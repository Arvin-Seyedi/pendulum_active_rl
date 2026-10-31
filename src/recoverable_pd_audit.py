"""Independent E09 saved-evidence checks; no simulation or optimization calls."""

import json
from pathlib import Path
import time

import numpy as np

from src.viability_checker import check_extremal_certificate


IDS = [3280, 0, 6560, 767, 44]


def _json(path):
    return json.loads(path.read_text())


def _arrays(path):
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _identical(a, b):
    assert a.keys() == b.keys()
    for name in a:
        assert a[name].dtype == b[name].dtype and a[name].shape == b[name].shape, name
        assert a[name].tobytes() == b[name].tobytes(), name


def audit_saved_pilot(root, output):
    root, output = Path(root), Path(output)
    summary = _json(output / 'summary.json')
    assert summary['pilot_ids'] == IDS and not summary['full_study_executed']
    assert [c['state_id'] for c in summary['cases']] == IDS
    with np.load(root / 'results/expert_safety_map/expert_safety_map.npz') as a:
        axes = {name: a[name].copy() for name in a.files}
    pairs, policies, certificates, checks = 0, 0, 0, 0
    reward_error = 0.; discounted_error = 0.; upper = []; lower = []
    started = time.perf_counter()
    for case in summary['cases']:
        sid = case['state_id']; folder = output / f'state_{sid:04d}'
        i, j = divmod(sid, 81)
        initial = np.array([axes['theta_values'][j], axes['omega_values'][i]], dtype=np.float64)
        assert case['identity']['initial_raw_bytes_hex'] == initial.tobytes().hex()
        assert case['identity']['initial_hex'] == [float(x).hex() for x in initial]
        archived_unsafe = bool(axes['margin_map'][i, j] < 0)
        assert case['comparison']['archived_unsafe'] == archived_unsafe
        gains = [d for d in (folder / 'gains').iterdir() if d.is_dir()]
        assert len(gains) == 9
        actual_pairs = set()
        for gain in gains:
            identity = _json(gain / 'identity.json'); kp, kd = identity['gains']
            actual_pairs.add((kp, kd))
            a = _arrays(gain / 'rollout.npz')
            _identical(a, _arrays(gain / 'policy_repeat.npz')); policies += 1
            _identical(a, _arrays(gain / 'replay.npz')); pairs += 1
            diag = _json(gain / 'diagnostics.json')
            assert diag == _json(gain / 'policy_repeat_diagnostics.json')
            legacy = []
            for t in range(200):
                theta, omega = a['raw_states'][0, t]
                obs = np.array([np.cos(theta), np.sin(theta), omega],
                               dtype=np.float64 if t == 0 else np.float32)
                angle, speed = np.arctan2(obs[1], obs[0]), obs[2]
                expected = np.array([np.clip(-kp*angle-kd*speed, -2., 2.)], dtype=np.float32)
                assert expected.tobytes() == a['actions'][0, t].tobytes(), (sid, kp, kd, t)
                legacy.append(float(min(1. - abs(angle), 1. - abs(speed)/3.)))
            assert np.array(legacy).tobytes() == np.array(diag['legacy_pre_action_margins']).tobytes()
            if (kp, kd) == (8., 2.):
                p = case['comparison']; old_time = axes['violation_time_map'][i, j]
                old_time = None if np.isnan(old_time) else float(old_time)
                crosses = np.flatnonzero(np.array(legacy) < 0)
                new_time = None if not len(crosses) else int(crosses[0])*.05
                assert p['archived_vs_legacy_label_disagreement'] == (archived_unsafe != bool(len(crosses)))
                assert p['archived_vs_legacy_crossing_disagreement'] == (old_time != new_time)
                assert p['legacy_margin_difference'] == min(legacy)-float(axes['margin_map'][i, j])
                assert p['archived_vs_raw_label_disagreement'] == (archived_unsafe != bool(np.any(a['margins'] < 0)))
        assert actual_pairs == {(kp, kd) for kp in (6.,8.,12.) for kd in (1.,2.,4.)}
        for trial in case['trials']:
            for attempt in trial['attempts']:
                action = trial['action']; direction = 'u_minus2' if action == -2 else 'u_plus2'
                certificate = _json(folder / 'extrema' / direction / f'precision_{attempt["precision_bits"]}' / 'certificate.json')
                checked = check_extremal_certificate(certificate, sid, [float(x).hex() for x in initial])
                assert checked == attempt['checker']
                checks += checked['check_count']; certificates += 1
                if checked['certified']:
                    (upper if action == -2 else lower).append(sid)
                path = folder / 'extrema' / direction / 'simulator'
            _identical(_arrays(path / 'rollout.npz'), _arrays(path / 'replay.npz')); pairs += 1
    # Audit every scientific rollout (gain, extremal, and optimization modes)
    # with direct raw-state formulas rather than the production metric helper.
    rollouts = list(output.glob('state_*/**/rollout.npz'))
    for path in rollouts:
        a = _arrays(path); raw = a['raw_states'][0]; actions = a['actions'][0, :, 0]
        assert raw.shape == (201, 2) and actions.shape == (200,) and actions.dtype == np.float32
        assert raw[0].tobytes() == a['initial_conditions'][0].tobytes()
        assert np.isfinite(raw).all() and np.isfinite(actions).all() and np.max(np.abs(actions)) <= 2
        margins = np.minimum(1-np.abs(raw[:, 0]), 1-np.abs(raw[:, 1])/3)
        assert margins.tobytes() == a['margins'][0].tobytes()
        metrics = _json(path.parent / 'metrics.json')
        assert float(margins.min()) == metrics['minimum_margins']
        assert bool(np.any(margins < 0)) == metrics['violated']
        assert bool(np.all(margins >= 0)) == metrics['witness']
        crossing = np.flatnonzero(margins < 0)
        expected_step = int(crossing[0]) if len(crossing) else None
        assert metrics['first_crossing_step'] == expected_step
        assert metrics['first_crossing_seconds'] == (None if expected_step is None else expected_step*.05)
        assert not a['terminated'].any() and not a['truncated'][0, :-1].any() and a['truncated'][0, -1]
        theta = (raw[:-1, 0]+np.pi)%(2*np.pi)-np.pi
        physical_reward = -(theta**2 + .1*raw[:-1, 1]**2 + .001*actions.astype(np.float64)**2)
        error = float(np.max(np.abs(physical_reward-a['rewards'][0])))
        reward_error = max(reward_error, error)
        assert error < 1e-7, (path, error)  # float32 action arithmetic, never a safety tolerance
        assert float(a['rewards'].sum()) == metrics['returns']
        independent_discounted = sum(float(r)*(.99**t) for t,r in enumerate(a['rewards'][0]))
        discounted_error = max(discounted_error, abs(independent_discounted-metrics['discounted_returns']))
        assert abs(independent_discounted-metrics['discounted_returns']) < 1e-10
    optimization = summary['optimization']
    assert optimization['state_id'] == 44
    for r in optimization['records']:
        if r['candidate_available']:
            path = output / 'state_0044/optimization' / r['start'] / r['mode']
            _identical(_arrays(path/'rollout.npz'), _arrays(path/'replay.npz')); pairs += 1
            assert r['witness'] == bool(np.all(_arrays(path/'rollout.npz')['margins'] >= 0))
    assert len(optimization['repetitions']) == 3
    assert all(r['passed'] for r in optimization['repetitions'])
    assert upper and lower, 'Pilot must exercise independently certified upper and lower directions'
    assert sorted(p.name for p in output.glob('state_*') if p.is_dir()) == sorted(f'state_{sid:04d}' for sid in IDS)
    return {'passed': True, 'policy_repeats_bitwise': policies,
            'recorded_action_replay_pairs': pairs, 'audited_scientific_rollouts': len(rollouts),
            'independent_certificate_records': certificates, 'independent_transition_rechecks': checks,
            'upper_certificate_ids': upper, 'lower_certificate_ids': lower,
            'maximum_reward_formula_difference': reward_error,
            'maximum_discounted_sum_difference': discounted_error,
            'all_201_raw_states_checked': True, 'archive_disagreements_preserved': True,
            'wall_seconds': time.perf_counter()-started,
            'extra_simulator_steps': 0, 'extra_solver_attempts': 0,
            'scope': 'Raw safety thresholds exact; reward/sum checks allow arithmetic-order differences only. Model certificates remain separate from simulator witnesses.'}
