"""Additive full-cohort orchestration; the reviewed pilot remains unchanged.

The interval generator, independent checker, precision escalation, deterministic
repeat, Gymnasium execution, and recorded-action replay are reused verbatim.
"""

from fractions import Fraction
import io
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import unittest

import gmpy2 as g
import gymnasium as gym
import numpy as np

from src.counterfactual import assert_bitwise_equal
from src.horizon_experiment import _assert_hashes, immutable_json, load_npz, read_json, save_npz, sha256, write_json
from src.horizon_study import exclusive
from src.recoverability import replay_and_measure, validate_execution
from src.recoverability_experiment import cached_evaluation, protected_manifest
from src.viability import MODEL, ratio
from src.viability_experiment import CASE_IDS, SOURCES, _attempt, compare_simulator, settings as pilot_settings, verify_completed as verify_pilot

CONFIG = 'configs/experiment_08_study.json'
ADDITIONS = [CONFIG, 'src/viability_study.py', 'src/viability_study_analysis.py',
             'experiments/08_viability_study.py', 'tests/test_viability_study.py']
COST_KEYS = ('primary_interval_updates', 'repeat_interval_updates', 'checker_transition_checks',
             'generation_seconds', 'repeat_seconds', 'checker_seconds', 'simulator_steps', 'simulator_resets')


def settings(root):
    scientific, source, pilot = pilot_settings(root)
    control = read_json(root / CONFIG)
    assert control['authorized_phase'] == 'remaining_states_after_reviewed_five_case_pilot'
    assert not control['scientific_method_changes'] and not control['training_data_use']
    assert control['scientific_config'] == 'configs/experiment_08_pilot.json'
    assert root / control['accepted_pilot'] == pilot and root / control['source_experiment'] == source
    assert control['reused_pilot_ids'] == CASE_IDS
    assert not control['global_viability_grid_if_all_40_certified']
    assert scientific == read_json(pilot / 'protocol.json')
    cases = read_json(source / 'state_manifest.json')['all_57_cases']
    ids = [c['state_id'] for c in cases]
    assert len(ids) == len(set(ids)) == 57 and set(CASE_IDS).issubset(ids)
    assert sum(c['PD_unsafe'] for c in cases) == 40
    assert sum(c['group'] in ('nominal_safe', 'challenging_safe') for c in cases) == 16
    assert [c['state_id'] for c in cases if c['group'] == 'equilibrium'] == [96]
    new = [c for c in cases if c['state_id'] not in CASE_IDS]
    assert len(new) == 52 and sum(c['PD_unsafe'] for c in new) == 38
    assert sum(c['group'] in ('nominal_safe', 'challenging_safe') for c in new) == 14
    return scientific, control, source, pilot, root / control['output'], cases


def verify_study(root):
    root = Path(root).resolve()
    _, _, source, pilot, output, cases = settings(root)
    verify_pilot(root)
    done = read_json(output / 'completion.json')
    assert done['passed'] and done['case_ids'] == [c['state_id'] for c in cases]
    _assert_hashes(output, done['artifacts_sha256'])
    _assert_hashes(root, read_json(output / 'source_hashes.json'))
    _assert_hashes(root, read_json(output / 'preservation_before.json')['sha256'])
    _assert_hashes(source, read_json(source / 'completion.json')['artifacts_sha256'])
    assert sha256(root / 'docs/experiment_08_report.md') == done['report_sha256']
    print('Completed full study verified; no interval calculations, checks, or simulations rerun.', flush=True)


def _new_case(root, output, source, case, scientific, simulation_config, new_ids):
    """Only authorization/aggregation are extended beyond the pilot wrapper."""
    sid = case['state_id']
    assert sid in new_ids and sid not in CASE_IDS
    folder = output / f'state_{sid:03d}'
    folder.mkdir(exist_ok=True)
    marker = folder / 'completion.json'
    if marker.exists():
        _assert_hashes(folder, read_json(marker)['artifacts_sha256'])
        return read_json(folder / 'summary.json')
    baseline = load_npz(source / f'state_{sid:03d}' / 'PD.npz')
    initial = baseline['raw_states'][0, 0].copy()
    assert_bitwise_equal({'initial': initial}, {'initial': np.array(case['initial'], dtype=np.float64)})
    initial_hex = [float(v).hex() for v in initial]
    identity = dict(case, initial_hex=initial_hex,
                    initial_ratios=[ratio(Fraction.from_float(float(v))) for v in initial],
                    initial_raw_bytes_hex=initial.tobytes().hex(),
                    PD_source_sha256=sha256(source / f'state_{sid:03d}' / 'PD.npz'))
    immutable_json(folder / 'identity.json', identity)
    validate_execution(baseline, initial)
    if not (folder / 'PD_replay.npz').exists():
        replay, costs, metrics = replay_and_measure(baseline, initial, case['reset_seed'], simulation_config)
        save_npz(folder / 'PD_replay.npz', replay)
        write_json(folder / 'PD_replay_costs.json', costs)
        write_json(folder / 'PD_metrics.json', metrics)
    assert_bitwise_equal(baseline, load_npz(folder / 'PD_replay.npz'))
    pd = read_json(folder / 'PD_metrics.json')
    assert pd['violated'] == case['PD_unsafe']
    trials = []
    for action in scientific['actions_in_order']:
        direction = folder / ('u_minus2' if action == -2 else 'u_plus2')
        attempts = []
        for precision in scientific['precision_bits_in_order']:
            certificate, record = _attempt(direction / f'precision_{precision}', sid, initial_hex, action, precision)
            attempts.append(record)
            if certificate['status'] != 'boundary_unresolved': break
        arrays, metrics = cached_evaluation(direction / 'simulator', initial, case['reset_seed'],
                                            np.full(200, action, dtype=np.float64), simulation_config)
        diagnostic = compare_simulator(certificate, arrays, metrics)
        if record['checker']['certified']:
            k = pd['first_crossing_step']
            assert k is not None and np.all(baseline['actions'][0, :k, 0] == action)
            assert_bitwise_equal({'raw_prefix': baseline['raw_states'][0, :k+1]},
                                 {'raw_prefix': arrays['raw_states'][0, :k+1]})
        trials.append({'action': action, 'attempts': attempts, 'final': record, 'simulator': diagnostic})
    certified = [t for t in trials if t['final']['checker']['passed'] and t['final']['checker']['certified']]
    if not case['PD_unsafe']:
        assert not certified, 'Nonviability claim conflicts with an archived PD-safe reference; investigate.'
    result = dict(state_id=sid, group=case['group'], initial_hex=initial_hex,
                  mathematical_classification='certified_nonviable' if certified else 'no_nonviability_certificate',
                  real_model_only=True, PD_simulator=pd, trials=trials, equilibrium_proof=None)
    write_json(folder / 'summary.json', result)
    hashes = {str(p.relative_to(folder)): sha256(p) for p in folder.rglob('*') if p.is_file() and p != marker}
    write_json(marker, {'passed': True, 'artifacts_sha256': hashes})
    return result


def costs_for(output, records):
    attempts = [a for r in records for t in r['trials'] for a in t['attempts']]
    simulator = [read_json(p) for p in output.glob('state_*/**/*costs.json')]
    return dict(primary_interval_updates=sum(a['diagnostics']['interval_steps'] for a in attempts),
                repeat_interval_updates=sum(a['diagnostics']['interval_steps'] for a in attempts),
                checker_transition_checks=sum(a['checker']['check_count'] for a in attempts),
                generation_seconds=sum(a['generation_seconds'] for a in attempts),
                repeat_seconds=sum(a['generation_repeat_seconds'] for a in attempts),
                checker_seconds=sum(a['checker_seconds'] for a in attempts),
                simulator_steps=sum(int(c['step_attempts']) for c in simulator),
                simulator_resets=sum(int(c['reset_attempts']) for c in simulator))


def validate_saved_records(source, output, cases, records):
    """Read-only archive audit; no new interval propagation or simulator calls."""
    comparisons = 0
    assert [r['state_id'] for r in records] == [c['state_id'] for c in cases]
    for case, record in zip(cases, records):
        sid = case['state_id']; folder = output / f'state_{sid:03d}'
        identity = read_json(folder / 'identity.json')
        baseline = load_npz(source / f'state_{sid:03d}' / 'PD.npz')
        initial = baseline['raw_states'][0, 0]
        assert identity['initial_raw_bytes_hex'] == initial.tobytes().hex()
        assert record['initial_hex'] == identity['initial_hex'] == [float(v).hex() for v in initial]
        assert identity['state_id'] == sid
        assert_bitwise_equal(baseline, load_npz(folder / 'PD_replay.npz')); comparisons += 1
        validate_execution(baseline, initial)
        assert len(record['trials']) == 2
        for trial in record['trials']:
            action = trial['action']; direction = folder / ('u_minus2' if action == -2 else 'u_plus2')
            arrays = load_npz(direction / 'simulator/rollout.npz')
            assert_bitwise_equal(arrays, load_npz(direction / 'simulator/replay.npz')); comparisons += 1
            validate_execution(arrays, initial)
            assert arrays['actions'].dtype == np.float32 and np.all(arrays['actions'] == action)
            for attempt in trial['attempts']:
                precision = attempt['precision_bits']; sub = direction / f'precision_{precision}'
                _assert_hashes(sub, read_json(sub / 'completion.json')['artifacts_sha256'])
                certificate = read_json(sub / 'certificate.json')
                assert certificate['state_id'] == sid and certificate['initial_hex'] == record['initial_hex']
                assert certificate['action'] == action and certificate['model'] == MODEL
                checked = attempt['checker']
                assert checked['passed'] and checked['model_only'] and attempt['generation_repeat_exact']
                assert checked['certified'] == (certificate['status'] == 'certified_nonviable')
                assert checked['all_transition_enclosures_verified'] and checked['prefix_enclosures_inside_safety_set']
                assert attempt['status'] == certificate['status']
            assert trial['final'] == trial['attempts'][-1]
    return {'passed': True, 'recorded_action_replay_pairs': comparisons,
            'extra_interval_propagations': 0, 'extra_simulator_steps': 0,
            'exact_binary_initial_values': True, 'all_201_raw_states_checked': True}


def run_study(root, resume=False, snapshot=None):
    root = Path(root).resolve()
    scientific, control, source, pilot, output, cases = settings(root)
    if output.exists() and not resume:
        raise FileExistsError('Full analysis exists; use --verify or explicitly --run --resume.')
    output.mkdir(parents=True, exist_ok=True)
    with exclusive(output):
        if (output / 'completion.json').exists():
            verify_study(root); return
        verify_pilot(root)
        source_hashes = {name: sha256(root / name) for name in SOURCES + ADDITIONS}
        immutable_json(output / 'source_hashes.json', source_hashes)
        immutable_json(output / 'scientific_protocol.json', scientific)
        immutable_json(output / 'authorization.json', control)
        immutable_json(output / 'case_manifest.json', cases)
        if not (output / 'preservation_before.json').exists():
            write_json(output / 'preservation_before.json', read_json(snapshot) if snapshot else protected_manifest(root, output))
        before = read_json(output / 'preservation_before.json')
        _assert_hashes(root, before['sha256'])
        for name in source_hashes:
            target = output / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists(): target.write_bytes((root / name).read_bytes())
            assert sha256(target) == source_hashes[name]
        versions = dict(gmpy2=g.version(), GMP=g.mp_version(), MPFR=g.mpfr_version(), MPC=g.mpc_version())
        assert versions == read_json(pilot / 'metadata.json')['arithmetic_versions']
        immutable_json(output / 'metadata.json', dict(python=sys.version, executable=sys.executable,
                       platform=platform.platform(), numpy=np.__version__, gymnasium=gym.__version__,
                       arithmetic_versions=versions, model=MODEL, reused_pilot_ids=CASE_IDS, new_case_count=52,
                       git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                       implementation_error_bound_established=False))
        if not (output / 'technical_tests.json').exists():
            stream = io.StringIO(); started = time.perf_counter()
            suite = unittest.defaultTestLoader.discover(str(root / 'tests'), pattern='test_viability*.py')
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
            (output / 'technical_tests.txt').write_text(stream.getvalue())
            write_json(output / 'technical_tests.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
                       failures=len(result.failures), errors=len(result.errors), wall_seconds=time.perf_counter()-started))
        tests = read_json(output / 'technical_tests.json'); assert tests['passed']
        new_ids = {c['state_id'] for c in cases} - set(CASE_IDS)
        simulation_config = read_json(source / 'protocol.json')
        records = []
        for i, case in enumerate(cases):
            sid = case['state_id']; folder = output / f'state_{sid:03d}'
            if sid in CASE_IDS:
                if not folder.exists():
                    temporary = output / f'.copy_state_{sid:03d}'
                    if temporary.exists(): raise RuntimeError(f'Incomplete pilot copy retained at {temporary}; inspect before resume.')
                    shutil.copytree(pilot / f'state_{sid:03d}', temporary)
                    temporary.rename(folder)
                _assert_hashes(folder, read_json(pilot / f'state_{sid:03d}/completion.json')['artifacts_sha256'])
                record = read_json(folder / 'summary.json')
            else:
                record = _new_case(root, output, source, case, scientific, simulation_config, new_ids)
            records.append({**record, 'reused_pilot': sid in CASE_IDS, 'artifact_directory': folder.name})
            print(f"Verified {i+1}/57: state {sid}, {record['mathematical_classification']}, reused_pilot={sid in CASE_IDS}", flush=True)
        from src.viability_study_analysis import summarize, write_outputs
        summary = summarize(records, CASE_IDS)
        audit = validate_saved_records(source, output, cases, records)
        for sid in CASE_IDS:
            origin = pilot / f'state_{sid:03d}'; copied = output / origin.name
            for p in origin.rglob('*'):
                if p.is_file(): assert sha256(p) == sha256(copied / p.relative_to(origin))
        combined = costs_for(output, records); old_costs = read_json(pilot / 'numerical_costs.json')
        costs = {'combined': combined, 'reused_pilot': old_costs,
                 'new': {key: combined[key] - old_costs[key] for key in COST_KEYS}}
        assert combined['simulator_steps'] == 57000 and combined['simulator_resets'] == 285
        assert costs['new']['simulator_steps'] == 52000 and costs['new']['simulator_resets'] == 260
        assert audit['recorded_action_replay_pairs'] == 171
        _assert_hashes(root, before['sha256']); _assert_hashes(root, source_hashes)
        _assert_hashes(pilot, read_json(pilot / 'completion.json')['artifacts_sha256'])
        _assert_hashes(source, read_json(source / 'completion.json')['artifacts_sha256'])
        verification = dict(passed=True, technical_tests=tests, protected_files=len(before['sha256']),
                            independent_archive_audit=audit, reused_pilot_files_bitwise=True,
                            case_count=57, new_case_count=52, primary_unsafe_count=40, safe_reference_count=16,
                            model_only=True, all_generation_repeats_exact=True,
                            implementation_error_bound_established=False,
                            additional_methods_run=False, viability_grid_run=False,
                            primary_unresolved_ids=summary['primary']['unresolved_ids'])
        write_json(output / 'verification.json', verification)
        write_json(output / 'numerical_costs.json', costs)
        report = write_outputs(output, summary, verification, costs)
        if summary['primary']['unresolved_ids']:
            write_json(output / 'additional_analysis_required.json',
                       {'state_ids': summary['primary']['unresolved_ids'],
                        'next': 'Approved additional analytical checks before numerical viability methods.'})
            print('Primary phase finished; unresolved PD-unsafe states require the approved follow-up analysis.', flush=True)
            return
        reviewed = root / 'docs/experiment_08_report.md'
        if not reviewed.exists(): reviewed.write_text(report)
        hashes = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                  if p.is_file() and p.name != '.lock' and p != output / 'completion.json'}
        write_json(output / 'completion.json', dict(passed=True, case_ids=[c['state_id'] for c in cases],
                   artifacts_sha256=hashes, report_sha256=sha256(reviewed), primary=summary['primary'],
                   no_additional_methods_needed=True))
        print('Full study complete; no additional analytical or grid calculations needed. No commit or push.', flush=True)
