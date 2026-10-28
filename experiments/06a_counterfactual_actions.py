"""Run the approved simulation-only Experiment 06A; never train a controller.

python3 experiments/06a_counterfactual_actions.py --run [--resume]
"""

import argparse
import io
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gymnasium as gym
import numpy as np

from src.counterfactual import (
    analyze, assert_bitwise_equal, compare_PD_reference, evaluate_first_actions, fixed_state_menu,
    make_figure, scientific_report, validate_rollouts, write_csv,
)
from src.horizon_experiment import (
    _assert_hashes, array_digest, immutable_json, load_npz, read_json,
    save_npz, sha256, write_json,
)
from src.horizon_study import exclusive
from src.policy_evaluation import evaluate_rollouts


OUTPUT = ROOT / 'results/offline_rl/experiment_06A'


def preservation_snapshot():
    files = [p for p in ROOT.iterdir() if p.is_file()]
    for directory in ('data', 'results', 'src', 'experiments', 'configs', 'tests', 'docs'):
        files.extend(p for p in (ROOT / directory).rglob('*') if p.is_file()
                     and OUTPUT not in p.parents and '__pycache__' not in p.parts and p.name != '.DS_Store')
    return {'sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sorted(set(files))}}


def frozen_arrays(path, arrays):
    if path.exists():
        assert_bitwise_equal(load_npz(path), arrays)
    else:
        save_npz(path, arrays)


def concatenate(groups):
    return {key: np.concatenate([group[key] for group in groups]) for key in groups[0]}


def run(resume=False, snapshot=None):
    output = OUTPUT
    if output.exists() and not resume:
        raise FileExistsError('Output exists; use --resume to verify/reuse completed artifacts.')
    output.mkdir(parents=True, exist_ok=True)
    with exclusive(output):
        config = read_json(ROOT / 'configs/experiment_06a.json')
        immutable_json(output / 'protocol.json', config)
        sources = [ROOT / name for name in ('src/counterfactual.py',
                   'experiments/06a_counterfactual_actions.py', 'configs/experiment_06a.json',
                   'tests/test_counterfactual.py', 'src/expert.py', 'src/safety.py',
                   'src/dataset.py', 'src/active_query.py', 'src/policy_evaluation.py', 'src/rollout_horizon.py')]
        source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in sources}
        immutable_json(output / 'source_hashes.json', source_hashes)
        if not (output / 'preservation_before.json').exists():
            before = read_json(snapshot) if snapshot else preservation_snapshot()
            write_json(output / 'preservation_before.json', before)
        before = read_json(output / 'preservation_before.json')
        _assert_hashes(ROOT, before['sha256'])
        if (output / 'completion.json').exists():
            complete = read_json(output / 'completion.json')
            _assert_hashes(output, complete['artifacts_sha256'])
            print('Completed Experiment 06A verified; no simulation or overwrite.', flush=True)
            return
        if not (output / 'completion.json').exists():
            stream = io.StringIO()
            suite = unittest.defaultTestLoader.discover(str(ROOT / 'tests'), pattern='test_counterfactual.py')
            test_result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
            (output / 'technical_tests.txt').write_text(stream.getvalue())
            write_json(output / 'technical_tests.json', dict(passed=test_result.wasSuccessful(),
                       tests=test_result.testsRun, failures=len(test_result.failures), errors=len(test_result.errors),
                       real_simulator_steps=0, note='Analytic/synthetic fixtures; no training or real simulator steps'))
        assert read_json(output / 'technical_tests.json')['passed']
        menu = fixed_state_menu(config)
        frozen_arrays(output / 'state_action_manifest.npz', menu)
        immutable_json(output / 'state_action_manifest.json', {key: value.tolist() for key, value in menu.items()})
        immutable_json(output / 'metadata.json', {
            'python': sys.version, 'numpy': np.__version__, 'gymnasium': gym.__version__,
            'platform': platform.platform(),
            'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            'state_action_digest': array_digest(*menu.values()),
            'state_manifest_frozen_before_any_rollout': True,
            'diagnostic_only': True, 'training_dataset_writes': 0,
        })
        env = gym.make(config['environment'])
        try:
            assert env.unwrapped.dt == config['dt']
            physics = {name: float(getattr(env.unwrapped, name)) for name in ('dt', 'g', 'm', 'l', 'max_speed', 'max_torque')}
            assert physics == dict(dt=.05, g=10., m=1., l=1., max_speed=8., max_torque=2.)
            immutable_json(output / 'physics.json', physics)
        finally:
            env.close()
        runs, sid = [], menu['run_state_ids']
        main_costs, replay_costs = [], []
        for state_id, initial in enumerate(menu['initial_conditions']):
            ids = np.flatnonzero(sid == state_id)
            path = output / 'main' / f'state_{state_id:03d}.npz'
            cost_path = path.with_name(path.stem + '_costs.npz')
            if not path.exists():
                arrays, costs = evaluate_first_actions(np.repeat(initial[None], len(ids), axis=0),
                            menu['first_actions'][ids], np.repeat(menu['reset_seeds'][state_id], len(ids)), config['horizon'])
                save_npz(cost_path, costs)
                save_npz(path, arrays)
            arrays = load_npz(path)
            validate_rollouts(arrays, menu['first_actions'][ids], config['horizon'])
            runs.append(arrays)
            main_costs.append(load_npz(cost_path))
            if (state_id + 1) % 20 == 0:
                print(f'Main: {state_id + 1}/97 states saved and verified.', flush=True)
        arrays = concatenate(runs)
        frozen_arrays(output / 'rollouts.npz', arrays)
        # Baseline replay uses the original evaluator, not the override runner.
        reference_path = output / 'PD_reference_replay.npz'
        if not reference_path.exists():
            start = time.perf_counter()
            reference, _ = evaluate_rollouts(None, menu['initial_conditions'], config['seeds']['environment'], config['horizon'])
            write_json(output / 'PD_reference_costs.json', {
                'resets': len(menu['initial_conditions']), 'steps': len(menu['initial_conditions']) * config['horizon'],
                'wall_seconds': time.perf_counter() - start,
                'count_basis': 'Verified complete preserved evaluator: one reset and 200 steps per initial state',
            })
            save_npz(reference_path, reference)
        reference = load_npz(reference_path)
        baseline_arrays = {key: value[menu['baseline_run_ids']] for key, value in arrays.items()}
        zero_signs = compare_PD_reference(baseline_arrays, reference)
        assert zero_signs == [[96, 0, 0]]
        # Preserve the exact offset formula and independently replay its +0.0.
        direct_ids = menu['baseline_run_ids'][[96]]
        direct_path = output / 'equilibrium_direct_replay.npz'
        if not direct_path.exists():
            direct, direct_cost = evaluate_first_actions(menu['initial_conditions'][[96]],
                    menu['first_actions'][direct_ids], menu['reset_seeds'][[96]], config['horizon'])
            save_npz(output / 'equilibrium_direct_replay_costs.npz', direct_cost)
            save_npz(direct_path, direct)
        assert_bitwise_equal({key: value[direct_ids] for key, value in arrays.items()}, load_npz(direct_path))
        replay_costs.append(load_npz(output / 'equilibrium_direct_replay_costs.npz'))
        for state_id, initial in enumerate(menu['initial_conditions']):
            ids = np.flatnonzero((sid == state_id) & (np.arange(len(sid)) != menu['baseline_run_ids'][state_id]))
            path = output / 'replay' / f'state_{state_id:03d}.npz'
            cost_path = path.with_name(path.stem + '_costs.npz')
            if not path.exists():
                replay, costs = evaluate_first_actions(np.repeat(initial[None], len(ids), axis=0),
                              menu['first_actions'][ids], np.repeat(menu['reset_seeds'][state_id], len(ids)), config['horizon'])
                save_npz(cost_path, costs)
                save_npz(path, replay)
            replay = load_npz(path)
            assert_bitwise_equal({key: value[ids] for key, value in arrays.items()}, replay)
            replay_costs.append(load_npz(cost_path))
            if (state_id + 1) % 20 == 0:
                print(f'Replay: {state_id + 1}/97 states bitwise verified.', flush=True)
        control = sid == len(menu['initial_conditions']) - 1
        assert arrays['episode_returns'][control].max() == 0.
        assert np.all(arrays['rewards'] <= 0.)
        assert reference['episode_returns'][-1] == 0.
        summary, episode_rows, state_rows = analyze(menu, arrays, config)
        def costs_total(items):
            return dict(resets=sum(int(c['reset_attempts']) for c in items),
                        steps=sum(int(c['step_attempts']) for c in items),
                        wall_seconds=sum(float(c['wall_seconds']) for c in items))
        main_cost = costs_total(main_costs)
        replay_cost = costs_total(replay_costs)
        reference_cost = read_json(output / 'PD_reference_costs.json')
        for key in ('resets', 'steps', 'wall_seconds'):
            replay_cost[key] += reference_cost[key]
        costs = dict(main=main_cost, replay=replay_cost, total_resets=main_cost['resets'] + replay_cost['resets'],
                     total_steps=main_cost['steps'] + replay_cost['steps'])
        assert main_cost['steps'] == len(sid) * config['horizon']
        assert replay_cost['steps'] == (len(sid) + 1) * config['horizon']
        assert main_cost['resets'] == len(sid) and replay_cost['resets'] == len(sid) + 1
        _assert_hashes(ROOT, before['sha256'])
        _assert_hashes(ROOT, source_hashes)
        checks = dict(passed=True, protected_files=len(before['sha256']), all_unique_executions_replayed=len(sid),
                      original_PD_evaluator_bitwise_matches=len(menu['initial_conditions']) - 1,
                      original_PD_evaluator_numerically_exact_matches=len(menu['initial_conditions']),
                      original_PD_evaluator_zero_sign_exceptions=zero_signs,
                      equilibrium_offset_formula_direct_replay_bitwise=True,
                      matched_resets=True, full_horizon_and_raw_safety=True, equilibrium_ceiling=True,
                      deterministic_state_action_manifest=True, no_training=True, training_dataset_writes=0,
                      technical_tests=read_json(output / 'technical_tests.json'))
        write_json(output / 'summary.json', summary)
        write_json(output / 'verification.json', checks)
        write_json(output / 'costs.json', costs)
        write_csv(output / 'episode_metrics.csv', episode_rows)
        write_csv(output / 'state_metrics.csv', state_rows)
        make_figure(output / 'counterfactual_diagnostics.png', menu, episode_rows, state_rows)
        report = scientific_report(summary, checks, costs)
        # The reviewed, versioned report is an immutable scientific record.
        # Reproduction reports contain run-specific timings and file counts.
        report_path = output / 'scientific_report.md'
        report_path.write_text(report)
        artifacts = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                     if p.is_file() and p.name not in ('completion.json', '.lock')}
        write_json(output / 'completion.json', dict(passed=True, artifacts_sha256=artifacts,
                   report_sha256=sha256(report_path)))
        print(f'Experiment 06A complete; all checks passed. Report: {report_path}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preservation-snapshot', type=Path)
    args = parser.parse_args()
    run(args.resume, args.preservation_snapshot)
