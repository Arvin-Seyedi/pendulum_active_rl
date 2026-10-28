"""Resume the approved 57-state study without editing or rerunning its pilot.

Only process orchestration is extended. All dynamics, transcription, solver
options, execution, and replay functions come from the accepted pilot.
"""

import io
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import unittest
import warnings

import casadi as ca
import gymnasium as gym
import numpy as np

from src.counterfactual import assert_bitwise_equal
from src.horizon_experiment import _assert_hashes, immutable_json, load_npz, read_json, save_npz, sha256, write_json
from src.horizon_study import exclusive
from src.policy_evaluation import evaluate_rollouts
from src.recoverability import candidate_diagnostics, evaluate_controls, pack, replay_and_measure, solve_attempt, unpack, validate_execution
from src.recoverability_checks import select_cases
from src.recoverability_experiment import cached_evaluation, protected_manifest

FULL_CONFIG = 'configs/experiment_07_full_study.json'


def settings(root):
    control = read_json(root / FULL_CONFIG)
    assert control['authorized_phase'] == 'full_57_state_study' and not control['methodology_changes']
    config = read_json(root / control['scientific_config'])
    pilot = root / control['accepted_pilot']
    assert config == read_json(pilot / 'protocol.json'), 'Scientific configuration changed.'
    assert ca.__version__ == config['solver']['casadi_version'] == '3.7.2'
    return control, config, pilot, root / control['output']


def full_worker(root, folder):
    control, config, pilot, output = settings(root)
    folder = Path(folder).resolve()
    assert output.resolve() in folder.parents
    request = read_json(folder / 'request.json')
    cases = read_json(output / 'state_manifest.json')['all_57_cases']
    case = next(c for c in cases if c['state_id'] == request['state_id'])
    assert request['initial'] == case['initial'] and request['start'] in config['starts']
    _assert_hashes(root, read_json(output / 'source_hashes.json'))
    solve_attempt(folder, config, np.asarray(case['initial']), load_npz(folder / 'initial_guess.npz')['decision'])


def read_attempt(folder, initial, config):
    record = read_json(folder / 'attempt.json')
    if record['process_exit_code'] != 0 and not record['watchdog_timeout']:
        raise RuntimeError(f'Worker failed technically: {folder / "solver.log"}; preserved for review.')
    candidate = folder / 'candidate.npz'
    if not candidate.exists():
        candidate = folder / 'last_iterate.npz'
    if (folder / 'solver_result.json').exists():
        record['solver'] = read_json(folder / 'solver_result.json')
    if not candidate.exists():
        return None, {**record, 'candidate_available': False, 'candidate_source': None}
    decision = load_npz(candidate)['decision']
    available = decision.shape == (3 * config['horizon'],) and np.isfinite(decision).all()
    record.update(candidate_available=bool(available), candidate_source=candidate.name)
    if not available:
        return None, record
    record['diagnostics'] = candidate_diagnostics(decision, initial, config['horizon'])
    return decision, record


def run_new_attempt(root, folder, request, guess, config):
    folder.mkdir(parents=True, exist_ok=True)
    immutable_json(folder / 'request.json', request)
    if (folder / 'initial_guess.npz').exists():
        assert_bitwise_equal(load_npz(folder / 'initial_guess.npz'), {'decision': guess})
    else:
        save_npz(folder / 'initial_guess.npz', {'decision': guess})
    if not (folder / 'attempt.json').exists():
        if (folder / 'solver.log').exists():
            raise RuntimeError(f'Interrupted attempt retained at {folder}; inspect before resuming.')
        env = dict(os.environ)
        thread_keys = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
        env.update({key: '1' for key in thread_keys})
        started = time.perf_counter()
        with (folder / 'solver.log').open('w') as log:
            process = subprocess.Popen([sys.executable, str(root / 'experiments/07_full_study.py'), '--worker', str(folder)],
                                       cwd=folder, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            write_json(folder / 'worker_identity.json', {'pid': process.pid, 'started_unix': time.time()})
            timeout = False
            try:
                process.wait(timeout=config['watchdog_seconds'])
            except subprocess.TimeoutExpired:
                timeout = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        write_json(folder / 'attempt.json', {'watchdog_timeout': timeout, 'process_exit_code': process.returncode,
                   'process_wall_seconds': time.perf_counter() - started, 'pid': process.pid,
                   'thread_environment': {key: env[key] for key in thread_keys}})
    decision, record = read_attempt(folder, request['initial'], config)
    write_json(folder / 'candidate_review.json', record)
    return decision, record


def state_records(root, output, case, config, reused):
    sid, initial, seed = case['state_id'], np.array(case['initial']), case['reset_seed']
    folder = output / f'state_{sid:03d}'
    folder.mkdir(exist_ok=True)
    if not (folder / 'PD.npz').exists():
        start = time.perf_counter()
        # The existing evaluator computes an unused sample SD for N=1.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='Degrees of freedom <= 0 for slice')
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='invalid value encountered in scalar divide')
            arrays, _ = evaluate_rollouts(None, initial[None], seed, config['horizon'])
        write_json(folder / 'PD_execution_costs.json', {'reset_attempts': 1, 'step_attempts': 200,
                   'wall_seconds': time.perf_counter() - start, 'count_basis': 'Preserved evaluator; validated full horizon'})
        save_npz(folder / 'PD.npz', arrays)
    baseline = load_npz(folder / 'PD.npz')
    validate_execution(baseline, initial)
    prior = load_npz(root / config['source_experiment'] / 'PD_reference_replay.npz')
    assert_bitwise_equal(baseline, {k: v[[sid]] for k, v in prior.items()})
    if not (folder / 'PD_replay.npz').exists():
        replay, costs, metrics = replay_and_measure(baseline, initial, seed, config)
        write_json(folder / 'PD_replay_costs.json', costs)
        write_json(folder / 'PD_metrics.json', metrics)
        save_npz(folder / 'PD_replay.npz', replay)
    assert_bitwise_equal(baseline, load_npz(folder / 'PD_replay.npz'))
    pd = read_json(folder / 'PD_metrics.json')
    assert pd['violated'] == case['PD_unsafe']
    candidates, attempts = [], []
    for start_index, name in enumerate(config['starts']):
        initialization = folder / f'initialization_{name}.npz'
        if not initialization.exists():
            if name == 'PD':
                init_arrays = baseline
            else:
                controls = np.zeros(200) if name == 'zero' else np.repeat(np.random.default_rng(
                    np.random.SeedSequence([config['seeds']['random_start'], sid])).uniform(-2, 2, size=10), 20)
                init_arrays, costs = evaluate_controls(initial, seed, controls)
                write_json(folder / f'initialization_{name}_costs.json', costs)
            save_npz(initialization, init_arrays)
        init_arrays = load_npz(initialization)
        guess = pack(init_arrays['raw_states'][0], init_arrays['actions'][0, :, 0])
        attempt_folder = folder / name / 'original'
        if reused:
            assert_bitwise_equal(load_npz(attempt_folder / 'initial_guess.npz'), {'decision': guess})
            vector, attempt = read_attempt(attempt_folder, initial, config)
        else:
            request = dict(state_id=sid, initial=initial.tolist(), start=name, authorization='full_57_state_study')
            vector, attempt = run_new_attempt(root, attempt_folder, request, guess, config)
        status = attempt.get('solver', {}).get('stats', {}).get('return_status', 'Watchdog_Timeout' if attempt['watchdog_timeout'] else 'No_Solver_Result')
        common = {'state_id': sid, 'group': case['group'], 'start': name, 'start_index': start_index,
                  'solver_status': status, 'solver_success': bool(attempt.get('solver', {}).get('stats', {}).get('success', False)),
                  'watchdog_timeout': attempt['watchdog_timeout'], 'candidate_available': vector is not None,
                  'reused_pilot': reused}
        attempts.append({**common, **attempt})
        if vector is None:
            for mode in ('open_loop', 'tracking'):
                candidates.append({**common, 'mode': mode, 'witness': False, 'missing_candidate': True})
            continue
        states, controls = unpack(vector, initial, config['horizon'])
        diagnostics = attempt['diagnostics']
        for mode in ('open_loop', 'tracking'):
            arrays, metrics = cached_evaluation(folder / name / mode, initial, seed, controls, config,
                                               states if mode == 'tracking' else None)
            error = np.abs(arrays['raw_states'][0] - states)
            candidates.append({**common, **diagnostics, **metrics, 'mode': mode, 'missing_candidate': False,
                               'delta_return': metrics['returns'] - pd['returns'],
                               'delta_discounted_return': metrics['discounted_returns'] - pd['discounted_returns'],
                               'maximum_plan_replay_state_discrepancy': error.max(axis=0).tolist(),
                               'terminal_plan_replay_state_discrepancy': error[-1].tolist(),
                               'nominally_feasible_at_solver_tolerance': bool(diagnostics['maximum_dynamics_residual'] <= config['solver']['constr_viol_tol']
                                   and diagnostics['maximum_bound_violation'] <= config['solver']['constr_viol_tol']
                                   and diagnostics['predicted_minimum_raw_margin'] >= 0)})
    result = {'case': case, 'PD': pd, 'candidates': candidates, 'attempts': attempts}
    write_json(folder / 'full_study_record.json', result)
    hashes = {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob('*')) if p.is_file() and p.name != 'full_state_completion.json'}
    write_json(folder / 'full_state_completion.json', {'passed': True, 'artifacts_sha256': hashes})
    return result


def log(output, message):
    with (output / 'execution.log').open('a') as stream:
        stream.write(message + '\n')
    print(message, flush=True)


def run_full(root, resume=False, snapshot=None):
    root = Path(root).resolve()
    control, config, pilot, output = settings(root)
    if output.exists() and not resume:
        raise FileExistsError('Full study exists; use --resume to reuse verified completed states.')
    output.mkdir(parents=True, exist_ok=True)
    with exclusive(output):
        _assert_hashes(pilot, read_json(pilot / 'completion.json')['artifacts_sha256'])
        _assert_hashes(root, read_json(pilot / 'source_hashes.json'))
        assert read_json(pilot / 'verification.json')['passed']
        sources = list(read_json(pilot / 'source_hashes.json')) + [FULL_CONFIG, 'src/recoverability_full_study.py',
                   'src/recoverability_full_analysis.py', 'experiments/07_full_study.py', 'tests/test_recoverability_full.py']
        hashes = {name: sha256(root / name) for name in sources}
        immutable_json(output / 'source_hashes.json', hashes)
        immutable_json(output / 'protocol.json', config)
        immutable_json(output / 'authorization.json', control)
        if not (output / 'preservation_before.json').exists():
            write_json(output / 'preservation_before.json', read_json(snapshot) if snapshot else protected_manifest(root, output))
        before = read_json(output / 'preservation_before.json')
        _assert_hashes(root, before['sha256'])
        if (output / 'completion.json').exists():
            _assert_hashes(output, read_json(output / 'completion.json')['artifacts_sha256'])
            print('Completed full study verified; no re-execution.', flush=True)
            return
        cases = select_cases(root, config)
        assert cases == read_json(pilot / 'state_manifest.json')
        immutable_json(output / 'state_manifest.json', cases)
        immutable_json(output / 'metadata.json', {'python': sys.version, 'executable': sys.executable,
                       'casadi': ca.__version__, 'numpy': np.__version__, 'gymnasium': gym.__version__,
                       'platform': platform.platform(), 'git_head': subprocess.check_output(['git','rev-parse','HEAD'], cwd=root, text=True).strip(),
                       'scientific_config_matches_accepted_pilot': True, 'reused_pilot_ids': cases['pilot_ids_in_order'],
                       'new_case_count': 53, 'new_solver_repeats': 0, 'all_candidates_have_recorded_action_replay': True})
        for name in ('technical_tests.json', 'model_verification.json', 'model_verification.npz', 'model_verification_costs.json', 'solver_build.json'):
            target = output / name
            if not target.exists(): shutil.copy2(pilot / name, target)
            assert sha256(target) == sha256(pilot / name)
        if not (output / 'full_technical_tests.json').exists():
            stream = io.StringIO()
            suite = unittest.defaultTestLoader.discover(str(root / 'tests'), pattern='test_recoverability*.py')
            test_result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
            (output / 'full_technical_tests.txt').write_text(stream.getvalue())
            write_json(output / 'full_technical_tests.json', {'passed': test_result.wasSuccessful(), 'tests': test_result.testsRun,
                       'failures': len(test_result.failures), 'errors': len(test_result.errors)})
        assert read_json(output / 'full_technical_tests.json')['passed']
        records = []
        for index, case in enumerate(cases['all_57_cases']):
            sid = case['state_id']; folder = output / f'state_{sid:03d}'
            reused = sid in cases['pilot_ids_in_order']
            if reused and not folder.exists():
                temporary = output / f'.copy_state_{sid:03d}'
                if temporary.exists():
                    raise RuntimeError(f'Incomplete pilot copy preserved at {temporary}; inspect before resuming.')
                shutil.copytree(pilot / f'state_{sid:03d}', temporary)
                temporary.rename(folder)
            if (folder / 'full_state_completion.json').exists():
                _assert_hashes(folder, read_json(folder / 'full_state_completion.json')['artifacts_sha256'])
                record = read_json(folder / 'full_study_record.json')
            else:
                record = state_records(root, output, case, config, reused)
            records.append(record)
            ol = any(r['witness'] for r in record['candidates'] if r['mode'] == 'open_loop')
            tr = any(r['witness'] for r in record['candidates'] if r['mode'] == 'tracking')
            log(output, f"Verified {index+1}/57: state {sid}, {case['group']}, open_loop={ol}, tracking={tr}, reused_pilot={reused}")
        from src.recoverability_full_analysis import finish_analysis
        finish_analysis(root, output, records, config, before, hashes)
        print('Full 57-state study complete; no commit or push.', flush=True)
