"""Experiment 09 pilot adapter for the unchanged Experiment 07 solver.

Only grid ID 44 is authorized here.  The optimization equations, solver options,
acceptance checks, and tracking law remain in the original reusable modules.
This adapter changes the worker entry point and archives paired pilot attempts.
"""

import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import numpy as np

from src.counterfactual import assert_bitwise_equal, episode_metrics
from src.horizon_experiment import (
    immutable_json, load_npz, read_json, save_npz, sha256, write_json,
)
from src.recoverability import (
    candidate_diagnostics, evaluate_controls, pack, solve_attempt, unpack,
    validate_execution,
)
from src.recoverability_experiment import cached_evaluation


OUTPUT_RELATIVE = 'results/recoverable_pd/pilot'
PILOT_IDS = [3280, 0, 6560, 767, 44]
REFERENCE_ID = 44
RANDOM_START_SEED = 909004
CONFIG_SOURCE = 'configs/experiment_07.json'
ARCHIVE = 'results/expert_safety_map/expert_safety_map.npz'
ARCHIVE_SHA256 = '2234e7c2c1d5e09595ed7f924286aa824748e747e1a4facc8be435c5cd6c622a'
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')


def _configuration(root):
    """Read the preserved E07 configuration without constructing an override."""
    config = read_json(root / CONFIG_SOURCE)
    assert config['horizon'] == 200 and config['dt'] == 0.05
    assert config['starts'] == ['PD', 'zero', 'random_blocks']
    assert config['random_blocks'] == {'count': 10, 'steps_per_block': 20,
                                       'range': [-2.0, 2.0]}
    assert config['solver']['max_iter'] == 1000
    assert config['solver']['max_wall_time'] == 30.0
    assert config['watchdog_seconds'] == 60.0
    assert config['tracking']['kp'] == 8.0 and config['tracking']['kd'] == 2.0
    return config


def _reference_initial(root):
    assert sha256(root / ARCHIVE) == ARCHIVE_SHA256
    archive = load_npz(root / ARCHIVE)
    i, j = divmod(REFERENCE_ID, 81)
    return np.array([archive['theta_values'][j], archive['omega_values'][i]],
                    dtype=np.float64)


def _validate_request(root, folder, request):
    expected = root / OUTPUT_RELATIVE / f'state_{REFERENCE_ID:04d}' / 'optimization'
    assert expected == folder.parent.parent
    assert folder.name in ('original', 'repeat')
    assert folder.parent.name in ('PD', 'zero', 'random_blocks')
    assert request['state_id'] == REFERENCE_ID
    assert request['authorized_pilot_ids'] == PILOT_IDS
    assert request['start'] == folder.parent.name
    assert request['config_source_sha256'] == sha256(root / CONFIG_SOURCE)
    assert request['random_start_seed'] == RANDOM_START_SEED
    initial = np.asarray(request['initial'], dtype=np.float64)
    assert initial.tobytes().hex() == request['initial_raw_bytes_hex']
    assert initial.tobytes() == _reference_initial(root).tobytes()
    assert [float(x).hex() for x in initial] == request['initial_hex']
    return initial


def worker(root, folder):
    """Run one authorized solve in the subprocess controlled by the parent."""
    root, folder = Path(root).resolve(), Path(folder).resolve()
    request = read_json(folder / 'request.json')
    initial = _validate_request(root, folder, request)
    config = _configuration(root)
    guess = load_npz(folder / 'initial_guess.npz')['decision']
    assert guess.shape == (600,) and np.isfinite(guess).all()
    solve_attempt(folder, config, initial, guess)


def _run_attempt(root, folder, request, guess, config):
    _validate_request(root, folder, request)
    folder.mkdir(parents=True, exist_ok=True)
    immutable_json(folder / 'request.json', request)
    if (folder / 'initial_guess.npz').exists():
        assert_bitwise_equal(load_npz(folder / 'initial_guess.npz'), {'decision': guess})
    else:
        save_npz(folder / 'initial_guess.npz', {'decision': guess})
    if not (folder / 'attempt.json').exists():
        if (folder / 'solver.log').exists():
            raise RuntimeError(f'Interrupted solver log preserved; inspect before resume: {folder}')
        env = dict(os.environ)
        for name in THREAD_VARIABLES:
            env[name] = '1'
        started = time.perf_counter()
        with (folder / 'solver.log').open('w') as stream:
            process = subprocess.Popen(
                [sys.executable, str(root / 'experiments/09_recoverable_pd.py'),
                 '--worker', str(folder)], cwd=folder, env=env,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            timeout = False
            try:
                process.wait(timeout=config['watchdog_seconds'])
            except subprocess.TimeoutExpired:
                timeout = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        write_json(folder / 'attempt.json', {
            'watchdog_timeout': timeout, 'process_exit_code': process.returncode,
            'process_wall_seconds': time.perf_counter() - started, 'pid': process.pid,
            'thread_environment': {name: env[name] for name in THREAD_VARIABLES},
            'technical_failure': bool(process.returncode != 0 and not timeout),
        })
    record = read_json(folder / 'attempt.json')
    candidate = folder / 'candidate.npz'
    if not candidate.exists():
        candidate = folder / 'last_iterate.npz'
    if not candidate.exists():
        record['candidate_available'] = False
        immutable_json(folder / 'candidate_review.json', record)
        return None, record
    vector = load_npz(candidate)['decision']
    record['candidate_available'] = bool(vector.shape == (600,) and np.isfinite(vector).all())
    record['candidate_source'] = candidate.name
    if (folder / 'solver_result.json').exists():
        record['solver'] = read_json(folder / 'solver_result.json')
    if record['candidate_available']:
        record['diagnostics'] = candidate_diagnostics(vector, request['initial'], config['horizon'])
    immutable_json(folder / 'candidate_review.json', record)
    return (vector if record['candidate_available'] else None), record


def _costs(folder):
    costs = [read_json(path) for path in folder.rglob('*_costs.json')]
    attempts = [read_json(path) for path in folder.glob('*/*/attempt.json')]
    solvers = [read_json(path) for path in folder.glob('*/*/solver_result.json')]
    return {
        'solver_attempts': len(attempts),
        'solver_seconds': sum(item['solver_seconds'] for item in solvers),
        'process_wall_seconds': sum(item['process_wall_seconds'] for item in attempts),
        'watchdog_timeouts': sum(item['watchdog_timeout'] for item in attempts),
        'technical_worker_failures': sum(item['technical_failure'] for item in attempts),
        'ipopt_iterations': sum(item['stats'].get('iter_count', 0) for item in solvers),
        'simulator_steps': sum(int(item['step_attempts']) for item in costs),
        'simulator_resets': sum(int(item['reset_attempts']) for item in costs),
        'simulator_wall_seconds': sum(float(item.get('wall_seconds', 0.0)) for item in costs),
        'nlp_evaluation_counts': {
            key: sum(item['stats'].get(key, 0) for item in solvers)
            for key in ('n_call_nlp_f', 'n_call_nlp_g', 'n_call_nlp_grad_f',
                        'n_call_nlp_jac_g', 'n_call_nlp_hess_l')},
    }


def run_reference_optimization(root, output, initial, seed, pd_arrays):
    """Three E07 starts at grid 44, each repeated; no other optimization allowed."""
    root, output = Path(root).resolve(), Path(output).resolve()
    assert output == root / OUTPUT_RELATIVE
    initial = np.asarray(initial, dtype=np.float64)
    assert initial.tobytes() == _reference_initial(root).tobytes()
    validate_execution(pd_arrays, initial)
    config = _configuration(root)
    source_hash = sha256(root / CONFIG_SOURCE)
    folder = output / f'state_{REFERENCE_ID:04d}' / 'optimization'
    folder.mkdir(parents=True, exist_ok=True)
    immutable_json(folder / 'unchanged_E07_configuration.json', config)
    immutable_json(folder / 'protocol.json', {
        'state_id': REFERENCE_ID, 'config_source': CONFIG_SOURCE,
        'config_source_sha256': source_hash, 'random_start_seed': RANDOM_START_SEED,
        'random_stream': 'numpy.SeedSequence([909004, 44])',
        'PD_initial_guess': 'Exact Experiment 01 observation/dtype path; supplied recorded raw trajectory.',
        'optimization_equations_and_settings_changed': False,
        'candidate_modes': ['open_loop', 'tracking'], 'repeat_each_start': True,
    })
    pd_metrics = episode_metrics(pd_arrays, config['discount'], config['dt'])
    records, repetitions = [], []
    for start_index, name in enumerate(config['starts']):
        initialization = folder / f'initialization_{name}.npz'
        if not initialization.exists():
            if name == 'PD':
                arrays = pd_arrays
            else:
                controls = np.zeros(200) if name == 'zero' else np.repeat(
                    np.random.default_rng(np.random.SeedSequence(
                        [RANDOM_START_SEED, REFERENCE_ID])).uniform(-2, 2, size=10), 20)
                arrays, costs = evaluate_controls(initial, seed, controls)
                write_json(folder / f'initialization_{name}_costs.json', costs)
            save_npz(initialization, arrays)
        arrays = load_npz(initialization)
        validate_execution(arrays, initial)
        if name == 'PD':
            assert_bitwise_equal(arrays, pd_arrays)
        guess = pack(arrays['raw_states'][0], arrays['actions'][0, :, 0])
        request = {
            'state_id': REFERENCE_ID, 'initial': initial.tolist(),
            'initial_hex': [float(x).hex() for x in initial],
            'initial_raw_bytes_hex': initial.tobytes().hex(), 'start': name,
            'authorized_pilot_ids': PILOT_IDS, 'config_source_sha256': source_hash,
            'random_start_seed': RANDOM_START_SEED,
        }
        attempt_folder = folder / name
        vector, attempt = _run_attempt(root, attempt_folder / 'original', request, guess, config)
        repeated, repeat = _run_attempt(root, attempt_folder / 'repeat', request, guess, config)
        status = attempt.get('solver', {}).get('stats', {}).get('return_status', 'no_solver_result')
        repeat_status = repeat.get('solver', {}).get('stats', {}).get('return_status', 'no_solver_result')
        repetition = {
            'state_id': REFERENCE_ID, 'start': name,
            'same_candidate_availability': (vector is None) == (repeated is None),
            'same_status': status == repeat_status,
            'original_status': status, 'repeat_status': repeat_status,
            'original_watchdog_timeout': attempt['watchdog_timeout'],
            'repeat_watchdog_timeout': repeat['watchdog_timeout'],
            'technical_worker_failure': attempt['technical_failure'] or repeat['technical_failure'],
        }
        if vector is not None and repeated is not None:
            repetition.update(decision_bitwise=vector.tobytes() == repeated.tobytes(),
                              maximum_decision_difference=float(np.max(np.abs(vector - repeated))))
        repetition['passed'] = bool(
            repetition['same_candidate_availability'] and repetition['same_status']
            and repetition.get('decision_bitwise', vector is None)
            and not repetition['original_watchdog_timeout']
            and not repetition['repeat_watchdog_timeout']
            and not repetition['technical_worker_failure'])
        immutable_json(attempt_folder / 'reproducibility.json', repetition)
        repetitions.append(repetition)
        if vector is None:
            records.append({'state_id': REFERENCE_ID, 'start': name,
                            'candidate_available': False, 'solver_status': status,
                            'watchdog_timeout': attempt['watchdog_timeout'],
                            'technical_worker_failure': attempt['technical_failure']})
            continue
        states, controls = unpack(vector, initial, 200)
        diagnostics = attempt['diagnostics']
        for mode in ('open_loop', 'tracking'):
            arrays, metrics = cached_evaluation(
                attempt_folder / mode, initial, seed, controls, config,
                states if mode == 'tracking' else None)
            difference = np.abs(arrays['raw_states'][0] - states)
            record = dict(
                state_id=REFERENCE_ID, start=name, start_index=start_index, mode=mode,
                candidate_available=True, solver_status=status,
                solver_success=bool(attempt.get('solver', {}).get('stats', {}).get('success', False)),
                watchdog_timeout=attempt['watchdog_timeout'],
                technical_worker_failure=attempt['technical_failure'], **diagnostics, **metrics,
                delta_return=metrics['returns'] - float(pd_metrics['returns'][0]),
                delta_discounted_return=metrics['discounted_returns'] - float(pd_metrics['discounted_returns'][0]))
            record['maximum_plan_replay_state_discrepancy'] = difference.max(axis=0).tolist()
            record['terminal_plan_replay_state_discrepancy'] = difference[-1].tolist()
            records.append(record)
        print(f'Grid 44, optimization {name}: {status}; paired decision bitwise={repetition.get("decision_bitwise")}', flush=True)
    assert sha256(root / CONFIG_SOURCE) == source_hash
    costs = _costs(folder)
    assert costs['solver_attempts'] == 6
    result = {
        'state_id': REFERENCE_ID, 'config_source_sha256': source_hash,
        'configuration_unchanged': read_json(folder / 'unchanged_E07_configuration.json') == config,
        'records': records, 'repetitions': repetitions, 'costs': costs,
        'passed': all(item['passed'] for item in repetitions) and not costs['technical_worker_failures'],
        'interpretation': 'Reference-state technical checks only. Witnesses are finite-horizon simulator evidence; failed optimization is inconclusive. Open-loop and raw-state feedback tracking are separate modes.',
    }
    immutable_json(folder / 'summary.json', result)
    return result
