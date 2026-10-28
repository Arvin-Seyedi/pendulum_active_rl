"""Pilot-only orchestration; solver processes are capped and independently replayed."""

import io
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import unittest

import casadi as ca
import gymnasium as gym
import numpy as np

from src.counterfactual import assert_bitwise_equal, write_csv
from src.horizon_experiment import _assert_hashes, immutable_json, load_npz, read_json, save_npz, sha256, write_json
from src.horizon_study import exclusive
from src.policy_evaluation import evaluate_rollouts
from src.recoverability import (
    candidate_diagnostics, evaluate_controls, pack, replay_and_measure, serializable,
    solve_attempt, unpack, validate_execution,
)
from src.recoverability_checks import check_dynamics, select_cases

OUTPUT_RELATIVE = 'results/recoverability/pilot'


def protected_manifest(root, output):
    files = [p for p in root.iterdir() if p.is_file()]
    for name in ('src', 'experiments', 'configs', 'tests', 'docs', 'data', 'results'):
        files.extend(p for p in (root / name).rglob('*') if p.is_file() and output not in p.parents
                     and '__pycache__' not in p.parts and p.name != '.DS_Store')
    return {'sha256': {str(p.relative_to(root)): sha256(p) for p in sorted(set(files))}}


def worker(root, folder):
    folder = Path(folder).resolve()
    assert (root / OUTPUT_RELATIVE).resolve() in folder.parents
    request = read_json(folder / 'request.json')
    assert request['state_id'] in request['authorized_pilot_ids']
    assert len(request['authorized_pilot_ids']) == 4
    config = read_json(root / 'configs/experiment_07.json')
    assert not config['full_study_authorized'] and config['horizon'] == 200
    initial = np.array(request['initial'], dtype=np.float64)
    guess = load_npz(folder / 'initial_guess.npz')['decision']
    solve_attempt(folder, config, initial, guess)


def run_attempt(root, folder, request, guess, config):
    folder.mkdir(parents=True, exist_ok=True)
    immutable_json(folder / 'request.json', request)
    if (folder / 'initial_guess.npz').exists():
        assert_bitwise_equal(load_npz(folder / 'initial_guess.npz'), {'decision': guess})
    else:
        save_npz(folder / 'initial_guess.npz', {'decision': guess})
    if not (folder / 'attempt.json').exists():
        # Refuse to silently overwrite an interrupted solver log/candidate.
        if (folder / 'solver.log').exists():
            raise RuntimeError(f'Interrupted worker preserved at {folder}; inspect before resuming.')
        env = dict(os.environ)
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            env[name] = '1'
        started = time.perf_counter()
        with (folder / 'solver.log').open('w') as stream:
            process = subprocess.Popen([sys.executable, str(root / 'experiments/07_recoverability.py'),
                                        '--worker', str(folder)], cwd=folder, env=env,
                                       stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            timeout = False
            try:
                process.wait(timeout=config['watchdog_seconds'])
            except subprocess.TimeoutExpired:
                timeout = True
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        record = {'watchdog_timeout': timeout, 'process_exit_code': process.returncode,
                  'process_wall_seconds': time.perf_counter() - started, 'pid': process.pid,
                  'thread_environment': {name: env[name] for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')}}
        write_json(folder / 'attempt.json', record)
    record = read_json(folder / 'attempt.json')
    if record['process_exit_code'] != 0 and not record['watchdog_timeout']:
        raise RuntimeError(f'Technical worker failure; inspect preserved log: {folder / "solver.log"}')
    candidate = folder / 'candidate.npz'
    if not candidate.exists():
        candidate = folder / 'last_iterate.npz'
    if not candidate.exists():
        return None, {**record, 'candidate_available': False}
    vector = load_npz(candidate)['decision']
    record['candidate_available'] = bool(vector.shape == (600,) and np.isfinite(vector).all())
    record['candidate_source'] = candidate.name
    if (folder / 'solver_result.json').exists():
        record['solver'] = read_json(folder / 'solver_result.json')
    if not record['candidate_available']:
        return None, record
    record['diagnostics'] = candidate_diagnostics(vector, request['initial'], config['horizon'])
    write_json(folder / 'candidate_review.json', record)
    return vector, record


def cached_evaluation(folder, initial, seed, actions, config, reference=None):
    folder.mkdir(parents=True, exist_ok=True)
    if not (folder / 'rollout.npz').exists():
        arrays, costs = evaluate_controls(initial, seed, actions, reference)
        write_json(folder / 'execution_costs.json', costs)
        save_npz(folder / 'rollout.npz', arrays)
    arrays = load_npz(folder / 'rollout.npz')
    validate_execution(arrays, initial)
    # Independently reconstruct every command from the saved raw states.
    requested = np.asarray(actions, dtype=np.float64).copy()
    if reference is not None:
        error = arrays['raw_states'][0, :-1] - reference[:-1]
        requested -= config['tracking']['kp'] * error[:, 0] + config['tracking']['kd'] * error[:, 1]
    expected = np.clip(requested, -2, 2).astype(np.float32)
    np.testing.assert_array_equal(arrays['actions'][0, :, 0], expected)
    if not (folder / 'replay.npz').exists():
        replay, costs, metrics = replay_and_measure(arrays, initial, seed, config, reference)
        write_json(folder / 'replay_costs.json', costs)
        write_json(folder / 'metrics.json', metrics)
        save_npz(folder / 'replay.npz', replay)
    assert_bitwise_equal(arrays, load_npz(folder / 'replay.npz'))
    return arrays, read_json(folder / 'metrics.json')


def run_pilot(root, resume=False, snapshot=None):
    root = Path(root).resolve()
    output = root / OUTPUT_RELATIVE
    if output.exists() and not resume:
        raise FileExistsError('Pilot output exists; inspect and use --resume.')
    output.mkdir(parents=True, exist_ok=True)
    with exclusive(output):
        config = read_json(root / 'configs/experiment_07.json')
        assert config['authorized_phase'] == 'four_case_technical_pilot_only'
        assert config['full_study_authorized'] is False and ca.__version__ == '3.7.2'
        immutable_json(output / 'protocol.json', config)
        sources = ['src/recoverability.py', 'src/recoverability_checks.py', 'src/recoverability_experiment.py',
                   'experiments/07_recoverability.py', 'tests/test_recoverability.py', 'configs/experiment_07.json',
                   'requirements_experiment_07.txt', 'src/expert.py', 'src/safety.py', 'src/active_query.py',
                   'src/dataset.py', 'src/policy_evaluation.py', 'src/rollout_horizon.py', 'src/counterfactual.py']
        hashes = {name: sha256(root / name) for name in sources}
        immutable_json(output / 'source_hashes.json', hashes)
        if not (output / 'preservation_before.json').exists():
            write_json(output / 'preservation_before.json', read_json(snapshot) if snapshot else protected_manifest(root, output))
        before = read_json(output / 'preservation_before.json')
        _assert_hashes(root, before['sha256'])
        if (output / 'completion.json').exists():
            _assert_hashes(output, read_json(output / 'completion.json')['artifacts_sha256'])
            print('Completed pilot verified; no solver or simulation rerun.', flush=True)
            return
        cases = select_cases(root, config)
        immutable_json(output / 'state_manifest.json', cases)
        immutable_json(output / 'metadata.json', {'python': sys.version, 'executable': sys.executable,
                       'numpy': np.__version__, 'gymnasium': gym.__version__, 'casadi': ca.__version__,
                       'platform': platform.platform(), 'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                       'full_study_executed': False, 'pilot_case_ids': cases['pilot_ids_in_order'],
                       'dynamics_rounding_note': 'Symbolic arithmetic is float64; actual Gymnasium receives float32 torques. Full independent replay decides safety.'})
        print('Frozen pilot IDs: ' + str(cases['pilot_ids_in_order']), flush=True)
        if not (output / 'technical_tests.json').exists():
            stream = io.StringIO()
            suite = unittest.defaultTestLoader.discover(str(root / 'tests'), pattern='test_recoverability.py')
            test_result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
            (output / 'technical_tests.txt').write_text(stream.getvalue())
            write_json(output / 'technical_tests.json', {'passed': test_result.wasSuccessful(), 'tests': test_result.testsRun,
                       'failures': len(test_result.failures), 'errors': len(test_result.errors), 'simulator_interactions': 0})
        assert read_json(output / 'technical_tests.json')['passed']
        if not (output / 'model_verification.json').exists():
            checks, arrays, costs = check_dynamics(config)
            save_npz(output / 'model_verification.npz', arrays)
            write_json(output / 'model_verification_costs.json', costs)
            write_json(output / 'model_verification.json', checks)
        assert read_json(output / 'model_verification.json')['passed']
        print('Dynamics, clipping, and derivative checks passed.', flush=True)
        prior_PD = load_npz(root / config['source_experiment'] / 'PD_reference_replay.npz')
        records, baselines, repetitions = [], [], []
        for case in cases['pilot_cases']:
            sid, initial, seed = case['state_id'], np.array(case['initial']), case['reset_seed']
            folder = output / f'state_{sid:03d}'
            folder.mkdir(exist_ok=True)
            if not (folder / 'PD.npz').exists():
                start = time.perf_counter()
                pd_arrays, _ = evaluate_rollouts(None, initial[None], seed, config['horizon'])
                write_json(folder / 'PD_execution_costs.json', {'reset_attempts': 1, 'step_attempts': 200,
                           'wall_seconds': time.perf_counter() - start, 'count_basis': 'Preserved evaluator, validated full horizon'})
                save_npz(folder / 'PD.npz', pd_arrays)
            pd_arrays = load_npz(folder / 'PD.npz')
            assert_bitwise_equal(pd_arrays, {k: v[[sid]] for k, v in prior_PD.items()})
            if not (folder / 'PD_replay.npz').exists():
                replay, costs, metrics = replay_and_measure(pd_arrays, initial, seed, config)
                write_json(folder / 'PD_replay_costs.json', costs)
                write_json(folder / 'PD_metrics.json', metrics)
                save_npz(folder / 'PD_replay.npz', replay)
            assert_bitwise_equal(pd_arrays, load_npz(folder / 'PD_replay.npz'))
            pd_metrics = read_json(folder / 'PD_metrics.json')
            assert pd_metrics['violated'] == case['PD_unsafe']
            baselines.append({**case, **pd_metrics})
            for start_index, name in enumerate(config['starts']):
                initialization = folder / f'initialization_{name}.npz'
                if not initialization.exists():
                    if name == 'PD':
                        init_arrays = pd_arrays
                    else:
                        controls = np.zeros(200) if name == 'zero' else np.repeat(np.random.default_rng(
                            np.random.SeedSequence([config['seeds']['random_start'], sid])).uniform(-2, 2, size=10), 20)
                        init_arrays, init_costs = evaluate_controls(initial, seed, controls)
                        write_json(folder / f'initialization_{name}_costs.json', init_costs)
                    save_npz(initialization, init_arrays)
                init_arrays = load_npz(initialization)
                guess = pack(init_arrays['raw_states'][0], init_arrays['actions'][0, :, 0])
                request = dict(state_id=sid, initial=initial.tolist(), start=name,
                               authorized_pilot_ids=cases['pilot_ids_in_order'])
                attempt_folder = folder / name
                vector, attempt = run_attempt(root, attempt_folder / 'original', request, guess, config)
                repeated, repeat = run_attempt(root, attempt_folder / 'repeat', request, guess, config)
                repetition = {'state_id': sid, 'start': name, 'same_candidate_availability': (vector is None) == (repeated is None),
                              'original_watchdog_timeout': attempt['watchdog_timeout'], 'repeat_watchdog_timeout': repeat['watchdog_timeout']}
                if vector is not None and repeated is not None:
                    repetition.update(decision_bitwise=vector.tobytes() == repeated.tobytes(),
                                      maximum_decision_difference=float(np.max(np.abs(vector - repeated))))
                a_status = attempt.get('solver', {}).get('stats', {}).get('return_status', 'no_solver_result')
                b_status = repeat.get('solver', {}).get('stats', {}).get('return_status', 'no_solver_result')
                repetition['same_status'] = a_status == b_status
                repetition['passed'] = bool(repetition['same_candidate_availability'] and repetition['same_status']
                                            and repetition.get('decision_bitwise', vector is None)
                                            and not attempt['watchdog_timeout'] and not repeat['watchdog_timeout'])
                repetitions.append(repetition)
                write_json(attempt_folder / 'reproducibility.json', repetition)
                if vector is None:
                    records.append({'state_id': sid, 'start': name, 'candidate_available': False, 'solver_status': a_status})
                    continue
                states, controls = unpack(vector, initial, 200)
                diagnostics = attempt['diagnostics']
                for mode in ('open_loop', 'tracking'):
                    arrays, metrics = cached_evaluation(attempt_folder / mode, initial, seed, controls, config,
                                                       states if mode == 'tracking' else None)
                    # Compare the plan for BOTH execution modes, not only tracking.
                    error = np.abs(arrays['raw_states'][0] - states)
                    record = dict(state_id=sid, group=case['group'], start=name, start_index=start_index, mode=mode,
                                  candidate_available=True, solver_status=a_status,
                                  solver_success=bool(attempt.get('solver', {}).get('stats', {}).get('success', False)),
                                  watchdog_timeout=attempt['watchdog_timeout'], **diagnostics, **metrics,
                                  delta_return=metrics['returns'] - pd_metrics['returns'],
                                  delta_discounted_return=metrics['discounted_returns'] - pd_metrics['discounted_returns'])
                    record['maximum_plan_replay_state_discrepancy'] = error.max(axis=0).tolist()
                    record['terminal_plan_replay_state_discrepancy'] = error[-1].tolist()
                    records.append(record)
                print(f"State {sid}, start {name}: {a_status}; repeated decision exact={repetition.get('decision_bitwise')}", flush=True)
        summary = summarize(cases, baselines, records, repetitions)
        write_json(output / 'summary.json', summary)
        write_csv(output / 'candidate_metrics.csv', [r for r in records if r.get('candidate_available')])
        write_csv(output / 'PD_metrics.csv', baselines)
        costs = collect_costs(output)
        write_json(output / 'interaction_costs.json', costs)
        _assert_hashes(root, before['sha256'])
        _assert_hashes(root, hashes)
        verification = {'technical_tests': read_json(output / 'technical_tests.json'),
                        'dynamics': read_json(output / 'model_verification.json'),
                        'solver_repetitions': repetitions, 'independent_execution_and_replay_passed': True,
                        'PD_matches_06A_bitwise': True, 'preservation_passed': True,
                        'protected_files': len(before['sha256']), 'executed_case_ids': cases['pilot_ids_in_order'],
                        'full_study_launched': False, 'solver_attempts': len(list(output.glob('state_*/*/*/attempt.json')))}
        verification['passed'] = all(r['passed'] for r in repetitions)
        write_json(output / 'verification.json', verification)
        plot_pilot(output, cases)
        report = pilot_report(summary, verification, costs)
        (output / 'pilot_report.md').write_text(report)
        report_path = root / 'docs/experiment_07_pilot_report.md'
        if report_path.exists() and report_path.read_text() != report:
            raise RuntimeError('Existing report differs; preserve and review it before replacing.')
        report_path.write_text(report)
        artifacts = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                     if p.is_file() and p.name not in ('.lock', 'completion.json')}
        write_json(output / 'completion.json', {'pilot_finished': True, 'verification_passed': verification['passed'],
                   'full_study_authorized': False, 'artifacts_sha256': artifacts})
        print(f"Pilot finished. Verification passed={verification['passed']}; full study not launched.", flush=True)


def summarize(cases, baselines, records, repetitions):
    outcomes = []
    for case, pd in zip(cases['pilot_cases'], baselines):
        entry = dict(case, PD=pd, modes={})
        for mode in ('open_loop', 'tracking'):
            available = [r for r in records if r['state_id'] == case['state_id'] and r.get('mode') == mode]
            best = max(available, key=lambda r: (r['minimum_margins'], r['returns'], -r['start_index'])) if available else None
            entry['modes'][mode] = {'witnesses': sum(r['witness'] for r in available), 'candidates': len(available),
                                    'any_witness': any(r['witness'] for r in available), 'oracle_best': best}
        outcomes.append(entry)
    return {'cases': outcomes, 'all_candidates': records, 'solver_repetitions': repetitions,
            'interpretation': 'Only finite-horizon empirical witnesses; failed optimization is inconclusive. Best candidate selection uses observed outcomes and is an oracle diagnostic.'}


def collect_costs(output):
    categories = {}
    for p in output.rglob('*costs.json'):
        item = read_json(p)
        name = 'model_verification' if p.name == 'model_verification_costs.json' else 'initialization' if p.name.startswith('initialization_') else 'replay' if 'replay' in p.name else 'evaluation'
        group = categories.setdefault(name, {'steps': 0, 'resets': 0, 'wall_seconds': 0.})
        group['steps'] += int(item['step_attempts']); group['resets'] += int(item['reset_attempts'])
        group['wall_seconds'] += float(item.get('wall_seconds', sum(item.get('step_seconds', [])) + sum(item.get('reset_seconds', []))))
    attempts = [read_json(p) for p in output.glob('state_*/*/*/attempt.json')]
    solvers = [read_json(p) for p in output.glob('state_*/*/*/solver_result.json')]
    return {'simulator_categories': categories, 'total_simulator_steps': sum(c['steps'] for c in categories.values()),
            'total_resets': sum(c['resets'] for c in categories.values()), 'solver_attempts': len(attempts),
            'solver_process_seconds': sum(a['process_wall_seconds'] for a in attempts),
            'solver_seconds': sum(s['solver_seconds'] for s in solvers),
            'watchdog_timeouts': sum(a['watchdog_timeout'] for a in attempts),
            'ipopt_iterations_total': sum(s['stats'].get('iter_count', 0) for s in solvers),
            'nlp_evaluation_counts': {k: sum(s['stats'].get(k, 0) for s in solvers) for k in ('n_call_nlp_f', 'n_call_nlp_g', 'n_call_nlp_grad_f', 'n_call_nlp_jac_g', 'n_call_nlp_hess_l')}}


def plot_pilot(output, cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(4, 2, figsize=(12, 11), sharex=True, constrained_layout=True)
    for row, case in enumerate(cases['pilot_cases']):
        folder = output / f"state_{case['state_id']:03d}"
        pd = load_npz(folder / 'PD.npz')
        for col, mode in enumerate(('open_loop', 'tracking')):
            ax = axes[row, col]
            ax.plot(np.arange(201)*.05, pd['margins'][0], color='black', label='PD')
            for start in ('PD', 'zero', 'random_blocks'):
                path = folder / start / mode / 'rollout.npz'
                if path.exists():
                    ax.plot(np.arange(201)*.05, load_npz(path)['margins'][0], label=start, alpha=.8)
            ax.axhline(0., color='red', linestyle='--', linewidth=.7)
            ax.set_title(f"State {case['state_id']} ({case['group']}): {mode}")
            ax.set_ylabel('Minimum normalized raw margin')
            if row == 0: ax.legend(fontsize=8)
            if row == 3: ax.set_xlabel('Time (s)')
    fig.savefig(output / 'raw_safety_margins.png', dpi=140)
    plt.close(fig)


def pilot_report(summary, checks, costs):
    lines = ['# Experiment 07 — Safety Recoverability Technical Pilot', '',
             'Only the four predefined cases were run. The 57-state study remains unauthorized. No training data were generated for RL or added to existing datasets.', '',
             '## Fixed method', '',
             'CasADi 3.7.2 / IPOPT / MUMPS, direct multiple shooting with 600 variables and 400 dynamics equalities, exact initial-state parameter, 200 independent torques, hard |theta|≤1, |omega|≤3, |u|≤2 constraints. Objective: original undiscounted reward cost; no terminal cost/constraint, safety penalty, slack acceptance, or tuning. Three starts (PD, zero, ten seeded random 20-step blocks); each solve repeated independently. Limits: 1000 iterations, 30 s solver wall time, 60 s process watchdog, tolerance and constraint tolerance 1e-8, no acceptable-termination heuristic or bound relaxation. All defaults/options and solver logs are retained.', '',
             'Open-loop executes clipped float32 optimized torques. Tracking separately applies clip(u_ref − 8*(theta−theta_ref) − 2*(omega−omega_ref),−2,2) using raw states. Each executed action sequence is replayed independently in Gymnasium. Safety uses all 201 raw states and strictly negative margin indicates violation; every episode continues through 200 steps.', '',
             '## Pilot results', '',
             '| Original state ID / group | Initial theta, omega | PD return / discounted / min margin | Open-loop witnesses | Tracking witnesses |',
             '|---|---|---|---:|---:|']
    for c in summary['cases']:
        p = c['PD']; ol, tr = c['modes']['open_loop'], c['modes']['tracking']
        lines.append(f"| {c['state_id']} / {c['group']} | {c['initial'][0]:.9f}, {c['initial'][1]:.9f} | {p['returns']:.6f} / {p['discounted_returns']:.6f} / {p['minimum_margins']:.6f} | {ol['witnesses']}/{ol['candidates']} | {tr['witnesses']}/{tr['candidates']} |")
    lines += ['', 'A witness requires independent Gymnasium safety AND exact recorded-action replay. It is not inferred from solver success or the optimizer’s state trajectory.', '',
              '| State | Start | Solver status | Dynamics residual | Mode | Return | Discounted return | Minimum raw margin | First crossing (s) | Max plan/replay angle, speed error |',
              '|---|---|---|---:|---|---:|---:|---:|---:|---|']
    for r in summary['all_candidates']:
        if not r.get('candidate_available'): continue
        crossing = 'none' if r['first_crossing_seconds'] is None else f"{r['first_crossing_seconds']:.2f}"
        lines.append(f"| {r['state_id']} | {r['start']} | {r['solver_status']} | {r['maximum_dynamics_residual']:.2e} | {r['mode']} | {r['returns']:.6f} | {r['discounted_returns']:.6f} | {r['minimum_margins']:.8f} | {crossing} | " + ', '.join(f'{v:.3e}' for v in r['maximum_plan_replay_state_discrepancy']) + ' |')
    m = checks['dynamics']
    lines += ['', '## Verification and costs', '',
              f"- Overall technical verification passed: {checks['passed']}. {checks['technical_tests']['tests']} pure technical tests passed; no RL training was run.",
              f"- Feasible-domain one-step checks: {m['random_feasible_domain_points']} random points plus {m['boundary_points']} boundary cases; maximum symbolic/Gym discrepancy {m['max_smooth_error_feasible_domain']:.3e} (tolerance 1e-7).",
              f"- In {m['speed_clipping_cases']} outside-domain cases, speed clipping activated: the smooth model differed by up to {m['max_unclipped_error_outside_domain']:.6f}; the explicitly clipped symbolic model matched Gym to {m['max_clipped_model_error_all_cases']:.3e}. No feasible-domain equivalence was claimed outside its domain.",
              '- Maximum constraint-Jacobian / cost-gradient / cost-Hessian central-difference errors: ' + ', '.join(f'{x:.3e}' for x in m['max_derivative_errors_jacobian_gradient_hessian']) + '.',
              '- Verified state/action indexing, fixed exact initial states, executed float32 controls, all 201 raw safety measurements, preserved PD baseline equality, and bitwise replay for every evaluated trajectory.',
              f"- All repeated solver decisions bitwise identical: {all(r.get('decision_bitwise', False) for r in checks['solver_repetitions'])}. Solver statuses and all repeat records are preserved; timing-related failures are not treated as reproducibility success.",
              f"- {costs['solver_attempts']} solver attempts including repeats; {costs['ipopt_iterations_total']} total IPOPT iterations; {costs['solver_seconds']:.3f} solver seconds and {costs['solver_process_seconds']:.3f} worker-process seconds. External watchdog timeouts: {costs['watchdog_timeouts']}.",
              f"- Actual Gymnasium use: {costs['total_simulator_steps']} steps and {costs['total_resets']} resets, including model tests, initial guesses, evaluation, and replay. Detailed categories and NLP evaluation counts are in interaction_costs.json.",
              f"- All {checks['protected_files']} protected pre-existing files are SHA-256 unchanged. Experiments 01–06A and unfinished IQL diagnostics were preserved.", '',
              '## Scientific interpretation and limits', '',
              'These results establish only which sampled initial states yielded a replay-verified, finite-horizon control-sequence witness within the fixed search. Optimization failures and missing witnesses are inconclusive about physical unrecoverability. The PD-safe cases already have a feasible baseline; optimizer failure there cannot make them unrecoverable.', '',
              'Open-loop and tracking outcomes are separate. Tracking is an additional diagnostic mechanism using fixed gains, not a certified backup or safety filter. A stable-looking optimized state sequence can disagree with an executed open-loop trajectory because dynamics residuals, float32 torque arithmetic, and rounding accumulate. Those discrepancies are measured and do not override actual raw-state safety.', '',
              'Best-of-start summaries in summary.json use observed minimum margin, then return, and are oracle diagnostics, not deployable policy performance. No safety, viability, robustness, or recoverability guarantee follows outside the exact sampled states, nominal discrete simulator, and 200-step horizon. Full-study execution, algorithm tuning, RL training, commits, and pushes remain outside this pilot.', '']
    return '\n'.join(lines)
