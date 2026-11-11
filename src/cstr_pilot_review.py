"""Read-only reconstruction of the stopped CSTR pilot; no solves/integration."""
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import json
import numpy as np
import casadi as ca
from src import cstr_verification as v
from src.cstr_expert_pilot import check_seal, verify_identities, compare_npz
from src.quadrotor_pilot import read, sha


def blocked(*args, **kwargs):
    raise RuntimeError('Read-only review prohibits optimization/integration')


def distribution(values):
    a = np.asarray(values)
    return {k: float(z) for k, z in zip(
        ['min', 'median', 'p95', 'p99', 'max', 'sum'],
        [a.min(), np.median(a), np.percentile(a, 95), np.percentile(a, 99), a.max(), a.sum()])}


def verify(root, *, full_preservation=False):
    root = Path(root).resolve()
    out = root / 'results/cstr_expert_pilot/technical_v1'
    config = check_seal(root, out)
    identities = verify_identities(root, config)
    ledger = read(out / 'resource_ledger.json')
    assert ledger['halted'] and ledger['actual'] == dict(solves=401, intervals=1200, monitors=12000)
    assert ledger['reserved'] == dict(solves=500, intervals=1300, monitors=12000)
    assert read(out / 'first_failure.json')['job'] == 'C2_original_scientific'
    snapshot = read(out / 'preservation_start.json')
    for group in ['tracked_sha256', 'untracked_sha256']:
        for n, h in snapshot[group].items():
            assert sha(root / n) == h, n
    stats_max = {k: 0. for k in ['primal_scaled_inf', 'scaled_stationarity_inf',
        'scaled_complementarity_inf', 'dual_sign_inf', 'objective_reconstruction_error',
        'constraint_reconstruction_error', 'initial_state_parameter_error']}
    cases, timers, native_timers, iterations, integration_timers = {}, [], [], [], []
    program = v.independent_program()
    load = lambda path: dict(np.load(path, allow_pickle=False))
    checked = 0
    for case in ['C0', 'C1', 'C2']:
        for phase in (['original', 'repeat'] if case != 'C2' else ['original']):
            f = out / 'runs' / f'{case}_{phase}_scientific'
            a = load(f / 'trajectory.npz'); x, u = a['states'], a['commands']
            initial = next(row['values'] for row in config['cases'] if row['id'] == case)
            assert x[0].tobytes() == np.asarray(initial).tobytes()
            attempts = sorted((f / 'attempts').glob('[0-9][0-9][0-9][0-9]'))
            assert len(attempts) == (1 if case == 'C2' else 100)
            for k, attempt in enumerate(attempts):
                candidate = load(attempt / 'candidate.npz'); before = load(attempt / 'prequery_memory.npz')
                metrics = v.candidate_metrics(candidate, program); native = read(attempt / 'native_stats.json')
                v.require_candidate(metrics, native, config)
                assert candidate['p'][:4].tobytes() == x[k].tobytes()
                previous = np.asarray(config['previous_input']) if k == 0 else u[k-1]
                assert candidate['p'][4:].tobytes() == previous.tobytes()
                assert before['previous_u'].tobytes() == previous.tobytes()
                assert abs(float(before['native_time'][0]) - k*.005) < 1e-14
                if k == 0:
                    assert not bool(before['initial_run'])
                saved = read(attempt / 'independent_metrics.json')
                for key in stats_max:
                    assert metrics[key] == saved[key], key
                    stats_max[key] = max(stats_max[key], metrics[key])
                checked += 1
                iterations.append(native['iter_count'])
                timers.append(read(attempt / 'solver_timing.json')['solver_wall'])
                if case == 'C2':
                    command = np.asarray(metrics['command'])
                    margins = np.minimum(command-v.ULO, v.UHI-command)
                    assert margins[0] < 0 and margins[1] >= 0
                    assert x.shape == (1,4) and u.shape == (0,2)
                    assert not (attempt/'postquery_memory.npz').exists()
                    assert not (attempt/'native_timing.json').exists()
                    assert not (f/'integrations').exists()
                    failure = dict(command=command.tolist(), binary64_hex=[float(t).hex() for t in command],
                        raw_input_margins=margins.tolist(), metrics=metrics, iterations=native['iter_count'],
                        solver_status=native['return_status'], applied=False,
                        missing=['postquery_memory', 'native_action_selection_timing', 'separate_returned_command_packet'],
                        stale_progress_candidate_count=read(f/'progress.json')['candidate_checks'])
                    continue
                assert np.asarray(metrics['command']).tobytes() == u[k].tobytes()
                v.require_input(u[k])
                post = load(attempt/'postquery_memory.npz')
                assert post['previous_u'].tobytes() == u[k].tobytes()
                native_timers.append(read(attempt/'native_timing.json')['native_action_selection_including_verification_wall'])
            if case == 'C2':
                continue
            assert x.shape == (101,4) and u.shape == (100,2)
            v.require_state(x, config)
            # Reconstruct stage/movement/terminal sums without the runner's cost helper.
            tracking = np.square(x[:-1,:2]-np.array([.7,.6])).sum(axis=1)
            du = u-np.vstack([np.array([5.,0.]),u[:-1]])
            movement = .1*np.square(du[:,0]/100)+.001*np.square(du[:,1]/2000)
            terminal = np.square(x[-1,:2]-np.array([.7,.6])).sum()
            completion = read(f/'completion.json')
            cost = float(tracking.sum()+movement.sum()+terminal)
            assert abs(cost-completion['costs']['total_cost']) < 1e-14
            assert np.allclose(np.sqrt(np.square(x[:-1,:2]-[.7,.6]).mean(0)),
                               completion['costs']['concentration_RMSE'], rtol=0, atol=1e-15)
            replay = out/'runs'/f'{case}_{phase}_native_replay'
            compare_npz(f/'trajectory.npz', replay/'trajectory.npz')
            for folder in [f, replay]:
                for k in range(100):
                    p = load(folder/'integrations'/f'{k:04d}_submitted.npz')
                    q = load(folder/'integrations'/f'{k:04d}_result.npz')
                    assert p['p'][:2].tobytes() == u[k].tobytes()
                    assert p['x0'].reshape(-1).tobytes() == x[k].tobytes()
                    assert q['xf'].reshape(-1).tobytes() == x[k+1].tobytes()
                    integration_timers.append(read(folder/'integrations'/f'{k:04d}_timing.json')['native_integration_wall'])
            independent = out/'runs'/f'{case}_{phase}_independent_replay'
            dense = load(independent/'monitoring_partial.npz')
            assert dense['primary'].shape == (100,11,4) and dense['refinement'].shape == (100,21,4)
            assert np.array_equal(dense['primary'], dense['refinement'][:,::2])
            v.require_state(dense['refinement'], config)
            err = float(np.max(abs(dense['states']-x)/v.WIDTH))
            assert err <= config['normalized_replay_tolerance']
            for k in range(100):
                assert read(independent/f'integration_{k:04d}.json')['success']
            dense_margins = v.state_margins(dense['refinement']).min(axis=(0,1))
            if phase == 'original':
                cases[case] = dict(costs=completion['costs'], raw_boundary_margins=v.state_margins(x).min(0).tolist(),
                    dense_margins=dense_margins.tolist(), raw_input_margins=np.minimum(u-v.ULO,v.UHI-u).min(0).tolist(),
                    normalized_reserve=float(np.min(dense_margins/v.WIDTH)-config['normalized_reserve']),
                    normalized_native_independent_error=err, violations=0)
            else:
                original = out/'runs'/f'{case}_original_scientific'
                compare_npz(original/'trajectory.npz', f/'trajectory.npz')
                for k in range(100):
                    for name in ['submitted.npz','candidate.npz','prequery_memory.npz','postquery_memory.npz']:
                        compare_npz(original/'attempts'/f'{k:04d}'/name,f/'attempts'/f'{k:04d}'/name)
                    st = read(original/'attempts'/f'{k:04d}'/'native_stats.json')
                    rt = read(f/'attempts'/f'{k:04d}'/'native_stats.json')
                    assert all(st[key] == rt[key] for key in ['return_status','success','iter_count'])
    assert checked == 401
    assert sum(j['status']=='complete' for j in ledger['jobs']) == 12 and len(ledger['jobs']) == 13
    assert not list((out/'runs').glob('C3*'))
    returns = [c['costs']['return'] for c in cases.values()]
    result = dict(passed_read_only_verification=True, pilot_acceptance_passed=False,
        new_solves=0, new_integrations=0, identities=identities, candidate_count=checked,
        scientific_complete=4, scientific_denominator=8, replay_complete=8, replay_denominator=16,
        cases=cases, failure=failure, candidate_maxima=stats_max,
        completed_original_only_mean_return=float(np.mean(returns)), completed_original_only_worst_return=min(returns),
        solver_wall_seconds=distribution(timers), native_action_selection_seconds=distribution(native_timers),
        native_action_selection_missing=1, native_integration_seconds=distribution(integration_timers),
        solver_iterations=distribution(iterations), solver_over_18_seconds=int(np.sum(np.array(timers)>18)),
        native_over_18_seconds=int(np.sum(np.array(native_timers)>18)),
        optimizer_initialization_diagnostics_complete=0, optimizer_initialization_denominator=16,
        frozen_source_and_artifact_seal=sha(out/'protocol_seal.json'))
    if full_preservation:
        from src.quadrotor_archive_finalization import historical_preservation
        result['historical_preservation'] = historical_preservation(root, root,
            read(root/'results/benchmark_selection/b0_archive_v1/preservation_start.json'))
    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--full-preservation', action='store_true')
    args = parser.parse_args()
    path = Path(args.output)
    if path.exists():
        raise FileExistsError(path)
    with ExitStack() as stack:
        for name in ['nlpsol', 'integrator']:
            stack.enter_context(patch.object(ca, name, blocked))
        stack.enter_context(patch.object(ca.Opti, 'solve', blocked))
        stack.enter_context(patch('scipy.integrate.solve_ivp', blocked))
        result = verify(Path.cwd(), full_preservation=args.full_preservation)
    path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
