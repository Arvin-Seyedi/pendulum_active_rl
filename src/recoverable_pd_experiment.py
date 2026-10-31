"""Pilot-only Experiment 09 orchestration; prior scientific code is read-only."""

import io
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import unittest

import casadi as ca
import gmpy2 as g
import gymnasium as gym
import numpy as np

from src.counterfactual import assert_bitwise_equal, write_csv
from src.horizon_experiment import (
    _assert_hashes, immutable_json, load_npz, read_json, save_npz, sha256, write_json,
)
from src.horizon_study import exclusive
from src.recoverability import replay_and_measure, validate_execution
from src.recoverability_experiment import cached_evaluation, protected_manifest
from src.recoverable_pd import (
    GAINS, PILOT_IDS, case_identity, compare_archive, evaluate_gain, load_benchmark,
)
from src.recoverable_pd_solver import run_reference_optimization
from src.viability_experiment import _attempt, compare_simulator


CONFIG = 'configs/experiment_09_pilot.json'
OUTPUT = 'results/recoverable_pd/pilot'
ADDITIONS = [CONFIG, 'requirements_experiment_09.txt', 'src/recoverable_pd.py',
             'src/recoverable_pd_solver.py', 'src/recoverable_pd_experiment.py',
             'src/recoverable_pd_audit.py', 'experiments/09_recoverable_pd.py',
             'tests/test_recoverable_pd.py', 'tests/test_recoverable_pd_solver.py']


def settings(root):
    config = read_json(root / CONFIG)
    assert config['authorized_phase'] == 'five_case_technical_pilot_only'
    assert config['pilot_ids_in_order'] == PILOT_IDS
    assert config['output'] == OUTPUT and config['optimization_grid_id'] == 44
    assert config['horizon'] == 200 and config['dt'] == .05
    assert config['discount'] == .99 and config['torque_limit'] == 2
    assert config['theta_limit'] == 1 and config['omega_limit'] == 3
    assert tuple((kp, kd) for kp in config['kp_in_order'] for kd in config['kd_in_order']) == GAINS
    assert config['interval_actions_in_order'] == [-2, 2]
    assert config['precision_bits_in_order'] == [128, 256, 512]
    assert config['seeds'] == {'environment': 909005, 'random_start': 909004}
    assert config['optimization_scientific_config'] == 'configs/experiment_07.json'
    assert not any(config[k] for k in ('full_study_authorized', 'additional_unresolved_case_authorized',
                                      'viability_grid_authorized', 'training_data_use'))
    assert config['real_model_only'] and not config['implementation_error_bound_established']
    assert ca.__version__ == '3.7.2' and g.version() == '2.3.2'
    return config, root / OUTPUT


def verify_completed(root):
    root = Path(root).resolve()
    _, output = settings(root)
    done = read_json(output / 'completion.json')
    assert done['passed'] and done['case_ids'] == PILOT_IDS
    _assert_hashes(output, done['artifacts_sha256'])
    _assert_hashes(root, read_json(output / 'source_hashes.json'))
    _assert_hashes(root, read_json(output / 'preservation_before.json')['sha256'])
    assert sha256(root / 'docs/experiment_09_pilot_report.md') == done['report_sha256']
    print('Completed five-case E09 pilot verified; no simulations or solves rerun.', flush=True)


def gain_name(kp, kd):
    return f'kp_{kp:g}_kd_{kd:g}'


def cached_gain(folder, initial, seed, kp, kd, config):
    folder.mkdir(parents=True, exist_ok=True)
    immutable_json(folder / 'identity.json', {'initial_hex': [float(x).hex() for x in initial],
                   'seed': seed, 'gains': [kp, kd]})
    if not (folder / 'rollout.npz').exists():
        arrays, costs, diagnostics = evaluate_gain(initial, seed, kp, kd)
        save_npz(folder / 'rollout.npz', arrays)
        write_json(folder / 'execution_costs.json', costs)
        write_json(folder / 'diagnostics.json', diagnostics)
    arrays = load_npz(folder / 'rollout.npz')
    validate_execution(arrays, initial)
    diagnostics = read_json(folder / 'diagnostics.json')
    if not (folder / 'policy_repeat.npz').exists():
        repeated, costs, repeated_diagnostics = evaluate_gain(initial, seed, kp, kd)
        save_npz(folder / 'policy_repeat.npz', repeated)
        write_json(folder / 'policy_repeat_costs.json', costs)
        write_json(folder / 'policy_repeat_diagnostics.json', repeated_diagnostics)
    assert_bitwise_equal(arrays, load_npz(folder / 'policy_repeat.npz'))
    assert diagnostics == read_json(folder / 'policy_repeat_diagnostics.json')
    if not (folder / 'replay.npz').exists():
        replay, costs, metrics = replay_and_measure(arrays, initial, seed, config)
        save_npz(folder / 'replay.npz', replay)
        write_json(folder / 'replay_costs.json', costs)
        write_json(folder / 'metrics.json', metrics)
    assert_bitwise_equal(arrays, load_npz(folder / 'replay.npz'))
    return arrays, diagnostics, dict(read_json(folder / 'metrics.json'), kp=kp, kd=kd,
                                    policy_repeat_bitwise=True, diagnostic_repeat_exact=True)


def process_case(root, output, benchmark, sid, config):
    assert sid in PILOT_IDS
    folder = output / f'state_{sid:04d}'
    folder.mkdir(exist_ok=True)
    identity = case_identity(benchmark, sid)
    immutable_json(folder / 'identity.json', identity)
    initial = np.array(identity['initial'], dtype=np.float64)
    assert initial.tobytes().hex() == identity['initial_raw_bytes_hex']
    seed = config['seeds']['environment'] + sid
    gains, baseline, baseline_diagnostics = [], None, None
    for kp, kd in GAINS:
        arrays, diagnostics, metrics = cached_gain(folder / 'gains' / gain_name(kp, kd),
                                                  initial, seed, kp, kd, config)
        gains.append(metrics)
        if (kp, kd) == (8., 2.):
            baseline, baseline_diagnostics = arrays, diagnostics
    comparison = compare_archive(identity, baseline, baseline_diagnostics)
    immutable_json(folder / 'archive_comparison.json', comparison)
    trials = []
    for action in config['interval_actions_in_order']:
        direction = folder / 'extrema' / ('u_minus2' if action == -2 else 'u_plus2')
        attempts = []
        for precision in config['precision_bits_in_order']:
            certificate, record = _attempt(direction / f'precision_{precision}', sid,
                                            identity['initial_hex'], action, precision)
            attempts.append(record)
            if certificate['status'] != 'boundary_unresolved':
                break
        arrays, metrics = cached_evaluation(direction / 'simulator', initial, seed,
                                            np.full(200, action), config)
        trials.append({'action': action, 'attempts': attempts, 'final': record,
                       'simulator': compare_simulator(certificate, arrays, metrics)})
    certs = [t for t in trials if t['final']['checker']['certified']]
    alternatives = [m for m in gains if [m['kp'], m['kd']] != [8., 2.] and m['witness']]
    result = {'state_id': sid, 'identity': identity, 'comparison': comparison,
              'gains': gains, 'trials': trials,
              'mathematical_status': 'certified_nonviable' if certs else
                  'exact_equilibrium_viable' if sid == 3280 else 'unresolved_model',
              'exact_equilibrium_witness': 'u=0 leaves theta=omega=0 exactly' if sid == 3280 else None,
              'empirically_safe_alternative_gains': [[m['kp'], m['kd']] for m in alternatives],
              'verified_PD_failure_rescue': bool(comparison['remeasured_raw_unsafe'] and alternatives),
              'model_simulator_conflict': bool(certs and any(m['witness'] for m in gains))}
    immutable_json(folder / 'summary.json', result)
    print(f'Grid {sid}: archived unsafe={identity["archived_unsafe"]}, raw unsafe={comparison["remeasured_raw_unsafe"]}, '
          f'{result["mathematical_status"]}; safe gain controllers={sum(m["witness"] for m in gains)}/9', flush=True)
    return result, baseline


def simulator_costs(folder):
    costs = [c for p in folder.rglob('*_costs.json')
             if 'step_attempts' in (c := read_json(p))]
    return {'steps': sum(int(c['step_attempts']) for c in costs),
            'resets': sum(int(c['reset_attempts']) for c in costs),
            'wall_seconds': sum(float(c.get('wall_seconds', 0)) for c in costs)}


def report_text(summary, verification, costs):
    lines = ['# Experiment 09 — Five-case technical pilot', '',
             'Only approved grid IDs 3280, 0, 6560, 767, and 44 were executed. The 3,230-state screening was not run. No training data were changed and no controller was trained.', '',
             'The archived Experiment 01 arrays and classifications remain immutable. E01-native observation scalar dtypes feed the original PD controller; all 201 physical raw states are recorded separately. Original archived trajectories/actions do not exist, so historical bitwise trajectory reproduction is not claimed. Current policy repeats and recorded-action replays are checked bitwise.', '',
             '## PD baseline and mathematical status', '',
             '| Grid ID | Archived unsafe | New legacy unsafe | Raw unsafe | Return | Raw minimum margin | First raw crossing (s) | Real-model status | Safe gain controllers |',
             '|---:|---|---|---|---:|---:|---:|---|---:|']
    for case in summary['cases']:
        p = case['comparison']; b = next(m for m in case['gains'] if [m['kp'], m['kd']] == [8., 2.])
        crossing = '—' if b['first_crossing_seconds'] is None else f'{b["first_crossing_seconds"]:.2f}'
        lines.append(f'| {case["state_id"]} | {p["archived_unsafe"]} | {p["remeasured_legacy_unsafe"]} | '
                     f'{p["remeasured_raw_unsafe"]} | {b["returns"]:.9f} | {b["minimum_margins"]:.12g} | '
                     f'{crossing} | {case["mathematical_status"]} | {sum(m["witness"] for m in case["gains"])}/9 |')
    lines += ['', 'Archived-label, crossing-time, and legacy-margin discrepancies are retained in archive_comparisons.csv. Differences between raw and wrapped minima are different safety conventions, not overwritten benchmark values.', '',
              '## Certificates', '',
              'Unchanged Experiment 08 producer and exact-rational independent checker; real-valued rational-parameter dynamics, exact binary initial states, |u|<=2, |theta|<=1, |omega|<=3, and 200 transitions. Every preceding interval box must be safe; -2 certifies only strict upper exits and +2 only strict lower exits. Opposite exits are noncertifying. Speed clipping is inactive on steps beginning in the safety set. All attempts, exact endpoints, repetitions and checks are retained.', '',
              '| Grid ID | Torque | Status | Component / crossing step | Certified separation (display) | Max theta width | Max omega width | Bits |',
              '|---:|---:|---|---|---:|---:|---:|---:|']
    for case in summary['cases']:
        for trial in case['trials']:
            r = trial['final']; d = r['diagnostics']; w = d['maximum_width_approx']
            sep = d['crossing_separation_lower_approx']
            lines.append(f'| {case["state_id"]} | {trial["action"]} | {r["status"]} | '
                         f'{r["crossing_component"]} / {r["decisive_step"]} | {sep} | {w["theta"]:.6e} | {w["omega"]:.6e} | {r["precision_bits"]} |')
    lines += ['', '## Frozen controller bank', '',
              'Kp in {6,8,12}, Kd in {1,2,4}; baseline (8,2). Every pair was evaluated on every pilot case with 200 steps, a full independent policy repeat and recorded-action replay. Full per-controller returns, discounted returns, raw margins, crossings, saturation and witness flags are in gain_metrics.csv. Per-state safe-pair lists are in summary.json. Any union of successful pairs is an oracle feasibility diagnostic, not a deployed controller. A safe alternative counts as a rescue only when the raw-state PD baseline violates safety.', '',
              '| Kp | Kd | Safe pilot cases | PD-unsafe pilot rescues |', '|---:|---:|---:|---:|']
    for kp, kd in GAINS:
        selected = [(c, next(m for m in c['gains'] if [m['kp'], m['kd']] == [kp, kd])) for c in summary['cases']]
        lines.append(f'| {kp:g} | {kd:g} | {sum(m["witness"] for c,m in selected)}/5 | '
                     f'{sum(c["comparison"]["remeasured_raw_unsafe"] and m["witness"] for c,m in selected)} |')
    lines += ['', '## Reference ID 44 optimization', '',
              'Unchanged Experiment 07 objective, transcription, constraints, solver settings and feedback tracker. Three starts (PD, zero, seeded 20-step random blocks), each solved twice with identical guesses. Open-loop and tracking executions are validated separately. Optimizer status does not establish safety; all finite candidates receive independent Gymnasium execution and recorded-action replay.', '',
              '| Start | Mode | Solver status | Dynamics residual | Predicted margin | Executed margin | Return | Safe witness |',
              '|---|---|---|---:|---:|---:|---:|---|']
    for r in summary['optimization']['records']:
        if not r['candidate_available']:
            lines.append(f'| {r["start"]} | missing | {r["solver_status"]} | — | — | — | — | False |')
        else:
            lines.append(f'| {r["start"]} | {r["mode"]} | {r["solver_status"]} | {r["maximum_dynamics_residual"]:.6e} | '
                         f'{r["predicted_minimum_raw_margin"]:.12g} | {r["minimum_margins"]:.12g} | {r["returns"]:.9f} | {r["witness"]} |')
    lines += ['', '## Verification, costs, and limitations', '',
              f'Technical verification passed: {verification["passed"]}. Technical tests: {verification["technical_tests"]["tests"]}, '
              f'failures: {verification["technical_tests"]["failures"]}, errors: {verification["technical_tests"]["errors"]}. '
              f'Independent saved-array audit: {verification["independent_audit"]["passed"]}. '
              f'Protected prior files unchanged: {verification["protected_files"]}. '
              f'Failed checks: {verification["failed_checks"]}.', '',
              f'Scientific simulator interactions: {costs["simulator"]["steps"]} steps and {costs["simulator"]["resets"]} resets, '
              f'{costs["simulator"]["wall_seconds"]:.6f} simulator wall seconds. Solver attempts: {costs["optimization"]["solver_attempts"]}, '
              f'{costs["optimization"]["solver_seconds"]:.6f} solver seconds and '
              f'{costs["optimization"]["process_wall_seconds"]:.6f} worker-process seconds. '
              f'Primary interval updates / exact repeats / rational checks: {costs["interval"]["updates"]} / '
              f'{costs["interval"]["updates"]} / {costs["interval"]["checks"]}. '
              'Unit-test interactions and final independent rational rechecks are recorded separately in technical_tests.json and independent_audit.json; they are excluded from scientific interaction totals.', '',
              'Certificates apply to the declared real-valued discrete model. Gymnasium observations are separate numerical evidence; no implementation-wide floating-point error bound is established. No continuous-time, uncertain-state, model-error, real-world, infinite-horizon, or deployable safety-filter guarantee follows. The five cases are technical controls, not a representative recovery-rate sample. Reference-state optimization outcomes are not rescues of unsafe states. Failure to find a witness remains inconclusive where a mathematical certificate is absent. No performance-based tuning, additional grid screening, commit, or push was performed.', '',
              'Reproduction: install requirements_experiment_09.txt and use experiments/09_recoverable_pd.py --pilot with the exact archived Experiment 01 NPZ. Existing outputs require explicit --resume; --verify checks saved hashes without new scientific computation. All generated evidence remains under results/recoverable_pd/pilot/.', '']
    return '\n'.join(lines)


def run_pilot(root, resume=False, snapshot=None):
    root = Path(root).resolve()
    config, output = settings(root)
    if output.exists() and not resume:
        raise FileExistsError('Pilot exists; inspect before using --resume or use --verify.')
    output.mkdir(parents=True, exist_ok=True)
    with exclusive(output):
        if (output / 'completion.json').exists():
            verify_completed(root)
            return
        try:
            _run(root, output, config, snapshot)
        except BaseException as exc:
            folder = output / 'failed_checks'
            folder.mkdir(exist_ok=True)
            write_json(folder / f'failure_{len(list(folder.glob("*.json")))+1:03d}.json',
                       {'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
            raise


def _run(root, output, config, snapshot):
    benchmark = load_benchmark(root / config['benchmark'])
    assert benchmark['metadata']['sha256'] == config['benchmark_sha256']
    immutable_json(output / 'protocol.json', config)
    immutable_json(output / 'benchmark_metadata.json', benchmark['metadata'])
    immutable_json(output / 'case_manifest.json', [case_identity(benchmark, sid) for sid in PILOT_IDS])
    if not (output / 'preservation_before.json').exists():
        write_json(output / 'preservation_before.json', read_json(snapshot) if snapshot else protected_manifest(root, output))
    before = read_json(output / 'preservation_before.json')
    _assert_hashes(root, before['sha256'])
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    sources = sorted(set(ADDITIONS + [n for n in tracked if n.startswith(('src/', 'tests/', 'configs/', 'requirements'))
                                    or n == 'experiments/01_expert_map.py']))
    hashes = {name: sha256(root / name) for name in sources}
    immutable_json(output / 'source_hashes.json', hashes)
    for name in sources:
        target = output / 'source_snapshot' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists(): target.write_bytes((root / name).read_bytes())
        assert sha256(target) == hashes[name]
    immutable_json(output / 'metadata.json', {'python': sys.version, 'executable': sys.executable,
                   'platform': platform.platform(), 'numpy': np.__version__, 'gymnasium': gym.__version__,
                   'casadi': ca.__version__, 'gmpy2': g.version(), 'MPFR': g.mpfr_version(),
                   'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                   'full_study_executed': False, 'pilot_ids': PILOT_IDS})
    if not (output / 'technical_tests.json').exists():
        stream = io.StringIO(); suite = unittest.TestSuite()
        for pattern in ('test_recoverable_pd*.py', 'test_viability*.py', 'test_recoverability*.py'):
            suite.addTests(unittest.defaultTestLoader.discover(str(root / 'tests'), pattern=pattern))
        started = time.perf_counter()
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
        (output / 'technical_tests.txt').write_text(stream.getvalue())
        write_json(output / 'technical_tests.json', {'passed': result.wasSuccessful(), 'tests': result.testsRun,
                   'failures': len(result.failures), 'errors': len(result.errors),
                   'wall_seconds': time.perf_counter()-started,
                   'E09_test_simulator_steps': 2600, 'E09_test_simulator_resets': 13,
                   'synthetic_test_steps': 200})
    tests = read_json(output / 'technical_tests.json')
    assert tests['passed'], 'Technical tests failed; see retained technical_tests.txt'
    cases, pd44 = [], None
    for sid in PILOT_IDS:
        record, baseline = process_case(root, output, benchmark, sid, config)
        cases.append(record)
        if sid == 44: pd44 = baseline
    optimization = run_reference_optimization(root, output, np.array(case_identity(benchmark, 44)['initial']),
                                               config['seeds']['environment']+44, pd44)
    summary = {'pilot_ids': PILOT_IDS, 'cases': cases, 'optimization': optimization,
               'full_study_executed': False, 'training_data_use': False}
    write_json(output / 'summary.json', summary)
    write_csv(output / 'archive_comparisons.csv', [c['comparison'] for c in cases])
    write_csv(output / 'gain_metrics.csv', [dict(state_id=c['state_id'], **m) for c in cases for m in c['gains']])
    write_csv(output / 'optimization_metrics.csv', optimization['records'])
    from src.recoverable_pd_audit import audit_saved_pilot
    audit = audit_saved_pilot(root, output)
    write_json(output / 'independent_audit.json', audit)
    attempts = [a for c in cases for t in c['trials'] for a in t['attempts']]
    costs = {'simulator': simulator_costs(output), 'optimization': optimization['costs'],
             'interval': {'updates': sum(a['diagnostics']['interval_steps'] for a in attempts),
                          'checks': sum(a['checker']['check_count'] for a in attempts),
                          'generation_seconds': sum(a['generation_seconds'] for a in attempts),
                          'repeat_seconds': sum(a['generation_repeat_seconds'] for a in attempts),
                          'checker_seconds': sum(a['checker_seconds'] for a in attempts)}}
    write_json(output / 'computational_costs.json', costs)
    _assert_hashes(root, before['sha256']); _assert_hashes(root, hashes)
    failed = []
    if not optimization['passed']: failed.append('optimization_reproducibility_or_worker_failure')
    if not audit['passed']: failed.append('independent_saved_array_audit')
    if any(c['model_simulator_conflict'] for c in cases): failed.append('model_simulator_conflict')
    verification = {'passed': not failed, 'failed_checks': failed, 'technical_tests': tests,
                    'independent_audit': audit, 'protected_files': len(before['sha256']),
                    'case_ids': PILOT_IDS, 'full_study_executed': False,
                    'optimization_only_grid_id': 44, 'unchanged_E07_configuration': optimization['configuration_unchanged']}
    write_json(output / 'verification.json', verification)
    report = report_text(summary, verification, costs)
    (output / 'scientific_report.md').write_text(report)
    doc = root / 'docs/experiment_09_pilot_report.md'
    if doc.exists(): assert doc.read_text() == report, 'Reviewed report exists and differs; retained generated report only.'
    else: doc.write_text(report)
    assert verification['passed'], 'Verification failed; all findings and failed checks retained.'
    artifacts = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                 if p.is_file() and p.name != '.lock' and p != output / 'completion.json'}
    write_json(output / 'completion.json', {'passed': True, 'case_ids': PILOT_IDS,
               'artifacts_sha256': artifacts, 'report_sha256': sha256(doc)})
    print('Five-case E09 pilot completed and verified; no full screening, commit or push.', flush=True)
