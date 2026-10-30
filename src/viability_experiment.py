"""Five-case Experiment 08 pilot; all prior artifacts remain read-only."""

from fractions import Fraction
import io
from pathlib import Path
import platform
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
from src.viability import MODEL, diagnostics, extremal_trajectory, ratio, rational
from src.viability_checker import check_extremal_certificate

CONFIG = 'configs/experiment_08_pilot.json'
CASE_IDS = [96, 2, 33, 37, 94]
SOURCES = [CONFIG, 'requirements_experiment_08.txt', 'src/viability.py',
           'src/viability_checker.py', 'src/viability_experiment.py',
           'experiments/08_viability_pilot.py', 'tests/test_viability.py',
           'tests/test_viability_checker.py']


def settings(root):
    config = read_json(root / CONFIG)
    assert config['authorized_phase'] == 'five_case_technical_pilot_only'
    assert config['case_ids_in_order'] == CASE_IDS
    assert config['actions_in_order'] == [-2, 2]
    assert config['precision_bits_in_order'] == [128, 256, 512]
    assert config['horizon'] == 200 and config['dt_exact'] == '1/20'
    for key in ('dt', 'gravity', 'mass', 'length', 'theta_limit', 'omega_limit', 'torque_limit'):
        assert config[key + '_exact'] == MODEL[key]
    assert not config['remaining_states_authorized'] and not config['viability_grid_authorized']
    assert not config['training_data_use'] and config['real_model_only']
    assert not config['implementation_error_bound_established']
    assert g.version() == '2.3.2'
    return config, root / config['source_experiment'], root / config['output']


def verify_completed(root):
    config, source, output = settings(root)
    completion = read_json(output / 'completion.json')
    assert completion['passed'] and completion['case_ids'] == CASE_IDS
    _assert_hashes(output, completion['artifacts_sha256'])
    _assert_hashes(root, read_json(output / 'source_hashes.json'))
    _assert_hashes(root, read_json(output / 'preservation_before.json')['sha256'])
    _assert_hashes(source, read_json(source / 'completion.json')['artifacts_sha256'])
    assert sha256(root / 'docs/experiment_08_pilot_report.md') == completion['report_sha256']
    print('Completed five-case pilot verified; no interval calculations or simulations rerun.', flush=True)


def _attempt(folder, sid, initial_hex, action, precision):
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / 'completion.json').exists():
        _assert_hashes(folder, read_json(folder / 'completion.json')['artifacts_sha256'])
        return read_json(folder / 'certificate.json'), read_json(folder / 'record.json')
    started = time.perf_counter()
    certificate = extremal_trajectory(sid, initial_hex, action, precision)
    generation_seconds = time.perf_counter() - started
    immutable_json(folder / 'certificate.json', certificate)
    started = time.perf_counter()
    repeated = extremal_trajectory(sid, initial_hex, action, precision)
    repeat_seconds = time.perf_counter() - started
    assert certificate == repeated
    started = time.perf_counter()
    checked = check_extremal_certificate(certificate, sid, initial_hex)
    checker_seconds = time.perf_counter() - started
    record = dict(state_id=sid, action=action, precision_bits=precision,
                  status=certificate['status'], decisive_step=certificate['decisive_step'],
                  crossing_component=certificate['crossing_component'],
                  diagnostics=diagnostics(certificate), checker=checked,
                  generation_repeat_exact=True, generation_seconds=generation_seconds,
                  generation_repeat_seconds=repeat_seconds, checker_seconds=checker_seconds)
    write_json(folder / 'record.json', record)
    write_json(folder / 'completion.json', {'passed': True, 'artifacts_sha256':
               {name: sha256(folder / name) for name in ('certificate.json', 'record.json')}})
    return certificate, record


def compare_simulator(certificate, arrays, metrics):
    """Observed discrepancies only: never infer a universal machine-error bound."""
    bounds = {name: Fraction(0) for name in ('theta', 'omega')}
    for row in certificate['trace']:
        for j, name in enumerate(('theta', 'omega')):
            actual = Fraction.from_float(float(arrays['raw_states'][0, row['step'], j]))
            lo, hi = rational(row[name]['lo']), rational(row[name]['hi'])
            bounds[name] = max(bounds[name], abs(actual - lo), abs(actual - hi))
    return {
        'first_crossing_step': metrics['first_crossing_step'],
        'minimum_raw_margin_over_201_states': metrics['minimum_margins'],
        'episode_return': metrics['returns'],
        'replay_bitwise': metrics['replay_bitwise'],
        'model_and_simulator_first_crossing_step_match': metrics['first_crossing_step'] == certificate['decisive_step'],
        'observed_prefix_error_upper_exact': {k: ratio(v) for k, v in bounds.items()},
        'observed_prefix_error_upper_approx': {k: float(v) for k, v in bounds.items()},
        'implementation_error_bound_established': False,
        'scope': 'Discrepancy for this saved execution prefix only; not a bound over all states or control sequences.',
    }


def process_case(root, output, source, case, config, simulation_config):
    sid = case['state_id']
    assert sid in CASE_IDS
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
    for action in config['actions_in_order']:
        direction = folder / ('u_minus2' if action == -2 else 'u_plus2')
        attempts = []
        for precision in config['precision_bits_in_order']:
            certificate, record = _attempt(direction / f'precision_{precision}', sid, initial_hex, action, precision)
            attempts.append(record)
            if certificate['status'] != 'boundary_unresolved':
                break
        arrays, metrics = cached_evaluation(direction / 'simulator', initial, case['reset_seed'],
                                            np.full(200, action, dtype=np.float64), simulation_config)
        diagnostic = compare_simulator(certificate, arrays, metrics)
        if record['checker']['certified']:
            # The existing unsafe PD prefix is precisely this executed extremum.
            k = pd['first_crossing_step']
            assert k is not None and np.all(baseline['actions'][0, :k, 0] == action)
            assert_bitwise_equal({'raw_prefix': baseline['raw_states'][0, :k+1]},
                                 {'raw_prefix': arrays['raw_states'][0, :k+1]})
        trials.append({'action': action, 'attempts': attempts, 'final': record,
                       'simulator': diagnostic})
    certified = [r for r in trials if r['final']['checker']['certified']]
    if not case['PD_unsafe']:
        assert not certified, 'Certificate conflicts with an independently verified PD-safe reference; investigate.'
    equilibrium_exact = sid == 96 and all(Fraction.from_float(float(v)) == 0 for v in initial)
    mathematical = ('certified_nonviable' if certified else 'exact_equilibrium_viable' if equilibrium_exact
                    else 'no_nonviability_certificate')
    result = dict(state_id=sid, group=case['group'], initial_hex=initial_hex,
                  mathematical_classification=mathematical, real_model_only=True,
                  PD_simulator=pd, trials=trials,
                  equilibrium_proof='u=0 gives sin(0)=0, omega_next=theta_next=0 exactly at every step.' if equilibrium_exact else None)
    write_json(folder / 'summary.json', result)
    hashes = {str(p.relative_to(folder)): sha256(p) for p in folder.rglob('*')
              if p.is_file() and p != marker}
    write_json(marker, {'passed': True, 'artifacts_sha256': hashes})
    return result


def report_text(summary, verification, costs):
    lines = ['# Experiment 08 — Five-case finite-horizon viability pilot', '',
             'This pilot contains only equilibrium and state IDs 2, 33, 37, and 94. No remaining cohort states or viability grids were evaluated. No optimization, RL training, or expert-training data collection was performed.', '',
             '## Model and mathematical justification', '',
             'The declared real-valued discrete map is v = omega + (15 sin(theta) + 3u)/20, theta_next = theta + v/20. Physical parameters, including h=1/20, are exact rationals; each archived binary64 initial value is treated as an exact dyadic rational. Bounds are |theta|<=1, |omega|<=3, |u|<=2, with 200 transitions and 201 sampled states. Initial hex strings, integer ratios, and raw bytes are saved.', '',
             'For |theta|<=1, cos(theta)>=1/2, so all state/action derivatives of this map are nonnegative. Any transition beginning in the safety set has |omega_next|<=81/20<8; simulator speed clipping is irrelevant to this comparison. A constant -2 path with a wholly safe prefix and a strict upper crossing bounds any hypothetical safe admissible path from below. Constant +2 gives the reflected lower-crossing argument. Thus such a crossing by k<=200 rules out every safe admissible control sequence for the declared model. Opposite-direction exits do not certify nonviability, and no trajectory is propagated after its first exit or ambiguous boundary.', '',
             '## Pilot results', '',
             '| State | Group | Real-model result | Certifying torque | Step / time | Certified crossing separation (approx. rad) | Precision |',
             '|---|---|---|---:|---:|---:|---:|']
    for case in summary['cases']:
        cert = next((r['final'] for r in case['trials'] if r['final']['checker']['certified']), None)
        values = ['—']*4 if cert is None else [str(cert['action']), f"{cert['decisive_step']} / {cert['decisive_step']/20:.2f} s",
                  f"{cert['diagnostics']['crossing_separation_lower_approx']:.12g}", str(cert['precision_bits'])]
        lines.append('| ' + ' | '.join([str(case['state_id']), case['group'], case['mathematical_classification'], *values]) + ' |')
    lines += ['', 'The reported separations are decimal displays of exact saved rational lower bounds. The certified counts apply only to the two PD-unsafe pilot states, not to the full set of 40. Equilibrium has a separate exact zero-torque witness. For states 2 and 33, absence of a certificate is not a mathematical viability proof; their full-horizon PD safety is verified separately in Gymnasium.', '',
              '| State | Torque | Extremal test status | First exit step | Maximum theta interval width | Maximum omega interval width | Safe-prefix minimum normalized margin lower bound |',
              '|---|---:|---|---:|---:|---:|---:|']
    for case in summary['cases']:
        for trial in case['trials']:
            r = trial['final']; d = r['diagnostics']; w = d['maximum_width_approx']
            lines.append(f"| {case['state_id']} | {trial['action']} | {r['status']} | {r['decisive_step']} | {w['theta']:.3e} | {w['omega']:.3e} | {d['safe_prefix_minimum_margin_lower_approx']:.12g} |")
    lines += ['', '## Independent checking and simulator comparison', '',
              f"All {verification['technical_tests']['tests']} technical tests pass. All {summary['precision_attempts']} interval records pass an independent checker that imports neither MPFR nor the producer. It evaluates exact rational alternating Taylor bounds for sine, checks enclosure inclusion at every transition, verifies exact initial IDs/values, every safe-prefix box, and the strict crossing direction. It also rejects forged records, wrong directions, ambiguous boundary crossings, unsafe prefixes, and altered model parameters. All generation repeats reproduce the exact interval records.", '',
              'MPFR directed rounding uses integer division by 20 rather than a rounded 0.05 literal. Endpoints are serialized as exact numerator/denominator strings. Precision escalation is fixed at 128, 256, then 512 bits only for boundary ambiguity; numerical tolerances never relax safety.', '',
              '| State | Archived PD return | Archived PD minimum raw margin | PD violation | New recorded-action replay |',
              '|---|---:|---:|---|---|']
    for case in summary['cases']:
        p = case['PD_simulator']
        lines.append(f"| {case['state_id']} | {p['returns']:.9f} | {p['minimum_margins']:.9f} | {p['violated']} | bitwise equal |")
    lines += ['', f"All ten extremal Gymnasium trajectories were evaluated over the full 200 steps using executed float32 torques and all 201 raw states, and independently replayed bitwise. Model and simulator first-exit steps match for {summary['matching_simulator_exit_steps']}/10 directions. Maximum observed prefix discrepancies bounded using the real-model enclosures are {summary['maximum_observed_prefix_error']['theta']:.6g} rad and {summary['maximum_observed_prefix_error']['omega']:.6g} rad/s. These are observations at the tested prefixes, not global implementation-error bounds.", '',
              'Certificates concern the declared real map, not NumPy/libm floating-point semantics. At u=+/-2 the torque product is exact, but binary 0.05, operation rounding, and sine implementation remain different. No justified implementation-wide error envelope is claimed. Bitwise simulator agreement does not upgrade a model certificate to a machine-level or physical-system proof.', '',
              '## Cost, preservation, and reproduction', '',
              f"Scientific interval propagation: {costs['primary_interval_updates']} steps; deterministic generation repeat: {costs['repeat_interval_updates']} steps; independent rational checker: {costs['checker_transition_checks']} checks. Generation took {costs['generation_seconds']:.6f} s, repetition {costs['repeat_seconds']:.6f} s, and rational checking {costs['checker_seconds']:.6f} s. Unit-test costs are separate from these five-case measurements.",
              f"New Gymnasium interactions: {costs['simulator_steps']} steps and {costs['simulator_resets']} resets, comprising five archived PD-action replays, ten extremal executions, and ten extremal replays. No new sampled initial states or training transitions were added. Numerical library versions: {summary['arithmetic_versions']}.", '',
              f"SHA-256 checks preserve {verification['protected_files']} pre-existing files, including Experiments 01–07, D0, the complete Experiment 07 pilot/full-study artifacts, and unfinished IQL diagnostics. The scientific configuration and source snapshots are saved with all certificates, noncertifying attempts, independent checks, and replay diagnostics under results/viability/pilot/.", '',
              'Install requirements_experiment_08.txt in a separate environment. Run `python -m unittest discover -s tests -p "test_viability*.py" -v`, then `python experiments/08_viability_pilot.py --pilot`. Existing outputs require explicit `--resume`; `--verify` only verifies completed hashes and never launches computations. Source Experiment 07 artifacts are required. Generated reports remain in results; the reviewed documentation is not overwritten by reproduction.', '',
              'These are sampled-time statements over the declared 200-step horizon, not continuous-time, infinite-horizon, or real-world safety claims. No conclusion is assigned to the other 38 PD-unsafe states or the other reference states before approval to proceed.', '']
    return '\n'.join(lines)


def run_pilot(root, resume=False, snapshot=None):
    root = Path(root).resolve()
    config, source, output = settings(root)
    if output.exists() and not resume:
        raise FileExistsError('Pilot exists; use --verify or explicitly --pilot --resume.')
    output.mkdir(parents=True, exist_ok=True)
    with exclusive(output):
        if (output / 'completion.json').exists():
            verify_completed(root)
            return
        _assert_hashes(source, read_json(source / 'completion.json')['artifacts_sha256'])
        _assert_hashes(root, read_json(source / 'source_hashes.json'))
        frozen = {name: sha256(root / name) for name in SOURCES}
        immutable_json(output / 'source_hashes.json', frozen)
        immutable_json(output / 'protocol.json', config)
        if not (output / 'preservation_before.json').exists():
            write_json(output / 'preservation_before.json', read_json(snapshot) if snapshot else protected_manifest(root, output))
        before = read_json(output / 'preservation_before.json')
        _assert_hashes(root, before['sha256'])
        for name in SOURCES:
            target = output / 'source_snapshot' / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists(): target.write_bytes((root / name).read_bytes())
            assert sha256(target) == frozen[name]
        versions = dict(gmpy2=g.version(), GMP=g.mp_version(), MPFR=g.mpfr_version(), MPC=g.mpc_version())
        immutable_json(output / 'metadata.json', dict(python=sys.version, executable=sys.executable,
                       platform=platform.platform(), numpy=np.__version__, gymnasium=gym.__version__,
                       arithmetic_versions=versions, model=MODEL,
                       git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                       simulation_policy='Separate model certificates and float32 Gymnasium diagnostics.'))
        if not (output / 'technical_tests.json').exists():
            stream = io.StringIO(); started = time.perf_counter()
            suite = unittest.defaultTestLoader.discover(str(root / 'tests'), pattern='test_viability*.py')
            result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
            (output / 'technical_tests.txt').write_text(stream.getvalue())
            write_json(output / 'technical_tests.json', dict(passed=result.wasSuccessful(), tests=result.testsRun,
                       failures=len(result.failures), errors=len(result.errors), wall_seconds=time.perf_counter()-started))
        tests = read_json(output / 'technical_tests.json')
        assert tests['passed']
        manifest = read_json(source / 'state_manifest.json')['all_57_cases']
        cases = [next(r for r in manifest if r['state_id'] == sid) for sid in CASE_IDS]
        immutable_json(output / 'case_manifest.json', cases)
        simulation_config = read_json(source / 'protocol.json')
        records = []
        for case in cases:
            r = process_case(root, output, source, case, config, simulation_config)
            records.append(r)
            print(f"Verified pilot state {case['state_id']}: {r['mathematical_classification']}", flush=True)
        attempts = [a for r in records for t in r['trials'] for a in t['attempts']]
        assert all(a['checker']['passed'] and a['generation_repeat_exact'] for a in attempts)
        new_costs = [read_json(p) for p in output.glob('state_*/**/*costs.json')]
        costs = dict(primary_interval_updates=sum(a['diagnostics']['interval_steps'] for a in attempts),
                     repeat_interval_updates=sum(a['diagnostics']['interval_steps'] for a in attempts),
                     checker_transition_checks=sum(a['checker']['check_count'] for a in attempts),
                     generation_seconds=sum(a['generation_seconds'] for a in attempts),
                     repeat_seconds=sum(a['generation_repeat_seconds'] for a in attempts),
                     checker_seconds=sum(a['checker_seconds'] for a in attempts),
                     simulator_steps=sum(int(c['step_attempts']) for c in new_costs),
                     simulator_resets=sum(int(c['reset_attempts']) for c in new_costs))
        assert costs['simulator_steps'] == 5000 and costs['simulator_resets'] == 25
        summary = dict(cases=records, precision_attempts=len(attempts), arithmetic_versions=versions,
                       certified_nonviable_ids=[r['state_id'] for r in records if r['mathematical_classification']=='certified_nonviable'],
                       certified_scope='Real-valued discrete map only',
                       matching_simulator_exit_steps=sum(t['simulator']['model_and_simulator_first_crossing_step_match'] for r in records for t in r['trials']),
                       maximum_observed_prefix_error={name:max(t['simulator']['observed_prefix_error_upper_approx'][name] for r in records for t in r['trials']) for name in ('theta','omega')})
        _assert_hashes(root, before['sha256']); _assert_hashes(root, frozen)
        _assert_hashes(source, read_json(source / 'completion.json')['artifacts_sha256'])
        verification = dict(passed=True, technical_tests=tests, protected_files=len(before['sha256']),
                            case_ids=CASE_IDS, exact_binary_initial_states=True, independent_rational_checker=True,
                            checked_records=len(attempts), recorded_action_replay_pairs=15,
                            reference_false_nonviability_count=0, source_07_artifacts_unchanged=True,
                            full_cohort_run=False, viability_grid_run=False, model_only=True)
        write_json(output / 'summary.json', summary)
        write_json(output / 'numerical_costs.json', costs)
        write_json(output / 'verification.json', verification)
        report = report_text(summary, verification, costs)
        (output / 'scientific_report.md').write_text(report)
        reviewed = root / 'docs/experiment_08_pilot_report.md'
        if not reviewed.exists(): reviewed.write_text(report)
        hashes = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                  if p.is_file() and p.name != '.lock' and p != output / 'completion.json'}
        write_json(output / 'completion.json', dict(passed=True, case_ids=CASE_IDS, artifacts_sha256=hashes,
                   report_sha256=sha256(reviewed)))
        print('Five-case pilot complete. Remaining states and viability grids were not run.', flush=True)
