"""Read-only aggregation and reporting for the approved Experiment 08 cohort.

No interval propagation, optimization, simulation, or training occurs here.
Exact evidence stays in the certificate artifacts; decimal tables are displays.
"""

from collections import Counter
import csv
import json
from pathlib import Path
import statistics


def _stats(values):
    values = list(values)
    if not values:
        return {'count': 0, 'min': None, 'max': None, 'mean': None}
    return {'count': len(values), 'min': min(values), 'max': max(values),
            'mean': statistics.fmean(values)}


def _direction_row(case, trial, reused):
    final = trial['final']
    checker = final['checker']
    if not checker['passed'] or not checker['model_only']:
        raise ValueError('Every reported direction requires a passed model-only checker.')
    certified = checker['certified']
    if certified != (final['status'] == 'certified_nonviable'):
        raise ValueError('Checker certification and declared status disagree.')
    if final['action'] != trial['action'] or final['state_id'] != case['state_id']:
        raise ValueError('Direction result identity mismatch.')
    for attempt in trial['attempts']:
        if not attempt['checker']['passed'] or not attempt['generation_repeat_exact']:
            raise ValueError('An attempted precision failed checking or deterministic replay.')
    if trial['attempts'][-1] != final:
        raise ValueError('Final direction result must be the last retained precision attempt.')
    desired = 'upper' if trial['action'] == -2 else 'lower'
    direction = desired if certified else ('lower' if desired == 'upper' else 'upper') if final['status'] == 'opposite_exit' else None
    diagnostic = final['diagnostics']
    step = final['decisive_step']
    artifact = case.get('artifact_directory', f"state_{case['state_id']:03d}")
    action_folder = 'u_minus2' if trial['action'] == -2 else 'u_plus2'
    return {
        'state_id': case['state_id'], 'group': case['group'], 'reused_pilot': reused,
        'action': trial['action'], 'status': final['status'],
        'independently_checked': True, 'certified_nonviable': certified,
        'crossing_direction': direction, 'crossing_component': final['crossing_component'],
        'crossing_step': step, 'crossing_time_seconds': None if step is None else step / 20,
        'crossing_separation_lower_exact': diagnostic['crossing_separation_lower_exact'],
        'crossing_separation_lower_approx': diagnostic['crossing_separation_lower_approx'],
        'maximum_theta_interval_width_exact': diagnostic['maximum_width_exact']['theta'],
        'maximum_omega_interval_width_exact': diagnostic['maximum_width_exact']['omega'],
        'maximum_theta_interval_width_approx': diagnostic['maximum_width_approx']['theta'],
        'maximum_omega_interval_width_approx': diagnostic['maximum_width_approx']['omega'],
        'safe_prefix_margin_lower_exact': diagnostic['safe_prefix_minimum_margin_lower_exact'],
        'safe_prefix_margin_lower_approx': diagnostic['safe_prefix_minimum_margin_lower_approx'],
        'precision_bits': final['precision_bits'],
        'attempted_precisions': [a['precision_bits'] for a in trial['attempts']],
        'interval_updates_all_attempts': sum(a['diagnostics']['interval_steps'] for a in trial['attempts']),
        'checker_transition_checks_all_attempts': sum(a['checker']['check_count'] for a in trial['attempts']),
        'simulator_first_crossing_step': trial['simulator']['first_crossing_step'],
        'simulator_model_exit_step_match': trial['simulator']['model_and_simulator_first_crossing_step_match'],
        'simulator_replay_bitwise': trial['simulator']['replay_bitwise'],
        'simulator_minimum_raw_margin': trial['simulator']['minimum_raw_margin_over_201_states'],
        'observed_theta_prefix_error_upper': trial['simulator']['observed_prefix_error_upper_approx']['theta'],
        'observed_omega_prefix_error_upper': trial['simulator']['observed_prefix_error_upper_approx']['omega'],
        'certificate_artifact': f'{artifact}/{action_folder}/precision_{final["precision_bits"]}/certificate.json',
        'real_model_only': True, 'implementation_error_bound_established': False,
    }


def summarize(records, reused_ids):
    """Summarize whatever was proved; never turn a failed search into a claim."""
    records = sorted(records, key=lambda r: r['state_id'])
    ids = [r['state_id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('Duplicate state IDs would corrupt the study denominator.')
    reused_ids = set(reused_ids)
    states, directions = [], []
    for case in records:
        if case['group'] not in ('PD_unsafe', 'nominal_safe', 'challenging_safe', 'equilibrium'):
            raise ValueError('Unknown study stratum.')
        if not case['real_model_only'] or sorted(t['action'] for t in case['trials']) != [-2, 2]:
            raise ValueError('Expected both unchanged extremal tests for the real model.')
        rows = [_direction_row(case, trial, case['state_id'] in reused_ids) for trial in case['trials']]
        certificates = [row for row in rows if row['certified_nonviable']]
        equilibrium = case['group'] == 'equilibrium' and all(float.fromhex(v) == 0 for v in case['initial_hex']) and bool(case.get('equilibrium_proof'))
        viable_extremum = any(row['status'] == 'horizon_safe_extremal' for row in rows)
        if certificates and (equilibrium or viable_extremum):
            raise ValueError('Conflicting mathematical nonviability and viability evidence.')
        status = ('certified_nonviable' if certificates else 'exact_equilibrium_viable' if equilibrium
                  else 'verified_model_viable_extremal' if viable_extremum else 'unresolved_model')
        declared = case['mathematical_classification']
        if (declared == 'certified_nonviable') != bool(certificates):
            raise ValueError('State classification disagrees with independently checked certificates.')
        # Prefer the earliest certified impossible-safe horizon when both
        # directions happen to certify. All evidence remains in directions.
        first = min(certificates, key=lambda r: (r['crossing_step'], r['action'])) if certificates else None
        pd = case['PD_simulator']
        states.append({
            'state_id': case['state_id'], 'group': case['group'],
            'initial_hex': case['initial_hex'], 'reused_pilot': case['state_id'] in reused_ids,
            'mathematical_status': status, 'certified_nonviable': bool(certificates),
            'certifying_actions': [row['action'] for row in certificates],
            'certifying_action': None if first is None else first['action'],
            'certified_boundary_direction': None if first is None else first['crossing_direction'],
            'certified_boundary_component': None if first is None else first['crossing_component'],
            'certified_crossing_step': None if first is None else first['crossing_step'],
            'certified_crossing_time_seconds': None if first is None else first['crossing_time_seconds'],
            'certified_separation_lower_exact': None if first is None else first['crossing_separation_lower_exact'],
            'certified_separation_lower_approx': None if first is None else first['crossing_separation_lower_approx'],
            'certificate_theta_width_approx': None if first is None else first['maximum_theta_interval_width_approx'],
            'certificate_omega_width_approx': None if first is None else first['maximum_omega_interval_width_approx'],
            'certificate_precision_bits': None if first is None else first['precision_bits'],
            'maximum_precision_bits_attempted': max(p for row in rows for p in row['attempted_precisions']),
            'maximum_theta_width_over_both_directions': max(row['maximum_theta_interval_width_approx'] for row in rows),
            'maximum_omega_width_over_both_directions': max(row['maximum_omega_interval_width_approx'] for row in rows),
            'PD_simulator_safe_witness': pd['witness'], 'PD_simulator_violated': pd['violated'],
            'PD_simulator_return': pd['returns'], 'PD_simulator_minimum_raw_margin': pd['minimum_margins'],
            'PD_simulator_first_crossing_step': pd['first_crossing_step'],
            'PD_recorded_action_replay_bitwise': pd['replay_bitwise'],
            'certificate_artifact': None if first is None else first['certificate_artifact'],
            'real_model_only': True,
        })
        directions.extend(rows)
    unsafe = [r for r in states if r['group'] == 'PD_unsafe']
    controls = [r for r in states if r['group'] in ('nominal_safe', 'challenging_safe')]
    certs = [r for r in directions if r['certified_nonviable']]
    unresolved = [r['state_id'] for r in unsafe if r['mathematical_status'] == 'unresolved_model']
    primary = {
        'denominator': len(unsafe),
        'certified_count': sum(r['certified_nonviable'] for r in unsafe),
        'certified_ids': [r['state_id'] for r in unsafe if r['certified_nonviable']],
        'unresolved_count': len(unresolved), 'unresolved_ids': unresolved,
        'model_viable_ids': [r['state_id'] for r in unsafe if r['mathematical_status'] == 'verified_model_viable_extremal'],
        'additional_methods_required_for_unresolved': bool(unresolved),
    }
    return {
        'primary': primary,
        'controls': {'denominator': len(controls),
                     'false_nonviability_count': sum(r['certified_nonviable'] for r in controls),
                     'false_nonviability_ids': [r['state_id'] for r in controls if r['certified_nonviable']],
                     'simulator_safe_count': sum(r['PD_simulator_safe_witness'] for r in controls),
                     'model_unresolved_ids': [r['state_id'] for r in controls if r['mathematical_status'] == 'unresolved_model']},
        'equilibrium': [r for r in states if r['group'] == 'equilibrium'],
        'state_count': len(states), 'direction_count': len(directions),
        'reused_state_ids': sorted(set(ids) & reused_ids), 'new_state_ids': sorted(set(ids) - reused_ids),
        'states': states, 'directions': directions,
        'numerical_summary': {
            'direction_status_counts': dict(Counter(r['status'] for r in directions)),
            'certificate_direction_counts': dict(Counter(r['crossing_direction'] for r in certs)),
            'certificate_component_counts': dict(Counter(r['crossing_component'] for r in certs)),
            'final_precision_counts': dict(Counter(str(r['precision_bits']) for r in directions)),
            'precision_attempts': sum(len(r['attempted_precisions']) for r in directions),
            'independently_checked_direction_count': sum(r['independently_checked'] for r in directions),
            'certificate_crossing_steps': _stats(r['crossing_step'] for r in certs),
            'certificate_crossing_times': _stats(r['crossing_time_seconds'] for r in certs),
            'certificate_separation_lower_approximations': _stats(r['crossing_separation_lower_approx'] for r in certs),
            'certificate_theta_widths': _stats(r['maximum_theta_interval_width_approx'] for r in certs),
            'certificate_omega_widths': _stats(r['maximum_omega_interval_width_approx'] for r in certs),
            'certificate_safe_prefix_margin_lower_approximations': _stats(r['safe_prefix_margin_lower_approx'] for r in certs),
            'all_direction_theta_widths': _stats(r['maximum_theta_interval_width_approx'] for r in directions),
            'all_direction_omega_widths': _stats(r['maximum_omega_interval_width_approx'] for r in directions),
            'model_simulator_exit_step_matches': sum(r['simulator_model_exit_step_match'] for r in directions),
            'extremal_simulator_replays_bitwise': sum(r['simulator_replay_bitwise'] for r in directions),
            'PD_simulator_replays_bitwise': sum(r['PD_recorded_action_replay_bitwise'] for r in states),
            'observed_theta_prefix_error_upper': _stats(r['observed_theta_prefix_error_upper'] for r in directions),
            'observed_omega_prefix_error_upper': _stats(r['observed_omega_prefix_error_upper'] for r in directions),
        },
        'scope': 'Certificates apply only to the real-valued discrete model with exact rational parameters and archived binary initial states; simulator evidence is separate.',
        'implementation_error_bound_established': False,
    }


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def _number(value, spec='.6g'):
    return '—' if value is None else format(value, spec)


def state_details(summary):
    lines = ['# Experiment 08 — State and direction results', '',
             'Exact rational endpoints and separations are authoritative. Values below are decimal displays. Upper/lower means the direction of the actual strict interval crossing; an opposite exit is explicitly noncertifying. A missing certificate is not a mathematical viability proof.', '',
             '## State classifications', '',
             '| State | Group | Mathematical status | Certifying torque | Boundary | Step / time (s) | Separation | Theta width | Omega width | Bits | Reused pilot |',
             '|---|---|---|---:|---|---|---:|---:|---:|---:|---|']
    for r in summary['states']:
        crossing = '—' if r['certified_crossing_step'] is None else f"{r['certified_crossing_step']} / {r['certified_crossing_time_seconds']:.2f}"
        boundary = '—' if r['certified_boundary_direction'] is None else r['certified_boundary_direction'] + ' ' + r['certified_boundary_component']
        lines.append('| ' + ' | '.join([str(r['state_id']), r['group'], r['mathematical_status'],
                     _number(r['certifying_action']), boundary, crossing,
                     _number(r['certified_separation_lower_approx'], '.12g'),
                     _number(r['certificate_theta_width_approx'], '.3e'),
                     _number(r['certificate_omega_width_approx'], '.3e'),
                     _number(r['certificate_precision_bits']), str(r['reused_pilot'])]) + ' |')
    lines += ['', 'The control states marked unresolved_model retain separate verified Gymnasium PD witnesses. Equilibrium has an exact zero-input mathematical witness. Opposite-direction extrema for either kind of control are intentionally not certificates.', '',
              '## Every extremal direction, including noncertifying results', '',
              '| State | Torque | Status | Crossing | Step / time (s) | Separation | Theta width | Omega width | Bits |',
              '|---|---:|---|---|---|---:|---:|---:|---:|']
    for r in summary['directions']:
        crossing = '—' if r['crossing_direction'] is None else r['crossing_direction'] + ' ' + r['crossing_component']
        when = '—' if r['crossing_step'] is None else f"{r['crossing_step']} / {r['crossing_time_seconds']:.2f}"
        lines.append('| ' + ' | '.join([str(r['state_id']), str(r['action']), r['status'], crossing, when,
                     _number(r['crossing_separation_lower_approx'], '.12g'),
                     _number(r['maximum_theta_interval_width_approx'], '.3e'),
                     _number(r['maximum_omega_interval_width_approx'], '.3e'), str(r['precision_bits'])]) + ' |')
    lines += ['', 'Angle separations/widths are in radians; angular-velocity separations/widths are in rad/s. The CSV files additionally retain exact rational values, initial hex values, artifact paths, and simulator comparisons.', '']
    return '\n'.join(lines)


def scientific_report(summary, verification, costs):
    p, c, n = summary['primary'], summary['controls'], summary['numerical_summary']
    tests = verification.get('technical_tests', {})
    grid_run = verification.get('viability_grid_run', False)
    lines = ['# Experiment 08 — Finite-Horizon Safety Viability Analysis', '',
             f"Primary outcome: **{p['certified_count']}/{p['denominator']} PD-unsafe initial states have independently verified nonviability certificates** for the declared real-valued discrete model. Unresolved PD-unsafe states: {p['unresolved_count']} ({p['unresolved_ids']}). This result is separate from Gymnasium execution and does not constitute a real-world impossibility proof.", '',
             '## Frozen protocol and scope', '',
             f"The fixed cohort contains {p['denominator']} PD-unsafe states, {c['denominator']} PD-safe references (eight nominal and eight challenging), and equilibrium. The five reviewed pilot cases were reused without recomputation; {len(summary['new_state_ids'])} remaining states were evaluated. Initial IDs, binary64 bit patterns, and both extremal torques remain unchanged. The model is v = omega + (15 sin(theta) + 3u)/20 and theta_next = theta + v/20, with exact rational parameters, |theta|<=1, |omega|<=3, |u|<=2, and horizon 200 transitions (201 sampled states).", '',
             'On the safe angle strip, cos(theta)>=1/2 and every state/action derivative is nonnegative. A constant -2 path with a safe prefix and a strict upper crossing bounds every hypothetical safe admissible path from below; constant +2 provides the reflected lower-crossing certificate. A certified exit by k<=200 therefore rules out all safe admissible sequences through 200 steps. All prefix interval boxes must remain inside the safety set. Opposite exits and boundary overlap never imply nonviability. Speed clipping is inactive on a step starting in the safety set because |omega_next|<=81/20<8.', '',
             'MPFR directed rounding uses exact integer division by 20. The independent checker imports neither MPFR nor the producer: it uses exact Fraction arithmetic and alternating Taylor bounds for sine to check every transition enclosure, identity, prefix, and strict crossing. Precision escalation remains 128, 256, then 512 bits only for boundary ambiguity. Exact rational endpoints and separations are preserved; decimal summaries are displays.', '',
             '## Outcomes and numerical diagnostics', '',
             f"Certificate directions: {n['certificate_direction_counts']}; components: {n['certificate_component_counts']}. Certified crossing steps range from {_number(n['certificate_crossing_steps']['min'])} to {_number(n['certificate_crossing_steps']['max'])}, or {_number(n['certificate_crossing_times']['min'])}–{_number(n['certificate_crossing_times']['max'])} s. Final precision counts across all extremal directions: {n['final_precision_counts']}.",
             f"Certified separation displays range from {_number(n['certificate_separation_lower_approximations']['min'], '.12g')} to {_number(n['certificate_separation_lower_approximations']['max'], '.12g')} (in the crossed component's units). Maximum certificate interval widths are {_number(n['certificate_theta_widths']['max'], '.6e')} rad and {_number(n['certificate_omega_widths']['max'], '.6e')} rad/s. The smallest certified-prefix normalized-margin lower bound is {_number(n['certificate_safe_prefix_margin_lower_approximations']['min'], '.12g')}.", '',
             f"False nonviability classifications among the {c['denominator']} PD-safe reference controls: **{c['false_nonviability_count']}**. Their PD trajectories remain simulator-safe in {c['simulator_safe_count']}/{c['denominator']} cases. No certificate for a control is not itself a mathematical viability proof. Equilibrium has the separate exact u=0 witness. All noncertifying directions are retained; final direction statuses are {n['direction_status_counts']}.", '',
             'Per-state certification status, direction, crossing step/time, separation, interval width, and precision appear in state_results.csv and state_details.md. direction_results.csv and the appendix also include every noncertifying extremal attempt.', '']
    if not p['unresolved_ids'] and not grid_run:
        lines += ['No PD-unsafe states remain unresolved, so additional analytical fallbacks and global viability-grid calculations were unnecessary and were not run.', '']
    else:
        lines += [f"Unresolved states remain unresolved in this report. Global viability grid run: {grid_run}. An absent certificate is not evidence of viability or nonviability; further authorized checks must preserve this distinction.", '']
    lines += ['## Simulator evidence and verification', '',
              f"{n['independently_checked_direction_count']}/{summary['direction_count']} extremal directions passed the independent checker. All retained precision attempts reproduced their interval records exactly. Technical tests passed: {tests.get('passed', verification.get('passed'))}; test count: {tests.get('tests', 'see verification.json')}. There are {n['PD_simulator_replays_bitwise']} bitwise archived-PD replay comparisons and {n['extremal_simulator_replays_bitwise']} bitwise extremal replay comparisons, including the reused pilot evidence.",
              f"Model and Gymnasium first-exit steps agree in {n['model_simulator_exit_step_matches']}/{summary['direction_count']} directions. Maximum observed prefix discrepancy bounds are {_number(n['observed_theta_prefix_error_upper']['max'], '.6g')} rad and {_number(n['observed_omega_prefix_error_upper']['max'], '.6g')} rad/s. These bounds concern only the saved prefixes and are not implementation-wide error bounds.", '',
              'Gymnasium uses float32 executed torques, binary floating-point arithmetic, and NumPy sine, whereas the certificate concerns a real-valued rational-parameter map. No justified global implementation-error enclosure is claimed. Simulator agreement does not upgrade the certificate to a proof about all floating-point executions, continuous-time motion, or physical hardware.', '',
              '## Computational cost and preservation', '',
              '| Work | Interval updates | Exact generation-repeat updates | Rational transition checks | Generation time (s) | Checker time (s) | Gym steps | Resets |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, key in [('New remaining states', 'new'), ('Reused pilot', 'reused_pilot'), ('Combined historical total', 'combined')]:
        row = costs.get(key, {})
        values = [row.get(k) for k in ('primary_interval_updates', 'repeat_interval_updates', 'checker_transition_checks', 'generation_seconds', 'checker_seconds', 'simulator_steps', 'simulator_resets')]
        lines.append('| ' + ' | '.join([name, *[_number(v) for v in values]]) + ' |')
    lines += ['', f"Preservation checks passed for {verification.get('protected_files', 'the recorded set of')} pre-existing files, including Experiments 01–07, the accepted Experiment 08 pilot, D0, and unfinished IQL diagnostics. Source hashes, initial-state identities, exact certificates, rejected/noncertifying attempts, and reproducibility metadata are retained. No previous data were modified; no RL agents were trained; no commit or push was performed.", '',
              '## Scientific interpretation and limitations', '',
              'The mathematical certificates answer a different question from the Experiment 07 local solver outcomes: they quantify over every admissible control sequence for the specified discrete model. A failed optimizer alone did not justify this conclusion. The present proof does not rely on solver infeasibility, action-grid failure, or extrapolating an extremal trajectory outside the comparison domain.', '',
              'The fixed sample was selected through prior PD failure, so these counts are not population-level failure probabilities. The constraints concern raw states at discrete samples, not continuous-time safety between samples. State/reference uncertainty, model mismatch, and an implementation-wide rounding contract are outside the proof. This analysis does not establish an infinite-horizon viable region or a deployable safety filter. Future acquisition can distinguish demonstrated model nonviability from poor learned control, but any changed acquisition domain and recursive safety conditions require separate study.', '']
    return '\n'.join(lines)


def write_outputs(output, summary, verification, costs):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(output / 'summary.json', summary)
    _write_csv(output / 'state_results.csv', summary['states'])
    _write_csv(output / 'direction_results.csv', summary['directions'])
    (output / 'state_details.md').write_text(state_details(summary))
    report = scientific_report(summary, verification, costs)
    (output / 'scientific_report.md').write_text(report)
    return report
