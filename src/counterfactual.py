"""Experiment 06A: one-action perturbations with an unchanged PD continuation.

Diagnostic only. This module does not load or write training datasets, fit a
controller, change the PD expert, or implement an online action selector.
"""

import csv
from pathlib import Path
import time

import numpy as np

from src.active_query import reset_at, sample_states
from src.dataset import get_state
from src.expert import KD, KP, MAX_TORQUE, pd_controller
from src.rollout_horizon import MeteredEnvironment
from src.safety import safety_margin


def fixed_state_menu(config):
    """Freeze states and proposal-to-execution identities before simulation."""
    nominal = sample_states(config['seeds']['nominal'], config['nominal_count'],
                            config['nominal_low'], config['nominal_high'])
    rng = np.random.default_rng(config['seeds']['challenging'])
    challenging, draws = [], 0
    low, high = np.array(config['nominal_low']), np.array(config['nominal_high'])
    while len(challenging) < config['challenging_count']:
        point = rng.uniform(config['challenging_low'], config['challenging_high'])
        draws += 1
        if not np.all((point >= low) & (point <= high)):
            challenging.append(point)
    initial = np.vstack([nominal, challenging, config['equilibrium_control']])
    canonical = np.array([get_state([np.cos(t), np.sin(t), w]) for t, w in initial])
    pd_actions = pd_controller(*canonical.T)
    offsets = np.asarray(config['offsets'], dtype=np.float64)
    proposals = pd_actions[:, None] + offsets
    executed = np.clip(proposals, -MAX_TORQUE, MAX_TORQUE).astype(np.float32)
    zero = int(np.flatnonzero(offsets == 0)[0])
    ids, actions, proposal_ids, baseline_ids, counts = [], [], [], [], []
    for i, row in enumerate(executed):
        unique, inverse = np.unique(row, return_inverse=True)
        proposal_ids.append(inverse + len(actions))
        baseline_ids.append(int(inverse[zero] + len(actions)))
        ids.extend([i] * len(unique))
        actions.extend(unique)
        counts.append(len(unique))
    assert all(safety_margin(*s) >= 0 for s in initial)
    return dict(
        initial_conditions=initial, canonical_initial_states=canonical,
        strata=np.array(['nominal'] * len(nominal) + ['challenging'] * len(challenging) + ['control']),
        saturated=np.abs(-KP * canonical[:, 0] - KD * canonical[:, 1]) >= MAX_TORQUE,
        offsets=offsets, proposed_torques=proposals, executed_proposals=executed,
        proposal_clipped=proposals != np.clip(proposals, -MAX_TORQUE, MAX_TORQUE),
        proposal_run_ids=np.array(proposal_ids), baseline_run_ids=np.array(baseline_ids),
        run_state_ids=np.array(ids), first_actions=np.array(actions, dtype=np.float32),
        distinct_action_counts=np.array(counts), challenging_draws=np.array(draws),
        reset_seeds=config['seeds']['environment'] + np.arange(len(initial)),
    )


def evaluate_first_actions(initial_conditions, first_actions, reset_seeds,
                           horizon=200, env_factory=MeteredEnvironment):
    """Full episodes; only t=0 is overridden, including after a safety crossing."""
    initial = np.asarray(initial_conditions, dtype=np.float64)
    first_actions = np.asarray(first_actions, dtype=np.float32)
    assert initial.shape == (len(first_actions), 2)
    assert len(reset_seeds) == len(initial)
    assert np.isfinite(initial).all() and np.isfinite(first_actions).all()
    assert np.all(np.abs(first_actions) <= MAX_TORQUE)
    n = len(initial)
    states = np.empty((n, horizon + 1, 2), dtype=np.float64)
    raw = np.empty_like(states)
    margins = np.empty((n, horizon + 1), dtype=np.float64)
    actions = np.empty((n, horizon, 1), dtype=np.float32)
    rewards = np.empty((n, horizon), dtype=np.float64)
    terminated = np.zeros((n, horizon), dtype=bool)
    truncated = np.zeros_like(terminated)
    env = env_factory(horizon)
    started = time.perf_counter()
    try:
        for i, initial_state in enumerate(initial):
            obs = reset_at(env, initial_state, int(reset_seeds[i]))
            np.testing.assert_array_equal(env.unwrapped.state, initial_state)
            states[i, 0] = get_state(obs)
            raw[i, 0] = env.unwrapped.state
            margins[i, 0] = safety_margin(*raw[i, 0])
            for t in range(horizon):
                action = first_actions[i] if t == 0 else pd_controller(*states[i, t])
                actions[i, t, 0] = action
                obs, reward, term, trunc = env.step(actions[i, t])[:4]
                states[i, t + 1] = get_state(obs)
                raw[i, t + 1] = env.unwrapped.state
                margins[i, t + 1] = safety_margin(*raw[i, t + 1])
                rewards[i, t] = reward
                terminated[i, t], truncated[i, t] = term, trunc
                if term or (trunc and t != horizon - 1):
                    raise RuntimeError('Unexpected termination semantics.')
        costs = env.costs()
    finally:
        env.close()
    costs['wall_seconds'] = np.array(time.perf_counter() - started)
    result = dict(initial_conditions=initial.copy(), states=states, raw_states=raw,
                  margins=margins, actions=actions, rewards=rewards,
                  terminated=terminated, truncated=truncated,
                  episode_returns=rewards.sum(axis=1), episode_min_margins=margins.min(axis=1))
    validate_rollouts(result, first_actions, horizon)
    assert int(costs['step_attempts']) == n * horizon
    assert int(costs['reset_attempts']) == n
    return result, costs


def validate_rollouts(arrays, first_actions, horizon):
    n = len(first_actions)
    assert arrays['raw_states'].shape == (n, horizon + 1, 2)
    assert arrays['actions'].shape == (n, horizon, 1)
    for value in arrays.values():
        assert np.isfinite(value).all()
    np.testing.assert_array_equal(arrays['raw_states'][:, 0], arrays['initial_conditions'])
    np.testing.assert_array_equal(arrays['actions'][:, 0, 0], first_actions)
    expected = pd_controller(*arrays['states'][:, 1:-1].transpose(2, 0, 1)).astype(np.float32)
    np.testing.assert_array_equal(arrays['actions'][:, 1:, 0], expected)
    expected_margins = np.array([[safety_margin(*s) for s in row] for row in arrays['raw_states']])
    np.testing.assert_array_equal(arrays['margins'], expected_margins)
    np.testing.assert_array_equal(arrays['episode_returns'], arrays['rewards'].sum(axis=1))
    np.testing.assert_array_equal(arrays['episode_min_margins'], expected_margins.min(axis=1))
    assert not arrays['terminated'].any()
    assert not arrays['truncated'][:, :-1].any()
    assert arrays['truncated'][:, -1].all()


def assert_bitwise_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        a, b = np.asarray(left[key]), np.asarray(right[key])
        assert a.dtype == b.dtype and a.shape == b.shape, key
        assert a.tobytes() == b.tobytes(), key


def compare_PD_reference(left, right):
    """Require identical arrays except explicitly enumerated zero torque signs.

    PD(0,0) is -0.0; adding the prescribed +0.0 offset produces +0.0.
    The exception is confined to first-step zero actions and does not permit
    tolerances in nonzero torques, states, rewards, margins, or flags.
    """
    assert left.keys() == right.keys()
    signed_zero_indices = []
    for key in left:
        a, b = left[key], right[key]
        assert a.dtype == b.dtype and a.shape == b.shape, key
        np.testing.assert_array_equal(a, b, err_msg=key)
        if a.tobytes() != b.tobytes():
            assert key == 'actions'
            indices = np.argwhere(np.signbit(a) != np.signbit(b))
            for index in indices:
                assert index[1] == 0 and a[tuple(index)] == b[tuple(index)] == 0.
            canonical = a.copy()
            canonical[tuple(indices.T)] = b[tuple(indices.T)]
            assert canonical.tobytes() == b.tobytes()
            signed_zero_indices.extend(indices.tolist())
    return signed_zero_indices


def episode_metrics(arrays, discount, dt):
    margins = arrays['margins']
    violation = (margins < 0).any(axis=1)
    first = np.where(violation, np.argmax(margins < 0, axis=1), -1)
    return dict(
        returns=arrays['rewards'].sum(axis=1),
        discounted_returns=arrays['rewards'] @ discount ** np.arange(arrays['rewards'].shape[1]),
        minimum_margins=margins.min(axis=1), violated=violation,
        unsafe_state_fraction=(margins < 0).mean(axis=1),
        first_crossing_step=first,
        first_crossing_seconds=np.where(violation, first * dt, -1.0),
    )


def classify_alternatives(pd_unsafe, alternative_unsafe, delta_return, delta_discounted, threshold):
    """Categories overlap only where stated; require both return improvements."""
    improves = (delta_return > threshold) & (delta_discounted > threshold)
    return {
        'safe_improvement': ~alternative_unsafe & improves,
        'safety_rescue': pd_unsafe & ~alternative_unsafe,
        'safety_rescue_and_improvement': pd_unsafe & ~alternative_unsafe & improves,
        'return_improvement_without_rescue': pd_unsafe & alternative_unsafe & improves,
        'undiscounted_improvement_without_rescue': pd_unsafe & alternative_unsafe & (delta_return > threshold),
        'discounted_improvement_without_rescue': pd_unsafe & alternative_unsafe & (delta_discounted > threshold),
        'introduced_violation': ~pd_unsafe & alternative_unsafe,
        'introduced_violation_with_improvement': ~pd_unsafe & alternative_unsafe & improves,
        'return_only_improvement': (delta_return > threshold) & ~(delta_discounted > threshold),
        'discounted_only_improvement': (delta_discounted > threshold) & ~(delta_return > threshold),
    }


def descriptive(values):
    values = np.asarray(values)
    if not len(values):
        return {'count': 0, 'mean': None, 'median': None, 'min': None, 'max': None}
    return {'count': len(values), 'mean': float(values.mean()), 'median': float(np.median(values)),
            'min': float(values.min()), 'max': float(values.max())}


def state_bootstrap(values, state_ids, strata, seed, repeats):
    """Keep the nominal/challenging composition fixed, with states as units."""
    values, state_ids = np.asarray(values, dtype=float), np.asarray(state_ids)
    if not len(values):
        return None
    rng = np.random.default_rng(seed)
    means = np.zeros(repeats)
    for name in ('nominal', 'challenging'):
        subset = values[strata[state_ids] == name]
        if len(subset):
            draws = rng.integers(len(subset), size=(repeats, len(subset)))
            means += subset[draws].sum(axis=1) / len(values)
    return np.quantile(means, [.025, .975]).tolist()


def analyze(menu, arrays, config):
    metrics = episode_metrics(arrays, config['discount'], config['dt'])
    sid, baseline = menu['run_state_ids'], menu['baseline_run_ids']
    b = baseline[sid]
    dr = metrics['returns'] - metrics['returns'][b]
    dg = metrics['discounted_returns'] - metrics['discounted_returns'][b]
    alt = np.arange(len(sid)) != b
    pd_bad = metrics['violated'][baseline]
    sampled = menu['strata'] != 'control'
    masks = {'all_sampled': sampled, 'nominal': menu['strata'] == 'nominal',
             'challenging': menu['strata'] == 'challenging',
             'saturated': sampled & menu['saturated'], 'unsaturated': sampled & ~menu['saturated'],
             'PD_safe': sampled & ~pd_bad, 'PD_violating': sampled & pd_bad}
    for name in ('nominal', 'challenging'):
        for suffix, condition in [('saturated', menu['saturated']), ('unsaturated', ~menu['saturated']),
                                  ('PD_safe', ~pd_bad), ('PD_violating', pd_bad)]:
            masks[name + '_' + suffix] = masks[name] & condition
    groups = {}
    for name, mask in masks.items():
        ids = np.flatnonzero(mask)
        selected = mask[sid] & alt
        all_runs = mask[sid]
        group = {
            'states': len(ids), 'distinct_actions': int(all_runs.sum()), 'alternatives': int(selected.sum()),
            'PD_violating_states': int(pd_bad[ids].sum()),
            'PD_return': descriptive(metrics['returns'][baseline[ids]]),
            'PD_discounted_return': descriptive(metrics['discounted_returns'][baseline[ids]]),
            'PD_minimum_margin': descriptive(metrics['minimum_margins'][baseline[ids]]),
            'all_action_minimum_margin': descriptive(metrics['minimum_margins'][all_runs]),
            'PD_first_crossing_seconds_observed': descriptive(metrics['first_crossing_seconds'][baseline[ids[pd_bad[ids]]]]),
            'thresholds': {}, 'per_offset': {},
        }
        for threshold in config['thresholds']:
            classes = classify_alternatives(pd_bad[sid], metrics['violated'], dr, dg, threshold)
            entry = {}
            for category, flags in classes.items():
                eligible = selected & flags
                hit_ids = np.unique(sid[eligible])
                state_hits = np.isin(ids, hit_ids)
                entry[category] = {
                    'actions': int(eligible.sum()), 'states': len(hit_ids),
                    'state_fraction': float(state_hits.mean()) if len(ids) else None,
                    'state_fraction_ci95': state_bootstrap(state_hits, ids, menu['strata'],
                                                          config['seeds']['bootstrap'], config['bootstrap_replicates']),
                    'state_ids': hit_ids.tolist(),
                }
            eligible = selected & classes['safe_improvement']
            entry['safe_improvement_gain_distribution'] = {
                'delta_return': descriptive(dr[eligible]),
                'delta_discounted_return': descriptive(dg[eligible]),
                'minimum_raw_margin': descriptive(metrics['minimum_margins'][eligible]),
            }
            group['thresholds'][str(threshold)] = entry
        for j, offset in enumerate(menu['offsets']):
            runs = menu['proposal_run_ids'][ids, j]
            group['per_offset'][str(offset)] = {
                'distinct_from_PD_states': int(alt[runs].sum()),
                'delta_return': descriptive(dr[runs]),
                'delta_discounted_return': descriptive(dg[runs]),
                'delta_return_mean_ci95': state_bootstrap(dr[runs], ids, menu['strata'], config['seeds']['bootstrap'], config['bootstrap_replicates']),
                'delta_discounted_mean_ci95': state_bootstrap(dg[runs], ids, menu['strata'], config['seeds']['bootstrap'], config['bootstrap_replicates']),
                'violations': int(metrics['violated'][runs].sum()),
                'minimum_margin': descriptive(metrics['minimum_margins'][runs]),
            }
        groups[name] = group
    rows, state_rows = [], []
    for i in range(len(sid)):
        state_id = int(sid[i])
        row = {'run_id': i, 'state_id': state_id, 'stratum': str(menu['strata'][state_id]),
               'theta0': float(menu['initial_conditions'][state_id, 0]),
               'omega0': float(menu['initial_conditions'][state_id, 1]),
               'PD_saturated': bool(menu['saturated'][state_id]), 'is_PD': not bool(alt[i]),
               'executed_action': float(menu['first_actions'][i]),
               'action_deviation': float(menu['first_actions'][i] - menu['first_actions'][b[i]]),
               'delta_return': float(dr[i]), 'delta_discounted_return': float(dg[i]),
               'PD_violated': bool(pd_bad[state_id])}
        row.update({key: value[i].item() for key, value in metrics.items()})
        if not row['violated']:
            row['first_crossing_step'] = row['first_crossing_seconds'] = None
        rows.append(row)
    for state_id in range(len(baseline)):
        candidates = np.flatnonzero((sid == state_id) & alt)
        safe = candidates[~metrics['violated'][candidates]]
        # A single oracle choice, maximizing discounted return among safe
        # alternatives. Both metrics refer to that SAME action, not two maxima.
        best = int(safe[np.argmax(metrics['discounted_returns'][safe])]) if len(safe) else None
        state_rows.append({
            'state_id': state_id, 'stratum': str(menu['strata'][state_id]),
            'PD_saturated': bool(menu['saturated'][state_id]), 'PD_violated': bool(pd_bad[state_id]),
            'distinct_actions': int(menu['distinct_action_counts'][state_id]),
            'safe_alternatives': len(safe), 'oracle_best_safe_alternative_run_id': best,
            'oracle_delta_return': None if best is None else float(dr[best]),
            'oracle_delta_discounted_return': None if best is None else float(dg[best]),
            **{f'safe_joint_gain_gt_{t}': bool(np.any((dr[safe] > t) & (dg[safe] > t))) for t in config['thresholds']},
        })
    for name, mask in masks.items():
        selected_oracles = [r for r in state_rows if mask[r['state_id']] and r['oracle_best_safe_alternative_run_id'] is not None]
        groups[name]['oracle_best_safe_alternative'] = {
            'available_states': len(selected_oracles),
            'missing_states': int(mask.sum()) - len(selected_oracles),
            'delta_return': descriptive([r['oracle_delta_return'] for r in selected_oracles]),
            'delta_discounted_return': descriptive([r['oracle_delta_discounted_return'] for r in selected_oracles]),
        }
    return {'groups': groups, 'total_states_with_control': len(baseline),
            'distinct_executions': len(sid), 'distinct_alternatives': int(alt.sum()),
            'duplicate_proposals_avoided': int(menu['executed_proposals'].size - len(sid)),
            'distinct_action_count_distribution': {str(k): int(np.sum(menu['distinct_action_counts'] == k)) for k in np.unique(menu['distinct_action_counts'])},
            'oracle_definition': 'Highest discounted return among distinct empirically safe alternatives; no PD fallback; both gains use the same action; not deployable policy performance',
            'bootstrap_note': 'State bootstrap within nominal/challenging strata; descriptive, no multiple-comparison correction; zero/all-hit intervals can degenerate and do not imply certainty',
            'minimum_raw_margin_all_runs': float(metrics['minimum_margins'].min())}, rows, state_rows


def write_csv(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figure(path, menu, rows, state_rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    nominal = menu['strata'] == 'nominal'
    hits = np.array([r['safe_joint_gain_gt_1e-06'] for r in state_rows])
    for mask, marker, label in [(nominal, 'o', 'Nominal'), (menu['strata'] == 'challenging', '^', 'Challenging')]:
        axes[0].scatter(*menu['initial_conditions'][mask].T, c=hits[mask], cmap='coolwarm', vmin=0, vmax=1,
                        marker=marker, label=label, edgecolor='black', linewidth=.3)
    axes[0].set(xlabel='Initial raw angle', ylabel='Initial angular velocity', title='Safe joint improvement: red=yes, blue=no')
    axes[0].legend()
    alternatives = [r for r in rows if not r['is_PD'] and r['stratum'] != 'control']
    for unsafe, label, color in [(False, 'No observed violation', 'tab:blue'), (True, 'Observed violation', 'tab:red')]:
        chosen = [r for r in alternatives if r['violated'] == unsafe]
        axes[1].scatter([r['delta_return'] for r in chosen], [r['delta_discounted_return'] for r in chosen],
                        s=14, alpha=.65, label=label, color=color)
        axes[2].scatter([r['minimum_margins'] for r in chosen], [r['delta_discounted_return'] for r in chosen],
                        s=14, alpha=.65, color=color)
    for ax in axes[1:]:
        ax.axhline(0, color='gray', linewidth=.5)
        ax.axvline(0, color='gray', linewidth=.5)
        ax.set_yscale('symlog', linthresh=.01)
    axes[1].set_xscale('symlog', linthresh=.01)
    axes[1].set(xlabel='Paired undiscounted gain', ylabel='Paired discounted gain', title='All distinct alternatives')
    axes[1].legend(fontsize=8)
    axes[2].set(xlabel='Minimum raw safety margin', ylabel='Paired discounted gain', title='Return improvement can remain unsafe')
    fig.savefig(path, dpi=170)
    plt.close(fig)


def scientific_report(summary, checks, costs):
    groups = summary['groups']
    lines = [
        '# Experiment 06A — Counterfactual Action Feasibility', '',
        'Simulation-only diagnostic. No controller was trained, no safety filter was introduced, and no diagnostic transition was added to a training dataset.', '',
        '## Fixed protocol', '',
        '32 nominal states uniformly sampled from [-0.25,0.25] × [-0.5,0.5] (seed 606201); 64 challenging states uniformly sampled from [-0.8,0.8] × [-2,2] excluding that nominal rectangle (seed 606202); one equilibrium technical control, excluded from sampled-state summaries. States were frozen before simulation.', '',
        'Offsets {-0.4,-0.2,0,+0.2,+0.4} N·m were added to the already clipped PD action and deduplicated after clipping and float32 conversion. Each branch used the identical float64 physical initial state and reset seed 606203 + state ID. One candidate action was followed by the unchanged PD controller for 199 steps. Discount 0.99; 200 steps (10 seconds); no stopping at safety crossings. All 201 raw states were checked against |theta| ≤ 1 and |omega| ≤ 3.', '',
        '## Safe alternatives improving both return metrics', '',
        'A hit requires a distinct action, minimum raw margin ≥ 0, and both return gains strictly greater than the stated threshold. Percentages are fractions of states, not fractions of action proposals.', '',
        '| Group | States | PD violations | >1e-6 | >0.01 | >0.1 |',
        '|---|---:|---:|---:|---:|---:|',
    ]
    for name in ('all_sampled', 'nominal', 'challenging', 'saturated', 'unsaturated', 'PD_safe', 'PD_violating'):
        g = groups[name]
        cells = []
        for threshold in ('1e-06', '0.01', '0.1'):
            hit = g['thresholds'][threshold]['safe_improvement']
            cells.append(f"{hit['states']}/{g['states']} ({100 * hit['state_fraction']:.2f}%)" if g['states'] else 'N/A')
        lines.append(f"| {name} | {g['states']} | {g['PD_violating_states']} | " + ' | '.join(cells) + ' |')
    lines += ['', '95% percentile intervals use 10,000 state-level bootstrap resamples within nominal/challenging strata (seed 606204). These describe sampled-state sensitivity, not safety guarantees; degenerate zero-hit intervals do not establish impossibility.']
    for threshold in ('1e-06', '0.01', '0.1'):
        hit = groups['all_sampled']['thresholds'][threshold]['safe_improvement']
        ci = hit['state_fraction_ci95']
        lines.append(f"- Joint-gain threshold {threshold}: {hit['state_fraction']:.6f}; 95% interval [{ci[0]:.6f}, {ci[1]:.6f}].")
    lines += ['', '## Safety and negative findings', '',
              '| Group | Rescue: actions / states | Both returns improve but PD and alternative remain unsafe: actions / states | New violation: actions / states |',
              '|---|---:|---:|---:|']
    for name in ('all_sampled', 'nominal', 'challenging', 'saturated', 'unsaturated'):
        entries = groups[name]['thresholds']['1e-06']
        cells = [f"{entries[k]['actions']} / {entries[k]['states']}" for k in ('safety_rescue', 'return_improvement_without_rescue', 'introduced_violation')]
        lines.append('| ' + name + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '| Group | PD mean / worst return | PD mean discounted return | PD minimum raw margin | All-action minimum raw margin |',
              '|---|---:|---:|---:|---:|']
    for name in ('nominal', 'challenging', 'saturated', 'unsaturated'):
        g = groups[name]
        lines.append(f"| {name} | {g['PD_return']['mean']:.6f} / {g['PD_return']['min']:.6f} | {g['PD_discounted_return']['mean']:.6f} | {g['PD_minimum_margin']['min']:.6f} | {g['all_action_minimum_margin']['min']:.6f} |")
    lines += ['', 'Every action’s paired gains, minimum raw margin, first-crossing step/time (null when censored), and unsafe-state fraction are in `episode_metrics.csv`. `summary.json` includes all stratum intersections, fixed-offset paired summaries and intervals, and cases improving only one return metric. `state_metrics.csv` records per-state eligibility and oracle results.', '',
              '## Oracle diagnostics', '',
              'The per-state oracle chooses the distinct, empirically safe alternative with the highest discounted return, if one exists. Both reported gains use that same action; states without a safe alternative remain missing. This uses rollout outcomes and is not deployable policy performance. Existence counts above do not require this oracle choice to maximize the undiscounted return.', '',
              '| Group | Available / missing states | Mean / minimum / maximum paired return gain | Mean / minimum / maximum paired discounted gain |',
              '|---|---:|---:|---:|']
    for name in ('nominal', 'challenging', 'saturated', 'unsaturated'):
        oracle = groups[name]['oracle_best_safe_alternative']
        cells = []
        for key in ('delta_return', 'delta_discounted_return'):
            d = oracle[key]
            cells.append(' / '.join(f'{d[k]:.6f}' for k in ('mean', 'min', 'max')) if d['count'] else 'N/A')
        lines.append(f"| {name} | {oracle['available_states']} / {oracle['missing_states']} | " + ' | '.join(cells) + ' |')
    gains = groups['all_sampled']['thresholds']['1e-06']['safe_improvement_gain_distribution']
    if gains['minimum_raw_margin']['count']:
        lines += ['', f"Across qualifying distinct alternatives, minimum raw safety margin was {gains['minimum_raw_margin']['min']:.6f}; joint-improvement undiscounted gains ranged from {gains['delta_return']['min']:.8f} to {gains['delta_return']['max']:.6f}, and discounted gains from {gains['delta_discounted_return']['min']:.8f} to {gains['delta_discounted_return']['max']:.6f}."]
    lines += ['', '## Verification and interaction costs', '',
              f"- {summary['distinct_executions']} distinct executions including PD and equilibrium; {summary['distinct_alternatives']} distinct alternatives; {summary['duplicate_proposals_avoided']} duplicate proposals avoided per pass.",
              f"- Distinct-action counts per state: {summary['distinct_action_count_distribution']}.",
              f"- Main pass: {costs['main']['resets']} resets and {costs['main']['steps']} transitions. Replay: {costs['replay']['resets']} resets and {costs['replay']['steps']} transitions. Total: {costs['total_resets']} resets and {costs['total_steps']} transitions.",
              f"- Main simulator wall time: {costs['main']['wall_seconds']:.3f} s; replay simulator wall time: {costs['replay']['wall_seconds']:.3f} s. File hashing, tests, analysis, and plotting are excluded from these simulator timings.",
              f"- All technical checks passed: {checks['passed']}. Every unique action has a bitwise replay. All 97 PD branches match the preserved evaluator numerically, including bitwise-identical physical states, rewards, and margins. The original evaluator emits -0.0 at the equilibrium first action while the prescribed offset menu emits +0.0; this sole sign-bit exception is documented, and an extra direct equilibrium replay is bitwise identical. It costs one extra reset and 200 steps, included above. Full-horizon/raw-safety checks and equilibrium reward-ceiling checks passed.",
              f"- SHA-256 preservation checks passed for {checks['protected_files']} pre-existing files, including all prior datasets, results, checkpoints, and source files. No prior experiment was rerun.", '',
              '## Interpretation and limitations', '',
              'Positive hits establish local counterfactual action opportunities under this fixed simulator, menu, PD continuation, and finite horizon. Return improvements on violating trajectories do not establish safe improvement. An absence of rescues does not establish that the unsafe states are unrecoverable under other actions or policies.', '',
              'The 32/64 mixture is a diagnostic sampling design, not a natural operating distribution. Saturation and PD-failure subgroups are descriptive; their sample counts differ. Exact replays are not independent observations. Bootstrap intervals are pointwise, without multiple-comparison correction. Very small positive gains can exceed the numerical threshold without practical importance.', '',
              'A one-step change followed by PD does not prove multi-step policy improvement. Repeated intervention would change future state visitation and safety. Discounted 200-step returns are not infinite-horizon IQL values. Best-of-menu outcomes use privileged outcome information. Deterministic simulator resets do not establish robustness to model error, noise, or real systems. PD is not a certified backup controller, and no observed violation over sampled discrete steps is not a formal safety guarantee.', '',
              'All diagnostic trajectories remain isolated under `results/offline_rl/experiment_06A/`. Existing training datasets and IQL implementations remain unchanged.', '']
    return '\n'.join(lines)
