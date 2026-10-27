"""Paired seed-level analysis only; never feeds acquisition or training."""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.acquisition_analysis import flatten_numeric, seed_summary
from src.horizon_experiment import (condition_name, immutable_json, load_npz,
    read_json, write_json, _assert_hashes)
from src.horizon_reporting import table
from src.horizon_study import SEEDS, completed, frozen_npz


def summarize_records(records, draws):
    flat = [flatten_numeric(r) for r in records]
    common = set.intersection(*(set(r) for r in flat))
    return {key: seed_summary([r[key] for r in flat], draws) for key in sorted(common)}


def summarize(output):
    output = Path(output)
    metrics, diagnostics, safety, contrasts, costs = [], [], [], [], []
    config = read_json(output / 'protocol.json')
    for seed in SEEDS:
        folder = output / f'seed_{seed}'
        if not completed(folder):
            raise RuntimeError(f'Seed {seed} incomplete: refusing partial or selected-seed inference.')
        metrics.append(read_json(folder / 'metrics.json'))
        diagnostics.append(read_json(folder / 'acquisition_diagnostics.json'))
        safety.append(read_json(folder / 'post_selection_safety_audit/condition_summaries.json'))
        contrasts.append(read_json(folder / 'paired_differences.json'))
        costs.append(read_json(folder / 'cost_accounting.json'))
    inference = config['inference']
    draws = np.random.default_rng(inference['bootstrap_seed']).integers(
        0, len(SEEDS), size=(inference['bootstrap_repeats'], len(SEEDS)))
    frozen_npz(output / 'bootstrap_indices.npz', {'indices': draws, 'master_seeds': np.array(SEEDS)})
    conditions = {name: summarize_records([m[name] for m in metrics], draws) for name in metrics[0]}
    differences = {str(b): {name: summarize_records([c[str(b)][name] for c in contrasts], draws)
                          for name in contrasts[0][str(b)]} for b in config['budgets']}
    collection = {name: summarize_records([d[name] for d in diagnostics], draws) for name in diagnostics[0]}
    audit = {name: summarize_records([a[name] for a in safety], draws) for name in safety[0]}
    pooled_audit = {}
    for name in safety[0]:
        eligible = [a[name]['safely_capped_at_20_count'] for a in safety]
        late = [a[name]['subsequent_violation_steps_21_to_200_count'] for a in safety]
        pooled_audit[name] = {'eligible_per_seed': eligible, 'late_per_seed': late,
            'eligible_total': sum(eligible), 'late_total': sum(late),
            'pooled_late_fraction': sum(late) / sum(eligible) if sum(eligible) else None,
            'note': 'Pooled fraction is descriptive. Seed-level fractions weight eligible seeds equally; no fictitious zero for undefined fractions.'}
    primary_name = 'uncertainty_H20_B5000_minus_uncertainty_H200_B5000'
    primary = {key: differences['5000'][primary_name][key]
               for key in ('wide.mean_return', 'wide.safety_violation_rate')}
    summary = {'seeds': list(SEEDS), 'pilot_excluded': True,
        'inference': dict(inference, conditioning='Fixed common evaluation states; intervals quantify acquisition/training seed variation.',
                          multiplicity='Separate marginal intervals; no joint or familywise adjustment. Secondary outcomes descriptive.'),
        'primary': primary, 'condition_statistics': conditions,
        'paired_differences_by_budget': differences, 'acquisition_statistics': collection,
        'safety_audit_statistics': audit, 'safety_audit_pooled_counts': pooled_audit,
        'cost_statistics': summarize_records(costs, draws),
        'conditional_fields': 'Only fields defined for all ten seeds are summarized with seed CIs. All per-seed diagnostics, including absent conditional strata, are preserved.',
        'acquisition_transitions_total': sum(c['acquisition_transitions'] for c in costs),
        'post_selection_audit_transitions_total': sum(c['post_selection_safety_audit_transitions'] for c in costs),
        'post_selection_audit_resets_total': sum(c['post_selection_safety_audit_resets'] for c in costs)}
    immutable_json(output / 'summary.json', summary)
    immutable_json(output / 'primary_comparison.json', primary)
    save_figures(output, summary, config)
    save_report(output, summary, config)
    return summary


def interval(r, factor=1, decimals=3):
    low, high = r['bootstrap_95_ci_mean']
    return f'[{factor*low:.{decimals}f}, {factor*high:.{decimals}f}]'


def contrast_rows(stats):
    rows = []
    for key, label, factor in [('wide.mean_return', 'Mean episode return', 1),
                               ('wide.safety_violation_rate', 'Violation rate (percentage points)', 100)]:
        r = stats[key]
        rows.append([label, f"{factor*r['mean']:+.4f}", f"{factor*r['sample_std']:.4f}",
                     f"{factor*r['minimum']:+.4f}", f"{factor*r['maximum']:+.4f}", interval(r, factor, 4)])
    return rows


def save_report(output, summary, config):
    c, a = summary['condition_statistics'], summary['acquisition_statistics']
    lines = ['# Experiment 05 — full study, seeds 52–61',
        'All ten prespecified paired seeds are included; seed 41 is excluded. Unchanged D0 (20,000 transitions), PD expert, raw-state safety stopping, five-member trajectory-bootstrap BC, 2×64 ReLU architecture, optimizer and OOB checkpoint selection. Each seed shares one 10,000-state pool and frozen rankings across horizons. The six conditions receive exact nested budgets of 2,000, 5,000 and 10,000 transitions.',
        'Evaluation uses the same independent 1,000 wide and 200 nominal initial conditions for every controller, full 200-step episodes even after safety violations, and 10,000 independent expert-action audit states. A deterministic PD reference is evaluated once. Raw unwrapped simulator states determine safety; individually clipped member actions are averaged for deployment.',
        '## Primary comparison: uncertainty H20 minus H200 at 5,000 transitions',
        'Each seed contributes one paired difference. SD uses ddof=1; intervals are marginal 95% percentile intervals from 10,000 paired seed resamples (seed 505010). Positive return differences favor H20; negative violation-rate differences favor H20. Intervals condition on the fixed held-out states and do not quantify test-population uncertainty or provide a joint two-outcome guarantee.',
        table(['Outcome', 'Paired mean', 'Sample SD', 'Minimum', 'Maximum', '95% CI for mean'], contrast_rows(summary['primary']))]
    rows = []
    names = ['PD', 'D0'] + [condition_name(s, h, 5000) for s in config['strategies'] for h in config['collection_horizons']]
    for name in names:
        r = c[name]
        rows.append([name, f"{r['wide.mean_return']['mean']:.3f}", f"{100*r['wide.safety_violation_rate']['mean']:.3f}%",
            f"{r['wide.worst_return']['mean']:.3f}", f"{r['wide.worst_return']['minimum']:.3f}",
            f"{r['wide.minimum_raw_safety_margin']['minimum']:.6f}",
            '—' if name == 'PD' else f"{r['audit.all.bounded_action_error.mae']['mean']:.6f}"])
    lines += ['## Controller outcomes at 5,000 transitions',
        table(['Condition', 'Mean return', 'Violating episodes', 'Mean seed worst return', 'Worst overall return', 'Minimum raw margin overall', 'Mean action MAE'], rows),
        'Mean seed minima and worst-decile returns, nominal-region outcomes, MAE/RMSE/p95/max errors, saturated/unsaturated strata and raw/clipped disagreement are retained with per-seed values, sample SD, ranges and seed-bootstrap intervals in `summary.json`. The minimum raw margin is taken over all 201 visited states per episode.']
    rows = []
    for name in names[2:]:
        r = c[name]
        rows.append([name, f"{r['wide.worst_10_percent_mean_return']['mean']:.3f}",
            f"{r['wide.minimum_raw_safety_margin']['mean']:.6f}",
            f"{r['nominal.mean_return']['mean']:.6f}", f"{100*r['nominal.safety_violation_rate']['mean']:.3f}%",
            f"{r['audit.unsaturated.bounded_action_error.mae']['mean']:.6f}",
            f"{r['audit.saturated.bounded_action_error.mae']['mean']:.6f}"])
    lines += [table(['Condition', 'Worst-decile mean return', 'Mean seed minimum margin', 'Nominal return', 'Nominal violations', 'Unsaturated MAE', 'Saturated MAE'], rows)]
    rows = []
    d = summary['paired_differences_by_budget']['5000']
    comparisons = [('random_H20_B5000_minus_random_H200_B5000', 'Random H20−H200'),
                   ('distance_H20_B5000_minus_distance_H200_B5000', 'Distance H20−H200'),
                   ('uncertainty_H20_B5000_minus_random_H20_B5000', 'Uncertainty−random at H20'),
                   ('uncertainty_horizon_effect_minus_random_horizon_effect', '(U20−U200)−(R20−R200)')]
    for key, label in comparisons:
        for row in contrast_rows(d[key]): rows.append([label, *row])
    lines += ['## Secondary paired comparisons at 5,000 transitions',
        table(['Comparison', 'Outcome', 'Paired mean', 'Sample SD', 'Minimum', 'Maximum', '95% CI'], rows)]
    rows = []
    for name in names[2:]:
        r = a[name]
        rows.append([name, f"{r['coverage.tail.distribution.mean']['mean']:.5f}",
            f"{r['coverage.tail.distribution.p95']['mean']:.5f}",
            f"{100*r['coverage.transition_fraction_near_equilibrium.0.05']['mean']:.2f}%",
            f"{100*r['coverage.transition_fraction_near_D0.0.01']['mean']:.2f}%",
            f"{r['coverage.bin_coverage.0.05.additional_occupied_bins']['mean']:.1f}",
            f"{r['operations.resets']['mean']:.1f}", f"{r['operations.collection_safety_crossings']['mean']:.1f}"])
    lines += ['## Acquired-transition coverage and collection costs',
        table(['Condition', 'Mean D0 distance', 'Mean seed p95', 'Near equilibrium', 'Near D0', 'Additional bins (.05)', 'Mean resets', 'Mean collection violations'], rows),
        'These summarize newly collected pre-action transitions, not only selected initial conditions. Distance uses [theta, omega/3]. Near-equilibrium radius is 0.05; near-D0 distance is 0.01. Both transition-weighted and equal-trajectory summaries, observed-step profiles and tails are saved. Unobserved continuation is never imputed as zero distance. Bin coverage depends on resolution.']
    rows = []
    for name in names[2:]:
        r = a[name]
        rows.append([name, *[f"{r[f'operations.transition_equivalent_cost.{cost}']['mean']:.0f}" for cost in (0, 20, 200)],
                     f"{r['operations.fraction_rollouts_with_observed_crossing']['mean']*100:.2f}%"])
    lines += [table(['Condition', 'Cost λ=0', 'Cost λ=20', 'Cost λ=200', 'Mean observed crossing fraction'], rows),
        'Cost is B + λ×resets, with illustrative reset equivalents. Fewer observed collection violations do not imply safer continuation: exposure, reset count and ranking depth differ. Acquisition, expert audit, simulator evaluation, training and technical checks have separate cost records.']
    rows = []
    for name in names[2:]:
        r = a[name]
        rows.append([name, f"{r['selection.bounded_disagreement.mean']['mean']:.5f}",
            f"{r['selection.raw_disagreement.mean']['mean']:.5f}",
            f"{r['selection.bounded_disagreement.minimum']['mean']:.5f}",
            f"{r['selection.bounded_disagreement.p95']['mean']:.5f}",
            f"{100*r['selection.fraction_zero_bounded_disagreement']['mean']:.2f}%",
            f"{r['selection.rank_positions_1based.uncertainty.median']['mean']:.1f}",
            f"{r['selection.rank_positions_1based.uncertainty.maximum']['mean']:.1f}"])
    lines += ['## Queried initial-condition score distributions',
        table(['Condition', 'Mean bounded std', 'Mean raw std', 'Mean minimum bounded std', 'Mean p95 bounded std', 'Zero bounded std', 'Mean median U-rank', 'Mean maximum U-rank'], rows),
        'Scores are initial-condition-weighted within seed, then seeds are equally weighted. Complete distributions, rank positions for every strategy, positive-score pool consumption, last-query scores, query order and selected support-distance scores are saved for all budgets. Random uses a uniform permutation and has no numerical acquisition score.']
    rows = []
    for strategy in config['strategies']:
        name = condition_name(strategy, 20, 5000)
        p = summary['safety_audit_pooled_counts'][name]
        r = summary['safety_audit_statistics'][name]
        key = 'subsequent_violation_fraction_among_safely_capped_20'
        rate = r.get(key)
        rows.append([strategy, p['eligible_total'], p['late_total'],
            'N/A' if p['pooled_late_fraction'] is None else f"{100*p['pooled_late_fraction']:.3f}%",
            'See per-seed records' if rate is None else f"{100*rate['mean']:.3f}%",
            'N/A' if rate is None else interval(rate, 100)])
    lines += ['## Separate post-selection expert safety audit',
        table(['H20 strategy', 'Safe caps at 20 (total)', 'Subsequent violations (total)', 'Pooled late fraction', 'Mean seed late fraction', '95% seed CI (pp scale)'], rows),
        f"The audit used {summary['post_selection_audit_transitions_total']:,} additional expert transitions and {summary['post_selection_audit_resets_total']:,} resets, outside the {summary['acquisition_transitions_total']:,} acquisition transitions. Each seed audited the union of selected initial conditions at B=5,000 only after all conditions' acquisition, training and evaluation were frozen. No audit labels entered training. Late events are first violations at steps 21–200 among exactly 20-step safe caps; partial budget cutoffs are excluded. Audit arrays and event timing are preserved."]
    rows = []
    for budget in config['budgets']:
        for strategy in config['strategies']:
            for horizon in config['collection_horizons']:
                name = condition_name(strategy, horizon, budget); r = c[name]
                rows.append([name, f"{r['wide.mean_return']['mean']:.3f}", interval(r['wide.mean_return']),
                    f"{100*r['wide.safety_violation_rate']['mean']:.3f}%", interval(r['wide.safety_violation_rate'], 100),
                    f"{r['audit.all.bounded_action_error.mae']['mean']:.6f}"])
    lines += ['## All budgets', table(['Condition', 'Mean return', '95% seed CI', 'Violation rate', '95% seed CI (%)', 'Action MAE'], rows),
        '## Verification and limitations',
        'Every planned seed must pass before this report is generated. Exact nested budgets, shared-candidate physical prefixes, whole-episode OOB, checkpoint reloads, loaded-policy imitation predictions, full evaluation horizons, raw safety recomputation and safety-audit isolation are checked. Artifact and preservation hashes are validated before aggregation. Pilot reproducibility replays remain preserved; full-study execution does not repeat completed acquisition or training.',
        'Ten seeds provide limited precision. The six conditions share evaluation states, so episodes are not independent seed replicates. These are marginal intervals, with secondary and other-budget comparisons descriptive. Shorter rollouts also change resets, ranking depth, queried-state diversity and trajectory-bootstrap composition; this study cannot isolate redundancy as the sole causal mechanism. Torque saturation suppresses bounded disagreement. The PD expert is not universally safe, BC need not maximize return, and empirical improvements are not safety guarantees. No performance-based tuning, seed exclusion, iterative querying, new experiment, commit or push occurred.']
    text = '\n\n'.join(lines) + '\n'
    path = output / 'full_study_report.md'
    if path.exists(): assert path.read_text() == text
    else: path.write_text(text)


def save_figures(output, summary, config):
    colors = {'uncertainty': 'tab:blue', 'random': 'tab:orange', 'distance': 'tab:green'}
    styles = {20: '-', 200: '--'}
    c = summary['condition_statistics']
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    keys = ['wide.mean_return', 'wide.safety_violation_rate', 'audit.all.bounded_action_error.mae']
    for s in config['strategies']:
        for h in config['collection_horizons']:
            for ax, key, factor in zip(axes, keys, [1, 100, 1]):
                records = [c[condition_name(s, h, b)][key] for b in config['budgets']]
                means = np.array([r['mean'] for r in records])*factor
                ci = np.array([r['bootstrap_95_ci_mean'] for r in records])*factor
                ax.plot(config['budgets'], means, marker='o', ls=styles[h], color=colors[s], label=f'{s} H{h}')
                ax.fill_between(config['budgets'], ci[:, 0], ci[:, 1], color=colors[s], alpha=.10)
    for ax, key, factor, label in zip(axes, keys, [1, 100, 1], ['Mean episode return', 'Safety violation rate [%]', 'Action MAE [N m]']):
        ax.axhline(c['D0'][key]['mean']*factor, color='black', ls=':', label='D0 only')
        ax.set(xlabel='Additional transitions', ylabel=label, xticks=config['budgets'])
        ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Seeds 52–61: means and marginal 95% seed-bootstrap intervals')
    fig.savefig(output / 'controller_outcomes.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, key, factor, label in zip(axes, keys[:2], [1, 100], ['Return difference', 'Violation-rate difference [percentage points]']):
        r = summary['primary'][key]
        ax.scatter(SEEDS, np.array(r['per_seed'])*factor)
        ax.axhline(0, color='black', lw=1)
        ax.axhline(r['mean']*factor, color='tab:orange')
        ax.axhspan(*(np.array(r['bootstrap_95_ci_mean'])*factor), color='tab:orange', alpha=.2)
        ax.set(xlabel='Paired master seed', ylabel=label, xticks=SEEDS)
    fig.suptitle('Primary: uncertainty H20 minus H200 at 5,000 transitions')
    fig.savefig(output / 'primary_paired_differences.png', dpi=180); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5), constrained_layout=True)
    for s in config['strategies']:
        for h in config['collection_horizons']:
            values = [[], [], []]
            for seed in SEEDS:
                target = output / f'seed_{seed}' / 'acquisition' / f'{s}_H{h}'
                values[0].extend(load_npz(target / 'coverage.npz')['pre_action_distances'][:5000])
                q = load_npz(target / 'selected_initial_conditions_5000.npz')
                values[1].extend(q['uncertainty']); values[2].extend(q['uncertainty_rank_1based'])
            for ax, v in zip(axes, values):
                ordered = np.sort(v)
                ax.plot(ordered, np.arange(1,len(v)+1)/len(v), ls=styles[h], color=colors[s], label=f'{s} H{h}')
    for ax, label in zip(axes, ['New-transition D0 distance', 'Queried bounded disagreement [N m]', 'Queried uncertainty rank (1=highest)']):
        ax.set(xlabel=label, ylabel='Pooled descriptive ECDF'); ax.grid(alpha=.2); ax.legend(fontsize=7)
    fig.suptitle('Coverage and queried scores at B=5,000; pooling is descriptive only')
    fig.savefig(output / 'coverage_and_selection.png', dpi=180); plt.close(fig)
