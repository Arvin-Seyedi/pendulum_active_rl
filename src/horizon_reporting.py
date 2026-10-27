"""Descriptive pilot reports and scientific plots; no selection or fitting."""

import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from src.horizon_experiment import condition_name


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                      '| ' + ' | '.join(['---'] * len(headers)) + ' |',
                      *['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]])


def save_pilot_report(folder, config, metrics, diagnostics, safety, contrasts, costs):
    strategies, horizons, budgets = config['strategies'], config['collection_horizons'], config['budgets']
    colors = {'uncertainty': 'tab:blue', 'random': 'tab:orange', 'distance': 'tab:green'}
    styles = {20: '-', 200: '--'}
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for strategy in strategies:
        for horizon in horizons:
            records = [metrics[condition_name(strategy, horizon, b)] for b in budgets]
            ys = [[r['wide']['mean_return'] for r in records],
                  [100 * r['wide']['safety_violation_rate'] for r in records],
                  [r['audit']['all']['bounded_action_error']['mae'] for r in records]]
            for ax, y in zip(axes, ys):
                ax.plot(budgets, y, marker='o', linestyle=styles[horizon], color=colors[strategy], label=f'{strategy}, H={horizon}')
    for ax, baseline, label in zip(axes,
            [metrics['D0']['wide']['mean_return'], 100 * metrics['D0']['wide']['safety_violation_rate'], metrics['D0']['audit']['all']['bounded_action_error']['mae']],
            ['Mean episode return', 'Episodes violating safety [%]', 'Held-out action MAE [N m]']):
        ax.axhline(baseline, color='black', linestyle=':', label='D0 only')
        ax.set(xlabel='Additional training transitions', ylabel=label, xticks=budgets)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle('Seed-41 technical pilot only: descriptive outcomes, no confidence intervals')
    fig.savefig(folder / 'pilot_controller_outcomes.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for strategy in strategies:
        for horizon in horizons:
            target = folder / 'acquisition' / f'{strategy}_H{horizon}'
            with np.load(target / 'coverage.npz', allow_pickle=False) as a:
                distance = np.sort(a['pre_action_distances'][:5000])
            with np.load(target / 'selected_initial_conditions_5000.npz', allow_pickle=False) as a:
                uncertainty = a['uncertainty'].copy()
                rank = np.sort(a['uncertainty_rank_1based'])
            label = f'{strategy}, H={horizon}'
            axes[0].plot(distance, np.arange(1, len(distance) + 1) / len(distance), styles[horizon], color=colors[strategy], label=label)
            axes[1].plot(np.arange(1, len(uncertainty) + 1), uncertainty, styles[horizon], color=colors[strategy], label=label, alpha=.8)
            axes[2].plot(rank, np.arange(1, len(rank) + 1) / len(rank), styles[horizon], color=colors[strategy], label=label)
    axes[0].set(xlabel='New-transition nearest-D0 distance', ylabel='Transition ECDF')
    axes[1].set(xlabel='Queried initial condition in acquisition order', ylabel='Bounded disagreement [N m]')
    axes[2].set(xlabel='Uncertainty rank in common pool (1=highest)', ylabel='Queried-initial-condition ECDF')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle('Seed 41, B=5,000: coverage and queried-candidate scores; no outcome-based selection')
    fig.savefig(folder / 'pilot_coverage_and_selection.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    names = [f'{s}\nH{h}' for s in strategies for h in horizons]
    near = [100 * diagnostics[condition_name(s, h, 5000)]['coverage']['transition_fraction_near_equilibrium']['0.05'] for s in strategies for h in horizons]
    axes[0].bar(names, near)
    axes[0].set(ylabel='New transitions near equilibrium [%]', ylim=(0, 100))
    rates = []
    labels = []
    for strategy in strategies:
        r = safety[condition_name(strategy, 20, 5000)]
        rate = r['subsequent_violation_fraction_among_safely_capped_20']
        rates.append(100 * rate if rate is not None else np.nan)
        labels.append(f"{r['subsequent_violation_steps_21_to_200_count']}/{r['safely_capped_at_20_count']}")
    bars = axes[1].bar(strategies, rates)
    axes[1].bar_label(bars, labels=labels, padding=3)
    maximum_rate = max((r for r in rates if np.isfinite(r)), default=0)
    axes[1].set(ylabel='Later expert violations among safe H20 caps [%]', ylim=(0, max(5, maximum_rate * 1.3)))
    fig.suptitle('B=5,000: redundancy and separate post-selection safety audit (counts on bars)')
    fig.savefig(folder / 'pilot_redundancy_and_late_safety.png', dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(6, 3, figsize=(13, 18), constrained_layout=True)
    for row, (strategy, horizon) in enumerate((s, h) for s in strategies for h in horizons):
        for col, budget in enumerate(budgets):
            name = condition_name(strategy, horizon, budget)
            histories = json.loads((folder / 'training' / name / 'histories.json').read_text())
            for record in histories:
                axes[row, col].semilogy(record['history']['epoch'], record['history']['oob_mse'], alpha=.7)
            axes[row, col].set(title=name, xlabel='Epoch', ylabel='OOB normalized MSE')
    fig.suptitle('All five member training histories; unchanged hyperparameters')
    fig.savefig(folder / 'pilot_training_histories.png', dpi=140)
    plt.close(fig)

    primary = contrasts['5000'][f"{condition_name('uncertainty', 20, 5000)}_minus_{condition_name('uncertainty', 200, 5000)}"]
    lines = [
        '# Experiment 05: seed-41 technical pilot',
        'This pilot tests implementation and reproducibility only. It is excluded from the proposed seeds 52–61 study. No method or hyperparameter tuning is permitted; single-seed outcomes are descriptive, without confidence intervals.',
        '## Verification scope',
        'See `checks.json` for final pass status. Checks cover 18 exact budget datasets; independently replayed collection prefixes; shared-candidate physical trajectories; raw safety and cap flags; 100 checkpoint reloads; whole-episode OOB; full-configuration member replay in six primary-budget conditions; bitwise agreement of the H200 condition with the accepted Experiment 04 pilot; full 200-step evaluation; score/rank diagnostics; and isolated expert-audit replay. Existing sources, D0, and prior results are protected by hashes.',
        '## Descriptive primary comparison at 5,000 transitions',
        f"Uncertainty H20 minus H200: mean return difference **{primary['wide.mean_return']:+.6f}**; violation-rate difference **{100 * primary['wide.safety_violation_rate']:+.3f} percentage points**. These are not multi-seed findings.",
    ]
    rows = []
    for name in ['PD', 'D0', *[condition_name(s, h, b) for b in budgets for s in strategies for h in horizons]]:
        r = metrics[name]
        rows.append([name, f"{r['wide']['mean_return']:.6f}", f"{100*r['wide']['safety_violation_rate']:.3f}%",
                     f"{r['wide']['worst_return']:.6f}", f"{r['wide']['minimum_raw_safety_margin']:.6f}",
                     '—' if name == 'PD' else f"{r['audit']['all']['bounded_action_error']['mae']:.6f}"])
    lines.extend(['## Full-horizon controller outcomes', table(['Condition', 'Mean return', 'Violating episodes', 'Worst return', 'Minimum raw margin', 'Action MAE'], rows)])
    rows = []
    for strategy in strategies:
        for horizon in horizons:
            name = condition_name(strategy, horizon, 5000)
            r = diagnostics[name]
            rows.append([name, r['operations']['resets'], r['operations']['collection_safety_crossings'],
                         f"{r['coverage']['tail']['distribution']['mean']:.6f}", f"{r['coverage']['tail']['distribution']['p95']:.6f}",
                         f"{100*r['coverage']['transition_fraction_near_equilibrium']['0.05']:.3f}%",
                         f"{100*r['coverage']['transition_fraction_near_D0']['0.01']:.3f}%",
                         r['coverage']['bin_coverage']['0.05']['additional_occupied_bins']])
    lines.extend(['## Acquisition at 5,000 transitions', table(['Condition', 'Resets', 'Safety stops', 'Mean D0 distance', 'p95 distance', 'Near equilibrium', 'Near D0', 'New bins (.05)'], rows),
                  'Near equilibrium uses raw normalized radius 0.05; near D0 uses distance 0.01. Bin occupancy is resolution-dependent. Later-step values for H20 are unobserved, not zero. Full distributions, per-trajectory records, and per-step counts are saved for every budget.'])
    rows = []
    for strategy in strategies:
        for horizon in horizons:
            name = condition_name(strategy, horizon, 5000)
            r = diagnostics[name]['selection']
            ranks = r['rank_positions_1based']['uncertainty']
            rows.append([name, r['queried_initial_conditions'], f"{r['bounded_disagreement']['mean']:.6f}",
                         f"{r['bounded_disagreement']['minimum']:.6f}", f"{r['bounded_disagreement']['p95']:.6f}",
                         f"{100*r['fraction_zero_bounded_disagreement']:.3f}%", f"{ranks['median']:.1f}", int(ranks['maximum'])])
    lines.extend(['## Queried initial-condition scores at 5,000 transitions', table(['Condition', 'Queries', 'Mean bounded std', 'Minimum', 'p95', 'Exactly zero', 'Median U-rank', 'Maximum U-rank'], rows),
                  'Ranks are 1-based in the common pool. Every selected initial condition contributes one observation, independently of its trajectory length. Saved arrays retain query order, all strategy ranks, support distance, bounded/raw disagreement, and remaining positive-score coverage. Random selection has no numerical acquisition score.'])
    rows = []
    for strategy in strategies:
        name = condition_name(strategy, 20, 5000)
        r = safety[name]
        rate = r['subsequent_violation_fraction_among_safely_capped_20']
        rows.append([strategy, r['safely_capped_at_20_count'], r['subsequent_violation_steps_21_to_200_count'], 'N/A' if rate is None else f'{100*rate:.3f}%', r['expert_violations_by_step_200']])
    lines.extend(['## Separate expert safety audit', table(['H20 strategy', 'Safe caps at step 20', 'First violation at steps 21–200', 'Late fraction among safe caps', 'All selected ICs violating by 200'], rows),
                  'The audit ran only after all acquisition, training, checkpoint selection, and policy evaluation were frozen. It audited the union of selected initial states once through first violation or step 200. No continuation labels entered training. Its replay and costs are recorded separately. A cap at step 20 does not imply safety through step 200; partial budget cutoffs are excluded from the safe-cap denominator.',
                  '## Cost accounting',
                  f"Main acquisition: {costs['acquisition_transitions']:,} transitions. Independent technical collection replays: {costs['technical_collection_replay_transitions']:,}. Post-selection expert safety audit: {costs['post_selection_safety_audit_transitions']:,} transitions and {costs['post_selection_safety_audit_resets']:,} resets; its technical replay adds {costs['technical_safety_audit_replay_transitions']:,} transitions. Policy evaluation and its replay are separately recorded in `cost_accounting.json`. Unit-test simulations are not acquisition data. Equal-transition budgets do not imply equal reset costs; per-budget reset-cost sensitivity uses 0, 20, and 200 transition-equivalents per reset.",
                  '## Limits',
                  'Only seed 41 was executed. No inferential conclusion about shorter-rollout superiority follows from this pilot. Shorter horizons change ranking depth, initial-condition diversity, trajectory-bootstrap composition, and observation exposure as well as equilibrium redundancy. Saturation can suppress bounded disagreement. Raw/clipped and saturation-stratified imitation diagnostics are preserved. The full ten-seed study has not been launched; nothing has been committed or pushed.'])
    (folder / 'pilot_report.md').write_text('\n\n'.join(lines) + '\n')
