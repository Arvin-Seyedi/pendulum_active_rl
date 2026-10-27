# Experiment 05 — reducing redundant expert-data collection

**Conclusion:** Shorter uncertainty-guided rollouts substantially reduced redundant near-equilibrium data, but the ten-seed study did **not establish an improvement in mean return or safety**, or an uncertainty-specific advantage over random acquisition. The favorable seed-41 pilot did not generalize consistently: the primary return difference improved in five seeds and worsened in five.

## Setup and verification

All prespecified paired seeds **52–61** were included; pilot seed 41 was excluded and preserved. Uncertainty, random and distance acquisition used the same per-seed 10,000-state pool and frozen rankings, at collection horizons 20 and 200. Exact nested additional-transition budgets were **2,000, 5,000 and 10,000**. D0, the PD expert, raw-state safety stopping, five-member trajectory-bootstrap ensemble, architecture, optimizer and whole-episode OOB selection were unchanged. No performance tuning or seed substitution occurred.

Every controller used the same independent 1,000 wide-region and 200 nominal initial conditions, full 200-step evaluation, and 10,000 imitation-audit states. All **21 technical tests, 180 budget datasets, 1,000 checkpoint checks and ten seed completion checks passed**. Independent saved-artifact verification covered **229,200 evaluation episodes**, dynamics/rewards, raw safety margins, 6,000 direct nearest-D0 distance checks, audit event times and bootstrap calculations. All **2,151 protected existing files**, including the pilot, D0 and Experiments 01–04, were unchanged.

## Predefined primary comparison

At **5,000 additional transitions**, uncertainty H20 minus uncertainty H200:

| Outcome | Paired mean difference | Sample SD | Seed range | 95% seed-bootstrap CI |
|---|---:|---:|---:|---:|
| Mean episode return | +8.164 | 78.283 | [−141.157, +129.858] | [−39.914, +52.714] |
| Safety violation rate, percentage points | −0.010 | 0.120 | [−0.300, +0.200] | [−0.090, +0.060] |

Intervals use 10,000 paired seed resamples, seed 505010; SD uses ddof=1. Positive return differences and negative violation-rate differences favor H20. Both primary intervals include zero. These are separate marginal intervals conditional on the fixed evaluation states, not evidence of equivalence or a joint guarantee.

## Controller outcomes at 5,000 transitions

All values average over seeds except the final column, which is the minimum over every seed and evaluation episode. Safety uses all 201 raw unwrapped visited states per episode.

| Strategy | Horizon | Mean return | Violating episodes | Action MAE | Mean seed worst return | Minimum raw margin overall |
|---|---:|---:|---:|---:|---:|---:|
| Uncertainty | 20 | −140.464 | 51.43% | 0.030488 | −715.112 | −5.284736 |
| Uncertainty | 200 | −148.628 | 51.44% | 0.038634 | −881.061 | −5.283281 |
| Random | 20 | −157.524 | 51.40% | 0.001739 | −941.284 | −5.283253 |
| Random | 200 | −188.396 | 51.47% | 0.013449 | −958.589 | −5.283326 |
| Distance | 20 | −151.339 | 51.42% | 0.015673 | −941.199 | −5.283235 |
| Distance | 200 | −107.126 | 51.85% | 0.084115 | −830.422 | −5.301401 |

D0-only mean return was −186.184; the PD reference achieved −159.559 with 51.40% violating episodes. Every condition at every budget had zero nominal-region violations. Uncertainty H20's mean seed worst return improved, but its paired interval included zero: +165.949 [−5.710, +370.865]. Its worst individual episode across seeds was **−1,002.968**, worse than H200's **−962.440**. Its imitation-MAE difference was −0.008146 [−0.018836, +0.003808]. Average improvement therefore did not imply a reliable tail, safety or imitation improvement.

Secondary paired comparisons at the same budget (mean difference [95% CI]):

| Comparison | Mean return difference | Violation-rate difference, percentage points |
|---|---:|---:|
| Random H20−H200 | +30.872 [3.677, 62.061] | −0.070 [−0.150, −0.010] |
| Distance H20−H200 | −44.213 [−62.708, −24.511] | −0.430 [−0.700, −0.180] |
| Uncertainty−random at H20 | +17.060 [−31.891, 57.760] | +0.030 [0.000, 0.090] |
| (U20−U200)−(R20−R200) | −22.707 [−78.501, 33.803] | +0.060 [−0.050, 0.160] |

Thus, random acquisition benefited from H20 at this budget. Distance H20 improved imitation and reduced violations, but worsened return and worst-case performance. The difference-in-differences analysis did not establish an uncertainty-specific benefit. Secondary intervals are descriptive and unadjusted for multiplicity. At 2,000 and 10,000 transitions, the uncertainty mean-return differences were +28.362 [−35.649, 89.720] and +17.001 [−24.539, 54.527], respectively; neither established an improvement.

## Coverage, selection and operational costs

At 5,000 transitions, these are seed means for **newly collected transitions**, not just selected initial states:

| Condition | Mean normalized nearest-D0 distance | Near-equilibrium transitions | Additional occupied bins (.05) | Resets | Collection violations |
|---|---:|---:|---:|---:|---:|
| Uncertainty H20 | 0.12445 | 14.33% | 105.7 | 250.1 | 0.1 |
| Uncertainty H200 | 0.01473 | 89.56% | 32.8 | 25.0 | 0.0 |
| Random H20 | 0.18802 | 12.64% | 684.5 | 330.9 | 146.4 |
| Random H200 | 0.02643 | 86.18% | 230.3 | 44.7 | 20.2 |
| Distance H20 | 0.25808 | 0.00% | 299.5 | 378.3 | 153.7 |
| Distance H200 | 0.02295 | 86.56% | 57.9 | 29.6 | 4.6 |

Normalization is [theta, omega/3]; near-equilibrium radius is 0.05. The uncertainty equal-trajectory near-equilibrium averages were also 14.33% versus 89.56%, supporting the redundancy finding beyond transition weighting. Full distance distributions, tails, per-trajectory and observed-step diagnostics are saved; bin coverage is resolution-dependent.

Uncertainty H20 queried approximately ranks 1–250 versus 1–25. Mean bounded scores declined from 0.34081 (H200) to 0.28684 (H20); mean cutoff scores declined from 0.33075 to 0.24952. **No uncertainty queries had zero bounded disagreement at any budget or seed.** Even at 10,000 transitions, H20 consumed only 10.93% of the positive-disagreement pool on average (maximum 12.14%). Positive-score exhaustion was not observed. Raw scores, other strategy ranks and saturation-stratified errors are retained; random/distance H20 queries had mean zero-disagreement fractions of 50.79%/40.60%.

The extra coverage required roughly ten times as many uncertainty resets. For cost **B + λ×resets**, uncertainty H20 versus H200 costs were **10,002 versus 5,500** at λ=20 and **55,020 versus 10,000** at λ=200. Corresponding random costs were 11,618/5,894 and 71,180/13,940; distance costs were 12,566/5,592 and 80,660/10,920. These are illustrative reset equivalents, excluding training, evaluation and the separate expert audit.

## Separate post-selection safety audit

Among trajectories safely capped at exactly 20 steps, subsequent first violations at steps 21–200 occurred in:

- Uncertainty: **6/2,499 (0.240% pooled)**; mean seed rate 0.241%, CI [0.000%, 0.723%].
- Random: **99/1,836 (5.392% pooled)**; mean seed rate 5.375%, CI [4.273%, 6.420%].
- Distance: **95/2,238 (4.245% pooled)**; mean seed rate 4.252%, CI [3.423%, 5.211%].

Partial budget cutoffs are excluded from these denominators. The audit used **1,115,728 expert transitions and 8,613 resets**, separate from **600,000 acquisition transitions**. It began only after each seed's six conditions were frozen; none of its labels or continuations entered training. A safe 20-step cap therefore did not establish 200-step safety, including for uncertainty acquisition.

## Interpretation and limitations

Reducing redundancy changed the collected data as intended, but did not establish a reproducible uncertainty-guided controller advantage. Random H20 gave substantially lower imitation error, while distance H200 combined better mean return with worse safety and imitation. These results reinforce that imitation fidelity, disagreement, return and safety measure different properties.

Ten seeds leave substantial uncertainty. Shorter horizons also change queried-state diversity, ranking depth, reset cost and trajectory-bootstrap composition, so redundancy cannot be isolated as the sole cause. The unsafe PD teacher and saturated actions limit what BC and bounded disagreement can establish. The study contains no performance tuning, favorable-seed selection, iterative querying, new experiment, commit or push.

Full per-seed values, sample SDs, ranges, confidence intervals, all-budget outcomes, raw/clipped diagnostics and plots are in [the detailed report](../results/rollout_horizon/full_study_52_61/full_study_report.md) and [summary.json](../results/rollout_horizon/full_study_52_61/summary.json). See [independent verification](../results/rollout_horizon/full_study_52_61/independent_verification.json) and [reproducibility instructions](../results/rollout_horizon/full_study_52_61/REPRODUCIBILITY.md).
