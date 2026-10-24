# Experiment 04: Active Expert-Data Acquisition

All ten paired seeds (42–51) completed and passed verification. The seed-41
technical pilot is excluded from inference. **Bounded-disagreement acquisition
did not demonstrate a reliable improvement over random acquisition at the
predefined 5,000-transition budget.** Distance acquisition improved returns in
secondary comparisons, with safety tradeoffs. No method or hyperparameter was
tuned using the pilot or evaluation results.

## Setup and reproducibility

- Pendulum-v1; unchanged PD expert `clip(-8 theta - 2 omega, -2, 2)`; unchanged
  D0 containing 100 trajectories and 20,000 transitions.
- Each master seed supplies a common pool of 10,000 initial conditions uniform
  in `theta ∈ [-0.8, 0.8]`, `omega ∈ [-2, 2]`. Rankings remain fixed: descending
  bounded ensemble disagreement, descending nearest-D0 distance, or a uniform
  random permutation. Selection has no expert labels or future safety outcomes.
- Equal nested budgets of **2,000, 5,000, and 10,000 additional transitions**.
  Each strategy collects one 10,000-transition stream per seed; smaller budgets
  are exact prefixes. Total executed expert collection: 300,000 transitions.
- Collection stops at the first negative raw successor-state margin, retaining
  and counting the crossing transition, or at 200 steps. The next ranked
  candidate supplies the remaining budget. Technical failures abort collection;
  there were none. Budget cutoffs are distinct from environment termination.
- Retrain five independently initialized 2–64–64–1 ReLU networks on D0 plus each
  prefix, using whole-trajectory bootstrap and out-of-bag checkpoint selection.
  Adam learning rate 0.001, batch size 256, maximum 300 epochs, patience 30;
  configurations and member seed streams are paired across conditions. Deploy
  the mean of five individually clipped outputs; preserve raw predictions.
- Shared independent evaluation: 1,000 wide-domain and 200 nominal initial
  conditions, 200 steps each; 10,000 independent states for PD-action imitation
  error. Evaluation continues after violations. Safety uses all 201 raw visited
  states and `min(1 - |theta|, 1 - |omega|/3)` without angle wrapping or clipping.

The archived [reproducibility configuration](../configs/experiment_04.json)
records the exact protocol, software versions, source hashes, D0 hash, and seed
configuration. It documents the source-defined configuration; it is not a new
runtime override. The study used deterministic, single-thread CPU PyTorch and
three worker processes. Verification covered 550 checkpoints, 90 budget
prefixes, 120,000 learned-policy evaluation episodes, and 24,120,000 raw states.
Ten technical tests passed, and independent training replay checks matched
saved parameters and histories bitwise for the checked members.

## Primary comparison: uncertainty minus random, 5,000 transitions

The two primary outcomes are separate. Intervals are 95% percentile intervals
from 10,000 paired seed-bootstrap resamples (seed 404010). SD is the sample SD
of the ten paired differences; ranges are across those differences.

| Outcome | Paired mean | Sample SD | Range | 95% CI for mean |
|---|---:|---:|---:|---:|
| Mean episode return | +15.6436 | 86.9953 | [−198.8566, +94.3137] | [−42.1756, +59.1252] |
| Episode violation rate, percentage points | −0.2000 | 0.5312 | [−1.7000, +0.1000] | [−0.5500, 0.0000] |

Return favored uncertainty in 7/10 seeds, but neither primary interval excludes
zero. Seed 49 is retained: its return difference was −198.8566 and its violation
rate difference was −1.7 percentage points, accounting for most of the average
safety reduction. The study does not establish equivalence between methods.

## Outcomes across all ten seeds

Entries are seed averages. Worst return and minimum margin are computed within
each seed's 1,000 wide-domain episodes and then averaged. Action MAE compares
the deployed bounded torque with the clipped expert on the independent audit.

| Controller | Added transitions | Mean return | Violating episodes | Mean seed-worst return | Mean seed-minimum raw margin | Action MAE |
|---|---:|---:|---:|---:|---:|---:|
| PD reference | — | −151.978 | 48.40% | −968.776 | −5.283185 | — |
| D0 only | 0 | −158.594 | 48.43% | −963.737 | −5.283222 | 0.050815 |
| Uncertainty | 2,000 | −158.288 | 48.43% | −944.710 | −5.283221 | 0.038850 |
| Distance | 2,000 | −126.004 | 48.62% | −810.825 | −5.284423 | 0.061264 |
| Random | 2,000 | −219.156 | 49.10% | −975.469 | −5.283363 | 0.057038 |
| Uncertainty | 5,000 | −183.633 | 48.46% | −972.573 | −5.283240 | 0.039027 |
| Distance | 5,000 | −103.584 | 49.05% | −789.213 | −5.291089 | 0.072252 |
| Random | 5,000 | −199.276 | 48.66% | −970.836 | −5.283270 | 0.020208 |
| Uncertainty | 10,000 | −162.379 | 48.47% | −824.822 | −5.283226 | 0.037413 |
| Distance | 10,000 | −115.275 | 48.60% | −890.338 | −5.283233 | 0.040366 |
| Random | 10,000 | −166.907 | 48.43% | −957.657 | −5.283216 | 0.004859 |

At 5,000 transitions, worst observed returns across all seeds were −1135.922
(uncertainty), −957.868 (distance), and −1014.868 (random); absolute minimum raw
margins were −5.283318, −5.352377, and −5.283619, respectively. All learned
controllers had zero violations on the 200 nominal conditions at every budget.
The PD reference is one shared deterministic evaluation, not ten independent
expert runs; it is not universally safe.

Secondary findings (95% paired seed-bootstrap intervals; no multiplicity
adjustment):

- At 5,000, distance improved return versus random by **+95.692**
  `[68.443, 121.135]` and versus D0 by **+55.010** `[37.697, 72.916]`. It improved
  mean seed-worst return versus random by +181.623 `[87.910, 308.973]`, but
  increased violation rate versus D0 by **+0.62 percentage points** `[0.15, 1.49]`.
  Its violation-rate difference versus random was +0.39 points `[−0.41, 1.44]`.
  Return contrasts against random and D0 favored distance at all three budgets.
- Uncertainty beat random in return at 2,000 by +60.869 `[24.757, 95.465]`, but
  its difference from D0 was only +0.306 `[−22.366, 22.442]`; this relative
  advantage largely reflects random degradation. At 10,000, uncertainty minus
  random was +4.527 `[−55.357, 54.258]`.
- At 5,000, uncertainty improved action MAE versus D0 by −0.011788
  `[−0.020062, −0.002703]`, but was worse than random by +0.018819
  `[0.003308, 0.031492]`. Its mean seed-worst-return difference versus random was
  −1.738 `[−49.202, 34.486]`. Imitation-error improvement did not establish
  improved closed-loop return or safety.

## Collection coverage and redundancy

At 5,000 transitions, the following are means across seeds. Distance is exact
nearest-D0 Euclidean distance in coordinates `[theta, omega/3]` among newly
collected pre-action states. P95 is averaged across per-seed percentiles.
Near equilibrium means normalized raw-state radius ≤0.05; added bins have
normalized width 0.05 and fixed origin.

| Strategy | Resets | Safety crossings | Mean distance | Mean p95 distance | Near equilibrium | Added occupied bins |
|---|---:|---:|---:|---:|---:|---:|
| Uncertainty | 25.0 | 0.0 | 0.012734 | 0.028458 | 90.77% | 30.8 |
| Distance | 31.8 | 6.8 | 0.023547 | 0.133809 | 86.37% | 61.2 |
| Random | 47.5 | 23.3 | 0.027763 | 0.214226 | 86.11% | 240.0 |

Uncertainty selected distant initial states (mean distance 0.62239), yet
**94.93%** of new transitions were within distance 0.01 of D0. After the first
20 steps, **98.58%** were near equilibrium, with mean distance **0.00004387**.
Every uncertainty rollout lasted 200 steps without a collection violation.
This demonstrates substantial redundant near-equilibrium data, but does not
prove redundancy alone caused the controller outcomes. Random produced broader
coverage at the cost of more resets and safety stops.

The pilot's mean distance 0.0065172 exceeding p95 0.0047242 was verified: the
largest 5% of distances contributed **97.93%** of the distance sum. Independent
distance calculations agreed within 1.11e−16; this was a right-tail effect.

## Scientific limitations

Ten seeds limit precision. Intervals describe seed variation conditional on the
fixed evaluation set, D0, model, and environment, not test-population uncertainty.
Primary intervals are marginal; secondary comparisons are descriptive. Equal
transition budgets imply unequal reset costs and trajectory counts; training
weights long trajectories more heavily despite whole-trajectory bootstrapping.
Coverage bins are resolution-dependent, and late-step diagnostics condition on
observed surviving trajectories. Evaluation continues after violations while
collection stops, so post-violation behavior is largely extrapolation. Raw
unwrapped safety margins are not directly comparable with older wrapped-angle
or pre-action-only benchmarks. Torque saturation can suppress disagreement;
raw/clipped predictions and saturation-stratified errors are preserved locally.
There is no formal safety guarantee and no demonstrated primary-outcome benefit
from bounded-disagreement acquisition. Iterative querying and offline RL were
not implemented.

## Reproduction and artifact policy

Run technical checks with `python3 -m unittest discover -s tests -v`. On a fresh
checkout, use the recorded software versions and the verified D0 hash. Generate
D0 with the unchanged Experiment 02 only if it is absent; then run
`experiments/04_active_acquisition.py --pilot` followed by
`experiments/04_full_study.py --workers 3` using `python3`. The full-study runner
requires the completed pilot at its default location. Existing run directories
have frozen source/Git metadata: do not overwrite them or resume them across
revisions. The reviewed study was not rerun for finalization.

Full local evidence remains under
`results/active_acquisition/full_study_42_51/`: protocol, per-seed metrics,
bootstrap indices, source snapshots, candidate rankings, training histories,
raw/clipped predictions, trajectory diagnostics, and verification records.
Generated datasets, checkpoints, evaluation trajectories, plots, and large
result files are intentionally excluded from this commit.
