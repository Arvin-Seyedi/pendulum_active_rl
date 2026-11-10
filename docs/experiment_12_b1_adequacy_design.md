# B1 learner adequacy — mathematical/data review and prospective design

**DESIGN ONLY; not approved for implementation or training.** No new learner,
MPC query, physics step, action probe, label, RL/IQL or acquisition experiment was
executed. B0's accepted NO-GO, all original criteria and failed E12 gates remain
unchanged. This review follows the completed B0 implementation and separate
decision-log commits. This document is an uncommitted review draft.

## Diagnosis: what the evidence supports

Ridge's advantage is specifically **MAE**, not universal superiority. A static
audit restricted to B0's ten training and two validation trajectories gives:

| Data size | MLP training MSE | Ridge training MSE | MLP validation MSE | Ridge validation MSE |
|---|---:|---:|---:|---:|
| 4 trajectories | 0.000478724 | 0.002243038 | 0.002441692 | 0.003124109 |
| 10 trajectories | 0.000982227 | 0.002682860 | 0.003147953 | 0.003432714 |

These are normalized squared-action units. MLP has lower MSE, its training loss,
while ridge has lower MAE. Thus B0's named ridge-adequacy gate failure is not
proof of optimization failure or insufficient expressive capacity. Different
error allocations/tails, clipped ridge output, regularization and inductive bias
are plausible explanations. Ridge and the tanh/sigmoid MLP are not proven nested
function classes; parameter count alone cannot order their adequacy. B0's original
MAE decision remains valid and is not retroactively replaced with MSE.

The original 5,000 updates expose about 1.28 million sampled records per fit:
roughly 1,067 epochs at four trajectories versus 427 at ten. Equal update counts
are an intentional compute control, not proof of equal convergence. Last logged
gradient norms at size10 range 0.0241–0.0303, but noisy minibatch gradients do not
prove lack of convergence. On size10 training inputs, hidden tanh saturation
(|activation|>0.99) is only 0.63–0.92% /2.17–3.22% in layers 1/2; extensive hidden
saturation is not observed there. Optimization, capacity/history and coverage
remain unresolved; no stronger causal conclusion follows from these diagnostics.

## Input audit and source-supported redundancy

The actual frozen feature builder and native MPC reference lookup are the sources:
[feature builder](../src/quadrotor_archive_data.py),
[fixed model/trainer](../src/quadrotor_archive_learner.py),
[protocol](experiment_12_b0_protocol.md), and pinned upstream `mpc.py:get_references`.

| Columns | Content | Deployment availability / limitations |
|---|---|---|
| 0–5 | Six current native physical observations | Causal measurement; state order x, xdot, z, zdot, theta, thetadot |
| 6–491 | Six reference components ×81 steps | Known planned task, including current reference; not future plant data |
| 492 | k/300 | Causal nonperiodic phase; necessary to retain terminal-preview semantics |
| 493–494 | Previous grouped submitted thrust | Causal command history, not physical sensing; teacher-forced in B0 |
| 495 | Reset flag | Causal; hover is a reset sentinel, not evidence of actual initial application |

Scaling uses fixed constraint midpoints/ranges, not assessment statistics. It is
not whitening or block balancing. MPC-only x/u plans, duals, optimizer guesses,
solver memory, costs and private simulator caches are excluded. A learner that
requires fresh expert plans each step would not be independent MPC imitation.
One previous command is not proven sufficient for teacher or plant history.

Training-only conditioning measurements (3,000 rows) show:

- **162/496 columns are exactly constant**, all in reference preview; 335 exact
  column patterns remain including the zero pattern. This is not a claim that
  every nonconstant preview column is an exact duplicate.
- The preview is identical across starts at each phase. For this one frozen
  reference, integer k determines all 486 entries, including terminal padding.
  Float32 phase uniquely recovers every k=0–299. Preview is therefore functionally
  redundant given the fixed task/phase, even when columns are not linearly identical.
- The centered nonconstant 3,000×334 matrix has numerical rank 307 at the stated
  binary64 roundoff tolerance 7.68e-11, but effective rank **93** at relative
  singular thresholds 1e-6 and 1e-4. Condition number is about **4.49e17**;
  the full centered matrix is singular. Ranks depend on tolerance and float32
  feature quantization; they are not physical model order or independent samples.
- Preview contributes about **96%** of summed centered coordinate variances,
  largely because it occupies 486 dimensions. This can affect initialization,
  gradient geometry and regularization; it does not prove the optimizer ignored
  physical state. Within a fixed regularized ridge solve, condition numbers are
  only about **4,538–4,550**, with backward errors around 1e-16. No ridge numerical
  failure was demonstrated. Removing redundant features also changes effective
  function-space regularization, even if numeric alpha stays unchanged.

Across the twelve training/validation starts, same-phase normalized physical
spread falls from **0.04950** in the first second to **2.882e-6** in the last two
seconds. Normalized action spread across starts falls **0.2010→7.173e-7**.
However late within-trajectory action increments remain about **0.00385** of
range: these are moving-reference trajectories, not equilibrium data. In the first
second, **33.17% /28.83%** of channel labels lie within 1% of an action bound;
the corresponding middle/late fractions are zero. Transient coverage and rare
near-bound behavior differ materially from repeated late reference tracking.

All new conditioning/activation/action statistics excluded B0 assessment IDs
3/7/11/15. Detailed static evidence is in
`results/benchmark_selection/b0_finalization/learner_adequacy_static_review.json`.
No model was updated. These facts support a controlled representation test, not
post-hoc selection of features that win on B0 assessment outcomes.

## Competing hypotheses

1. **Metric/inductive-bias trade-off:** ridge wins absolute error but loses the
   measured training/validation squared-error comparison. Inspect both loss and
   decision metric; do not call this universal underfit or change B0's gate.
2. **Redundant parameterization/conditioning:** hundreds of task-determined
   features alter optimization and regularization. A compact, information-
   preserving fixed-task representation could help; no such result exists yet.
3. **Finite optimization budget:** more updates at fixed data/representation may
   help. It must be tested prospectively, not assumed from nonzero gradients.
4. **Capacity/history limitation:** a one-lag feed-forward map may be inadequate;
   source data do not identify this separately from optimization/inductive bias.
5. **Informative-data limitation:** transient state/history support may remain
   sparse. Better coverage alone did not establish B0's required improvement;
   this hypothesis is not the default explanation after the others fail.

## Proposed minimal B1, requiring separate approval

Keep all B0 data, controller, task, loss, bounds and original results unchanged.
Freeze the B1 protocol before any new learning outcomes or external predictions.

**One compact candidate only:** 16 inputs = current physical state (6), current
reference (6), nonperiodic phase (1), previous submitted command (2), reset (1).
For the single known reference, full preview is reconstructible from phase and
the frozen reference table, so this is not an information-deletion claim for that
task. Preserve the same boundary scaling and command history. Do not replace
phase with a periodic angle that loses terminal-padding information. This
equivalence fails for arbitrary unknown reference families; future varying tasks
would need explicit task/preview information. It is not a Markov-sufficiency proof.

Compare the existing full-input MLP/ridge with compact-input MLP/ridge. Both MLPs
retain two 64-unit tanh layers and sigmoid output. Their parameter counts differ:
**36,098 versus 5,378**. Consequently any compact advantage identifies a practical
parameterization/inductive-bias effect, not pure causal isolation of representation
from capacity. No additional capacity sweep is included; unresolved capacity or
history effects stay unresolved rather than being called data scarcity.

Use the original ten training IDs, original order and nested sizes **4 and 10**;
validation stays [2,10]. The four old assessment IDs are now a reused development
dashboard only, never a new primary test. Original B0 seed results are historical
references, not selectively reused favorable runs. Proposed B1 seeds are fixed
**123000–123004**. Within each representation, pair initial weights and minibatch
uniforms across data sizes. Across representations, input shapes differ; do not
claim identical first-layer initialization. Record corresponding hidden/output
RNG streams and all hashes.

Retain Adam, batch256, normalized MSE, scaling, single-thread CPU and numerical
checks. Train each of twenty MLP runs to a fixed **20,000 updates**, retaining
the preregistered **5,000 and 20,000** checkpoints from the same continuation.
One fixed seed/compact/size4 run repeats through 20,000 updates for determinism:
**420,000 gradient updates total**. Four ridge fits keep alpha1e-3/intercept rules.
Report raw/clipped ridge outputs separately. No loss change to MAE, phase
reweighting, architecture sweep, validation-selected stopping or test-based model
selection. Validation reports fit/generalization diagnostics only. Unlock the
external assessment once after every planned fit and determinism check finishes.
Equal update counts control gradient exposures, not equal FLOPs/CPU across the
two input dimensions. Report actual CPU and parameter counts; do not present the
representation contrast as simultaneous equal-parameter/equal-compute isolation.

## Independent assessment and the zero-solve limitation

The four B0 assessment starts are consumed evidence. Repartitioning their adjacent
transitions or changing only training seeds does not make them untouched again.

Existing **MMAQ H0–H3 MPC80 archives** are possible external label-test candidates:
they have four complete 300-step original trajectories, 6,001 raw states each,
the independently sampled PCG64-120101 provenance and no exact raw-IC duplicates
of B0. B0 fitted only the sixteen other trajectories; no B1 learner predictions
were computed on H0–H3 in this review. However these states were globally reviewed
for expert/model diagnostics; do not call them globally unseen, fresh, or a new
closed-loop validation cohort.

Before accepting them as a prospectively locked learner test, require:

- Proven source/initial-coordinate/reference/task identity, exclusion from all
  prior learner fitting/error-driven selection, and all four cases retained.
- A sealed manifest before new model predictions; no H0–H3 normalization,
  hyperparameter selection or error inspection during B1 development.
- Causal input reconstruction. These older archives **lack direct motor-submission
  streams**, although post-interval RPM is recorded. An inferred previous command
  must be explicitly labeled as inferred, never fabricated direct telemetry.
  A prospective preflight must verify the unchanged source recurrence and reproduce
  all 4,800 B0 float32 feature rows exactly using RPM-based reconstruction before
  applying that method to H0–H3. Initial sentinels, phase and terminal preview must
  match. No tolerance repair or alternative feature is chosen after test errors.

Availability is verified, but this reconstruction/exposure qualification has
**not** been executed in this design review. If it fails, STOP before new B1
training and report the blocker. A development-only variant could be reviewed
separately; it is not an automatic fallback and cannot turn B0's reused assessment
data into independent evidence. A new expert-data/physics allocation is not
automatically authorized to repair this blocker.

If qualified, H0–H3 support only a limited, independently sampled archival
supervised comparison with known prior exposure disclosed. A genuinely fresh
confirmation cohort is still preferable for broader claims. For example, four
new fully qualified starts would require up to 2,400 MPC calls/144,000 physics
steps with paired repeats/both replay types; eight would require 4,800/288,000.
These are **unfunded dependencies, not approvals**, and exceed the remaining
1,870-call allowance. Fresh cohort generation/selection requires separate review.

## Prespecified contrasts, metrics and decisions

Three proposed primary paired contrasts, with no best-model selection:

1. Full MLP20k versus full MLP5k at size10: optimization-budget contrast.
2. Compact versus full MLP at 20k/size10: parameterization contrast.
3. Size10 versus size4 at compact/20k: data-size contrast, fixed whether or not
   compact wins the preceding comparison. It is not a post-hoc selected winner.

Use whole-trajectory macro MAE **and MSE/RMSE**, physical N and normalized units,
per-start and phase errors, p95/max, near-bound errors, train/validation/external
gaps, optimizer variability, finite-value and conditioning diagnostics. Report
worst-trajectory behavior and all seeds. No closed-loop return or safety metric
exists in this archive-only study. Cluster uncertainty by training seed and
external initial trajectory (PCG64-123999, 20,000 crossed draws), never transitions.
For the three planned contrasts report nominal Bonferroni-adjusted 98.33% bootstrap
intervals alongside descriptive 95% intervals; with four clusters this is not a
certified simultaneous-coverage guarantee. Four external starts remain
low-power; wide intervals mean unresolved, not equivalence or proof of no effect.

Prospective operational criteria, separate from unchanged B0:

- A practical optimization/parameterization benefit needs >=25% relative and
  >=0.001 absolute normalized external MAE reduction, positive adjusted paired
  interval, and no start's mean deterioration >0.005. Optimization attribution
  additionally needs lower full-training MSE at fixed data/representation.
  If MSE improves but MAE does not, classify a metric/generalization trade-off,
  not a positive acquisition finding. Parameterization/capacity confounding stays
  explicit even if compact wins.
- Reconsider a reducible data gap only if the fixed adequate learner's training
  MAE is <=0.005, external MAE remains >=0.01, it is within 10% of ridge MAE,
  and the preregistered 4→10 data contrast meets >=25% reduction with positive
  adjusted interval. Examine phase-conditioned coverage/error relationships as
  supporting diagnostics, not independent-transition significance or a causal
  proof that an acquisition ranking will work.
- If errors fall below the practical floor, do not collect data merely to claim
  a benefit. If fit remains poor, capacity/history/optimization are unresolved;
  do not infer missing data by elimination. If no contrast is robust, retain
  NO-GO/inconclusive status. No post-hoc changes or extra fitting arms.

Even a positive B1 result would justify only reviewing an equal-budget acquisition
design. That later study needs frozen candidate contexts/rankings based on causal
information, paired seeds/compute, identical added-transition budgets and failures
accounted without future rewards/margins selecting favorable cases. It also needs
fresh independent learned-policy rollouts with all raw physics states, actual
inputs, contacts, incomplete episodes, mean/worst return/effort and safety outcomes,
expert comparisons, deterministic repeats/replays and the declared mismatch
diagnostics. None of those executions is approved here. Beyond-expert RL retains
the separate safe-action support and meaningful-headroom prerequisites.

## Smallest bounded resource proposal and stop

One worker, pinned environment, 4 GiB, **zero new MPC calls/physics steps**.
Twenty MLP fits with two fixed checkpoints, one 20k repeat, four ridge systems,
external/development predictions and independent checks are bounded at
**600 additional CPU /1,200 additional active-wall seconds**, including a
**60 CPU/120 wall-second preflight subcap**. B0 used 55.519 CPU seconds for
105,000 updates plus persistence; 420,000 full-cost updates extrapolate to about
222 CPU seconds before data checks/audits. The finite cap leaves conservative
overhead, not a guarantee of completion. Stop rather than shortening schedules,
dropping seeds or extending the allocation.

This budget is a proposal, not authorization to spend B0's unused time. Source,
runtime, split and external-input preflight must pass before training; otherwise
stop. No new labels, calls, physical steps or acquisition allocation is requested
or granted by the design. Preserve cumulative historical charges. Recommendation:
**GO for scientific review of this limited adequacy protocol; NO-GO for execution
until separately approved, and continued NO-GO for acquisition on B0 evidence.**
