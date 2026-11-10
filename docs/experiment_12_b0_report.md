# Experiment 12 B0 — archive-only fixed-learner gap assessment

**Recommendation: NO-GO for designing new targeted-versus-random acquisition on
the present B0 evidence.** A measurable supervised gap exists, and very small
training sets benefit from more archived demonstrations. However, the frozen
primary 4→10-trajectory comparison fails both the 25% reduction and uncertainty
criteria, and the largest-data neural learner is worse than a simple ridge model.
No architecture, threshold, partition or schedule was tuned after learning began.

## Frozen data, inputs and learner

Use the accepted sixteen original MPC80 trajectories: **4,800 labels** at 50 Hz,
one six-second figure-eight reference, no disturbances/parameter randomization.
The 4,800 deterministic-repeat labels were verified but excluded as samples.
All 4,800 native observation/reference/action records, native submitted inputs,
raw resets versus exact requested ICs and provenance were checked. Pilot B's
alternative data were not added. Source, Python 3.10.19, Torch 2.8.0, dependency,
native binary and upstream identities passed before training. Torch's CPU library
also matched its installed package RECORD.

Inputs are 496 causal scalars: six current physical observations, known 6×81
reference preview with terminal padding, k/300, previous directly submitted grouped
thrust and reset flag. At reset the previous-input slot is a known hover sentinel,
not a claim that hover thrust was applied. Preview is the available task reference,
not future plant observations. No rewards, successor states, current target action,
MPC plans, duals, guesses, optimizer memory or hidden simulator caches enter the
learner. Previous submitted thrust is command telemetry, not physical sensing.
Teacher-forced action history in this assessment differs from a learner's own
history in autonomous execution. One lag is not proven sufficient for teacher or
plant memory; residual errors can involve representation as well as coverage.

The approved instruction to freeze a deliberately simple learner was implemented
as a new feed-forward B0 version, rather than the earlier unimplemented GRU draft:
**496–64–64–2**, tanh hidden layers, sigmoid normalized requested-thrust output.
State/reference/previous-action scaling uses fixed archived bounds only. Target
is native requested thrust normalized by the fixed channel range
**0.24035951867699623 N**. Float32, deterministic single-thread CPU, Adam 1e-3,
batch256, normalized MSE, exactly 5,000 updates per fit, last checkpoint only.
Five seeds **122000–122004**, identical initialization and underlying minibatch
uniforms across subsets; no early stopping or validation-based selection.

Training order: **[0,4,8,12,5,6,13,14,9,1]**; validation **[2,10]**; assessment
**[3,7,11,15]**. Nested training prefixes contain 2/4/8/10 trajectories
(600/1,200/2,400/3,000 labels). No trajectory crosses partitions. The fixed order
uses the original four probe IDs first and PCG64-122101 for the remainder. The
assessment partition was not used for fitting, normalization or model selection;
it was unlocked only after all twenty fits and the prescribed reproducibility
repeat passed. It is still a partition of previously reviewed scientific data,
not a fresh independent policy-evaluation cohort.

Comparator: ridge regression on identical features/targets, fixed regularization
1e-3, unpenalized intercept, float64 solve. Raw predictions/clipping are retained;
clipped [0,1] predictions are the prespecified primary baseline. Four deterministic
fits, not five fictitiously independent ridge seeds. No baseline tuning.

## Learning curves and uncertainty

MAE is averaged over channels/time within each complete trajectory, then equally
over trajectories and seeds. Percentages use the full action range. Intervals
are 20,000 crossed seed/assessment-trajectory bootstrap draws, PCG64-122999;
adjacent transitions are never resampled as independent observations. With only
four fixed assessment starts, these are limited conditional uncertainty estimates,
not population safety/generalization intervals. Seeds measure optimizer variation,
not five independent data-subset draws. Only one frozen nested subset order was used.

| Training trajectories /labels | Train MAE % | Validation MAE % | Assessment MAE % [95% CI] | Assessment MAE mN | Ridge assessment MAE % |
|---|---:|---:|---:|---:|---:|
| 2 /600 | 0.8307 | 3.1156 | 2.6179 [1.7681, 3.4510] | 6.2923 | 1.4785 |
| 4 /1200 | 1.1314 | 1.7567 | 1.6129 [1.2356, 1.9907] | 3.8767 | 1.1678 |
| 8 /2400 | 1.5988 | 1.9903 | 1.5906 [1.3101, 1.8634] | 3.8231 | 1.1747 |
| 10 /3000 | 1.4247 | 1.8077 | 1.4592 [1.2244, 1.7054] | 3.5073 | 1.1450 |

Primary **base4 minus size10 error reduction**: **0.001536739** normalized (0.369370 mN), 95% CI **[-0.001220991, 0.004092489]** normalized ([-0.293477, 0.983669] mN). Relative mean reduction **9.5279%**, below the frozen 25%. Paired seed-mean sample SD 0.002458142 normalized; range [-0.002706481, 0.003681301]. At least one seed worsened. All five seeds were retained.

| Seed | Base4 minus size10 MAE reduction, mN |
|---|---:|
| 122000 | +0.518520 |
| 122001 | +0.496508 |
| 122002 | -0.650528 |
| 122003 | +0.597514 |
| 122004 | +0.884836 |

Positive descriptive finding: 2→10 trajectories reduced mean assessment MAE by **44.2596%**. This does not replace the preregistered primary 4→10 comparison. The largest MLP MAE is **1.2744× ridge** (ridge 2.752139 mN); the <=1.10 adequacy gate fails. Ridge's own 4→10 point improvement is only 1.949%.

## Upper tails and per-trajectory results

RMSE and p95 below are means of per-trajectory statistics over the five seeds; they are not transition-independent confidence statements. Worst-trajectory mean averages its five seeds; largest single action error takes the maximum over seeds, trajectories, times and channels.

| Training trajectories | Macro RMSE % | Mean trajectory p95 % | Worst-trajectory mean MAE % | Largest single action error N |
|---|---:|---:|---:|---:|
| 2 | 9.8029 | 11.3942 | 3.5061 | 0.240341 |
| 4 | 4.1897 | 4.2654 | 1.9111 | 0.158687 |
| 8 | 3.9169 | 3.6216 | 1.7357 | 0.165223 |
| 10 | 3.7083 | 3.3450 | 1.5998 | 0.148714 |

| Assessment ID | Base4 MAE % | Size10 MAE % | Size10 MAE mN | Size10 seed SD % | Size10 mean p95 mN | Size10 maximum point error N |
|---|---:|---:|---:|---:|---:|---:|
| 3 | 1.6142 | 1.5246 | 3.6644 | 0.2566 | 7.9826 | 0.119919 |
| 7 | 1.1357 | 1.2611 | 3.0311 | 0.2869 | 6.4333 | 0.119614 |
| 11 | 1.9111 | 1.5998 | 3.8454 | 0.2589 | 9.0075 | 0.148714 |
| 15 | 1.7904 | 1.4513 | 3.4884 | 0.2791 | 8.7363 | 0.066183 |

The largest-data worst point is **0.148713966 N** at ID11, step2 (0.04 s), channel0, seed122004: expert request 0.0590077681 N versus prediction 0.207721734 N. Mean errors hide substantial early transients. ID7 mean error worsened with more data, within the frozen 0.005 normalized deterioration allowance. No state-safety inference follows from any action-error statistic.

## Phase-conditioned error and coverage

| Assessment ID | Phase | Base4 MAE % /mN | Size10 MAE % /mN |
|---|---|---:|---:|
| 3 | early | 5.2797 /12.6902 | 4.7229 /11.3520 |
| 3 | middle | 0.6894 /1.6570 | 0.8395 /2.0179 |
| 3 | late | 1.1688 /2.8094 | 0.9529 /2.2905 |
| 7 | early | 2.7732 /6.6657 | 3.1787 /7.6403 |
| 7 | middle | 0.5681 /1.3654 | 0.8273 /1.9886 |
| 7 | late | 1.1685 /2.8086 | 0.9529 /2.2903 |
| 11 | early | 6.9449 /16.6927 | 5.0883 /12.2301 |
| 11 | middle | 0.7279 /1.7495 | 0.8683 /2.0871 |
| 11 | late | 1.1690 /2.8098 | 0.9530 /2.2905 |
| 15 | early | 5.5797 /13.4112 | 3.5971 /8.6460 |
| 15 | middle | 0.9413 /2.2626 | 1.0683 /2.5676 |
| 15 | late | 1.1695 /2.8110 | 0.9530 /2.2907 |

Early = steps0–49 (0–1 s), middle =50–199 (1–4 s), late =200–299 (4–6 s). Aggregate base4→size10 phase MAEs are early **5.1444→4.1467%**, middle **0.7317→0.9009%**, late **1.1690→0.9529%**. Reported improvements are not uniform across phases. Complete normalized/physical channel, RMSE, p95, max and per-seed metrics for every trajectory/size are in `assessment.json`.

Coverage uses same-phase nearest-training physical-state RMS distances normalized by the fixed state ranges; it is a partial context descriptor, not a safety/viability distance. Assessment early mean distance falls **0.0264019→0.0210226→0.0131833→0.0131833** across the four sizes. Late mean distance falls from **1.18976e-6** at sizes2/4 to **4.22564e-7** at sizes8/10. Data are far more redundant late in the reference. Distance/error Spearman correlations are **0.3830, 0.1114, −0.1197, 0.2722**, descriptive only, without transition-level p-values. Improved state coverage did not establish the required primary error reduction or an acquisition advantage.

## Frozen decision and interpretation

| Criterion | Outcome |
|---|---|
| meaningful_base_gap | PASS |
| at_least_25_percent_reduction | FAIL |
| positive_paired_improvement_CI | FAIL |
| no_large_trajectory_deterioration | PASS |
| early_phase_does_not_worsen | PASS |
| not_clear_underfit_relative_to_ridge | FAIL |

The operational meaningful-gap threshold (MAE >=1% of channel range, bootstrap lower bound >=0.5%) passes. These scales are about 2.404/1.202 mN, anchored to the earlier 5%-range action menu, not validated physical safety tolerances or a general small-signal control-sensitivity model. A single safe Pilot B perturbation does not calibrate all states.

There is a measurable supervised prediction gap and a reducible low-data component. Nevertheless, the primary 4→10 reduction is only 9.53%, its interval includes zero, and ridge beats the frozen neural learner. More expert collection is not yet justified under the sealed gate. Fixed compute gives smaller datasets more epochs; no convergence guarantee or architecture/history sufficiency was established. Approximation, optimization schedule, omitted teacher history and reference representation remain possible contributors. This study did not isolate their causes and does not prove that targeted acquisition or another learner cannot help. Do not retune B0 or move the primary endpoint after these results. Review learner/baseline adequacy before proposing a new acquisition experiment.

One reference family, sixteen reviewed starts, one fixed data order, four assessment starts and five optimizer seeds constrain inference. Assessment error is measured on expert-visited states with teacher-forced previous commands, not on a learned controller’s own state/action distribution. No closed-loop return, tracking, safety, contacts, hard real-time readiness or beyond-expert policy improvement was evaluated. Bounded requested-action predictions do not establish admissible executed motor inputs or state safety. Original FAILED overall/model-agreement gates and unfinished IQL remain unchanged.

## Verification, preservation and resources

Preflight passed 20 focused tests; final applicable quadrotor regressions passed **196/196**, with no failures/errors/skips. All real MPC/engine APIs were blocked. One prescribed full training replay matched parameters, optimizer moments, schedules and histories bitwise; all twenty saved checkpoints restored predictions bitwise. Independent calculations reconstructed all 4,800 causal feature rows, all twenty metric sets and the paired bootstrap. Maximum independent FP64-versus-FP32 readout discrepancy was **6.44752e-7 normalized**, below the 1e-5 numerical diagnostic tolerance, not a safety certificate. There were no nonfinite scientific losses/gradients/predictions or numerical training failures.

Before freezing, a finite ridge fixture emitted NumPy matmul floating-status warnings. Its prediction matched an independent contraction exactly and backward error was 1.98e-16. Explicit contractions replaced that B0 numerical path before any learning outcomes; objective, alpha and runtime stayed fixed. The diagnostic and warning log are retained. Final preflight had no such warnings; no warnings were suppressed.

Preservation passed before/after: 250 prior tracked source files, 87,315 accepted MPC80 artifacts, 59,268 earlier result/data entries, 12,147 Pilot B artifacts, 87 authoritative drilling files, prior failed attempts/reports/decision history and nine unchanged untracked IQL files. No old ledger or generated dataset was rewritten. The source seal remains `0ffc97e5731bfde8c22f1f7fac435ebfd4a551421de0d7cf76aad4478a62927e`. The configuration, source, tests and protocol were sealed before scientific fitting. Models, splits, predictions and plots remain outside Git. No commit/push occurred.

Largest observed process RSS: **485,539,840 bytes**, below 4 GiB. One CPU worker was used. Twenty scientific fits plus one deterministic training repeat used **105,000 supervised gradient updates**; four ridge fits were analytic. These are not MPC solver calls or RL updates. Scientific fitting itself, including repeat and persistence, charged 55.519 CPU /57.090 wall seconds.

Resource accounting starts from the accepted post-finalization cumulative **2,164.941 CPU /3,013.063 active-wall seconds**. The new B0 ceilings add exactly the authorized increments: **3,964.941 CPU /6,613.063 active-wall seconds**. Earlier unused headroom was not added again. All preflight, synthetic tests, fitting, baseline, assessment, independent verification, plots and preservation costs are charged.

**Total B0 charge: 134.306 CPU /210.959 active-wall seconds.** Final cumulative charge: **2299.247 CPU /3224.022 wall seconds**. Remaining B0 envelope: **1665.694 CPU /3389.041 wall seconds**, unspent and not permission for other work. Final report/hash/metadata charge includes a conservative 3 CPU/10 wall-second allowance. Original **1,870 MPC calls /294,000 physics steps remain unchanged**; B0 used zero of either.

## Artifacts and stop point

- Frozen source configuration: `configs/experiment_12_b0_archive_v1.json`; protocol: `docs/experiment_12_b0_protocol.md`.
- Reusable implementations: `src/quadrotor_archive_data.py`, `quadrotor_archive_learner.py`, `quadrotor_archive_assessment.py`, `quadrotor_archive_experiment.py`; entrypoint `experiments/12_b0_archive.py`.
- External evidence: `results/benchmark_selection/b0_archive_v1/` contains immutable splits/provenance, 20 final checkpoints plus the deterministic repeat, ridge coefficients, all prediction arrays, complete per-seed/per-trajectory metrics, independent audit, limits and hashes. Do not rerun/overwrite completed phases; any further fitting needs a separately reviewed protocol.
- Plots: [learning curves](../results/benchmark_selection/b0_archive_v1/plots/learning_curve.png), [phase errors](../results/benchmark_selection/b0_archive_v1/plots/phase_errors.png), [trajectory errors](../results/benchmark_selection/b0_archive_v1/plots/trajectory_errors.png).

Stop here for scientific review. Do not collect new labels, run action probes, execute learned controllers, train RL/IQL or start an acquisition comparison. The frozen B0 result is NO-GO for that next design on current evidence, despite a measurable prediction gap and the positive low-data learning-curve finding.
