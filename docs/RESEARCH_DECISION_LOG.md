# Research Decision Log — Experiments 01–10

This retrospective log records the evidence and decisions in experiment order.
Commit references identify versioned milestones, not experiment execution or
review dates. No dates are inferred. Reports under `docs/` are the concise
scientific record; links under `results/` point to preserved local artifacts
that are intentionally excluded from Git. Experiment 06 remains unfinished and
uncommitted. Decisions below describe the subsequent approved research sequence,
not authorization to start another experiment.

## 01 — Map the existing PD expert's safety region

- **Question / previous understanding:** Could the clipped PD controller serve
  as the data-generating expert across the initial-condition domain? Nominal
  stabilization did not establish safety away from equilibrium.
- **Evidence:** On the 81×81 grid over `theta ∈ [-0.8,0.8]`,
  `omega ∈ [-2,2]`, 3,331/6,561 trajectories were safe (50.77%); 3,230 were
  unsafe. First violations ranged from 0.10 to 2.10 s, mean 0.4939 s.
- **Interpretation / limits:** The expert is not universally safe. This original
  benchmark uses wrapped observation angles and pre-action safety samples; its
  margins must not be silently equated with later all-201-raw-state diagnostics.
  It measures PD performance, not feasibility under every admissible controller.
- **Decision / reason:** Establish a nominal D0 before testing targeted
  acquisition; keep expert safety failure distinct from learner uncertainty.
- **Sources:** [benchmark implementation](../experiments/01_expert_map.py),
  [saved benchmark arrays](../results/expert_safety_map/expert_safety_map.npz);
  commit [82f8949](https://github.com/Arvin-Seyedi/pendulum_active_rl/commit/82f8949).

## 02 — Establish reproducible nominal demonstrations

- **Question / previous understanding:** Could the nominal benchmark become a
  fixed, reproducible offline dataset for comparisons? Broad PD coverage was unsafe.
- **Evidence:** Seed 42 produced 100×200 = 20,000 transitions from
  `theta0 ∈ [-0.25,0.25]`, `omega0 ∈ [-0.5,0.5]`. Mean/best/worst returns were
  −0.220083/−0.000562/−0.974336. Minimum pre-action margin was 0.734792, with
  zero violations; final successors also had zero violations.
- **Interpretation / limits:** D0 supplies safe nominal demonstrations in this
  simulator, not broad state coverage, alternative actions, or a safety guarantee.
- **Decision / reason:** Freeze D0 as the shared baseline and test ensemble
  disagreement outside its concentrated support.
- **Sources:** [summary](../results/D0/summary.txt),
  [generator](../experiments/02_generate_D0.py); commit
  [c5d850c](https://github.com/Arvin-Seyedi/pendulum_active_rl/commit/c5d850c).

## 03 — Test disagreement as an imitation-error ranking signal

- **Question / previous understanding:** Could a five-member trajectory-bootstrap
  BC ensemble identify poorly learned actions? D0's narrow support motivated
  uncertainty-based selection, but disagreement was unvalidated.
- **Evidence:** On the fixed 161×161 grid, seed-42 bounded disagreement/error
  Spearman correlation was 0.988652. Removing 16,433 jointly zero states reduced
  it to 0.827565 on 9,488 states. Across seeds 42–46, top-10%-error capture on
  positive-only subsets averaged 52.82% (sample SD 10.06 points), versus 19.02%
  for support distance and 10.017% for repeated random selection. Full-grid
  capture averaged 65.62%; raw-disagreement capture averaged only 5.39%, with
  a zero-capture seed retained.
- **Interpretation / limits:** Clipping and zero ties inflated the initial
  correlation, but did not remove ranking utility. Positive-only masks depend
  on expert errors and are retrospective diagnostics. Neither calibrated
  uncertainty nor improved closed-loop control was established.
- **Decision / reason:** Test bounded disagreement prospectively against random
  and distance acquisition with equal data budgets, rather than infer acquisition
  benefit from error detection.
- **Sources:** [robustness interpretation](../results/ensemble_uncertainty/robustness/interpretation.md),
  [tables](../results/ensemble_uncertainty/robustness/report.md); commit
  [e037a25](https://github.com/Arvin-Seyedi/pendulum_active_rl/commit/e037a25).

## 04 — Test whether targeted collection improves the learned controller

- **Question / previous understanding:** Does disagreement-guided collection
  outperform random collection? Experiment 03 supported error ranking only.
- **Evidence:** Ten paired seeds 42–51 compared uncertainty, distance, and random
  acquisition at 2,000/5,000/10,000 added transitions, H200. At the primary 5,000
  budget, uncertainty minus random mean return was +15.6436, 95% seed-bootstrap
  CI [−42.1756, +59.1252]; violation-rate difference was −0.2000 percentage
  points [−0.5500, 0.0000]. Distance minus random return was +95.692
  [68.443, 121.135], but distance increased violations versus D0 by +0.62 points
  [0.15, 1.49]. Uncertainty collected 90.77% near-equilibrium transitions;
  98.58% of its transitions after step 20 were near equilibrium.
- **Interpretation / limits:** Neither primary interval excluded zero; this is
  not evidence of equivalence. Distance's secondary reward benefit came with
  safety tradeoffs. Redundancy was demonstrated, not established as the cause.
  The pilot's mean distance exceeding p95 was a verified heavy-tail effect:
  the largest 5% contributed 97.93% of the distance sum.
- **Decision / reason:** Preserve the negative primary result and compare
  shorter rollouts without tuning selection or network hyperparameters.
- **Source:** [Experiment 04 report](experiment_04_report.md); commit
  [297828a](https://github.com/Arvin-Seyedi/pendulum_active_rl/commit/297828a).

## 05 — Reduce rollout redundancy without changing acquisition scores

- **Question / previous understanding:** Would H20 instead of H200 improve
  uncertainty acquisition? Experiment 04 exposed mostly near-equilibrium data.
- **Evidence:** Ten fresh paired seeds 52–61 tested all three strategies at both
  horizons and the same nested budgets. At 5,000, uncertainty H20−H200 mean
  return was +8.164 [−39.914, +52.714]; violation-rate difference was −0.010
  points [−0.090, +0.060]. Near-equilibrium transitions fell from 89.56% to
  14.33%, while resets rose from 25.0 to 250.1. No uncertainty query exhausted
  positive clipped-disagreement scores. The uncertainty-versus-random return
  difference-in-differences was −22.707 [−78.501, 33.803]. In the separate audit,
  6/2,499 safely H20-truncated uncertainty trajectories later violated safety.
- **Interpretation / limits:** Redundancy reduction did not establish an
  uncertainty-specific controller benefit. Random H20 improved return in a
  secondary comparison; distance H20 reduced violations but worsened return.
  Shorter rollouts also change reset cost, ranking depth, and bootstrap
  composition. Safe truncation is not full-horizon safety.
- **Decision / reason:** Retain both favorable coverage and negative primary
  findings; assess offline RL versus matched BC on existing datasets instead
  of assuming better imitation or less redundancy implies better control.
- **Source:** [Experiment 05 report](experiment_05_report.md); commit
  [92fe329](https://github.com/Arvin-Seyedi/pendulum_active_rl/commit/92fe329).

## 06 — IQL feasibility and critic diagnostics (unfinished)

- **Question / previous understanding:** Can offline reward-based learning beat
  matched BC on the same PD demonstrations? BC acquisition results did not
  establish reliable safe improvement.
- **Evidence:** The seed-41 50,000-update pilot passed implementation and replay
  checks, but expanded-data IQL reached minimum V = −33,196,702 and twin-Q
  residual 7.16×10¹⁰, far outside the continuing-return envelope
  [−1627.360440109, 0]. D0's residual was 5.80×10⁻⁷, without proving value
  accuracy. Expanded IQL's descriptive +0.584 return versus matched BC had
  unchanged 47.4% violations. The equation/gradient audit found no defect
  explaining the divergence; the first fixed-probe Q lower-bound crossing was
  at update 7,531 at a safety-stopped successor.
- **Controlled diagnostics:** Freezing only 138 safety-stopped continuation
  values at update 5,000 prevented observed negative divergence through 50,000;
  the control reproduced the original checkpoint bitwise. However, 92.264% of
  final training-state V predictions were positive. A separate smooth bounded
  readout enforced the physical range, but 99.608% of V readout derivatives
  were below 10⁻⁶. Its 67-state PD-reference MAE was 590.760; violation rate
  remained 47.4%.
- **Interpretation / limits:** The frozen-target intervention causally affected
  this fixed training trajectory; it does not isolate missing support as the
  sole cause or justify changed production targets. Bounded outputs are not
  accurate values. PD-reference discrepancies are not exact IQL value errors.
  The deterministic PD behavior supplies essentially one recorded action per
  state, so wider state coverage does not supply direct counterfactual action
  evidence. In the ideal single-action conditional distribution, expectile
  regression has no better observed action to favor; finite approximators can
  still alter state weighting and extrapolate. This is a theoretical support
  limitation, not a claim that all possible learned policies must equal PD.
- **Decision / reason:** Keep the ten-seed IQL study unlaunched and production
  IQL unchanged; do not adopt either critic diagnostic as a repair. Investigate
  counterfactual action opportunities in isolated simulation before new training
  data or a safety filter. No completed offline-RL improvement claim is supported.
- **Sources (local, uncommitted):** [pilot](../results/offline_rl/pilot_seed41/pilot_report.md),
  [diagnosis](../results/offline_rl/diagnosis_seed41/diagnosis_report.md),
  [frozen-continuation ablation](../results/offline_rl/safety_bootstrap_ablation_seed41/comparison_report.md),
  [bounded-value diagnostic](../results/offline_rl/bounded_value_seed41/comparison_report.md),
  and [PD definition](../src/expert.py). No Experiment 06 commit exists; all nine
  unfinished IQL files and generated diagnostic artifacts remain excluded.

## 06A — Test nearby actions under unchanged PD continuation

- **Question / previous understanding:** Do alternative actions at identical
  states offer safe return gains? Existing training data lacked those comparisons.
- **Evidence:** At 96 sampled states plus equilibrium, five clipped/deduplicated
  offsets tested one action followed by 199 PD steps. Empirically safe gains in
  both undiscounted and discounted return occurred at 38/96 states above 10⁻⁶,
  21/96 above 0.01, and 0/96 above 0.1. No alternative rescued any of the 40
  PD-unsafe states; two alternatives introduced violations. All qualifying gains
  occurred at initially unsaturated states.
- **Interpretation / limits:** Small local improvements exist under the fixed
  menu and continuation. Best-of-menu results use oracle outcome information,
  not a deployable controller. A one-step menu cannot establish multi-step
  optimality or unrecoverability, and empirical safety is not certification.
- **Decision / reason:** Keep all diagnostic transitions outside training and
  test whether longer admissible control sequences could rescue the failing states.
- **Source:** [Experiment 06A report](experiment_06a_report.md); commit
  [ef2d4b6](https://github.com/Arvin-Seyedi/pendulum_active_rl/commit/ef2d4b6).

## 07 — Search for finite-horizon safe control sequences

- **Question / previous understanding:** Were the 40 PD failures avoidable by
  other control sequences? Experiment 06A's unsuccessful local menu was inconclusive.
- **Evidence:** CasADi/IPOPT direct multiple shooting used three fixed starts
  per state for 40 PD-unsafe states, 16 PD-safe references, and equilibrium.
  Independently replayed safe witnesses among PD-unsafe states were **0/40**
  open-loop and **0/40** with feedback tracking. The 171 starts yielded 51 solver
  successes and 120 infeasibility statuses, without timeouts. All 48 successful
  reference-state plans became unsafe open-loop; tracking preserved safety for
  all 16 reference states. Equilibrium passed both modes.
- **Interpretation / limits:** Solver infeasibility was not a proof of physical
  impossibility. Even solver success and small residuals did not imply safe
  float32 execution; open-loop numerical sensitivity was substantial. Tracking
  witnesses were separate empirical results, not a certified backup controller.
- **Decision / reason:** Preserve unsuccessful candidates and investigate
  mathematical viability to distinguish inadequate optimization from genuine
  model-level infeasibility.
- **Source:** [Experiment 07 report](experiment_07_report.md); commit
  [040dea5](https://github.com/Arvin-Seyedi/pendulum_active_rl/commit/040dea5).

## 08 — Certify finite-horizon nonviability under an explicit real model

- **Question / previous understanding:** Could any admissible control sequence
  keep the same 40 PD-unsafe states safe? Experiment 07 had found no witness,
  but had not answered that universal question.
- **Evidence:** Directed-rounding extremal trajectories and an independent exact
  rational checker certified **40/40**, with **zero unresolved PD-unsafe cases**.
  All crossings were in theta: 20 upper and 20 lower, at steps 2–23
  (0.10–1.15 s). All 114 directional records passed checking at 128 bits.
  Smallest certified separation was 0.000527185965922 rad. None of the 16 safe
  references was falsely classified; equilibrium has an exact zero-input witness.
- **Assumptions / limits:** The real map is
  `v = omega + (15*sin(theta) + 3*u)/20`, `theta_next = theta + v/20`,
  with exact rational parameters, archived binary64 initial points interpreted
  exactly, `|u|≤2`, `|theta|≤1`, `|omega|≤3`, and 200 sampled transitions.
  On the safe strip the map is order-preserving; a safe extremal prefix followed
  by a strict crossing in the certifying direction excludes every safe admissible
  sequence. Speed clipping is inactive on steps starting inside the safe set.
  Gymnasium first-exit agreement in 114/114 directions is separate numerical
  evidence, not an implementation-wide rounding bound. No continuous-time,
  uncertain-state, real-system, or infinite-horizon guarantee is established.
  Reference-state mathematical viability remains unproved except at equilibrium.
- **Decision / reason:** Accept model-level nonviability for these exact 40
  states, rather than attribute their safety failures solely to PD or a learned
  controller. No global viability grid was needed. Future research must separate
  impossible initial conditions from improvable control; no new acquisition
  domain, safety filter, or Experiment 09 is authorized by this result.
- **Sources:** [Experiment 08 report](experiment_08_report.md),
  [per-state evidence](../results/viability/full_study/state_details.md),
  [verification](../results/viability/full_study/verification.json);
  Experiment 08 commit: [74f3888](https://github.com/Arvin-Seyedi/pendulum_active_rl/commit/74f3888a1ecd2a21fc7224ea00488bc272078378).

## 09 — Resolve every original-grid PD failure before choosing an improvement task

- **Question / previous understanding:** Experiment 08 certified nonviability
  for a separate 40-state sample. It did not establish whether the 3,230 PD failures
  on Experiment 01's original grid included recoverable states. Absence of an
  optimizer witness in Experiment 07 was still distinct from a mathematical proof.
- **Evidence:** All **3,230/3,230** archived PD-unsafe initial points received
  independently checked real-model nonviability certificates: 1,615 upper and
  1,615 lower angle crossings at steps 2–42 (0.10–2.10 s), all at 128 bits.
  Minimum certified separation was 2.21447048634e−5 rad. The complete **6,561-state**
  baseline audit checked all 201 raw states and bitwise action replay. Archived
  labels, first-crossing times and legacy minimum margins matched exactly.
  All 6,526 primary/reference directional records passed independent rechecking;
  the 32 non-equilibrium reference controls had no false certificates.
- **Search gate / negative finding:** No primary model cases remained unresolved.
  Consequently, every full-study alternative-controller evaluation and optimization
  solve was skipped by the predefined gate. Zero alternatives found must not be
  described as empirical failure of the eight-controller bank or the optimizer.
  The pilot's controller and reference-tracking results remain separate evidence.
- **Interpretation / limits:** Under the explicit real map and original constraints,
  no admissible control sequence can maintain 200-step sampled-time safety from
  those exact 3,230 points. This is stronger than unsuccessful search, and makes
  rescue searches on that cohort an infeasible improvement objective under the
  certified model. It is not a universal floating-point, continuous-time,
  uncertain-state, real-world, or infinite-horizon impossibility claim. The other
  3,331 grid points retain empirical PD safety: mean/worst return −1.369392/−7.518568,
  minimum raw margin 0.2. They do not define a mathematically certified continuous
  viable region or prove safety for another policy. PD reward optimality is unproved.
- **Decision / reason:** Preserve the certified failures and stop using this
  original cohort as a safety-rescue target under the same assumptions. Propose
  policy improvement within a defensible viable operating region, with independent
  safety outcomes, fair PD/BC comparisons and adequate counterfactual action
  evidence. First review the scientific design and justification of that region;
  do not implement Experiment 10, collect data, modify prior methods, or resume
  unfinished IQL on the strength of this result alone.
- **Sources:** [Experiment 09 report](experiment_09_report.md),
  [accepted pilot](experiment_09_pilot_report.md),
  [reproduction](experiment_09_reproduction.md),
  [all primary outcomes](../results/recoverable_pd/full_study/primary_state_results.csv),
  [verification](../results/recoverable_pd/full_study/verification.json);
  Experiment 09 commit: [9a7943f](https://github.com/Arvin-Seyedi/pendulum_active_rl/commit/9a7943fda1dfa251653cee47014d3efa2ba7a193).


## 10 — Measure safe improvement headroom before further learning

- **Question / previous understanding:** Experiment 09 certified all 3,230
  original-grid PD failures nonviable under the explicit real model, so rescue
  searches there could not test achievable improvement. The other 3,331 points
  had empirical PD safety, not a certified continuous viable region. PD reward
  optimality remained unproved; Experiment 06A had shown only small one-action
  oracle gains, and unfinished IQL lacked counterfactual action support.
- **Evidence:** Ten frozen controllers were compared on all **3,331** PD-safe
  points. LQR improved mean return from −1.369391670 to −1.313225930:
  **+0.056165740**, a **4.10% mean-cost reduction**, below the frozen practical
  threshold **+0.136939167**. Worst return improved from −7.518567842 to
  −7.435439946. LQR had **zero violations**, minimum raw margin 0.2, minimum
  paired return difference 0 and no observed per-state margin loss, but used
  **6.43% more squared-torque effort**. No tested controller met the approved
  practical reward, effort or margin-improvement criteria.
- **Optimization / negative findings:** On the frozen 64-state stratified sample,
  all **192 original solves succeeded**, yet **0/192 open-loop executions were
  safe** and **192/192 feedback-tracking executions were safe**. The best-safe
  tracking oracle improved over matched LQR by only **+0.000007367 mean return**;
  no selected state gained above 0.01. Tracking oracles genuinely improved PD at
  64/64 states; open-loop oracles used PD fallback at 64/64, which is not an
  optimization success. Twelve accepted pilot repeats were technical verification,
  not extra scientific starts. All 83 included tests passed in the workspace and
  clean export; the disposable pilot reproduced 330 NPZ bundles and 24 solver
  candidate vectors bitwise. Full saved-record replay and numerical audits passed.
- **Interpretation / limits:** A strong classical LQR baseline accounts for almost
  all demonstrated optimization gain on the selected sample. This is evidence of
  limited headroom under the tested methods, not proof that PD or LQR is globally
  optimal, or that no other method can improve. Solver feasibility is separate
  from safe float32 execution. State-specific, outcome-selected oracle choices
  are not a deployable single controller. Empirical finite-horizon simulator safety
  does not establish mathematical, continuous-time, robust or hardware guarantees.
  The fixed cohort and selected 64 points are not independent population samples;
  deterministic repeats do not supply statistical replication.
- **Decision / reason:** Together, Experiments 09–10 remove the original unsafe
  grid as a model-level rescue target and show little practical reward headroom
  in the tested safe-cohort methods. Preserve Pendulum as a reproducible
  methodological benchmark for acquisition, support, replay, safety diagnostics
  and numerical failure analysis. Before further RL, separately review control
  tasks or objectives with meaningful, safely achievable improvement headroom,
  strong classical baselines and justified counterfactual action support. Do not
  manufacture a positive result through retrospective thresholds or reward tuning.
  The broader safe-learning/active-expert-acquisition objective remains; these
  findings do not authorize Experiment 11, new data collection or IQL resumption.
  Experiment 06 remains unfinished, and its nine untracked files and diagnostic
  artifacts are preserved unchanged.
- **Sources:** [Experiment 10 full report](experiment_10_report.md),
  [accepted pilot](experiment_10_pilot_report.md),
  [reproduction and finalization checks](experiment_10_reproduction.md),
  [saved study summary](../results/safe_policy_improvement/full_study/summary.json),
  [independent verification](../results/safe_policy_improvement/finalization/independent_scientific_audit.json);
  Experiment 10 commit:
  [0705df3](https://github.com/Arvin-Seyedi/pendulum_active_rl/commit/0705df37295890e2134202f7c0fe61cc08f75306).
