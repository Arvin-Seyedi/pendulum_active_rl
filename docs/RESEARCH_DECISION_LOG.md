# Research Decision Log — Experiments 01–12

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


## 11 / 11A — Establish native drilling reproducibility before physical-model claims

- **Question / previous understanding:** Experiments 09–10 excluded the original
  Pendulum grid as a model-level rescue target and demonstrated limited practical
  headroom in the tested safe-cohort methods. The next task was benchmark
  feasibility, not another assumed RL improvement. Experiment 11 Gate 0 inspected
  the actual authoritative directional-drilling ZIP, distinct script variants and
  stored parameters. Provenance review passed; physically bounded-actuator safety
  remained blocked because no applicable engineering actuator contract was sourced.
- **Approved scope / evidence:** Experiment 11A separately reproduced ideal-input
  native MATLAB numerics only. P0 used the top-level script; P1–P4 used exact
  Chart_Plot IC1–IC4 through native IC_Test; P5 was a newly approved zero-error
  control, not an archived case. The standalone Chart script was not substituted
  for IC3. Six original/repeat pairs (**12 executions**) matched all **85 numeric
  workspace fields bitwise**, on MATLAB R2024b Update 5/MACA64. All 200,001 raw
  states per execution were checked. All 2,400,000 recorded transitions passed
  independent reconstruction, with maximum absolute/state-scaled discrepancies
  3.55271e−15/3.46945e−15. No cross-language bitwise equivalence is claimed.
- **Safety and lookup findings:** P0/P1/P2/P4/P5 were strictly sampled-safe.
  P3 had **401 lower-rotational violations** from **0.1323 s** and **358
  lower-axial violations** from **0.1347 s**. Its **411 lookup failures** began
  at **0.1318 s**, preceding the first crossing; this does not establish causation.
  There were 401 attempts with complex indices. Original stale-memory fallback
  was preserved. No first lookup failed, leaving that failure path dynamically
  unexercised. No physical trajectory was nonfinite/nonreal, and no native run
  failed or timed out. Initial-sample inclusion explains the smaller P1/P4 axial
  margins relative to earlier historical summaries.
- **Physical-reference limitation:** For `q_d=IC+Vs*t`, define
  `r0=F_M,d-D_M*Vs-K_M*IC`. Reconstructed physical acceleration balance includes
  `r0-t*K_M*Vs`, omitted from the implemented reduced error dynamics. Native
  maximum component magnitudes, in their dominant angular components, were
  **2.36554798527244e−6 rad/s²** for r0,
  **0.00762011974485406 rad/s³** for K_M*Vs, and
  **4.66281145639069e−11 rad/s²** for r0 minus saved err.mat. At 20 seconds the
  omitted term has maximum magnitude **0.15240057922651 rad/s²**. This is a
  reference acceleration-balance discrepancy, **not a measured trajectory
  deviation**. Independent FEM reassembly confirms nonzero K_M*Vs. P5 remains
  exactly at the zero-error equilibrium of the implemented reduced map, not a
  demonstrated exact physical reference trajectory. No compensation was inserted.
- **Interpretation / limits:** Native bitwise reproducibility, sampled raw-state
  safety, physical-reference consistency and physically admissible bounded-input
  safety are distinct conclusions. The numerical milestone passes; P3 is unsafe;
  physical consistency is limited by nonzero reference residuals; bounded-input
  feasibility remains blocked. Deterministic repeats are verification rather than
  population samples or formal safety guarantees. The native ideal-input results
  do not establish a safe expert, a backup controller, or RL improvement headroom.
- **Decision / reason:** Accept and preserve Experiment 11A as a reproducible
  numerical milestone. Before safe RL controller development, separately review
  a focused investigation of the physical-reference/error-equation relationship;
  otherwise learning could optimize an error map whose physical interpretation is
  not established. Do not silently compensate residuals, recalculate parameters,
  insert actuator limits or redesign the controller. No integration refinement,
  new control experiment or IQL resumption follows automatically. Experiment 06
  remains unfinished and its nine untracked files remain unchanged.
- **Finalization / sources:** All **92 included tests passed** in workspace and
  clean export. The export re-audited the accepted records without new MATLAB
  processes or control rollouts. Original reports/manifests, source identities,
  the authority ZIP/scripts and Experiments 01–10 remain preserved; the versioned
  report adds reviewed clarifications. See [scientific report](experiment_11a_report.md),
  [reproduction and read-only checks](experiment_11a_reproduction.md),
  [original native verification](../results/drilling_reproduction/native_pilot/final_verification.json),
  [clean-export verification](../results/drilling_reproduction/finalization/clean_export_final_verification.json);
  Experiment 11A commit:
  [3757db5](https://github.com/Arvin-Seyedi/pendulum_active_rl/commit/3757db51f87f2ff84b7c1f123cd60620c0a53630).

## 11B — Confirm numerical reference inconsistency before changing the physical model

- **Question / previous understanding:** Experiment 11A established bitwise native
  reproduction and observed nonzero reference-balance terms. It did not resolve
  whether those terms reflected numerical assembly/conditioning effects or an
  exact error-coordinate inconsistency. Physical contact assumptions and an
  engineering actuator contract were still unverified. The approved 11B scope
  was static mathematical verification using the original data and archived
  trajectories, without any plant correction or new control execution.
- **New evidence:** Source provenance, executed FEM/coordinate reconstruction
  and the algebraic error transformation passed. Exact rational products of
  archived binary64 operands give maximum angular components
  **2.36554798765712e−6 rad/s²** for `r0=F_M,d-D_M*Vs-K_M*IC` and
  **0.00762011974485422 rad/s³** for `K_M*Vs`. Independent **128/256-bit**
  directed-rounding reassembly separates both residual vectors from zero. The
  dominant reassembled stiffness-reference term is **0.00762011974664 rad/s³**;
  tested dot rounding (~6.325e−15) and assembly perturbations (~1.694e−11 in that
  component) do not explain it. The native error equation omits `r0-t*K_M*Vs`,
  approximately **−0.152400579 rad/s²** in its dominant component at 20 seconds.
  This is a **matched-state equation discrepancy, not a trajectory deviation**.
- **Independent checks:** All **108 archived points** passed. The independent
  rational checker verified **54 unique equations and 540 componentwise rounding
  bounds**, with six bitwise-identical archived sample pairs. The maximum floating
  identity discrepancy was 1.28210e−11, below the frozen 1e−8 scaled tolerance.
  Static finalization replay reproduced exact products, both interval endpoint
  sets, four nominal-load variants and all 108 points directly from the saved
  native workspaces. No state was advanced or controller called.
- **Physical interpretation remains unresolved:** The **14.07%** stiffness
  comparison is `||K_alt-K_M||_F/||K_alt||_F`, between the native projection and
  an ideal-holonomic-constraint alternative in fixed SI coordinates. It is not
  measured physical modeling error and does not establish that the conventional
  alternative is correct. Chapter 5's stated geometric constraints do not by
  themselves establish the necessary contact/reaction closure. Retain the printed
  rotation-sign discrepancy, `cond2(K_M)≈4.03e17`, threshold-dependent numerical
  rank 9/10, displacement origins and nodal/element-frame distinctions. Nominal
  bit loads are constant on the reference; native initial nominal-load bookkeeping
  is separate from the omitted forcing. P5 is an equilibrium of the implemented
  error map, not a verified exact physical reference trajectory. No cause for
  P3's safety crossing is established by these checks.
- **Scientific decision / reason:** Accept 11B as validation of properties of
  the archived numerical model, **not validation of the physical drilling plant**.
  Preserve the original idealized error simulator with that qualification.
  Do not automatically insert residual compensation, recalculate the reference,
  substitute the conventional reduction, or redesign physical control: the
  physical reference and contact/reaction assumptions are not yet established.
  Missing engineering actuator specifications remain a separate blocker. A
  focused next physical-model investigation requires its own reviewed design;
  none is implemented here. Experiment 06/IQL remains unfinished and all nine
  local files remain unchanged; no RL training or new expert data follow.
- **Finalization / sources:** All **108 applicable regression tests passed in
  both workspace and clean export**. The export excluded the nine IQL files.
  Accepted reports, source snapshots, failed technical attempts and generated
  artifacts remain unchanged; preservation covered 53,435 historical project
  files and 87 authority files, with exact historical Git checks for authorized
  context/log updates. Verification wrote zero accepted-archive files and launched
  zero MATLAB processes or new research/control trajectories. See the
  [scientific report](experiment_11b_report.md),
  [reproduction and read-only checks](experiment_11b_reproduction.md),
  [accepted integrity manifest](../results/drilling_static_audit/approved_static/final_integrity.json),
  [workspace verification](../results/drilling_static_audit/finalization/workspace_verification.json),
  [clean-export verification](../results/drilling_static_audit/finalization/clean_export_verification.json);
  Experiment 11B commit:
  [f2b0696](https://github.com/Arvin-Seyedi/pendulum_active_rl/commit/f2b0696269693cca98f263de8bdf97aedf2e7824).

## 11C — Separate enforced geometry from mechanical closure; pause physical development

- **Question / previous understanding:** Experiment 11A reproduced the native
  ideal-input error map bitwise; 11B confirmed nonzero reference-balance terms
  beyond tested rounding/assembly effects. Neither established whether the
  mechanical projection or reference followed from a closed physical drilling
  model. Experiment 11C was approved only as a static mathematical pilot on
  P5 indices **0/200000** and P0/P3 index **1000**, with unchanged matrices,
  inputs, reference and 128/256-bit arithmetic settings.
- **New evidence:** All five fixed operator groups (KT, DT, surface mapping,
  signed bit-load mapping and gravity) and all four snapshots have validated
  nonzero `C*M^-1*w` witnesses without additional reactions. Ideal reactions
  have nonzero retained-acceleration effects in every snapshot; the largest
  saved angular effect is **82.850353 rad/s²** at P3's 0.1 s state. These are
  static equation comparisons, not integrated or measured trajectory deviations.
- **Essential distinction:** The native embedding `U=Tq` enforces geometric
  coordinates through `CT=0`; this is **not an observed geometric escape**.
  What fails is compatibility between that embedded motion and the supplied
  full mechanical force balance. Preserving its retained acceleration would
  generally require extra forces with nonzero virtual work. Under fixed ideal
  bilateral constraints, reaction-Schur, virtual-work and full-block formulations
  agree exactly for **31 RHS columns**, with a unique block solution and
  independent interval/LDL checks. This establishes conditional mathematical
  consistency, not that these ideal reactions represent actual drilling contact.
- **Reference and numerical limitations:** The archived affine reference fails
  under both tested formulations without changing IC, Vs or nominal inputs.
  The native dominant balance at 20 s remains approximately **−0.152400579
  rad/s²**; the ideal interpretation already has a maximum retained translational
  balance of **0.408512575 m/s²** at reference time zero. Neither is a trajectory
  deviation or evidence of physical controller instability. The approximately
  **14.07%** relative Frobenius stiffness-matrix comparison is coordinate-dependent,
  not measured physical modeling error and not a reason to select the conventional
  reduction. Literal dyadic rank, numerical rank and structured-real energy/mode
  statements remain distinct; the latter do not establish physical cyclic modes.
- **Scientific decision / reason:** Experiments **11A–11C establish numerical
  reproducibility and specific mathematical inconsistencies, not validation of a
  physical drilling plant**. Physical applicability of ideal reactions and actual
  plant invalidity remain unresolved. Preserve the original MATLAB error map as
  an explicitly ideal-input methodological benchmark, with safety claims confined
  to its implemented equations. Pause further physical drilling-model development
  and new numerical drilling investigations pending external model/engineering
  evidence and a separately reviewed design. Do not automatically substitute a
  reduction, insert compensation, construct a new reference, redesign a controller
  or invent actuator limits. Frozen-result replay is verification, not a new study.
- **External information required:** Obtain the original author's reaction and
  friction closure, including effects already represented in the load vector;
  rotational-coordinate and work-conjugate torque conventions; material versus
  spatial coordinates and wellbore geometry; support/boundary conditions,
  stress-free geometry/preload and reference/cyclic-mode/gauge construction; and
  physically admissible absolute surface force/torque limits, correction-channel
  meaning and any documented rate restrictions. Nominal commands and observed
  peaks provide no actuator contract. Engineering/author clarification is needed
  before any further physical numerical investigation, not another arbitrary fix.
- **Verification / preservation / sources:** All **35 reviewed pilot tests**
  passed, with identical arithmetic repeats. Finalization passed **122 applicable
  regression tests in both workspace and clean export**. Replay reproduced all
  four original/repeat snapshot pairs, exact arithmetic, both interval endpoint
  sets, five operator groups and all scientific classifications. Independent
  checks verified **3,100 interval residuals and 950 cross-precision overlaps**,
  plus exact block uniqueness and the coordinate/missing-force identity. Accepted
  reports, source snapshots and generated records remain unchanged; preservation
  covers 53,435 historical files, 87 authority files and all nine unchanged IQL
  files. IQL remains unfinished and unimported; no RL training or new expert data
  was authorized. No new MATLAB process, trajectory or physical model was executed.
  See [scientific report](experiment_11c_report.md),
  [reproduction and read-only checks](experiment_11c_reproduction.md),
  [accepted review seal](../results/drilling_constraints/pilot/review_integrity.json),
  [workspace replay](../results/drilling_constraints/finalization/workspace_verification.json),
  [clean-export replay](../results/drilling_constraints/finalization/clean_export_verification.json);
  Experiment 11C commit:
  [024bec5](https://github.com/Arvin-Seyedi/pendulum_active_rl/commit/024bec5ae66e11298ae7a07a6bc07f33b12e1dfd).

## 12 — Preserve failed benchmark qualification; scope the next expert-feasibility question

- **Question / previous understanding:** Pendulum demonstrated little practical
  improvement headroom under the tested methods; deterministic demonstrations
  lacked counterfactual action support. Experiments 11A–11C did not validate a
  physical drilling plant, so that direction remains paused. The published planar
  quadrotor figure-eight was selected for a reproducible simulation-feasibility
  assessment, without assuming RL superiority or changing objectives to create it.
- **New evidence:** The pinned native Darwin linear MPC fails conic/IPOPT
  dispatch. A separately versioned QRQP adapter preserves the mathematical control
  problem and completes the four-case pilot safely, but held-out H0/H2 remain
  incomplete. H0 cycles to 1,000 iterations; H2's printed trace ends at iteration
  12 on a singular/regularity-loss path. Neither proves infeasibility. Warm-start
  aliasing is verified, but its causal contribution is unestablished. Historical
  failed vectors and some full-call timing observations remain unavailable.
- **Prediction and safety distinction:** The original **0.001 symbolic/PyBullet
  agreement gate remains FAILED**. Geometry, cached attitude, conversion and
  integration explain retained discrepancies, not a global floating-point safety
  theorem. MMAQ_v1 has **26/32 completed executions** and remains failed overall.
  NMPC40/80 each complete all four held-out cases twice with zero observed
  violations/contacts. NMPC80 mean/worst returns are **−39.331860180/−65.816857638**,
  minimum normalized raw margin **0.200745220**; its original maximum scaled model
  discrepancy remains **0.01326088346**. Empirical finite-horizon execution is
  neither model equivalence nor formal/hardware safety.
- **Verifier integrity and new targeted result:** Versioned v2 corrects finite-side
  complementarity, dual-sign and nonfinite handling with unchanged tolerances.
  All **7,932** retained MMAQ candidates pass corrected checks, without rewriting
  incomplete outcomes. The separately approved H2/MPC20 v2 executions both finish
  **300 controls and 6,001 raw states**, with zero observed violations, bitwise
  repetitions/action replays and **300 independently checked candidates/run**.
  Return is **−25.138729807869687**, minimum normalized raw margin
  **0.2699882352585654**. After bitwise-matched 54-control prefixes, the v1 defect
  is reproduced on a **newly captured step-54 candidate**; the missing historical
  rejection vector was not recovered and old full-horizon outcomes remain absent.
- **Computation and limitations:** H2 solver-only median/p95/p99/max is
  **7.187/11.802/12.889/72.745 ms**, with **2/600** over 20 ms; supervised calls
  overrun **600/600**. MPC80 often exceeds 20 ms even in archived kernel timing.
  Preserve all overruns and missing telemetry; offline simulation does not imply
  hard real-time readiness. Future v2 failure capture retains available candidates,
  residuals, nonfinite values and partial records without fallback actions.
  Remaining conservative allowance is **12,970 solver calls / 948,000 physics
  steps**. No new full-study allocation or sixteen-state/action-probe execution
  is authorized by this milestone.
- **Scientific decision / reason:** Do not continue open-ended baseline debugging
  or declare the published benchmark fully qualified. Propose a **limited,
  prospectively defined MPC80-only simulation-expert feasibility study** for
  separate design and budget review: MPC80 has complete repeated technical and
  held-out sampled-safety evidence and verified retained numerical candidates,
  enough to justify a focused question rather than further baseline repairs.
  This narrows the proposed expert-feasibility scope; it does not establish
  superiority over other controllers or remove their failed outcomes. Freeze
  the next cohort, resource/failure accounting and decision criteria in advance.
  Establish distinct safe alternative actions/continuations and informative data
  support before any active-versus-random acquisition experiment with equal
  transition budgets and independent held-out policy evaluation. Learning
  headroom, sample-efficiency benefit and learned-policy safety remain untested.
  Drilling development and Experiment 06/IQL remain paused; no new study is
  implemented or launched here.
- **Verification / preservation / sources:** **191 applicable regression tests
  pass in both workspace and clean export**, excluding unfinished IQL.
  Saved-record audits agree in canonical scientific JSON, checking **106** attempt
  records, **650,640** physics transitions, **10,932** retained candidates,
  **51,192** exact majorant components and **168** complete-episode replay records.
  Preservation checks cover **53,819 historical files, 87 authority files,
  289 upstream files, 5,438 sealed artifacts, 43 accepted sources and nine IQL
  hashes**. Generated records stay outside Git; finalization uses zero new solves
  or physics steps. Prior decision entries remain unchanged. See the
  [consolidated report](experiment_12_report.md),
  [read-only reproduction](experiment_12_finalization.md),
  [original pilot](experiment_12_pilot_report.md),
  [diagnostic](experiment_12_diagnostic_report.md),
  [adapter revalidation](experiment_12_linear_revalidation_report.md),
  [MMAQ_v1](experiment_12_mmaq_v1_report.md),
  [integrity audit](experiment_12_qualification_integrity_report.md),
  [H2 v2 report](experiment_12_h2_requalification_v2_report.md) and
  [clean-export verification](../results/benchmark_selection/finalization/clean_export_comparison.json).
  Implementation milestone:
  [a11e36c](https://github.com/Arvin-Seyedi/pendulum_active_rl/commit/a11e36c3e34317810f95e64ad8587895b3faef6b).

## 12 — Accept scoped MPC80 expert feasibility; retain failed benchmark qualification

- **Question / previous understanding:** Earlier E12 evidence supported complete
  MPC80 executions on technical and held-out cases but did not qualify the original
  four-controller benchmark. The original 0.001 symbolic/PyBullet criterion and
  overall benchmark qualification remain FAILED; no safe counterfactual actions,
  learning headroom or active-acquisition benefit had been demonstrated.
- **New evidence:** The sealed sixteen-state PCG64-120001 cohort completed under
  unchanged MPC80 in E12_MPC80_FEAS_v2: **16/16 states, 32/32 scientific runs,
  64/64 replays and 9,600 independently verified candidates**. All 6,001 raw states
  per run were checked, with zero observed state/input violations or contacts,
  strictly positive causal reserves, bitwise scientific repeats and replay
  agreement. Mean/worst original return is **−20.241625399 / −78.787551231**;
  position RMSE **0.087969885 / 0.197426305 m**; squared-thrust effort
  **0.228677863 / 0.242540074 N²s**; minimum normalized raw margin
  **0.0756528087895**.
- **Negative findings / integrity:** V1 remains FAILED after one actual solve and
  one physics step because its callback read a stale reset command cache. This
  was not a demonstrated thrust violation. The separate v2 observer records
  current motor and force/torque submission arguments without changing native
  behavior; thrust derived from those arguments is not physical sensing. V1's
  failed 300-solve / 6,000-step reservation remains charged. Maximum v2 prediction
  disagreement is **0.0189680501631**: neither failed original gate is superseded.
  All **9,600** native-action latency estimates exceed 20 ms (median 24.337480 ms,
  maximum 309.950500 ms); no hard real-time readiness is established.
- **Interpretation / limits:** Accept finite-cohort offline simulation-expert
  feasibility only. This is not formal robust/continuous-time/hardware safety,
  a certified operating region, global optimality or proof of supported policy
  improvement. The sixteen qualification states cannot become independent
  held-out learned-policy evaluation data. Demonstration action variation at
  different states/phases does not establish safe alternatives at identical
  decision contexts, especially with reference preview and warm-start history.
- **Scientific decision / reason:** Preserve the scoped positive result and stop
  open-ended baseline debugging. Review a prospectively specified safe-action
  diversity study before any execution or learning. Keep the original nine-action
  menu, four probe IDs, objective and constraints; first establish reproducible
  matched contexts and independently safe continuations. Beyond-expert RL still
  needs informative counterfactual action/outcome support. Expert imitation and
  equal-budget active-versus-random acquisition are separate future questions.
  Physical drilling development and unfinished IQL remain paused.
- **Resources / verification / references:** Remaining original allowance is
  **3,070 solves / 366,000 physics steps**, with no refund/reset. No new execution
  allocation is granted here. Finalization used zero solves/steps, passed
  **248 tests in both workspace and clean export**, and reproduced the accepted
  scientific digest `6ada71395cb84324cad4f6c28638b200b8b82cc52daab188564af895d62a2f6c`.
  It checked 87,315 sealed v2 files, prior research/authority hashes and all nine
  unchanged untracked IQL files. A new finalization comparator's ledger-snapshot
  mismatch was explicitly reconciled against the two recorded audit charges;
  accepted evidence was not changed. See [accepted report](experiment_12_mpc80_feas_v2_report.md),
  [v1 failure](experiment_12_mpc80_feas_v1_report.md),
  [v2 preflight](experiment_12_mpc80_feas_v2_preflight_report.md),
  [read-only reproduction](experiment_12_mpc80_reproduction.md) and
  [clean-export comparison](../results/benchmark_selection/mpc80_feas_finalization/clean_export_comparison.json).
  Implementation milestone: [9848529](https://github.com/Arvin-Seyedi/pendulum_active_rl/commit/9848529c4c4cd831af991a7141af4b5cc9eeb892).
