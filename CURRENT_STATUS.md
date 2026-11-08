# Safe RL Research — Current Status

## Current stage

Experiments 01–05, 06A, 07–10 and 11A–11C are complete and scientifically reviewed.
Experiment 12's reviewed pilot/diagnostic/qualification milestone is complete;
**overall benchmark qualification remains FAILED**. The targeted H2/MPC20 v2
requalification passes without changing any historical outcome or threshold.
The separately reviewed **E12_MPC80_FEAS_v2 scoped expert-feasibility gate PASSED**
on all sixteen frozen states; it does not supersede either failed benchmark gate.
Experiment 06/IQL remains unfinished: its full ten-seed study was not launched.
Experiment 11 Gate 0 passed provenance review, but physically bounded-actuator
benchmark selection remains BLOCKED by missing engineering specifications.
The separately approved Experiment 11A reproduced six native ideal-input drilling
cases; it did not establish physical-reference consistency or bounded-input safety.
Experiment 11B confirms nonzero reference residuals for the archived numerical
model beyond tested rounding/assembly effects. Physical contact/reaction assumptions
remain unresolved. Experiment 11C confirms supplied-force incompatibility and
conditional ideal-reaction consistency without validating a physical plant.
Further physical drilling-model development and numerical investigation are
paused pending external model/engineering clarification and a new reviewed design.
Frozen-result reproduction is verification only. No compensation, dynamics change,
actuator limits, controller redesign,
integration refinement, RL training or IQL resumption is authorized here.

The long-term objective remains safe learning with active expert-data acquisition.
The evidence now separates imitation-error detection, actual controller improvement,
and whether an initial state admits any safe control sequence.

## Fixed benchmark and data

- Pendulum-v1; PD expert `clip(-8*theta - 2*omega, -2, 2)` N·m.
- Safety set: `|theta| <= 1 rad`, `|omega| <= 3 rad/s`.
- Normalized margin: `min(1 - |theta|, 1 - |omega|/3)`.
- Experiment 01: 81×81 grid over `[-0.8,0.8] × [-2,2]`, 200 steps;
  3,331/6,561 trajectories safe (50.77%). First violations: 0.10–2.10 s,
  mean 0.4939 s. This original benchmark uses observation-derived wrapped angles
  and pre-action samples; later all-201-raw-state safety results are distinct.
- D0: seed 42, 100×200 = 20,000 transitions from
  `theta0 ∈ [-0.25,0.25]`, `omega0 ∈ [-0.5,0.5]`.
  Mean/best/worst return: −0.220083 / −0.000562 / −0.974336;
  minimum pre-action margin 0.734792; zero violations, including final successors.
  D0 remains unchanged at `data/pendulum_D0.npz`.

## Completed evidence

| Experiment | Main finding | Reviewed source / existing commit |
|---|---|---|
| 01–02: expert map and D0 | PD is not universally safe; nominal demonstrations are safe and narrowly supported. | `experiments/01_expert_map.py`, `experiments/02_generate_D0.py`; `82f8949`, `c5d850c` |
| 03: ensemble disagreement | Useful for detecting imitation error in this setting, with saturation/tie and seed qualifications; this did not test acquisition benefit. | `experiments/03_robustness_analysis.py`; `e037a25` |
| 04: active acquisition | At 5,000 transitions, uncertainty−random return +15.644, 95% CI [−42.176, 59.125]; violation-rate difference −0.200 percentage points [−0.550, 0.000]. No demonstrated primary benefit. Distance improved return with safety tradeoffs. | [Report](docs/experiment_04_report.md); `297828a` |
| 05: rollout horizon | Uncertainty H20−H200 return +8.164 [−39.914, 52.714]; violation-rate difference −0.010 points [−0.090, 0.060]. Near-equilibrium data fell from 89.56% to 14.33%, but no reliable primary improvement; resets increased about tenfold. | [Report](docs/experiment_05_report.md); `92fe329` |
| 06A: counterfactual actions | Safe gains in both return metrics at 38/96 states above 1e−6, 21/96 above 0.01, none above 0.1. No rescues among 40 PD-unsafe states; two alternatives introduced violations. Oracle diagnostics, not a deployed policy. | [Report](docs/experiment_06a_report.md); `ef2d4b6` |
| 07: recoverability optimization | 0/40 PD-unsafe states had verified safe open-loop or tracking witnesses. All 48 converged non-equilibrium reference candidates failed open-loop execution; tracking remained safe on all 16 references. Solver failure alone did not prove nonviability. | [Report](docs/experiment_07_report.md); `040dea5` |
| 08: finite-horizon viability | **40/40 PD-unsafe states independently certified nonviable for the declared real-valued discrete model**, no unresolved PD-unsafe cases; no false certificates on 16 reference controls. | [Report](docs/experiment_08_report.md); commit reference in [decision log](docs/RESEARCH_DECISION_LOG.md) |
| 09: original-grid PD failures | **3,230/3,230 original PD failures certified nonviable** under the same real model. All 6,561 baselines audited with unchanged labels/times. Full-study controller search and optimization were skipped by the predefined gate. | [Report](docs/experiment_09_report.md); commit reference in [decision log](docs/RESEARCH_DECISION_LOG.md) |
| 10: safe policy improvement | LQR mean-return gain +0.056165740 (4.10%), below the frozen +0.136939167 threshold; 0/3,331 violations, 6.43% more squared-torque effort. No tested controller passed any practical improvement criterion. Tracking-oracle gain over matched LQR was only +0.000007367 across 64 states. | [Report](docs/experiment_10_report.md); commit reference in [decision log](docs/RESEARCH_DECISION_LOG.md) |
| 11A: native drilling reproduction | Six original/repeat pairs (12 executions) matched 85 numeric fields bitwise. Five cases were sampled-safe; P3 had 401 rotational/358 axial violating samples and 411 lookup failures. Nonzero physical-reference residuals limit interpretation. | [Report](docs/experiment_11a_report.md); commit reference in [decision log](docs/RESEARCH_DECISION_LOG.md) |
| 11B: static reference audit | Exact-rational and 128/256-bit interval checks confirm both reference identities fail for the archived model. Omitted forcing is approximately −0.152400579 rad/s² at 20 s; it is an equation discrepancy, not trajectory drift. Physical constrained reduction remains unresolved. | [Report](docs/experiment_11b_report.md); [reproduction](docs/experiment_11b_reproduction.md) |
| 11C: constraint/reference pilot | Four snapshots and all five operator groups reject unreacted full-force compatibility. Ideal-reaction, virtual-work and block equations agree conditionally, with nonzero retained reaction effects. The affine reference fails under both formulations; physical applicability remains unresolved. | [Report](docs/experiment_11c_report.md); [reproduction](docs/experiment_11c_reproduction.md) |
| 12: quadrotor numerical qualification | Native linear dispatch and original 0.001 symbolic/plant agreement fail. Adapted linear pilot succeeds, but held-out QRQP failures remain. NMPC40/80 complete all four held-out cases safely; new H2/MPC20 v2 runs complete twice and pass corrected KKT checks. Overall qualification remains failed. | [Consolidated report](docs/experiment_12_report.md); [finalization reproduction](docs/experiment_12_finalization.md) |

Experiment 04 uses paired seeds 42–51; Experiment 05 uses 52–61. Reported intervals
are seed-bootstrap intervals conditional on each study's fixed evaluation set.
Zero-inclusive intervals do not establish equivalence. Both studies retain all
seeds, negative findings, and separate return/safety outcomes.

## Experiment 08 model and scope

The exact real-valued map is
`omega_next = omega + (15*sin(theta) + 3*u)/20`,
`theta_next = theta + omega_next/20`, with `h=1/20`, `g=10`, `m=l=1`,
continuous admissible torque `|u|<=2`, and the closed safety set above.
Initial states are the exact archived binary64 values, not rounded display values.
The horizon is 200 transitions, with safety required at all 201 sampled states.

On the safe strip all state/action derivatives are nonnegative; a step beginning
in the safe set satisfies `|omega_next|<=81/20<8`, so speed clipping is inactive.
A safe-prefix constant −2 trajectory with a strict upper crossing bounds every
hypothetical safe path from below; +2 gives the reflected lower-crossing proof.
Directed-rounding MPFR intervals and an independent exact-rational/Taylor checker
verify the premises and crossing. Opposite exits do not certify nonviability.

All 40 certificates cross an angle boundary (20 upper, 20 lower), at steps 2–23
(0.10–1.15 s), using 128-bit precision. Minimum certified separation is
0.000527185965922 rad. The five reviewed pilot cases were reused unchanged;
the remaining 38 PD-unsafe and 14 reference states completed verification.
All 21 technical tests, 114 direction checks, exact interval repeats, and
171 bitwise simulator replay comparisons passed. No global viability grid was needed.

The 16 references have empirical Gymnasium PD witnesses, not mathematical
viability proofs from this method. Equilibrium has an exact zero-input witness.
No global floating-point implementation-error bound, continuous-time guarantee,
robustness certificate, infinite-horizon viable region, or safety filter is claimed.
The 40 states were selected by prior PD failure; 40/40 is not a population estimate.
Within this model, no admissible controller can prevent their finite-horizon exits.
This conclusion does not retroactively turn Experiment 07 solver statuses into proofs.

## Experiment 09: full original-grid result and research decision

The original stored Experiment 01 axes and masks, exact binary initial states,
and fixed IDs `81*omega_index + theta_index` were reused. All 6,561 PD episodes
were evaluated for 200 steps and independently replayed, preserving the original
float64 initial observation and subsequent native float32 observation/action path.
Archived, remeasured legacy, and raw-state classifications and first crossings
agreed everywhere; legacy minimum margins matched exactly. All 201 raw states
were checked, including final successors.

All **3,230/3,230** archived PD-unsafe points received independently checked
real-model certificates: 1,615 upper and 1,615 lower angle crossings, steps 2–42
(0.10–2.10 s), all at 128 bits. Smallest certified separation:
`2.21447048634e-5 rad`. All 6,526 primary/reference direction records were rechecked;
32 non-equilibrium reference controls had no false certificates. No primary model
cases remained unresolved. Full-study alternative-controller evaluations and
optimization solves were **zero because the prespecified certificate gate skipped
them**, not because a controller search empirically failed.

Whole-grid PD mean/worst return: −155.382357/−947.709757, violation rate 49.2303%,
minimum raw margin −5.283185. The 3,331 empirically PD-safe points had mean/worst
return −1.369392/−7.518568 and minimum raw margin 0.2. These observations do not
prove a continuous or robust viable region, PD reward optimality, or safety for a
modified policy. No global floating-point error bound or continuous-time,
uncertain-state, physical-system, or infinite-horizon guarantee was established.

Experiment 08 had resolved a separate 40-state sample; Experiment 09 resolves
all original grid failures under the explicit real model. Searching for safety
rescues at these certified points cannot succeed under those assumptions.
This motivated Experiment 10's performance-headroom study on the empirically
PD-safe grid, with safety evaluated independently. It did not authorize treating
those points as a certified continuous viable region.

The accepted five-case pilot is unchanged. Three new full-study tests passed;
6,561 replay pairs and 1,318,761 raw samples passed independent audit. Preservation
covered 13,870 pre-existing files at execution. Finalization checks preserve all
scientific artifacts and distinguish the authorized context/log edits from their
historical hashes; original manifests remain immutable.

## Experiment 10: limited demonstrated performance headroom

Ten frozen controllers were compared on the same 3,331 archived PD-safe points,
with 200 transitions and all 201 raw states checked. LQR mean/worst returns were
−1.313225930/−7.435439946 versus PD −1.369391670/−7.518567842. Its mean gain
**+0.056165740 (4.10% cost reduction)** fell below the frozen practical threshold
**+0.136939167**. LQR had **0/3,331 violations**, no observed per-state margin
loss, minimum raw margin 0.2, and minimum paired return difference 0. However,
squared-torque effort increased **6.43%**. No tested controller met the approved
practical reward, effort, or margin-improvement criteria.

The 64-state optimization subset was frozen by four PD-cost strata, PCG64 seed
101001 and grid-ID tie-breaking before new evaluations. All **192 original
solves succeeded**, but **0/192 open-loop executions were safe**; **192/192
feedback-tracking executions were safe**. Every execution was independently
replayed with float32 torques. The best-safe tracking oracle improved mean return
over matched LQR by just **+0.000007367** (no state gained more than 0.01).
All 64 tracking-oracle choices genuinely improved on PD; all 64 open-loop oracle
outcomes used PD fallback, which is not optimization success. Twelve pilot repeat
solves are retained separately, not additional scientific starts.

Solver feasibility is not execution safety. Tracking oracles select a different
outcome-informed candidate per state; they are not a deployable single controller.
Empirical simulator safety does not establish a mathematical safety guarantee.
These deterministic finite-cohort results neither prove global reward optimality
nor exclude better methods or other tasks. They show little demonstrated reward
headroom under the tested methods and meaningful reward/effort tradeoffs.
Experiments 09–10 support retaining this benchmark for methodology checks while
reviewing more meaningful control objectives, rather than manufacturing a positive
learning result. The deterministic PD action-support limitation remains unresolved.

The accepted pilot, frozen subset, original reports, full-study records and
verification manifests are preserved. See [reproduction and finalization
checks](docs/experiment_10_reproduction.md) for clean-export verification and the
explicit distinction between authorized context edits and historical hashes.

## Experiment 11A: reproducible error map, unresolved physical consistency

The exact authoritative ZIP and its distinct source variants were preserved.
P0 uses the top-level simulation; P1–P4 use Chart_Plot IC1–IC4 with the native
parameterized IC_Test kernel; P5 is the newly approved zero-error technical input,
not an archived case. The standalone Chart script is not P3. MATLAB R2024b
Update 5/MACA64 ran each case twice, with unchanged ideal inputs, 20 seconds at
1e−4 seconds, and all 200,001 raw error states checked per execution.

All six repeat pairs matched 85 numerical workspace fields bitwise (510
comparisons). All 2,400,000 recorded transitions were independently reconstructed;
maximum absolute/state-scaled discrepancies were 3.55271e−15/3.46945e−15.
No cross-language bitwise equivalence is claimed. There were no execution failures,
timeouts or nonfinite/nonreal physical trajectories.

P0/P1/P2/P4/P5 remained strictly inside the sampled raw safety domain. P3 had
401 lower-rotational violations from 0.1323 s and 358 lower-axial violations from
0.1347 s. Its 411 lookup failures began at 0.1318 s, before the first crossing;
this temporal ordering does not establish causation. The native stale-memory
fallback was preserved; 401 lookup attempts had complex indices. No first lookup
failed, so that failure path was not dynamically exercised. These are finite,
sampled results for the implemented error dynamics, not mathematical safety proofs.

For the moving physical reference `q_d = IC + Vs*t`, define
`r0 = F_M,d - D_M*Vs - K_M*IC`. The reconstructed acceleration balance contains
`r0 - t*K_M*Vs`, which is omitted from the implemented error dynamics. Native
MATLAB maximum component magnitudes (the largest components are angular) are:

| Quantity | Magnitude | Unit |
|---|---:|---|
| r0 | 2.36554798527244e−6 | rad/s² |
| K_M*Vs | 0.00762011974485406 | rad/s³ |
| r0 minus saved err.mat | 4.66281145639069e−11 | rad/s² |
| r0 − 20*K_M*Vs | 0.15240057922651 | rad/s² |

The last value is an acceleration-balance discrepancy evaluated on the reference,
not a measured trajectory deviation. Independent FEM reassembly confirms the
nonzero reference-stiffness term. P5 remains exactly at zero in the implemented
reduced error map; this does not prove an exact physical reference trajectory.
No residual compensation or recalculation of the original parameters was applied.

Keep four findings separate: native bitwise reproducibility passes; sampled
raw-state safety holds for five cases and fails for P3; 11A did not establish exact
physical-reference consistency; physically admissible bounded-input
safety remains blocked. No engineering actuator bounds were supplied or inferred
from the observed inputs. Physical consistency must be reviewed before safe RL
controller development. [Reproduction and read-only verification](docs/experiment_11a_reproduction.md).

## Experiment 11B: numerical inconsistency confirmed, physical assumptions unresolved

The reviewed static audit froze the authoritative sources, binary64 operators,
IC, Vs and nominal inputs. Source provenance, executed matrix/coordinate
reconstruction and algebraic error transformation passed. Exact archived products
give maximum angular components `|r0|=2.36554798765712e−6 rad/s²` and
`|K_M*Vs|=0.00762011974485422 rad/s³`. Independently reassembled 128/256-bit
directed-rounding enclosures exclude zero for both identities. The dominant
reassembled reference-stiffness term is `0.00762011974664 rad/s³`; tested dot
rounding (~6.325e−15) and assembly perturbations (~1.694e−11 in that component)
do not explain it.

The derived error equation includes `r0-t*K_M*Vs`, omitted by the native map.
At 20 s its dominant component is approximately **−0.152400579 rad/s²**.
This is a matched-state acceleration discrepancy, not an integrated or measured
trajectory deviation, and does not establish a cause for P3's safety failure.

The reported **14.07%** difference compares the native reduced stiffness matrix
with an ideal-holonomic-constraint alternative, using a relative Frobenius norm
in the fixed SI coordinates. It is not measured physical modeling error, and
the alternative is not assumed physically correct. Chapter 5's contact/reaction
closure remains unresolved. Retain the printed rotation-sign discrepancy,
`cond2(K_M)≈4.03e17`, threshold-dependent numerical rank 9/10, nodal versus
element-chord frames, constant nominal loads and initial native-load bookkeeping.
Nominal commands do not supply actuator limits.

All 108 archived sample points passed; the independent checker verified 54 unique
equations and 540 exact-rational rounding bounds. The original 21 pilot tests,
source identities and failed technical attempts are preserved. Finalization adds
read-only static replay and clean-export checks; see the [report](docs/experiment_11b_report.md)
and [reproduction guide](docs/experiment_11b_reproduction.md). No new drilling
trajectory or controller was executed. Numerical-model validation does not validate
the physical drilling plant. Preserve the original idealized map; residual
compensation and physical-controller redesign are not justified by this audit alone.

## Experiment 11C: physical development paused pending external clarification

Only P5 at indices 0/200000 and P0/P3 at index 1000 were used. Exact arithmetic
and 128/256-bit directed rounding checked 25 fixed operator columns, the four
snapshots and two reference states. The five operator groups (KT, DT, surface
inputs, signed bit loads and gravity) all contain nonzero `C*M^-1*w` witnesses.
Ideal reactions affect retained accelerations in all four snapshots; the largest
retained angular effect was 82.850353 rad/s² at P3's saved 0.1 s state. These are
static equation comparisons, not simulated or measured trajectory changes.

Coordinate enforcement passes: `U=Tq` and `CT=0` preserve geometric coordinates
by construction. The incompatibility is between that embedded motion and the
supplied full mechanical force balance without additional forces. It must not be
reported as an observed geometric escape. Exact reaction-Schur, virtual-work and
full block equations agree for all 31 RHS columns under fixed ideal bilateral
constraint assumptions. This is conditional mathematical consistency, not proof
that those reactions describe drilling contact.

The archived affine reference fails under both tested formulations. The native
dominant forcing at 20 s remains about −0.152400579 rad/s²; the ideal formulation
already has a maximum retained translational balance of 0.408512575 m/s² at
reference time zero. Neither is an integrated deviation. The approximately 14.07%
relative Frobenius matrix comparison remains coordinate-dependent and is not a
measured physical modeling error or evidence that the alternative is correct.

The reviewed pilot passed 35 mathematical/provenance tests, identical repeats at
all three arithmetic settings, 3,100 independent interval residual checks and
950 cross-precision overlaps. Finalization adds the full regression suite and a
clean-export replay without altering the accepted records. See the report and
reproduction guide above. No original equation, input, reference or controller
was changed, and no new trajectory was executed.

Physical applicability and actual plant invalidity both remain unresolved. Obtain
external clarification of all five outstanding contracts before further work:

- Constraint reactions and friction, including which effects are already included.
- Rotational coordinates and work-conjugate torque conventions.
- Material versus spatial coordinates and frozen/moving wellbore geometry.
- Boundary conditions, stress-free geometry/preload, cyclic modes and reference construction.
- Physically admissible absolute surface force/torque limits, correction channels,
  and any documented rate/slew restrictions; nominal values and peaks are not limits.

Preserve the original ideal-input numerical error-map benchmark with qualified
safety claims. Experiments 11A–11C establish reproducibility and specific numerical
inconsistencies, not a validated physical drilling plant. No further numerical
drilling investigation, residual compensation or controller redesign is authorized
without new external evidence and scientific review.

## Experiment 12: completed evidence, failed overall qualification

The published six-second planar-quadrotor figure-eight task is frozen at
safe-control-gym `6b5391d014f36fdfa0f9d22d92c77387e5274308`, with 50-Hz control,
1,000-Hz physics and unchanged model, constraints, objectives and MPC horizons.
Initial four-case evidence, refinement, diagnostic, versioned QRQP adapter,
held-out MMAQ_v1, integrity audit and targeted H2 v2 records remain separate.
The native Darwin conic/IPOPT dispatch failure remains a software outcome.
The adapter changes dispatch/options only, not the control problem.

The original **0.001** symbolic/PyBullet agreement gate remains **FAILED**.
Model-mismatch-aware checks do not establish exact model equivalence or a general
floating-point safety guarantee. NMPC40/80 completed all four held-out cases twice
without observed state/input violations or contacts; NMPC80 mean/worst returns
were −39.331860180/−65.816857638, minimum normalized raw margin 0.200745220.
Adapted linear H0/H2 remain incomplete. H0 cycles to QRQP's 1,000-iteration limit;
H2's trace ends at iteration 12 with a singular/regularity-loss failure, not a
proved infeasibility or demonstrated iteration-limit event. Missing failed
vectors/latencies cannot be recovered retrospectively.

The v2 KKT verifier corrects absent-side complementarity, sign and nonfinite
handling without changing numerical tolerances. The new H2/MPC20 runs both
complete **300 controls and 6,001 raw states**, with zero observed violations,
bitwise repetitions/action replays and **300 checked candidates each**. Return
is −25.138729807869687, minimum normalized raw margin 0.2699882352585654.
The first 54 historical controls/candidates match bitwise; the v1 defect is
reproduced on the **newly captured step-54 candidate**, not on a recovered old
rejection vector. Historical v1 runs remain incomplete with that vector absent.
Future v2 capture records failures/nonfinite data explicitly and applies no fallback.

Solver-only H2 median/p95/p99/max latency was 7.187/11.802/12.889/72.745 ms,
with two 20-ms overruns. All 600 supervised calls exceeded 20 ms; MPC80 also
frequently exceeded it in archived kernel timing. Offline simulation success is
not hard real-time readiness. Overall benchmark qualification remains failed;
successful sampled executions are neither formal nor hardware safety guarantees.

At the earlier qualification milestone the balance was 12,970 solves / 948,000
physics steps. The following scoped feasibility study is now accepted separately.

### Accepted MPC80-only feasibility milestone

E12_MPC80_FEAS_v1 remains **FAILED** at its first physics callback: one actual
solve/step, zero complete controls. Reading the stale reset command cache was an
instrumentation defect, not demonstrated applied-thrust infeasibility. Its full
300-solve / 6,000-step failed reservation remains charged. The separately sealed
v2 observer records current native RPM and submitted force/torque arguments;
KF-derived thrust is not hardware telemetry. Historical v1 evidence was not repaired.

**V2 passed 16/16 states, 32/32 scientific executions and 64/64 independent
replays**, with bitwise repetitions and replay agreement. Each scientific run
contains 300 controls and all 6,001 raw states. All **9,600 optimization candidates**
passed unchanged native and corrected-v2 independent checks. No observed raw
state/input violations or contacts occurred; every numerical reserve was positive.
Mean/worst original return: **−20.241625399 / −78.787551231**; position RMSE:
**0.087969885 / 0.197426305 m**; squared-thrust effort:
**0.228677863 / 0.242540074 N²s**. Minimum normalized raw margin: **0.0756528087895**.
All six componentwise raw margins and causal reserves are in the
[accepted scientific report](docs/experiment_12_mpc80_feas_v2_report.md).

Maximum prediction disagreement is **0.0189680501631**, so the original **0.001**
gate remains **FAILED**. Source-recurrence closure is at most 3.552714e−15; this
supports numerical reconstruction, not exact symbolic/plant equivalence or a
formal robust safety theorem. All **9,600/9,600** native-action latency estimates
exceeded 20 ms (median 24.337480 ms; maximum 309.950500 ms). The accepted scope is
finite-cohort offline simulation-expert feasibility, not hard real-time readiness,
physical flight safety, reward optimality, safe action diversity or learning benefit.

V2 charged 9,600 solves / 576,000 physics steps. **Remaining original allowance:
3,070 solves / 366,000 physics steps**, with no failed reservation refunded.
The accepted execution ledger carries 1,501.732 CPU / 1,730.196 active-wall seconds,
including v1/preflights/record audit; additive finalization overhead is recorded
separately without rewriting that ledger. These sixteen states remain development/
qualification data, never independent held-out learned-policy evaluation data.
No action probes or learning runs have occurred. A safe-action diversity design
requires separate execution/budget approval. [Read-only reproduction and
finalization checks](docs/experiment_12_mpc80_reproduction.md).

## Experiment 06/IQL remains unfinished

The seed-41 expanded-data critic diverged. Freezing 138 safety-stopped continuation
values prevented that catastrophic divergence in the paired diagnostic, but did
not establish accurate values. A smooth physically bounded readout prevented
unbounded predictions while producing extensive saturation and inaccurate
PD-return reference estimates. These were diagnostic-only interventions; they were
not adopted as production IQL or combined. The planned full study remains on hold.

The deterministic PD datasets lack direct counterfactual actions at identical
states. Greater state coverage does not establish action coverage or supported
improvement beyond the teacher. Advantage weighting can reweight imitation;
it does not supply missing counterfactual evidence. IQL source/config/test/report
files remain untracked and excluded from Experiment 12 finalization. All local
pilot/diagnostic artifacts are preserved under `results/offline_rl/`.

## Handoff and preservation

Read [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md), the
[decision log](docs/RESEARCH_DECISION_LOG.md), and the current reports before
proposing further work. Do not restart completed experiments or tune methods using
reviewed results. No next experiment is authorized by this status update.

- Latest milestone: `src/quadrotor*.py`, separately versioned E12 configurations,
  `requirements_experiment_12.txt`, historical entry points under `experiments/12_*`
  and read-only `experiments/12_verify_milestone.py`. Accepted evidence is under
  `results/benchmark_selection/`; additive finalization evidence is in its
  `finalization/` folder. [Reproduction](docs/experiment_12_finalization.md).
  Generated outputs and all nine IQL files are excluded from Git.
- Preserved 11C implementation: `src/drilling_constraint_*.py`, frozen
  `configs/experiment_11c.json`, and `requirements_experiment_11c.txt`.
  Read-only finalization uses `experiments/11c_verify_constraints.py`; the historical
  pilot entry point is `experiments/11c_constraint_audit.py`. Accepted results are
  `results/drilling_constraints/pilot/`; finalization evidence is separate under
  `results/drilling_constraints/finalization/`. No IQL code is a dependency.
- Preserved 11A implementation: `src/drilling_reproduction.py`, `src/drilling_numerical_audit.py`,
  `src/drilling_pilot_verification.py` and `src/matlab/e11a_*.m`. Entry point:
  `experiments/11a_native_drilling.py`; configuration: `configs/experiment_11a.json`;
  dependencies: `requirements_experiment_11a.txt`. Frozen native execution files
  retain their original hashes. Additive read-only verification uses
  `experiments/11a_verify_native.py`; tests: `tests/test_drilling*.py`.
- Preserved 11A evidence: `results/drilling_reproduction/native_pilot/`;
  all 12 native workspaces, histories, source snapshots, audits and original
  completion markers remain unchanged and outside Git. Finalization audits are
  separate, under `results/drilling_reproduction/finalization/`.
- Preserved Experiment 10 implementation: `src/policy_improvement*.py`; entry points
  `experiments/10_safe_policy_improvement.py` and `experiments/10_full_study.py`.
  Frozen configurations: `configs/experiment_10_*.json`; dependencies:
  `requirements_experiment_10.txt`; tests: `tests/test_policy_improvement*.py`.
- Preserved Experiment 10 evidence: `results/safe_policy_improvement/pilot/` and
  `results/safe_policy_improvement/full_study/`; 33,310 fixed-controller records,
  192 original solves, 384 optimization executions and 128 oracle records.
  Generated outputs remain outside Git; the reviewed scientific reports remain
  unchanged. [Reproduction](docs/experiment_10_reproduction.md).
- Preserved Experiment 09 implementation: `src/recoverable_pd*.py`; entry points
  `experiments/09_recoverable_pd.py` and `experiments/09_full_study.py`.
  Configurations: `configs/experiment_09_*.json`; dependencies:
  `requirements_experiment_09.txt`; tests: `tests/test_recoverable_pd*.py`.
- Preserved Experiment 09 evidence: `results/recoverable_pd/pilot/` and
  `results/recoverable_pd/full_study/`; 6,561 baseline rows, 3,230 primary
  certificate rows, 33 reference rows and 6,526 direction rows. Generated outputs
  remain outside Git. See [reproduction](docs/experiment_09_reproduction.md).
- Preserved Experiment 08 implementation: `src/viability*.py`; entry points:
  `experiments/08_viability_pilot.py` and `experiments/08_viability_study.py`.
- Frozen configuration/dependency files: `configs/experiment_08_*.json` and
  `requirements_experiment_08.txt`; tests: `tests/test_viability*.py`.
- Saved evidence: `results/viability/pilot/` and `results/viability/full_study/`;
  57 state rows, 114 direction rows, exact certificates, identity/source hashes,
  replay arrays, and verification records. Generated artifacts remain outside Git.
- [Reproduction and archive verification](docs/experiment_08_reproduction.md)
  explain dependencies and the historical preservation hashes. The authorized
  context-document updates must be distinguished from changes to scientific artifacts;
  original manifests must not be rewritten to accept new content.
