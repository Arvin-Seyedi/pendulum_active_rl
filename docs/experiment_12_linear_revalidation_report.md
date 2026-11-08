# Experiment 12 — Limited QRQP adapter four-case revalidation

**Adapted execution passed on the four technical cases. The original native baseline still fails and the original symbolic/PyBullet model-agreement gate remains FAILED.** No screening, action probes, nonlinear reruns, tuning, model changes or RL training occurred.

## Frozen scope and provenance

Variant `linear40_qrqp_v1` uses the accepted `linear_conic_qrqp_v1` dispatch helper. The upstream LinearMPC class and all of its objective, constraints, model, state/input conventions, information and warm-start methods remain unchanged. Only its final solver-selection call is scoped through the local adapter: requested QRQP replaces the incompatible conic-IPOPT dispatch, and the IPOPT-specific max_iter option is removed. QRQP defaults remain unchanged. No upstream file or accepted adapter/pilot/diagnostic source is edited.

Upstream commit `6b5391d014f36fdfa0f9d22d92c77387e5274308`; original pinned Python 3.10.19/CasADi 3.7.2/PyBullet 3.2.7 environment. The accepted frozen.json, full reference and environment lock were copied byte-for-byte. The four original T0–T3 initial vectors, six-second figure-eight, 50 Hz control, 1,000 Hz physics, horizon 40, Q/R, bounds, no-disturbance/no-randomization settings and seed 1337 are retained. No 2,000 Hz runs were authorized or executed in this revalidation.

Generated records are separate under `results/benchmark_selection/linear_qrqp_revalidation/`. The original failed native linear40 results remain in the accepted technical_pilot archive.

## Complete optimization-graph verification

The complete horizon-40 QP has **326 variables, 252 parameters and 898 constraint rows**, including 246 full-row-rank shooting/initial-state equalities. Unused declared slack variables do not enter the active hard-constrained problem. Hessian eigenvalues range from 0.1 to 5.0; condition number 50.0 in these coordinates. Strict convexity supports a unique feasible QP minimizer; it is not a plant-safety certificate.

The serialized mathematical-program hash before and after dispatch is identical: `9749a1048fe71ddf9e967beb6d5af544aeb5bfc592dabcfecbb37be3954e9d4b`. Independent reconstruction derives the hover linearization from the frozen physical constants and uses a finite nilpotent exponential polynomial, rather than the upstream derivative/discretization helpers. It checks all Hessian and constraint coefficients, offsets, initial equalities, native 1e-8 constraint tightening and reference parameters at all four cases and early/middle/end reference windows.

Graph reconstruction maxima: `{'H': 0.0, 'A': 2.220446049250313e-16, 'q': 0, 'lower': 0, 'upper': 0, 'constant': 2.842170943040401e-14}`. All are below the predeclared 1e-10 reconstruction tolerance. State/input variable and parameter ordering are verified by the full coefficient comparison and independent per-solve reconstruction.

An initial checker assertion treated structural MX derivative-call dependencies as evidence of a nonconstant derivative. Exact SX expansion resolves those dependencies and proves the Hessian/Jacobian constant before coefficient comparison. The failed checker/source and log are retained. No equation or threshold was changed to make the check pass.

Every solved QP is independently checked before its action is applied. Native and independently reconstructed objective values, primal bounds, stationarity, multiplier signs and normalized complementarity are checked; the latter use native QRQP 1e-8 tolerances plus a declared 1e-10 roundoff allowance. After execution, all 2,400 primal/dual/parameter records are independently rechecked.

| Quantity | Maximum over 2,400 solves |
|---|---:|
| primal_inf | 5.20114439414e-13 |
| stationarity_inf | 3.19072459426e-13 |
| complementarity_inf | 7.88732164979e-17 |
| normalized_complementarity_inf | 2.05307729851e-17 |
| dual_sign_inf | 0 |
| objective_reconstruction_error | 1.23564353194e-13 |

All 2,400 native statuses are success. Solver feasibility/optimality concerns the frozen linear prediction QP; actual nonlinear simulator safety is checked independently below.

## Four-case execution and archived nonlinear comparisons

All eight episodes (four original plus four repeats) completed 300 applied controls and six seconds. Each contains 301 control-boundary states and **6,001 raw physics states**, including initial and terminal states. No failed or incomplete episode is assigned a safe result. Returns are negative unscaled sums of the same native post-transition quadratic cost.

| Case | Original native linear40 | Adapted linear40 | Archived MPC20 | Archived MPC40 | Archived MPC80 |
|---|---|---:|---:|---:|---:|
| T0 | Failed before action | -4.324699811 | -8.057783340 | -3.597928022 | -3.590174987 |
| T1 | Failed before action | -0.402233982 | -0.559915650 | -0.406094597 | -0.436372836 |
| T2 | Failed before action | -23.886580866 | -37.010559045 | -23.337107456 | -23.297156231 |
| T3 | Failed before action | -9.099879612 | -13.316673203 | -8.981828894 | -8.960638822 |
| Mean | N/A | -9.428348568 | -14.736232809 | -9.080739742 | -9.071085719 |
| Worst | N/A | -23.886580866 | -37.010559045 | -23.337107456 | -23.297156231 |

The original native macOS implementation has no completed performance or safety rate. Adapter success does not retroactively convert that baseline into a successful upstream execution. Nonlinear values are read from accepted records at identical cases/settings; zero nonlinear episodes were rerun.

| Adapted minus archived comparator | Mean paired return difference |
|---|---:|
| nonlinear20 | +5.307884242 |
| nonlinear40 | -0.347608826 |
| nonlinear80 | -0.357262849 |

The adapted controller beats MPC20 on these four cases, but has worse mean/worst return than MPC40 and MPC80. T1 improves slightly over both longer-horizon nonlinear baselines; T0/T2/T3 deteriorate. This is a deterministic technical cohort, not a population estimate, learning result or policy oracle.

| Case | Position RMSE (m) | Thrust-square integral (N² s) | Near-bound fraction, T1/T2 | Min normalized margin | Max QRQP iterations |
|---|---:|---:|---|---:|---:|
| T0 | 0.037762846 | 0.220997585 | 0.667% / 0.333% | 0.303407062 | 4 |
| T1 | 0.008756717 | 0.211045932 | 0.000% / 0.000% | 0.304991145 | 1 |
| T2 | 0.107645214 | 0.230865038 | 4.000% / 0.333% | 0.292502806 | 17 |
| T3 | 0.066578732 | 0.218438165 | 1.000% / 0.000% | 0.304374766 | 4 |

Mean per-case position RMSE: **0.055185877 m**; worst: **0.107645214 m**. Mean absolute squared-thrust integral: **0.220336680 N² s**. This is an effort proxy, not electrical energy. Per-channel RMS/peaks, hover-relative effort, total variation and terminal errors are retained in verification.json.

Near-bound means within the previously used 1e-7 N of either software input limit. The native linear QP tightens bounds by 1e-8, whereas nonlinear MPC uses 1e-6; this numerical difference limits cross-controller interpretation of that particular saturation metric. Actual requested-action clipping is **zero** in every adapted episode. Peak executed grouped thrust is 0.296682884230 N, within the unchanged 0.296682894230 N bound.

## Raw safety, replay and numerical limitations

All eight episodes have **zero raw state violations and zero contacts**. First-crossing times are absent. Minimum normalized margin: **0.29250280587**. Minimum raw margins by (x,x_dot,z,z_dot,theta,theta_dot), retaining each component unit: `[1.1781498724748363, 28.40049370535385, 0.6239682705078926, 28.0840735694428, 1.060361784544939, 5.1051371293217915]`. All twelve separate lower/upper margins are retained. Negative margins are never rounded into safety.

Original/repeat control and all physics states, inputs, rewards, predictions and complete QP primal/dual records match bitwise. Each episode has an uninstrumented native replay and an independent replay driven by the recorded four-motor RPMs. The latter also checks that native requested-thrust conversion produces those exact RPMs; requested commands reconstruct cost bookkeeping only. Motor replay reproduces all 6,001 physics states, contacts and rewards bitwise. The uninstrumented replay independently confirms that physics-state capture does not change observed execution.

The original nonlinear-symbolic/PyBullet one-step agreement gate remains **FAILED at its unchanged 0.001 threshold**. On the adapted trajectories its largest discrepancy is **0.00933780954872**, and three of four cases exceed 0.001. This check is distinct from the linear-QP shooting residual. The accepted geometry/integration/cached-frame decomposition is preserved; no diagnostic recurrence replaces the native model or gate.

Numerical QP feasibility is not physical safety certification. These results establish only sampled simulator safety for the four cases, finite horizon, fixed assets and inputs. They do not establish a continuous viable region, robustness to perturbations or full-cohort performance.

## Failures, runtime and preservation

No actual controller execution, solver, replay or KKT verification failed. Failure supervision is tested with injected unsuccessful/nonfinite outputs and deliberately bad residuals, without another simulator episode. A nonfinite-output mock exposes an unchanged legacy serialization limitation: the runner stops before acting, but JSON writing rejects a NaN failure payload, leaving a partial artifact rather than a completion marker. The failed test/log and revised fail-closed postcondition test are retained. This path did not occur in the real runs; it is not silently declared repaired.

Opti exposes iter_count=-1 for this conic wrapper. Actual iterations were independently recovered from every retained QRQP Converged row: 300 rows per episode, matching repeat-by-repeat. Counts range from 1 to **17**, mean 1.130833333. The original -1 fields are preserved rather than overwritten with inferred counts.

Worker CPU (episodes plus replays): **38.687790 s**. Mean action-selection wall time, including QP/KKT checks: **1.794248 ms**; maximum **80.281417 ms**. Peak worker RSS: **309.234375 MiB**. Solver calls: **2,400**; physics steps including replays: **144,000**. Static graph checks, tests, verification and preservation hashing are separate.

The accepted pilot plus revalidation gives 16,816 solver attempts and 1,440,000 physics steps; including the 14 previously accepted standalone diagnostic attempts gives 16,830 solves overall, below 40,000/3,000,000. Execution CPU and memory remain far below four CPU-hours/four GiB; the one-second per-action cap was unchanged and never triggered. One worker is used throughout.

All 18 focused simulator-free mathematical, artifact and failure-supervision tests passed. Independent verification was repeated and produced an identical verification JSON hash.

All previous experiments, the authoritative drilling files, accepted E12 pilot and diagnostic evidence, frozen configurations and nine unfinished IQL files are checked against their prior identities. No accepted source/report/artifact is overwritten. New generated records remain outside Git. No commit or push is made.

## Decision and reproduction

Compatibility revalidation is successful on the four cases. Preserve four distinct outcomes: original native linear failure; successful separately labelled adapted linear execution; unchanged archived nonlinear evidence; and still-failed original symbolic-model agreement. **Do not proceed to screening, action probes or training.** Stop for scientific review.

For a separately authorized fresh reproduction, use the pinned environment and dedicated entry point with a new --output directory: --prepare, then --preflight, then --run, then --verify. The preparer checks accepted seals and copies frozen inputs; completed attempts are reused and partial attempts are not overwritten. The CLI permits only T0–T3, original/repeat phases and 1,000 Hz physics. No nonlinear, screening, action-probe or training mode exists. Source versions at execution and final verification are retained, including the initial preflight checker and subsequent verifier-only iteration parsing.

Comparison figures show control-boundary traces. All safety classifications use the complete raw physics-step records, not plotted samples.

The controller frequency is 50 Hz in simulated time. 2 of 2400 measured action-selection calls exceeded 20 ms wall time; the maximum was 0.080281 s. The approved one-second watchdog was not exceeded. Fixed-step simulation waits for computation, so this does not alter the recorded simulated horizon, but it prevents a hard real-time guarantee from these measurements. Cold-start/runtime overhead was not discarded.
