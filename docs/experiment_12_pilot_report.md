# Experiment 12 — Initial four-case quadrotor technical pilot

Technical execution completed with two unresolved blockers. No screening, counterfactual action study or RL training was run.

## Frozen model and task

safe-control-gym commit `6b5391d014f36fdfa0f9d22d92c77387e5274308`, Python 3.10.19, CasADi 3.7.2, PyBullet 3.2.7, Gymnasium 0.28.1 and NumPy 2.2.6. Full package versions, platform, binary/source hashes and the built PyBullet wheel are saved outside Git.

All four cases use the published six-second figure-eight task: 50 Hz control, 1,000 Hz physics, scale 0.75 m, one cycle, offset (0,1). T0 denotes a hover initial state, not a different constant-reference task. Q=diag(5,.1,5,.1,.1,.1), R=.1 I. Random initial states, disturbances, parameter randomization and action normalization are disabled.

State order: (x, x_dot, z, z_dot, theta, theta_dot); units (m,m/s,m,m/s,rad,rad/s). Two grouped physical thrust inputs are in N. Native state/input constraint values and their binary32 bounds are preserved.

Grouped thrust bounds: 0.0563233755529 to 0.29668289423 N per channel; nominal hover 0.1323 N per channel. Native MPC observations/actions remain binary64; space bounds are binary32. No extra action dtype conversion was introduced.

| Case | Frozen initial state |
|---|---|
| T0 | [0, 0, 1, 0, 0, 0] |
| T1 | [0.0, 0.7853981633974483, 1.0, 0.7853981633974483, 0.0, 0.0] |
| T2 | [-0.5, 0, 0.75, 0, 0.15, 0] |
| T3 | [0.5, 0, 1.25, 0, -0.15, 0] |

T1 is read directly from the native reference: both initial translational velocities are π/4 m/s. Initial binary64 identities and the full reference are archived.

Maximum reset representation discrepancy: 2.77556e-17; quaternion/Euler representation need not reproduce a requested angle bitwise. Frozen requested values are unchanged.

## Results at 1,000 Hz

Returns are negative cumulative native quadratic costs, evaluated at post-transition states against reference index k+1. They are unscaled sums; integral costs equal 0.02 times these costs. The four states are technical cases, not a population sample.

| Controller | Complete cases | Mean return | Worst return | Safe/contact-free completed cases | Mean squared-thrust integral (N² s) |
|---|---:|---:|---:|---:|---:|
| linear40 | 0/4 | N/A | N/A | N/A | N/A |
| nonlinear20 | 4/4 | -14.736232809 | -37.010559045 | 4/4 | 0.225366305 |
| nonlinear40 | 4/4 | -9.080739742 | -23.337107456 | 4/4 | 0.221651044 |
| nonlinear80 | 4/4 | -9.071085719 | -23.297156231 | 4/4 | 0.221585256 |

| Case | MPC20 return | MPC40 return | MPC80 return |
|---|---:|---:|---:|
| T0 | -8.057783340 | -3.597928022 | -3.590174987 |
| T1 | -0.559915650 | -0.406094597 | -0.436372836 |
| T2 | -37.010559045 | -23.337107456 | -23.297156231 |
| T3 | -13.316673203 | -8.981828894 | -8.960638822 |

| Case/controller | Position RMSE (m) | Min normalized margin | Squared-thrust integral (N² s) | Peak channel thrust (N) |
|---|---:|---:|---:|---:|
| T0/nonlinear20 | 0.05063687 | 0.27912501 | 0.22192545 | 0.25043370 |
| T0/nonlinear40 | 0.03014315 | 0.30432072 | 0.22461036 | 0.29668187 |
| T0/nonlinear80 | 0.02840933 | 0.30500876 | 0.22507022 | 0.29668188 |
| T1/nonlinear20 | 0.01232723 | 0.30478043 | 0.21087866 | 0.15466809 |
| T1/nonlinear40 | 0.00882451 | 0.30510677 | 0.21103227 | 0.15470352 |
| T1/nonlinear80 | 0.00941050 | 0.30500799 | 0.21098333 | 0.15501907 |
| T2/nonlinear20 | 0.12124031 | 0.26441625 | 0.24931463 | 0.29668190 |
| T2/nonlinear40 | 0.10679632 | 0.30346550 | 0.23376920 | 0.29668190 |
| T2/nonlinear80 | 0.10670534 | 0.30500773 | 0.23333968 | 0.29668190 |
| T3/nonlinear20 | 0.07457475 | 0.28190468 | 0.21934647 | 0.18367909 |
| T3/nonlinear40 | 0.06639244 | 0.30067599 | 0.21719234 | 0.16047365 |
| T3/nonlinear80 | 0.06666443 | 0.30190108 | 0.21694779 | 0.15824118 |

Linear MPC has no completed returns or safety-rate estimate. Its original/repeat attempts at both physics resolutions fail before any action is applied; an empty safe prefix is not a safe episode.

## Safety, repeatability and independent checks

All 24 completed nominal-frequency nonlinear executions (12 conditions plus repeats) and all 24 nonlinear refinement executions (including controller-level repeats) were observed safe and contact-free. There were zero raw state violations. This is finite sampled simulator evidence, not continuous-time, robust or physical-flight certification.

Each completed 1,000 Hz episode contains 6,001 raw physics states; each 2,000 Hz episode contains 12,001. The global normalized minimum margin among completed executions was 0.264416252445.

Minimum raw margins across completed executions, by (x,x_dot,z,z_dot,theta,theta_dot), with each component retaining its own units: `[1.0576650097785207, 27.905142458812925, 0.60830986817208, 27.88060975628325, 0.9453932257601079, 5.391562142318828]`. All twelve lower/upper margins, trajectories, first-crossing fields, terminal behavior and per-controller costs/effort/RMS/peak/saturation/variation metrics are in verification.json.

All 32 original/repeat condition pairs across both physics resolutions match bitwise for their available state/action/reward, prediction and solver-iteration records. Eight pairs contain only failed zero-transition linear-MPC attempts. Each of all 64 attempts has both an uninstrumented action replay and an independently instrumented action replay. The 48 completed attempts reproduce control states/rewards bitwise; instrumented replays additionally reproduce all physics states and contacts. Zero-transition replays are bookkeeping checks only.

Instrumentation delegates to the unchanged native stepSimulation, then queries state/contact information. It never refreshes upstream cached states or changes forces. Uninstrumented replay confirms no observed change to control-boundary states/rewards on these trajectories.

Independent reference reconstruction maximum error: 1.27676e-15. Cost reconstruction maximum error: 0. Executed thrust reconstruction maximum error: 1.11022e-16 N.

Independent/native RK4 agreement maximum error: 5.57887e-15. Maximum reconstructed shooting-dynamics residual: 4.35149e-10. Predicted bound excesses remain separate from executed-state checks.

Requested thrusts, clipping, motor groups (1,4)/(2,3), PWM/RPM conversions and realized thrusts were independently checked. Raw safety checks use no rounding tolerance. Small arithmetic tolerances in cost/motor reconstruction do not relax safety boundaries.

## Failed checks and preserved upstream behavior

**Linear-MPC execution blocker:** the published configuration requests QRQP, but its unchanged Darwin branch selects IPOPT on an Opti conic problem. The installed CasADi has no conic-IPOPT plugin. The native controller catches the initial error and returns a fallback; the runner then detects absent solved statistics, records failure and applies no fallback action. No claim of control infeasibility follows.

**One-step numerical tolerance failure:** symbolic predictions differ from PyBullet by more than the frozen 0.001 scaled tolerance in several completed conditions. The overall maximum is 0.00859824313788. The independent RK4 reconstruction agrees with native RK4; the discrepancy concerns the symbolic/PyBullet model pair.

Only 4/12 original nonlinear case/controller pairs pass the one-step tolerance. The dominant discrepancy is pitch-rate prediction in rad/s: a matched-state one-step discrepancy, not a closed-loop trajectory difference.

The symbolic pitch lever is L/sqrt(2)=0.0280721392131 m, whereas cf2x.urdf uses 0.028 m rotor offsets. Saved-transition fits are close to the URDF value. Rotation diagnostics quantify the much smaller residual after using that documented lever in a matched-state equation comparison. This is evidence of a source-geometry contribution, not a changed model, a complete implementation-error bound or a corrected trajectory.

The nonlinear warm start shifts rows of the input array (`u_guess[:-1]=u_guess[1:]`), not time columns. This behavior is unchanged and independently checked against the actual initial guesses. No repaired controller is evaluated. Linear/nonlinear classes also retain their native numerical constraint tightenings. Nonlinear terminal cost includes a constant hover-input offset absent from linear MPC; this constant does not change its optimizer. Evaluated trajectory costs are identical in definition across controllers.

All 12 completed nonlinear refinement pairs preserve safety classification and remain within the frozen 1% relative-cost tolerance. Maximum relative cost change: 0.81799449%. Linear-MPC refinement is unavailable, not passed.

## Computational cost and preservation

Recorded worker CPU: 515.793 s; solver attempts: 14,416 (14,400 successful nonlinear solves and 16 pre-action linear failures); physics steps including replays: 1,296,000. Peak worker RSS: 311.922 MiB. Maximum completed action-selection time: 0.947808 s. Maximum solver iterations: 23.

The one-second action-selection watchdog was not triggered. Runtime includes native solve preparation in each action call; the largest cold-start call is reported rather than discarded. Native solver logs and unsuccessful candidates are retained. Physics integration and readout overhead are separately timed in each record. Setup/build time is separate from worker-execution CPU.

The measured nonlinear throughput is compatible with a four-CPU-hour study, but is not a guarantee for unseen states. A single 16-state/four-controller pass costs up to 19,200 solve calls; four nine-action/300-step probes cost up to 10,800 more before reuse. These fit 40,000 as a separate phase, but adding the nominal/repeated/refinement technical allocation can exceed a study-wide 40,000 cap. Exact accounting, replays, repeats and baseline reuse must be reviewed before scaling. No cap was raised.

Independent preservation verified 53,819 pre-existing project files, 87 authority files and 289 upstream files. All nine IQL files also match previously sealed identities and remain unimported. All 127 applicable regression tests passed in the historical Python 3.13 environment; the five new contract tests separately passed under pinned Python 3.10. Unfinished IQL tests were excluded.

A CLI postprocessing/logging attempt exited 120 after writing a complete verification JSON, with no captured diagnostic output. Independent verification was rerun directly twice and its JSON hashes compared. This attempt and the PyBullet build failure are retained as technical execution records, not silently removed. Neither altered simulation records.

## Decision

Nonlinear MPC execution, deterministic replay, full physics capture and cost reconstruction are demonstrated on the four technical cases. The pilot does not pass the complete readiness gate: linear MPC remains blocked and the frozen symbolic/PyBullet discrepancy tolerance fails. Stop for scientific review. No 16-state screen, counterfactual menu, data acquisition, controller repair or training is authorized by these results. Horizon comparisons are descriptive technical diagnostics; they do not establish RL headroom.

Generated evidence: results/benchmark_selection/technical_pilot/. Authoritative source archive, configuration/reference identities, full environment lock, build failure/workaround logs, per-attempt logs/trajectories, verification.json, usage.json and plots remain outside Git.
