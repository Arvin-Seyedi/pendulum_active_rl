# Experiment 09 — Five-case technical pilot

Only approved grid IDs 3280, 0, 6560, 767, and 44 were executed. The 3,230-state screening was not run. No training data were changed and no controller was trained.

The archived Experiment 01 arrays and classifications remain immutable. E01-native observation scalar dtypes feed the original PD controller; all 201 physical raw states are recorded separately. Original archived trajectories/actions do not exist, so historical bitwise trajectory reproduction is not claimed. Current policy repeats and recorded-action replays are checked bitwise.

## PD baseline and mathematical status

| Grid ID | Archived unsafe | New legacy unsafe | Raw unsafe | Return | Raw minimum margin | First raw crossing (s) | Real-model status | Safe gain controllers |
|---:|---|---|---|---:|---:|---:|---|---:|
| 3280 | False | False | False | 0.000000000 | 1 | — | exact_equilibrium_viable | 9/9 |
| 0 | True | True | True | -130.371899597 | -5.28318530402 | 0.10 | certified_nonviable | 0/9 |
| 6560 | True | True | True | -130.371899597 | -5.28318530402 | 0.10 | certified_nonviable | 0/9 |
| 767 | True | True | True | -140.629447137 | -5.28318503772 | 2.10 | certified_nonviable | 0/9 |
| 44 | False | False | False | -2.555300897 | 0.333333333333 | — | unresolved_model | 6/9 |

Archived-label, crossing-time, and legacy-margin discrepancies are retained in archive_comparisons.csv. Differences between raw and wrapped minima are different safety conventions, not overwritten benchmark values.

## Certificates

Unchanged Experiment 08 producer and exact-rational independent checker; real-valued rational-parameter dynamics, exact binary initial states, |u|<=2, |theta|<=1, |omega|<=3, and 200 transitions. Every preceding interval box must be safe; -2 certifies only strict upper exits and +2 only strict lower exits. Opposite exits are noncertifying. Speed clipping is inactive on steps beginning in the safety set. All attempts, exact endpoints, repetitions and checks are retained.

| Grid ID | Torque | Status | Component / crossing step | Certified separation (display) | Max theta width | Max omega width | Bits |
|---:|---:|---|---|---:|---:|---:|---:|
| 3280 | -2 | opposite_exit | omega / 8 | 0.4330653740395946 | 2.057115e-38 | 8.228460e-38 | 128 |
| 3280 | 2 | opposite_exit | omega / 8 | 0.4330653740395946 | 2.057115e-38 | 8.228460e-38 | 128 |
| 0 | -2 | opposite_exit | theta / 2 | 0.129127122118319 | 1.175494e-38 | 4.701977e-38 | 128 |
| 0 | 2 | certified_nonviable | theta / 2 | 0.038451792568206695 | 1.175494e-38 | 2.350989e-38 | 128 |
| 6560 | -2 | certified_nonviable | theta / 2 | 0.038451792568206695 | 1.175494e-38 | 2.350989e-38 | 128 |
| 6560 | 2 | opposite_exit | theta / 2 | 0.129127122118319 | 1.175494e-38 | 4.701977e-38 | 128 |
| 767 | -2 | opposite_exit | omega / 4 | 0.3468170542909376 | 1.175494e-38 | 7.052966e-38 | 128 |
| 767 | 2 | certified_nonviable | theta / 42 | 0.09331998572618358 | 1.165503e-35 | 3.427742e-35 | 128 |
| 44 | -2 | opposite_exit | omega / 4 | 0.5120630046857385 | 1.175494e-38 | 5.877472e-38 | 128 |
| 44 | 2 | opposite_exit | omega / 26 | 0.5705425948519679 | 8.169686e-37 | 2.821186e-36 | 128 |

## Frozen controller bank

Kp in {6,8,12}, Kd in {1,2,4}; baseline (8,2). Every pair was evaluated on every pilot case with 200 steps, a full independent policy repeat and recorded-action replay. Full per-controller returns, discounted returns, raw margins, crossings, saturation and witness flags are in gain_metrics.csv. Per-state safe-pair lists are in summary.json. Any union of successful pairs is an oracle feasibility diagnostic, not a deployed controller. A safe alternative counts as a rescue only when the raw-state PD baseline violates safety.

| Kp | Kd | Safe pilot cases | PD-unsafe pilot rescues |
|---:|---:|---:|---:|
| 6 | 1 | 1/5 | 0 |
| 6 | 2 | 2/5 | 0 |
| 6 | 4 | 2/5 | 0 |
| 8 | 1 | 1/5 | 0 |
| 8 | 2 | 2/5 | 0 |
| 8 | 4 | 2/5 | 0 |
| 12 | 1 | 1/5 | 0 |
| 12 | 2 | 2/5 | 0 |
| 12 | 4 | 2/5 | 0 |

## Reference ID 44 optimization

Unchanged Experiment 07 objective, transcription, constraints, solver settings and feedback tracker. Three starts (PD, zero, seeded 20-step random blocks), each solved twice with identical guesses. Open-loop and tracking executions are validated separately. Optimizer status does not establish safety; all finite candidates receive independent Gymnasium execution and recorded-action replay.

| Start | Mode | Solver status | Dynamics residual | Predicted margin | Executed margin | Return | Safe witness |
|---|---|---|---:|---:|---:|---:|---|
| PD | open_loop | Solve_Succeeded | 8.881784e-15 | 0.333333333333 | -5.33915021227 | -128.425708593 | False |
| PD | tracking | Solve_Succeeded | 8.881784e-15 | 0.333333333333 | 0.333333333333 | -2.473052246 | True |
| zero | open_loop | Solve_Succeeded | 5.551115e-16 | 0.333333333333 | -5.33915021227 | -128.425708593 | False |
| zero | tracking | Solve_Succeeded | 5.551115e-16 | 0.333333333333 | 0.333333333333 | -2.473052246 | True |
| random_blocks | open_loop | Solve_Succeeded | 3.330669e-16 | 0.333333333333 | -5.33915021227 | -128.425708593 | False |
| random_blocks | tracking | Solve_Succeeded | 3.330669e-16 | 0.333333333333 | 0.333333333333 | -2.473052246 | True |

## Verification, costs, and limitations

Technical verification passed: True. Technical tests: 38, failures: 0, errors: 0. Independent saved-array audit: True. Protected prior files unchanged: 13140. Failed checks: [].

Scientific simulator interactions: 33800 steps and 169 resets, 0.461068 simulator wall seconds. Solver attempts: 6, 0.151295 solver seconds and 5.945213 worker-process seconds. Primary interval updates / exact repeats / rational checks: 100 / 100 / 100. Unit-test interactions and final independent rational rechecks are recorded separately in technical_tests.json and independent_audit.json; they are excluded from scientific interaction totals.

Certificates apply to the declared real-valued discrete model. Gymnasium observations are separate numerical evidence; no implementation-wide floating-point error bound is established. No continuous-time, uncertain-state, model-error, real-world, infinite-horizon, or deployable safety-filter guarantee follows. The five cases are technical controls, not a representative recovery-rate sample. Reference-state optimization outcomes are not rescues of unsafe states. Failure to find a witness remains inconclusive where a mathematical certificate is absent. No performance-based tuning, additional grid screening, commit, or push was performed.

Reproduction: install requirements_experiment_09.txt and use experiments/09_recoverable_pd.py --pilot with the exact archived Experiment 01 NPZ. Existing outputs require explicit --resume; --verify checks saved hashes without new scientific computation. All generated evidence remains under results/recoverable_pd/pilot/.
