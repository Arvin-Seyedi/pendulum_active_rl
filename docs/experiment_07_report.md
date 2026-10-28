# Experiment 07 — Safety Recoverability Analysis

Primary outcome: **0/40 PD-unsafe initial states** admitted at least one independently replayed safe sequence. Open-loop: 0/40; feedback tracking: 0/40. These are finite-horizon numerical witnesses, not formal recoverability or controller guarantees.

## Frozen protocol and verification

The accepted 57-state manifest contains all 40 PD-unsafe states, eight nominal PD-safe references, eight challenging PD-safe references, and equilibrium. The four accepted pilot cases were reused byte-for-byte; 53 additional cases were executed. The scientific configuration, model, objective, solver, gains, and safety acceptance criteria are unchanged from the pilot.

CasADi 3.7.2, IPOPT 3.14.11, MUMPS 5.4.1; 600-variable, 400-equality direct multiple shooting; three fixed starts (PD, zero, seeded 20-step random blocks). Each start has 1000 iterations, 30 s solver limit, 60 s watchdog; tolerances 1e-8, exact derivatives, no bound relaxation or acceptable-termination heuristic. Hard constraints: |theta|≤1, |omega|≤3, |u|≤2. Horizon 200 steps (10 s), discount 0.99. Safety is checked on all 201 raw states with no tolerance for negative margin.

Every available candidate is executed independently in Gymnasium as float32 torques, once open-loop and separately with fixed feedback tracking. Each actual action sequence is then independently replayed. Unsuccessful and timed-out candidates are retained and evaluated when finite. No training dataset is changed and no RL controller is trained.

All checks passed: True. 399 evaluated trajectories and their replays passed the independent audit; the original PD traces match Experiment 06A bitwise. 7278 pre-existing files are unchanged, including the accepted pilot and unfinished IQL files. The twelve paired pilot solves remain the solver reproducibility check; no extra starts were introduced.

## Safe-witness counts

| Group | States | PD-safe | Open-loop witness | Tracking witness | Either mode |
|---|---:|---:|---:|---:|---:|
| PD_unsafe | 40 | 0 | 0 | 0 | 0 |
| nominal_safe | 8 | 8 | 0 | 8 | 8 |
| challenging_safe | 8 | 8 | 0 | 8 | 8 |
| equilibrium | 1 | 1 | 1 | 1 | 1 |

PD-unsafe states with an open-loop witness: [].
PD-unsafe states with a tracking witness: [].
PD-unsafe states where neither mode found a safe witness: [32, 34, 37, 38, 39, 40, 41, 44, 45, 47, 48, 50, 51, 53, 55, 56, 57, 58, 59, 62, 65, 66, 67, 70, 71, 72, 73, 78, 79, 80, 81, 82, 83, 84, 85, 89, 90, 91, 93, 94].

The complete state IDs, initial coordinates, individual start outcomes, and missing-candidate flags are recorded in state_outcomes.csv and candidate_metrics.csv. No-witness states are not classified as unrecoverable.

## Return and safety outcomes

For optimized modes, each state uses the prespecified oracle choice: greatest executed minimum margin, then greatest return, then earliest start. This is outcome-informed selection among three candidates, not deployable policy performance. Failed and unsafe candidates remain included; the tables do not condition on safe outcomes.

| Group | Method | Mean return | Worst return | Mean discounted return | Violation rate | Minimum raw margin |
|---|---|---:|---:|---:|---:|---:|
| PD_unsafe | PD | -310.704165 | -899.305647 | -172.306215 | 100.00% | -5.283185 |
| PD_unsafe | open_loop | -1113.984779 | -1360.130821 | -444.804299 | 100.00% | -45.209016 |
| PD_unsafe | tracking | -483.031303 | -952.608920 | -257.352419 | 100.00% | -3.210534 |
| PD_safe_references | PD | -0.860884 | -3.515687 | -0.812420 | 0.00% | 0.335935 |
| PD_safe_references | open_loop | -126.703974 | -129.152206 | -39.319409 | 100.00% | -5.383381 |
| PD_safe_references | tracking | -0.806955 | -3.435786 | -0.767410 | 0.00% | 0.335935 |
| all_non_equilibrium | PD | -222.177513 | -899.305647 | -123.307988 | 71.43% | -5.283185 |
| all_non_equilibrium | open_loop | -831.904549 | -1360.130821 | -328.951473 | 100.00% | -45.209016 |
| all_non_equilibrium | tracking | -345.252918 | -952.608920 | -184.042417 | 71.43% | -3.210534 |

## Solver outcomes and numerical sensitivity

The 171 scientific starts produced these statuses: {'Solve_Succeeded': 51, 'Infeasible_Problem_Detected': 120}. External watchdog timeouts: 0; solver time-limit terminations: 0.

| Group / mode | Solver-success candidates | Success but executed unsafe | Nominally feasible but executed unsafe | Largest angle deviation | Largest velocity deviation |
|---|---:|---:|---:|---:|---:|
| PD_unsafe / open_loop | 0 | 0 | 0 | 54.2628 | 10.4957 |
| PD_unsafe / tracking | 0 | 0 | 0 | 4.74441 | 7.26471 |
| PD_safe_references / open_loop | 48 | 48 | 48 | 6.46542 | 7.75809 |
| PD_safe_references / tracking | 48 | 0 | 0 | 4.89133e-09 | 3.20916e-08 |
| equilibrium / open_loop | 3 | 0 | 0 | 1.04519e-07 | 3.67499e-07 |
| equilibrium / tracking | 3 | 0 | 0 | 6.23886e-22 | 1.2278e-20 |

Nominal feasibility here requires small dynamics and bound residuals (≤1e-8) and nonnegative predicted raw margins. Merely satisfying state bounds at shooting nodes is insufficient. Residuals, predicted versus executed margins, terminal deviations, first-crossing times, and constraint statuses are retained for every candidate.

The discrepancy traces and time summaries use already saved plans and executions; no precision, solver, dynamics, control, or acceptance settings were changed. Large open-loop divergence despite small local numerical errors is consistent with amplification in unstable dynamics. This is a sensitivity observation, not a controlled causal decomposition. Feedback tracking results are reported separately and remain empirical.

## Computation and preservation

New execution: 159 solver attempts, 11.786 solver seconds, 135.726 worker-process seconds, 169600 simulator steps, 848 resets.
Including reused pilot work: 183 solver attempts (171 scientific starts plus 12 pilot repeats), 8326 IPOPT iterations, 12.663 solver seconds, 156.728 worker-process seconds, 183431 simulator steps, 1943 resets. These totals include initial guesses, evaluation, replay, and the pilot's 1031 one-step model checks.

Solver logs, defaults/options, initial guesses, last iterates, unsuccessful candidates, and timeout records are retained. interaction_costs.json separates new and reused work and gives simulator categories and NLP evaluation counts. Source, configuration, case, state-result, and final-output hashes protect reproducibility. No previous files, datasets, or checkpoints were overwritten.

## Scientific scope

A verified witness establishes an observed admissible 200-step sequence for an exact sampled state in the nominal simulator. It does not establish infinite-horizon viability, robustness, safety between discrete samples, or a certified backup controller. A tracking-generated witness is not evidence that its fixed feedback law is safe from other states.

A local solver reporting infeasibility, exhausting its budget, or failing to produce a safe replay does not prove physical unrecoverability. The 40 PD-unsafe cases were conditioned on prior PD failure; the 16 reference states and equilibrium are controls, not independent population-level validation. Counts are descriptive for this fixed sample. These results can inform later backup-controller research but do not justify deployment or alter the reviewed IQL conclusions.
