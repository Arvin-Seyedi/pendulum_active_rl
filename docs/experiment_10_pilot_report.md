# Experiment 10 — Safe Policy Improvement Feasibility: technical pilot

Only grid IDs 3280, 2956, 5383, 2568 and 6194 were evaluated. The complete 64-state optimization subset was frozen from archived PD costs using PCG64 seed 101001, four cost strata and grid-ID tie-breaking before new controller evaluations. The remaining 60 optimization states and the full 3,331-state controller benchmark were not run. No RL or data collection for training occurred.

## LQR and fixed protocol

Clipped LQR gain K = [19.69321654676335, 5.262500081365323]. Maximum absolute discrete Riccati residual 7.105427e-15; spectral radius 0.853708043137; poles [[0.8537080431369131, 0.0], [0.2467178205575636, 0.0]]. Repeated solve bitwise: True; independent Riccati iteration discrepancy 2.096101e-13. This is a local unconstrained linear design, not nonlinear optimality or safety certification.

All controllers use identical exact archived initial states, original native observation scalar conventions and float32 executed torques. Horizon 200 steps with all 201 raw states checked; |u|<=2, |theta|<=1, |omega|<=3. Fixed PD gains are Kp in {6,8,12}, Kd in {1,2,4}; LQR uses the approved linearization, Q=diag(1,0.1), R=0.001 without tuning.

## Fixed-controller results across five pilot states

| Controller | Mean return | Mean paired return gain | Minimum paired return difference | Worst return | Mean squared-torque effort | Minimum raw margin | Violations | Practical reward criterion |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| PD_6_1 | -83.049994766 | -82.362105001 | -410.536382933 | -412.571512303 | 6.155931349 | -5.287371085 | 1/5 | False |
| PD_6_2 | -1.084551536 | -0.396661770 | -0.942225111 | -2.936698506 | 1.353376896 | 0.400000000 | 0/5 | False |
| PD_6_4 | -1.779183572 | -1.091293806 | -3.282234864 | -4.238739608 | 2.232059220 | 0.400000000 | 0/5 | False |
| PD_8_1 | -29.571724829 | -28.883835063 | -143.796228868 | -145.831358239 | 4.383707832 | -5.334083078 | 1/5 | False |
| PD_8_2 | -0.687889766 | +0.000000000 | +0.000000000 | -2.035129370 | 0.905698781 | 0.400000000 | 0/5 | False |
| PD_8_4 | -0.876840657 | -0.188950892 | -0.646069498 | -2.402142663 | 1.147231968 | 0.400000000 | 0/5 | False |
| PD_12_1 | -131.588093039 | -130.900203274 | -653.742145153 | -655.777274523 | 8.633074640 | -4.549707227 | 1/5 | False |
| PD_12_2 | -0.664991668 | +0.022898097 | +0.000000000 | -1.981746567 | 0.965963706 | 0.400000000 | 0/5 | False |
| PD_12_4 | -0.657041731 | +0.030848034 | +0.000000000 | -1.990560474 | 0.915780233 | 0.400000000 | 0/5 | False |
| LQR | -0.636495190 | +0.051394576 | +0.000000000 | -1.951276720 | 0.950885348 | 0.400000000 | 0/5 | False |

The pilot mean-return threshold is max(0.10, 10% of pilot PD mean cost) = 0.100000000. Worst-case deterioration is the **minimum per-state paired return difference**, with threshold -0.10; it is not the difference between independently minimized episode returns. The safety/margin criterion requires zero violations and no per-state margin loss >0.01. Effort success requires >=10% mean reduction with mean return loss <=0.01 and the same worst-paired-return and safety conditions. Every raw margin must be nonnegative; the comparison allowance is never a safety tolerance.

Full per-state undiscounted and discounted returns, paired differences, reward components, squared-torque effort, RMS/peak torque, saturation, total variation, raw margins, violations and terminal/last-20-step diagnostics are in fixed_metrics.csv. Effort is dt*sum(u^2), a torque-effort proxy rather than physical energy. Total variation excludes the initial-to-first-action jump, which is retained separately.

The corresponding full-benchmark mean-return threshold, frozen from existing E09 data before evaluation, is 0.136939167034; it is not a full-benchmark pilot result. Margin-eligible pilot states: 0; a margin-improvement criterion with no eligible states is not applicable. Controllers meeting the pilot effort criterion: [].

| Controller | Mean discounted return | Mean RMS torque | Max peak torque | Mean saturation | Mean total variation | Max terminal abs(theta) | Max terminal abs(omega) |
|---|---:|---:|---:|---:|---:|---:|---:|
| PD_6_1 | -40.569967239 | 0.496557119 | 2.000000000 | 0.132000 | 3.679376167 | 6.285353e+00 | 2.733836e-03 |
| PD_6_2 | -0.937630403 | 0.284539995 | 2.000000000 | 0.010000 | 1.136031711 | 1.653750e-03 | 9.076940e-04 |
| PD_6_4 | -1.339039911 | 0.358749069 | 2.000000000 | 0.013000 | 1.451225392 | 2.478405e-02 | 6.332454e-03 |
| PD_8_1 | -19.193033720 | 0.428957058 | 2.000000000 | 0.095000 | 2.866355963 | 6.283179e+00 | 1.737425e-04 |
| PD_8_2 | -0.647141637 | 0.230177589 | 2.000000000 | 0.012000 | 1.244000011 | 5.876668e-10 | 1.336119e-09 |
| PD_8_4 | -0.790939048 | 0.263861401 | 2.000000000 | 0.014000 | 1.312533390 | 1.575892e-04 | 1.263025e-04 |
| PD_12_1 | -48.163683794 | 0.573450462 | 2.000000000 | 0.204000 | 6.551904807 | 4.279348e+00 | 4.425957e+00 |
| PD_12_2 | -0.630896500 | 0.238838603 | 2.000000000 | 0.017000 | 1.573925296 | 5.252924e-16 | 4.946171e-15 |
| PD_12_4 | -0.620235062 | 0.233746224 | 2.000000000 | 0.017000 | 1.360000000 | 2.353966e-09 | 4.884731e-09 |
| LQR | -0.604506823 | 0.240601212 | 2.000000000 | 0.022000 | 1.410500002 | 3.705362e-14 | 1.269906e-13 |

## Optimization and safe-oracle results

The four non-equilibrium pilot states each use PD, zero and seeded 20-step random-block starts (SeedSequence([101004, grid_id])). Every start was solved twice with unchanged Experiment 07 equations, objective, constraints, IPOPT options and 30-second solver/60-second watchdog limits. Every finite candidate from both original and repeated solves was independently executed and replayed under both open-loop and existing tracking modes. Original solves alone enter the oracle; repeat solves are reproducibility checks. Solver feasibility remains separate from execution safety.

| Grid | Mode | Safe original candidates | Selected source | Selected start | Paired return gain | Gain >0.01 | Gain >0.1 |
|---:|---|---:|---|---|---:|---|---|
| 2956 | open_loop | 0/3 | PD_fallback | None | +0.000000000 | False | False |
| 2956 | tracking | 3/3 | optimized | PD | +0.006698528 | False | False |
| 5383 | open_loop | 0/3 | PD_fallback | None | +0.000000000 | False | False |
| 5383 | tracking | 3/3 | optimized | PD | +0.083790881 | True | False |
| 2568 | open_loop | 0/3 | PD_fallback | None | +0.000000000 | False | False |
| 2568 | tracking | 3/3 | optimized | PD | +0.082632161 | True | False |
| 6194 | open_loop | 0/3 | PD_fallback | None | +0.000000000 | False | False |
| 6194 | tracking | 3/3 | optimized | random_blocks | +0.083853154 | True | False |

Original candidate solver statuses by execution mode: {'Solve_Succeeded': 24}. Safe executions: open-loop 0/12; tracking 12/12. All unsuccessful and unsafe records remain included in optimization_metrics.csv. A PD fallback is explicitly not an optimized improvement, even when PD itself is safe. Best-of-start results are state-specific oracle comparisons, not performance of one deployable controller.

## Verification, costs and scientific scope

Verification passed: True. Technical tests: {'passed': True, 'tests': 16, 'failures': 0, 'errors': 0, 'simulator_steps': 0, 'solver_attempts': 0}. Independent audit: {'passed': True, 'all_64_selection_identities_verified': True, 'pilot_ids_verified': [3280, 2956, 5383, 2568, 6194], 'policy_repeats_bitwise': 50, 'recorded_action_replay_pairs': 98, 'all_finite_original_and_repeat_candidates_executed': True, 'solver_execution_count': 24, 'paired_solver_reproducibility_checks': 12, 'optimized_oracle_improvements': 4, 'PD_fallback_oracles': 4, 'maximum_reward_formula_difference': 3.753399258066281e-10, 'maximum_reward_decomposition_difference': 3.78985305360402e-08, 'all_201_raw_states_checked': True, 'extra_simulator_steps': 0, 'extra_solver_attempts': 0, 'wall_seconds': 0.18400733299495187}. Protected pre-existing files unchanged: 43953. Failed checks: [].

Executed costs: {'simulator_steps': 50800, 'simulator_resets': 254, 'simulator_wall_seconds': 0.6392777019063942, 'solver_executions': 24, 'solver_seconds': 0.49869612301699817, 'worker_process_seconds': 21.54229412697896, 'solver_iterations': 340, 'watchdog_timeouts': 0, 'technical_worker_failures': 0}. Unit tests make no simulator/solver calls; the independent audit uses saved arrays only. Numerical reward decomposition permits float32 arithmetic-order discrepancies without relaxing any safety threshold.

The five states are technical pilot cases selected from baseline cost strata, not an independent estimate of full-grid improvement. Practical criteria are displayed diagnostically and do not authorize full-benchmark claims or tuning. The PD-safe cohort has empirical finite-horizon simulator evidence, not a certified continuous viable region. Clipped LQR and local optimization cannot establish global optimality or its absence. Small NLP residuals do not imply safe float32 open-loop execution. No continuous-time, robustness, physical-system or infinite-horizon safety guarantee is claimed.

All prior experiments, datasets, Research Decision Log and unfinished IQL files are preserved. The pilot does not establish that diagnostic trajectories are suitable offline-RL training data. No remaining-state evaluations, learning, commit or push was performed.

Reproduce with requirements_experiment_10.txt and the exact E01 archive plus E09 baseline archive. First run experiments/10_safe_policy_improvement.py --freeze-subset; then --pilot --resume. Existing completed output can be checked with --verify without regeneration. Scientific outputs remain under results/safe_policy_improvement/pilot/.
