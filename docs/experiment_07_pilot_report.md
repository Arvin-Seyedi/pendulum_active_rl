# Experiment 07 — Safety Recoverability Technical Pilot

Only the four predefined cases were run. The 57-state study remains unauthorized. No training data were generated for RL or added to existing datasets.

## Fixed method

CasADi 3.7.2 / IPOPT 3.14.11 / MUMPS 5.4.1, direct multiple shooting with 600 variables and 400 dynamics equalities, exact initial-state parameter, 200 independent torques, hard |theta|≤1, |omega|≤3, |u|≤2 constraints. Objective: original undiscounted reward cost; no terminal cost/constraint, safety penalty, slack acceptance, or tuning. Three starts (PD, zero, ten seeded random 20-step blocks); each solve repeated independently. Limits: 1000 iterations, 30 s solver wall time, 60 s process watchdog, tolerance and constraint tolerance 1e-8, no acceptable-termination heuristic or bound relaxation. All defaults/options and solver logs are retained.

Open-loop executes clipped float32 optimized torques. Tracking separately applies clip(u_ref − 8*(theta−theta_ref) − 2*(omega−omega_ref),−2,2) using raw states. Each executed action sequence is replayed independently in Gymnasium. Safety uses all 201 raw states and strictly negative margin indicates violation; every episode continues through 200 steps.

## Pilot results

| Original state ID / group | Initial theta, omega | PD return / discounted / min margin | Open-loop witnesses | Tracking witnesses |
|---|---|---|---:|---:|
| 96 / equilibrium | 0.000000000, 0.000000000 | 0.000000 / 0.000000 / 1.000000 | 3/3 | 3/3 |
| 2 / nominal_safe | 0.023232378, 0.223727468 | -0.028574 / -0.026971 / 0.925424 | 0/3 | 3/3 |
| 33 / challenging_safe | 0.150274077, 0.812635956 | -0.899955 / -0.837163 / 0.729121 | 0/3 | 3/3 |
| 32 / PD_unsafe | 0.795492365, -0.580330788 | -274.006377 / -188.815677 / -4.626068 | 0/3 | 0/3 |

A witness requires independent Gymnasium safety AND exact recorded-action replay. It is not inferred from solver success or the optimizer’s state trajectory.

| State | Start | Solver status | Dynamics residual | Mode | Return | Discounted return | Minimum raw margin | First crossing (s) | Max plan/replay angle, speed error |
|---|---|---|---:|---|---:|---:|---:|---:|---|
| 96 | PD | Solve_Succeeded | 0.00e+00 | open_loop | 0.000000 | 0.000000 | 1.00000000 | none | 0.000e+00, 0.000e+00 |
| 96 | PD | Solve_Succeeded | 0.00e+00 | tracking | 0.000000 | 0.000000 | 1.00000000 | none | 0.000e+00, 0.000e+00 |
| 96 | zero | Solve_Succeeded | 0.00e+00 | open_loop | 0.000000 | 0.000000 | 1.00000000 | none | 0.000e+00, 0.000e+00 |
| 96 | zero | Solve_Succeeded | 0.00e+00 | tracking | 0.000000 | 0.000000 | 1.00000000 | none | 0.000e+00, 0.000e+00 |
| 96 | random_blocks | Solve_Succeeded | 3.47e-23 | open_loop | -0.000000 | -0.000000 | 0.99999988 | none | 1.045e-07, 3.675e-07 |
| 96 | random_blocks | Solve_Succeeded | 3.47e-23 | tracking | -0.000000 | -0.000000 | 1.00000000 | none | 6.239e-22, 1.228e-20 |
| 2 | PD | Solve_Succeeded | 1.55e-13 | open_loop | -126.077105 | -36.974183 | -5.28358270 | 5.75 | 6.284e+00, 7.732e+00 |
| 2 | PD | Solve_Succeeded | 1.55e-13 | tracking | -0.012089 | -0.011931 | 0.92542418 | none | 4.663e-10, 2.171e-09 |
| 2 | zero | Solve_Succeeded | 1.65e-17 | open_loop | -125.922489 | -38.474960 | -5.28312730 | 5.55 | 6.283e+00, 7.741e+00 |
| 2 | zero | Solve_Succeeded | 1.65e-17 | tracking | -0.012089 | -0.011931 | 0.92542418 | none | 4.373e-10, 3.816e-09 |
| 2 | random_blocks | Solve_Succeeded | 3.12e-16 | open_loop | -125.692856 | -38.866892 | -5.28572838 | 5.45 | 6.286e+00, 7.755e+00 |
| 2 | random_blocks | Solve_Succeeded | 3.12e-16 | tracking | -0.012089 | -0.011931 | 0.92542418 | none | 6.880e-10, 5.007e-09 |
| 33 | PD | Solve_Succeeded | 4.42e-13 | open_loop | -126.477254 | -38.859249 | -5.31582547 | 5.55 | 6.316e+00, 7.756e+00 |
| 33 | PD | Solve_Succeeded | 4.42e-13 | tracking | -0.816196 | -0.769636 | 0.72912135 | none | 3.680e-09, 2.953e-08 |
| 33 | zero | Solve_Succeeded | 3.09e-13 | open_loop | -127.044955 | -36.295759 | -5.28561792 | 5.95 | 6.286e+00, 7.722e+00 |
| 33 | zero | Solve_Succeeded | 3.09e-13 | tracking | -0.816196 | -0.769636 | 0.72912135 | none | 2.294e-09, 2.953e-08 |
| 33 | random_blocks | Solve_Succeeded | 2.42e-13 | open_loop | -126.936252 | -39.255060 | -5.28289413 | 5.55 | 6.283e+00, 7.729e+00 |
| 33 | random_blocks | Solve_Succeeded | 2.42e-13 | tracking | -0.816196 | -0.769636 | 0.72912135 | none | 4.275e-09, 2.953e-08 |
| 32 | PD | Infeasible_Problem_Detected | 2.81e-01 | open_loop | -1162.820692 | -460.256062 | -3.52126281 | 0.45 | 4.521e+00, 5.094e+00 |
| 32 | PD | Infeasible_Problem_Detected | 2.81e-01 | tracking | -951.027533 | -406.273517 | -3.06850611 | 0.45 | 4.066e+00, 5.094e+00 |
| 32 | zero | Infeasible_Problem_Detected | 2.81e-01 | open_loop | -1089.922799 | -442.603158 | -4.51859540 | 0.45 | 5.523e+00, 7.944e+00 |
| 32 | zero | Infeasible_Problem_Detected | 2.81e-01 | tracking | -951.027533 | -406.273517 | -3.06850611 | 0.45 | 3.709e+00, 5.056e+00 |
| 32 | random_blocks | Infeasible_Problem_Detected | 2.81e-01 | open_loop | -984.570300 | -413.476428 | -10.98736104 | 0.45 | 1.185e+01, 8.755e+00 |
| 32 | random_blocks | Infeasible_Problem_Detected | 2.81e-01 | tracking | -952.119768 | -406.422332 | -3.06850611 | 0.45 | 4.274e+00, 5.680e+00 |

## Verification and costs

- Overall technical verification passed: True. 4 pure technical tests passed; no RL training was run.
- Feasible-domain one-step checks: 1000 random points plus 27 boundary cases; maximum symbolic/Gym discrepancy 1.192e-08 (tolerance 1e-7).
- In 4 outside-domain cases, speed clipping activated: the smooth model differed by up to 1.050000; the explicitly clipped symbolic model matched Gym to 1.192e-08. No feasible-domain equivalence was claimed outside its domain.
- Maximum constraint-Jacobian / cost-gradient / cost-Hessian central-difference errors: 3.043e-10, 2.192e-10, 5.751e-11.
- Verified state/action indexing, fixed exact initial states, executed float32 controls, all 201 raw safety measurements, preserved PD baseline equality, and bitwise replay for every evaluated trajectory.
- All repeated solver decisions bitwise identical: True. Solver statuses and all repeat records are preserved; timing-related failures are not treated as reproducibility success.
- 24 solver attempts including repeats; 592 total IPOPT iterations; 0.877 solver seconds and 21.002 worker-process seconds. External watchdog timeouts: 0.
- Actual Gymnasium use: 13831 steps and 1095 resets, including model tests, initial guesses, evaluation, and replay. Detailed categories and NLP evaluation counts are in interaction_costs.json.
- An independent post-run audit checked all 28 evaluated trajectories and recorded-action replays, recomputed rewards and raw margins, checked candidate control indexing and dynamics residuals, and confirmed that only IDs 96, 2, 33, and 32 ran. It used no additional simulator calls.
- The original PD evaluator emitted an expected sample-standard-deviation warning for one-episode calls; that unused aggregate was discarded. Every saved trajectory and reported per-episode metric is finite.
- All 6835 protected pre-existing files are SHA-256 unchanged. Experiments 01–06A and unfinished IQL diagnostics were preserved.

## Scientific interpretation and limits

Nine of the twelve original solves reported success: all three starts for equilibrium and both PD-safe cases. Nevertheless, all six non-equilibrium optimized open-loop executions violated safety, first crossing between 5.45 and 5.95 s, despite dynamics equality residuals at most 4.43e-13. Independent audits reproduced the actual simulator transitions and checked the applied action indices. The resulting large plan-versus-execution discrepancies are consistent with accumulation and amplification of small numerical differences; no claim is made that solver success certifies executable safety.

The fixed tracking diagnostic produced replay-verified safe sequences for all three starts at both PD-safe cases. Nominal return changed from -0.028574 under PD to approximately -0.012089, and challenging-safe return from -0.899955 to approximately -0.816196. Their minimum raw margins were 0.925424 and 0.729121, respectively. These are descriptive pilot results at states already known to admit safe PD trajectories, not evidence of recovery from PD failure or a validated learned controller.

At the one PD-unsafe pilot state, all three starts returned Infeasible_Problem_Detected, with maximum dynamics residual approximately 0.281335. Their bound-respecting optimizer states were therefore not feasible trajectories. All six executed candidates violated safety and no recovery witness was found (0/1 PD-unsafe states). This is an inconclusive search result, not a proof that this state is unrecoverable.

These results establish only which sampled initial states yielded a replay-verified, finite-horizon control-sequence witness within the fixed search. Optimization failures and missing witnesses are inconclusive about physical unrecoverability. The PD-safe cases already have a feasible baseline; optimizer failure there cannot make them unrecoverable.

Open-loop and tracking outcomes are separate. Tracking is an additional diagnostic mechanism using fixed gains, not a certified backup or safety filter. A stable-looking optimized state sequence can disagree with an executed open-loop trajectory because dynamics residuals, float32 torque arithmetic, and rounding accumulate. Those discrepancies are measured and do not override actual raw-state safety.

Best-of-start summaries in summary.json use observed minimum margin, then return, and are oracle diagnostics, not deployable policy performance. No safety, viability, robustness, or recoverability guarantee follows outside the exact sampled states, nominal discrete simulator, and 200-step horizon. Full-study execution, algorithm tuning, RL training, commits, and pushes remain outside this pilot.

## Reproduction and artifacts

The pilot runs with `.venv/experiment07/bin/python experiments/07_recoverability.py --pilot`; use `--resume` only for matching existing artifacts. CasADi was installed in an isolated virtual environment using the existing packages without altering them. `requirements_experiment_07.txt` records the dependencies. The entry point has no full-study mode.

Solver logs include all IPOPT options/defaults, actual solver versions, statuses, and timings. Each attempt retains its initial guess, last finite iterate, final candidate, iteration trace, and repeat. `candidate_metrics.csv`, `summary.json`, `verification.json`, and `independent_verification.json` preserve detailed outcomes. The raw-margin plot uses black for the original PD baseline and colored curves for the three optimization starts. No unsuccessful candidate, divergent open-loop execution, or negative finding was omitted.
