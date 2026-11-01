# Experiment 10 — Safe Policy Improvement Feasibility: full study

Primary LQR-versus-PD result: mean return gain +0.056165740, threshold 0.136939167034; violations 0/3331; minimum paired return difference +0.000000000; maximum observed per-state margin loss -0. Frozen practical reward criterion passed: **False**.

## Protocol and provenance

All 3,331 original archived PD-safe initial states were evaluated with ten frozen controllers, identical original observation/action dtypes, torque bounds ±2, 200 transitions and all 201 raw states. PD uses the verified E09 baseline archive; accepted pilot controller evidence and 12 original pilot solves are reused. The exact 64-state optimization subset, four cost strata, PCG64 seed 101001, grid-ID tie-breaking, LQR gain, objective, constraints, three starts, random streams, solver/time limits and practical criteria are unchanged. Remaining 60 states used 180 new original solves; no new solver-repeat starts or training occurred.

Worst-case deterioration means the minimum per-state paired return difference (required >=−0.10). A practical reward success also needs mean gain >=max(0.10, 10% of PD mean cost), zero violations, and no per-state margin loss >0.01. No negative margin is accepted. Effort and margin criteria remain separate and unchanged. All conditions, including unsafe executions and unsuccessful optimizations, remain represented.

## Complete fixed-controller cohort

| Controller | Mean return | Worst return | Mean discounted return | Mean gain vs PD | Minimum paired gain | Mean effort | Minimum raw margin | Violations | Reward criterion |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| PD_6_1 | -5.622538519 | -734.902882275 | -3.351177190 | -4.253146848 | -732.347581378 | 1.541461281 | -5.375291656 | 38/3331 | False |
| PD_6_2 | -4.986469735 | -745.105389479 | -2.875934670 | -3.617078065 | -737.586821637 | 1.936645622 | -4.609892965 | 14/3331 | False |
| PD_6_4 | -193.081566967 | -1137.017522717 | -75.004776507 | -191.712175297 | -1131.112513631 | 9.462684441 | -3.560276964 | 588/3331 | False |
| PD_8_1 | -3.970299268 | -682.053293378 | -2.693392620 | -2.600907597 | -680.266683621 | 1.540603780 | -5.443468688 | 24/3331 | False |
| PD_8_2 | -1.369391670 | -7.518567842 | -1.303757284 | +0.000000000 | +0.000000000 | 0.888159542 | 0.200000000 | 0/3331 | False |
| PD_8_4 | -71.978837341 | -1014.902149512 | -27.913608283 | -70.609445671 | -1009.823524324 | 4.358987509 | -3.951166620 | 246/3331 | False |
| PD_12_1 | -11.010513738 | -808.639744924 | -5.819362296 | -9.641122068 | -806.369357231 | 2.764369561 | -5.553316577 | 80/3331 | False |
| PD_12_2 | -1.404172618 | -7.479826538 | -1.345217035 | -0.034780947 | -0.356707047 | 1.057094601 | 0.200000000 | 0/3331 | False |
| PD_12_4 | -1.366133339 | -7.487630479 | -1.298390351 | +0.003258331 | -0.194698301 | 0.978436106 | 0.200000000 | 0/3331 | False |
| LQR | -1.313225930 | -7.435439946 | -1.257373558 | +0.056165740 | +0.000000000 | 0.945293388 | 0.200000000 | 0/3331 | False |

| Controller | Mean RMS torque | Max peak torque | Mean saturation | Mean total variation | Effort reduction vs PD | Worst margin change | Effort criterion | Margin criterion |
|---|---:|---:|---:|---:|---:|---:|---|---|
| PD_6_1 | 0.337254455 | 2.000000 | 0.012573 | 1.894191683 | -0.735568 | -5.837650765 | False | False |
| PD_6_2 | 0.381966399 | 2.000000 | 0.009592 | 1.850187396 | -1.180515 | -4.868560519 | False | False |
| PD_6_4 | 0.769668667 | 2.000000 | 0.173257 | 8.293278320 | -9.654262 | -4.098327643 | False | False |
| PD_8_1 | 0.354482578 | 2.000000 | 0.018043 | 3.287875489 | -0.734603 | -5.876802021 | False | False |
| PD_8_2 | 0.251424853 | 2.000000 | 0.010802 | 1.406280551 | 0.000000 | +0.000000000 | False | False |
| PD_8_4 | 0.487495108 | 2.000000 | 0.081516 | 5.207497416 | -3.907888 | -4.347552304 | False | False |
| PD_12_1 | 0.469138047 | 2.000000 | 0.046328 | 6.248171337 | -2.112470 | -5.958061734 | False | False |
| PD_12_2 | 0.291548050 | 2.000000 | 0.018400 | 2.460175084 | -0.190208 | -0.016119942 | False | False |
| PD_12_4 | 0.280943814 | 2.000000 | 0.015158 | 1.992337059 | -0.101645 | +0.000000000 | False | False |
| LQR | 0.267535765 | 2.000000 | 0.020615 | 1.836773657 | -0.064328 | +0.000000000 | False | False |

Margin-eligible baseline states: 98/3331; the identical initial state caps attainable episode minimum margin. Per-state paired differences, crossing events, effort, reward components and terminal diagnostics are retained in fixed_metrics.csv. Total variation excludes the separately recorded initial-action jump; squared torque is an effort proxy, not physical energy.

## Prespecified 64-state optimization sample

Original scientific solver statuses (one status per solve): {'Solve_Succeeded': 192}. Solver success is distinct from a safe executed trajectory. Open-loop and tracking are independently executed/replayed with actual float32 torques. All 192 original scientific starts remain in the record; the accepted 12 pilot repeat solves remain separate historical technical evidence.

| Oracle mode | Safe original executions | Genuine optimized improvements over PD | PD fallbacks | Mean gain vs PD | Minimum gain vs PD | Mean gain vs LQR | Minimum gain vs LQR | Optimized gains vs LQR >1e−6 / >0.01 / >0.1 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| open_loop | 0/192 | 0/64 | 64/64 | +0.000000000 | +0.000000000 | -0.053746854 | -0.084512741 | 0 / 0 / 0 |
| tracking | 192/192 | 64/64 | 0/64 | +0.053754221 | +0.000943436 | +0.000007367 | -0.000000010 | 31 / 0 / 0 |

Oracle outcomes use the best safely executed original candidate only when it improves PD return; otherwise the source is explicitly PD fallback, which is not an optimization success. These are state-specific outcome-informed comparisons, not results for a deployable universal controller. Comparisons against LQR use the same 64 states and preserve LQR safety flags. Missing finite candidates are unresolved numerical cases, never evidence of optimality. All-start results and oracles are in optimization_metrics.csv and oracle_metrics.csv.

Pilot-only diagnostic before scaling found tracking-minus-LQR gains between −1.02116e−9 and +1.34021e−6, with maximum torque differences up to 0.0130317 N·m. It did not motivate any method or threshold change. Full-sample return and executed-torque difference distributions are recorded in the oracle table and summary.json.

## Verification and computational costs

Verification passed: True; protected pre-existing files: 45126. Independent fixed replay pairs: 33310; optimization replay pairs: 384; raw samples reconstructed: 6772494. Rewards, discounted sums, effort, torque sequences, raw margins and crossing events were independently reconstructed. Technical tests: {'passed': True, 'tests': 4, 'failures': 0, 'errors': 0, 'simulator_steps': 0, 'solver_executions': 0}. All accepted pilot evidence, full scientific inputs and prior results passed preservation/source hashes.

New execution costs: {'fixed_new_steps': 11973600, 'fixed_new_resets': 59868, 'fixed_new_wall_seconds': 162.45861624449026, 'reused_E09_PD_conditions': 3331, 'reused_pilot_nonbaseline_conditions': 45, 'new_solver_executions': 180, 'new_solver_seconds': 3.799599203863181, 'new_worker_process_seconds': 157.67202079597337, 'new_solver_iterations': 2444, 'new_optimization_steps': 168000, 'new_optimization_resets': 840, 'new_optimization_wall_seconds': 1.7177060882968362, 'new_watchdog_timeouts': 0, 'new_technical_worker_failures': 0, 'reused_original_pilot_solves': 12, 'retained_historical_pilot_repeat_solves': 12, 'scientific_original_solves': 192}. These exclude historical reused executions, saved-array audit and file handling. Timings are descriptive, not new independent observations. Numerical execution diagnostics: {'open_loop': [7.048866955509343, 7.758198846584366], 'tracking': [5.055059038028276e-09, 3.352014443791518e-08]}; NLP residual range: {'count': 384, 'mean': 1.312556450066132e-12, 'min': 6.938893903907228e-18, 'max': 1.5671131059491472e-11, 'median': 1.9373391779708982e-14}. Any timeouts, technical failures, missing candidates and failed checks are retained.

## Scientific interpretation and limitations

Judge LQR against every frozen practical criterion rather than mean return alone. Positive state-specific optimization gains do not imply deployable controller improvement, and PD fallback must not be counted as optimized success. Local solver convergence and small residuals cannot prove global optimality; open-loop failures can reflect numerical sensitivity. The selected 64 points are a prespecified stratified sample for diagnostics, not a proof over the continuous domain or a globally optimal reference.

The 3,331-point cohort has empirical PD safety, not a certified continuous viable region. All claims concern this fixed nominal simulator, exact initial states and 200-step sampled horizon. No formal robustness, continuous-time, model-error, hardware or infinite-horizon guarantee is established. Reward, actuator effort and safety margins can trade off. Deterministic repeats are verification, not independent statistical samples. No favorable seed selection, tuning, reward change, new controller, RL training, IQL work, commit or push occurred. Any future learning or acquisition design requires review of these results first.
