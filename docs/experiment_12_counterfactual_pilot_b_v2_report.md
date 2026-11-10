# Experiment 12 — Counterfactual Pilot B v2

**Result: PASS for the frozen one-state technical pilot.** Both nominal runs and both menu-3 alternative runs completed with all eight replays. The alternative admits one reproducible empirically safe six-second continuation; it worsened cumulative return. No other action probe or controller was executed.

## Frozen context and execution fidelity

Only manifest ID 0 at reference step zero was used. Native MPC80, the six-second 50/1,000-Hz task, upstream revision `6b5391d014f36fdfa0f9d22d92c77387e5274308`, CPython 3.10.19, source assets, optimization program, solver settings, constraints, warm starts and acceptance thresholds matched the reviewed seal `cfb6a56fd40158f26602d536bab80dd9636467f92b5912aed3721605874f8126`.

The twelve jobs ran in frozen order, one worker. The six-job nominal archive/repeat/replay gate passed before any alternative reservation. No jobs failed or were incomplete; there were no retries, fallback controls or parameter changes. The sealed 173-test wiring evidence was reused; execution source code was not edited.

Both nominal runs reproduced accepted scientific arrays and all 300 primal, dual and parameter vectors bitwise. Requested/clipped/RPM/force/torque submissions matched bitwise in physical columns; transient body/client identifiers are metadata. Every reference preview, initial guess and prescribed native x/u memory shift was independently checked. The original first query ran once and was verified before detached substitution. Its nominal post-query plans/memory were preserved. Only the first 20-ms action interval changed; MPC80 continued unchanged for steps 1–299 using its new observations. Both branch repetitions were bitwise identical in scientific data and candidates.

Each scientific run contained 300 controls and all 6,001 raw physics states, including reset and terminal states. All 1,200 solves returned Solve_Succeeded (maximum 20 iterations). Every candidate passed corrected v2 KKT, finite-value and program checks during execution and independent saved-record audit. There were zero raw state/input violations and zero contacts. Uninstrumented native replays matched 301 control states, rewards and end-interval RPM. Independent motor replays also matched all 6,001 raw states/contacts and direct submissions. Do not infer full-rate capture from the uninstrumented replay alone.

Frozen requested alternative: `[0.056324369506392545, 0.13664132686280034]` N. Direct recorded grouped-force sums: `[0.056324369506392545, 0.1366413268628003]` N, exactly matching the sealed bytes. Actual normalized separation was `39363019590861/787260391817216 = 0.05000000000000025`, satisfying >=5% without tolerance adjustment. First-interval RPM, force and tiny nonzero yaw-torque bytes matched. These are native API command submissions, not physical force sensors. Separately reconstructed RPM-derived thrusts differed from direct force sums by zero on these records.

All 36 original labels remain explicit: two executed labels, nine source-predicted input-inadmissible labels NOT RUN, and 25 other admissible labels NOT RUN. Repetitions do not increase the label denominator.

## Matched scientific outcomes

Original/repeat outcomes are identical within each branch. This is one initial state with deterministic technical repetitions, not independent samples or a cohort estimate. No confidence interval is appropriate. Returns use the native quadratic reward; discounted return uses gamma=0.99. Effort is the executed-thrust squared integral (dt=0.02 s), in N² s.

| Metric | Nominal menu 4 | Alternative menu 3 | Alternative minus nominal |
|---|---:|---:|---:|
| Cumulative return | -6.26232336758 | -6.27648132969 | -0.0141579621107 |
| Discounted return | -5.36047163172 | -5.36044252209 | +2.91096340295e-05 |
| Position RMSE (m) | 0.040699194783 | 0.0412281977927 | +0.000529003009727 |
| Terminal position-error norm (m) | 0.105803526763 | 0.105803530016 | +3.25366611431e-09 |
| Absolute squared-thrust effort (N² s) | 0.228686296728 | 0.228726280876 | +3.99841488834e-05 |
| Hover-relative squared-thrust effort (N² s) | 0.0145235175216 | 0.014557121732 | +3.36042103857e-05 |

State order: (x, xdot, z, zdot, theta, thetadot), units m, m/s, m, m/s, rad, rad/s. Margins are the closer raw lower/upper bound over all physics samples; normalization uses the full component range. Reserves use the actual action and causal raw/cache context.

| Component | Nominal raw margin | Alternative raw margin | Nominal normalized | Alternative normalized | Nominal reserve | Alternative reserve |
|---|---:|---:|---:|---:|---:|---:|
| x | 1.238939171 | 1.238838761 | 0.3097347927 | 0.3097096903 | 1.238837139 | 1.23873683 |
| xdot | 29.13636683 | 29.13457045 | 0.4856061138 | 0.4855761742 | 29.13635272 | 29.13455795 |
| z | 0.624384162 | 0.6242726929 | 0.3045776399 | 0.3045232647 | 0.6241790252 | 0.62407244 |
| zdot | 28.03364365 | 28.03826156 | 0.4672273941 | 0.4673043594 | 28.0331525 | 28.03773608 |
| theta | 1.222989474 | 1.222299687 | 0.4121890469 | 0.4119565652 | 1.222722106 | 1.22192711 |
| thetadot | 5.01996558 | 5.450830367 | 0.2876228357 | 0.312309569 | 5.000901002 | 5.45057296 |

Minimum submitted-input margins (N), nominal: [9.939534998601451e-07, 1.0016499831719261e-06]; alternative: [9.922343503498965e-07, 1.0006000933787673e-06]. All margins/reserves are strictly positive. Angular-rate margin improved while several other component margins worsened; this does not establish improved overall safety.

Terminal error vectors in the same state order/units:

- Nominal: `[-0.08762458611008783, -0.39588997012613764, -0.05929855128485895, -0.39780091816144947, -0.09841731231268162, -0.046428366414330055]`.
- Alternative: `[-0.08762458998589981, -0.39588996245194, -0.059298551362987784, -0.3978009210944587, -0.09841730836836897, -0.04642839702531926]`.

The alternative's cumulative-return difference is -0.014157962111; its discounted-return gain is only +0.000029109634. Position RMSE and both effort measures increase. This supports a distinct empirically safe action with continuation, not meaningful reward headroom or beyond-expert improvement. No best-of-menu/oracle search occurred. The original >=10% practical improvement and three-of-four action-support criteria are neither established nor replaced by this pilot.

## Numerical agreement and computational reliability

Original overall E12 qualification and 0.001 symbolic/PyBullet agreement remain **FAILED**. Maximum symbolic one-step discrepancies were 0.009523534792 nominal and 0.008282330205 alternative. Maximum source closure was 1.7763568394002505e-15; maximum contract excess was 3.1168506941154945e-16, within unchanged 1e-10 closure tolerance. Positive reserves are numerical diagnostics, not a general PyBullet floating-point certificate, robust safety theorem or hardware safety evidence.

KKT maxima over all 1,200 candidates: `{'primal_inf': 7.884274426771708e-09, 'stationarity_inf': 5.2512732651860405e-09, 'dual_sign_inf': 0.0, 'complementarity_inf': 9.866945440860662e-09, 'normalized_complementarity_inf': 9.866253644841711e-09}`.

Latency in ms; overruns count strict >20 ms, with no dropped samples. Solver-only/native compute exclude separately recorded verification/persistence overhead; supervised latency includes the wrapper.

| Branch/run | Timing | Median | p95 | p99 | Max | >20 ms /300 |
|---|---|---:|---:|---:|---:|---:|
| menu 4 original | solver_only | 21.896 | 40.310 | 46.007 | 772.764 | 300 |
| menu 4 original | native_compute | 22.699 | 41.114 | 46.742 | 773.481 | 300 |
| menu 4 original | supervised | 89.142 | 102.933 | 142.317 | 830.185 | 300 |
| menu 4 repeat | solver_only | 22.571 | 41.595 | 46.861 | 141.746 | 300 |
| menu 4 repeat | native_compute | 23.446 | 42.389 | 47.709 | 142.633 | 300 |
| menu 4 repeat | supervised | 93.648 | 106.538 | 140.580 | 229.295 | 300 |
| menu 3 original | solver_only | 22.588 | 41.684 | 44.250 | 124.734 | 300 |
| menu 3 original | native_compute | 23.458 | 42.558 | 45.033 | 125.525 | 300 |
| menu 3 original | supervised | 90.224 | 104.809 | 133.131 | 211.556 | 300 |
| menu 3 repeat | solver_only | 23.234 | 41.792 | 44.321 | 140.425 | 300 |
| menu 3 repeat | native_compute | 24.452 | 42.577 | 45.119 | 141.331 | 300 |
| menu 3 repeat | supervised | 91.397 | 116.637 | 157.996 | 202.127 | 300 |

All 1,200 calls exceeded 20 ms in each category; hard real-time readiness is not established. Cold-start solver maximum was 0.772764334 s; supervised maximum was 0.830184708 s, below the unchanged one-second watchdog. Solver/native/supervised telemetry is complete; separate pure physics-integration time is not identifiable. Replay, verification, journaling and hashing are included in aggregate charges, not reported as solver time. All latency vectors and overrun flags are saved in `all_latency_observations.npz`.

Maximum scientific RSS: 464,879,616 bytes, below 4 GiB. Work was sequential. No optimizer or physics calls occurred during post-run audits.

## Resource and preservation accounting

Starting historical charge: 1,889.205 CPU /2,638.108 active-wall seconds. Launch including preparation, workers, replays and initial conservative inspection charge: 178.509 CPU /185.218 wall seconds. Independent audit: 6.654 CPU /8.464 wall; preservation: 19.367 CPU /47.824 wall. Final reporting/hashing/metadata: 3.876 CPU /14.385 wall (includes conservative 2 CPU/10 wall seconds for remaining reporting/sealing and metadata work, including the failed report-file construction command; no scientific execution failed).

Final cumulative charge: **2097.611 CPU /2893.999 active-wall seconds**, within approved 2,189.205 /3,238.108. Remaining: **91.594 CPU /344.109 wall seconds**. All 1,200 solves /72,000 physics steps were reserved and used. Remaining original allowance: **1,870 solves /294,000 physics steps**. No historical charges or reservations were reset/refunded.

Independent preservation passed: 227 tracked files, 289 pinned upstream files/runtime identities, accepted preflight/wiring seals, 87,315 accepted MPC80 artifacts, 59,268 earlier result/data entries, 87 drilling sources, 50 restoration artifacts, decision history and nine unchanged untracked IQL files. Original failed native linear/qualification/model-agreement outcomes, v1 instrumentation failure, stopped Pilot B and accepted MPC80 feasibility evidence remain unchanged. No controller/source code was edited, committed or pushed.

## Scientific interpretation and scope

This pilot validates matched-context execution fidelity and one empirically safe non-nominal action/continuation at ID 0. It does not establish population safety, robust/physical safety, three-of-four action diversity, meaningful performance headroom, safe policy improvement or adequate offline counterfactual support. No other menu labels, cohort states, acquisition methods, RL or IQL were evaluated. Stop here for review; expanded studies require separate approval and prospective budgets.

Detailed trajectories, both replay types, complete candidate/history/telemetry logs, independent audit, all 36 labels, latency vectors, ledger and artifact hashes are under `results/benchmark_selection/counterfactual_pilot_B_v2/`; launch/audit scripts and authorization are in the sibling `_authorization/` directory. Generated evidence remains outside Git. This report is a new document; prior reports and decision history are unchanged.
