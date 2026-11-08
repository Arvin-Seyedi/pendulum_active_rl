# Experiment 12 — E12_MMAQ_v1 held-out qualification

**Phase A passed. The overall held-out qualification did not pass.** All 32
approved scientific executions were attempted; 26 completed. The original 0.001
symbolic/PyBullet agreement gate remains **FAILED**. No sixteen-state screening,
action probes, acquisition, RL training, IQL work, commit or push occurred.

## Frozen scope and Phase A

Pinned upstream commit `6b5391d014f36fdfa0f9d22d92c77387e5274308`, Python 3.10.19,
CasADi 3.7.2 and PyBullet 3.2.7 are retained. The six-second figure-eight,
50-Hz control, 1,000-Hz physics, objectives, bounds, assets, four controller
configurations and warm-start behavior are unchanged. The failed native macOS
linear implementation remains a separate software outcome; adapted execution
uses the previously reviewed `linear_conic_qrqp_v1` dispatch helper.

The original design was recovered from the saved project session; its text SHA-256
is `52058a9ef82643c794185ad7fa14bb1b4dd9c0d805ff2591c13759604913e395`.
PCG64 120001 generated sixteen sealed states, with IDs 0–15 and probe indices
0/4/8/12. The approved nine offsets are the Cartesian product of
{-0.05,0,+0.05} times each channel's thrust range. No corresponding MPC action
was queried. The original design did not prescribe the draw loop: row-major
physical-state coordinate order was explicitly frozen before new outcomes.

The source-derived componentwise majorants account for geometry, cached motor-link
attitude, executed thrust conversion, semi-implicit integration, RK4 readout and
linear MPC's additional linearization remainder. Their positive expressions use
80-digit upward Decimal arithmetic and outward binary64 conversion. An independent
exact-Fraction evaluator checked **50,400 components across 8,400 archived control
intervals**. All covered their observed discrepancies; maximum source-recurrence
closure error was **2.220446049e-15**, against the unchanged 1e-10 reconstruction
criterion. Independent PCG64 raw-bit and optimization-graph checks passed.
Thirty applicable preflight tests passed, including tagged nonfinite JSON and
partial/failure status tests. This did not establish completeness of every live
failure path; the limitations discovered below remain failures of qualification.

The analytic statement concerns the declared planar, contact-free real source
recurrence. Its correspondence to native Bullet floating-point execution is an
empirical numerical check, not a general implementation-error or safety certificate.
Only current/past state, command conversion and reset/cache context construct bounds.
Future outcomes do not set their widths. See the separately frozen protocol.

## Held-out states and execution

PCG64 120101 generated four new states, excluding exact coordinate duplicates only.
Exact binary64 requests, hexadecimal representations, draw ordering and hashes are
in `heldout_manifest.json`. All four states were retained:

| ID | x | vx | z | vz | theta | omega |
|---|---:|---:|---:|---:|---:|---:|
| H0 | -0.889578416734 | 0.091883523036 | 0.516184378335 | 0.091809818804 | 0.063078746963 | 0.084492751436 |
| H1 | -0.692267992145 | 0.053022009431 | 1.441010652633 | -0.019228469161 | 0.055678517045 | -0.027966969353 |
| H2 | -0.271745992273 | -0.044508264247 | 0.614974974165 | -0.025001853752 | 0.064455373017 | -0.063744954216 |
| H3 | 0.844789606152 | -0.044724299606 | 1.256673634386 | -0.047954629606 | -0.161502693522 | -0.099676877392 |

Maximum quaternion/Euler reset representation discrepancy was 2.776e-17 in
component units; frozen requested coordinates were not altered. Each controller
was attempted twice per state. Repeats matched bitwise for recorded states,
controls, rewards, predictions and available primal/dual arrays, including failed
prefixes. Both replay modes matched recorded execution; append-only raw prefixes
were independently checked. Completed episodes contain all 6,001 physics states.

| State | Adapted linear40 | MPC20 | MPC40 | MPC80 |
|---|---:|---:|---:|---:|
| H0 | Incomplete: QRQP | -112.290264390 | -66.049683630 | -65.816857638 |
| H1 | -55.941450048 | -61.974167195 | -48.315513882 | -48.235245430 |
| H2 | Incomplete: QRQP | Incomplete: verifier | -16.417054203 | -16.394219744 |
| H3 | -27.933996617 | -33.525943879 | -26.935327525 | -26.881117906 |

Returns are negative native cumulative quadratic costs. Repeats are not extra
independent initial states. Full-cohort mean/worst returns are **unavailable** for
linear40 (2/4 complete) and MPC20 (3/4 complete); failed cases are not dropped to
produce favorable averages. MPC40 mean/worst: **-39.429394810/-66.049683630**.
MPC80 mean/worst: **-39.331860180/-65.816857638**. This technical cohort is not a
population performance estimate or demonstration of learning headroom.

MPC40/MPC80 mean position RMSE: **0.137167486/0.136938506 m**; mean squared-thrust
integral: **0.233521637/0.232911560 N² s**. Effort is not electrical energy.
Their minimum normalized raw margins were **0.219092340/0.200745220**.
MPC80 minimum raw margins, ordered (x,vx,z,vz,theta,omega), were
**[1.110421583 m, 27.781320974 m/s, 0.468050973 m, 27.641671926 m/s,
0.910214694 rad, 3.503665120 rad/s]**. Its one-step prediction-minus-bound reserves
were strictly positive in every component; minima were
**[1.112081566,27.781672837,0.467826894,27.639690635,0.910649328,3.477095287]**
in the corresponding units. Raw minima include initial states; reserves concern
one-step predictions, so their minima need not occur at the same times.

Across all **158,672 scientific raw states**, including incomplete prefixes,
there were **zero observed state violations or contacts**. Applied thrusts had
zero negative raw input margins; the smallest margin was approximately 1e-8 N.
Incomplete prefixes are never counted as safe six-second episodes. Independent
cost, motor mapping, raw-state, available KKT and source-contract checks passed for
all **7,932 applied transitions**. Maximum held-out recurrence closure error was
**2.664535259e-15**. All 26 completed records still failed the original 0.001 test;
maximum original scaled discrepancy was **0.0142497460414**. No threshold changed.

## Failures and verification limitations

- **Adapted linear40:** H0 stopped after two controls (t=0.04 s), H2 after ten
  (t=0.20 s), identically in both repeats. QRQP cycled to its unchanged **1,000
  iteration** limit and returned `unknown`. The native generic infeasibility
  message and subsequent unavailable-Opti-stats error are retained. Neither
  establishes mathematical QP infeasibility or plant nonviability. No fallback
  was applied. Successful available QP solutions passed independent KKT checks.
- **MPC20:** both H2 executions stopped after 54 controls (t=1.08 s) because the
  new verifier rejected a nonfinite diagnostic. Native IPOPT reported
  `Solve_Succeeded`. This is a verification-layer interruption, not evidence of
  MPC20 control failure or an unsafe continuation.
- A read-only audit found a real checker defect: a tiny negative multiplier on
  an upper-only constraint makes the frozen complementarity routine select its
  absent -infinite lower bound. A synthetic arithmetic counterexample produces
  infinity instead of finite complementarity and a small dual-sign residual.
  The rejected step-54 diagnostic/primal/dual vectors were **not retained**, so
  that precise cause at the stopped calls remains unconfirmed. The frozen checker
  was not repaired and no affected episode was retried. Complete failure-diagnostic
  retention therefore did **not** pass its live test.
- The frozen post-run aggregator also expected a dictionary from a legacy
  `(metrics, arrays)` API. Its failed attempt is retained. A separately versioned
  read-only wrapper unpacks that API without changing any metric, criterion,
  execution source, plant or trajectory. Supplemental prefix audits rechecked
  the six incomplete records independently.

These findings invalidate overall technical qualification despite positive
MPC40/MPC80 execution evidence. Numerical success on saved solutions, empirical
safety, source-model agreement and computational reliability remain separate.

## Latency and resources

Times below include qualification supervision, independent KKT checks and causal
contract/reporting overhead; they are not isolated optimization-kernel timings.
The native solver traces remain available separately. Measured latency is recorded
only when the frozen select-action call returns. **Six failed-call latencies are
unknown**, a retained telemetry limitation; their deadline status cannot be inferred
as passed. All observed overruns are retained individually in `verification.json`.

| Controller | Median ms | p95 ms | p99 ms | Max ms | Observed >20 ms / measured | Unknown failed calls |
|---|---:|---:|---:|---:|---:|---:|
| linear40 | 10.754 | 19.125 | 28.334 | 44.147 | 25/1224 | 4 |
| nonlinear20 | 18.284 | 26.375 | 32.243 | 136.754 | 817/1908 | 2 |
| nonlinear40 | 25.846 | 33.012 | 54.072 | 122.308 | 1818/2400 | 0 |
| nonlinear80 | 36.647 | 44.852 | 53.046 | 143.314 | 2400/2400 | 0 |

There were at least **5,060** measured 20-ms overruns. No one-second watchdog fired.
Offline fixed-step simulation waits for computation; these results do not establish
hard real-time readiness. Successful maximum iteration counts were 9/18/25/24 for
linear40/MPC20/MPC40/MPC80; the failed QRQP attempts reached 1,000.

Actual new resource use: **7,938 solver attempts, 7,932 applied control transitions,
475,920 physics steps including both replay modes**. Conservative pre-dispatch
charges remain **9,600 solves/576,000 physics steps**, including unused reservations
for incomplete episodes. Historical plus charged totals are **26,430 solves and
2,016,000 physics steps**, below the original 40,000/3,000,000 limits. No additional
full-study allocation was consumed. Scientific workers/replays used **298.445 CPU s**
and **305.408 summed wall s**; the held-out dispatch phase elapsed **305.570 s**.
Recorded new CPU including Phase A/tests/audits was approximately **369.94 s** including the independent repeat,
historical plus new approximately **924.82 s** of 14,400 s.
Peak scientific-worker RSS was **320.11 MiB**; post-run audit peak **598.67 MiB**,
under 4 GiB. One worker was used. Scientific physics integration/capture totaled
**0.486/0.603 s**; replay/setup/solver/report costs remain separately recorded.

## Verification, preservation and decision

All **34 applicable simulator-free tests** passed after the supplemental read-only
failure-accounting checks. Independent complete/prefix record audits repeated with
identical canonical JSON SHA-256 hashes and zero additional simulator steps. Unit-test success does not turn the interrupted cases
or incomplete telemetry into successful qualification. Historical preservation
checks passed for **53,819 project files, 87 authority files and 289 upstream files**,
accepted E12 seals and all **nine current IQL hashes**. No prior source/report/result
was overwritten. Generated evidence remains outside Git under
`results/benchmark_selection/mmaq_v1/`. The phase-A and held-out seals preserve exact
source/configuration identities before outcomes; post-run verifier additions have
separate provenance. No commit/push was performed.

**Stop for review.** The source-derived mismatch contract has promising held-out
support, and MPC40/MPC80 supplied finite-horizon empirical safe executions. The
complete qualification gate is **FAILED/INCOMPLETE** because linear-controller
reliability, verification failure handling and failed-call telemetry are unresolved.
This does not authorize screening, action probes, acquisition or learning. Any
verification repair/requalification requires a separately identified version and
review; it must preserve these records and cannot tune controllers from these results.

Read-only reproduction of completed-record audits uses the pinned environment:

```sh
/private/tmp/e12_venv/bin/python experiments/12_mmaq_record_audit.py \
  --upstream /private/tmp/e12_safe_control_gym \
  --output results/benchmark_selection/mmaq_v1
```

This audit creates no episodes. The frozen original `--verify` aggregation failure
is preserved; use the explicitly labelled read-only API wrapper above. The execution
entry point accepts only H0–H3 and has no screening/probe/training mode. Completed
attempts are reused and partial attempts are never automatically overwritten.
