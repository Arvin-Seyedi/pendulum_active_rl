# Experiment 12 — Targeted H2/MPC20 v2 requalification

**The new targeted requalification PASSED.** Both original and repeat completed
six simulated seconds, 300 applied controls and all 6,001 physics states. No
numerical controller failure or v2-verifier failure occurred. Historical v1 H2
records remain incomplete verifier rejections; the original missing vectors were
not backfilled. Overall benchmark qualification remains **FAILED**, including the
unchanged original **0.001 symbolic/PyBullet agreement criterion**. This result
authorizes no screening, action probes, acquisition or training.

## Frozen protocol and preflight

Only MPC20 at the original held-out H2 was evaluated. The requested state is
`[-0.2717459922733778, -0.044508264247253675, 0.6149749741645008,
-0.025001853752066136, 0.06445537301651993, -0.06374495421574869]`.
Its exact binary64/hex representation is sealed in `initial_state.json` and matches
row 2 of the accepted held-out array. No state was resampled.

Upstream commit `6b5391d014f36fdfa0f9d22d92c77387e5274308`, the pinned Python 3.10.19
and dependency environment, native MPC20/IPOPT settings, objective, constraints,
dynamics, reference, warm starts (including the native row-shift convention),
seed 1337, 50-Hz control and 1,000-Hz physics remain unchanged. No linear controller
was instantiated for an execution and no QRQP option/limit changed.

Before execution, all **63 focused tests** passed. Preflight independently
rechecked **7,932 retained historical candidates** with v2 finite-side KKT
mathematics. All 54 archived H2 candidates were also substituted into the actual
native graph; maximum independent/native difference was **7.105427358e-15**.
The native candidate extractor captured all mandatory fields. Rejected/mismatched,
nonfinite and failed-call tests retained raw candidates, diagnostics, status,
partial prefixes and timing without fallback actions.

Preflight patched actual optimization and physics stepping to fail if called:
**zero solves and zero physics steps** occurred before its pass. Source/configuration,
initial-state and program hashes were sealed first. The unchanged native graph
SHA-256 is `097a983684b181ba458f8a3929c88d135fa423fd80696be3401bc548ecc65a49`.
The new path is separately versioned `E12_H2_MPC20_requalification_v2`; accepted
v1/v2 verifier, controller, telemetry and historical result sources were not edited.

## Numerical and empirical results

Both repetitions produced identical state/action/reward/prediction/iteration
histories and primal/parameter/dual arrays, bitwise. Native warm-start checks have
zero error. The original requested initial state resets with zero recorded
representation error for H2. Each execution's uninstrumented replay and independent
recorded-motor replay reproduced its recorded results; the latter checks all raw
states, contacts and actual four-motor commands. Append-only raw streams match the
complete trajectory records.

| Quantity | Each execution |
|---|---:|
| Applied controls / raw physics states | 300 / 6,001 |
| Return | -25.138729807870 |
| Discounted return, gamma=0.99 | -18.694905608520 |
| Position tracking RMSE | 0.101105482 m |
| Squared-thrust integral | 0.242895121 N² s |
| Thrust total variation | 1.839951718 N |
| Minimum normalized raw state margin | 0.269988235 |
| State/input violations or contacts | 0 / 0 / 0 |
| Maximum IPOPT iterations | 17 |

Squared thrust is an effort proxy, not electrical energy. Mean/worst return across
the two identical executions is the same number; these are repetitions at **one
initial condition**, not independent population samples.

Minimum raw margins in state order (x,vx,z,vz,theta,omega) are
**[1.079952941 m,28.407665276 m/s,0.615840249 m,27.587954595 m/s,
1.070104394 rad,6.605141414 rad/s]**. Minimum actual applied input margins were
approximately **9.904e-7 and 9.905e-7 N**. All raw values, including initial and
terminal states, were checked without rounding negative margins into safety.

All **600 candidate records** passed independent native/program reconstruction
and corrected KKT checks. Maximum residuals across the runs:

| Residual | Maximum |
|---|---:|
| Primal feasibility | 9.619895185e-9 |
| Stationarity | 4.931490272e-9 |
| Dual-sign error | 1.094061792e-9 |
| Raw complementarity | 4.501724972e-8 |
| Normalized complementarity | 9.790449116e-9 |

The **unchanged** acceptance criterion uses normalized complementarity <=1.01e-8;
raw complementarity is retained separately. No tolerance was increased. Native
status was Solve_Succeeded throughout. Cost reconstruction error is zero.

The original symbolic/PyBullet scaled error is **0.005539872046**, so both new
records also fail the original 0.001 test. The separate source-recurrence closure
is **3.552713679e-15**, and componentwise prediction-minus-bound reserves remain
positive. This is finite sampled simulator evidence, not exact model equivalence,
continuous-time/hardware safety or a certified operating region.

## Historical distinction and the critical step

For each repetition, the new first 54 controls, first 55 control-boundary states,
all corresponding physics states/rewards/motor commands/predictions and all 54
retained solver candidate vectors match the corresponding v1 prefix **bitwise**.
The old full-horizon return/safety outcome remains unavailable and is not compared
with the new complete return as though the horizons matched.

At control index **54** (t=1.08 s), the new stored candidate contains upper-only
constraint row 18, the stage-0 lower-thrust side `g=umin-T1 <= -1e-6`:

- g = **-0.037577772047087454**; lower = -infinity; upper = -1e-6.
- Signed multiplier = **-1.0940617921979725e-9**.
- The corrected finite-side check records that dual-sign residual explicitly;
  it is within the frozen 1.01e-8 numerical tolerance.
- Applying the frozen v1 checker to this **new** saved point selects the absent
  lower side and produces infinite normalized complementarity, rejecting it.

The same mechanism appears at index 54 in both new repetitions and nowhere else
in these runs. This reproduces the legacy verifier defect after bitwise-matched
retained history and strongly supports that explanation for the earlier stop.
It does **not** recover the absent historical candidate or retrospectively change
its incomplete classification. The new v2 outcome passes; there is no observed
remaining numerical controller or v2-verifier failure in this targeted task.

## Timing and overhead

All 600 calls have candidate, solver-status, residual and timing records; no new
failed-call latency is missing. Actual solver-call time is measured around the
unchanged native Opti.solve call, before snapshot/verification. Initial solver
setup is included. Readout, candidate extraction, persistence, partial snapshots,
model diagnostics and physics instrumentation have separate timing categories.

| Scope | Median ms | p95 ms | p99 ms | Maximum ms | >20-ms observations |
|---|---:|---:|---:|---:|---:|
| solver_only | 7.187 | 11.802 | 12.889 | 72.745 | 2/600 |
| native_action | 7.786 | 12.479 | 13.545 | 73.829 | 2/600 |
| supervised_action | 39.569 | 52.566 | 56.237 | 89.244 | 600/600 |

`native_action` is an adjusted estimate from exclusive spans: native solver
compute is included and labelled capture/verification spans excluded. Small
wrapper dispatch/status-bookkeeping remains measurement overhead; it is not an
independently isolated hardware execution time. `supervised_action` includes the
full qualification callback and journals; it is not solver-only time.

Both solver overruns were at control index 0: **72.745 ms** (original) and
**31.177 ms** (repeat). They are preserved, not discarded as warm-up. All 600
supervised callbacks exceeded 20 ms; none exceeded the unchanged one-second
watchdog. Offline fixed-step simulation waits for computation. No hard real-time
readiness is claimed.

Across both executions, labelled exclusive wall times total approximately:
solver calls **4.739 s**, native-candidate extraction **0.385 s**, numerical native
plus independent verification **0.258 s**, candidate/diagnostic/partial-prefix
persistence **11.352 s**, and model diagnostics **0.136 s**. Remaining callback
work includes full-history reporting/bookkeeping and is not reassigned to the
solver. Physics integration/query capture/stream writing are separately retained
in each record; an execution totals approximately **0.020/0.022/0.042 s** for these
three components. Replay costs are measured separately.

## Budget, preservation and decision

Actual use is exactly **600 additional native solves and 36,000 physics steps**:
12,000 scientific-execution steps plus 24,000 steps for four recorded-action
replays. No standalone preflight optimization was run. There were two scientific
executions only. Reserved and actual solver/physics counts match, so the task
stayed within its entire approved allowance without retries or extra cases.

Recorded new CPU, including preflight and two independent record-verification
passes, is approximately **52.37 s**. Scientific-worker peak RSS is **321.22 MiB**,
under the existing 4-GiB limit; one worker was used. The conservative original
remaining allocation is **12,970 solver calls, 948,000 physics steps and about
13,397 CPU seconds**. No unused prior failure reservation was released and no
new full-study allocation was requested/consumed.

Preservation checks confirm unchanged Experiments 01–11C, accepted E12 pilot,
diagnostic/adapter/MMAQ/integrity archives, Research Decision Log, drilling authority
and all nine IQL hashes. They cover 53,819 historical project files, 87 authority
files, 289 upstream files and the accepted result seals. Source/configuration
identities and native program parity are retained. Independent verification
repeated with identical canonical JSON SHA-256, with zero additional solves or
physics steps. Generated outputs remain outside Git under
`results/benchmark_selection/H2_MPC20_requalification_v2/`.

**Stop for scientific review.** This targeted v2 numerical and sampled-safety
requalification passes. The overall benchmark gate remains failed; the original
0.001 model-agreement result and outstanding linear-MPC outcomes are untouched.
No sixteen-state screening, action probes, data acquisition, training, IQL work,
controller/solver change, commit or push was performed. Results do not establish
learning headroom or an acquisition benefit.

The new entry point permits only the frozen H2/MPC20 original/repeat and enforces
600 solves/36,000 steps. Completed attempts are reused, interrupted attempts are
not automatically overwritten, and historical files are never the output target.
For future read-only record reproduction, use a disposable copy of the completed
archive and run `experiments/12_h2_requalification_v2.py --verify` with that copy
as `--output`, the pinned environment and original source archives. Do not launch
another controller execution without separate approval.
