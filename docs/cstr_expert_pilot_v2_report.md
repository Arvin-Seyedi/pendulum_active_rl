# CSTR revised-controller technical pilot v2

**Verified PASS for the frozen four-case technical qualification.** This is a
separate result for the sole solver-option change `ipopt.bound_relax_factor=0`.
The stock-controller pilot remains **FAILED**, with its C2 rejection, missing
post-query telemetry and unexecuted C3/history diagnostics unchanged.

## Scope, provenance and completion

Pinned do-mpc v5.1.1: `90e0d7ada888a4cc75ea3769214de5105764c640`.
The accepted CPython 3.10.19 runtime, 40 package identities (including CasADi
3.7.2/IPOPT 3.14.11) and all 269 upstream source files were reverified. No upstream
source, dynamics, objective, bounds, scaling, simulator, horizon, warm-start logic
or other solver option was changed. A version-scoped setup wrapper adds exactly
the allowed option. Static original/revised objective, constraints and derivative
readouts agree bitwise; independent reconstruction differs by at most 1.336e-15.

Actual IPOPT `OptionsList` getters on the constructed backend observed zero
relaxation, unchanged other tolerances and `honor_original_bounds=no`.
Observations were retained for all **816** queries. This goes beyond checking a
configuration dictionary. The read-only C++ inspector uses the exact CasADi
version's layout headers and pinned libraries; preflight showed unchanged
controller memory/serialized backend after inspection. Its ABI validity is scoped
to the verified binary/compiler/architecture. Two preliminary helper builds failed
on missing build declarations; their logs are retained. Neither evaluated a solver.

The mandatory zero-solve preflight passed **33 tests**, then the separately
authorized execution completed **40/40 jobs**:

- Four frozen cases × two fresh scientific executions = **8/8**, 100 controls each.
- Native recorded-command replays = **8/8**; independent-equation/DOP853 replays = **8/8**.
- Eight frozen decision contexts × two initializations = **16/16** diagnostic queries.
- **816/816** optimization candidates independently checked: 800 trajectory queries
  plus 16 diagnostic queries. Diagnostic commands were not integrated.

Sampling is 0.005 h (18 s), horizon 20, degree-2 Radau/one element; each episode
lasts 0.5 h. Exact binary64 initial states and original ordering remain:
C0=(0.8,0.5,134.14,130), C1=(0.6,0.4,130,125), C2=(1,0.7,136,135),
C3=(0.8,0.5,138,134), in `[C_a,C_b,T_R,T_K]` order. Previous input is `[5,0]`.
Concentrations are mol/L, temperatures °C, F h^-1 and Q kJ/h.

Protocol seal SHA-256:
`8a1bd307f0a16dbc3b6b9d1426a79662d72aeb6e139553708459a4381f78ae21`.
See [protocol](cstr_expert_pilot_v2_protocol.md) and
[configuration](../configs/cstr_expert_pilot_v2.json).

## Per-case performance and margins

Results below use the four original executions, with repetitions treated as
reproducibility evidence rather than additional independent cases. Return is
negative total cost: 100 original concentration-tracking stages, scaled
input-movement penalties and one terminal concentration cost. No new reward or
discount factor was introduced.

| Case | Return | C_a RMSE (mol/L) | C_b RMSE (mol/L) | Dense minimum T_R margin (°C) | Minimum F lower margin (h^-1) |
|---|---:|---:|---:|---:|---:|
| C0 | -0.028003696619 | 0.010395665 | 0.013074748 | 0.008484769318 | 0.881817201450 |
| C1 | -0.129603688249 | 0.012542635 | 0.033593313 | 0.009537294829 | 1.640387576138 |
| C2 | -0.157807516664 | 0.032305808 | 0.023055886 | 0.009295368425 | 0.000000450093 |
| C3 | -0.025811765980 | 0.010305894 | 0.012249642 | 0.008025236230 | 3.102271469294 |

Complete-cohort **mean return -0.085306666878**, **worst -0.157807516664 (C2)**,
best -0.025811765980 (C3). These are four predefined technical cases, not an
estimate of broad operating-domain performance. No statistical confidence claim
is based on the deterministic repetitions.

| Case | Tracking sum | Scaled movement sum | Terminal cost | Feed volume (L) | Removed heat (kJ) |
|---|---:|---:|---:|---:|---:|
| C0 | 0.027901888538 | 0.000094006130 | 0.000007801951 | 58.546249 | 350.681107 |
| C1 | 0.128582834682 | 0.001013051506 | 0.000007802061 | 55.414535 | 280.308532 |
| C2 | 0.157523911516 | 0.000275803121 | 0.000007802027 | 60.144786 | 415.989993 |
| C3 | 0.025626517426 | 0.000177446523 | 0.000007802031 | 60.754131 | 421.649194 |

Movement is `.1*(delta F/100)^2 + .001*(delta Q/2000)^2`, using the actual previous
command, including `[5,0]` at k0. Feed/heat totals are descriptive physical-use
quantities, not added objective terms. Componentwise squared-input sums and all
terminal states are retained in `independent_review.json`.

Minimum dense two-sided raw state margins, in each component's units:

| Case | C_a (mol/L) | C_b (mol/L) | T_R (°C) | T_K (°C) |
|---|---:|---:|---:|---:|
| C0 | 0.601538755246 | 0.400000000000 | 0.008484769318 | 1.108604186333 |
| C1 | 0.500000000000 | 0.300000000000 | 0.009537294829 | 1.111788639490 |
| C2 | 0.581268452345 | 0.497706160235 | 0.009295368425 | 1.110926401066 |
| C3 | 0.601567836683 | 0.400000000000 | 0.008025236230 | 1.105208877678 |

All native initial/control-boundary/terminal states and all prescribed independent
dense samples had **zero observed state violations**. All returned commands,
including diagnostics, met the original F∈[5,100], Q∈[-8500,0] limits with
**zero input violations**, without clipping, projection or negative-margin masking.
Across all returned commands the lower-margin minima `[F,Q]` were
`[4.500926991e-7,7505.585672626]`; upper-margin minima were
`[86.948462799,0.007812470506]`, each in its own units. Both sides are saved
separately for every state/input, including predicted input sequences.

In particular revised C2's first F is **5.000000450092699 h^-1**. This passes its
own strict bound check. It does not alter the rejected stock value
4.999999450092826 h^-1 or supply a missing stock trajectory.

## Optimization, integration and reproducibility

All 816 queries returned native `Solve_Succeeded` and passed the original finite
candidate, objective/constraint reconstruction and scaled KKT criteria. Maxima:

| Check | Observed maximum | Frozen tolerance |
|---|---:|---:|
| Width-normalized primal | 5.15873e-9 | 1e-7 |
| Scaled stationarity | 7.56878e-9 | 1e-6 |
| Scaled complementarity | 8.40690e-9 | 1e-6 |
| Dual-sign residual | 0 | 1e-6 |
| Objective reconstruction error | 1.11022e-16 | 1e-10 |
| Constraint reconstruction error | 1.33530e-15 | 1e-10 |

Maximum native-coordinate equality residual was 9.80159e-9; it uses different
coordinates from the width-normalized primal metric. The minimum finite NLP
variable-bound slack was positive, 4.50093e-9. Independent readouts matched exact
saved state/previous-command context, input rescaling and integrator submissions.

Scientific repetitions match bitwise for raw states, commands, submitted/candidate
vectors, pre/post optimizer memory and solver status/iterations. All eight native
replays match the corresponding scientific trajectory bitwise. Both native and
independent replay records also match bitwise between repetitions. Independent
equation/integrator replays all pass the frozen 1e-7 normalized endpoint criterion;
the maximum discrepancy is **1.997876309e-9**. All dense normalized state reserves
after subtracting 2e-7 are positive; the smallest is **8.896929144e-5** (C3).

Native CVODES supplies 101 control-boundary states per complete episode.
Independent DOP853 reconstruction samples the original 10- and 20-subinterval
grids with shared samples agreeing. These are reconstructed numerical states,
not unrecorded native CVODES internal steps. Refinement resamples the same dense
solution rather than establishing integration convergence or continuous-time safety.
The preserved native prediction constraints still bound indices 1–19 and their
collocation slots, with an initial equality and no independent terminal box bound.

## Optimizer-history diagnostics

All eight native-history reconstructions reproduced the full saved candidate
bitwise. All eight cold-initialization queries passed status/KKT/input checks.
Cold-minus-native action differences:

| Context | delta F (h^-1) | delta Q (kJ/h) |
|---|---:|---:|
| C0 k0 | 0 | 0 |
| C0 k1 | -6.05915e-12 | +5.25821e-12 |
| C1 k0 | 0 | 0 |
| C1 k1 | -8.95284e-13 | -3.46546e-14 |
| C2 k0 | 0 | 0 |
| C2 k1 | +8.88178e-16 | +1.43264e-11 |
| C3 k0 | 0 | 0 |
| C3 k1 | +3.56335e-10 | +1.097665e-4 |

Maximum normalized action difference is **1.291370585e-8** of input ranges, at
C3 k1. The largest absolute objective difference was about 3.92223e-13. No new
action-equality threshold was introduced. At k0 both initializations coincide
with the fresh-start construction; the four k1 contexts supply the actual
warm-versus-cold comparisons. This finite evidence does not establish global
six-input sufficiency, uniqueness of the MPC law or absence of history dependence.

## Verification progress, latency and resources

The independent audit had already finished before interruption: all candidates,
raw margins/costs, repetitions/replays, contexts and full historical preservation
passed. This resumption reused that receipt and reconstructed the numerical
results identically, reran 33 zero-solve/static tests successfully, and completed
the report/bookkeeping. It generated **zero new solves/integrations**. Real-operation
factories were blocked or replaced by inert stubs throughout resumed verification.

Across all 816 queries, backend-evaluation envelope latency was median 15.251 ms,
p95 16.996 ms, p99 19.264 ms and max 23.880 ms. This includes the read-only option
getter and is not an isolated IPOPT-only measurement. Native action-selection
time including instrumentation/verification was median 20.520 ms, p95 24.365 ms,
p99 27.586 ms, max 43.247 ms. There were **zero measured 18-second deadline
overruns**. Native solver `t_wall_total` was unavailable. Iterations ranged 19–26
(median 25). Recorded native integration calls totaled 0.623435 s over 1,600
calls. A 4 GiB watchdog was enforced; measured peak scientific RSS was not retained.
This offline pilot does not establish hard real-time readiness.

Actual and reserved counts both equal their frozen ceilings: **816 solves,
2,400 control intervals** (800 scientific, 800 native replay, 800 independent
replay), **24,000 charged monitoring subinterval checks**. The two grids share
samples: they represent 16,000 unique dense subintervals, not 24,000 independent
observations. No count was reset or borrowed from historical allocations.

The ledger's stale `PREFLIGHT_PENDING` label was preserved in
`resume_ledger_before.json`; closing metadata updates that label only. The actual
preflight/completion/candidate evidence was already complete and is unchanged.
The separate incremental 600 CPU/1,200 active-wall ceiling includes preparation,
preflight, execution, verification and report/integrity work. Final charges are
appended below and retained in its ledger. Unused time does not authorize more
scientific work; all solve/integration/monitoring count allowances are exhausted.

## Preservation and scientific interpretation

The completed full preservation audit checked 87,315 accepted MPC80 files,
59,268 prior result/data files, 12,147 Pilot B files, 87 authoritative drilling
assets and nine unchanged IQL files; B0/B1 seals also passed. Resumed checks
reverified protected CSTR scientific files/evidence, prior tracked/untracked
sources and drilling/IQL assets. All prior outcomes, decision history, models
and historical ledgers remain unchanged. Generated results stay outside Git;
no commit or push occurred.

**This revised controller passes finite-cohort numerical expert feasibility.**
It does not certify physical reactor safety, robustness or a viable operating
box. Temperature margins remain only about 0.008–0.010 °C. Parameter error,
disturbances, actuator/sensor effects, independent learner performance,
beyond-expert headroom and active-versus-random acquisition were not tested.
No learner or acquisition study is authorized by this report. The original stock
pilot remains FAILED, separately from this revised PASS.

Evidence: `results/cstr_expert_pilot/no_bound_relax_v2/`, especially
`protocol_seal.json`, `study_completion.json`, `independent_review.json`,
`resumed_verification.json`, both unit-test logs, actual backend-option records
and complete per-attempt/per-interval data. Read-only reproduction is through
`experiments/cstr_verify_pilot_v2.py` with a fresh output path and the accounting
wrapper; it blocks optimizer/integrator execution. Do not invoke the pilot runner
again or overwrite accepted evidence.

Final incremental charges: **83.358 CPU seconds / 150.614 active-wall seconds**, including conservative closeout allowance. Remaining time ceiling: 516.642 CPU / 1049.386 active-wall seconds. All solve, interval and monitoring count allowances are exhausted.
