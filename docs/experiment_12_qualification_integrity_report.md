# Experiment 12 — Qualification-integrity audit v2

**Static audit only: zero new quadrotor episodes, optimization solves, physics
steps, controller evaluations, screening, probes or training.** The accepted
E12_MMAQ_v1 qualification remains **FAILED**. The original 0.001 symbolic/PyBullet
agreement gate remains **FAILED**. The native linear software failure, adapted
linear results, all archived incomplete executions and all original reports are
preserved unchanged. This additive report corrects two earlier QRQP summaries.

## Correct KKT conditions and verified defects

The actual Opti formulation is min f(z), lower <= g(z) <= upper. The frozen NLP
represents each state/input box side as an **upper-only** inequality, with lower
=-infinity and upper=-1e-6. Shooting and initial-state constraints are equalities.
The linear QP similarly includes upper-only box rows and equality rows, with its
unchanged 1e-8 tightening. This bound representation is consistent with the
[official CasADi NLP interface](https://web.casadi.org/docs/#nonlinear-programming).

For finite inequality sides use nonnegative multipliers mu_L, mu_U:

    L = f + mu_L'*(lower-g) + mu_U'*(g-upper)
    lambda = mu_U - mu_L
    grad(f) + J(g)'*lambda = 0
    mu_L*(g-lower) = 0; mu_U*(upper-g) = 0

Thus upper-only lambda >= 0, lower-only lambda <= 0; equality lambda is free and
has no inequality complementarity test. A fully unbounded row requires lambda=0.
Primal feasibility, stationarity, dual-sign errors and complementarity must be
reported separately. Products are evaluated **only on finite, existing sides**.
Stationarity always uses the original signed lambda, not a projected substitute.

The separately versioned `canonical_bound_kkt_v2` corrects independently
reproduced defects:

1. **Spurious infinite one-sided complementarity.** With g=-0.1, upper=0,
   lower=-infinity and lambda=-1e-30, v1 selects the absent lower side and
   generates infinity. V2 reports a finite dual-sign residual 1e-30; this is within
   the unchanged numerical tolerance. A substantial wrong sign still fails.
2. **NaN equality false acceptance.** V1 can report zero primal/reconstruction
   error when an equality value is NaN but its objective/gradient remain finite.
   V2 explicitly rejects nonfinite constraint values, multipliers and gradients.
3. **Infinite-bound sign mismatch.** V1's reconstruction check tests only
   `isinf`, allowing -infinity and +infinity to compare as equivalent. V2 compares
   signs, dimensions and output counts and rejects NaN operands.

Deterministic counterexamples exercise the actual frozen v1 functions; an
independent scalar exact-Fraction oracle covers 20 active/inactive, lower/upper,
two-sided, equality, free-row and invalid cases. Further tests cover infeasibility,
wrong dual signs, genuine nonzero complementarity, overflow and failure recording.
These are infrastructure defects, not evidence that historical trajectories had
NaN states or that a solver's native success report was false.

Existing numerical tolerances are retained: stationarity, dual-sign and normalized
complementarity <=1.01e-8; NLP primal <=1e-4, QP primal <=1.01e-8; reconstruction
<=1e-10. Raw complementarity is also reported. NLP stationarity does not prove a
global optimum, and numerical KKT checks do not certify plant safety. No model,
constraint, objective, solver option, horizon or gain changed.

## Retained candidates and unresolved MPC20 outcomes

**All 7,932 retained candidates pass corrected numerical KKT checks:** 1,224 QP
and 6,708 NLP candidates. Mathematical readouts reuse the frozen independent
shooting/loss equations through a symbolic shell, without constructing an
environment or controller. No `Opti.solve` is called. This does not turn an
incomplete episode into a completed or safe one.

For H2/MPC20, both original and repeat attempted 55 selections, retained only 54
candidate/check records, and stopped before applying control index 54 (t=1.08 s).
Both are still **UNRESOLVED verifier rejections**. The rejected primal, dual and
independent diagnostic vector are absent. No missing vector was inferred from a
previous solution, rounded log, re-solve or retrospective simulation.

Native IPOPT's last reported status is Solve_Succeeded, nine iterations. Both
logs print native residuals:

- Dual infeasibility: **1.645289367e-9**.
- Constraint violation: **1.267613792e-10**.
- Complementarity/overall NLP error: **7.187619828e-9**.

These retained scalar reports support native numerical success; they cannot
identify the independent checker's missing vector or prove that the one-sided
bug caused these particular rejections. Historical full-horizon returns and
safety outcomes remain unavailable for the interrupted cases.

## QRQP failures: what the actual traces establish

| State | Last printed iteration | Printed primal residual | Printed dual residual | Trace classification |
|---|---:|---:|---:|---|
| H0 | 1000 | 0.044 | 0.025 | iteration-limit cycling |
| H2 | 12 | 0.049 | 0.027 | numerical failure with singular active-set trace before iteration limit |

Both outcomes repeat identically. **H0 reaches the unchanged 1,000-iteration
limit. H2 has no Max-iter marker and its trace ends at iteration 12**, on a
singular/pending-loss-of-regularity path before generic conic failure. The exact
final internal iteration/status flag is not retained; the Opti status is unknown.
The previous claim that all four failures reached 1,000 iterations is incorrect.
It remains preserved in the historical report and is corrected here.

H0 repeatedly drops/enforces combined row 520 (stage-8 lower T1 bound), with
stagnant residuals and printed steps as small as 2.9e-308. H2's last noted change
is row 344 (stage-0 lower T1 bound). `Sing=1` is not itself an infeasibility
certificate. The installed CasADi 3.7.2 runtime can set this flag for a pending
rank loss without refreshing the displayed minimum-R value; the printed 0.0044
is not a condition estimate for the failed working set.

Available static matrix diagnostics, in the frozen coordinate units:

- Hessian eigenvalues **0.1 to 5**, condition 2-norm **50**.
- Equality matrix rank **246/246**, condition approximately **2,747**.
- Full constraint matrix rank **326**, condition approximately **40.14**.
- Last-successful diagnostic near-active matrices have ranks **277/277** (H0)
  and **264/264** (H2), conditions approximately **4,213/4,214**. Their KKT matrix
  conditions are approximately **99,420/91,037**.

Near-active means slack <=1e-7, a fixed diagnostic threshold, not a changed solver
or acceptance tolerance. These are last-successful sets, **not** recovered failed
working sets. The independent problem matrices agree with archived native
matrices within 2.22e-16. Coordinate-dependent conditioning does not establish a
physical modeling error or identify the failure cause.

Source and input-dump checks confirm a native warm-start consistency problem:
`action=u_val[:,0]` is a NumPy view, and in-place addition of the 0.1323-N nominal
hover offset also changes the first column of `u_prev`. The next optimization
uses input **deviations**, but its warm start contains first-column absolute
thrusts around 0.296683 N versus an allowed deviation upper bound around
0.164383 N. The mismatch can reach **0.1323 N**. Reconstructed guesses agree
with rounded dumps within 4.92e-6. This is a warm-start issue, not an observed
applied-thrust violation; infeasible starting guesses are allowed by QP methods.
It is a plausible contributor, **not a controlled causal finding**. Nothing was
repaired in the native controller or adapter.

The logs retain rounded problem/input guesses, including zero initial dual
vectors, and scalar iteration diagnostics. They do not retain final failed
primal/dual vectors or an infeasibility certificate. Native generic text saying
'Infeasible MPC Problem' is not such a certificate. Nonlinear MPC safety also
does not prove feasibility of a different linear prediction QP.

A second historical summary correction: the largest successful QRQP iteration
count is **37** when successful prefixes of incomplete episodes are included;
9 was the maximum among completed episodes only.

## Failure retention and prospective timing components

The additive `candidate_guard_v2` and native solve-capture hook persist the
CURRENT raw candidate before any independent rejection, plus objective,
parameters, multipliers, constraint values/bounds, gradient, solver status and
available native statistics. Missing backend/debug data are explicitly marked;
`opti.initial()` or previous controller solutions are never substitutes. Debug
points are labelled unverified and are not claimed to be final failed iterates.

Binary NPZ preserves nonfinite values and NaN payload bits. Atomic tagged JSON
preserves diagnostics, per-row primal/dual/complementarity information, exceptions,
raw/partial trajectory snapshots and an explicit failed/incomplete status. Invalid
arithmetic residual infinities are labelled rejection sentinels, not measured
solver divergence. Absent-side slack slots are masked placeholders. No fallback
action is applied; the guard returns an action only after checks pass.

This is tested reusable instrumentation for a future approved runner. It is not
wired into the immutable v1 runner, and live integration has **not** been tested
with new episodes in this audit. Future qualification must use the native capture
point and independently reconstructed readouts before allowing actions.

TimingLedger separates native selection, native optimization, extraction,
verification, persistence and prefix capture. It handles exceptions in `finally`.
Nested solver compute is counted as native MPC work; nested audit/persistence is
excluded from adjusted native latency. Unit tests exercise this accounting.

## Historical timing: measured and unavailable quantities

Full exclusive native action-selection latency cannot be recovered from v1's
inclusive measurements. The difference between inclusive time and solver time
contains both native Python work and qualification overhead, so it is retained
as **unallocated remainder**, not labelled entirely as verification.

The IPOPT logs do retain rounded optimizer-kernel wall times, excluding native
warm-start/setup/action extraction. They provide a partial decomposition:

| Controller | Kernel median/p95/p99/max, ms | Printed kernel >20 ms / calls | Preserved inclusive >20 ms / measured |
|---|---|---:|---:|
| linear40 | Unavailable | 0/0 | 25/1224 |
| nonlinear20 | 7.410 / 12.021 / 13.460 / 20.270 | 1/1910 | 817/1908 |
| nonlinear40 | 11.650 / 21.650 / 28.260 / 106.890 | 194/2400 | 1818/2400 |
| nonlinear80 | 21.250 / 38.591 / 45.640 / 82.070 | 2209/2400 | 2400/2400 |

MPC80 has six additional kernel entries rounded to exactly 20 ms, whose threshold
side is ambiguous. The **5,060 measured inclusive overruns remain unchanged**;
six failed full-call latencies remain missing. The two interrupted MPC20 calls
have printed solver-only times of 7.83 ms, which do not recover their full action
latencies. The earlier adapter revalidation's 2/2,400 overruns and 80.281-ms maximum
also remain unchanged. Offline feasibility is distinct from hard real-time
readiness; MPC80 already exceeds 20 ms on many solver-kernel measurements alone.

## MPC80 evidence and remaining qualification decision

MPC80 retains four held-out initial conditions, two bitwise repetitions each,
300 controls and all 6,001 raw physics states per completed episode. It has zero
observed state/input violations or contacts, positive one-step discrepancy-aware
reserves, and **2,400 retained candidates passing the corrected KKT check**.
Mean/worst returns remain **-39.331860180/-65.816857638**, minimum normalized raw
margin **0.200745220**. Its largest original symbolic/PyBullet scaled discrepancy
remains **0.01326088346**, above the unchanged 0.001 gate.

This supports finite-horizon empirical simulation-expert behavior on those four
states. It does not establish a certified operating region, broad robustness,
full benchmark qualification, real-time/hardware readiness, safe alternative
conditional actions or an acquisition advantage. No historical failure is relabelled.

The conservative remaining original allocation is **13,570 solver calls and
984,000 physics steps**; recorded CPU balance after this audit is approximately
**13,450 seconds**. Unused failure reservations were not released. Actual-count
balances (15,232 calls and 1,084,080 steps) are shown separately and are not the
execution allowance. No additional full-study allocation was approved or consumed.

Prospective allocations below assume full completion and two action replays per
scientific execution; they are estimates, **not authorizations**:

| Future scope | Solver calls | Physics steps including replays | Fits conservative remainder? |
|---|---:|---:|---|
| targeted_H2_MPC20_two_repeats_only | 600 | 36,000 | Yes |
| both_failed_linear_cases_two_repeats_only | 1,200 | 72,000 | Yes |
| complete_four_state_four_controller_qualification_twice | 9,600 | 576,000 | Yes |
| original_sixteen_state_four_controller_screen_once | 19,200 | 1,152,000 | No |

The original complete sixteen-state screen is short **5,630 solver calls and
168,000 physics steps**, even before further qualification. Do not assume early
failures, remove baselines or silently reset counters to make it fit. A full
four-state redo would leave only 3,970 calls and 408,000 steps. Prior costs suggest
tens of seconds for a targeted two-episode check and roughly 5–10 minutes for a
four-state redo; new journal I/O is not benchmarked and these are not guarantees.
Existing watchdog, one-worker, 4-GiB and cumulative original CPU limits remain.

**Recommended next review:** retain MPC80 evidence; do not repeat all controllers
blindly. Consider authorizing only H2/MPC20 original/repeat with the corrected,
fully capturing verifier (600 calls/36,000 physics steps), matching saved prefixes
and preserving new versus historical outcomes separately. Before another linear
controller episode, review a minimal saved-QP reproduction to distinguish
warm-start aliasing from the QRQP active-set mechanism. Any warm-start or solver
change needs a separate versioned design/approval; no iteration-limit increase
or solver replacement is justified by this audit alone. Screening/probes remain
blocked by qualification and resource requirements.

## Verification, preservation and reproduction

All **58 focused tests** pass, including exact rational reference cases, both
one-sided signs, equality/free rows, invalid bounds, real complementarity errors,
NaN/overflow, native-candidate capture before rejection, no-fallback behavior and
exclusive timing. Optimizer entry points were patched to reject actual solves;
PyBullet was never imported. Reconstructed audit JSON results repeat with
identical canonical SHA-256 hashes.

Preservation passes for **53,819 historical project files, 87 drilling authority
files, 289 upstream files, all 498 sealed MMAQ artifacts, prior E12 seals/reports,
and nine current IQL file hashes**. Generated evidence remains under
`results/benchmark_selection/qualification_integrity_v2/`, outside Git. The audit
used approximately **25.46 logged CPU seconds**, peak **104.69 MiB RSS**, and
zero solver/physics calls. Ad hoc development test invocations were not individually
timed. No commit/push, training, new drilling investigation or controller change
occurred. The decision log and prior reports are unchanged.

From the repository, a fresh read-only reproduction is:

```sh
/private/tmp/e12_venv/bin/python experiments/12_qualification_integrity_audit.py \
  --source results/benchmark_selection/mmaq_v1 \
  --output /private/tmp/e12_integrity_recheck
```

It verifies accepted seals, performs static arithmetic only, rejects optimizer
calls, and refuses to overwrite a sealed audit. Reproduction requires the accepted
scientific archives and pinned upstream sources; it imports no IQL files.
