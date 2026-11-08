# E12_MPC80_FEAS_v1 — stopped execution report

**Campaign STOPPED on its first failed acceptance check. Completion is 0/16
states and 0/32 scientific executions.** State 0 original is incomplete; states
1–15, every repetition and every replay were not run. The stop was caused by a
new execution-instrumentation defect, not an established MPC80 thrust violation.
No repair, retry, altered tolerance or remaining-state execution was performed.
The original overall E12 qualification and original 0.001 model-agreement gate
remain **FAILED**.

## Authorization, frozen inputs and execution

The explicit user approval is retained as `execution_approval_request.txt`,
SHA-256 `68fc6f7029fcad27ec48cc1650ac76ff4e284357d3d94e46c4d0bf1ca4d29824`.
The unchanged accepted protocol seal is
`6dbb0ccd3076bad0f5513237a14317413a1eca05e152a26a154f1fc7a607b2bf`.
All 96 binary64 coordinates, original IDs/order, PCG64 120001 provenance,
source/configuration identities, dependency versions and core binary hashes
passed checks before dispatch. The native MPC80 program hash remained
`d52a79bf7f360acf8d9831952a3632c40479dfeac02f9351df0f7751c86bb32d`.

The accepted preflight ledger was copied into a separately linked execution
ledger; the original was never modified. Its **58.0 CPU s and 95.2 active wall s
were already charged** against this study's 1,800/3,600-second caps. New wiring
reused unchanged native MPC/plant execution and the validated v2 candidate guard.
All **97 zero-solve tests** passed before dispatch. They did not detect the
subsequent live cache-timing error.

Exactly one job, `00_original_episode`, was reserved before dispatch. One native
solve returned `Solve_Succeeded` in 20 IPOPT iterations. Its candidate passed
independent v2 checks; one physics step was applied and recorded. The campaign
then stopped at t=0.001 s, before completing any 0.020-s control transition.

## First failure: observing the wrong RPM field

The new wrapper checked `require_executed(env.last_clipped_action, model)` from
inside the intercepted `pybullet.stepSimulation` call. This was incorrect.
Pinned `base_aviary.py` initializes that cache to zero at line 201, calls
`_physics(clipped_action)` before physics integration, calls `stepSimulation`
at line 282, then assigns `last_clipped_action = clipped_action` at line 284.
The callback runs before the assignment. Thus it checked reset zeros rather
than the RPM argument actually supplied to that physics step.

The resulting apparent margin was −0.056323375552892685 N per channel. This
is **an invalid measurement of executed input**, not evidence that the plant
received zero thrust. The initialized cache value is established from source
ordering; it was not separately captured in the failure JSON.

Retained requested thrust was `[0.056324369506392545, 0.14865930279665016]` N.
Independent application of the unchanged native conversion gives expected
grouped thrust `[0.056324369506392545, 0.14865930279665018]` N and RPMs
`[9440.383351674363, 15336.897203068307, 15336.897203068307, 9440.383351674363]`.
Its lower input margins are `[9.939534998601451e−7, 0.0923359272437575]` N;
upper margins are `[0.24035852472349636, 0.14802359143323873]` N.
These are explicitly **source-derived**, not a recovered direct `_physics`
argument log. The stored per-control RPM/clipped-command arrays are empty
because native `env.step` did not return. They have not been retrospectively filled.

The first recorded physical transition matches the source equation under this
converted command to maximum **2.7755575615628914e−17** in component units.
Zero RPM instead gives velocity discrepancies
`[0.0003614009312536831, 0.0075833811113498115, 0.18466986658051526]`
in `(vx,vz,omega)` units. This supports the stale-cache diagnosis without another
simulation or solve. It does not retroactively pass the input telemetry gate or
establish full-horizon controller safety.

## Completion and performance denominators

| State ID | Original | Repeat | Complete pair | Six-second return/cost |
|---|---|---|---|---|
| 0 | Incomplete after one physics step | Not run | No | Unavailable |
| 1 | Not run | Not run | No | Unavailable |
| 2 | Not run | Not run | No | Unavailable |
| 3 | Not run | Not run | No | Unavailable |
| 4 | Not run | Not run | No | Unavailable |
| 5 | Not run | Not run | No | Unavailable |
| 6 | Not run | Not run | No | Unavailable |
| 7 | Not run | Not run | No | Unavailable |
| 8 | Not run | Not run | No | Unavailable |
| 9 | Not run | Not run | No | Unavailable |
| 10 | Not run | Not run | No | Unavailable |
| 11 | Not run | Not run | No | Unavailable |
| 12 | Not run | Not run | No | Unavailable |
| 13 | Not run | Not run | No | Unavailable |
| 14 | Not run | Not run | No | Unavailable |
| 15 | Not run | Not run | No | Unavailable |

Full-cohort mean/worst cumulative or discounted return, tracking RMSE, terminal
error, effort and saturation are **unavailable**. There are no completed native
reward samples; an empty reward sum must not be reported as a zero-return
episode. The solver's predicted horizon objective is not realized six-second
performance. Zero completed/qualified states does not mean sixteen unsafe states.

Both recorded raw physics states (initial and first successor) are finite and
inside the box, with zero contacts. Minimum observed raw margins, ordered
`(x,vx,z,vz,theta,omega)`, are
`[1.9916903426932646, 29.923011677451427, 0.8802147761377608,
29.954989999042112, 1.4359088709321746, 8.528685181012593]`, in component units.
Minimum normalized margin is **0.4293730613745624**. Full six-second safety and
violation incidence cannot be estimated from this prefix.

The first prospective 20-ms reserves were positive:
`[1.9900167708707233, 29.915958868432476, 0.880418777175266,
29.997179207597906, 1.4712521910124736, 5.0009010019626]`.
The full 20-ms interval was not executed, so its discrepancy coverage was not
tested. The 1-ms source-equation check is a separate matched-state diagnostic;
it does not pass the original 20-ms 0.001 model-agreement test or certify safety.

## Numerical verification and timing

The sole retained native candidate is complete and finite. Independent/native
graph discrepancy is **3.552713678800501e−15**. Its v2 residuals are primal
**7.793313798853018e−9**, stationarity **1.2591316878030057e−13**, dual sign **0**,
and normalized complementarity **2.273857571620207e−9**; all meet unchanged
tolerances. One observed solver success out of one attempt is not a reliability
estimate for sixteen initial states. No numerical MPC failure was observed.

| Latency scope | n | Median/p95/p99/maximum, ms | >20 ms |
|---|---:|---:|---:|
| Native `Opti.solve` boundary, including cold setup | 1 | 134.233500 | 1/1 |
| Adjusted native action selection | 1 | 135.235582 | 1/1 |
| Complete supervised action callback | 1 | 180.971667 | 1/1 |

All four quantiles coincide because n=1; no latency distribution is inferred.
The separate printed IPOPT kernel time was approximately 69.312 ms, excluding
some solve-boundary setup. It is not substituted for measured solve time.
Verification/capture/persistence spans remain separate in the raw status record.
The one-second watchdog was not reached. Hard real-time readiness is not established.

## Resources, preservation and decision

Actual experimental work is **one solver call and one physics step**, with
zero completed controls, repetitions or replays. The failed job retains its
entire **300-solve/6,000-step reservation**; no refund or automatic retry occurred.
The remaining conservative original allowance is **12,670 solves and 942,000
physics steps**, not an allowance based on the smaller actual counts. Unattempted
jobs remain unexecuted; the campaign ledger is halted.

After two read-only audits, charged study CPU/wall were approximately
**101.930/176.969 seconds**, including the accepted preflight charges; final
report/integrity accounting is retained separately. Recorded worker peak RSS was
**361,267,200 bytes (344.53 MiB)**, below 4 GiB; aggregate/parent peak RSS was not
retained and is not invented. No recorded CPU, wall, solve or physics cap was
exhausted. The stop was an instrumentation failure, not resource exhaustion.

Two fresh-process read-only audits produced identical canonical scientific JSON:
`50d684290b3d82822f7ca10dc373fca44b9bc0f8da7c4a047843e6a6cc32c2f2`.
The raw two-sample stream matches its NPZ bitwise; full solver/candidate and partial
records remain intact. New controller trajectory repetition and independent action
replay were **not performed**, as required by the first-failure stop rule.

Preservation checks pass for **197 tracked files, 59,183 prior result/data files,
87 authority files, 289 upstream files, 5,438 sealed E12 artifacts and all nine
IQL hashes**, including accepted preflight evidence and its untouched ledger.
Execution sources and failed records remain unchanged after the stop. Read-only
audits created no optimization or physics calls. Evidence is outside Git under
`results/benchmark_selection/mpc80_feas_v1_study/`, including the explicit stop,
failed reservation, raw prefix, candidate, source evidence and audit repeats.

The MPC80 expert-feasibility question remains unresolved. The invalid mid-step
input observation must not be described as an MPC80 safety failure, nor repaired
post hoc to declare success. Any instrumentation correction must observe the
actual native command at its application boundary without changing plant/cache
semantics, and requires separate review before any new execution. No correction,
retry, remaining-state evaluation, probe, acquisition, training, commit or push
was performed. Stop for scientific review.
