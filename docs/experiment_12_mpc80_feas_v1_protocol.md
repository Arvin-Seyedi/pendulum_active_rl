# E12_MPC80_FEAS_v1 — frozen protocol and zero-solve preflight

Only static preflight is authorized. This version does not change the original
FAILED overall qualification or FAILED 0.001 symbolic/PyBullet criterion. A passing
preflight is readiness evidence, **not permission to execute** the cohort.

## Immutable cohort and model

Load the original `mmaq_v1/screening_manifest.json` and `screening_states.npy`
directly, under `results/benchmark_selection/`. Check the SHA-256 identities in
`configs/experiment_12_mpc80_feas_v1.json`, all 96 binary64 coordinates, hex/byte
representations, IDs 0–15, row-major draw ordering and PCG64 seed 120001. Raw-bit
reconstruction is an independent verification only; it never replaces the inputs.
Keep the frozen probe indices/menu untouched and unevaluated. The sixteen states
are qualification/development data, never fresh held-out learned-policy evaluation.

Retain upstream `6b5391d014f36fdfa0f9d22d92c77387e5274308`, Python 3.10.19, all
archived package versions, vehicle/source files, task/reference arrays and native
MPC definitions. Six seconds, 50-Hz control, 1,000-Hz physics, horizon 80, original
Q/R and hard constraints, physical action convention and seed 1337 are unchanged.
Freeze actual source identities for objective, terminal cost, constraints,
reference indexing/padding, solver dispatch/options, controller information and
warm starts. In particular, do not repair the native nonlinear warm-start row
shift. No disturbance, randomization, controller redesign or solver tuning.

## Prospective execution, currently disabled

Order: original IDs 0–15; for each ID, original then repeat; for each phase,
scientific episode then native uninstrumented action replay then independent
instrumented executed-motor replay. Fresh controller/environment memory between
scientific executions; native warm starts within each execution. No replacement,
favorable reordering or retry. Exact requested coordinates are preserved;
measured reset representation differences must be <=1e-14.

Each complete scientific execution has 300 applied controls and 6,001 raw physics
states. Acceptance requires 16/16 states twice, zero negative raw lower/upper
state margins, zero contacts and no executed-input bound violations. There is no
rounding tolerance for empirical safety. Requested/clipped/executed actions and
RPMs remain separately logged. Failure or truncation is not a safe episode.

Require native `Solve_Succeeded`, finite candidate/action arithmetic, independent
primal residual <=1e-4; stationarity, dual-sign and normalized complementarity
<=1.01e-8; native/independent program reconstruction <=1e-10. Use unchanged v2
finite-side KKT rules. Retain raw complementarity as well. Local numerical KKT
conditions do not establish global NLP optimality or state safety. All scientific
arrays and available primal/dual records must repeat bitwise; both independent
replay types must agree bitwise. Timing need not repeat.

Reuse the approved componentwise mismatch majorants, upward arithmetic and exact
rational cross-checks. Their arguments are current/past state, current command
conversion and documented preceding-physics attitude cache (current angle at
reset). Require observed error <=bound+1e-10, empirical source closure <=1e-10,
and strictly positive componentwise predicted boundary distance minus bound at
every step. These are one-step numerical reserves, not recursive robust or
physical safety certificates. Contacts/chart/context failures invalidate the
contract. A nonpositive conservative reserve does not prove physical unsafety.

Pause the campaign on the first failed check or resource limit, preserving all
already-generated data. Distinguish controller failure, verifier rejection,
empirical violation, contract failure and resource/instrumentation failure.
Unattempted states remain in the 16-state denominator. No fallback, retry or
threshold adjustment. Historical MPC20/40 and linear results are references on
their own cohorts, not a newly completed four-controller benchmark.

## Resources and failure accounting

The frozen 96-job plan contains 32 scientific executions (9,600 solves/192,000
physics steps), 32 native replays (192,000 steps), and 32 motor replays (192,000
steps): **9,600 solves and 576,000 steps**. It fits the conservative remaining
12,970/948,000 allowance, leaving 3,370/372,000 if fully reserved. This envelope
is earmarked; no scientific job is authorized or charged by preflight.

One worker, aggregate CPU 1,800 s, aggregate active wall 3,600 s, 4 GiB, unchanged
one-second action watchdog. Include preflight, parent and child CPU without double
counting, verification, instrumentation, persistence and replay. Exclude human
review idle time from active wall accounting; record it separately. The latest
historical CPU balance predates some finalization checks and is not relabelled as
an exact current balance. No new full-study allocation is assumed.

Before a future dispatcher could launch a job, it must durably debit the entire
job reservation and receive a receipt from the versioned ledger. Execution also
requires separate approval bound to the frozen protocol seal and a passing
preflight. Failed/reserved work never refunds resources, is never automatically
retried, and halts the ledger. Verification overhead is charged even when no
scientific job runs. Partial/failed backend data are explicit, not fabricated.

## Static checks and reporting

Preflight disables native optimization entry points and physics stepping; it
also blocks controller action selection and environment step APIs. Native graph
setup may create a disposable environment at a previously archived initial
condition, but does not select actions, integrate physics or produce an episode.
Record this distinction explicitly. Reconstruct the entire MPC80 graph from
pinned native code and independently compare objective, all constraint rows,
bounds and Lagrangian derivatives at retained candidates. Validate all accepted
held-out MPC80 candidates; use archived trajectories only for contract checks.

Exercise failure capture with static archived/synthetic inputs: save candidate,
primal/dual arrays, nonfinite diagnostics, status, timing and raw partial prefix
before rejecting it. Tests must cover missing candidates, numerical rejection,
exceptions, no fallback, reservation ordering, failure consumption and CPU/wall
limits. Do not execute a test optimization to demonstrate these properties.

After separate execution approval, report per-state and cohort mean/worst native
quadratic costs/returns, discounted returns, tracking/terminal errors, absolute
and hover-relative thrust effort, saturation, componentwise raw margins and first
crossings. Solver/native/supervised latency median/p95/p99/max and every 20-ms
overrun remain separate from logging/verification/physics costs; no hard real-time
claim. Missing/partial outcomes remain explicit. Repeats are not independent
states; full-cohort cost is unavailable if any case is incomplete. Cost is
descriptive, with no invented post-outcome reward cutoff.

A successful future cohort would justify reviewing safe-action diversity, not
automatically running it. The original four-state/nine-action continuation study
does not fit the resulting balance. Imitation/acquisition would additionally need
a reducible learning gap, equal transition budgets and fresh policy evaluation;
beyond-expert RL needs conditional alternative-action support and meaningful safe
performance headroom. No such claims follow from preflight.

Generated evidence is separate under
`results/benchmark_selection/mpc80_feas_v1_preflight/`. The entry point
`experiments/12_mpc80_feasibility_preflight.py` exposes **no execution mode**.
All previous source/config/report/result identities and nine IQL hashes are
preserved. A repeat archived audit may write only new preflight evidence files;
it never edits accepted archives or recomputes experimental outcomes.
