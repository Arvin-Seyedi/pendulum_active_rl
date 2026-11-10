# E12_MPC80_COUNTERFACTUAL_v1 — zero-solve preflight and budget review

This version authorizes static/mocked and archived-record checks only. The CLI
`experiments/12_counterfactual_preflight.py` has no execution mode. Future use of
native context/selection helpers needs separate execution approval, sealed job
reservations and an approved incremental CPU envelope. No counterfactual episode,
solver or physics execution is authorized. Original FAILED E12 overall and 0.001
model-agreement gates, v1 failure, and accepted MPC80_FEAS_v2 remain unchanged.

## Fixed scientific comparison

Keep original probe IDs 0/4/8/12, exact manifest binary64 values, step zero,
PCG64-120001 provenance, original nine Cartesian offsets {-0.05,0,+0.05} times each
channel's unchanged range, lexicographic first-channel-outer ordering. Nominal
menu index is 4. Never regenerate states, replace actions, inward-shift a bound,
change a tolerance or choose by future outcomes. Keep all 36 labels, including
input-inadmissible/duplicate/not-run labels. Source-predicted input admissibility
is not observed state safety. Force-equivalent labels deduplicate to the earliest
original representative; motor RPM and force signatures are retained separately.

Original full-study criteria stay at three of four states with two additional
safe actions and >=5% channel-range separation, >=10% mean cost improvement,
zero violations, <=0.01 normalized-margin loss and per-state cost deterioration
<=5% of comparator mean cost, against the strongest matched classical comparator.
Exact rational evaluation of binary64 submitted thrusts distinguishes tiny numeric
differences from the required separation; no epsilon turns a shortfall into a pass.
This technical pilot cannot settle these scientific full-study criteria.

## Native decision context and one-step substitution

Preserve native construction order from p.run_episode: create the plant; make
MPC80 with its native reset; then reset the plant once. No second controller reset
or kinematic refresh. Match actual archived reset raw values bitwise (also require
native reset tolerance 1e−14), original request coordinates, seed, info step zero,
empty x_prev/u_prev and zero reset motor cache. Observe full base pose/quaternion,
velocities, joint states, contacts, native cached arrays/counters and RNG states
using passive read APIs only; never query forward kinematics or restore a synthetic
six-state checkpoint. Old evidence lacks full internal engine checkpoints; do not
claim archived equality of unrecorded Bullet state or a formal hidden-state proof.

Run the unchanged first MPC query, with the original 6×81 reference preview and
terminal padding. Require the original first action, primal/dual/parameters,
initial guesses and x/u plans to match; exclude timing from bitwise equality.
The returned action aliases u_prev[:,0]. Copy it before offset addition. Leave
nominal post-query optimizer memory/prev_action untouched, then apply the detached
candidate once and use normal MPC80 at steps 1–299. Preserve the native x-column
and u-row warm-start shifts. Zero-offset must reproduce the baseline throughout.
Never reconstruct live optimizer internals solely from stored candidate vectors.

CounterfactualPath accepts an independently verified original-query callback;
reuse IndependentCandidateGuard, v2 KKT mathematics and full candidate capture.
It does not replace the optimizer or assert that the candidate is the nominal
optimizer's action. Candidate-specific prediction/reserve must be computed from
the actual detached request and current raw state/cache, not nominal forecasts.
Preflight uses inert predictor spies/archived predictions only, no new integration.

Retain the accepted v2 application observer/current RPM and force/torque submission
records. Requested/clipped thrust, direct RPM/submission arguments, derived
KF*RPM² thrust and unobserved physical outputs remain distinct. This version's
static motor mapping is independently checked against native arithmetic and
scripted native application callbacks; it is not live actuator telemetry.

## Acceptance and failure records for a future technical pilot

Each qualified run completes 300 controls and 6,001 raw states. Require strict
nonnegative raw state/input margins, zero contacts, native Solve_Succeeded,
unchanged corrected v2 KKT/finite/program checks, positive causal reserves,
source/contract closure <=1e−10 and bitwise deterministic repeats plus native
recorded-action and independent motor-command replays. Preserve instrumented vs
uninstrumented comparison. First nominal query must match archived context for
both nominal and alternative branches; only the zero-offset entire trajectory
must match baseline, since true alternative successors are expected to differ.

All prospective calls need reservations before dispatch. Retain rejected solver
vectors, primal/dual diagnostics, nonfinite values, partial physics records,
status and separated solver/native/supervised timing. Native candidate capture
and the v2 journal precede verification; additional intervention failures use
lossless JSON/nonfinite tags and raw NPZ. No fallback, retries or overwriting.
Incomplete/unverified is distinct from empirical safety or observed unsafety;
input-inadmissible and duplicate labels remain distinct. Stop the technical pilot
on its first acceptance, telemetry, source, context, replay or resource failure.
No failed reservation is refunded. Do not label failures physical nonviability.

## Two pilot scopes (neither approved for execution)

A: nominal controls at all four IDs, each original/repeat/both replays, followed
by one admissible ID-0 alternative pair: 10 scientific runs +20 replays,
3,000 solves /180,000 physics steps. Remaining original allowance: 70 /186,000.

B (recommended minimum): nominal ID 0/menu 4 pair and ID 0/menu 3 alternative
pair, each with both replays: 4 scientific runs +8 replays,
1,200 solves /72,000 physics steps. Remaining original allowance: 1,870 /294,000.
Freeze ID 0/menu 3 as the first lexicographic nonzero, input-admissible,
nonduplicate action meeting exact 5% separation. No returns/continuation safety
are used. Exact request/RPM/force/torque byte representations are sealed by
preflight. No fallback alternative is selected if this frozen branch fails.

A adds live reset/nominal-path checks at the other three archived starts. B already
exercises both zero/nonzero substitution paths, continuation memory, direct
telemetry and independent replays at one start; static tests cover all four
archived warm-start/reference sequences. B is sufficient as a machinery pilot,
not a cohort safety/action-support test. A need not precede B; any later expanded
context verification requires separate review and retains original denominators.

B's execution time envelope is proposed at 300 incremental CPU/600 active-wall
seconds; A at 600/900. Both retain one worker, 4 GiB, native 1-second watchdog,
and all 20-ms overrun reporting. Carry prior conservative finalization charges
1,711.201929 CPU /2,182.657953 active-wall seconds, add this preflight's measured
charges and conservative source-preparation overhead, and preserve all old
ledgers. Remaining old local CPU headroom cannot fund either pilot; an explicit
cumulative ceiling extension is required, not a fresh/reset clock. Proposed
resource figures grant no allocation. No new full-study budget is authorized.

Run only the static CLI with pinned requirements_experiment_12.txt and upstream
6b5391d014f36fdfa0f9d22d92c77387e5274308, choosing a fresh output directory.
All genuine solver, environment stepping/creation, force application and engine
APIs are blocked. Native Python controller flow uses archived solution lookup;
native application flow uses a scripted backend with no dynamics. Passing these
checks prepares a separately approved live B pilot; it cannot establish live
trajectory equivalence, action support, practical headroom or learned-policy safety.
