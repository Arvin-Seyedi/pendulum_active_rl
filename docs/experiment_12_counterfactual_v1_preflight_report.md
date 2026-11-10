# E12_MPC80_COUNTERFACTUAL_v1 — zero-solve preflight

**Static/mocked preflight PASSED. Execution is NOT approved. Recommend pilot B,
subject to separate execution permission and a carried CPU-ceiling extension.**
No actual optimizer solve, engine creation, physics step, or counterfactual
continuation occurred. The original overall E12 and 0.001 agreement gates remain
FAILED; accepted MPC80_FEAS_v2, v1 failure and prior research remain unchanged.

## Exact admissibility results

All 96 original manifest binary64 coordinates, ID ordering and PCG64-120001
provenance were verified directly. No state or menu label was regenerated.
Original probe IDs remain 0/4/8/12, at step zero. All 36 labels are retained.

| State ID | Input-inadmissible menu indices | Affected channel (zero-based) |
|---|---|---|
| 0 | 0, 1, 2 | 0 |
| 4 | 0, 3, 6 | 1 |
| 8 | 0, 1, 2 | 0 |
| 12 | none | — |

Thus **9/36 fail the unchanged input contract; 27/36 pass the static input check**.
All nine command signatures per state are distinct: no duplicate labels are
removed. Bitwise distinctness does not imply the original 5% practical separation.
Neither input admissibility nor mock data establishes safe state continuation.

Each flagged request clips to **0.056323375552892685 N**
(`0x1.cd66ae0000000p-5`) but the native conversion/submission arithmetic gives
**0.05632337555289267 N** (`0x1.cd66adffffffep-5`). The exact lower margin is
**−1/72057594037927936 N = −2^−56 N = −1.3877787807814457e−17 N**,
two binary64 ULPs below the bound. Zero negative-margin tolerance means failure;
no epsilon, inward correction, bound change or replacement is introduced. This
is a source/mock numerical input-contract finding, not an observed flight violation
or evidence of physical danger.

Independent pinned native clipping, PWM/RPM and force/torque arithmetic matched
**720 scripted native application boundaries** (36 labels ×20). Submitted force
arithmetic uses the native vector operation order, not a reassociated scalar
approximation. All four archived nominal first RPM/force/torque records also
matched bitwise. No candidate-dependent state was integrated.

## Prepared infrastructure and test evidence

Reusable versioned helpers provide native factory/reset ordering, passive reset
context capture, original-query archive parity, detached one-step substitution,
classification and additional failure records. The CLI has **no live execution
mode**. A future approved dispatcher must combine them with the unchanged native
candidate guard/KKT checker, v2 command observer, raw journal, replays and explicit
job reservations. No controller, plant, asset, solver option or tolerance changed.

Preserve the accepted ordering: construct the plant, construct/reset the controller
through the native factory, then reset the plant once. Require actual archived
reset raw-state bytes and the original 1e−14 request/reset tolerance; phase zero,
empty x_prev/u_prev and reset command cache are checked. Passive full-base,
quaternion/velocity, joint/contact, cached-array/counter and RNG capture uses no
forward-kinematic refresh or state restoration.

The verified original query is followed by a parity gate, then substitution.
First-query primal/dual/parameters, initial guess/history, observation, reference
preview, requested action and post-query x/u plan must match the archive. Nominal
branches demand archive parity throughout; alternative branches demand it at the
first query only. Changed alternative successor states are expected, not falsely
compared with the old nominal path.

The returned action aliases u_prev[:,0]: it is copied before offset addition.
Nominal post-query x_prev/u_prev/prev_action and optimizer parameter/initial-guess
state remain unchanged. Original x-column and u-row warm-start shifts are retained.
The 6×81 preview spans 1.6 s; terminal padding begins at index 221. Steps
0, 220, 221, 299 and 300 were checked explicitly.

Direct native RPM arguments, forwarded force/torque arguments, requested/clipped
thrust and derived KF*RPM² thrust remain separate. The v2 observer matches the
current command to its physics callback, including first-after-reset and later
transitions; last_clipped_action is stale debug context, not the applied input.
Deduplication is by actual submitted force/torque signature, with direct motor
records retained and earliest original labels preserved as representatives.

**154/154 applicable static tests pass**, zero failures/errors/skips and zero
forbidden real solver/engine calls. Native Python MPC flow uses archived solution
lookups; mocked physics only advances a scripted-state array index. All four
300-step nominal sequences match unwrapped native flow in commands, parameters,
guesses and memory, including **1,200 archived initial-guess comparisons**.
Independent program/KKT/reference checks pass for **20 retained candidates**;
all 20 archived native statuses are Solve_Succeeded. Tests reject changed duals,
guesses, observations/previews, nonfinite values and incorrect reset context.
Zero-offset backend traces/scripted state bytes remain unchanged.

Candidate-specific contract wiring is tested with inert predictor spies/archived
predictions: the altered request, not nominal forecast, reaches the contract.
No new alternative-state model integration or future-safety evaluation was done.
Failure tests retain partial arrays and explicit nonfinite diagnostics without
fallback; existing candidate capture retains available primal/dual/status/timing
before parity or application checks can reject the action.

**Limits:** these tests do not establish live trajectory equivalence. Historical
archives lack complete PyBullet internal checkpoints/full-3D/cache snapshots.
Exact identity of unrecorded opaque engine or solver internals cannot be certified.
Native fresh resets, prospective passive snapshots, first-query parity and live
nominal/replay agreement define operational matching. Allocation IDs and unused
RNG metadata must be distinguished from physically/numerically relevant context;
no new seed changes or randomization are introduced. Instrumentation passivity
still requires live nominal versus uninstrumented/motor replay verification.

## Frozen alternative and full-study criteria

Freeze **state ID 0, menu index 3**: first lexicographic nonzero, nonduplicate,
current-input-admissible ID-0 item with exact submitted-thrust separation >=1/20.
Indices 0–2 fail current input admissibility. No continuation return/safety outcome
was used, and no substitute alternative is permitted if this one fails.

- Native nominal request: [0.056324369506392545, 0.14865930279665016] N.
- Offset: [0, −0.012017975933849812] N, preserving the original nine-action menu.
- Detached request/clipped command: **[0.056324369506392545, 0.13664132686280034] N**.
- Native predicted RPM: **[9440.383351674363, 14703.898500466665,
  14703.898500466665, 9440.383351674363]**.
- Submitted motor z forces: **[0.028162184753196273, 0.06832066343140016,
  0.06832066343140016, 0.028162184753196273] N**, transverse components zero,
  links 0–3, native LINK_FRAME application points [0,0,0].
- Native submitted link-4 torque: **[0,0,1.0842021724855044e−19] N·m**. The
  cancellation residual is preserved rather than rounded to zero.
- Submitted grouped thrust: **[0.056324369506392545, 0.1366413268628003] N**.
- Exact normalized separation: **39363019590861/787260391817216
  = 0.05000000000000025**, strictly >=1/20; both input channels have positive margins.

Decimal/float.hex/little-endian representations are frozen in
`frozen_command_representation.json`. Runtime direct submissions must confirm them;
these source-derived values are not yet live alternative telemetry.

Original full-study criteria remain: three of four states with two additional
verified-safe actions satisfying 5% separation; >=10% mean cost reduction;
zero violations; maximum normalized-margin loss 0.01; per-state cost deterioration
<=5% of comparator mean cost, against the strongest matched classical baseline.
An MPC80-only oracle does not establish that strongest-baseline claim. Neither
technical pilot can pass the three-of-four criterion or establish beyond-expert
improvement. Safe alternatives, performance gains, incomplete/numerically failed
branches, empirical violations, duplicates and best-of-menu/fallback oracles
remain separate classifications.

## Pilot budget comparison and recommendation

One fresh nominal query plus 299 continuation queries = **300 solves per scientific
run**, with **6,000 physics steps**. Each native or motor replay adds 6,000 physics
steps and no solves. Original/repeat trials each use fresh native construction.
Reserve each job before dispatch; failures retain their full reservation.

| Scope | Scientific runs | Replays | Solves | Physics steps | Original balance afterwards |
|---|---:|---:|---:|---:|---|
| A: four nominal pairs plus one ID-0 alternative pair | 10 | 20 | 3,000 | 180,000 | 70 solves /186,000 steps |
| B: one ID-0 nominal pair plus one ID-0 alternative pair | 4 | 8 | 1,200 | 72,000 | 1,870 solves /294,000 steps |

**Recommend B as the minimum machinery pilot.** It exercises both zero/nonzero
substitution, initial-query context, continuation memory, direct telemetry and
both independent replay types. A adds live baseline/reset-context verification
at IDs 4/8/12; it adds no alternative-action evidence there. Static tests already
cover all four archival controller sequences. Broader live context coverage is
useful before a full study but not necessary for the smallest execution-fidelity
pilot. Neither scope implies population safety or a qualified operating region.

B's frozen order is menu 4 original →native replay →motor replay →repeat →both
replays, then the same six jobs for menu 3 only after all nominal checks pass.
The 12 B jobs and 30 comparison A jobs are frozen individually, with no live job
dispatched or reserved. No cached-first-query solve discount is assumed.

Observed mean CPU rates imply about **156 s (B)** versus **389 s (A)** before
extra setup/verification/failure overhead; these are estimates, not guarantees.
Proposed incremental envelopes are **300 CPU/600 active-wall seconds (B)** and
**600/900 (A)**, retaining one worker, 4 GiB, unchanged 1-second selection watchdog
and reporting every 20-ms overrun. No hard real-time readiness is claimed.

Preflight cumulative accounting is **1,744.401929 CPU /2,274.157953 active-wall
seconds**, carrying prior conservative charges and including source-preparation,
supplementary checks and reporting allowances. Peak measured RSS is
**499,073,024 bytes**. No accepted ledger or v1 reservation was changed/refunded.
The exact original solve/step allowance remains **3,070 /366,000**.

Only **55.598071 CPU seconds** remain below the old local 1,800 ceiling; neither
pilot fits it. B requires an explicitly approved cumulative CPU ceiling of up to
**2,044.401929 s**; A up to **2,344.401929 s**, before any additional future preflight
charges. Their prospective wall totals are 2,874.157953 /3,174.157953 s, both within
the old 3,600 wall ceiling. This is a carried-accounting proposal, not a reset or
an approved allocation. Current final_resource_accounting.json is the carry for
any subsequent approval, and all intervening work must also be charged.

## Preservation and execution go/no-go

Preservation passed for **87,315 sealed v2 artifacts; 59,268 prior data/results;
227 currently tracked files; 87 authority files; 289 upstream files; 5,438 earlier
E12 stage artifacts; and all nine IQL hashes**. Existing tracked files and the
accepted resource ledger remain unchanged. Historical context updates are proven
against original Git bytes. Generated evidence stays outside Git; new source,
configuration, tests and these documents are uncommitted. No commit/push occurred.

**GO only to separate approval review for pilot B; execution remains NO-GO now.**
Before dispatch, require current seals/dependencies, complete journals and budget
receipts plus explicit execution/CPU approval. Nominal original/repeat must match
all archived commands/candidates/raw states bitwise and pass both replay types
before any alternative. Both future branches require all 6,001 raw states, strict
nonnegative state/input margins, zero contacts, native success/corrected KKT,
finite records, their own causal positive reserves, source/contract closure
<=1e−10 and bitwise repeat/replay agreement. No negative margin is accepted.

The alternative must reproduce the original first query and retain nominal
post-query memory, then follow its own unchanged MPC continuation. Any numerical,
context, telemetry, safety, replay or budget failure stops the pilot with partial
records, no fallback/retry/replacement or threshold adjustment. Incomplete is not
safe; solver failure or a failed reserve is not physical nonviability.

Successful B validates operational execution fidelity at one frozen context, not
the original 3-of-4 action-support criterion, safe arbitrary action sequences,
meaningful learning headroom or beyond-expert performance. Full-study design and
practical criteria stay unchanged. No other probes, acquisition, RL/IQL training
or drilling-model investigation were implemented or executed.
