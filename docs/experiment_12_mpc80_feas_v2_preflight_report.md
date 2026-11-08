# E12_MPC80_FEAS_v2 — instrumentation correction and static preflight

**Zero-solve preflight PASSED. Fresh execution remains unauthorized.** No real
quadrotor episode, engine creation, physics step, optimizer solve, action probe
or training was performed. V1 remains a FAILED execution with its original
one-solve/one-physics-step evidence and full failed reservation unchanged. The
original FAILED overall E12 qualification and FAILED 0.001 model-agreement gate
are not reclassified.

## Root cause and correction

The pinned action path is requested MPC thrust -> `BenchmarkEnv.before_step` ->
native `_preprocess_control` -> physical command/clipping -> PWM/RPM conversion
-> `_advance_simulation(rpm)` -> `_physics(rpm)` -> four motor-link force calls
and one torque call -> physics integration -> diagnostic callback -> update of
`last_clipped_action`. In `base_aviary.py`, that cache is zero-initialized at line
201, `stepSimulation` is called at line 282, and the cache assignment follows at
line 284. V1 checked the cache before that assignment and therefore read reset
zeros on its first callback. Its negative diagnostic did not establish a true
thrust-bound violation.

The separate `AppliedActionObserver` copies the current argument delivered to
native `_physics(rpm)`, records the forwarded force/torque API arguments, and
associates this packet with the target drone/client and next successful physics
step. The callback receives that packet. It never uses the stale cache as the
executed-input source. V1 source and artifacts were not repaired.

The observer forwards original argument objects exactly once, retains native
return objects/order, does not update cache/kinematics, touches no controller or
optimization input, and restores hooks after normal/error exits. Missing or
ambiguous application records fail closed without a fallback action. Logging
and callback failures retain the application/step distinction.

Direct records are the native preprocessing output, `_physics` motor-command
argument and submitted per-link force/torque arguments. `KF*rpm²` thrust and
grouped force sums are labelled derived quantities. These are command/submission
observations, not sensor readings of hardware RPM or internal Bullet net forces.
The direct source is available before integration, including the first step and
subsequent command transitions. V1's missing direct RPM record remains missing;
no equation reconstruction has been inserted as historical telemetry.

## Independent checks

| Check | Outcome |
|---|---|
| V1 stale-action regression | PASS: pinned native Python control flow on a mock backend reproduces the first-step zero-cache failure with the unchanged v1 input check. |
| Corrected first/reset and transition timing | PASS: current motor arguments, not reset/previous-command cache values, reach diagnostics at all tested boundaries. |
| Native pass-through invariance | PASS in paired mocks: force/torque/step call traces, command values, argument/return identity, cache updates and scripted state bytes match. Hooks restore; global RNG state is unchanged. |
| Observability/error handling | PASS: missing application/preprocessing, invalid motor data, incomplete force application, backend failure and callback failure are distinguished and retained. Unsupported disturbed/non-PYB contexts are rejected. |
| Archived application-boundary cross-check | PASS: 160 observed mock boundaries across the first two archived commands of H0–H3; 640 force and 160 torque submissions observed. Native control flow is real source code; the backend is scripted, with no equations integrated. Baseline mocks execute the matching control flow separately. |
| Actuator/source recurrence consistency | PASS: direct mock submission values compared with genuine archived RPM/state data; maximum matched-state source-equation discrepancy 8.881784197001252e−16. Grouped force sums and KF-derived thrust agree on these records. |
| MPC80 numerical definitions | PASS: 2,400 accepted archived candidates satisfy unchanged v2 KKT checks. Full independent/native graph discrepancy <=1.4210854715202004e−14. No new graph/controller solve. |
| Unchanged discrepancy contract | PASS: 7,200 archived intervals and 43,200 exact-rational components; source closure <=2.6645352591003757e−15, maximum bound excess 4.237264222008711e−15 within the existing 1e−10 numerical allowance; all checked reserves positive. |
| Regression | PASS: 120 zero-solve tests, zero failures/errors/skips, including 18 observer and five v2 protocol tests. All real engine-creation/force/step and optimizer APIs were guarded; no forbidden calls occurred. |
| Reproducibility | PASS: fresh-process repeat yields identical canonical scientific JSON, SHA-256 `9fa5bc03735306255c0243ebb044157bdcf0bf63b472a5f205dbd7a875d7fd40`. |
| Preservation | PASS: 197 tracked files, 59,226 prior result/data files, 87 authority files, 289 upstream files, 5,438 sealed E12 artifacts and nine IQL hashes preserved. Supplemental checks verify 13 parent source/report identities, including all failed v1 execution sources. |

These mocked and archived checks establish the timing and pass-through contract.
They do **not** empirically measure new live v2 scientific-trajectory invariance.
That would require fresh, separately authorized execution and independent replays.
No full-horizon v2 performance, safety or latency result exists. The source-based
mismatch checks are not formal robust/continuous-time/physical safety guarantees.

## Frozen identities and unchanged scientific protocol

Upstream: `6b5391d014f36fdfa0f9d22d92c77387e5274308`.
V2 protocol seal: `920a6281aa0abd007d7a838ccefba393c5aa8f2d1e767939179958b06fadf969`.
Configuration SHA-256:
`91a17c5d45c537d0cc59672b36bd1a2f7d508f6e554538039635b8bc222a3839`.
Observer SHA-256:
`66e484bdd592468d00ee24b91c8264e96d89a1c32c466888131ae8691d152c77`.
All nine source/configuration/dependency identities are in `protocol_seal.json`;
their exact bytes are preserved in source snapshots.

Python 3.10.19, CasADi 3.7.2, PyBullet 3.2.7, all archived dependency versions,
core binary identities, vehicle assets and model constants are unchanged. The
original manifest/array hashes remain
`0f13eaf7313623d5c77f8851b090dab147aab3aae95b346307242e04869b9bd9` /
`c97f1fed3fcdd17a17de70d61042bef0ce7e0323a2312e38000d1a788fbb4f17`.
All 96 binary64 coordinates and IDs/order verify directly against those files.
Their historical metadata are not rewritten to conceal v1's partial state-0 attempt.

Automated inheritance checks retain the full v1 acceptance dictionary, stopping
rule, horizon 80, six-second task, 300 controls, 6,001 raw states, native warm
starts, objective/constraints/reference semantics, seed 1337, ascending IDs,
original/repeat ordering and both replay types. Negative raw margins still have
zero tolerance. No altered MPC solver settings or actuator limits were introduced.
The cohort remains qualification/development data, not independent learned-policy
evaluation. The static entry point has no cohort execution mode.

## Resources and readiness

V1's failed **300-solve/6,000-step** reservation is retained despite actual work
of one solve/one physics step. Original remaining allowance is **12,670 solves /
942,000 physics steps**. V2's unchanged prospective 96-job envelope of **9,600 /
576,000** fits, leaving **3,070 / 366,000** if eventually reserved. No v2 job is
dispatched or solve/physics-debited, and no prior reservation is refunded.

V2 carries v1's complete charged balance, approximately **102.23 CPU seconds /
177.169 active wall seconds**, into the same **1,800 / 3,600-second** shared caps.
The caps were not reset. After the two static passes and supplemental checks,
shared charges were **158.23 CPU s / 270.569 wall s**, with final integrity/report
overhead accounted separately in the live v2 ledger. One worker is used. The
largest conservative sum of recorded parent/child process peaks was
**837,419,008 bytes**, below 4 GiB. Runtime and logging costs of a real v2 cohort
remain unmeasured and must fit the remaining shared caps; no extra allocation
is granted by these estimates.

**No static numerical, provenance or command-boundary observability blocker was
identified for separate execution review in the pinned environment.** Internal
engine force sensing and new live trajectory invariance remain outside this
preflight's evidence. Mandatory fresh repetition/replay and first-failure checks
cannot be waived on the basis of mocks. V2 is not an approved or qualified expert
execution, and v1 cannot be resumed.

Evidence is under `results/benchmark_selection/mpc80_feas_v2_preflight/`, separate
from all v1 records. Static reproduction uses
`experiments/12_mpc80_feasibility_v2_preflight.py --upstream
/private/tmp/e12_safe_control_gym --preflight --output <fresh-directory>`;
`--verify-static` performs a fresh-process read-only repeat on its completed
preflight. Existing attempts are never overwritten. New baseline preservation
counts can differ for a new output directory; numerical input identities remain
fixed. No simulation, solver execution, commit or push was performed. Stop for
scientific review; fresh cohort execution requires separate explicit approval.
