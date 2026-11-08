# E12_MPC80_FEAS_v2 — instrumentation correction, zero-solve preflight only

No fresh cohort execution is authorized. Preserve the v1 FAILED execution,
one-solve/one-physics-step prefix, source identities and full 300-solve/6,000-step
failed reservation. Neither the original FAILED overall E12 qualification nor
the FAILED 0.001 symbolic/PyBullet criterion is superseded.

V2 inherits the exact original sixteen-state manifest (PCG64 120001, 96 binary64
coordinates and IDs/order), MPC80, six-second task, 50/1,000-Hz timing, native
objectives/constraints/solver/warm starts/reference information, 32 scientific
executions, both replay types and every acceptance threshold/stop rule from v1.
No new initial conditions, counterfactual actions or controller comparisons.
These states remain qualification/development data; v1 already partially attempted
state 0. They are not fresh held-out learned-policy evaluation data.

## Exact instrumentation correction

Native path: requested action -> `BenchmarkEnv.before_step` -> native
`Quadrotor._preprocess_control` (physical command, clipping, PWM/RPM conversion)
-> `BaseAviary._advance_simulation(rpm)` -> `_physics(rpm)` -> submitted motor-link
forces/torque -> `stepSimulation` -> diagnostic callback -> native update of
`last_clipped_action`. The latter is reset to zero and is not current at the
callback boundary. V1 remains failed; no historical RPM array is filled in.

The separate `AppliedActionObserver` records a detached copy of the actual native
`_physics(rpm)` argument before forwarding the original object exactly once.
It records the original forwarded `applyExternalForce`/`applyExternalTorque`
arguments and return/error status. A packet is associated with the same target
drone/client and next physics step. The post-step callback receives this packet,
never the stale cache as the applied-input source. Missing, ambiguous or incomplete
application records fail closed; no fallback or altered motor command is provided.

Preprocessing-returned RPM, applied motor-command argument, submitted link forces
and torque arguments are direct API observations. Grouped thrust from `KF*rpm²`
is explicitly derived; grouped sums of submitted force arguments are separately
labelled. None is a sensor reading of internal Bullet force integration or
hardware motor speed. `last_clipped_action` is optional stale debug context only.
The observer does not initialize/update that field, call a kinematic refresh,
change frames/assets, touch controller/optimization inputs, or integrate a model.

Install the observer inside the existing raw-physics capture context so native
integration and raw capture finish before the qualification callback. Successful
calls preserve original argument objects, ordering, numerical values, return
objects, cache-update timing and stepping. Hooks are restored on normal/error
exit. Logging/validation failures still stop supervision under the inherited rule.

## Allowed validation and its limits

All real optimizer, engine creation, force-submission and physics-step APIs are
blocked. Deterministic tests execute the pinned native preprocessing/application
control-flow methods against a mock backend. Its `stepSimulation` advances only
an index through archived/scripted rows, with no equations integrated. Compare
unobserved/observed native call traces, return/input identity, cache updates,
scripted state bytes, first reset step and subsequent action transitions.
Reproduce v1's failure with its unchanged check at the old callback boundary.

Use archived command/RPM/state records for force-argument and independent
matched-state source-recurrence checks; retain mock provenance explicitly. Do not
call reconstructed signals new live telemetry. Recheck sealed MPC80 graphs,
candidate acceptance and the unchanged mismatch contract on saved vectors only.
Past v1 live applied RPM remains unavailable. A successful equation reconstruction
cannot replace a missing direct record. Live scientific-trajectory invariance
of v2 cannot be empirically measured in this authorization; future approved
repetitions/replays remain mandatory. No formal robust or hardware safety claim.

## Resource reconciliation

The original conservative balance after v1 is **12,670 solves / 942,000 physics
steps**. The unchanged prospective v2 plan needs **9,600 / 576,000**, leaving
**3,070 / 366,000**. This is a proposed envelope, not dispatched work or permission.
No failed reservation is released. Carry v1's entire charged CPU/active wall
balance into v2 against the same **1,800/3,600-second** ceiling, rather than
resetting either cap. Charge v2 tests, preflight, repeats and verification too.
One worker and 4 GiB remain. The historical project CPU ledger is not silently
represented as an exact post-finalization total; shared local caps are stricter
than its recorded remaining headroom.

Seal source/configuration/asset identities and the version-specific ledger before
any outcomes. New results go only under `mpc80_feas_v2_preflight/`; no accepted
v1 output or source is writable. The entry point exposes static preflight and
archived re-verification only. A passing report supports a separate execution
decision; it neither resumes v1 nor authorizes any fresh cohort, probe or training.
