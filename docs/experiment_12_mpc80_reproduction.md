# Experiment 12 — accepted MPC80 feasibility reproduction

This finalizes E12_MPC80_FEAS_v1's failed execution, both reviewed preflights and
E12_MPC80_FEAS_v2's successful scoped execution. It does not supersede the original
FAILED overall E12 qualification or FAILED 0.001 symbolic/PyBullet criterion.
The original scientific reports, source/configuration seals, failed reservations,
accepted ledgers and generated evidence remain immutable.

Use Python 3.10.19, `requirements_experiment_12.txt`, the original pinned native
binaries, and safe-control-gym commit `6b5391d014f36fdfa0f9d22d92c77387e5274308`.
See the original [environment reproduction](experiment_12_reproduction.md).
Generated scientific prerequisites are intentionally absent from Git. This is
saved-record verification, not a request to regenerate scientific trajectories.
No IQL source is imported or needed for numerical checks; the original nine
untracked files are preservation targets only in the external data root.

## Read-only entry point

From a source checkout or clean source export, make the original archives
available at `results/` (a read-only data link suffices), then use a fresh output:

```sh
/private/tmp/e12_venv/bin/python experiments/12_verify_mpc80_milestone.py \
  --data-root "$DATA_ROOT" \
  --upstream /private/tmp/e12_safe_control_gym \
  --output /private/tmp/mpc80_record_verification.json
```

`DATA_ROOT` is the original scientific repository, including its Git history and
accepted archives. The source checkout supplies implementation/configuration files;
the original data root supplies preserved results, datasets and nine IQL hashes.
The verifier refuses existing outputs and outputs inside the accepted MPC80
archives. Optimization constructors and solve APIs are blocked; PyBullet is not
imported. It validates the sealed 87,315-file v2 evidence manifest, then independently
reconstructs costs, thrust effort, all raw margins, dense source equations,
submitted commands, retained horizon-80 programs and all 9,600 KKT candidates.
It verifies recorded bitwise scientific repetitions and both replay types; it
does not run new simulator replays. Complete scientific JSON must equal the
accepted audit, including its still-failed original gates.

Only `CURRENT_STATUS.md`, `RESEARCH_CONTEXT.md`, and the decision log may differ
from historical preservation hashes. Every exception is proved from the exact
original Git bytes; scientific sources/results have no exception. No historical
manifest is rewritten to accept a newer file. Context edits and the later separate
decision-log commit are authorized research records, not changes to experimental
methodology.

Run the applicable regression groups using the existing two environments:

```sh
/private/tmp/e12_venv/bin/python experiments/12_regression.py \
  --group quadrotor --output /private/tmp/mpc80_quadrotor_tests.json
"$DATA_ROOT/.venv/experiment08/bin/python" experiments/12_regression.py \
  --group prior --output /private/tmp/mpc80_prior_tests.json
```

Exclude `test_offline_rl.py`. New action-observer tests use scripted native Python
control flow on a mock backend: importing PyBullet for its module definitions is
not engine creation or physics integration. No native MPC solve occurs.

For clean export, copy only Git's intended source/configuration/tests/documentation
set to a fresh directory, without unfinished IQL files, generated datasets/results,
checkpoints or environments. Supply read-only `results`/`data` links for fixtures
that locate archives relative to the source root. Run both regressions and the
record entry point there. Compare the complete canonical scientific JSON and SHA
against workspace verification, excluding only verification timing/paths.

## Accepted findings and resource boundary

V1 remains failed after one actual solve/step with its full 300-solve / 6,000-step
reservation. V2 records the current physics RPM and force/torque submission
arguments rather than the late-updated command cache. Derived thrust is not
hardware or internal-engine sensing. The full sixteen-state v2 cohort passes:
32 scientific executions, 64 replays, 9,600 accepted candidates, zero observed
state/input violations or contacts, all 6,001 raw states/run checked, bitwise
repetitions/replays and positive causal numerical reserves.

Mean/worst return is −20.241625399 / −78.787551231, position RMSE
0.087969885 / 0.197426305 m, and squared-thrust effort
0.228677863 / 0.242540074 N²s. Minimum normalized raw margin is 0.0756528087895.
Maximum prediction disagreement remains 0.0189680501631. All 9,600 measured native
selection estimates exceed 20 ms. This is finite-cohort offline expert feasibility,
not a robust/hardware/continuous-time safety certificate or real-time readiness.

Accepted execution accounting remains 9,600 solves / 576,000 physics steps and
1,501.732 shared CPU / 1,730.196 active-wall seconds. Remaining original solve/step
allowance is **3,070 / 366,000**. Finalization consumes no solves/steps. Its CPU/wall
and verification artifacts are recorded additively under
`results/benchmark_selection/mpc80_feas_finalization/`; they do not reset or alter
the accepted ledger. The old 1,800/3,600-second local ceilings remain the reference
for reporting carried headroom, not an automatically renewed execution allocation.

The original sixteen states are qualification/development data, never independent
held-out learned-policy evaluation. Action diversity, beyond-expert headroom and
active-versus-random acquisition remain untested and require separate approval.
The [accepted report](experiment_12_mpc80_feas_v2_report.md) retains all per-state,
componentwise margin/reserve, latency, source-hash and numerical diagnostics.

## Finalization verification results

Workspace and clean source export each pass **248/248 tests** (126 quadrotor and
122 prior-experiment tests), zero failures/errors/skips. Mock action-path tests
import native module definitions but do not instantiate a physics engine or solve
an MPC problem. Both independent record audits produce exactly the accepted
scientific JSON, SHA-256
`6ada71395cb84324cad4f6c28638b200b8b82cc52daab188564af895d62a2f6c`.
Each verifies 87,315 sealed v2 files, 9,600 retained candidates, all 32 scientific
records and 64 replay records. Preservation also checks 197 prior tracked files,
59,268 prior results/data files, 87 drilling authority files, 289 upstream files,
5,438 historical E12 stage artifacts and all nine IQL hashes. The clean source
export has 227 files, with no generated scientific data or IQL source copied.

The initial new finalization comparator rejected a bookkeeping difference: the
accepted scientific report snapshots usage before its two post-run read-only
checks, while the final ledger includes their charges. The new verifier now
proves the exact +32.9/+11.8 CPU and +51.3/+29.4 wall charge reconciliation before
comparing the original snapshot. No scientific record, accepted ledger or
experimental verifier was changed. The failed finalization comparison is retained;
its missing exact execution timing is disclosed and conservatively charged at
90 CPU / 180 wall seconds, not reported as recovered telemetry.

Successful record checks used approximately 42.08 and 40.31 CPU seconds. Including
tests, conservative startup/documentation allowances and the failed comparison,
combined carried accounting is **1,711.202 CPU / 2,182.658 active-wall seconds**,
within the old 1,800/3,600 reference ceilings. These additive finalization charges
are external to the immutable accepted ledger; no new execution allocation is
implied. There were **zero new solver calls and zero new physics steps**.
Finalization evidence and exact export comparison remain outside Git under
`results/benchmark_selection/mpc80_feas_finalization/`.

The staged whitespace check flags one pre-existing final blank line in the sealed
`src/quadrotor_applied_action_v2.py`. It is deliberately retained to preserve its
reviewed source hash; there are no other staged whitespace findings.
