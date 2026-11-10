# Experiment 12 — Pilot B stopped before dispatch

**Status: NOT STARTED — runtime identity gate FAILED.** The explicit first-failure
rule was applied. No nominal or alternative scientific execution, replay, MPC solve
or physics step occurred. No restoration, replacement runtime, dependency change,
controller repair or automatic retry was attempted. The original FAILED overall
E12 qualification and FAILED 0.001 symbolic/PyBullet criterion remain unchanged;
the previously accepted MPC80_FEAS_v2 result remains historical accepted evidence.

## Approved scope and failed prerequisite

The November 9 approval permits only state ID 0, menu 4 nominal twice followed
by menu 3 alternative twice, each with native and motor-command replay: four
scientific executions, eight replays, at most 1,200 new solves /72,000 physics
steps, 300 incremental CPU /600 active-wall seconds, one worker and 4 GiB.
First-failure stopping and all frozen criteria apply before dispatch as well.

The accepted counterfactual preflight seal, its nine source/configuration/report
files and all 23 sealed artifacts match their stored SHA-256 identities. The
frozen command values, original 36-label accounting and selection remain unchanged.
However the external pinned native runtime is no longer intact:

- `/private/tmp/e12_safe_control_gym` retains directories but **all 289 expected
  upstream files are absent**, including controller code, simulator code, URDF and
  task/configuration files. No existing expected source file could be hashed.
- `git -C /private/tmp/e12_safe_control_gym rev-parse HEAD` exits **128** with
  `fatal: not a git repository (or any of the parent directories): .git`.
- `/private/tmp/e12_venv/bin/python -c 'import sys; print(sys.version)'` exits **1**.
  It cannot initialize filesystem encoding: `ModuleNotFoundError: No module named
  'encodings'`. The expected venv configuration and base `encodings/__init__.py`
  are absent. An interpreter executable/symlink alone is not the approved environment.

The cause of this external file loss has not been established. No environment
version equivalence, solver availability or native model identity can be asserted
from the remaining directory names. The available system Python was used only for
read-only file hashing and failure bookkeeping, never as a scientific substitute.
No frozen MATLAB, quadrotor, controller, objective, constraint or tolerance changed.

Full interpreter output, missing-file list, native Git failure and exact approval
are retained under `results/benchmark_selection/counterfactual_pilot_B_v1/`.

## Scientific results and classifications

| Required outcome | Completed | Classification |
|---|---:|---|
| Nominal scientific executions | 0/2 | Not started |
| Nominal replays | 0/4 | Not started |
| Alternative scientific executions | 0/2 | Not started; nominal gate never reached |
| Alternative replays | 0/4 | Not started |
| New optimization candidates | 0 | No candidate generated |
| New physics steps | 0 | No new trajectory generated |

Nominal fidelity, live reset/context equivalence, KKT correctness of new calls,
raw safety, contacts, numerical reserves, deterministic repetition and replay
agreement are **not evaluated**. Neither branch may be classified empirically safe
or unsafe. The alternative's earlier static input admissibility/distinctness is
unchanged, but its live continuation remains untested. Return, tracking-error,
effort, raw-margin differences and actual submitted separation are unavailable.
No solver-only, native-action or supervised latency samples or deadline overruns
were generated; zeros would be fabricated measurements. The accepted 154-test
preflight remains preserved historical evidence; its pinned-environment regression
was not rerun because that interpreter cannot initialize.

## Resources and preservation

No job was dispatched or reserved: **zero reserved/actual solves and physics
steps**, with no failed scientific reservation to refund. Original remaining
allowance stays **3,070 solves /366,000 physics steps**.

Starting accounting adopts the approved upward-rounded **1,744.402 CPU /
2,274.158 active-wall seconds**, rather than decreasing the exact accepted
preflight values (1,744.401929 /2,274.1579532499345). The approved cumulative ceilings
are sealed at **2,044.402 CPU /2,874.158 active-wall seconds**. Read-only diagnosis,
preservation hashing and conservative preparation/reporting allowances add
**20.5 CPU /70.9 active-wall seconds**, yielding **1,764.902 /2,345.058 seconds**.
The unspent increment is 279.5 CPU /529.1 wall seconds; it is an accounting balance,
not permission to resume after the failed gate. Historical ledgers remain untouched;
new accounting is additive in this stopped-attempt directory.

Preservation passed for all nine sealed preflight source files, 23 preflight
artifacts, 227 tracked project files, 87,315 accepted v2 artifacts, 59,268 earlier
data/result files, 87 authoritative drilling files and nine unchanged IQL hashes.
Archived evidence remains outside Git. No prior research file or decision-log
entry was modified. No commits or pushes occurred.

An additive execution-wiring draft was started before the external runtime loss
was detected. `src/quadrotor_counterfactual_execution_v1.py` is **untested, unsealed
and unexecuted**; it is not represented as a qualified live runner. Implementation
work stopped at the failed prerequisite. This report records no completed Pilot B
scientific result and does not weaken any acceptance criterion.

## Next decision required

Review restoration of the **exact** pinned source/runtime from preserved artifacts
and zero-solve identity/revalidation checks. Do not silently install a different
version, substitute another environment or retry the stopped campaign. Any later
execution needs separate authorization after source/dependency identities and
execution wiring tests pass. No other states/menu items, debugging simulations,
acquisition, RL/IQL training or drilling investigations were run.
