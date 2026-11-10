# Experiment 12 — Pilot B v2 final zero-solve wiring review

Technical readiness: **GO for fresh, separately approved Pilot B execution**.
Scientific execution remains **PAUSED**. This review performed zero real optimizer
solves, zero physics steps and zero scientific/replay executions. No execution
approval or live campaign directory was created. Dry receipts belong to inert
fixtures and did not consume the original scientific reservation ledger.

## Scope and frozen comparison

Parent counterfactual preflight seal SHA-256:
`facb3a19264f111290c33982bfd959f237bf6bae60ad0ddaee470206ecec934d`.
Frozen state ID 0, reference step zero, nominal menu 4 and alternative menu 3
remain unchanged. The stored 16-state/96-coordinate manifest, task, reference
preview, native MPC80, IPOPT settings, plant/URDF, objectives, constraints,
acceptance criteria and all historical source files were preserved.

The alternative request is `[0.056324369506392545, 0.13664132686280034]` N;
its little-endian binary64 bytes are
`73dcfe398cd6ac3fee311487767dc13f`. Submitted grouped force is
`[0.056324369506392545, 0.1366413268628003]` N. The original exact normalized
separation is `39363019590861/787260391817216 = 0.05000000000000025`,
meeting the unchanged >=5% rule. Preserve the tiny nonzero yaw-torque submission
`1.0842021724855044e-19` N m. All 36 original labels match, including nine
source-predicted input-inadmissible labels. No alternative was reselected.

## Identified orchestration defects and bounded corrections

The retained unexecuted v1 draft referenced a missing execution entrypoint and
had no complete dispatch supervisor. Its frozen-command check ran after the
trajectory, and constructor exceptions outside the inherited episode try-block
could escape without a complete experiment failure record. Its JSON reset-context
comparison treated signed zeros as equal. These are demonstrated wiring gaps;
they do not explain the historical missing-runtime stop or establish a control
failure. The original v1 draft and stopped report remain bytewise unchanged.

The separate v2 entrypoint, dispatcher and wrapper now:

- Verify exact pinned runtime/source identities before prospective dispatch;
  require fresh seal/report-bound approval and explicit cumulative ceilings.
- Preserve native plant construction, native controller construction/reset and
  one plant reset. Guard and capture the original native query before archive
  parity, detached substitution, candidate-specific contract and native stepping.
- Scope the solve observer to the one expected Opti instance; do not change its
  solver or optimization program. Native post-query memory, including x-column
  and u-row warm-start shifts, is preserved. No second query is introduced.
- Journal current direct motor/force/torque submissions before physics. Check
  actual/derived input bounds and every first-interval alternative command byte
  before integration; retain packets and incomplete records on rejection.
- Compare declared reset-context floating-point bytes, including signed zeros.
  Preserve available startup/execution failures, candidate records, nonfinite
  diagnostics and partial histories without fallback or retry.
- Reserve every one of the original 12 jobs before dispatch, enforce a six-job
  nominal archive/repeat/replay gate before any alternative reservation, and
  stop on the first failure without releasing its reservation.
- Charge startup, preparation, verification, child CPU, replay and active-wall
  overhead to cumulative clocks. Keep one worker, 4 GiB, native one-second
  action watchdog, worker CPU limits and remaining-wall timeouts.

Nominal branch uses a detached byte-identical copy of the native submitted
request. Its live acceptance still requires full archived-array/candidate parity.
Alternative substitution changes only step-zero applied action, not nominal
post-query optimizer memory. At steps 1–299 the unchanged controller receives
its newly observed physical states and original indexed preview. No optimizer
state is restored from archived solutions during live execution.

## Independent zero-solve verification

**173/173 applicable quadrotor tests passed** (154 inherited plus 19 new tests).
They exercise the exact CLI, immutable preparation/approval checks, shared
12-job dispatcher, all 12 first-failure positions, no retry/refund, nominal gate,
worker routing for all job types, budget exhaustion, both replay paths,
pre-physics rejection/journaling, signed-zero handling, startup failures and
incomplete-run rejection.

The actual new episode wrapper ran with inert numerical providers and scripted
archived native Python MPC flow for 300 queries in each of nominal and alternative
paths. It checked exactly one query per step, detached action memory, 300 native
initial guesses per branch, planned x/u arrays, unchanged post-query memory and
preview/terminal-padding indices. These fixtures deliberately feed archived
nominal successor states to both paths and end with a labeled mock stop. They
are not simulations, new safe continuations, or evidence of live counterfactual
trajectory equivalence. Native construction/control flow and motor observation
are independently exercised by the existing archived/mocked tests.

Additional independent reconstruction passed: 1,200 archived warm starts,
1,200 archived numerical contracts, 7,200 exact-rational majorant components,
20 independent KKT/program/reference candidates, and 720 mocked motor boundaries.
Maximum source recurrence closure was `3.1086244689504383e-15`; maximum independent
program reconstruction discrepancy was `3.552713678800501e-15`.
All real optimizer/engine API counters remained empty. Peak regression RSS was
437,682,176 bytes, below 4 GiB.

One initial preservation-audit helper stopped because it requested the wrong
JSON field name (`additional_report_sha256` instead of the restoration seal's
`report_sha256`). Its failed log/script and charge are retained. The corrected
read-only audit passed; no evidence, acceptance threshold or scientific result
was changed to resolve this harness error.

## Preservation and numerical limitations

The final independent new execution checkpoint passed. All 289 upstream files
match clean commit `6b5391d014f36fdfa0f9d22d92c77387e5274308`.
Interpreter `/private/tmp/e12_venv/bin/python` is the accepted CPython 3.10.19;
its executable SHA-256 is
`1d583a3812cabef5b1db8bfd437e08e00f4ae1b5341ede784945934a9da30e7d`.
All 38 package versions, four native binaries and four package RECORD identities
match. The accepted preflight's nine sources/21 artifacts, all 227 tracked files,
87,315 accepted MPC80 artifacts, 59,268 earlier result/data entries, 87 drilling
source files, 50 restoration artifacts and nine untracked IQL files match their
recorded hashes. The stopped Pilot B v1, its unexecuted draft and historical
reports are unchanged. No commit/push was made.

Direct telemetry records native motor and force/torque API arguments, not physical
sensor output. Thrust inferred from RPM remains labeled as derived. Full opaque
Bullet state is not archived or certified. Future uninstrumented native replay
checks 301 control states/rewards and end-interval RPM; instrumented motor replay
also checks all 6,001 raw states/contacts and direct submissions. Do not claim
full-rate direct observation from the uninstrumented replay alone.

Original overall qualification and original 0.001 symbolic/PyBullet criterion
remain **FAILED**. Scoped accepted MPC80 feasibility remains empirical, finite-
cohort evidence, not formal robust/physical safety or hard real-time readiness.
Passing this wiring review does not establish action diversity, the original
three-of-four action-support criterion or beyond-expert performance improvement.

## Resource reconciliation and execution decision

| Quantity | CPU seconds | Active-wall seconds |
|---|---:|---:|
| Accepted carried cumulative charge | 1830.740281 | 2517.051232 |
| This review, measured work plus conservative preparation/finalization allowance | 58.464719 | 121.056768 |
| Updated cumulative charge, rounded upward | 1889.205 | 2638.108 |
| Existing approved ceiling used for this review | 2044.402 | 2874.158 |
| Proposed fresh Pilot B increment (not authorized) | 300.000 | 600.000 |
| **Proposed final cumulative ceiling** | **2189.205** | **3238.108** |

No historic charges or reservations were reset. Real original allowance remains
**3,070 solves /366,000 physics steps**. The unchanged four scientific runs and
eight replays reserve exactly **1,200 solves /72,000 steps**, leaving
**1,870 /294,000** if the complete pilot is later authorized. All failed work
consumes its reservation and actual overhead. Future preparation/dispatch and
verification use the proposed increment; this review grants no larger budget.

No numerical/provenance wiring blocker remains under the tested scope.
Recommendation: **GO for separate execution approval**, contingent on accepting
the exact proposed cumulative ceilings and seal-bound fresh approval. Until then
scientific execution remains **PAUSED**. Stop after this review; no live validation
or counterfactual performance/safety conclusion is available.

## Versioned implementation identities

| File | SHA-256 |
|---|---|
| `src/quadrotor_counterfactual_execution_v2.py` | `4b843b0558d32a53a6f51181fbc1837f7d58428d171f5a0a1da93e650c377329` |
| `src/quadrotor_counterfactual_dispatch_v2.py` | `5e38485d5d928a1f069b359e96b63e6c148908c7c7547972864dd30e459a5be4` |
| `experiments/12_counterfactual_pilot_b_v2.py` | `5dfe63440d3fe304384f38138e8543d982708908805c2eda7d319a2a36d8bc11` |
| `tests/test_quadrotor_counterfactual_execution_v2.py` | `cfb465dece1c5f44d06cc4cc775926f19a53881b10a0a49ea2e1508b1966dadf` |
| `configs/experiment_12_counterfactual_pilot_b_v2.json` | `7ed766f92a523a34cacec3a404be7cc239dccb4c0b63cfd10a7e2ba85746e5af` |
| `docs/experiment_12_counterfactual_pilot_b_v2_protocol.md` | `875157857910b62d0f29976bbcb5019ff382f3819900a0f54fca8352b5779a72` |

All detailed zero-solve records, retained failed harness log, resource ledger and
review seal are in `results/benchmark_selection/counterfactual_wiring_review_v2/`,
excluded from Git. The report and protocol are new v2 documentation; historical
reports were not edited.
