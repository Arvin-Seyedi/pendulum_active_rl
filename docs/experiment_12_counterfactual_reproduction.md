# Experiment 12 — Counterfactual Pilot B reproduction

This milestone preserves the preflight, stopped runtime attempt, exact restoration,
sealed wiring review and accepted Pilot B v2. Original overall qualification and
0.001 symbolic/PyBullet gates remain FAILED. Historical protocols remain frozen
with their original authorization flags; later approvals are external records.

Use the exact restored Python 3.10.19 environment, pinned
`requirements_experiment_12.txt` and clean upstream revision
`6b5391d014f36fdfa0f9d22d92c77387e5274308`. See the
[restoration identities](experiment_12_runtime_restoration_report.md). Generated
scientific prerequisites are intentionally absent from Git. A source checkout
alone cannot recreate unavailable accepted data or native runtime identities.

## Read-only verification

From this checkout or a clean source export:

```sh
/private/tmp/e12_venv/bin/python experiments/12_verify_counterfactual_pilot.py \
  --data-root "$DATA_ROOT" --upstream /private/tmp/e12_safe_control_gym \
  --output /private/tmp/fresh_pilot_b_verification.json --full-preservation
```

`DATA_ROOT` is the original scientific repository containing the accepted archives,
original Git history, authority files and nine preservation-only IQL files.
Output must be fresh and outside historical evidence directories. The verifier
does not call a controller or engine. Real optimization/physics APIs are blocked.
It validates 12,147 Pilot B artifacts, the external authorization records and
review seal; reconstructs rewards, direct/derived inputs, all raw margins,
reserves, all 1,200 independent KKT candidates, reference/warm-start histories,
bitwise nominal fidelity, deterministic repeats and both recorded replay types.
It does not execute new simulator replays. Its canonical scientific JSON must
equal the accepted independent audit exactly.

`--full-preservation` additionally hashes the earlier 87,315 MPC80 files and
59,268 prior result/data entries, authority and IQL files. Only CURRENT_STATUS,
RESEARCH_CONTEXT and the decision log may differ from old context hashes, proved
using the original Git bytes. No historical manifest is rewritten. The retained
unqualified v1 draft is checked in the external data root; it is not a runnable
dependency or part of the new committed execution implementation.

## Regressions and clean source export

Run `experiments/12_regression.py --group quadrotor` with the pinned E12 Python
and `--group prior` with the existing `.venv/experiment08/bin/python`. Use fresh
JSON outputs with distinct supervisor log names. Exclude `test_offline_rl.py`.
Quadrotor fixtures use archived solutions/scripted native flow and a mock backend;
use `zero_real_operations` to guard native engine and optimizer entrypoints.
No scientific episode is executed by these regressions.

Export only intended Git source/configuration/tests/documentation to a fresh
directory; exclude IQL, generated outputs, datasets, checkpoints and environments.
Provide read-only `results`/`data` fixture links to the original archive; supply
the original data root explicitly to the verifier. Run both regression groups and
the read-only verifier there. Compare complete canonical scientific results and
digest against workspace verification. Full historical data hashing need only
be performed once in the same finalization; both exports check the complete
Pilot B/review seals and reconstruct the scientific results independently.

Verification evidence and additive resource accounting reside in
`results/benchmark_selection/counterfactual_pilot_B_finalization/`. The accepted
Pilot B ledger stays unchanged. Its completion allowance was 1,870 solves,
294,000 physics steps, 91.594 CPU and 344.109 active-wall seconds; finalization
uses zero solves/steps and reduces only the remaining time allowance. It is not
a renewed experiment allocation. The accepted scientific report retains exact
metrics and historical findings; this document adds reproduction instructions.

## Execution boundary

`experiments/12_counterfactual_pilot_b_v2.py` is the reviewed execution entrypoint,
but it rejects stopped/completed/partial output trees and needs fresh explicit
seal/report-bound approval and matching cumulative ceilings. It must not be used
to rerun the accepted pilot during finalization. No additional action probes,
acquisition, learner training or IQL work is authorized by this milestone.

## Milestone verification

Workspace and clean export pass **298 tests** (176 quadrotor, 122 prior), with
zero failures/errors/skips; the new read-only guards are also checked after the
clean-export path-resolution adjustment. Both independent audits reconstruct all
1,200 candidates, four scientific records and eight replays and produce the same
complete scientific JSON, canonical SHA-256
`979088d5d69d4ca7c630ba12e0bbca14f4b2ce36745f34d47cfe893aa83cfeff`.
Full historical preservation is checked once; both audits validate all 12,147
Pilot B files and the unchanged review seal and numerical evidence.

An initial regression supervisor chose the same log basename as the regression
script; the latter refused the existing output before any tests ran. That failure
and charge are retained. Fresh distinct paths passed, without modifying historical
test code or scientific evidence. The proposed next-direction design document is
a separate uncommitted review draft, excluded from this reproducibility milestone.
