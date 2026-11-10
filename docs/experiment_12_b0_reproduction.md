# B0 accepted archive-only assessment: read-only reproduction

B0's 4→10-trajectory comparison remains NO-GO for acquisition readiness. Preserve
all original thresholds, seeds, partitions, fixed 5,000-update fits, negative
findings and the positive descriptive 2→10 learning-curve result. No learned
policy was executed. No finalization or design review authorizes new fitting,
MPC calls, physics steps or collection.

Use the pinned Python 3.10.19 /Torch 2.8.0 environment and existing
`requirements_experiment_12.txt`, with upstream safe-control-gym revision
`6b5391d014f36fdfa0f9d22d92c77387e5274308`. The exact environment, native library
hashes and accepted data are external prerequisites, not silently substituted.

```sh
/private/tmp/e12_venv/bin/python experiments/12_verify_b0.py \
  --data-root "$DATA_ROOT" --upstream /private/tmp/e12_safe_control_gym \
  --output /private/tmp/fresh_b0_verification.json --full-preservation
```

DATA_ROOT supplies accepted `results/benchmark_selection/b0_archive_v1/`, the
original MPC80 data, original Git history, authority and preservation-only IQL
files. The source checkout supplies committed code/configuration/docs. Output
must be fresh, outside historical evidence directories. All scientific fitting
functions are blocked; the numerical audit blocks real MPC/engine APIs. No
optimizer is created and no gradients or physical trajectories are generated.

The verifier checks the B0 source seal and all 204 evidence hashes; independently
reconstructs 4,800 causal feature rows, twenty checkpoint readouts, per-trajectory
and phase errors, all curve intervals and paired primary CI; checks four saved
ridge coefficient systems, clipping/error metrics, five-seed initialization and
minibatch identities, the saved deterministic training repeat/optimizer state,
and worst actions. It never retrains ridge or the MLP. The complete accepted
assessment JSON is canonically hashed for workspace/clean-export comparison.

Full preservation additionally verifies 250 earlier tracked files, 87,315 MPC80
artifacts, 59,268 prior result/data entries, 12,147 Pilot B artifacts, 87 authority
files and nine unchanged IQL files. Only the three authorized context/log files
may differ from older hashes, proved against exact original Git bytes. No
historical manifest, report or ledger is rewritten.

For a clean export, copy only intended Git source/config/tests/docs, excluding
generated outputs, checkpoints, datasets, environments, IQL and unreviewed design
drafts. Provide read-only `results`/`data` fixture links, supply DATA_ROOT explicitly
and rerun the verifier and applicable quadrotor regressions. Compare complete
canonical scientific JSON/digest. Full historical hashing need only run once in
the same finalization; both exports check the entire B0 seal and scientific audit.
Fixtures use synthetic updates/scripted state lookups, not new scientific fitting
or physics. Do not run `12_b0_archive.py --phase train` for finalization or retry
an accepted output tree.

Additive finalization records/time charges are under
`results/benchmark_selection/b0_finalization/`. Accepted B0 consumption remains
134.306 CPU /210.959 wall seconds, with its completion balance of 1,665.694 CPU /
3,389.041 wall seconds under cumulative ceilings 3,964.941 /6,613.063. Finalization
reduces the time balance without altering that historical snapshot. Original
1,870 solver calls /294,000 physics steps remain unused by B0/finalization.
No new resource allocation is granted; a B1 proposal is design only.

## Finalization verification

Workspace and clean source export each pass **199 applicable quadrotor tests**,
zero failures/errors/skips. All real engine/MPC APIs are blocked. Synthetic
gradient fixtures are unit tests, not additional B0 scientific fits. Unchanged
prior experiments are covered by their preserved source/artifact hashes.
Both independent audits reconstruct the same complete accepted scientific JSON,
SHA-256 `49452bca6e5e849aaee45bd35e0939c89f77338f2440bdc4a49838fdeb93d6a2`.
They check all twenty models, four ridge systems, all 4,800 causal records, all
learning-curve intervals, paired bootstrap, seeds/schedules, worst actions and
the saved deterministic optimizer/parameter replay. Full historical preservation
passes separately; no accepted evidence or failed criterion is changed.
The first staging check reported a final blank line in the already sealed
`quadrotor_archive_experiment.py`. Its exact accepted bytes are preserved rather
than changing the source hash for formatting. The remaining whitespace check
passes with only `blank-at-eof` disabled for that command; no persistent Git
configuration or scientific code was changed. The original check log is retained.
