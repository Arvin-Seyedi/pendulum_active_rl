# B1 milestone: read-only finalization

The accepted [report](experiment_12_b1_report.md), source/configuration seal,
checkpoints, predictions and original NO-GO remain immutable. Finalization adds
context updates and verification only, with no fitting, MPC solve or integration.
Generated evidence is under `results/benchmark_selection/b1_finalization/`;
accepted B1 evidence stays under `b1_archive_v1/`, outside Git.

Use the exact pinned CPython3.10.19 environment and upstream revision
`6b5391d014f36fdfa0f9d22d92c77387e5274308`, with NumPy2.2.6/Torch2.8.0 CPU.
Package/native/upstream identities are verified against accepted archives.

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/private/tmp/e12_matplotlib E12_FORCE_ZERO_SOLVE=1
/private/tmp/e12_venv/bin/python experiments/12_finalize_b1.py \
  --data-root "$PWD" --full-preservation \
  --output /private/tmp/b1_finalization_read_only.json
```

Choose a new output and charge it against the additive remaining resource ledger.
No runtime substitution, fitting or new environment creation is permitted. The
new wrapper checks all 196 accepted B1 artifact hashes and accepted source hashes
before invoking the unchanged independent B1 verifier. It reconstructs 40
checkpoint readouts, four ridge equations, three paired intervals, schedules,
initialization pairing and bitwise repeat parameters/Adam moments/histories.
Scientific fitter calls are forbidden; real simulator/optimizer calls remain
blocked by the original zero-operation guard.

The original strict verifier also checks frozen context-document bytes. To retain
that behavior while permitting authorized status/log updates, finalization proves
their original bytes using the recorded Git revision, then supplies them only in
a disposable documentation view. Every scientific source in that view points to
the current verified code. Live context hashes are reported separately; no source
hash is relabeled, evidence repaired or historical document overwritten. Other
historical file changes fail. Full preservation independently verifies the live
scientific repository and previous research/drilling/IQL artifacts.

A clean export contains tracked source plus the explicitly intended B1 files;
generated results are accessed through an explicit original data root. All nine
unfinished IQL files and unrelated drafts are excluded. Regression fixtures use
archived records and mocks. Neither export verification nor tests refit scientific
models. Historical design wording is retained in the frozen B1 design document;
later acceptance is recorded by the status and decision log, not retroactive edits.

Finalization records reconcile charges from accepted cumulative 2608.115 CPU /
3649.412 active-wall seconds against unchanged shared ceilings3964.941/6613.063.
The 1,870-call/294,000-step allowance is unchanged. Separate research-direction
analysis is not part of either milestone commit and authorizes no new experiment.

Finalization verification passed in the workspace and clean export: **216 tests**,
zero failures/errors/skips, plus independent numerical reconstruction and full
historical preservation. The accepted original B1 run remains at 211 tests. Five
new finalization tests cover immutable outputs and documentation/evidence guards.
An initial clean-export test caught an unresolved-symlink output guard in the new
wrapper; the guard was corrected, the failing record retained, and both suites
passed again. This changes no scientific equation, fit, metric or acceptance rule.
