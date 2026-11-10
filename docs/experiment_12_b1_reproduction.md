# B1 archive-only reproduction and evidence

Use the pinned `/private/tmp/e12_venv/bin/python` (CPython3.10.19, Torch2.8.0 CPU,
NumPy2.2.6) and clean upstream revision
`6b5391d014f36fdfa0f9d22d92c77387e5274308`. All38 package identities, installed
RECORD/native-library hashes and289 upstream file hashes are validated against
accepted local evidence. Do not substitute another runtime or regenerate archives.

Generated evidence lives outside Git under
`results/benchmark_selection/b1_archive_v1/`. The [protocol](experiment_12_b1_protocol.md)
and [configuration](../configs/experiment_12_b1_archive_v1.json) specify the
prospective models, checkpoints, partitions, statistical contrasts and stop rules.
The reviewed design draft is preserved unchanged; its earlier approval wording is
superseded by the user's staged B1 authorization, not retroactively edited.

Read-only verification (no fitting, controller construction, solves or integration):

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR=/private/tmp/e12_matplotlib E12_FORCE_ZERO_SOLVE=1
/private/tmp/e12_venv/bin/python experiments/12_verify_b1.py \
  --data-root "$PWD" \
  --study "$PWD/results/benchmark_selection/b1_archive_v1" \
  --output /private/tmp/b1_read_only_verification.json --full-preservation
```

Choose a fresh output; existing evidence is never overwritten. Charge verification
against the remaining task allowance using the study's `account.py` wrapper, which
retains inherited cumulative charges and enforces the local remaining CPU/wall cap.
The verifier reconstructs all40 scientific checkpoint readouts in independent
binary64 NumPy arithmetic, all four regularized ridge systems, all paired CIs,
fixed minibatch schedules and initialization pairing, and bitwise repeat weights,
Adam moments and histories. Full preservation validates previous evidence and
the authoritative drilling sources. The unfinished IQL files are checked only
by hash and are never imported.

Scientific fitting is **not** part of read-only reproduction. Its historical
stage order was: immutable initial preservation/resource snapshot; exact native
RPM/B0 feature test; full B0/provenance/mocked-boundary preflight;211-test regression;
Phase-A approval marker; sealed20 fits plus fixed repeat/four ridge systems;
external prediction unlock; fixed assessment; independent audits. The original
preflight output-path rejection is retained as `preflight.log`; the B0 verifier
was then called with a disposable output and its result copied to B1, preserving
its original guard. No scientific provenance check failed or was relaxed.

The historical account and preparation records are ignored artifacts, not new
training authorization. Re-running scientific fitting would need a new approved
resource/provenance snapshot and fresh output directory. The entry point rejects
existing fits and sealed output. Do not rerun fitting merely to inspect results.

Clean-export verification copies tracked source plus the explicitly listed B1
implementation/configuration/test/docs to a disposable directory, excludes all
nine IQL files and unrelated drafts, and reads original archives through an
explicit data-root. Regression fixtures can access a link to the accepted results;
no dataset/model is committed or regenerated. Both code roots must match the
source seal; the original accepted artifacts remain in their original location.

H0–H3 history values are source reconstructions of commands, not observed historical
motor submissions or physical actuator sensing. Requested-action labels match
saved MPC solver first actions; integer control indices and known references
provide causal alignment. No learned controller was run and no safety, acquisition
effectiveness or beyond-expert claim is supported by these supervised records.
