# Experiment 12 — Read-only milestone reproduction

Use Python 3.10.19 and `requirements_experiment_12.txt`, plus the unmodified
upstream checkout at `6b5391d014f36fdfa0f9d22d92c77387e5274308`.
The native build/platform details are in [original reproduction](experiment_12_reproduction.md).
No unpublished IQL source is required. Generated evidence is external scientific
input, not part of the source commit; a source-only checkout does not contain
trajectories, the pinned wheel or authoritative drilling package.

With the original research tree accessible as `DATA_ROOT`, run from the source
checkout or a clean source export:

```sh
/private/tmp/e12_venv/bin/python experiments/12_verify_milestone.py \
  --data-root "$DATA_ROOT" \
  --upstream /private/tmp/e12_safe_control_gym \
  --output /private/tmp/e12_saved_record_verification.json
```

Choose a fresh output path. The verifier refuses existing outputs and paths
inside accepted archives. It reads all six accepted stage seals, archived physics
states, actions, candidate vectors, replay records and frozen source manifests.
It independently recomputes costs/margins/thrust conversion, dense one-step
physics equations, finite-side KKT checks, rational discrepancy majorants and the
new step-54 v1 rejection. It checks historical prefixes without inventing the
missing old candidate. `Opti.solve` and `solve_limited` are disabled and PyBullet
is not imported. This verifies archived independent action-replay **evidence**;
it does not run another simulator replay.

Preservation checks use the original data root for generated data and the nine
IQL preservation targets, and the exported source for committed implementations.
They hash 53,819 historical project files, 87 authority files, 289 upstream files,
all accepted stage artifacts and sources, and the nine unchanged IQL files.
Only the three authorized research-context/log edits can differ from historical
hashes; each exception must be proved against its exact historical Git bytes.
IQL files are not imported or present in the clean source export.

Regression is split between the pinned E12 environment and the existing prior
experiment environment because their Python/dependency requirements differ.
The E12 suite is `tests/test_quadrotor*.py`; prior regression includes all other
`test_*.py` except unfinished `test_offline_rl.py`. Solver entry points are blocked
during verification. Unit fixtures do not constitute new scientific trajectories.

```sh
/private/tmp/e12_venv/bin/python experiments/12_regression.py \
  --group quadrotor --output /private/tmp/e12_quadrotor_tests.json
"$DATA_ROOT/.venv/experiment08/bin/python" experiments/12_regression.py \
  --group prior --output /private/tmp/e12_prior_tests.json
```

The legacy environment is the previously established Python 3.13.2 environment
with prior experiment dependencies (including gmpy2); it is separate from the
pinned E12 Python 3.10.19 environment. Both commands refuse an existing report/log.

For clean-export verification, export the prior committed tree and overlay only
the intended milestone source/configuration/tests/documentation. Exclude all nine
IQL files and generated outputs. Read-only data links are supplied only for tests
that resolve archived prerequisites relative to the source root. Repeat both
regression groups and the saved-record entry point there; compare the complete
canonical `scientific_results` JSON with the working-tree result, excluding paths,
timing and the separately documented authorized context edits. Do not use the
historical execution entry points to reproduce this finalization.

Finalization evidence, test logs, export manifests and exact scientific-result
comparison are retained separately at
`results/benchmark_selection/finalization/`. The reviewed pilot/diagnostic/
qualification archives and original reports are immutable. See the
[consolidated scientific report](experiment_12_report.md) for the still-failed
overall gate and the remaining resource allowance.

## Finalization results

**191/191 tests pass in both workspace and clean export**: 122 prior-experiment
tests and 69 E12 tests, zero failures/errors/skips. PyBullet is not imported by
either regression group or the record verifier; optimization entry points are
blocked. The export excludes unfinished IQL files and contains no generated data.

The two record audits produce identical canonical scientific JSON (SHA-256
`85289849b54892a1ecd699d24c37e7d800cc055121cbf15d40d90781752a565c`). They verify
106 attempted execution records (84 complete), 168 complete-episode independent
replay records, **650,640** saved physics transitions (maximum source-equation
reconstruction difference **8.881784197001252e−16**), **10,932** retained KKT
candidates and **51,192** exact-rational majorant-component checks. The original
failed model criterion and overall qualification remain failed.

Preservation verifies **5,438 sealed stage artifacts**, **43 accepted source
files**, all **53,819 historical project files**, **87 authority files**,
**289 upstream files** and all nine IQL hashes. Only authorized context updates
use proven historical-byte exceptions. Record-verification CPU was approximately
20.50 s in the workspace and 21.08 s in the export; timing/path metadata are
excluded from the scientific equality comparison. Regressions and preliminary
checks have separate logs. There were **zero new solver calls and zero new
physics steps**, so the remaining solve/step allowance is unchanged.
