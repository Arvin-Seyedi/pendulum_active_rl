# Experiment 11B — Static reproduction and immutable finalization

This milestone validates mathematical/numerical properties of the archived
five-node model. It does not validate the physical drilling plant, justify the
native constrained reduction, or establish bounded-input safety. The
[scientific report](experiment_11b_report.md) preserves failed reference identities,
conditioning, printed-sign and initialization findings. No residual compensation
or alternative plant is adopted.

## Required inputs

Install `requirements_experiment_11b.txt`: NumPy 2.3.2, SciPy 1.18.1 and gmpy2
2.3.2. The complete historical regression suite additionally uses
`requirements_experiment_10.txt`. No unfinished IQL files are imported.

Retain the exact external authority ZIP configured in `configs/experiment_11a.json`
(SHA-256 `b86ce2a62fe26871f40eb674377b1b17ddedac84ef422019c47e59f1787be1b5`),
the original authority/Chapter 5 files referenced by the manifests, and accepted
scientific records at:

- `results/drilling_reproduction/native_pilot/`
- `results/drilling_static_audit/approved_static/`

These are explicitly required external scientific inputs, excluded from Git.
There is no silent regeneration of source matrices, IC, Vs or trajectories.
MATLAB, controller synthesis, RL and actuator specifications are not execution
dependencies of these static checks. Their absence does not resolve the physical
research blockers.

## Read-only verification

From the reviewed repository, choose a new output file:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/experiment08/bin/python \
  experiments/11b_verify_static.py \
  --output /private/tmp/e11b-unused-verification.json
```

The entry point enforces one numerical thread, the 30 CPU-minute/2 GiB limits,
and an output path outside both accepted archives. Existing output files are
rejected. It independently recomputes exact rational residuals, reproduces all
128/256-bit directed-rounding matrix/residual endpoints and nominal-load variants,
and re-extracts the nine frozen indices from all 12 saved 11A workspaces. It checks
the saved 108 points without advancing any state. The separate rational checker
verifies 54 unique equations and 540 componentwise rounding bounds.

Accepted artifact hashes are checked before and after verification. The original
`verify`, `analyze` and pilot-run/report writers are historical execution interfaces
that can write derived outputs; do not invoke them with the accepted archive as
their output. The new entry point supplies a separate report destination.

`CURRENT_STATUS.md`, `RESEARCH_CONTEXT.md` and `docs/RESEARCH_DECISION_LOG.md` have
authorized finalization edits. Their E11B pilot hashes are verified against exact
Git objects at `f558a13b3eab01d9bc7ad1b628cc72d751958aa9`; no other preserved
project file receives that exception. The older 11A manifest retains its own
historical context anchor. Original manifests and reports are never rewritten.

The reviewed resumed pilot recorded technical source amendments and a preserved
pre-resume supervisor-status snapshot. Verification checks those exact bytes and
the final execution status; it does not waive scientific artifact hashes. Accepted
source versions, including the original verifier/report, are additionally frozen
under `results/drilling_static_audit/finalization/accepted_source_snapshot/`.
The versioned report clarifies the Frobenius norm used by the 14.07% comparison;
the exact execution-time report remains untouched in the pilot archive.

## Clean-export checks

Create a disposable source export containing tracked files plus intended 11B
source/configuration/tests/documentation. Exclude generated results and all nine
unfinished IQL files. Run its code with an explicit scientific-data root:

```sh
PYTHONDONTWRITEBYTECODE=1 /path/to/python experiments/11b_verify_static.py \
  --project-root /path/to/reviewed/project \
  --output /private/tmp/e11b-unused-export-check.json
```

Imports come from the export. Required authority, accepted scientific records and
historical Git/preservation information come from the explicitly named project.
Local IQL bytes may be hashed for preservation there; they are not imported or
used in any numerical calculation.

For the full historical regression suite, restore only these existing ignored
fixtures, byte-for-byte: Experiment 01 `expert_safety_map.npz` and Experiment 10
pilot `selection.json`/`protocol.json`, at their original relative paths. Then:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  MPLCONFIGDIR=/private/tmp/e11b-export-mpl \
  /path/to/python -m unittest discover -s tests -v
```

The unfinished IQL test is absent from the export. All historical experiment
implementations and tests remain unchanged. No new research/control trajectories
or MATLAB processes are part of finalization; ordinary regression fixtures remain
isolated from scientific datasets and results. Additive verification logs and
export manifests are retained separately under
`results/drilling_static_audit/finalization/` and excluded from Git.

Finalization passed **108 tests in each environment**. Both workspace and export
reproduced the exact rational residuals and both directed-rounding endpoint sets,
four nominal-load variants and all 108 native archived points. The independent
checker verified 54 unique equations and 540 rational rounding bounds. Both
checks reported zero writes to accepted evidence, zero new MATLAB processes
and zero new research trajectories. The 49-file accepted 11B integrity manifest
and historical preservation checks passed unchanged.
