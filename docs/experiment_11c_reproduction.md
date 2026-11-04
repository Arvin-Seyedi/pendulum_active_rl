# Experiment 11C — Immutable static replay and finalization

The [report](experiment_11c_report.md) records a reviewed four-snapshot mathematical
pilot. Geometric coordinate enforcement, full-acceleration compatibility, ideal
reaction mathematics and physical validation are distinct. The original model
remains an ideal-input numerical benchmark. Further physical development and new
numerical drilling investigations are paused pending external clarification;
reproducing these frozen calculations does not authorize another investigation.

## Dependencies and exact external inputs

Install `requirements_experiment_11c.txt` (NumPy 2.3.2, SciPy 1.18.1, gmpy2 2.3.2
through the 11B/11A requirement files). The full historical test suite additionally
uses `requirements_experiment_10.txt`. No unfinished IQL code is imported.

Keep the unchanged authority ZIP at the path in `configs/experiment_11a.json`,
SHA-256 `b86ce2a62fe26871f40eb674377b1b17ddedac84ef422019c47e59f1787be1b5`,
the source/Chapter 5 files referenced by preservation manifests, and the accepted
11A and 11B archives and finalization records. The 11C input archive is
`results/drilling_constraints/pilot/`. These are explicitly required scientific
inputs outside Git; no missing input is silently regenerated. Native MATLAB,
controller synthesis and a simulator are not used for this replay.

The frozen cohort is exactly P5 indices 0/200000 and P0/P3 index 1000. The initial
P5 loads are explicit nominal values, not native zero placeholders. Its terminal
command is computed but unused by integration. All five operator groups and 31
RHS columns, original inputs/reference, exact arithmetic and 128/256-bit settings
remain fixed. No new force limit or corrected equation is introduced.

## Read-only command

Choose an unused output file outside all accepted archives:

```sh
PYTHONDONTWRITEBYTECODE=1 .venv/experiment08/bin/python \
  experiments/11c_verify_constraints.py \
  --output /private/tmp/e11c-unused-verification.json
```

The entry point enforces one numerical thread, 30 CPU-minutes and 2 GiB. It:

- Verifies the original completion, integrity and clarified-review seals.
- Re-extracts the same four original/repeat native snapshots and compares all
  frozen values bitwise, including command/load bookkeeping.
- Reproduces exact and 128/256-bit results and the accepted repeat hashes.
- Uses the unchanged independent fraction/Bareiss/LDL checker for 31 exact RHS,
  3,100 interval scalar residuals and 950 cross-precision overlaps.
- Reproduces all five operator groups, hypothesis classifications and rank checks.
- Independently verifies `C*T*qddot_N=0` and the missing-force virtual-work identity
  `T'*(M*T*qddot_N-w)=M_H*(qddot_N-qddot_H)` without applying that force.

The legacy verifier and report generator write summaries. They are therefore run
only in a disposable evidence copy, never in the accepted archive. A private
hash-only root is not added to the import path. Its historical context files use
exact Git bytes from `e1a2e00415f45d31810415bed66bfddc9480c454`; only
`CURRENT_STATUS.md`, `RESEARCH_CONTEXT.md` and `docs/RESEARCH_DECISION_LOG.md` can
use this documentation exception. All other live preservation hashes must match.
Source files used by the original mathematical pilot must remain byte-identical.
Local unfinished IQL bytes are preservation targets, not numerical dependencies.

Accepted source versions are additionally preserved in
`results/drilling_constraints/finalization/accepted_source_snapshot/`. The original
and clarified pilot reports, all rejected/unresolved findings and all numerical
outputs remain unchanged. The versioned report adds finalization and explicit
external-information requirements. Existing output files are rejected.

## Clean source export

Export committed files and intended 11C source/configuration/tests/documentation
into an unused temporary directory. Exclude generated outputs and all nine
unfinished IQL files. From that export, run:

```sh
PYTHONDONTWRITEBYTECODE=1 /path/to/python experiments/11c_verify_constraints.py \
  --project-root /path/to/reviewed/project \
  --output /private/tmp/e11c-unused-export-verification.json
```

Imports come from the export. The explicit project root supplies the authority,
accepted scientific data and historical Git/preservation information. The replay
does not import source from its temporary hash-only view.

For the complete historical regression suite, restore just three existing ignored
fixtures with original bytes: Experiment 01 `expert_safety_map.npz` and Experiment
10 pilot `selection.json`/`protocol.json`, at their original relative paths. Then:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  MPLCONFIGDIR=/private/tmp/e11c-export-mpl \
  /path/to/python -m unittest discover -s tests -v
```

The unfinished IQL test is absent from the export. Ordinary regression fixtures
remain isolated from scientific results; finalization executes no new research
trajectories, drilling controller, RL agent or new mechanical model. Logs,
preservation checks and export manifests are stored separately under
`results/drilling_constraints/finalization/` and excluded from Git.

## External clarification required before further physical work

The original author/engineering sources must specify the reaction/friction law;
rotational coordinates and torque work conventions; material/spatial mesh and
wellbore geometry; supports, preload/stress-free state and reference/gauge; and
physically admissible surface force/torque limits and any documented rate limits.
Internal agreement of ideal-reaction formulations cannot establish those facts.

## Finalization results

All **122 applicable tests passed in workspace and clean export**. Both static
replays reproduced exact arithmetic and both interval endpoint sets, the original
repeat hashes, four archived snapshot pairs, all five operator groups and all
scientific classifications. Independent verification passed 31 exact RHS columns,
3,100 interval scalar residual checks, 950 cross-precision overlaps and the
geometric-enforcement/missing-force identity. All accepted seals and historical
preservation checks passed with zero accepted-archive writes, new MATLAB processes
or new research trajectories. These are reproduction checks, not evidence that
the physical drilling plant or either physical reduction has been validated.
