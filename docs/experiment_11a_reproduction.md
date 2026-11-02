# Experiment 11A reproduction and immutable archive verification

The [scientific report](experiment_11a_report.md) records a successful native
numerical-reproducibility milestone with an unresolved physical-reference
consistency limitation. Six original/repeat pairs (12 executions) reproduced
85 numerical workspace fields bitwise. Five cases were strictly sampled-safe;
P3 was not. Lookup failures began at 0.1318 s, before the first safety crossing
at 0.1323 s; their temporal ordering does not establish causation.

The physical-reference balance contains `r0-t*K_M*Vs`, where
`r0=F_M,d-D_M*Vs-K_M*IC`. Neither reference identity is exactly zero. The largest
angular component of the omitted term at 20 seconds is 0.15240057922651 rad/s²,
not a measured trajectory deviation. P5 is an equilibrium of the implemented
reduced error map; it does not prove an exact physical reference trajectory.
Physically admissible bounded-input safety remains blocked by missing engineering
specifications. No residual compensation, limits or controller redesign is adopted.

## Required inputs and environments

Native execution used MATLAB **24.2.0.2863752 (R2024b Update 5), MACA64**, with
`-singleCompThread`. Base MATLAB is required. The original stored LPV gain table
is loaded; LMI synthesis functions are not required or invoked. License checks
and installed toolbox metadata are in the archived logs and per-run environment
files. Python audits used CPython 3.13.2, NumPy 2.3.2 and SciPy 1.18.1; install
`requirements_experiment_11a.txt`. The complete earlier Python suite additionally
uses `requirements_experiment_10.txt`. No unfinished IQL files are dependencies.
Cross-platform bitwise equivalence is not assumed.

Restore the exact external archive at the location in
`configs/experiment_11a.json`:

```text
8_Downhole_Directional_Drilling_Barrier_Avoidance.zip
SHA-256 b86ce2a62fe26871f40eb674377b1b17ddedac84ef422019c47e59f1787be1b5
```

The archive and unpacked authority files are excluded from this Git milestone.
For read-only verification, retain the accepted output archive at
`results/drilling_reproduction/native_pilot/` and the authority files referenced
by its preservation manifest. This is an explicitly required scientific input,
not a silently regenerated dataset. Source/member hashes, exact six-case bindings,
executed initial binary64 values, MATLAB histories and environment records remain
in that archive.

P0 is the top-level script; P1–P4 are exact Chart_Plot IC1–IC4 through native
IC_Test; P5 is the separately approved new zero-error input. The standalone Chart
script is not IC3. All six source bindings are in `cases()` and the frozen
`case_manifest.json`. Relative loads are resolved only within the matching
isolated source/data copies.

## Finalization and read-only checks

Do not use the original report-generating audit functions for archival checks:
they write derived outputs. The additive verifier reuses their numerical functions
without calling the simulator, MATLAB or controller, and writes no accepted data:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  .venv/experiment08/bin/python experiments/11a_verify_native.py
```

An optional `--output PATH` saves a **new** verification JSON outside the accepted
archive. Existing output files and paths inside the accepted archive are rejected.
The verifier checks:

- Original completion/final-verification artifact hashes and source snapshots.
- Exact authority ZIP/member hashes, gain-table values and request identities.
- All six native numeric original/repeat comparisons and initial-state identities.
- Strict safety on all 200,001 states per execution, lookup/fallback semantics,
  reconstructed bit loads, transformations and recorded commands.
- All 2,400,000 one-step transitions using recorded states and previous-interval
  loads; these calculations are not sequential rollouts.
- Native MATLAB safety/residual records against independently reconstructed data.
- Historical preservation, allowing exactly the three authorized context updates.

`CURRENT_STATUS.md`, `RESEARCH_CONTEXT.md` and `docs/RESEARCH_DECISION_LOG.md`
intentionally differ from their execution-time contents at commit `a164fa6`.
Their historical hashes are checked against that Git commit. All other preserved
files must retain their original bytes. Do not rewrite historical manifests to
accept documentation updates.

The versioned scientific report adds the reviewed equations, units, noncausal
interpretation and distinctions. Its exact execution-time predecessor remains
at `native_pilot/scientific_report.md`, matching the original completion markers'
report hashes. The verifier preserves that historical report and separately
records the current versioned report hash. Original native/analysis implementation
files remain byte-identical to their execution-time source snapshots; finalization
adds only read-only verification infrastructure and documentation.

## Clean-export verification without new control experiments

Create a disposable export containing committed files and this milestone's
intended source/configuration/tests/documentation, excluding all nine unfinished
IQL files and generated outputs. Run the export's read-only verifier against the
accepted archive through the explicit data-root argument:

```sh
PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 \
  /path/to/python experiments/11a_verify_native.py \
  --project-root /path/to/reviewed/project
```

The code comes from the export; only explicitly identified scientific archives
and historical preservation/Git records are read from the reviewed project.
No import depends on IQL, and no MATLAB/native trajectory is executed.

For all historical Python tests in that export, restore only these ignored test
inputs with their original bytes: the exact Experiment 01
`results/expert_safety_map/expert_safety_map.npz` and Experiment 10 pilot
`selection.json` and `protocol.json`. Then run:

```sh
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/private/tmp/pendulum_11a_mpl \
  /path/to/python -m unittest discover -s tests -v
```

This finalization performs source/test and saved-record reproducibility checks,
not a fresh native rerun. The accepted 12 MATLAB executions provide the native
repeat evidence. Generated outputs remain outside Git.

## Original native execution, for an independently approved fresh reproduction

Never run into a reviewed output directory. In a disposable research checkout
with the exact authority ZIP available, first check base MATLAB availability:

```sh
/Applications/MATLAB_R2024b.app/bin/matlab -singleCompThread -batch \
  "disp(version); disp(computer); assert(license('test','MATLAB')); disp(ver);"
```

Then `experiments/11a_native_drilling.py --run` executes only P0–P5 original/repeat
pairs in isolated temporary directories with read-only authoritative copies.
The native observation hook is a false conditional breakpoint at IC_Test line
547, before display clipping, plus post-script capture for P0. It does not assign
to the controller workspace. Initial nominal inputs/loads, stale-memory fallback,
200,000 applied commands and the unused terminal computed command are preserved.
Do not interpret initial zero entries in native load arrays as applied zero loads.

The original `--audit`, MATLAB `e11a_reference_checks`, and
`python -m src.drilling_pilot_verification` are the execution-time derived-output
steps. The legacy report generator additionally expects the original preservation
and authority manifests, static MATLAB reference outputs and test summary; it is
not a one-command fresh-build pipeline. Those records are preserved in the
accepted archive and are explicitly required by the finalization checks. They are retained unchanged for provenance; their
execution-time paths and preservation/report checks are intentionally strict.
They are not the finalization verifier and should not be used to rewrite the
reviewed archive after documentation changes. No fresh native reproduction,
integration refinement or new control experiment is authorized by this document.

## Finalization evidence

All **92 included Python tests passed** in the reviewed workspace and clean export,
including the five original E11A fixtures and four new fail-closed archive checks.
The clean export independently rechecked all six native pairs, 510 numerical-field
comparisons, the frozen gain table, all 2,400,000 recorded transitions and all
2,400,012 raw states. Maximum absolute/state-scaled transition discrepancies
remained 3.552713678800501e−15 / 3.469446951953614e−15. All 87 authority files
and historical archive hashes passed. This used **zero new MATLAB processes, zero
new plant rollouts and zero writes to the accepted archive**. Finalization evidence
is saved separately under `results/drilling_reproduction/finalization/`.
