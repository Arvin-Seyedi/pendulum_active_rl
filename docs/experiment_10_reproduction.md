# Experiment 10 reproduction and archive verification

The [reviewed full report](experiment_10_report.md) and [accepted pilot
report](experiment_10_pilot_report.md) remain unchanged. The study compared ten
fixed controllers on all 3,331 archived PD-safe states, plus three optimization
starts on a frozen 64-state sample. LQR gained +0.056165740 mean return (4.10%
mean-cost reduction), below the +0.136939167 practical threshold, with zero
observed violations and 6.43% more squared-torque effort. No tested controller
passed the practical reward, effort or margin-improvement criteria.

All 192 original solves succeeded, but none executed safely open-loop; all 192
tracking executions were safe. Best-safe tracking gained only +0.000007367 over
matched LQR. Open-loop oracles used PD fallback, not optimized success. Solver
feasibility, empirical simulator safety, state-specific oracle choices and formal
safety guarantees must remain distinct. Neither global optimality nor a safety
guarantee is established.

## Environment and prerequisite archives

Use CPython 3.13.2 and `requirements_experiment_10.txt`. Its pinned chain includes
SciPy 1.18.1, Gymnasium 1.2.3, NumPy 2.3.2, PyTorch 2.10.0, CasADi 3.7.2 and
gmpy2 2.3.2. The recorded macOS solver build is IPOPT 3.14.11 / MUMPS 5.4.1;
arithmetic builds are GMP 6.3.0 / MPFR 4.2.2 / MPC 1.4.1. Cross-platform bitwise
reproduction is not assumed. Runners use POSIX locks and process groups.

Restore these exact, ignored scientific inputs:

- `results/expert_safety_map/expert_safety_map.npz`, SHA-256
  `2234e7c2c1d5e09595ed7f924286aa824748e747e1a4facc8be435c5cd6c622a`.
- `results/recoverable_pd/full_study/baseline_summary.json` and
  `source_hashes.json` from the accepted Experiment 09 archive.
- Experiment 09 `baseline/chunk_START_END/` directories. The pilot needs the
  five chunks containing IDs 3280, 2956, 5383, 2568 and 6194; a fresh full study
  needs all baseline chunks. The chunk start is `64 * (grid_id // 64)`, and its
  inclusive end is `min(start + 63, 6560)`, both formatted with four digits.

Selection records hash the benchmark and baseline summary. Coordinates are the
original binary64 values; never regenerate axes or substitute rounded points.
No IQL files, D0, acquired training datasets or Experiment 07/08 output archives
are required. Prior committed source helpers and frozen Experiment 07 solver
configuration are required through the normal repository dependency chain.

```sh
python3 -m venv .venv/experiment10
.venv/experiment10/bin/python -m pip install -r requirements_experiment_10.txt
```

## Fresh verification in a disposable checkout

Never rerun into reviewed result folders. Use a separate checkout or clean export
with a local Git baseline for provenance, intended committed files only, and the
inputs above restored. Exclude all nine unfinished IQL files. In that disposable
copy only, move the copies of the two reviewed E10 reports aside **before** any
execution snapshot. Do not move the original reports in the reviewed workspace.

```sh
mkdir reviewed_reports
mv docs/experiment_10_pilot_report.md docs/experiment_10_report.md reviewed_reports/
MPLCONFIGDIR=/tmp/pendulum_10_mpl .venv/experiment10/bin/python experiments/10_safe_policy_improvement.py --freeze-subset
MPLCONFIGDIR=/tmp/pendulum_10_mpl .venv/experiment10/bin/python experiments/10_safe_policy_improvement.py --pilot --resume
MPLCONFIGDIR=/tmp/pendulum_10_mpl .venv/experiment10/bin/python -m unittest discover -s tests -v
```

Freezing selects the exact 64 states before any new controller evaluation, using
PCG64 seed 101001, four ascending-PD-cost strata and grid-ID tie-breaking. The
pilot executes only its five IDs and 24 prescribed solver executions, including
the 12 original/repeat pairs. Full-suite tests read the frozen selection and pilot
protocol, so run them after the pilot. The two E10 modules contain 13 tests;
earlier tests are preserved. Timings, paths, Git metadata and protected-file counts
can differ; compare scientific arrays and solver decisions separately.

For a fresh full reproduction, restore all Experiment 09 baseline chunks, finish
the pilot, then run:

```sh
MPLCONFIGDIR=/tmp/pendulum_10_mpl .venv/experiment10/bin/python experiments/10_full_study.py --run
```

This reuses 12 original pilot solves and runs the remaining 180 once, with no new
repeat starts. Interrupted outputs require inspection and explicit `--resume`;
completed matching outputs are verified, not overwritten. No new scientific
full-study execution is needed for finalization.

## Immutable historical manifests and authorized documentation edits

Before finalization edits, both original strict commands passed:

```sh
python experiments/10_safe_policy_improvement.py --verify
python experiments/10_full_study.py --verify
```

These commands also enforce whole-workspace execution-time hashes. The authorized
updates to `CURRENT_STATUS.md`, `RESEARCH_CONTEXT.md` and
`docs/RESEARCH_DECISION_LOG.md` intentionally differ from commit `67cc51d`.
Consequently strict historical verification rejects those updated documents.
Do not rewrite original snapshots, completion markers, source hashes or reports.
The following read-only check allows exactly those three context-document changes
while verifying their historical contents against Git:

```sh
python - <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess

root = Path.cwd()
allowed = {'CURRENT_STATUS.md', 'RESEARCH_CONTEXT.md', 'docs/RESEARCH_DECISION_LOG.md'}
def digest(data):
    return hashlib.sha256(data).hexdigest()
def read(path):
    return json.loads(path.read_text())
def check(base, hashes):
    for name, expected in hashes.items():
        assert digest((base / name).read_bytes()) == expected, str(base / name)

for phase, report in [('pilot', 'experiment_10_pilot_report.md'),
                      ('full_study', 'experiment_10_report.md')]:
    folder = root / 'results/safe_policy_improvement' / phase
    done = read(folder / 'completion.json')
    assert done['passed']
    check(folder, done['artifacts_sha256'])
    check(root, read(folder / 'source_hashes.json'))
    before = read(folder / 'preservation_before.json')['sha256']
    for name in allowed:
        original = subprocess.check_output(['git', 'show', '67cc51d:' + name])
        assert digest(original) == before[name], name
    check(root, {name: value for name, value in before.items() if name not in allowed})
    assert digest((root / 'docs' / report).read_bytes()) == done['report_sha256']
    print(phase, 'scientific hashes and preservation passed; three authorized context exceptions')
PY
```

The independent saved-record audit is `independent_audit` in
`src/policy_improvement_study_analysis.py`. It takes the repository root, full
output path, `fixed_records.json` and `optimization_records.json`. It checks
33,310 fixed replay pairs, 384 optimization replay pairs, all 6,772,494 raw samples,
original initial-state bits, native-dtype action reconstruction, rewards, effort,
raw margins, crossings and oracle accounting without new simulation or solves.
Generated records remain excluded from Git; reproducibility requires the archived
inputs and, for read-only full-result verification, the preserved output archive.

## Finalization verification

All **83 included tests passed** in both the reviewed workspace and a clean
export without unfinished IQL dependencies. A disposable five-case pilot reproduced
**330 NPZ bundles (2,676 individual arrays) bitwise**, including all **24 solver
candidate vectors**. Selection, LQR, controller definitions, protocol and frozen
threshold files also matched byte-for-byte. This reran only the technical pilot
in the disposable export, not the original pilot or full study.

The export's independent audit rechecked all 33,310 fixed and 384 optimization
replay pairs from the preserved full study. Maximum independently reconstructed
reward discrepancy was 5.26478694e−10; maximum NLP dynamics residual was
1.56711311e−11. LQR's DARE residual was 7.10542736e−15 and its local closed-loop
spectral radius was below one. These numerical checks do not add a global safety
or optimality guarantee.

Preservation checks covered **53,424 pre-existing files** at finalization, with
only the three explicitly authorized context-document exceptions. Earlier
experiment source, tests, configurations, reports, datasets, generated artifacts,
accepted pilot evidence and all nine unfinished IQL files were unchanged.
Original pilot/full manifests still validate 1,160/8,287 artifact hashes and their
68/73 scientific source hashes. Supplemental finalization evidence is retained
locally under `results/safe_policy_improvement/finalization/`, outside Git.
