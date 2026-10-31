# Experiment 09 reproduction and archive verification

The [reviewed report](experiment_09_report.md) records **3,230/3,230 real-model
nonviability certificates** and the complete **6,561-state PD baseline audit**.
The [pilot report](experiment_09_pilot_report.md) is a separate historical record.
The full study skipped every alternative-controller and optimization search by
the predefined all-certified gate. Zero alternatives found is not an empirical
failure rate for the controller bank or optimizer.

## Environment and required input

Use CPython 3.13.2 and `requirements_experiment_09.txt`. Its pinned dependency chain
includes Gymnasium 1.2.3, NumPy 2.3.2, PyTorch 2.10.0, CasADi 3.7.2 and gmpy2 2.3.2.
The recorded arithmetic build is GMP 6.3.0 / MPFR 4.2.2 / MPC 1.4.1; the solver
build is IPOPT 3.14.11 / MUMPS 5.4.1 on macOS. Cross-platform bitwise reproduction
is not assumed. The runners use POSIX locks and process groups.

Restore the exact archived `results/expert_safety_map/expert_safety_map.npz`:

```text
SHA-256 2234e7c2c1d5e09595ed7f924286aa824748e747e1a4facc8be435c5cd6c622a
```

This is the only prerequisite result archive for fresh Experiment 09 execution.
It contains the original axes, margins and crossing-time map, not historical
actions or trajectories. Coordinates must not be regenerated or rounded, and
approximately symmetric points cannot share certificates. The source imports
committed earlier helpers but needs no IQL files, D0, acquired datasets, model
checkpoints, or Experiment 07/08 result archives. The checker itself uses standard
library rational arithmetic, independently of MPFR.

```sh
python3 -m venv .venv/experiment09
.venv/experiment09/bin/python -m pip install -r requirements_experiment_09.txt
MPLCONFIGDIR=/tmp/pendulum_09_mpl .venv/experiment09/bin/python -m unittest discover -s tests -p 'test_recoverable_pd*.py' -v
```

The E09 suite has 13 tests. It includes actual unchanged-Experiment-01 parity on
all five pilot cases, exact initial-state and dtype checks, replay checks, worker
authorization, and full-study outcome gates. Existing earlier tests remain intact.
Tests using this archive require its exact checksum; absence is not silently
repaired with substitute data.

## Preserve the reviewed workspace

Before finalization documentation updates, these read-only commands passed:

```sh
MPLCONFIGDIR=/tmp/pendulum_09_mpl .venv/experiment09/bin/python experiments/09_recoverable_pd.py --verify
MPLCONFIGDIR=/tmp/pendulum_09_mpl .venv/experiment09/bin/python experiments/09_full_study.py --verify
```

They also verify historical whole-workspace preservation snapshots. The authorized
updates to `CURRENT_STATUS.md`, `RESEARCH_CONTEXT.md` and
`docs/RESEARCH_DECISION_LOG.md` intentionally differ from their execution-time
contents at commit `4008b7f`. These strict commands therefore reject those updated
documents. Do not rewrite saved snapshots, source hashes, completion markers or
scientific reports to accept the changes.

The following read-only check preserves every scientific source/report/artifact
hash while allowing exactly those three documentation changes. It checks their
historical hashes against actual Git history:

```sh
.venv/experiment09/bin/python - <<'PY'
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

for phase, report in [('pilot', 'experiment_09_pilot_report.md'),
                      ('full_study', 'experiment_09_report.md')]:
    folder = root / 'results/recoverable_pd' / phase
    done = read(folder / 'completion.json')
    assert done['passed']
    check(folder, done['artifacts_sha256'])
    check(root, read(folder / 'source_hashes.json'))
    before = read(folder / 'preservation_before.json')['sha256']
    for name in allowed:
        original = subprocess.check_output(['git', 'show', '4008b7f:' + name])
        assert digest(original) == before[name], name
    check(root, {name: value for name, value in before.items() if name not in allowed})
    assert digest((root / 'docs' / report).read_bytes()) == done['report_sha256']
    print(phase, 'artifact/source/report and preservation hashes passed; exactly three authorized documentation exceptions')
PY
```

The full archive retains 6,561 bitwise baseline replay pairs, 6,526 independently
checked extremal records, exact initial identities, all 201 raw states, and the
stage gate. Hash verification preserves those records without new simulation.
Independent saved-array and rational-checker revalidation can be run separately
using `independent_baseline_audit` and `independent_certificate_audit` from
`src/recoverable_pd_study_analysis.py`; neither calls the simulator or optimizer.

## Fresh reproduction in a disposable checkout or clean export

Never rerun into the reviewed result folders. Use a separate Git checkout with
the exact benchmark restored and dependencies installed. A clean export used for
runner verification also needs a local Git baseline because source/Git provenance
is recorded. Keep all nine unfinished IQL files and all other generated outputs
out of that export.

In the disposable copy only, move the copies of the two reviewed E09 reports
aside **before** creating execution snapshots:

```sh
mkdir reviewed_reports
mv docs/experiment_09_pilot_report.md docs/experiment_09_report.md reviewed_reports/
MPLCONFIGDIR=/tmp/pendulum_09_mpl .venv/experiment09/bin/python experiments/09_recoverable_pd.py --pilot
MPLCONFIGDIR=/tmp/pendulum_09_mpl .venv/experiment09/bin/python experiments/09_full_study.py --run
```

The runners refuse to overwrite a differing reviewed report. Timings, paths,
Git metadata and protected-file counts differ between checkouts; retain the
reviewed copies and compare scientific arrays/certificates separately. Do not
move reports in the original reviewed workspace. Existing outputs require an
explicit `--resume`; inspect interrupted files first. Completed matching runs
verify rather than regenerate their scientific artifacts.

The completed full-study branch certifies every primary state and skips the
searches. The pilot exercises all nine gains and the unchanged three-start E07
solver on reference ID 44. If screening ever leaves an uncertified case, the full
runner retains `followup_required.json` and returns without a completion marker;
it does not silently declare viability, nonviability, or a successful search.
Automatic full-grid controller/optimizer continuation was not needed or tested
for this milestone. Such a branch must not be represented as completed evidence.

Finalization checks use a clean export with only intended source/configuration,
tests and documentation, plus the exact benchmark as an explicitly restored,
ignored input. They run the full included test suite, reproduce the five-case
pilot, compare its trajectories, certificates and paired solver decisions against
the accepted pilot, and recheck the full saved study with the export's audit code.
Generated trajectories, logs, checkpoints, datasets and result archives remain
excluded from Git. No new scientific initial conditions or training are required.

The reviewed finalization check passed all **70 included tests**. Its fresh pilot
matched **188 NPZ array bundles bitwise**, all **10 exact certificate records**,
all five archived-baseline comparisons, and all **six paired solver candidates**
against the accepted pilot. The clean export's audit code also independently
verified all **6,561 baseline replay pairs** and **6,526 full-study extremal
records** (43,682 rational transition rechecks), without new full-study simulation.
These checks validate the included dependency set and preserved scientific
evidence; they do not broaden the model or safety claims.
