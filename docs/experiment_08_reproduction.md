# Experiment 08 reproduction and archive verification

The [full report](experiment_08_report.md) records **40/40 independently checked
nonviability certificates**, with no unresolved PD-unsafe states. The
[pilot report](experiment_08_pilot_report.md) remains the historical five-case
record. Certificates apply to the declared real-valued discrete model and exact
archived initial values; Gymnasium replay is separate evidence.

## Dependencies and tests

Use CPython 3.13.2 and the pinned dependency chain in
`requirements_experiment_08.txt`. The recorded arithmetic libraries are gmpy2
2.3.2, GMP 6.3.0, MPFR 4.2.2, and MPC 1.4.1. The runners reuse committed
Experiments 04–07 helpers; **no unfinished Experiment 06 IQL files are required**.
The independent certificate checker itself uses only the Python standard library.

```sh
python3 -m venv .venv/experiment08
.venv/experiment08/bin/python -m pip install -r requirements_experiment_08.txt
MPLCONFIGDIR=/tmp/pendulum_08_mpl .venv/experiment08/bin/python -m unittest discover -s tests -p 'test_viability*.py' -v
```

The 21 tests exercise interval arithmetic, independent certificate validation,
invalid-record rejection, and reporting. They do not run the cohort study or train
controllers. Bitwise simulator reproducibility across different platforms or
library builds is not assumed.

## Verify the historical reviewed archive

The original read-only commands are:

```sh
MPLCONFIGDIR=/tmp/pendulum_08_mpl .venv/experiment08/bin/python experiments/08_viability_pilot.py --verify
MPLCONFIGDIR=/tmp/pendulum_08_mpl .venv/experiment08/bin/python experiments/08_viability_study.py --verify
```

They check completed hashes and never regenerate trajectories or certificates.
They also enforce the **historical whole-workspace preservation snapshot**. The
authorized finalization updates to `CURRENT_STATUS.md` and `RESEARCH_CONTEXT.md`
therefore cause those strict commands to reject the changed documentation. This
is an expected historical-snapshot mismatch, not changed scientific evidence.
Both documents had their `040dea5` contents when the study ran. Do not rewrite
saved preservation manifests, source hashes, completion markers, or reports to
make the old commands accept new documentation.

After those two authorized updates, the following read-only check verifies all
saved source, artifact, report, and preservation hashes, except exactly those two
documentation paths. It also verifies that the historical exceptions match their
actual `040dea5` contents:

```sh
.venv/experiment08/bin/python - <<'PY'
import hashlib
import json
from pathlib import Path
import subprocess

root = Path.cwd()
allowed = {'CURRENT_STATUS.md', 'RESEARCH_CONTEXT.md'}
def digest(data):
    return hashlib.sha256(data).hexdigest()
def read(path):
    return json.loads(path.read_text())
def check(base, hashes):
    for name, expected in hashes.items():
        assert digest((base / name).read_bytes()) == expected, str(base / name)

for phase, report in [('pilot', 'experiment_08_pilot_report.md'),
                      ('full_study', 'experiment_08_report.md')]:
    folder = root / 'results/viability' / phase
    done = read(folder / 'completion.json')
    assert done['passed']
    check(folder, done['artifacts_sha256'])
    check(root, read(folder / 'source_hashes.json'))
    before = read(folder / 'preservation_before.json')['sha256']
    for name in allowed:
        original = subprocess.check_output(['git', 'show', '040dea5:' + name])
        assert digest(original) == before[name], name
    check(root, {name: value for name, value in before.items() if name not in allowed})
    assert digest((root / 'docs' / report).read_bytes()) == done['report_sha256']
    print(phase, 'artifact/source/report hashes and preservation checks passed; only the two authorized context documents excluded')
source = root / 'results/recoverability/full_study'
check(source, read(source / 'completion.json')['artifacts_sha256'])
print('Experiment 07 prerequisite artifacts unchanged')
PY
```

The reviewed artifacts also retain independent checker outcomes, exact generation
repeats, 171 bitwise replay comparisons, initial-state identities, and all 201 raw
states per simulator trajectory. Hash verification preserves that evidence; it
does not recompute the independent checks or simulator rollouts.

## Fresh reproduction

Use a separate disposable checkout and output archive, never the reviewed
workspace. Generated artifacts are intentionally excluded from Git. Restore the
complete reviewed `results/recoverability/full_study/` archive, including its
completion manifest and matching source hashes, or reproduce Experiment 07 using
its [reproduction instructions](experiment_07_reproduction.md). Experiment 08
requires those source artifacts for exact initial states, archived PD actions,
and verification. It needs no D0, acquired training datasets, model checkpoints,
or IQL diagnostics.

With no existing Experiment 08 output directories, run:

```sh
MPLCONFIGDIR=/tmp/pendulum_08_mpl .venv/experiment08/bin/python experiments/08_viability_pilot.py --pilot
MPLCONFIGDIR=/tmp/pendulum_08_mpl .venv/experiment08/bin/python experiments/08_viability_study.py --run
```

The full runner requires the completed matching pilot and reuses its five cases;
only the remaining 52 states are computed. Existing directories require explicit
`--resume`; inspect incomplete work before using it. Both runners preserve
existing reviewed documentation and write fresh generated reports under
`results/viability/`. Compare scientific records separately from timing,
environment paths, Git metadata, protected-file counts, and the historical pilot
test count. No solver or global viability grid is needed for the reviewed result.
