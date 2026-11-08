# E12_MPC80_FEAS_v1 — zero-solve preflight report

**Static preflight PASSED; ready for separate execution review. Cohort execution
is NOT authorized.** No new controller action selection, optimization solve,
physics step, replay episode, action probe or training occurred. The original
FAILED E12 overall qualification and FAILED 0.001 symbolic/PyBullet criterion
remain unchanged. This is not a new four-controller benchmark.

## Checks and evidence

| Check | Result |
|---|---|
| Original cohort provenance | PASS: original manifest/NPY hashes, all 96 binary64 coordinates, IDs 0–15, hex/byte representations, PCG64 120001 raw-bit reconstruction and row-major draw order match. Originals were loaded directly and never rewritten. |
| Environment and source | PASS: Python 3.10.19 and all archived package versions match; 289 pinned upstream files, configurations and vehicle assets preserved. Actual solver setup remains IPOPT with `expand=True, error_on_fail=False`. |
| Complete MPC80 graph | PASS: 646 variables, 492 parameters and 1,778 constraints. Independent objective, every constraint/bound and Lagrangian derivative agree with the native graph at all 2,400 retained held-out MPC80 candidates. Maximum discrepancy 1.4210854715202004e−14, below 1e−10. |
| Corrected numerical KKT | PASS: all 2,400 archived candidates satisfy v2 checks and strict native success. No thresholds changed. |
| Failure capture | PASS: static native extractor and archived-value substitution; available candidate/primal/dual/status retained before injected rejection, NaN/Inf preserved, timing and raw partial prefix retained, no fallback. Missing-field, exception and nonfinite paths are also covered by regression tests. |
| Archived discrepancy contract | PASS: 7,200 archived MPC80 intervals from 24 technical/held-out records, including 1,000/2,000-Hz evidence; 43,200 independent exact-rational component checks. Bounds use only current/past state, command conversion and prior attitude cache. |
| Repeatability | PASS: fresh-process static repeat produces identical canonical scientific JSON. It does not rerun historical trajectories or recover missing historical solver vectors. |
| Regression | PASS: 89 zero-solve E12 tests, zero failures/errors/skips, including 20 new preflight/ledger tests. Backend solve, action selection and physics-step entry points were blocked. |
| Preservation | PASS: 197 tracked files, 59,126 prior result/data files, 87 authority files, 289 upstream files, 5,438 sealed E12 artifacts and all nine IQL hashes unchanged. Context and decision-log files were not edited. |
| Evaluation separation | PASS: original qualification and four held-out qualification cohorts have distinct exact coordinates. No future learned-policy evaluation set is defined; these sixteen states are explicitly qualification/development data. |

Native setup instantiated one disposable graph/environment at the previously
archived H0 initial condition per verification process. It selected no action,
integrated no physics and evaluated none of the sixteen cohort states. Guards
recorded zero forbidden production calls. The injected failures are static test
fixtures, not newly observed controller failures.

Maximum archived MPC80 KKT residuals:

| Quantity | Maximum | Frozen acceptance |
|---|---:|---:|
| Primal | 9.921097914421705e−9 | <=1e−4 |
| Stationarity | 6.206087907467683e−9 | <=1.01e−8 |
| Dual sign | 0 | <=1.01e−8 |
| Normalized complementarity | 9.743975138633095e−9 | <=1.01e−8 |
| Raw complementarity | 1.143085299424281e−7 | Diagnostic; not the normalized acceptance metric |

Maximum matched-state source closure is **2.6645352591003757e−15**.
Maximum observed error minus the analytic bound is **4.237264222008711e−15**,
within the unchanged 1e−10 empirical numerical allowance. Minimum one-step
reserves by `(x,vx,z,vz,theta,omega)` are
`[1.1120815658767285, 27.78167283676896, 0.4678268936540328,
27.639690634629165, 0.9106493282687961, 3.477095286890007]`, in component units.
These are archived-record diagnostics, not sixteen-state screening results.
All reserve components are positive. No general Bullet floating-point enclosure,
recursive robust safety, continuous-time safety or hardware guarantee follows.

## Frozen identities

Upstream commit: `6b5391d014f36fdfa0f9d22d92c77387e5274308`.
Protocol seal: `6dbb0ccd3076bad0f5513237a14317413a1eca05e152a26a154f1fc7a607b2bf`.

| Source/configuration | SHA-256 |
|---|---|
| `configs/experiment_12_mpc80_feas_v1.json` | `f465c20596d068127f84e13f41c3b6b50d8417a0eb0b9ad488cb45062c4ccad9` |
| `src/quadrotor_mpc80_feasibility_v1.py` | `94eb8b8f501993d243d99ebfacd17c189358072969e13d011e437ce999ad8e6b` |
| `experiments/12_mpc80_feasibility_preflight.py` | `6ac7e4a5e3951363633f54b86d670794454812f17c6af1b5b2c444e03ed92f7e` |
| `tests/test_quadrotor_mpc80_feasibility_v1.py` | `8b4504b7719c72cfeb13b5ab9eab42c50233e61903c7b8bb3da718297dd4e057` |
| `docs/experiment_12_mpc80_feas_v1_protocol.md` | `c43363f3be8797b098b56e98c2483e1a0fde7713debc6607b8b4db401318b7c8` |
| `requirements_experiment_12.txt` / environment lock | `d047dd760e7cab1dec1e08f4a3abee8f87e1531542536c73b0d42c4f68c09999` |

Original cohort manifest/array SHA-256:
`0f13eaf7313623d5c77f8851b090dab147aab3aae95b346307242e04869b9bd9` /
`c97f1fed3fcdd17a17de70d61042bef0ce7e0323a2312e38000d1a788fbb4f17`.
Native MPC80 program/readout hashes:
`d52a79bf7f360acf8d9831952a3632c40479dfeac02f9351df0f7751c86bb32d` /
`69e1e31b94ef1149d8ff0e61463e4dc4ba559e811ddf4c1161ff1885a620d03a`.
Canonical scientific JSON:
`f551c34a1a882e3f520e840315709818d059901feaf44a630f907e3c87a4c631`.

`controller_contract.json` contains all upstream/configuration/asset identities,
physical constants, exact software bounds, Q/R, dependency versions and native
memory/solver conventions. `native_dependency_identity.json` additionally freezes
the Python executable and PyBullet/CasADi/NumPy/SciPy core binaries, checked against
installed distribution RECORD hashes. The historical PyBullet wheel/build identity
is preserved; the wheel payload was not independently revalidated here. This
does not establish cross-platform binary equivalence. Rebuilding/changing the
environment requires renewed preflight.

## Reservation and execution boundary

The frozen 96-job ordering is state ID ascending, original then repeat, each
followed by native replay then motor replay. Its earmarked envelope is:

| Work | Jobs | Solves | Physics steps |
|---|---:|---:|---:|
| Scientific executions | 32 | 9,600 | 192,000 |
| Native replays | 32 | 0 | 192,000 |
| Motor replays | 32 | 0 | 192,000 |
| Total | 96 | 9,600 | 576,000 |

**Zero jobs are dispatched or execution-charged.** The conservative original
allowance remains 12,970 solves/948,000 steps. The envelope would leave
3,370 solves/372,000 steps, without reclaiming failed reservations. CPU/wall
verification overhead is charged separately within the study's 1,800/3,600-second
caps; the live ledger is authoritative for its current balance. Both static
passes plus supplemental dependency checks charged 57.7 CPU s and 95.0 active
wall s before final integrity/report accounting. Peak measured process RSS was
**588,742,656 bytes**, below 4 GiB. Final accounting is retained separately.

The ledger requires a separate approval tied to this seal and preflight report,
a durable reservation before a dispatch receipt, fixed ordering and completion
of prior work. Failed/interrupted jobs retain reservations and halt the campaign.
No automatic retry or release exists. The entry point exposes only static
preflight/reverification; there is no cohort execution mode. Future execution
wiring remains outside this authorization and must use the frozen ledger and
validated capture primitives after explicit review approval.

## Decision, limitations and reproduction

There are **no failed static checks or unresolved numerical/provenance blockers
for review of execution in this frozen local environment**. Historical overall
qualification/model disagreement, incomplete baselines, missing telemetry and
hard real-time limitations remain failures/limitations, not repaired outcomes.
The historical CPU balance predates later checks and is not represented as an
exact current cumulative balance; this preflight's own metered/billed overhead
is explicit. Native graph/KKT agreement does not establish good control on new
states, global optimality or safe alternative-action support.

Recommend separate approval review of the frozen MPC80-only cohort. A passing
cohort would still require another design/budget review before action-diversity
probes, acquisition or learning. The remaining envelope cannot fund the original
full nine-action continuation study with complete replay verification.

Evidence is outside Git at
`results/benchmark_selection/mpc80_feas_v1_preflight/`: protocol/source seal,
preservation baseline, candidate capture fixtures, both archived audits,
repeatability, tests, dependency identities and resource ledger. Do not overwrite
this completed evidence. To reproduce the static checks, use the pinned environment
and the entry point with `--upstream /private/tmp/e12_safe_control_gym` and either
`--preflight --output <fresh-directory>` or, once, `--verify-archived` on a completed
preflight. Neither option runs episodes or optimization. Existing verification
attempts are retained and never overwritten. No commit or push was performed.
