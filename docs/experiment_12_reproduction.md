# Experiment 12 initial technical-pilot reproduction

This entry point supports only cases T0–T3 and the four approved controllers.
It has no screening, counterfactual-menu, acquisition or training mode.
No earlier research implementation or authoritative model is modified.

## Frozen dependencies

Use CPython 3.10.19 and the exact package versions in
`requirements_experiment_12.txt`. The accepted execution used a disposable
environment under `/private/tmp/e12_venv` and the unmodified upstream checkout
under `/private/tmp/e12_safe_control_gym`, at commit
`6b5391d014f36fdfa0f9d22d92c77387e5274308`.

The minimal upstream environment/MPC dependency closure is used directly from
the checkout. Optional GP, RL and synthesis dependencies are not installed or
imported. No third-party source patches or alternative controllers are used.

On this macOS ARM64 SDK, PyBullet 3.2.7's bundled legacy zlib fails because
it defines `fdopen` to NULL. Build configuration `CFLAGS=-Dfdopen=fdopen`
prevents that macro definition without source edits. The built platform wheel,
its SHA-256, compiler identity, complete environment lock, upstream source
archive, file hashes and unsuccessful build logs are preserved under the
generated pilot output. Prefer the saved wheel on the identical platform.
Rebuilding on another compiler/platform does not promise binary identity.

## Commands

Run from the project root, with an isolated environment and source checkout:

```sh
git clone https://github.com/learnsyslab/safe-control-gym.git /private/tmp/e12_safe_control_gym
git -C /private/tmp/e12_safe_control_gym checkout 6b5391d014f36fdfa0f9d22d92c77387e5274308
```

Install `requirements_experiment_12.txt` in the isolated Python 3.10.19
environment, using the documented build flag if a PyBullet source build is
required. Set OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, MKL_NUM_THREADS and
VECLIB_MAXIMUM_THREADS to 1. Put MPLCONFIGDIR in a writable temporary directory.
The runner supplies these settings to its sequential workers too.

```sh
/private/tmp/e12_venv/bin/python experiments/12_quadrotor_pilot.py --upstream /private/tmp/e12_safe_control_gym --prepare
/private/tmp/e12_venv/bin/python experiments/12_quadrotor_pilot.py --upstream /private/tmp/e12_safe_control_gym --run
/private/tmp/e12_venv/bin/python experiments/12_quadrotor_pilot.py --upstream /private/tmp/e12_safe_control_gym --verify
```

**Do not rerun the accepted output.** The runner reuses completed attempts,
including failures, and refuses to overwrite partial folders. For a fresh
authorized reproduction, supply `--output` pointing to a disposable location.
Independent preservation verification requires a baseline manifest there;
the reviewed baseline records pre-existing project files and the 87 original
authority files. The actual run's baseline predates every E12 source addition.
For an authorized reproduction against the same preserved research tree, copy
the accepted `preservation_before.json` into the new output directory before
`--prepare`. For another research tree, capture and review its own baseline
instead of claiming to preserve files it does not contain.

Both physics resolutions receive controller-level repeats and both types of
recorded-action replay. The initial driver scheduled one 2,000 Hz execution per
condition; final verification added its controller-level repeat. The initial
driver/source snapshot is retained, and the final driver reproduces the complete
64-attempt plan without rerunning any completed attempt. The episode executor,
upstream source, seeds, configuration and controllers were unchanged throughout.

`--verify` reads saved trajectories and upstream sources; it does not run a
controller or simulator. It independently reconstructs costs, references,
thrusts, safety margins and one-step equations, checks saved independent
replays, compares repeats/refinement and checks preservation hashes.

Generate the report and plots from `src.quadrotor_reporting.report(output)`
and `src.quadrotor_reporting.plots(output)`. These use saved records only.

## Interpretation and platform blocker

All cases use the figure-eight tracking task; T0 is a hover **initial state**.
Native observations and commands are float64, while state/input space bounds
are float32. All 6,001 nominal or 12,001 refined physics states are captured
for a completed episode, including initial and terminal states. Failed and
zero-transition attempts receive no complete-episode safety classification.

The upstream Darwin linear-MPC implementation selects IPOPT for a conic
problem even though its published YAML requests QRQP. The pinned CasADi
distribution has no conic-IPOPT plugin. This failure is a technical finding
of this pilot; do not switch solvers, spoof the platform or repair the
controller during reproduction. Likewise retain the native nonlinear warm
start's row shift and the symbolic/URDF geometry discrepancy. The 0.001
one-step discrepancy threshold is not relaxed.

The scientific report and machine-readable `verification.json` distinguish
successful archive checks from failed numerical-readiness gates. No full
screening or counterfactual study is authorized by these commands.
