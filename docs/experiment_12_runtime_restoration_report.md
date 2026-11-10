# Experiment 12 — exact runtime restoration and zero-solve revalidation

**Pinned runtime restored; frozen zero-solve preflight PASSED. No Pilot B execution
or fresh execution approval occurred.** The accepted stopped-attempt report remains
unchanged. Original FAILED E12 overall and 0.001 agreement gates remain FAILED;
accepted MPC80_FEAS_v2 remains historical finite-cohort evidence.

## Phase A: confirmed failure mechanism and remaining causal uncertainty

The project working directory and mounts were accessible. `/tmp` and `/private/tmp`
resolved to the same original device/inode; the upstream directory was not a
redirecting symlink. All 289 expected source/configuration/asset files and required
Git metadata were absent at that location, while directories remained. This was
a damaged/incomplete original working checkout, not a wrong working directory or
an unresolved `/tmp` alias. No evidence establishes which process removed files;
an operating-system cleanup explanation remains unconfirmed.

The interpreter symlink resolved to the original surviving CPython executable,
whose frozen SHA matched. The surviving libpython and four core package binaries
were present. However `pyvenv.cfg` and standard-library files including
`encodings/__init__.py` were absent. PYTHONHOME/PYTHONPATH were unset. The observed
encoding initialization failure follows from the incomplete installation/config,
not from a verified Python-version substitution or environment-variable override.
Damaged roots were retained at `/private/tmp/e12_damaged_runtime_20261109/`.

The preserved source archive was intact: SHA-256
`5433f6d28c9b81378ce95f0caddd29bd30953269cd3c4112cfa865078c9519d9`,
containing all 289 expected files with zero hash mismatches. Original native
package RECORDs were inspected individually during recovery; PyBullet's authentic
RECORD survived even though its generated installation metadata did not.

## Phase B: exact identities restored without exceptions

The checkout at `/private/tmp/e12_safe_control_gym` is clean and detached at
**6b5391d014f36fdfa0f9d22d92c77387e5274308**. Trusted exact-revision Git objects and
the independent archived inventory agree on all **289/289 file hashes**, including
controller, task, constraints, dynamics and URDF. Damaged checkout metadata was
quarantined, not fabricated or silently reused.

The original installer version, uv 0.9.8, restored CPython **3.10.19 (main,
Oct 28 2025, 12:01:32) [Clang 20.1.4 ]**. The restored platform string matches
`macOS-27.2-arm64-arm-64bit`. The interpreter is
`/private/tmp/e12_venv/bin/python`, based on
`/private/tmp/e12_python/cpython-3.10.19-macos-aarch64-none`.

| Identity | Verified SHA-256 |
|---|---|
| Python executable | `1d583a3812cabef5b1db8bfd437e08e00f4ae1b5341ede784945934a9da30e7d` |
| Original/restored libpython | `7b9f3a9c1241f40ee56ce6a97d9cab89f64a478a40be11abbf199a9d9a224031` |
| PyBullet native binary | `e95ab89254c3c64340cd5bbaff3ad9792352943bb882f917dd4ddb9b622a7de3` |
| CasADi native binding | `49a479a5c79efe5ceaf1494b1f16b67f8bb4ca9cb28b277617185ed4032eda96` |
| NumPy native core | `4e23a325a373848240c6ac122d4749c084001dae43f4b547774577f91f5c2ced` |
| SciPy fblas | `8c8b35b60894b025648d4fc66ce3534eeb5240c7ec44153be0d5b3ad1fa0fca7` |

All **38 pinned package versions**, all **four frozen native-binary hashes** and
all **four frozen package RECORD hashes** match. Standard-library initialization
and required native/scientific dependency imports succeed.

Two intermediate identity checks failed and were preserved before exact recovery:

1. A staged Python libpython hash differed because uv embeds its installation
   prefix in the Mach-O library ID. `otool` confirmed the two paths. Installing at
   the original prefix reproduced the full surviving library hash exactly; no
   different library was accepted. The executable hash matched throughout.
2. Installing the verified archived PyBullet wheel reproduced its native binary
   but added wheel-origin/cache metadata, so its installed RECORD initially
   differed. The quarantined authentic RECORD matched frozen hash
   `2bc6f4f4e97c071cf84f14d8d0305a7e495d0508d797c7379f9e71da8aaf6004`.
   All **4,020 hashed payload entries** were validated against it. Its missing
   two-byte `uv_build.json` marker was recovered with its original size/hash;
   only newly generated nonoriginal `direct_url.json`/`uv_cache.json` were removed
   after preserving copies. The authentic RECORD bytes were restored. This is
   exact restoration of a recovered manifest, not a substituted hash or exception.

The rejected candidate installation records and intermediate failures remain in
`results/benchmark_selection/runtime_restoration_v1/`. The original runtime and
frozen protocol hashes were not weakened. Subsequent Python installation used
`--no-bin`; the staging-created version-specific convenience shim was corrected
to the verified original interpreter target, with no default-Python/PATH change.

## Phase C: zero-solve revalidation

**154/154 applicable tests pass**, zero errors/failures/skips. Genuine optimizer,
engine creation, force application and physics-step APIs were blocked; forbidden
real-call counters remain empty. Source/controller flow was tested only with
scripted archived solution/state lookups and mocked application backends.

Revalidation confirmed the original 36 labels, nine predicted input-inadmissible
labels, ID-0/menu-3 selection and exact frozen command representations. It checked
nominal-query/post-query memory, detached substitution, native warm-start behavior,
reference indexing/padding, telemetry timing, corrected KKT logic, failure recording
and stop-rule tests. Specifically:

- 1,200 archived initial-guess/warm-start comparisons.
- 20 independent archived candidate KKT/program/reference checks.
- 1,200 archived numerical-contract/reserve checks, including 7,200 exact-rational
  majorant components; maximum matched-state source closure 3.10862446895e−15.
- 720 mocked native application-boundary comparisons.
- Maximum program reconstruction discrepancy 3.5527136788e−15.

These are archived equation evaluations and mock operations, not new control
trajectories. There were **zero MPC solves, zero new physics steps, zero
counterfactual continuations and zero new training transitions**. No empirical
alternative-action safety/performance or live nominal fidelity was established.

## Preservation and resources

Independent post-test preservation passes for all manifested preflight sources
and artifacts, 227 tracked project files, 87,315 accepted v2 artifacts, 59,268
prior data/result files, 87 drilling authority files and nine unchanged untracked
IQL files. The accepted stopped Pilot B report, integrity records and resource
ledger remain unchanged, as does its unexecuted wiring draft. No scientific
protocol, threshold, controller behavior, solver setting, asset, historical report
or decision-log entry was modified. No commits or pushes occurred.

Carried usage remains **1,764.902 CPU /2,345.058 active-wall seconds**. This task's
charged restoration, rejected identity checks, revalidation and preservation cost
**65.838281 CPU /171.993232 active-wall seconds**, including explicitly recorded
conservative preparation/reporting allowances. New cumulative charges are
**1,830.740281 CPU /2,517.051232 active-wall seconds**, within approved ceilings
**2,044.402 /2,874.158**. Raw stage/subprocess timing is retained; conservative
charges are not represented as exact measured kernel time. The revalidation
process's measured peak RSS is 410,435,584 bytes (below 4 GiB); installer-wide
aggregate peak RSS was not logged and is not retrospectively invented.

Remaining approved time is **213.661719 CPU /357.106768 active-wall seconds**.
The original solve/step allowance is unchanged at **3,070 /366,000**, with no
reservation refunded or new scientific job reserved. Historical charges were
never reset. The remaining time is smaller than Pilot B's original 300/600
planning envelope; renewed execution review must address this before dispatch.
No extra allocation is requested or silently assumed.

## Readiness decision

**Runtime identity and accepted zero-solve preflight: READY for renewed review.**
The runtime-integrity blocker is resolved without a provenance exception.
**Immediate Pilot B dispatch: NO-GO.** Fresh execution authorization is still
required. The previously documented execution-wiring draft remains untested and
unsealed; this restoration-only task did not implement or qualify a dispatcher.
Its final zero-solve wiring/source seal and pre-dispatch resource accounting remain
required, alongside review of the reduced available time. Do not resume the old
stopped attempt. No other experiment, acquisition/RL/IQL work or drilling
investigation was performed.
