# CSTR source, evidence and reproduction

The accepted CSTR milestone is an archive-backed reproducibility result. The
repository contains implementation, tests, reports and the
[runtime/source identity lock](../configs/cstr_reproducibility_lock.json), not
the generated optimizer records, trajectories or source bundles. A clean source
export alone cannot reproduce the numerical audit.

The saved evidence must be supplied read-only under `results/cstr_source_audit/`,
`results/cstr_expert_pilot/technical_v1/`,
`results/cstr_numerical_diagnosis/zero_solve_v1/` and
`results/cstr_expert_pilot/no_bound_relax_v2/`. Full historical preservation
also needs the prior research archives and the nine unfinished IQL files as
external, unchanged inputs. Their presence for verification does not make them
part of this commit. Preserve the original FAILED stock-pilot record and its
missing post-query telemetry.

The authoritative upstream is do-mpc v5.1.1 commit
`90e0d7ada888a4cc75ea3769214de5105764c640`; the audit hashes all 269 files.
The preserved source bundle has SHA-256
`2160f5fae2bf0aa0be614f100b4e55503a3d1db786b679c207a929062d0c3c4c`.
The accepted runtime is macOS arm64, CPython 3.10.19, with the exact 40 package
versions, distribution RECORD hashes and native-library hashes in the lock.
The original layout uses `/private/tmp/cstr_do_mpc_90e0d7a` and
`/private/tmp/cstr_audit_venv_90e0d7a`; the latter uses read-only dependencies
from the frozen E12 environment. `src/cstr_expert_pilot.py::verify_identities`
checks these and the original audit's completion seal before archived review.
Version numbers alone do not prove byte-identical wheels or native libraries.

The numerical pilot uses the nominal `CSTR_approximate_mpc` variant. F is a
dilution rate in h^-1, Q is kJ/h, temperatures are Celsius and Arrhenius terms
use Kelvin. It does not import the robust CSTR example's uncertain parameters,
objective or soft state constraint. The original source-audit results are
external evidence, including the documented plot-label/unit mismatch and the
fact that a constraint box does not establish viability.

The read-only option inspector uses CasADi 3.7.2 internal layout and is only
validated on the pinned ABI. Its C++ source is committed. The original build
command, compiler identity, exact header archive identity and library hash are
retained in the excluded v2 evidence. A recompiled library on another platform
must not be assumed identical or safe to use without a separately reviewed ABI
check. `configs/cstr_reproducibility_lock.json` records the key identities.

For a clean export, supply the excluded evidence at the paths above and the
exact runtime/source checkout, then run from the export root:

```sh
/private/tmp/cstr_audit_venv_90e0d7a/bin/python experiments/cstr_verify_pilot_v2.py \
  --output /private/tmp/cstr_v2_read_only_review.json
```

The output must not already exist. This verifies saved candidates, costs,
margins, action replays, initialization diagnostics and source seals; optimizer
and integrator entry points are blocked. Full historical preservation adds
`--full-preservation` and requires the archived datasets and read-only Git
history. The clean-export verification for this milestone used the preserved
audit/results and prior untracked preservation inputs as external inputs. It
was not a network-independent reconstruction of those excluded artifacts.

The prior audit snapshots encode the original tracked research context. After
later authorized updates to `CURRENT_STATUS.md`, `RESEARCH_CONTEXT.md` and the
decision log, a verifier must supply their historical bytes in a disposable source
view or use the pre-update Git export; do not rewrite the sealed historical
snapshots to make a current working-tree check pass.

This milestone authorizes no rerun, expert-data generation, learner training,
acquisition comparison or IQL work. The [v2 report](cstr_expert_pilot_v2_report.md)
documents empirical finite-cohort success, narrow margins and remaining limits.
