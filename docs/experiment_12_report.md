# Experiment 12 — Quadrotor qualification milestone

The reviewed pilot and diagnostics are complete. **Overall benchmark qualification
remains FAILED**, including the original **0.001** symbolic/PyBullet agreement
criterion. Successful controller executions do not reverse that finding or
authorize screening, action probes, acquisition, training or another controller run.

## Frozen task and outcome history

The upstream safe-control-gym source is commit
`6b5391d014f36fdfa0f9d22d92c77387e5274308`. The pinned Python 3.10.19 environment,
dependencies, source/vehicle hashes and exact states are recorded in the archived
manifests and `requirements_experiment_12.txt`. The published six-second planar
figure-eight uses states `(x, vx, z, vz, theta, omega)`, two grouped thrusts,
50-Hz control and 1,000-Hz physics. Published objectives, constraints, models,
vehicle parameters and MPC horizons are unchanged. Initial pilot refinement at
2,000 Hz is retained separately.

| Stage | Accepted outcome; historical failures remain immutable |
|---|---|
| Four-case technical pilot | NMPC20/40/80 completed both repetitions at both physics resolutions without observed violations. Native Darwin LinearMPC failed before applying any action. The original model-agreement gate failed. |
| Focused diagnostic | The linear conic program dispatches to unavailable conic/IPOPT on Darwin despite requesting QRQP. Symbolic/plant differences include motor-link geometry, cached attitude and integration. Small standalone checks did not alter the plant or controller definitions. |
| Linear adapter revalidation | Separately versioned `linear_conic_qrqp_v1` changes solver dispatch/backend-specific options only. Eight complete executions were empirically safe and independently replayed; this is not successful execution of the unchanged native implementation. |
| E12_MMAQ_v1 | 26/32 executions completed. NMPC40/80 completed all four held-out cases twice without observed violations. Adapted linear H0/H2 and NMPC20 H2 were incomplete. Overall qualification failed. |
| Integrity audit v2 | Corrected finite-side complementarity/sign and nonfinite handling. All 7,932 retained qualification candidates pass corrected KKT checks; missing rejected candidates remain missing. No episodes or solves were run in this audit. |
| Targeted H2/MPC20 v2 | Two new complete 300-control executions pass v2 verification, repeat bitwise and have independent recorded-action replays. This new success does not relabel the historical incomplete v1 executions. |

On the original four cases, mean returns were −14.736232809 (NMPC20),
−9.080739742 (NMPC40), −9.071085719 (NMPC80), and −9.428348568 (adapted linear40).
On the four held-out cases, NMPC40/80 mean returns were −39.429394810 and
−39.331860180; NMPC80's worst return was −65.816857638 and minimum normalized
raw margin 0.200745220. These are small technical cohorts, not population estimates.

## Targeted H2 verification and verifier correction

Both new executions contain 300 controls, **6,001 raw physics states**, zero
observed state/input violations or contacts, and **300 independently checked
candidates per run**. Repetitions and instrumented/uninstrumented action replays
are bitwise identical. Return is **−25.138729807869687** (discounted:
−18.694905608519996), tracking RMSE 0.101105482, squared-thrust effort
0.242895121 N²s and minimum normalized raw margin **0.2699882352585654**.

For `lower <= g <= upper`, the signed multiplier is `lambda=mu_upper-mu_lower`.
Equality multipliers are free; complementarity is evaluated only for finite
inequality sides, with unsupported signs checked separately. The original v1
checker could multiply a tiny wrong-sign multiplier by an absent infinite side;
it also had nonfinite/equality handling defects. Separately versioned v2 rejects
invalid/nonfinite iterates and preserves original numerical tolerances. Numerical
KKT conditions do not prove global NLP optimality or state safety.

At newly captured control index **54**, upper-only row 18 has multiplier
**−1.0940617921979725e−9**, within the frozen 1.01e−8 sign tolerance. The v1
calculation on this **new** candidate produces infinite complementarity and
rejects it; v2 passes. The first 54 retained controls, solver vectors and matching
state histories reproduce the old prefix bitwise. **The missing historical
step-54 rejection vector was not recovered.** Its full-horizon outcome remains
unavailable. The new point reproduces the mathematical defect, rather than
retrospectively supplying absent historical evidence.

## Numerical, software and timing limitations

The symbolic lever arm is 0.0280721392131 m versus the URDF's 0.028 m.
Recorded-transition reconstruction also accounts for motor-link attitude caching,
executed RPM/thrust conversion and semi-implicit integration. This explains
recorded discrepancies but is not a general PyBullet floating-point safety bound.
The prospective MMAQ componentwise contract remains a separate, limited
qualification mechanism; it does not turn the original 0.001 failure into a pass.
NMPC80's held-out maximum original scaled disagreement remains 0.01326088346.

QRQP H0 reaches its unchanged 1,000-iteration limit with cycling. H2's printed
trace ends at iteration 12 on a singular/regularity-loss path, **not a demonstrated
iteration-limit event or infeasibility certificate**. Earlier contrary summaries
remain preserved and are corrected in the integrity report. Final failed QP
vectors and complete historical call latency were not retained. The verified
linear warm-start view/nominal-offset aliasing is a possible contributor, not an
established cause. Native nonlinear warm-start behavior is also preserved.

Future v2 failure capture retains available current candidates, parameters,
primal/dual diagnostics, solver status, nonfinite values, timing and partial
trajectories, explicitly marking missing backend information. It introduces no
fallback control action. Failed or incomplete episodes are never counted as safe.

For H2 v2, solver-only latency median/p95/p99/max was
**7.187/11.802/12.889/72.745 ms**; **2/600** exceeded 20 ms (both initial calls).
Full supervised latency was 39.569/52.566/56.237/89.244 ms, with **600/600**
overruns. Historical adapter overruns and missing telemetry remain recorded;
MPC80 frequently exceeds 20 ms even in printed solver-kernel time. Offline
fixed-step execution is feasible on these records; **hard real-time readiness is
not established**. Verification/persistence overhead is not labelled solver time.

## Reproducibility, resources and decision

Generated evidence is preserved outside Git under `results/benchmark_selection/`.
Finalization uses only retained records: it verifies source/artifact hashes,
costs, actions, raw margins, physics equations, KKT conditions, repeats and saved
independent replay evidence. It does not execute new episodes or optimizations.
See [finalization reproduction](experiment_12_finalization.md) for the clean-export
procedure and regression results.

The targeted task consumed exactly 600 solves and 36,000 physics steps, including
four action replays. The conservative remaining original allowance is
**12,970 solver calls and 948,000 physics steps** (about 13,397 recorded CPU
seconds before finalization verification overhead). No unused failure reservation
was released. Read-only finalization incurs no additional solves or physics steps;
its CPU is recorded separately. No new full-study allocation is approved.

Successful finite-horizon executions, solver feasibility, prediction agreement,
empirical simulator safety and mathematical/hardware safety are distinct findings.
No safe-action support, learning headroom, acquisition benefit or learned-policy
safety has been established. The proposed next decision is a **limited,
prospectively specified MPC80-only simulation-expert feasibility study**, subject
to separate design and budget review, rather than continued open-ended baseline
debugging. The overall gate remains failed. Drilling physical-model development
and unfinished IQL remain paused.

## Traceable reviewed records

- [Original pilot](experiment_12_pilot_report.md) and [original reproduction](experiment_12_reproduction.md).
- [Focused diagnostic](experiment_12_diagnostic_report.md) and [adapter revalidation](experiment_12_linear_revalidation_report.md).
- [MMAQ_v1 protocol](experiment_12_mmaq_v1_protocol.md) and [qualification report](experiment_12_mmaq_v1_report.md).
- [Integrity audit](experiment_12_qualification_integrity_report.md) and [H2 v2 requalification](experiment_12_h2_requalification_v2_report.md).

These historical reports retain their original source hashes, outcomes and
stage-specific resource balances. The latest balance above supersedes earlier
balances without rewriting their evidence.
