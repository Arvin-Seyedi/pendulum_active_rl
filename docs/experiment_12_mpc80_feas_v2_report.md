# E12_MPC80_FEAS_v2 — MPC80-only expert feasibility

**Scoped expert-feasibility gate PASSED.**
Completion: 16/16 initial-state pairs; 32/32 complete scientific rollouts; 32/32 fully verified executions.

V1 remains FAILED. The original FAILED overall E12 qualification and original
FAILED 0.001 symbolic/PyBullet agreement criterion are unchanged. This study
qualifies only the frozen MPC80 expert scope; it is not a new four-controller benchmark.

## Frozen setup and verification

The original sixteen PCG64-120001 initial states and exact binary64 values/IDs
were loaded directly. Each state used original/repeat, then native uninstrumented
and independent recorded-motor replay, in the frozen order. All native controller,
model, URDF, Q/R, bounds, IPOPT, horizon-80, warm-start, reference, seed-1337,
six-second, 50-Hz control and 1,000-Hz physics definitions were preserved.

The sealed v2 observer records current `_physics(rpm)` arguments and actual
submitted force/torque API arguments; it does not use the stale command cache
inside physics callbacks. Grouped thrust/effort are derived from these commands,
not measured hardware outputs. Comparisons ignore only transient world/body IDs,
not motor commands, forces, frames, application points or control/physics indices.

## Per-state original-run performance

| ID | Qualified original/repeat | Return | Discounted return | Position RMSE, m | Absolute effort, N²s | Hover-relative effort, N²s | Min normalized margin |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | pass/pass | -6.262323368 | -5.360471632 | 0.040699195 | 0.228686297 | 0.014523518 | 0.287622836 |
| 1 | pass/pass | -6.185165842 | -5.469635227 | 0.048750649 | 0.220945768 | 0.007587719 | 0.290720578 |
| 2 | pass/pass | -16.713018346 | -14.904452995 | 0.094946913 | 0.222137649 | 0.008468812 | 0.305008357 |
| 3 | pass/pass | -5.637751411 | -4.781712085 | 0.034327481 | 0.228235647 | 0.013588267 | 0.304649115 |
| 4 | pass/pass | -8.905610161 | -8.012704033 | 0.066437011 | 0.221272051 | 0.007268175 | 0.260225542 |
| 5 | pass/pass | -78.787551231 | -70.262168161 | 0.197426305 | 0.242540074 | 0.020098821 | 0.099286685 |
| 6 | pass/pass | -25.659737363 | -23.019394189 | 0.118663135 | 0.226873722 | 0.012527913 | 0.292861229 |
| 7 | pass/pass | -8.686273832 | -7.733363699 | 0.066513465 | 0.228544916 | 0.015427773 | 0.305008320 |
| 8 | pass/pass | -5.252799778 | -4.580616163 | 0.041981329 | 0.219708759 | 0.006015841 | 0.301204977 |
| 9 | pass/pass | -9.490286858 | -8.487390764 | 0.067349912 | 0.231155308 | 0.017360418 | 0.294652572 |
| 10 | pass/pass | -34.927901282 | -31.363931143 | 0.140200771 | 0.223366313 | 0.009365598 | 0.261398890 |
| 11 | pass/pass | -7.905405508 | -6.697898180 | 0.050689825 | 0.228773808 | 0.013148190 | 0.305008609 |
| 12 | pass/pass | -9.285868565 | -8.315502483 | 0.068538787 | 0.229963744 | 0.016741500 | 0.305008130 |
| 13 | pass/pass | -24.522659756 | -21.887142707 | 0.100899579 | 0.234610027 | 0.018699609 | 0.163833538 |
| 14 | pass/pass | -38.573395035 | -34.419497246 | 0.130186688 | 0.236460951 | 0.018958143 | 0.075652809 |
| 15 | pass/pass | -37.070258047 | -32.800931962 | 0.139907108 | 0.235570779 | 0.018710731 | 0.278746033 |

## Full-cohort statistics

Statistics below use sixteen originals, not thirty-two independent samples.
Returns are negative unscaled native cumulative quadratic cost; tracking
RMSE uses the native control-boundary reference. Raw safety uses every
physics state including initial and terminal samples.

| Metric | Mean over 16 | Worst over 16 |
|---|---:|---:|
| Return | -20.241625399 | -78.787551231 |
| Cost | 20.241625399 | 78.787551231 |
| Discounted return, gamma=0.99 | -18.006050792 | -70.262168161 |
| Position RMSE, m | 0.087969885 | 0.197426305 |
| Absolute squared-thrust effort, N²s | 0.228677863 | 0.242540074 |
| Hover-relative squared-thrust effort, N²s | 0.013655689 | 0.020098821 |
| Thrust total variation, N | 1.327881584 | 2.343759446 |

Mean near-bound saturation fractions by channel: [0.0, 0.0].
Mean absolute terminal error `(x,vx,z,vz,theta,omega)`: [0.0876244006999539, 0.3958905939730853, 0.05929854022021724, 0.39780081625337693, 0.09841736808921375, 0.04642700875000208].
Worst absolute terminal error by component: [0.08762494760352564, 0.3958942729384611, 0.05929864127520279, 0.39780093818653606, 0.09841866589361302, 0.046430330918786004].

| Component | Minimum lower raw margin | Minimum upper raw margin | Minimum one-step reserve |
|---|---:|---:|---:|
| x, m | 1.00010194475 | 1.14358350077 | 0.999913370359 |
| vx, m/s | 29.2149569895 | 27.4649536769 | 27.4650082344 |
| z, m | 0.571429367922 | 0.422647868131 | 0.422449304472 |
| vz, m/s | 28.7822812546 | 27.5745652633 | 27.5735472865 |
| theta, rad | 1.08793957493 | 0.732937623893 | 0.734136732691 |
| omega, rad/s | 6.08315337341 | 1.32039062648 | 1.28231603225 |

Minimum normalized raw margin: 0.0756528087895.
Minimum derived input margins, N: [9.900963646169991e-07, 9.901579765339563e-07].
Zero raw state/input violations and contacts were observed in all qualified executions.
No negative margin was rounded into safety. Derived effort is not electrical energy.

## Numerical reliability, model discrepancy and latency

Independently accepted retained candidates: 9600.
Maximum independent/native graph reconstruction difference: 1.42108547152e-14.
KKT residual maxima: `{'primal_inf': 9.903635438511981e-09, 'stationarity_inf': 5.324486612029444e-09, 'dual_sign_inf': 0.0, 'complementarity_inf': 1.0746384046651986e-07, 'normalized_complementarity_inf': 9.869311626219633e-09}`.
Native success, finite candidates, frozen v2 numerical tolerances and positive
one-step reserves are separate checks. KKT conditions do not imply global NLP optimality.

| Timing scope | Measured calls | Median ms | p95 ms | p99 ms | Max ms | >20 ms | Missing |
|---|---:|---:|---:|---:|---:|---:|---:|
| solver_only | 9600 | 23.351042 | 43.327646 | 66.151440 | 286.198375 | 9600 | 0 |
| native_action | 9600 | 24.337480 | 44.585866 | 68.383920 | 309.950500 | 9600 | 0 |
| supervised_action | 9600 | 87.165438 | 123.085932 | 190.496364 | 861.457917 | 9600 | 0 |

Maximum retained complete-record source closure: 3.5527136788e-15.
Maximum symbolic/plant one-control-step disagreement: 0.0189680501631.
Maximum discrepancy-contract excess: 3.71604125506e-15.
Maximum recorded native solver iteration count: 26.

Cold starts and every 20-ms overrun are retained. Native action latency is
an adjusted exclusive-span estimate; supervised latency includes qualification
work. Solver time is the native solve boundary, not just printed IPOPT kernel
time. Fixed-step offline simulation waits for computation; hard real-time
readiness is not established. All individual overrun records and timing categories
are retained in the machine-readable audit.

## Resource accounting and preservation

Actual recorded native solver attempts: 9600.
Actual physics steps: 576000 (recorded lower bound 576000).
Conservative reservations: 9600 solves / 576000 steps.
Remaining original allowance: 3070 solves / 366000 steps.
Shared study CPU/wall charges after this audit: 1489.932000 / 1700.796000 s.
Maximum recorded worker peak RSS: 507297792 bytes.
These carry v1 and both preflights; no cap or failed reservation was reset or refunded.
One worker and the original 1,800-CPU-second, 3,600-active-wall-second, 4-GiB
and one-second action-watchdog limits remain. Scientific-worker peak RSS is
retained per episode; aggregate parent peak is not separately reconstructed.

Preservation checks: `{'passed': True, 'tracked_files': 197, 'prior_result_and_data_files': 59268, 'authority_files': 87, 'upstream_files': 289, 'IQL_files': 9, 'accepted_artifacts': 5438}`.
Accepted protocols, source/configuration hashes, v1 failure, historical reports,
drilling authority, decision history and all nine IQL files remain unchanged.

## Scientific interpretation and limitations

The unchanged MPC80 controller generated complete, reproducible, empirically
constraint-satisfying demonstrations on the sixteen frozen initial states.
Native replays and independently driven motor replays support live instrumentation
nonintrusiveness on this cohort, including submitted force/motor command equality.
This supports a finite-cohort offline simulation-expert role. It does not establish
a certified operating region, formal robust/continuous-time/hardware safety,
reward optimality, safe action diversity, learning headroom or acquisition benefit.

This is a deterministic fixed-cohort study; repetitions are reproducibility
checks, not independent statistical replicates. The sixteen states remain
qualification/development data. Any future learned-policy evaluation requires
a fresh independent cohort. Safe-action diversity, equal-budget active acquisition
and beyond-expert learning each require separate evidence and approval.
No action probes, counterfactual continuations, additional baselines, RL/IQL work,
controller tuning, commits or pushes were performed. Stop for scientific review.

## Frozen identities

```json
{
  "accepted_config_sha256": "91a17c5d45c537d0cc59672b36bd1a2f7d508f6e554538039635b8bc222a3839",
  "accepted_preflight_seal_sha256": "920a6281aa0abd007d7a838ccefba393c5aa8f2d1e767939179958b06fadf969",
  "execution_binding_sha256": "4c0c31ebe0658a420bc8bfe764bdf92a33610908456f63928b1b20c6df3e3c96",
  "frozen_model_sha256": "90bbdccb6ecaaa08a5e435dc9694fe7c34984ae26c2ae5fd05ef87a6464641d3",
  "frozen_reference_sha256": "37bc9f54732b761af119de91a1df1d5f0d932876d24468d8dfd7994ad27e503a",
  "screening_manifest_sha256": "0f13eaf7313623d5c77f8851b090dab147aab3aae95b346307242e04869b9bd9",
  "screening_states_npy_sha256": "c97f1fed3fcdd17a17de70d61042bef0ce7e0323a2312e38000d1a788fbb4f17",
  "source_sha256": {
    "experiments/12_mpc80_feasibility_v2_study.py": "4656a3ada696ef3cb1cc0a2adc9ca57f27cfa54fee5c18be78689c6e6475b793",
    "src/quadrotor_h2_requalification_v2.py": "16c916da7f5af4a9c4b87ffc8a6b2d8528e350e6f14e355c557c17eebe2f5049",
    "src/quadrotor_mpc80_execution_v1.py": "c1a8b4993bb034df7fdc251682d049c04cf175420505b770c06c2021d7e0f65e",
    "src/quadrotor_mpc80_execution_v2.py": "67052b3393a0b72ccbeda30df517fb2508392f395d1aa43f7c01cdd5bd79a74f",
    "src/quadrotor_mpc80_feasibility_v1.py": "94eb8b8f501993d243d99ebfacd17c189358072969e13d011e437ce999ad8e6b",
    "src/quadrotor_qualification_telemetry_v2.py": "3ac4433f9e17894fdb86e8dca5dcdff78348d0715baef2ad0ad214e521fda0ed",
    "tests/test_quadrotor_mpc80_execution_v2.py": "bafa884498cd9efaa1d26d418adeddf3eb6270a27c44ce70e7cc574e1c18d7b8"
  }
}
```
