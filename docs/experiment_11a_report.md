# Experiment 11A — Authoritative ideal-input native reproduction pilot

Numerical reproduction completed. This is not a bounded-input safety study, controller redesign or RL experiment. The physical actuator contract remains blocked.

## Frozen execution and provenance

P0 uses the top-level script byte-for-byte. P1–P4 use the exact Chart_Plot initial errors and native IC_Test kernel. P5 is the newly approved zero-error input, not an archived case. The standalone Chart script is not substituted for P3. The ZIP SHA-256 is b86ce2a62fe26871f40eb674377b1b17ddedac84ef422019c47e59f1787be1b5. Source/data identities, exact expressions and executed initial binary64 values are retained.

MATLAB 24.2.0.2863752 (R2024b Update 5), MACA64, single computational thread, seed 111001. Base MATLAB was available. LMI synthesis functions were absent but are not a dependency of these native stored-table runs. Sandboxed startup failed before MATLAB output; authorized native execution succeeded.

Twelve separate processes used isolated temporary directories and read-only, byte-identical native files. IC_Test workspaces were saved using a false conditional breakpoint immediately before line 547, before display clipping; a fixture verified caller access and nonbreaking behavior. P0 was captured after native script completion. No authoritative source was edited.

## Six-case results

| Case | min omega lower (rad/s) | min omega upper (rad/s) | min axial lower (m/s) | min axial upper (m/s) | Violating states | First crossing (s) | Failed lookups |
|---|---:|---:|---:|---:|---:|---:|---:|
| P0 | 2.76972507697 | 2.0875211352 | 4.66523798995e-05 | 0.000919121164881 | 0 | none | 0 |
| P1 | 0.573627799362 | 0.0670798087964 | 7.85890092462e-05 | 3.04250719781e-06 | 0 | none | 0 |
| P2 | 2.7697289576 | 2.07363093726 | 4.66523155331e-05 | 0.000921801383809 | 0 | none | 0 |
| P3 | -0.436432830732 | 0.396460427224 | -7.97052998834e-05 | 0.000580493837415 | 401 | 0.1323 | 411 |
| P4 | 1.06436208499 | 2.33917140206 | 0.000199296895519 | 4.22019883481e-05 | 0 | none | 0 |
| P5 | 4.77 | 5.7 | 0.000506895816719 | 0.00243000789475 | 0 | none | 0 |

All 200,001 raw error states were checked in every execution. P0/P1/P2/P4/P5 remained strictly inside the sampled safety domain. P3 had 401 lower-rotational violations starting at 0.1323 s and 358 lower-axial violations starting at 0.1347 s. Its 411 lookup failures began at 0.1318 s; 401 attempts had at least one complex index. Lookup failures precede the first safety crossing; this temporal ordering does not establish that they caused it. The native stale-gain/transformed-memory fallback was preserved and verified. No first lookup failed, so that failure path was not dynamically exercised or repaired.

All physical error/input/load histories were finite and real. P3 has complex transformed lookup indices, not complex physical trajectories. There were no native execution failures, timeouts or incomplete horizons. Failed or truncated histories are explicitly ineligible for safe classification.

P1/P4 have smaller axial margins than the historical context because the initial sample is now included. Excluding it gives 2.01414884471e-5 / 5.98700570543e-5 m/s, reproducing the earlier approximate values; including it gives 3.04250719781e-6 / 4.22019883481e-5. Earlier records were not changed.

## Reference consistency and equilibrium

P5 retained exactly zero error throughout and constant nominal inputs: it is an equilibrium of the implemented reduced error map, not proof of an exact physical reference trajectory.

For the moving reference `q_d(t)=IC+Vs*t`, define `r0=F_M,d-D_M*Vs-K_M*IC`. Its reconstructed physical acceleration balance is `F_M,d-D_M*Vs-K_M*q_d(t)=r0-t*K_M*Vs`, which is omitted from the implemented error dynamics. Native MATLAB gives:

- max |r0| = 2.36554798527244e-06; the largest component is angular acceleration in rad/s^2.
- max |K_M Vs| = 0.00762011974485406; the largest component is angular acceleration rate in rad/s^3.
- max |r0 - saved err| = 4.66281145639069e-11; the largest component is angular acceleration in rad/s^2.
- max |r0 - 20 K_M Vs| = 0.15240057922651 rad/s^2 in the dominant angular component.

The 0.152400579 rad/s^2 value is an acceleration-balance discrepancy evaluated on the reference at 20 seconds, not a measured trajectory deviation.

The conditions r0=0 and K_M Vs=0 are not satisfied exactly by the assembled matrices and stored reference. Independent FEM reassembly confirms the nonzero K_M Vs. The implemented error equation omits r0 - t K_M Vs. No compensation, recalculated IC, new controller or altered dynamics was introduced. The stiffness matrix condition estimate is approximately 4.03e17; small balance-residual differences must not be conflated with the much larger nonzero reference-stiffness term. These are failed physical-reference identities, separate from successful native reproduction.

The four conclusions remain separate: same-platform bitwise numerical reproduction passes; five cases satisfy sampled raw-state safety and P3 does not; physical-reference consistency is limited by the nonzero residuals; physically admissible bounded-input safety remains blocked. No formal safety guarantee or causal explanation for P3 is established.

## Verification and cost

All 88 included tests passed, including five new E11A fixtures; unfinished IQL tests were excluded. Six native original/repeat pairs matched all 85 numerical workspace fields bitwise (510 field comparisons); MAT-file timestamps are excluded and gain-table identity is established by its source checksum. No cross-language bitwise equivalence is claimed.
Independent matrix reassembly worst norm-relative difference: 2.74316e-12. All 2,400,000 recorded transitions were independently reconstructed without a sequential rollout; maximum absolute discrepancy 3.55271e-15, maximum state-scaled discrepancy 3.46945e-15. Full controller correction reconstruction maximum absolute discrepancy 4.09273e-12, worst norm-relative discrepancy 7.12924e-16. Representative load/transform checks and independent MATLAB raw-safety checks passed. Absolute/scaled/per-variable diagnostics are retained in audit JSON files.

State scaling for transition errors is 1 m/rad for displacement coordinates and 0.001 m/s / 10 rad/s for axial/angular velocity coordinates. The prespecified source checks require matrix norm-relative error <1e-10 and transition/load/transformation/correction scaled error <1e-8. These numerical tolerances never relax the strict safety boundaries.

Twelve native runs: 2,400,000 integration transitions and 2,400,012 raw states. Process wall total 216.319 s; native wrapper time including capture/plot 117.852 s; core capture time including MAT serialization 108.509 s. Independent representative audit 16.056 s; full-history verification and preservation overhead are separate. No broader integration refinement was run.

Preservation passed for 53435 pre-existing project files and 87 original authority files, including the ZIP, unpacked MATLAB/data files, Experiments 01–10, decision log and nine unchanged IQL files. All disposable authoritative copies remain byte-identical. No commit/push or prior experiment changes occurred.

## Readiness and reproduction

Ready for scientific review of further numerical validation of the implemented ideal-input map. Not ready to assert exact physical-reference equivalence or bounded-input safety. Review the omitted reference term, initial physical-array/error mismatch and P3 fallback/domain exits before approving any next validation study. Optimization, alternatives, controller changes, training and integration refinement remain unperformed.

Generated evidence is under results/drilling_reproduction/native_pilot and is excluded from Git. native.mat retains original histories; derived_histories.npz and applied_vector_histories.npz label independent reconstructions, including the initial nominal loads and the 200,000 applied commands. The final computed command is retained separately and was not applied. Initial WOB/TOB array zeros are placeholders, not initial applied loads.

Reproduction requires the pinned native MATLAB release and exact external archive path/hash in configs/experiment_11a.json, plus the existing Python NumPy/SciPy environment. First run the documented MATLAB license check. Run experiments/11a_native_drilling.py --run only with appropriate native execution permission. Completed executions are verified/reused; interrupted artifacts cause a refusal to overwrite. --audit reads saved records only. e11a_reference_checks(root,output) performs static MATLAB safety/residual checks; python -m src.drilling_pilot_verification generates the final verification/report from those saved outputs. Use a disposable checkout/output location for any fresh reproduction.

## Finalization clarification

This versioned report adds the reviewed distinctions and equations without changing numerical findings or methodology. The exact execution-time report remains at `results/drilling_reproduction/native_pilot/scientific_report.md`, with its original report hash and completion manifests unchanged. Finalization uses the additive read-only verifier; it does not rerun the six control cases. See [reproduction and archive verification](experiment_11a_reproduction.md). A physical-model consistency investigation requires a separate scientific design review before safe RL controller development.
