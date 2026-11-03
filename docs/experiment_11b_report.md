# Experiment 11B — Static physical-reference consistency audit

Static pilot completed for scientific review. No new trajectory, controller call,
compensation, actuator bound, synthesis or training was performed. The original
11A safety classifications and evidence are unchanged.

## Scope and provenance

The authoritative ZIP SHA-256 remains
`b86ce2a62fe26871f40eb674377b1b17ddedac84ef422019c47e59f1787be1b5`.
The audit freezes the native five-node binary64 matrices, IC, Vs, nominal inputs,
loads, geometry, section/material parameters, source-member hashes and accepted
11A completion manifests. No decimal display was used to regenerate a reference.
The dissertation PDF and Chapter 5 TeX hashes are separately recorded.

Generated evidence: `results/drilling_static_audit/approved_static/` (ignored by
Git). Implementation: `src/drilling_static_audit.py`,
`src/drilling_static_interval.py`, `src/drilling_static_verification.py` and
`src/drilling_static_analysis.py`; entry point and numerical contract are
`experiments/11b_static_audit.py` and `configs/experiment_11b.json`.

## Mathematical comparison

The full FEM equation is `M Uddot + D Udot + K U = Fext`, with Rayleigh damping.
Each of five nodes has `(x,y,alpha,beta)` displacement coordinates. The native
geometry embedding `U=T q` sets `y=x tan(theta)` and `beta=alpha tan(theta)`.
`S` selects global x/alpha coordinates, giving the native reduction
`KM=S M^-1 K T`, `DM=S M^-1 D T`, `FM=S M^-1 Fext`.

For `q_d=IC+Vs*t`, `e_q=q-q_d`, `e_v=qdot-Vs`, exact substitution gives

```
e_q_dot = e_v
e_v_dot = -KM e_q - DM e_v + (FM-FM_d) + r0 - t KM Vs
r0 = FM_d - DM Vs - KM IC
```

The authoritative error equation omits `r0-t KM Vs`. Chapter 5, equations
(5.35)–(5.37), invokes the nominal balance and `KM Vs=0` to make this cancellation.
`Simulation/IC_check.m:85` instead computes the saved displacement using normal
equations and records its residual at line 87. No IC was recalculated here.

## Results and status

| Audit item | Status | Evidence / qualification |
|---|---|---|
| Source provenance | PASS | Original ZIP, all 87 authority files and accepted 11A artifacts verified. |
| Executed matrix/coordinate reconstruction | PASS | Worst binary64 norm-relative discrepancy 2.74316e-12; worst interval-midpoint comparison 6.18106e-12, both below 1e-10. |
| Algebraic error transformation | PASS | The difference is exactly `(0,r0-t KM Vs)` in rational arithmetic. The native omission remains. |
| Exact reference identities | FAIL | Both residual vectors are nonzero for exact archived operands; all components exclude zero in both interval reassemblies. |
| Conditioning audit | PASS, with severe ill-conditioning | Numerical ranks, singular values and backward errors recorded; this does not mean the matrices are well-conditioned. |
| Physical constrained-reduction justification | UNRESOLVED | The stated geometric constraints alone do not justify the native projection; reaction closure remains unspecified. |

Exact rational products of archived binary64 operands give:

| Quantity | Largest component |
|---|---:|
| `r0` | 2.36554798765712e-6 rad/s², component 6 |
| `KM Vs` | 0.00762011974485422 rad/s³, component 4 |

Independent directed-rounding reassembly starts from frozen primitive geometry,
angles, material/section data, IC, Vs and nominal loads. It does not use saved
rotation, FEM or reduced matrices as assembly inputs. MPFR elementary functions
and interval Gaussian elimination enclose the calculations. At both precisions,
the largest components are 2.36554154352788e-6 rad/s² and
0.00762011974663970 rad/s³. Cross-precision enclosures overlap in all 20 residual
components.

| Precision | Maximum r0 enclosure width | Maximum KM Vs enclosure width |
|---|---:|---:|
| 128 bits | 2.16753e-21 | 3.29318e-22 |
| 256 bits | 6.07334e-60 | 9.22738e-61 |

These widths describe arithmetic uncertainty conditional on the frozen inputs,
not uncertainty in physical parameters. Exact endpoint ratios are retained;
printed values are approximate displays.

In the dominant component, the archived dot-product rounding bound is about
6.325e-15 rad/s³. A bound on the native-versus-reassembled matrix perturbation
times Vs is about 1.694e-11 rad/s³. Neither explains a 0.00762012 rad/s³ term.
The term persists after independent high-precision assembly. This conclusion
does not rule out different physical conventions or uncertain engineering data.

At 20 seconds, the dominant derived-minus-implemented acceleration discrepancy
is approximately -0.152400579227 rad/s² for archived operators and
-0.152400579251 rad/s² for the independently reassembled operators. This is a
matched-state acceleration-balance discrepancy, **not a trajectory deviation**.
No corrected dynamics were integrated and no causal explanation for P3 is claimed.

## Conditioning and mechanical assumptions

With fixed SI normalization (1 m, 1 rad, 1 s), native condition estimates are:
`cond2(M)=2.77128e6`, `cond2(KM)=4.03124e17`, and `cond2(DM)=1323.16`.
The declared SVD threshold `max(shape)*eps*sigma_max` gives numerical ranks
20/20 for M, 18/20 for full K, 9/10 for KM, and 10/10 for DM. These are
threshold-dependent numerical ranks, not proofs of exact rank.

For the saved IC balance, the normwise backward error is 4.31320e-12 and
componentwise backward error 5.75686e-8. Small backward error does not establish
accurate or unique IC in this ill-conditioned problem. The small r0 persists
in high precision, but its size must be separated from the much larger
reference-stiffness effect and from unknown physical-model accuracy.

`ST=I` and `CT=0` hold exactly in the reconstructed binary64 operators. Executed
planar rotations have maximum orthogonality error 4.44089e-16 and determinant +1
to rounding. The literal printed plus sign in Chapter 5 equation (5.13) instead
produces frame orthogonality errors 0.1305–0.7934 for these elements. The executed
cross-product convention was retained; no source equation was repaired.

For ideal holonomic constraints, the usual reduced stiffness operator is
`K_alt=(T' M T)^-1 T' K T`. The reported comparison is
`||K_alt-KM||_F / ||K_alt||_F = 0.140737` in the fixed SI coordinates;
the corresponding damping comparison is 0.140550. The implementation uses the
Frobenius matrix norm, not an induced spectral norm. The **14.07%** figure is
a numerical operator comparison, not measured physical modeling error; it does
not establish that the conventional alternative is physically correct for this
plant. Also `S M^-1 C'` is nonzero.
Thus omitted normal reactions can affect retained accelerations. Chapter 5
asserts radial contact-force cancellation, but that assertion and the supplied
force vector do not establish the required complete reaction closure. This is
an unresolved mechanical assumption, not approval to replace the reduction.

The native nodal bit-frame reference velocities are 0.001 m/s and 10 rad/s.
Projection into the terminal element chord frame instead gives
0.000997858923239 m/s and 9.97858923239 rad/s. These are different frame
conventions; the audit retained the native nodal definition for bit loads.

## Archived-point and nominal-load checks

All six cases were checked at the predefined nine indices in both archived
executions: 108 points, with six bitwise-identical original/repeat sample pairs.
The independent checker separately verified all 54 unique point equations and
540 componentwise exact-rational rounding bounds without importing the MPFR
producer. Maximum floating identity discrepancy was 1.28210e-11, below the
1e-8 scaled tolerance and at most 0.01484 of the conservative rounding bound.
The position part of the matched-state transformation cancels exactly.

`X1-U1_global-UU1` was zero at all checked points. After initialization, physical
position reconstruction agreed within 5.68434e-14. Initial nonzero-error cases
retain the known mismatch between nominal native position arrays and positions
reconstructed from the error, rather than silently repairing it.

At 0, 1, 5, 10 and 20 seconds, independently recomputed nominal loads are constant:
WOB=40616.28276328623 N, TOB=1860.358879951857 N·m to binary64 rounding.
Mathematical-pi and native-binary64-pi load calculations are recorded separately.
Full nominal balance agrees with `r0-t KM Vs` plus its separately recorded
load-reconstruction term; maximum floating identity discrepancy is 3.46324e-12.
The nonlinear nominal load law supplies no compensating time-dependent term.
The nominal surface commands remain -457923.7912416592 N and
1860.358879951857 N·m, **not actuator limits**.

Initial native nominal loads differ from the memoryless load law at perturbed
initial errors. Native-minus-memoryless WOB/TOB differences are:

| Case | WOB difference (N) | TOB difference (N·m) |
|---|---:|---:|
| P0 | +15213.936779 | +712.540861 |
| P1 | -5338.111978 | -733.990397 |
| P2 | +15213.936779 | +712.540861 |
| P3 | -389.068540 | -53.496924 |
| P4 | -12341.191360 | -1696.913812 |
| P5 | 0 | 0 |

This is native initialization bookkeeping, separate from reference forcing; it
is not evidence of an additional persistent one-sample physical delay.

## Reviewed pilot verification and costs

During the reviewed pilot, all 21 applicable tests passed. Independent exact-rational verification passed,
all 53,435 prior project files and 87 authority files were preserved, and the
nine unfinished IQL files remained unchanged. At that stage no previous experiment,
context file or decision-log entry was modified. Generated evidence remains ignored by Git.

There was one computational worker. The initial partial static pass took 3.373 s
wall time; resumed archived-point work took 7.264 s, with 6.403 CPU s. The two
interval assemblies each took about 0.185 s and were not repeated. Independent
verification including full preservation took about 11 s wall time. Peak worker
RSS was 1,280,180,224 bytes, below the 2 GiB cap; all work remained well below the
30 CPU-minute limit. No cap or numerical tolerance was relaxed.

Failed setup/read/check attempts are preserved: sandbox denial of external `ps`,
unsupported DATA-limit setup, a gravity vector-shape comparison, and a row-vector
timestamp reader. The completed interval results were reused after the reader
fix. A worker/supervisor metadata-lifecycle mismatch was resolved by verifying
the exact initial status bytes at their preserved snapshot and independently
checking final status. Scientific artifact hashes were never exempted. Source
snapshots and amendments distinguish execution-time code from subsequent I/O-only
maintenance; no arithmetic method or scientific result changed.

For finalization, use the read-only entry point with a new output path:

```
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python experiments/11b_verify_static.py --output /private/tmp/e11b-check-unused.json
```

This reassembles the two approved interval precisions, recomputes exact products,
checks all archived sample lineages and independent rational identities, and
writes outside accepted evidence. It executes no controller or trajectory. The
original writer/report functions remain available for a fresh static reproduction
into an unused directory; they must not rewrite the reviewed archives. See the
[reproduction guide](experiment_11b_reproduction.md) for clean-export inputs,
historical context exceptions and preserved source snapshots. MATLAB is not used.

## Research decision for review

The reference inconsistency is confirmed for the declared reduced numerical
model and is not explained by binary64 dot-product cancellation or tested FEM
assembly rounding. Engineering input uncertainty and physical constraint-reaction
conventions remain separate unresolved questions. Preserve the original idealized
error simulator; do not insert the residual term automatically. Next review should
resolve the physical reference and contact/reaction assumptions before approving
physical-model simulation changes or drilling safe-RL development. The missing
bounded-actuator contract remains independently blocking.

## Finalization verification

All **108 applicable regression tests passed** in both the workspace and clean
source export. The nine unfinished IQL files were excluded from the export and
were not imported. Workspace/export static replay reproduced both exact-rational
vectors, all 128/256-bit interval endpoints, four nominal-load variants, and all
108 sample points re-extracted directly from the 12 archived 11A workspaces.
Independent rational verification again passed 54 unique equations and 540
componentwise rounding bounds. The frozen model, tolerances and arithmetic
methods were unchanged.

Accepted 11A/11B reports, manifests, failed attempts and numerical outputs remain
byte-identical. All 53,435 historical project files, 87 authority files and nine
IQL files passed preservation checks, allowing only the separately authorized
context/log documentation updates through their exact historical Git hashes.
The finalization verifier wrote zero accepted-archive files and launched zero
MATLAB processes or new research/control trajectories. Logs and export identities
are retained separately in `results/drilling_static_audit/finalization/`.
The versioned report clarifies the Frobenius comparison and records finalization;
the original pilot report and its source snapshots remain preserved.
