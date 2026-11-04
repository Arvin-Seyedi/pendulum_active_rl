# Experiment 11C — Static physical constraint and reference pilot

Completed the four approved archived snapshots only. No new trajectory, control
execution, plant replacement, compensation, actuator bound or RL training was
performed. This is a mathematical audit, not physical drilling-plant validation.

## Frozen inputs and interpretations

Snapshots are P5 at indices 0 and 200000, P0 at index 1000, and P3 at index 1000.
Their binary64 error states, saved `U1_global` positions/velocities, full force
vectors and input/load channels are frozen, with original/repeat bitwise checks.
P5's first bit-load array entries are placeholders; the explicit nominal loads
are used. Its terminal command is computed but unused by the original integration.
No extra terminal transition is inferred.

Two arithmetic interpretations are kept separate:

1. Exact rational operations on archived binary64 full M/K/D, reconstructed
   binary64 T/S/C, and saved forces/states.
2. Independent structured-real FEM assembly from frozen primitive data at
   128/256-bit directed rounding, with unchanged IC, Vs, input/load values and
   the same saved snapshot vectors. Operator gravity and mappings are reassembled;
   their rounding differences from saved forces are reported separately.

The native mechanical projection is evaluated as `S M^-1 w`. This is not a
derivative measured from the archived trajectory. The stored reduced KM/DM have
their own original assembly rounding; their prior 11B reference residuals remain
separately recorded. No stored matrix or equation was replaced.

Authority ZIP, source-member and Chapter 5 identities are inherited and verified
against 11A/11B. M/K/D and rotations are independently reconstructed with the
existing verified FEM functions. Coordinate order is `(x,y,alpha,beta)` per node;
reduced coordinates are `(x1,alpha1,...,x5,alpha5)`. Translational components use
m, m/s and N; rotational components use rad, rad/s and N·m. The fixed tangent
relations generate `U=Tq`, `ST=I`, `CT=0`. The kernel is force-actuated with no
Dirichlet row replacement in the assembled matrices. This numerical convention
does not settle physical support, preload or contact assumptions.

The executed cross-product rotations remain orthogonal and right-handed to the
original tolerance. The printed Chapter 5 rotation-sign discrepancy is retained
in the convention ledger; no sign was repaired. `U1_global` is displacement plus
velocity, while `X1` adds the fixed geometry origin `UU1`. These are not silently
interchanged. The surface/bit mapping signs and the reconstructed full forces
pass the unchanged numerical tolerance.

## Static formulations

For the supplied force vector f, define `w=f-D T v-K T q`. The three ideal-reaction
expressions compared on identical operands are:

```
native acceleration:       a_N = M^-1 w;  qddot_N = S a_N
ideal reaction:           lambda = -(C M^-1 C')^-1 C a_N
                         R = C' lambda; a_H = M^-1 (w+R)
virtual-work reduction:  qddot_H = (T' M T)^-1 T' w
full block equations:    [M -C'; C 0] [a_H; lambda] = [w; 0]
```

The fixed RHS columns are KT (10), DT (10), surface inputs (2), signed bit loads
(2), and gravity (1), followed by the four snapshots and two reference states.
All 31 columns share the same factorization; operator tests are predefined and
not selected using snapshot outcomes. Basis coefficients are not physical input
excitations or actuator limits.

## Native compatibility and retained reaction effects

All five operator groups contain interval-certified nonzero components of
`C M^-1 w`. Each of the four snapshots also fails compatibility without additional
reactions. The 256-bit midpoint displays below separate translation and rotation;
exact ratios and interval endpoints are retained in the generated evidence.

This does **not** mean the reconstructed native coordinates leave the geometric
constraint set: `U=Tq` and `CT=0` enforce that geometry by construction, including
`C(T qddot_N)=0`. What fails is consistency between that embedded acceleration
and the supplied full FEM force balance. Its required extra generalized force
would be `R_needed=M T qddot_N-w`, with virtual work
`T' R_needed=M_H*(qddot_N-qddot_H)`. The nonzero retained differences show why
that force cannot generally be only an ideal workless normal reaction while
preserving the native acceleration. No such force was applied or added to a plant.

Rows of C are unnormalized constraint functions, so their residuals are not
unit-normal contact accelerations. Normalizing the rows changes magnitudes,
not the zero/nonzero conclusion. Retained translational accelerations below are
global q-coordinate components, not measured bit accelerations.

| Snapshot | max translational compatibility error (m/s²) | max rotational compatibility error (rad/s²) | max retained translational reaction effect (m/s²) | max retained rotational reaction effect (rad/s²) |
|---|---:|---:|---:|---:|
| P5, 0 s | 3.885487997 | 0.0002061453 | 0.408512575 | 0.001307879 |
| P5, 20 s | 3.885488574 | 0.005592283 | 0.408512585 | 0.077866717 |
| P0, 0.1 s | 3.947793745 | 3.429781369 | 0.409507383 | 40.717323984 |
| P3, 0.1 s | 3.883081584 | 6.979665484 | 0.408013449 | 82.850352727 |

These are static accelerations evaluated at identical recorded states and forces,
not trajectory differences, measured plant responses or safety outcomes.
The unreacted projection cannot be reconciled with this ideal-reaction formulation
while also retaining its original accelerations: the required ideal reactions
have nonzero retained effects. Multipliers are conjugate to the unnormalized
constraint rows `(-tan(theta),1)`; they are not measured contact-force magnitudes.

## Ideal-reaction consistency

Virtual work, reaction-Schur and full block equations agree exactly for all 31
archived-dyadic RHS columns. The checker independently verifies the equations and
a nonzero full-block determinant using fraction-free integer elimination. Thus
the agreement is not an artifact of a shared floating-point solution.

For the declared structured-real interpretation, validated solves and independent
fraction/dyadic-rounding checks support the same identities. Worst 128-bit
enclosure magnitudes across all columns include:

- Constraint acceleration residual: 1.025e-23 in normalized component units.
- Reaction virtual-work residual: 5.484e-18 in normalized generalized-force units.
- Virtual-work versus reaction acceleration: 5.050e-23.
- Full block versus reaction acceleration: 1.091e-22.

At 256 bits the corresponding bounds are approximately 2.873e-62, 1.537e-56,
1.415e-61 and 2.945e-61. Exact algebraic identities justify zero; enclosing zero
alone would not prove it. All reconstruction/identity checks satisfy the original
1e-10 and 1e-8 scaled tolerances; none was changed.

This supports a mathematically consistent model under fixed bilateral, workless
normal-reaction assumptions. It does not show that drilling contact is bilateral,
workless, or correctly represented by these reactions. Prescribed friction may
already incorporate contact assumptions; adding normal reactions to a physical
model without resolving that issue could double-count or omit effects.

## Rank, conditioning and modes

| Matrix | Binary64 SVD numerical rank | condition estimate in fixed SI normalization |
|---|---:|---:|
| Full M | 20/20 | 2.77128e6 |
| Full K | 18/20 | 8.05375e16 |
| M_H = T' M T | 10/10 | 39.0919 |
| K_H = T' K T | 10/10 | 1.33368e9 |
| Recomputed native stiffness acceleration operator | 9/10 | 3.41955e16 |

The last operator differs by original solve rounding from saved KM, whose 11B
condition estimate was 4.03124e17. Near-singular condition estimates are sensitive
to such differences. Numerical rank uses the frozen `max(shape)*eps*sigma_max`
threshold, not a claim about exact physical modes.

Exact determinants show the literal archived dyadic M, K, M_H and K_H all have
full rank. In particular, dyadic full-K rank 20 must not be confused with its
numerical rank 18: assembly rounding perturbs structural zero modes. In the
structured-real assembly, the two common global translation columns have zero
stiffness by the element-difference identity and consistent interval enclosures.
Their constraint images have rank two, so no nonzero combination is compatible
with the frozen nodal tangent constraints.

Both independent outward-rounded LDL checks certify positive M, M_H and K_H for
the symmetric structured-real assembly. The smallest K_H LDL pivot lower bound
is approximately 0.122297; this is not a minimum-eigenvalue estimate. No SPD claim
is made by silently symmetrizing the slightly nonsymmetric archived dyadics.
Positive K_H excludes a nonzero stiffness-null velocity within that conditional
ideal model. The native numerical near-null vector has positive ideal stiffness
quadratic form 0.0202948 in the fixed normalization; it is not an established
rigid or cyclic physical mode.

The existing axial ×1000 diagnostic scaling changes condition estimates:
M_H to 3.90919e7 and K_H to 1.05850e12. This demonstrates unit/coordinate
sensitivity, not a new plant or a conditioning repair. The native-versus-ideal
stiffness comparison remains approximately **14.073721% in relative Frobenius
norm** in the original SI coordinates. It is a numerical operator comparison,
not physical modeling error or evidence that the ideal alternative is correct.

## Reference solvability

IC, Vs and nominal inputs/loads were not changed. Both native `K_M Vs` and ideal
`K_H Vs` are nonzero, with strict interval witnesses. The fixed affine reference
is therefore not an exact constant-input solution under either tested reduction.
At the exact affine reference states, the maximum component magnitudes are:

| Reference time | Native translation (m/s²) | Native rotation (rad/s²) | Ideal translation (m/s²) | Ideal rotation (rad/s²) |
|---|---:|---:|---:|---:|
| 0 s | 3.375e-11 | 2.365542e-6 | 0.408512575 | 0.001307879 |
| 20 s | 2.745386e-7 | 0.152400579 | 0.408512548 | 0.155054546 |

The native dominant signed component at 20 s is approximately
**-0.152400579251 rad/s²**, consistent with 11B. These are matched-state equation
discrepancies, not integrated deviations. The larger ideal reference residual at
time zero shows that switching reduction alone does not establish the nominal
reference. It is not a prediction of physical controller instability.

Changing IC cannot remove a nonzero time coefficient. If a true cyclic or moving
reference coordinate was intended, its kinematics and work-conjugate loading
must be derived; no nullspace projection, compensation or new reference was used.
P5 remains a zero-error equilibrium of the original implemented error map only.

## Hypotheses and limits

| Hypothesis | Result |
|---|---|
| Supplied forces without additional reactions enforce the declared constraints | Rejected for the declared fixed numerical model; all operator groups and four snapshots supply counterexamples. |
| Ideal normal reactions have no retained-acceleration effect | Rejected for the declared model. |
| Ideal-reaction, virtual-work and full-block formulations are internally consistent | Supported conditionally by exact identities, uniqueness and interval verification. |
| Archived affine reference is exact under the native reduction | Rejected for the declared model. |
| Archived affine reference is exact under the ideal reduction | Rejected for the declared conditional model. |
| Ideal bilateral reactions represent actual drilling contact | Unresolved physical assumptions. |
| The actual physical drilling plant is invalid | Not established; unresolved. |

The coefficient audit tests a fixed linear operator structure; four archived
snapshots demonstrate specific incompatibilities. They do not validate a contact
law, identify the correct physical reduction, establish global physical validity,
or assess full-horizon safety and performance of an alternative controller.

## Verification, preservation and cost

During the reviewed pilot, all 35 applicable drilling mathematical/provenance tests passed, including ten
new E11C fixtures. Exact, 128-bit and 256-bit calculations each repeated identically
on frozen inputs. The independent checker imported no producer math or MPFR:
it verified 31 exact RHS columns, a nonzero block determinant, 3,100 interval scalar
residual checks, 950 cross-precision overlaps and independent LDL certificates.
Additional exact determinant checks independently confirm the reported full
archived ranks. Original/repeat archived snapshot records match bitwise.

One worker used 43.823 s total wall time. Main audit CPU was 18.698 s before its
final independent/preservation steps; the hard 1,800 CPU-second limit remained
active. Peak recorded main-worker RSS was 1,222,115,328 bytes, below 2 GiB.
No timeout, failed technical check, threshold change or unresolved solve occurred.
Physical assumptions remain unresolved as scientific findings.

All prior tracked research, 53,435 historical files, 87 authority files, accepted
11A/11B archives and the nine unchanged IQL files passed preservation checks.
Generated snapshots/certificates are at `results/drilling_constraints/pilot/`,
outside Git. They are diagnostic records, not expert-training data. Existing
research documentation was not rewritten at pilot completion, and no commit or
push was performed during that phase.

## Recommendation for review

An internally consistent ideal-reaction reduction is mathematically supported;
a physically justified drilling reduction is **not yet established**. Physical
drilling-model development should remain blocked. Obtain clarification of:

- Whether the supplied loads already include normal reactions, and the contact,
  clearance/unilateral and friction laws, including the role of string tension.
- The rotational constraint and work-conjugate torque definitions, material versus
  spatial mesh, reference-frame motion, stress-free geometry/preload and supports.
- The intended cyclic modes/reference and gauge, the nominal-force construction,
  and the actual higher-fidelity plant used for the dissertation comparison.
- Physically admissible absolute surface force and torque ranges, the meaning of
  controller-correction channels, and any documented slew/rate restrictions.
  Nominal inputs and observed control peaks do not supply these limits.

In particular, the physical derivation of the `0.973839772515` force-construction
factor in `IC_check.m` and the simultaneous use of an ordinary stiffness inverse
and a nonzero stiffness-null reference in Chapter 5 require clarification. No
new IC or reference was computed to resolve those issues.

Engineering actuator specifications remain a separate unresolved contract.
Retain the original MATLAB implementation as an explicitly ideal-input numerical
error-map benchmark, with safety claims limited to that implemented model. It is
not presently a physically bounded safe-RL benchmark. This pilot authorizes no
physical-model replacement, controller redesign or RL experiment.
Further physical drilling-model development and numerical investigation are paused
pending external model/engineering clarification. Do not begin another numerical
drilling investigation without new engineering evidence and a reviewed design.

Read-only reproduction uses `requirements_experiment_11c.txt`, the unchanged
external authority and accepted 11A–11C records, and a new output file:

```
python experiments/11c_verify_constraints.py --output /private/tmp/e11c-unused-check.json
```

The entry point reproduces only the frozen mathematical expressions and data,
enforces one numerical worker/thread and the original resource limits, and writes
outside accepted archives. Legacy pilot/checker/report writers are preserved and
run only in disposable copies during finalization. Existing outputs are never
overwritten. See the [reproduction guide](experiment_11c_reproduction.md) for
clean-export inputs and exact historical context verification. No MATLAB process,
simulator, new model, state/action search or trajectory is executed.

## Finalization verification

All **122 applicable regression tests passed in both workspace and clean export**.
The export excludes all nine unfinished IQL files and imports its own code.
Both environments reproduced the exact and 128/256-bit results, accepted repeat
hashes, all four archived original/repeat snapshot pairs, five operator groups,
reference/rank diagnostics and hypothesis classifications. Independent checks
again passed all 31 exact RHS columns, 3,100 interval residuals and 950
cross-precision overlaps. The exact missing-force virtual-work identity was also
checked independently of coordinate enforcement; no force was applied.

The accepted completion, numerical and clarified-review seals remain unchanged.
Original reports, inputs, source snapshots and negative/unresolved findings are
preserved. Only the authorized current-status, research-context and decision-log
documents receive historical-hash exceptions. All 53,435 historical files, 87
authority files and nine IQL files passed preservation checks. Verification wrote
zero accepted-archive files, launched zero MATLAB processes and executed zero
new research/control trajectories. Additive finalization records and accepted
source copies are separate under `results/drilling_constraints/finalization/`.

Finalization records the pause on further physical development and new numerical
drilling investigations pending external clarification. It does not authorize
another model, reduction, compensation, controller, actuator contract or RL study.
