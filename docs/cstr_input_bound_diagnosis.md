# CSTR strict-input numerical diagnosis — zero-solve v1

**Recommendation: investigate one separately versioned numerical controller,
subject to new approval; do not retain the stock controller as qualified under
the strict raw-input contract.** The accepted pilot remains FAILED and halted.
No solve, integration, trajectory or optimizer-history probe was run here.

## Verified pathway and numerical evidence

The pinned do-mpc source remains v5.1.1,
`90e0d7ada888a4cc75ea3769214de5105764c640`. All 269 upstream files, 40 package
identities and the accepted source-audit/pilot seals were verified. The failed
pilot completion-seal SHA-256 remains
`98b877e134cd024492e5fe7987f29bba18b69677ccc960aad1a3d6d3f38b8819`.

The native template declares F in [5,100] h^-1 and its scale as 100. The optimizer
bound setters divide by that scale, giving [0.05,1] for z_F. MPC `_update_bounds`
applies this as a variable bound at every input stage. It is not an additional
nonlinear inequality in g: the 324 g rows are equalities. The first F is
zero-based index 336, label `[_u,0,0,F,0]`. All archived bound arrays match both
the original native graph and the independent reconstruction exactly.
See the pinned [template](https://github.com/do-mpc/do-mpc/blob/90e0d7ada888a4cc75ea3769214de5105764c640/examples/CSTR_approximate_mpc/template_mpc.py),
[bound setters](https://github.com/do-mpc/do-mpc/blob/90e0d7ada888a4cc75ea3769214de5105764c640/do_mpc/optimizer.py), and
[MPC extraction](https://github.com/do-mpc/do-mpc/blob/90e0d7ada888a4cc75ea3769214de5105764c640/do_mpc/controller/_mpc.py).

| Quantity | Reconstructed value |
|---|---:|
| Saved z_F | 0.04999999450092826 |
| Original scaled lower margin | -5.499071739634154e-9 |
| Returned F = binary64(100 z_F) | 4.999999450092826 h^-1 |
| Physical lower margin | -5.499071740189265e-7 h^-1 |
| Returned Q | -0.013781197157147958 kJ/h |
| Absolute rounding error in 100 z_F | 3.330669073875470e-16 h^-1 |

Native extraction multiplies the first optimized input by its original scale,
without input clipping. An inert-backend reconstruction of that exact native
structured readout agrees bytewise with the saved independent readout. The
candidate is already below its scaled bound before multiplication. Rescaling
roundoff is over nine orders of magnitude smaller than the physical violation;
there is no demonstrated extraction, ordering, scaling-implementation or unit
conversion defect. The archived runner checked equality to this readout before
rejecting it; no separate returned-command packet was saved.

## Inherited IPOPT behavior

The sealed native options change display settings only. IPOPT 3.14.11 defaults
include `bound_relax_factor=1e-8` and `honor_original_bounds=no`. Its relaxation is

`delta = min(constr_viol_tol, bound_relax_factor * max(abs(bound),1))`.

For 0.05 this gives delta=1e-8, relaxed lower bound 0.04999999, or 4.999999 h^-1.
C2 is inside this relaxed bound by 4.500928262041182e-9 in NLP coordinates and
uses 54.9907% of the relaxation. IPOPT's reported constraint violation excludes
violations of original, unrelaxed variable bounds. These are source-supported
properties, not a reinterpretation of the raw-input requirement.
[IPOPT 3.14.11 bound handling](https://github.com/coin-or/Ipopt/blob/releases/3.14.11/src/Algorithm/IpOrigIpoptNLP.cpp).

The retained lower multiplier is 0.5567526082070251 and final mu is
2.5059035596800618e-9. Their product with the default-relaxed slack is
2.5059035492441402e-9, differing from mu by -1.044e-17. The central-path-implied
lower bound, `z_F - mu/(-lam_x)`, is 0.04999998999999998: within 2.082e-17 of
the default-relaxed bound. This is strong additional evidence for the relaxation
mechanism. It is an inferred bound, not recovered internal-bound telemetry.

Other inherited defaults are tol=1e-8, constr_viol_tol=1e-4, dual_inf_tol=1 and
compl_inf_tol=1e-4. Successful stopping requires the combined scaled NLP error
and the absolute criteria; these individual limits cannot be interpreted as
independent physical guarantees. `acceptable_tol=1e-6` is a separate stopping
path; the recorded status was `Solve_Succeeded`, not acceptable-level termination.
Here, “unscaled” IPOPT coordinates are already do-mpc-scaled variables, not
physical F units. See the versioned [convergence check](https://github.com/coin-or/Ipopt/blob/releases/3.14.11/src/Algorithm/IpOptErrorConvCheck.cpp)
and [relative tolerance](https://github.com/coin-or/Ipopt/blob/releases/3.14.11/src/Algorithm/IpIpoptData.cpp).

| Saved/reconstructed residual | Value |
|---|---:|
| Maximum original variable-bound violation, NLP coordinates | 5.499071740e-9 |
| Maximum original equality violation / final native inf_pr | 5.952789390e-9 |
| Final native inf_du | 9.464925012e-10 |
| Pilot width-normalized primal residual | 5.788496564e-9 |
| Pilot scaled stationarity residual | 4.201719048e-10 |
| Pilot scaled complementarity residual | 3.156081623e-9 |
| Pilot dual-sign residual | 0 |

Only the first F violates an original variable bound in this saved candidate.
The normalized F violation is `(0.05-z_F)/0.95`, explaining the pilot primal
figure; this normalization differs from native IPOPT diagnostics. Independent
constraint reconstruction agrees to 1.334e-15. Both native and independent
objective readouts agree within 1e-16 near 0.1572819024327726.

CasADi 3.7.2 defaults to no post-solve bound-consistency projection, and the
native code does not request one. See its [NLP wrapper](https://github.com/casadi/casadi/blob/3.7.2/casadi/core/nlpsol.cpp)
and [IPOPT interface](https://github.com/casadi/casadi/blob/3.7.2/casadi/interfaces/ipopt/ipopt_interface.cpp).
Versioned sources were downloaded and SHA-256 recorded outside Git. There is no
saved effective-option dump or internal relaxed-bound vector. No ipopt.opt is
present in the current working directory; its historical absence was not separately
recorded. Thus source defaults plus the numerical central-path match strongly
explain the result, without claiming complete recovery of solver internals or
an intervention-based causal test.

## Nonapplication and completed-case limitations

C2 still has one initial state, zero applied commands and no integration records.
Post-query memory and overall action-selection timing remain missing. The stale
progress counter was not repaired. C3 and all 16 initialization diagnostics remain
unrun, leaving the six-input supervision contract unresolved.

C0/C1 dense minimum reactor-temperature margins remain 0.00848337485 and
0.00953589809 °C. Maximum native-versus-independent T_R endpoint differences
were only 3.26434e-8 and 3.29710e-8 °C. These comparisons support nominal numerical
agreement, not error bounds for the physical reactor. Monitoring resamples an
independent dense solution; unrecorded CVODES internal states, continuous-time
safety, parameter uncertainty, disturbances, sensors and actuator dynamics were
not validated. The small positive margins provide little demonstrated physical
robustness. Native initial/terminal prediction-bound limitations remain unchanged.

## Prospective options and scientific decision

The stock controller cannot be considered compliant with the strict contract:
the existing C2 counterexample suffices, even though C0/C1 passed. No conclusion
of mathematical infeasibility follows.

1. **Preferred option for separate review:** a numerical-only version disabling
   IPOPT bound relaxation, with prospectively specified feasibility/optimality
   tolerances and the unchanged strict post-return rejection. Tighter tol alone
   leaves relaxed bounds; tighter constr_viol_tol can reduce relaxation but does
   not ensure exact physical admissibility. Disabling relaxation removes the
   identified mechanism, not every finite-precision failure mode. No success is
   promised.
2. **Alternative:** explicitly request final original-bound projection. This changes
   returned-candidate handling; objective, equality/KKT residuals and controller
   memory consistency would require fresh verification after projection.
3. **Alternative:** prospectively inset optimizer input bounds using a declared
   numerical error budget while keeping the original physical limits. This changes
   the feasible optimization set and must be labeled accordingly. Choosing an
   inset merely above the observed shortfall would be unjustified.

None was implemented or tested. Any revision must retain the failed stock result,
repair failure-record completeness prospectively, freeze a single configuration
before outcomes and obtain fresh execution/resource approval. Do not resume the
halted pilot or cycle through settings until a passing result appears.

One bounded revision is more defensible than immediately switching tasks: this
failure has a specific numerical explanation, while completed solver calls were
cheap (about 15 ms median). Switching introduces new provenance, context and expert
validation costs. Nevertheless another controller-validation cycle only addresses
a prerequisite; it does not test active acquisition. Reliable expert execution,
resolved supervision context, a fixed learner's reducible gap and independent
closed-loop evaluation would still be required. Pause CSTR rather than prolong
validation if those prerequisites remain unavailable after a separately approved
bounded reassessment. No active/random study is currently justified.

## Tests, preservation and evidence

All 25 applicable static/archive tests passed on the final run. A new arithmetic
test initially compared a binary64 multiplication with a differently rounded
decimal literal; its expected result was corrected using exact dyadic arithmetic.
The failed test log remains saved. No historical acceptance threshold changed.

Checks preserved all 6,078 sealed pilot artifacts, 280 prior tracked and 12 prior
untracked files, accepted CSTR source/runtime, earlier research/evidence,
87 authoritative drilling files and all nine IQL files. Additional B0/B1 seals
passed. No tracked/staged changes, commit or push occurred. Diagnostic source,
tests and this report are new, untracked files; generated evidence is ignored.

Evidence: `results/cstr_numerical_diagnosis/zero_solve_v1/diagnosis.json`,
`independent_checks_v2.json`, both test logs, versioned external source identities,
and the separate resource ledger. Optimizer/integrator factories were blocked
or replaced with inert stubs throughout. All historical ledgers remain unchanged.

Diagnostic charges, including conservative inspection/report allowances: **33.664 CPU / 86.040 active-wall seconds**, zero solves/integrations. Within the existing original CSTR pilot time ceiling, the combined charges are 92.790 CPU / 200.982 active-wall seconds; the remaining ceiling balance is 507.210 CPU / 999.018 active-wall seconds. The accepted failed-pilot ledger itself is unchanged. This accounting does not authorize new scientific execution.
