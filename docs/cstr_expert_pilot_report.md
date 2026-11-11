# CSTR expert-only technical pilot v1

**Outcome: STOPPED at the first mandatory acceptance failure. The four-case
expert-feasibility gate did not pass.** No retry, fallback, clipping, altered
tolerance, replacement case, learner training or acquisition experiment followed.

## Frozen protocol and preflight

Source: do-mpc v5.1.1, commit
`90e0d7ada888a4cc75ea3769214de5105764c640`, with the accepted CPython 3.10.19
environment (40 package identities; CasADi 3.7.2, IPOPT 3.14.11, NumPy 2.2.6,
SciPy 1.15.3). All 269 upstream file hashes, dependency RECORD/native-library
identities and accepted source-audit evidence were rechecked. Exact manifests
are retained in the source audit and pilot identity records.

The mandatory zero-solve preflight passed 20 tests. Independent reconstruction
of the 376-variable, 324-equality Radau NLP agreed with native objective,
constraints, gradients and multiplier products to a maximum absolute difference
of 1.336e-15. Full bound indexing, original solver options, fresh-object reset
independence, within-episode native warm starts, current-command telemetry,
failure stopping and budget dispatch were tested without real operations.

Sampling is 0.005 h (18 s), horizon 20, Radau degree 2, one element; each
scientific episode is 100 transitions (0.5 h). State order is
`[C_a,C_b,T_R,T_K]`, concentrations mol/L and temperatures Celsius. Inputs are
`[F,Q_dot]`, dilution rate h^-1 and cooling heat rate kJ/h; previous input is
`[5,0]`. Original concentration tracking and scaled movement penalties are
unchanged. State bounds apply at prediction indices 1–19 and their stored
collocation slots; the initial state has an equality, and the terminal prediction
state has no independent box bound. All 20 input stages are bounded.

The protocol, exact binary64 cases, numerical contract and seven implementation
files were sealed before dispatch. Protocol seal SHA-256:
`e7e388a314b320108d9accab147f0e1393d12ce4e522fd871132a1f1045d435d`.
See [protocol](cstr_expert_pilot_protocol.md) and
[configuration](../configs/cstr_expert_pilot_v1.json).

## Complete denominators and first failure

| Case | Initial state | Scientific runs complete | Native/independent replays complete | Outcome |
|---|---|---:|---:|---|
| C0 | (0.8,0.5,134.14,130) | 2/2 | 4/4 | All prescribed checks passed |
| C1 | (0.6,0.4,130,125) | 2/2 | 4/4 | All prescribed checks passed |
| C2 | (1,0.7,136,135) | 0/2 | 0/4 | First query rejected before application |
| C3 | (0.8,0.5,138,134) | 0/2 | 0/4 | Not dispatched after mandatory stop |

Thus 4/8 scientific executions and 8/16 replay jobs completed. C2's original
run contains only its initial state and no applied commands. All 16 optimizer
initialization diagnostics (eight contexts, two initializations) remain unrun.
The proposed six-input learner's numerical-history sufficiency is unresolved.

C2's first returned command was `[4.999999450092826,-0.013781197157147958]`.
Its dilution-rate lower margin is **-5.499071740189265e-7 h^-1**. Exact binary64
representations are `0x1.3ffffdb18aaacp+2` and `-0x1.c3950f8b7a971p-7`.
The strict physical requested-input check rejected this negative margin before
the simulator was called. There is no observed C2 reactor-state violation.

The candidate reported `Solve_Succeeded`, 26 iterations, scaled primal residual
5.78850e-9, stationarity 4.20172e-10, complementarity 3.15608e-9 and zero dual-sign
residual. Its reconstructed objective was 0.1572819024327726. These meet the
separately frozen numerical tolerances; they do not override exact nonnegative
requested-input margins. The evidence demonstrates a small bound-violating
numerical command, not mathematical infeasibility. No solver-bound-relaxation
causal explanation was established and no settings were changed.

## Performance of completed original cases only

Return is the negative sum of 100 concentration-tracking stages, their scaled
input-movement penalties, and one terminal concentration cost. Repetitions are
reproducibility checks, not additional independent cases.

| Metric | C0 | C1 |
|---|---:|---:|
| Total cost | 0.028003696583 | 0.129603688203 |
| Return | -0.028003696583 | -0.129603688203 |
| Tracking cost sum | 0.027901888508 | 0.128582834642 |
| Movement cost sum | 0.000094006124 | 0.001013051501 |
| Terminal cost | 0.000007801951 | 0.000007802061 |
| C_a/C_b RMSE (mol/L) | 0.010396 / 0.013075 | 0.012543 / 0.033593 |
| Feed volume (L) | 58.546252 | 55.414538 |
| Removed heat (kJ) | 350.680903 | 280.308357 |
| Sum F² ((h^-1)²) | 13820.022108 | 12577.271873 |
| Sum Q² ((kJ/h)²) | 64264271.238474 | 50357968.455318 |

Conditional on these two completed original cases, mean return is
**-0.078803692393**, worst return **-0.129603688203**. These are not full
four-case performance estimates. No full-horizon return is assigned to C2/C3.

Minimum independently reconstructed dense raw state margins in state order:

| Case | C_a (mol/L) | C_b (mol/L) | T_R (°C) | T_K (°C) |
|---|---:|---:|---:|---:|
| C0 | 0.6015387552 | 0.4000000000 | 0.00848337485 | 1.10860231791 |
| C1 | 0.5000000000 | 0.3000000000 | 0.00953589809 | 1.11178675646 |

Native-boundary minimum T_R margins were 0.00850379471 and 0.00954262065 °C,
respectively. All completed scientific runs and their reconstructions had zero
observed state/applied-input violations. Minimum applied-input margins `[F,Q]`
were `[0.8818172014,0.0857629176]` and `[1.6403875774,0.00779247075]`, with each
component in its own units. Dense normalized reserves after subtracting the
frozen 2e-7 allowance were 9.40597e-5 and 1.05754e-4. These small reactor-temperature
margins warrant retaining the dense diagnostics; no continuous-time certificate
or box-wide viability follows.

## Numerical verification and repeatability

All **401/401** saved candidates passed independently reconstructed objective,
constraints and scaled KKT checks. Across them, maximum primal, stationarity and
complementarity residuals were 5.78850e-9, 7.56890e-9 and 8.40419e-9; maximum
dual-sign residual was zero. Objective and constraint reconstruction errors were
at most 8.32667e-17 and 1.33511e-15. Measured-state parameter agreement was exact;
the candidate's initial equality had physical residual at most 1.42109e-14.

C0/C1 deterministic repetitions agreed bitwise for states, commands, candidate
and submitted vectors, pre/post optimizer memory, and solver status/iterations.
All four native recorded-command replays agreed bitwise. All four independent
equation/DOP853 replays passed; the largest component-range-normalized endpoint
discrepancy was **1.60230e-9**, versus the frozen 1e-7 limit.

Each complete native episode supplies 101 control-boundary states. Independent
replays evaluate 10-subinterval and 20-subinterval dense-output grids per interval;
shared grid samples agree. These are independently reconstructed numerical
states, not unrecorded native CVODES internal states. Refinement resamples the
same independent solution, not a separate integration-convergence proof.

## Failure-record limitations

The rejected C2 query retained the complete available solver candidate, exact
backend submission, multipliers, bounds, native status/iterations, independent
residuals, solver-only timing and prequery memory. It did **not** retain postquery
controller memory or overall native action-selection timing: raw-input rejection
preceded those writes. No separate returned-command packet was persisted;
bytewise equality to the saved candidate readout was checked before rejection.
The progress counter remains zero although one independently checked candidate
is present. These records were not repaired or retrospectively reconstructed.
Failure-telemetry completeness is therefore limited despite the passed preflight.

## Latency, resources, preservation and decision

Solver-only wall latency (401 calls): median 14.867 ms, p95 16.064 ms, p99
19.388 ms, maximum 25.368 ms. Native action-selection time including verification
(400 retained observations): median 19.668 ms, p95 22.266 ms, p99 28.471 ms,
maximum 35.771 ms; one rejected-call observation is missing. No measured call
exceeded the **18-second** CSTR sampling interval. This is offline evidence, not
a hard real-time guarantee. Iterations ranged 19–26 (median 25). The 800 recorded
native integration calls totaled 0.351873 s; independent integration and file/
verification overhead are included in aggregate accounting.

Actual work: **401 solves, 1,200 simulated control intervals, 12,000 charged
monitoring subinterval checks**. Reservations remain **500 solves, 1,300
intervals, 12,000 checks**, including unused failed-job reservations. CPU/wall
charges and remaining cap are in the separate pilot resource ledger; previous
CSTR-audit and E12 allowances were not reset or debited. All counters remain
halted after the first failure.

Read-only verification rechecked 87 authoritative drilling files, nine unchanged
IQL files, 87,315 accepted MPC80 files, 59,268 prior result/data files and 12,147
Pilot B artifacts. Additional B0/B1 seals, current tracked files, accepted CSTR
audit/runtime and the seven frozen pilot implementation hashes are checked.
Applicable zero-solve regression tests were repeated; original evidence remains
unchanged. Generated outputs stay outside Git. No commit or push was performed.

**Decision:** scoped feasibility remains unestablished. Preserve this stopped
pilot for scientific review. Neither the tiny requested-input violation nor the
two successful cases establishes infeasibility of the CSTR task, formal reactor
safety, six-input learner sufficiency, or an acquisition advantage. Any proposal
to address numerical bound handling or failure telemetry requires separate review;
this report does not authorize resuming the halted execution.

Evidence: `results/cstr_expert_pilot/technical_v1/`, especially `first_failure.json`,
`independent_review.json`, `final_zero_checks.json`, `protocol_seal.json`,
`completed_records.json`, `resource_ledger.json` and the per-attempt records.
Read-only reproduction (no solve/integration; refuses existing output):

```sh
/private/tmp/cstr_audit_venv_90e0d7a/bin/python experiments/cstr_verify_pilot.py \
  --output /private/tmp/cstr_read_only_review.json --full-preservation
```

Use the accounting wrapper for any additional authorized verification work.

Final separate pilot charges, including conservative report/integrity allowances:
**59.126 CPU seconds / 114.942 active-wall seconds**. Remaining time allowance: 540.874 CPU / 1085.058 active-wall seconds. The execution remains halted; this balance does not authorize additional work.
