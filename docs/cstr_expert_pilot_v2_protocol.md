# CSTR expert technical pilot v2: no bound relaxation

This prospective version changes exactly one native controller option:
`ipopt.bound_relax_factor=0`. All other original CSTR controller/model/task,
scaled objective, bounds, 20-stage degree-2 Radau graph, 0.005-hour sample,
display settings and warm-start semantics remain unchanged. No clipping,
projection, feasibility-tolerance change or tightened physical bound is allowed.
The original pilot remains FAILED; its records and missing telemetry are immutable.

Exact original C0–C3 states, ordering and previous command [5,0] are loaded
byte-for-byte from the accepted audit. Two fresh native executions per case,
100 controls each, with native recorded-command replay and independent DOP853
replay after each. All eight original k0/k1 contexts and their two original
optimizer initializations follow only after every scientific/replay job passes.
First failed mandatory criterion stops all dispatch. No retry or replacement.

Reuse immutable v1 accounting, integration, independent dynamics/NLP, KKT,
costs and replay code through explicitly scoped v2 orchestration bindings.
The source template is invoked unchanged; its setup call is wrapped only to
add the one allowed option before native backend creation. Native source files
are not edited. Static original/revised graphs, derivatives and bounds must
agree bitwise and match the independent reconstruction within the original
1e-10 graph tolerance. All numerical/raw-input/state/replay acceptance values
are exactly those in the original protocol/configuration.

Effective option verification uses actual IPOPT OptionsList getters on the live
constructed CasADi backend memory, not a configured-dictionary assertion. A small
read-only C++ helper uses exact CasADi 3.7.2 source layout headers, pinned runtime
headers/libraries and the recorded compiler/architecture. It checks out/releases
backend memory without evaluating it. Preflight must observe stock 1e-8 versus
revised zero, identical other tolerances and unchanged serialized backend/controller
memory. During every scientific/diagnostic backend call, actual option observations
are persisted and must equal the frozen expected values before evaluation. This
inspection is valid only for the verified pinned ABI. Any uncertainty stops preflight.

Candidate submissions/results, multipliers, bound residuals, native status and
timing remain recorded. Returned command bytes, both raw input margins, timing
and post-query memory are written before raw-command rejection. A native exception
records available post-attempt memory with native-query-returned=false; absent
commands/candidates are never invented. Raw lower and upper state/input margins
are retained, including partial trajectories and dense replay samples.

Native CVODES observations are control-boundary outputs, not unobserved internal
steps. Independent replay uses original 1e-11 DOP853 tolerances, 10/20 subinterval
dense grids and range-normalized endpoint agreement <=1e-7. Require finite
samples, nonnegative physical margins and normalized state reserve >2e-7.
Refinement resamples one numerical dense solution; no formal continuous-time or
physical reactor safety claim follows. Numerical solver acceptance remains native
Solve_Succeeded/success, primal <=1e-7, stationarity/complementarity/dual <=1e-6,
objective/constraint reconstruction <=1e-10. Strict raw commands allow no epsilon.

Deterministic repetitions compare state/command/candidate/submission/memory and
solver-status/iteration bytes; timings are excluded. Native replay is bitwise;
independent replay uses the original error criteria. Native-history diagnostic
reconstruction must reproduce the saved candidate bitwise. Cold-initialization
action differences are reported without a newly invented action-equality gate.
These finite tests cannot establish global six-input sufficiency.

New separate allocation: 816 solves, 2400 simulated control intervals, 24000
charged monitoring subintervals, 600 incremental CPU/1200 active-wall seconds,
one worker/4 GiB RAM. Every job is reserved before dispatch; failures retain
their reservations. Actual and reserved counters remain separate. Preflight,
inspection, reporting and preservation count in this allocation. All historical
CSTR, Pendulum, drilling and E12 resource ledgers remain unchanged. No learner,
acquisition, alternative-action probe, RL or IQL work; no commit/push.
