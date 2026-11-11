# CSTR expert-only technical pilot v1

Approved scope is four frozen cases, expert-only, conditional on a mandatory
zero-solve preflight. No learner, dataset acquisition comparison or RL/IQL.
The accepted source audit and all prior outcomes remain immutable.

Source: do-mpc v5.1.1, `90e0d7ada888a4cc75ea3769214de5105764c640`, exact audit
runtime/dependency RECORD/native hashes. Use only the three native templates;
never execute upstream main.py or stock sampling/training. Preserve .005h sampling,
N20 Radau2/one element, original objective and scaled movement penalties,
native display-only solver options and CVODES1e−10 tolerances. F is h^-1,
Q is kJ/h and temperatures are Celsius with Arrhenius Kelvin conversion.

Cases C0–C3 are byte-identical to the accepted prospective audit manifest.
Each has previous-command memory (5,0),100 transitions, original/repeat,
fresh native MPC/simulator per independent execution, native warm starts within
episodes. Source state bounds cover prediction indices1–19/all stored collocation
slots; initial state is equality-constrained and terminal state independently
unbounded. All20 input stages are bounded. Raw simulation checks include the
actual initial/terminal states and never assume the box is a viable region.

Dispatch each case: original scientific/native replay/independent replay;
then repeat scientific/native replay/independent replay. Stop at the first failed
mandatory check; no retries, replacement cases, fallback or tuning. Unexecuted
cases remain unexecuted. Deterministic source states, commands, parameters,
candidates and relevant memory must agree bitwise; timings are excluded.

Independent NLP reconstruction uses the known376-variable layout,324 equalities,
analytic degree2 Radau coefficients, independent CSTR balances and original
scaled objective. Preflight compares native and independent f,g,gradient and
J-transpose-multiplier readouts at fixed fixtures and all bounds. Acceptance:
native success and Solve_Succeeded, finite candidate, primal<=1e−7,
scaled stationarity/complementarity/dual signs<=1e−6; graph/objective reconstruction
<=1e−10. Normalize coordinates by physical box widths; transform multipliers
contragrediently. Stationarity denominator is1+max(norms of the three gradient
contributions). Reuse canonical corrected two-sided KKT logic; equality duals
are free. This does not change the original solver settings.

Record exact backend kwargs/result, all available z,p,lambda_x,lambda_g,bounds,
status/iterations/latency, pre/post memory and partial trajectories. Rejected
candidates are persisted before rejection. An exception with no backend return
has candidate_available=False; do not substitute previous values. Current-step
command telemetry is copied from the integrator submission parameter, not a
stale previous-command cache. These are software commands, not physical sensors.

Native CVODES observations are101 control-boundary states per complete run.
No internal accepted-step state access is claimed. Independent DOP853 uses
recorded ZOH commands,1e−11 tolerances,10-subinterval primary and20-subinterval
dense-output refinement grids; these are independent reconstructed samples.
Shared-grid values and native endpoints must agree within1e−7 of component
range; every primary/refined raw margin must be nonnegative, with normalized
state reserve strictly above2e−7. Requested inputs require exact nonnegative raw
margins; no epsilon repair or clipping. Monitoring refinement resamples the same
independent dense solution, not an additional feedback trajectory or a proof
of integration-error convergence. No formal continuous-time safety follows.

Eight diagnostic contexts are fixed: each original case at k0/k1, before its
native query. After all scientific/replay jobs pass, execute two initializations:
(A) exact original prequery primal/duals/initial-run flag and relevant native
memory, requiring the original candidate bitwise; (B) fresh constant-current-state
and previous-input primal guess, zero duals/initial-run=False. Same source
configuration and identical OCP parameters, no diagnostic actions applied.
Report all raw/normalized action and solver differences. B need not equal A;
there is no invented action-agreement cutoff. Absence of sensitivity is only a
finite-context observation, not proof of six-input sufficiency. Numerical/status
failures in either branch are mandatory stops.

Reserve every job before dispatch, including failed work. Maximum816 solves,
2400 simulated control intervals,24000 charged monitoring subinterval checks,
600 CPU/1200 active-wall seconds, one worker/4GiB RSS. Scientific/native replay
use800 intervals each; independent replay uses800, each with10+20=30 charged
monitoring checks. The20-grid contains the10-grid: at most16000 distinct dense
subintervals, not24000 independent observations. No extra integration is used for
monitoring-grid refinement. Ledger actual work and reservations are separate;
unused failed-job reservations are retained. Time includes preflight and reporting.
Use aggregate external CPU/wall limits plus the in-process watchdog and durable
per-step records. Historical CSTR-audit and E12 ledgers are unchanged.

Report full four-case/eight-run/sixteen-replay denominators, per-case native and
dense raw margins, original stage/terminal costs, tracking RMSE, physical input
use, solver reliability, history sensitivity, numerical discrepancies and costs.
Incomplete runs never count safe or contribute an invented full-horizon return.
All generated evidence stays under ignored results/cstr_expert_pilot/technical_v1.
No commits/pushes are authorized by this pilot.
