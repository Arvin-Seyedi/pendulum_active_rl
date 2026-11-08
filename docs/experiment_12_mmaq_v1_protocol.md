# E12_MMAQ_v1 — prospective simulation-expert qualification

This additive version preserves the historical **FAILED** 0.001 symbolic/PyBullet
test, the failed native macOS linear baseline, adapted linear40_qrqp_v1 and the
unchanged nonlinear20/40/80 baselines. No plant, model, assets, controller graph,
warm start, objective, constraint or horizon is replaced. No hard real-time or
hardware safety claim is made.

## Mathematical contract, frozen before held-out outcomes

State order is (x,vx,z,vz,theta,omega). Let D=1/50 s, h=1/f_physics,
beta_s=sum(model thrust)/m, beta_p=sum(motor-derived thrust)/m,
alpha_s=(L/sqrt(2))*diff(model thrust)/Iyy and alpha_p=.028*diff(executed thrust)/Iyy.
All are frozen source parameters, not regression fits. Model thrust is the native
clipped command; executed thrust is calculated causally by the pinned PWM/RPM map.
At reset c=theta. Otherwise c is the preceding recorded physics attitude, available
before the current control step. Future states are never arguments to a bound.

For the declared contact-free planar source recurrence:

    v_next = v + h*a(c)
    omega_next = omega + h*alpha_p*cos(theta-c)
    position_next = position + h*velocity_next
    c_next = theta

define Omega_p=|omega0|+D|alpha_p| and
delta=max(|theta0-c0|,h*Omega_p). The angular acceleration discrepancy is bounded by
A=|alpha_p-alpha_s|+|alpha_p|*delta^2/2. Thus
Eomega=D*A and Etheta=A*D*(D+h)/2+|alpha_s|*D*h/2.
The second term explicitly bounds right-rectangle/semi-implicit integration.

Using the global unit Lipschitz constants of sine/cosine, bound translation's
left-rectangle velocity discrepancy by

    Ev = D*(|beta_p-beta_s|+beta_p*(delta+Etheta))
         + beta_s*(|omega0|+D|alpha_s|)*D*h/2.

Position bounds are D*Ev+beta_s*D*h/2 (x) and
D*Ev+(beta_s+g)*D*h/2 (z). These are conservative analytic majorants, not calibrated
prediction intervals. Angular output must remain in the planar Euler chart;
contacts/nonplanar dynamics are outside the contract.

For nonlinear MPC, add bounds for its unchanged RK4 readout versus the continuous
symbolic solution. The two RK4 midpoint angles deviate from the exact midpoint
by |alpha_s|D^2/8. Simpson's remainder uses
|d^4 sin(theta(t))/dt^4| <= Omega_s^4+6*Omega_s^2*|alpha_s|+3*alpha_s^2.
The velocity remainder is beta_s*|alpha_s|D^3/12 plus beta_s*D^5/2880 times that
derivative bound. Position uses the weighted-integral first-derivative bound
implemented explicitly in the contract. Angular RK4 components are exact for
this constant-torque symbolic equation in real arithmetic.

For linear MPC, use the exact hover-linearized ZOH polynomial and add its
linearization remainder instead of RK4's remainder. With
Theta=|theta0|+|omega0|D+|alpha_s|D^2/2 and g_lin=sum(U_EQ)/m:

    Ax <= beta_s*Theta^3/6 + |beta_s-g_lin|*Theta
    Az <= beta_s*Theta^2/2 + |g_lin-g|.

These contribute (D^2*Ax/2,D*Ax,D^2*Az/2,D*Az,0,0).
All bound expressions use 80-digit ROUND_CEILING Decimal arithmetic followed by
outward binary64 conversion. An independent exact-Fraction evaluator checks each
bound against the same real-valued majorant on archived operands.

The proof concerns the declared source recurrence. Agreement with native
binary64 Bullet is an **empirical numerical contract** using the existing 1e-10
reconstruction tolerance, separately from the analytic majorants. It is not a
uniform libm/Bullet floating-point enclosure. Only current/past state/cache and
the deterministic command conversion are used to construct predictions/bounds.
Safety reserves are componentwise distance of each one-step prediction from its
raw box boundaries minus its bound. This is not recursive safety certification.

## Frozen gates and scope

Phase A verifies all historical seals, the original design recovery, independent
PCG64 raw-bit sampling, archived discrepancy coverage, exact bound arithmetic,
optimization-graph reconstruction and failure-safe/nonfinite logging. Any failure
or unresolved contract blocks new episodes. The approved sixteen-state screening
coordinates and nine offsets are sealed but **never evaluated here**. The original
design specified seed/ranges, not the draw-loop arrangement: row-major physical
state order and lexicographic menu order are explicitly frozen before outcomes.

Only after Phase A passes, four held-out states are drawn with PCG64 120101 from
the published ranges, rejecting exact coordinate duplicates of screening/pilot
states. No performance/safety replacement is allowed. H0-H3 are each executed
twice by adapted linear40 and nonlinear20/40/80: 32 scientific executions.
Each completed execution has 300 controls and all 6,001 raw physics states.
Each receives native uninstrumented and recorded-motor instrumented replays.

The independent linear-QP KKT threshold remains 1.01e-8. NLP requires native
Solve_Succeeded and independently reconstructed native default feasibility/KKT
criteria (unscaled primal <=1e-4; stationarity and normalized complementarity
<=1.01e-8). Reconstruction <=1e-10. No solver settings change. Local NLP
stationarity is not global optimality. Actual negative raw safety margins have
no rounding tolerance. Failures, invalid arithmetic, contacts or incompleteness
never count as safe. MPC80 qualification additionally requires positive
one-control-step reserves on every held-out step. All baseline outcomes remain
reported even when a comparison controller fails an empirical criterion.

The original cumulative allocation, including all accepted work, is retained:
40,000 solves, 3,000,000 physics steps, 14,400 CPU seconds, 4 GiB and one worker;
one-second action watchdog. The held-out phase reserves 9,600 solves and 576,000
physics steps including both replays. Attempts/failures are charged before dispatch.
The proposed additional full-study allocation is not authorized. Screening,
counterfactual probes, data acquisition, training and IQL remain disabled.

Failure JSON explicitly tags NaN/Inf, writes atomically and retains partial arrays,
unapplied rejected actions, logs, progress and failed/incomplete status. It adds no
fallback action. Every 20-ms wall-clock overrun is reported; offline simulation
waits for computation and does not establish hard real-time readiness.
