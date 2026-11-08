# Experiment 12 — Compatibility and recorded-transition diagnostic

The linear-MPC failure is a solver-interface dispatch incompatibility. The model
discrepancy is explained on the archived samples by geometry, semi-implicit
integration and a one-physics-step cached-link attitude lag. **The original
0.001 acceptance gate still fails.** No model, threshold, controller, prior
result or frozen configuration was changed. No quadrotor episode or physics
step was executed.

## Scope and provenance

The accepted four-case pilot, its 485 generated artifacts and nine final source
identities were verified before analysis. Earlier preservation checks cover
53,819 historical project files, 87 drilling authority files and 289 pinned
safe-control-gym files. The unchanged upstream commit is
`6b5391d014f36fdfa0f9d22d92c77387e5274308`; the original Python 3.10.19,
CasADi 3.7.2 and PyBullet 3.2.7 installation is used.

Recorded analysis uses all 24 completed original/refinement nonlinear records:
four cases × three horizons × two physics frequencies. This comprises **7,200
control intervals and 216,000 physics intervals**. Controller repeats are
preserved but not counted as additional independent evidence. Failed linear
attempts contain no transitions and contribute no prediction observations.
Outputs are under `results/benchmark_selection/compatibility_diagnostic/`,
separate from the accepted pilot.

## A. Linear MPC compatibility

The pinned `LinearMPC.setup_optimizer` constructs `Opti('conic')` when its
requested solver is QRQP. Its Darwin branch subsequently calls
`opti.solver('ipopt', {'expand': True, 'ipopt.max_iter': 100})`. This asks for
the **conic** IPOPT interface, not the available nonlinear IPOPT interface.

The installed plugin checks and standalone solves establish:

| Interface | Available and tested |
|---|---|
| Conic QRQP | Yes |
| Nonlinear IPOPT | Yes |
| Conic IPOPT | No; native dispatch reproduces the missing-plugin failure |

This is not evidence of an infeasible MPC problem, absent IPOPT generally, or
inadequate actuator authority. CasADi distinguishes conic/QP and nonlinear
solver interfaces in its [official documentation](https://web.casadi.org/docs/#quadratic-programming).

### Minimal proposed compatibility remedy

The separate, **unconnected** local adapter
`src/quadrotor_compatibility_v1.py` provides `linear_conic_qrqp_v1`:

```python
# After the unmodified native linear-QP construction:
configure_linear_conic_qrqp_v1(
    opti, requested_solver='qrqp', native_backend='ipopt',
    native_options={'expand': True, 'ipopt.max_iter': 100})
# Dispatches only: opti.solver('qrqp', {'expand': True})
```

It retains the published requested QRQP backend and removes the incompatible
IPOPT-specific option. It does not invent a QRQP iteration limit or access the
objective, constraints, parameters, horizon, model, observations or warm starts.
Unexpected dispatch/options are rejected. No upstream file is edited and no
pilot runner is wired to this adapter. A future binding must be explicitly
labelled and source-guarded; it must not replace the original failed baseline.

Running the unchanged implementation on Linux is another possibility, but
requires a separate platform/environment reproduction. Recasting the problem as
an IPOPT NLP also has the same mathematical optimum for these convex examples,
but changes more of the intended solver path than retaining published QRQP.

### Standalone mathematical checks

Three two-variable strictly convex QPs exercise an active sum constraint, an
interior solution, and simultaneous lower/upper bounds. The common Hessian is
`[[4,1],[1,2]]`; constraints are `0 <= x,y <= 0.8` and `x+y <= 1`.
Their linear terms and exact solutions/objectives are:

| Linear term | Exact optimizer | Exact objective |
|---|---|---|
| (-5,-3) | (3/4,1/4) | -25/8 |
| (-1,-1) | (1/7,3/7) | -2/7 |
| (2,-3) | (0,4/5) | -44/25 |

Exact-rational KKT checks independently verify these references. Each problem
is solved twice using adapter QRQP and reference nonlinear IPOPT. Serialized
objective/constraint program graphs are identical before/after solver dispatch.
The rational references describe the stated decimal problems; the solvers
evaluate binary64 inputs, including the usual representation of 0.8.

- QRQP maximum optimizer/objective error: **2.22e-16**; primal excess zero;
  maximum stationarity residual **6.66e-16**.
- Nonlinear IPOPT maximum optimizer error: **2.573e-8**; objective error
  **3.699e-8**; primal excess **9.102e-9**, retained rather than rounded away.
- Maximum QRQP/IPOPT optimizer difference: **2.573e-8**.
- An analytically infeasible scalar QP is rejected by QRQP. Its returned
  status is `unknown` with a solve exception: failure handling must check the
  exception and success flag, not infer feasibility from a status string.

There were 14 tiny solve attempts total: 12 successful solves, the intentional
native-interface failure and the infeasible control. These tests establish
availability and mathematical-program preservation, **not full horizon-40
numerical conditioning, quadrotor performance or safe closed-loop operation**.

## B. Recorded model-discrepancy diagnosis

All intervals use the same archived initial states and executed thrusts. Control
states match physics states at their exact stride; counters and endpoints
confirm `k -> k+1` alignment. No reference shift is used to improve agreement.
The original scale vector remains `(1,1,1,1,1,1)`, so the following absolute
values are numerically equal to the frozen scaled errors.

| Component | Unit | Native maximum, 1,000 Hz | Native maximum, 2,000 Hz | Full diagnostic recurrence residual, both frequencies |
|---|---|---:|---:|---:|
| x | m | 9.34665e-5 | 4.67365e-5 | 1.11e-16 |
| x_dot | m/s | 2.14578e-3 | 1.07357e-3 | 4.44e-16 |
| z | m | 1.21140e-4 | 6.05659e-5 | 2.22e-16 |
| z_dot | m/s | 5.36117e-4 | 2.68418e-4 | 4.44e-16 |
| theta | rad | 1.58159e-3 | 7.47839e-4 | 9.44e-16 |
| theta_dot | rad/s | 8.59824e-3 | 8.59400e-3 | 2.22e-15 |

The unchanged native model exceeds 0.001 on **37/3,600** nominal intervals and
**29/3,600** refined intervals; **8/12 case/controller records fail at each
frequency**. Existing failed classifications are not replaced by the diagnostic
recurrence's agreement.

The largest native errors all occur in T2 at 1,000 Hz. Their control-interval
start times are fixed by the archived index; each interval lasts 0.02 s:

| Component | Controller | Interval start (s) |
|---|---|---:|
| x | MPC20 | 0.18 |
| x_dot | MPC80 | 0.02 |
| z | MPC80 | 0.02 |
| z_dot | MPC80 | 0.04 |
| theta | MPC80 | 0.00 |
| theta_dot | MPC80 | 0.00 |

Full signed contributions, exact initial states and executed commands for each
worst interval are retained in `transition_diagnosis.json`.

### Identifiable contributions

1. **Geometry:** symbolic pitch acceleration uses `L/sqrt(2)` =
   **0.0280721392131 m**. The four URDF motor inertial offsets use x-coordinates
   ±**0.028 m**. The lever difference is **0.257640%** relative to the URDF value.
   At the worst pitch-rate interval, T2/MPC80 at 0–0.02 s, its signed contribution
   is **+0.00859251653678 rad/s** of the total **+0.00859824313788 rad/s**;
   cached-frame torque projection contributes another **+5.72660e-6 rad/s**.

2. **Integration:** the recorded positions obey the semi-implicit update
   `position_next = position + h*velocity_next`, including pitch angle, rather
   than the continuous/RK4 map. The maximum pitch-angle contribution from this
   difference is **0.001667542885 rad** at 1,000 Hz and halves at 2,000 Hz.
   At the original worst pitch-angle interval this partly cancels the geometry
   contribution; component maxima therefore must not be added as percentages.

3. **Cached link-frame timing:** PyBullet transforms `LINK_FRAME` rotor forces
   using `m_cachedWorldTransform`. Its forward-kinematics cache is refreshed
   during external-force solution, after the application command has already
   transformed the force. Subsequent integration updates collision transforms,
   not that cache. Therefore the force direction uses the preceding physics
   sample's attitude, while force propagation uses the current configuration.
   This is a **one-physics-step attitude lag**, not a delayed control command.
   Its maximum horizontal-velocity contribution is **0.001430482466 m/s** at
   1,000 Hz and **0.000715788550 m/s** at 2,000 Hz.

For these recorded planar, contact-free cases, with `c_j` the cached preceding
attitude and `h=1/f_physics`, the source-supported diagnostic identities are:

```text
vx_next = vx + h*(T1+T2)/m * sin(c_j)
vz_next = vz + h*((T1+T2)/m * cos(c_j) - g)
omega_next = omega + h*0.028/Iyy*(T2-T1)*cos(theta-c_j)
(x,z,theta)_next = (x,z,theta) + h*(vx,vz,omega)_next
c_next = theta
```

The initial cache equals the reset attitude. All subsequent preceding attitudes
are available in the archived physics records. Checking every saved physics
interval yields maximum velocity-equation acceleration residuals below
**5.8e-13**, and position-update residuals below **4.3e-16** in their component
units. Independently reconstructing each control interval from its archived
start state, preceding attitude and command reproduces its endpoint within
**2.23e-15**. Predictions are reset at every interval; no full trajectory or
corrected controller is simulated.

### Other checks and remaining limitations

- Grouped motors are (1,4)/(2,3). The PWM conversion returns RPM despite the
  helper docstring's squared-RPM wording; physics squares RPM once. Executed
  grouped-thrust reconstruction differs by at most **1.11e-16 N**. There is no
  torque-factor-of-two or action-normalization explanation.
- URDF mass is 0.027 kg; pitch inertia is 1.4e-5 kg m²; gravity is 9.8 m/s².
  Recorded equations with cached-frame timing recover mass within 1e-16 kg,
  gravity within 3e-14 m/s² and lever/Iyy = 2000 within 4e-12. Lever and inertia
  are not independently identifiable from their ratio alone; source inertia
  and the native `URDF_USE_INERTIA_FROM_FILE` flag supply the stated convention.
- Pitch is Euler Y, and recorded angular velocity is world Y; they agree under
  the observed planar motion. This is not a claim about arbitrary 3D motion.
- Independent/native RK4 agreement is **5.58e-15**. Continuous one-interval
  Gaussian quadratures at orders 32/64 agree within **2.23e-16**. Native RK4
  discretization itself contributes at most **1.042e-5**, below 0.001.
- NumPy/Apple Accelerate emitted arithmetic warnings in the batched quadrature
  matrix products. These warnings are retained. All operands/results were
  finite; acceleration operands were bounded by 11.84. An independent
  elementwise-sum implementation with arithmetic errors set to raise compared
  all 14,400 lever/scenario interval predictions and agreed within **2.23e-16**
  without an arithmetic exception. The warning origin remains incompletely
  localized; it does not change the measured errors or gate classifications.
- Geometry alignment **alone** leaves maximum horizontal-velocity error
  **0.002145793196 m/s** and pitch-angle error **0.001667512049 rad**: still
  failures. Adding semi-implicit integration but omitting cache timing still
  leaves horizontal-velocity error **0.001430482466 m/s**.

The signed decomposition is native RK4 truncation + geometry + semi-implicit
integration + cached-frame timing + remaining residual. It closes on the
archived samples with zero measured binary64 closure residual. Its ordering is
explicit and attribution is not a global causal-percentage calculation.
The result explains these numerical observations; it is not a formal uniform
error bound, a hardware validation or proof of safety for untested policies.

## Recommendation

**Authorize only a separately labelled four-case linear-MPC compatibility
revalidation**, if scientifically approved, before considering screening. Use
`linear_conic_qrqp_v1`, retain the original failed baseline, verify the complete
horizon-40 objective/constraint graph before/after the scoped solver dispatch,
and retain the original model, URDF, gains, constraints, reference, warm-start
semantics and failure-stopping rules. Test both physics frequencies, repeat
deterministically and independently replay applied commands. Existing unchanged
nonlinear episodes need not be rerun.

Do not authorize the 16-state or counterfactual study yet. The original 0.001
model-equivalence gate still fails, although its recorded causes are now
identified. No corrected recurrence may silently replace the native predictor
for acceptance. A separate reviewed model-contract decision is required before
full screening; compatibility success alone would not clear that gate.

## Reproduction and preservation

`experiments/12_quadrotor_diagnostic.py --solver` runs only standalone QPs;
`--transitions` reads saved records. The versioned adapter is not imported by the
accepted pilot's execution path. Completed diagnostic outputs are not
overwritten. Source evidence snapshots, exact version/file hashes, plugin
errors, successful/infeasible toy results, all signed residual arrays and
per-case/component worst intervals are retained in the diagnostic output.
Seven focused mathematical/artifact tests cover analytic quadrature, exact
QP KKT conditions, closed-form angular attributions and telescoping closure.
Together with the five existing simulator-free E12 contract tests, **12 tests
passed**. Diagnostic numerical kernels consumed approximately **0.402 CPU-s**
in the primary solver/transition runs, with peak process memory below **51 MiB**;
imports, preservation hashing and repeat/checker runs are separate. The numerical
diagnosis and all signed arrays reproduced identically on a second calculation.
The accepted pilot and all historical preservation identities are rechecked
afterward. No previous source/configuration/report or IQL file is edited; no
commit or push is made.
