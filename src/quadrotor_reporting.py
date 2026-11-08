"""Summaries and diagnostic plots from saved E12 records only."""
from pathlib import Path
import json
import numpy as np


def report(out):
    out=Path(out); v=json.loads((out/'verification.json').read_text()); f=json.loads((out/'frozen.json').read_text())
    usage=json.loads((out/'usage.json').read_text()); rows=v['records']
    base=[r for r in rows if r['phase']=='original']; completed=[r for r in rows if r['complete']]
    groups={}; lines=['# Experiment 12 — Initial four-case quadrotor technical pilot','',
      'Technical execution completed with two unresolved blockers. No screening, counterfactual action study or RL training was run.','',
      '## Frozen model and task','',
      f"safe-control-gym commit `{f['config']['upstream_commit']}`, Python 3.10.19, CasADi 3.7.2, PyBullet 3.2.7, Gymnasium 0.28.1 and NumPy 2.2.6. Full package versions, platform, binary/source hashes and the built PyBullet wheel are saved outside Git.",
      '', 'All four cases use the published six-second figure-eight task: 50 Hz control, 1,000 Hz physics, scale 0.75 m, one cycle, offset (0,1). T0 denotes a hover initial state, not a different constant-reference task. Q=diag(5,.1,5,.1,.1,.1), R=.1 I. Random initial states, disturbances, parameter randomization and action normalization are disabled.',
      '', 'State order: (x, x_dot, z, z_dot, theta, theta_dot); units (m,m/s,m,m/s,rad,rad/s). Two grouped physical thrust inputs are in N. Native state/input constraint values and their binary32 bounds are preserved.',
      '', f"Grouped thrust bounds: {f['action_low'][0]:.12g} to {f['action_high'][0]:.12g} N per channel; nominal hover {f['hover_input'][0]:.12g} N per channel. Native MPC observations/actions remain binary64; space bounds are binary32. No extra action dtype conversion was introduced.",
      '', '| Case | Frozen initial state |','|---|---|']
    for case,state in f['cases'].items():lines.append(f'| {case} | {state} |')
    lines += ['', 'T1 is read directly from the native reference: both initial translational velocities are π/4 m/s. Initial binary64 identities and the full reference are archived.',
      '', f"Maximum reset representation discrepancy: {max(r['reset_maximum_absolute_error'] for r in rows):.6g}; quaternion/Euler representation need not reproduce a requested angle bitwise. Frozen requested values are unchanged.",
      '', '## Results at 1,000 Hz','',
      'Returns are negative cumulative native quadratic costs, evaluated at post-transition states against reference index k+1. They are unscaled sums; integral costs equal 0.02 times these costs. The four states are technical cases, not a population sample.',
      '', '| Controller | Complete cases | Mean return | Worst return | Safe/contact-free completed cases | Mean squared-thrust integral (N² s) |',
      '|---|---:|---:|---:|---:|---:|']
    for ctrl in f['config']['controllers']:
        r=[x for x in base if x['controller']==ctrl]; good=[x for x in r if x['complete']]
        mean=float(np.mean([x['return'] for x in good])) if good else None
        worst=min([x['return'] for x in good],default=None)
        effort=float(np.mean([x['absolute_thrust_squared_integral'] for x in good])) if good else None
        groups[ctrl]={'complete_cases':len(good),'mean_return':mean,'worst_return':worst,
                      'mean_absolute_thrust_squared_integral':effort,'safe_cases':sum(x['safe_complete_no_contact'] for x in good)}
        cells=['N/A' if a is None else f'{a:.9f}' for a in (mean,worst,effort)]
        safe_count=f'{sum(x["safe_complete_no_contact"] for x in good)}/{len(good)}' if good else 'N/A'
        lines.append(f'| {ctrl} | {len(good)}/4 | {cells[0]} | {cells[1]} | {safe_count} | {cells[2]} |')
    lines += ['', '| Case | MPC20 return | MPC40 return | MPC80 return |','|---|---:|---:|---:|']
    for case in f['cases']:
        rr={r['controller']:r for r in base if r['case']==case}
        lines.append('| '+case+' | '+' | '.join(f'{rr[c]["return"]:.9f}' for c in ['nonlinear20','nonlinear40','nonlinear80'])+' |')
    lines += ['', '| Case/controller | Position RMSE (m) | Min normalized margin | Squared-thrust integral (N² s) | Peak channel thrust (N) |',
              '|---|---:|---:|---:|---:|']
    for r in base:
        if r['complete']:
            lines.append(f"| {r['case']}/{r['controller']} | {r['position_tracking_rmse']:.8f} | {r['minimum_normalized_margin']:.8f} | {r['absolute_thrust_squared_integral']:.8f} | {max(r['peak_thrust_by_channel']):.8f} |")
    lines += ['', 'Linear MPC has no completed returns or safety-rate estimate. Its original/repeat attempts at both physics resolutions fail before any action is applied; an empty safe prefix is not a safe episode.',
      '', '## Safety, repeatability and independent checks','',
      'All 24 completed nominal-frequency nonlinear executions (12 conditions plus repeats) and all 24 nonlinear refinement executions (including controller-level repeats) were observed safe and contact-free. There were zero raw state violations. This is finite sampled simulator evidence, not continuous-time, robust or physical-flight certification.',
      '', f"Each completed 1,000 Hz episode contains 6,001 raw physics states; each 2,000 Hz episode contains 12,001. The global normalized minimum margin among completed executions was {min(r['minimum_normalized_margin'] for r in completed):.12g}.",
      '', 'Minimum raw margins across completed executions, by (x,x_dot,z,z_dot,theta,theta_dot), with each component retaining its own units: `'+str(np.min([r['minimum_margins_by_state'] for r in completed],axis=0).tolist())+'`. All twelve lower/upper margins, trajectories, first-crossing fields, terminal behavior and per-controller costs/effort/RMS/peak/saturation/variation metrics are in verification.json.',
      '', 'All 32 original/repeat condition pairs across both physics resolutions match bitwise for their available state/action/reward, prediction and solver-iteration records. Eight pairs contain only failed zero-transition linear-MPC attempts. Each of all 64 attempts has both an uninstrumented action replay and an independently instrumented action replay. The 48 completed attempts reproduce control states/rewards bitwise; instrumented replays additionally reproduce all physics states and contacts. Zero-transition replays are bookkeeping checks only.',
      '', 'Instrumentation delegates to the unchanged native stepSimulation, then queries state/contact information. It never refreshes upstream cached states or changes forces. Uninstrumented replay confirms no observed change to control-boundary states/rewards on these trajectories.',
      '', f"Independent reference reconstruction maximum error: {v['reference_maximum_error']:.6g}. Cost reconstruction maximum error: {max(r['cost_reconstruction_error'] for r in rows):.6g}. Executed thrust reconstruction maximum error: {max(r['executed_thrust_reconstruction_error'] for r in rows):.6g} N.",
      '', f"Independent/native RK4 agreement maximum error: {max(r['independent_native_RK4_maximum_error'] for r in rows):.6g}. Maximum reconstructed shooting-dynamics residual: {max(r['shooting_dynamics_max_error'] or 0 for r in rows):.6g}. Predicted bound excesses remain separate from executed-state checks.",
      '', 'Requested thrusts, clipping, motor groups (1,4)/(2,3), PWM/RPM conversions and realized thrusts were independently checked. Raw safety checks use no rounding tolerance. Small arithmetic tolerances in cost/motor reconstruction do not relax safety boundaries.',
      '', '## Failed checks and preserved upstream behavior','',
      '**Linear-MPC execution blocker:** the published configuration requests QRQP, but its unchanged Darwin branch selects IPOPT on an Opti conic problem. The installed CasADi has no conic-IPOPT plugin. The native controller catches the initial error and returns a fallback; the runner then detects absent solved statistics, records failure and applies no fallback action. No claim of control infeasibility follows.',
      '', '**One-step numerical tolerance failure:** symbolic predictions differ from PyBullet by more than the frozen 0.001 scaled tolerance in several completed conditions. The overall maximum is '+f"{max(r['symbolic_pybullet_one_step_max_scaled'] for r in completed):.12g}. "+'The independent RK4 reconstruction agrees with native RK4; the discrepancy concerns the symbolic/PyBullet model pair.',
      '', f"Only {sum(r['one_step_tolerance_passed'] for r in base if r['complete'])}/12 original nonlinear case/controller pairs pass the one-step tolerance. The dominant discrepancy is pitch-rate prediction in rad/s: a matched-state one-step discrepancy, not a closed-loop trajectory difference.",
      '', f"The symbolic pitch lever is L/sqrt(2)={f['arm_length']/np.sqrt(2):.12g} m, whereas cf2x.urdf uses 0.028 m rotor offsets. Saved-transition fits are close to the URDF value. Rotation diagnostics quantify the much smaller residual after using that documented lever in a matched-state equation comparison. This is evidence of a source-geometry contribution, not a changed model, a complete implementation-error bound or a corrected trajectory.",
      '', 'The nonlinear warm start shifts rows of the input array (`u_guess[:-1]=u_guess[1:]`), not time columns. This behavior is unchanged and independently checked against the actual initial guesses. No repaired controller is evaluated. Linear/nonlinear classes also retain their native numerical constraint tightenings. Nonlinear terminal cost includes a constant hover-input offset absent from linear MPC; this constant does not change its optimizer. Evaluated trajectory costs are identical in definition across controllers.',
      '', 'All 12 completed nonlinear refinement pairs preserve safety classification and remain within the frozen 1% relative-cost tolerance. Maximum relative cost change: '+f"{max(x['relative_cost_change'] for x in v['refinement_checks'] if x['both_complete']):.8%}. "+'Linear-MPC refinement is unavailable, not passed.',
      '', '## Computational cost and preservation','',
      f"Recorded worker CPU: {usage['cpu_seconds']:.3f} s; solver attempts: {usage['solver_calls']:,} (14,400 successful nonlinear solves and 16 pre-action linear failures); physics steps including replays: {usage['physics_steps']:,}. Peak worker RSS: {max(r['peak_rss_bytes'] for r in rows)/2**20:.3f} MiB. Maximum completed action-selection time: {max(r['maximum_solver_wall_seconds'] for r in completed):.6f} s. Maximum solver iterations: {max(r['maximum_solver_iterations'] for r in completed)}.",
      '', 'The one-second action-selection watchdog was not triggered. Runtime includes native solve preparation in each action call; the largest cold-start call is reported rather than discarded. Native solver logs and unsuccessful candidates are retained. Physics integration and readout overhead are separately timed in each record. Setup/build time is separate from worker-execution CPU.',
      '', 'The measured nonlinear throughput is compatible with a four-CPU-hour study, but is not a guarantee for unseen states. A single 16-state/four-controller pass costs up to 19,200 solve calls; four nine-action/300-step probes cost up to 10,800 more before reuse. These fit 40,000 as a separate phase, but adding the nominal/repeated/refinement technical allocation can exceed a study-wide 40,000 cap. Exact accounting, replays, repeats and baseline reuse must be reviewed before scaling. No cap was raised.',
      '', f"Independent preservation verified {v['preserved_project_files']:,} pre-existing project files, {v['preserved_authority_files']} authority files and {v['pinned_upstream_files']} upstream files. All nine IQL files also match previously sealed identities and remain unimported. All 127 applicable regression tests passed in the historical Python 3.13 environment; the five new contract tests separately passed under pinned Python 3.10. Unfinished IQL tests were excluded.",
      '', 'A CLI postprocessing/logging attempt exited 120 after writing a complete verification JSON, with no captured diagnostic output. Independent verification was rerun directly twice and its JSON hashes compared. This attempt and the PyBullet build failure are retained as technical execution records, not silently removed. Neither altered simulation records.',
      '', '## Decision','',
      'Nonlinear MPC execution, deterministic replay, full physics capture and cost reconstruction are demonstrated on the four technical cases. The pilot does not pass the complete readiness gate: linear MPC remains blocked and the frozen symbolic/PyBullet discrepancy tolerance fails. Stop for scientific review. No 16-state screen, counterfactual menu, data acquisition, controller repair or training is authorized by these results. Horizon comparisons are descriptive technical diagnostics; they do not establish RL headroom.',
      '', 'Generated evidence: results/benchmark_selection/technical_pilot/. Authoritative source archive, configuration/reference identities, full environment lock, build failure/workaround logs, per-attempt logs/trajectories, verification.json, usage.json and plots remain outside Git.']
    summary={'controllers':groups,'usage':usage,'ready_for_full_study':False,
        'blockers':['Native macOS linear-MPC conic-IPOPT plugin mismatch','Frozen symbolic/PyBullet one-step tolerance exceeded']}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    text='\n'.join(lines)+'\n';(out/'scientific_report.md').write_text(text)
    return text


def plots(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out=Path(out);ref=np.load(out/'reference.npy');folder=out/'plots';folder.mkdir(exist_ok=True)
    labels=['x (m)','vx (m/s)','z (m)','vz (m/s)','pitch (rad)','pitch rate (rad/s)']
    for case in ['T0','T1','T2','T3']:
        fig,axes=plt.subplots(4,2,figsize=(10,11),sharex=True)
        for c in ['nonlinear20','nonlinear40','nonlinear80']:
            a=np.load(out/'runs'/f'{case}_{c}_1000_original'/'trajectory.npz')
            t=np.arange(len(a['states']))/50
            for j in range(6):axes.flat[j].plot(t,a['states'][:,j],label=c)
            for j in range(2):axes.flat[j+6].plot(t[:-1],a['clipped_actions'][:,j],label=c)
        for j in range(6):
            axes.flat[j].plot(np.arange(301)/50,ref[:,j],'k--',alpha=.65,label='reference')
            axes.flat[j].set_ylabel(labels[j])
        for j in range(2):axes.flat[j+6].set_ylabel(f'Grouped thrust {j+1} (N)');axes.flat[j+6].set_xlabel('Time (s)')
        axes.flat[0].legend(fontsize=8);fig.suptitle(case+' — Technical tracking trajectories; linear MPC unavailable')
        fig.tight_layout();fig.savefig(folder/f'{case}_trajectories.png',dpi=150);plt.close(fig)
