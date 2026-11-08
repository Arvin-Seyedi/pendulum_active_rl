"""Independent record mathematics and preservation checks for E12.

No producer dynamics, reward, margin or motor conversion functions are imported.
No trajectory is integrated here. RK4 calculations are matched-state one-step
predictions against archived transitions, not new closed-loop experiments.
"""
from pathlib import Path
import hashlib
import json
import subprocess
import numpy as np


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def mechanics(x, u, f):
    total = u[0] + u[1]
    return np.array([x[1], np.sin(x[4])*total/f['mass'], x[3],
                     np.cos(x[4])*total/f['mass']-f['gravity'], x[5],
                     f['arm_length']*(u[1]-u[0])/(np.sqrt(2)*f['Iyy'])])


def rk4(x, u, f, h=.02):
    a = mechanics(x, u, f); b = mechanics(x+h*a/2, u, f)
    c = mechanics(x+h*b/2, u, f); d = mechanics(x+h*c, u, f)
    return x+h*(a+2*b+2*c+d)/6


def audit_record(folder, f, reference):
    meta = load(folder/'completion.json')
    assert digest(folder/'trajectory.npz') == meta['trajectory_sha256']
    a = dict(np.load(folder/'trajectory.npz')); n = meta['transitions']
    states = a['states']; physics = a['physics_states']; actions = a['clipped_actions'].reshape(n,2)
    low, high = np.array(f['state_low']), np.array(f['state_high'])
    margins = np.minimum(physics-low, high-physics)
    norms = margins/(high-low)
    assert len(physics) == n*(meta['frequency']//50)+1
    assert physics[::meta['frequency']//50].tobytes() == states.tobytes()
    assert np.array_equal(a['current_steps'][:n], np.arange(n))
    assert np.max(np.abs(states[0]-f['cases'][meta['case']])) < 1e-14
    assert np.isfinite(physics).all()
    violation = np.any(margins < 0, axis=1)
    q, r = np.array(f['Q']), np.array(f['R']); hover = np.array(f['hover_input'])
    error = states[1:n+1] - reference[1:n+1]
    command_error = actions-hover
    costs = .5*np.einsum('ni,ij,nj->n', error, q, error)+.5*np.einsum('ni,ij,nj->n', command_error, r, command_error)
    cost_error = float(np.max(np.abs(costs+a['rewards']))) if n else 0.
    assert cost_error < 1e-10
    # Directly reconstruct the two grouped thrusts from actual four-motor RPM.
    rpms = a['rpms'].reshape(n,4); executed = f['KF']*np.column_stack([rpms[:,0]**2+rpms[:,3]**2, rpms[:,1]**2+rpms[:,2]**2]) if n else np.empty((0,2))
    motor_error = float(np.max(np.abs(executed-actions))) if n else 0.
    assert motor_error < 1e-12
    pwm = (rpms-f['PWM2RPM_CONST'])/f['PWM2RPM_SCALE']
    if n:
        assert np.min(pwm) >= f['MIN_PWM']-1e-8 and np.max(pwm) <= f['MAX_PWM']+1e-8
        assert np.max(np.abs(executed-np.clip(executed, f['action_low'], f['action_high']))) < 1e-12
    prediction = np.array([rk4(states[i], executed[i], f) for i in range(n)])
    native_model_error = float(np.max(np.abs(prediction-a['native_model_predictions'][:n]))) if n else 0.
    assert native_model_error < 1e-12
    discrepancy = np.abs(prediction-states[1:n+1]) if n else np.zeros((1,6))
    residual = []
    # Predicted shooting feasibility: linear controller uses exact linearized dynamics;
    # nonlinear controllers use RK4. Actual safety is always checked independently.
    if n:
        for x, u in zip(a['predicted_states'], a['predicted_inputs']):
            residual.append(float(np.max(np.abs(x[:,0]-states[len(residual)]))))
    shooting_error = None; predicted_state_excess = None; predicted_input_excess = None
    if n:
        xs=a['predicted_states']; us=a['predicted_inputs']
        predicted_state_excess=float(max(0,np.max(low[None,:,None]-xs),np.max(xs-high[None,:,None])))
        predicted_input_excess=float(max(0,np.max(np.array(f['action_low'])[None,:,None]-us),
                                         np.max(us-np.array(f['action_high'])[None,:,None])))
        if meta['controller']!='linear40':
            flatx=np.moveaxis(xs[:,:,:-1],1,0).reshape(6,-1)
            flatu=np.moveaxis(us,1,0).reshape(2,-1)
            nextx=np.moveaxis(xs[:,:,1:],1,0).reshape(6,-1)
            shooting_error=float(np.max(np.abs(rk4(flatx,flatu,f)-nextx)))
    solver_success = bool(len(a['solver_success']) == n and np.all(a['solver_success']))
    safe = bool(meta['complete'] and solver_success and not violation.any() and not np.any(a['physics_contacts']))
    result = {'case': meta['case'], 'controller': meta['controller'], 'frequency': meta['frequency'],
        'phase': meta['phase'], 'complete': meta['complete'], 'failure': meta['failure'], 'transitions': n,
        'reset_maximum_absolute_error':meta['reset_maximum_absolute_error'],
        'physics_samples': len(physics), 'safe_complete_no_contact': safe,
        'violating_physics_samples': int(violation.sum()), 'violating_samples_by_state': (margins<0).sum(axis=0).tolist(),
        'minimum_margins_by_state': np.min(margins, axis=0).tolist(), 'minimum_normalized_margin': float(norms.min()),
        'minimum_lower_margins_by_state': np.min(physics-low,axis=0).tolist(),
        'minimum_upper_margins_by_state': np.min(high-physics,axis=0).tolist(),
        'first_crossing_seconds': float(np.flatnonzero(violation)[0]/meta['frequency']) if violation.any() else None,
        'contact_physics_samples': int(np.count_nonzero(a['physics_contacts'])),
        'return': -float(costs.sum()) if meta['complete'] else None,
        'partial_return': -float(costs.sum()), 'cost_reconstruction_error': cost_error,
        'discounted_return_gamma_099': -float(np.dot(.99**np.arange(n), costs)),
        'position_tracking_rmse': float(np.sqrt(np.mean(error[:,[0,2]]**2))) if n else None,
        'tracking_rmse_by_state': np.sqrt(np.mean(error**2, axis=0)).tolist() if n else None,
        'absolute_thrust_squared_integral': float(np.sum(executed**2)/50),
        'hover_relative_thrust_squared_integral': float(np.sum((executed-hover)**2)/50),
        'thrust_rms_by_channel': np.sqrt(np.mean(executed**2,axis=0)).tolist() if n else None,
        'peak_thrust_by_channel': executed.max(axis=0).tolist() if n else None,
        'saturation_fraction_by_channel': np.mean((np.abs(executed-np.array(f['action_low']))<1e-7)|
                      (np.abs(executed-np.array(f['action_high']))<1e-7),axis=0).tolist() if n else None,
        'total_variation': float(np.sum(np.abs(np.diff(executed,axis=0)))),
        'clipping_steps': int(np.any(a['actions'][:n]!=actions,axis=1).sum()) if n else 0,
        'executed_thrust_reconstruction_error': motor_error,
        'maximum_requested_clipped_difference': float(np.max(np.abs(a['actions'][:n]-actions))) if n else None,
        'symbolic_pybullet_one_step_max_by_state': discrepancy.max(axis=0).tolist(),
        'symbolic_pybullet_one_step_max_scaled': float(discrepancy.max()),
        'independent_native_RK4_maximum_error': native_model_error,
        'one_step_tolerance_passed': bool(discrepancy.max() <= f['config']['one_step_scaled_tolerance']) if n else None,
        'terminal_state': states[-1].tolist(), 'terminal_tracking_error': (states[-1]-reference[n]).tolist(),
        'shooting_initial_state_max_error': max(residual, default=0),
        'shooting_dynamics_max_error':shooting_error,
        'predicted_state_bound_excess':predicted_state_excess,
        'predicted_input_bound_excess':predicted_input_excess,
        'maximum_solver_iterations': int(a['solver_iterations'].max()) if len(a['solver_iterations']) else None,
        'solver_status_counts': {s: meta['solver_statuses'].count(s) for s in set(meta['solver_statuses'])},
        'mean_solver_wall_seconds': float(a['solver_wall'].mean()) if len(a['solver_wall']) else None,
        'maximum_solver_wall_seconds': float(a['solver_wall'].max()) if len(a['solver_wall']) else None,
        'maximum_warm_u_error': float(a['warm_u_error'].max()) if len(a['warm_u_error']) else None,
        'maximum_warm_x_error': float(a['warm_x_error'].max()) if len(a['warm_x_error']) else None,
        'physics_integration_seconds': float(a['physics_integration_seconds'].sum()),
        'physics_capture_seconds': float(a['physics_capture_seconds'].sum()),
        'cpu_seconds': meta['cpu_seconds'], 'wall_seconds': meta['wall_seconds'], 'peak_rss_bytes': meta['peak_rss_bytes']}
    for name in ('replay_native.json','replay_instrumented.json'):
        replay = load(folder/name)
        assert replay['control_states_bitwise'] and replay['rewards_bitwise']
        if 'physics_states_bitwise' in replay:
            assert replay['physics_states_bitwise'] and replay['contacts_bitwise']
    if n:
        lever = .028  # Archived cf2x.urdf rotor x-offset; diagnostic only, no model replacement.
        difference = executed[:,1]-executed[:,0]
        acceleration = np.diff(states[:,5])/.02
        fit = f['Iyy']*np.dot(difference,acceleration)/np.dot(difference,difference)
        result['rotation_diagnostic'] = {'symbolic_lever_m': f['arm_length']/np.sqrt(2),
            'URDF_lever_m': lever, 'record_fitted_effective_lever_m':float(fit),
            'maximum_URDF_one_step_angular_velocity_residual':float(np.max(np.abs(np.diff(states[:,5])-.02*lever/f['Iyy']*difference))),
            'interpretation':'Matched-state source-geometry comparison only; no corrected trajectory or controller'}
    return result, a


def verify(root, upstream, out):
    root, out, upstream = Path(root), Path(out), Path(upstream)
    f = load(out/'frozen.json'); reference = np.load(out/'reference.npy')
    assert digest(out/'reference.npy') == f['reference_sha256']
    # Reconstruct reference independently from the published figure-eight formula.
    t = np.arange(301)/50; w = 2*np.pi/6
    ref = np.column_stack([.75*np.sin(w*t), .75*w*np.cos(w*t),
        1+.75*np.sin(w*t)*np.cos(w*t), .75*w*np.cos(2*w*t), np.zeros(301), np.zeros(301)])
    assert np.max(np.abs(ref-reference)) < 1e-14
    assert set(f['cases']) == {'T0','T1','T2','T3'}
    records=[]; data={}; repeats=[]; refinements=[]
    for folder in sorted((out/'runs').iterdir()):
        result, arrays = audit_record(folder, f, reference)
        records.append(result); data[folder.name]=arrays
    for case in f['cases']:
        for controller in f['config']['controllers']:
            prefix = f'{case}_{controller}_'
            a,b = data[prefix+'1000_original'], data[prefix+'1000_repeat']
            keys = ['states','physics_states','physics_contacts','actions','clipped_actions','rpms','rewards',
                    'dones','predicted_states','predicted_inputs','native_model_predictions','current_steps','solver_iterations','solver_success']
            equal = {k:a[k].tobytes()==b[k].tobytes() for k in keys}
            assert all(equal.values()), (case,controller,equal)
            repeats.append({'case':case,'controller':controller,'frequency':1000,'bitwise':equal})
            fine_a,fine_b=data[prefix+'2000_refine'],data[prefix+'2000_repeat']
            fine_equal={k:fine_a[k].tobytes()==fine_b[k].tobytes() for k in keys}
            assert all(fine_equal.values()),(case,controller,fine_equal)
            repeats.append({'case':case,'controller':controller,'frequency':2000,'bitwise':fine_equal})
            base = next(r for r in records if r['case']==case and r['controller']==controller and r['phase']=='original')
            fine = next(r for r in records if r['case']==case and r['controller']==controller and r['phase']=='refine')
            valid = base['complete'] and fine['complete']
            change = abs(fine['partial_return']-base['partial_return'])/max(abs(base['partial_return']),1e-15)
            refinements.append({'case':case,'controller':controller,'both_complete':valid,
                'relative_cost_change': change if valid else None, 'same_safety_classification':base['safe_complete_no_contact']==fine['safe_complete_no_contact'] if valid else None,
                'cost_tolerance_passed': bool(valid and change<=f['config']['refinement_relative_cost_tolerance'])})
    preserved=load(out/'preservation_before.json')
    for name,h in preserved['files'].items():
        assert digest(root/name)==h, 'Historical project file changed: '+name
    for name,h in preserved['authority_files'].items():
        assert digest(name)==h, 'Authority changed: '+name
    for name,h in f['upstream_files'].items():
        assert digest(upstream/name)==h, 'Upstream changed: '+name
    summary={'records':records,'repeat_checks':repeats,'refinement_checks':refinements,
        'preserved_project_files':len(preserved['files']),'preserved_authority_files':len(preserved['authority_files']),
        'pinned_upstream_files':len(f['upstream_files']), 'reference_maximum_error':float(np.max(np.abs(ref-reference))),
        'all_replays_passed':True,'all_bitwise_repeats_passed':True,'scope':'Four technical cases only; no screening, counterfactuals or training'}
    (out/'verification.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return {k:v for k,v in summary.items() if k not in ['records','repeat_checks','refinement_checks']}
