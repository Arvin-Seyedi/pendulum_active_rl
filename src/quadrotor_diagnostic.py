"""E12 compatibility and matched-state equation audit. No simulator calls.

Each diagnostic prediction is reset to an archived control-boundary state.
Alternative readouts/recurrences are attribution calculations only, never plant
or controller replacements, and their outputs are not training data.
"""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import time
import numpy as np
from src.quadrotor_verification import rk4
from src.quadrotor_compatibility_v1 import configure_linear_conic_qrqp_v1


def write(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def continuous_prediction(x, u, frozen, lever, order=32):
    """Analytic pitch plus fixed Gaussian quadrature, independent intervals."""
    nodes,weights=np.polynomial.legendre.leggauss(order);t=(nodes+1)*.01;weights=weights*.01
    alpha=lever*(u[:,1]-u[:,0])/frozen['Iyy'];th=x[:,4,None]+x[:,5,None]*t+.5*alpha[:,None]*t*t
    ax=u.sum(axis=1)[:,None]/frozen['mass']*np.sin(th)
    az=u.sum(axis=1)[:,None]/frozen['mass']*np.cos(th)-frozen['gravity']
    y=x.copy();y[:,0]+=x[:,1]*.02+ax@(weights*(.02-t));y[:,1]+=ax@weights
    y[:,2]+=x[:,3]*.02+az@(weights*(.02-t));y[:,3]+=az@weights
    y[:,4]+=x[:,5]*.02+.5*alpha*.02**2;y[:,5]+=alpha*.02
    return y


def interval_reconstruction(x,u,previous_theta,frozen,lever,frequency,cached):
    """Diagnostic finite sum for one archived control interval, not a rollout."""
    y=x.copy();old=previous_theta.copy();h=1/frequency
    total=u.sum(1)/frozen['mass'];alpha=lever*(u[:,1]-u[:,0])/frozen['Iyy']
    for _ in range(frequency//50):
        theta=y[:,4].copy();force_theta=old if cached else theta
        y[:,1]+=h*total*np.sin(force_theta)
        y[:,3]+=h*(total*np.cos(force_theta)-frozen['gravity'])
        y[:,5]+=h*alpha*np.cos(theta-force_theta)
        y[:,0]+=h*y[:,1];y[:,2]+=h*y[:,3];y[:,4]+=h*y[:,5]
        old=theta
    return y


def component_metrics(error, scale):
    return {'maximum_absolute_by_state':np.max(np.abs(error),axis=0).tolist(),
            'maximum_scaled_by_state':np.max(np.abs(error)/scale,axis=0).tolist(),
            'rms_by_state':np.sqrt(np.mean(error**2,axis=0)).tolist(),
            'maximum_scaled':float(np.max(np.abs(error)/scale)),
            'transitions_exceeding_0001':int(np.any(np.abs(error)/scale>.001,axis=1).sum())}


def saved_transition_audit(pilot, output):
    pilot,output=Path(pilot),Path(output);f=json.loads((pilot/'frozen.json').read_text());scale=np.array(f['config']['state_scale'])
    all_errors={};labels=[];individual=[];physics_audits=[];worst=[];native_max=quad_max=closure_max=motor_max=0.
    for folder in sorted((pilot/'runs').iterdir()):
        meta=json.loads((folder/'completion.json').read_text())
        if not meta['complete'] or (meta['frequency']==1000 and meta['phase']!='original') or (meta['frequency']==2000 and meta['phase']!='refine'):
            continue
        a=np.load(folder/'trajectory.npz');x=a['states'][:-1];target=a['states'][1:];hz=meta['frequency'];N=hz//50;h=1/hz
        rpm=a['rpms'];u=f['KF']*np.column_stack([rpm[:,0]**2+rpm[:,3]**2,rpm[:,1]**2+rpm[:,2]**2])
        motor_max=max(motor_max,float(np.max(np.abs(u-a['clipped_actions']))))
        p=a['physics_states'];boundary=np.arange(300)*N;previous=p[np.maximum(0,boundary-1),4]
        assert p[::N].tobytes()==a['states'].tobytes()
        assert np.array_equal(a['current_steps'],np.arange(300))
        native=np.array([rk4(xx,uu,f) for xx,uu in zip(x,u)])
        native_max=max(native_max,float(np.max(np.abs(native-a['native_model_predictions']))))
        ode_symbolic=continuous_prediction(x,u,f,f['arm_length']/np.sqrt(2))
        ode_urdf=continuous_prediction(x,u,f,.028)
        quad_max=max(quad_max,float(np.max(np.abs(ode_symbolic-continuous_prediction(x,u,f,f['arm_length']/np.sqrt(2),64)))))
        direct=interval_reconstruction(x,u,previous,f,.028,hz,False)
        cache=interval_reconstruction(x,u,previous,f,.028,hz,True)
        errors={'native_total':native-target,'native_RK4_quadrature':native-ode_symbolic,
            'geometry':ode_symbolic-ode_urdf,'semiimplicit_discretization':ode_urdf-direct,
            'cached_link_timing':direct-cache,'remaining_recurrence_residual':cache-target,
            'geometry_aligned_continuous_error':ode_urdf-target,
            'geometry_and_discretization_error':direct-target}
        closure=errors['native_total']-sum(errors[k] for k in ['native_RK4_quadrature','geometry','semiimplicit_discretization','cached_link_timing','remaining_recurrence_residual'])
        closure_max=max(closure_max,float(np.max(np.abs(closure))))
        for k,v in errors.items():all_errors.setdefault(k,[]).append(v)
        individual.append({'record':folder.name,'frequency':hz,'case':meta['case'],'controller':meta['controller'],
            'transitions':len(x),'components':{k:component_metrics(v,scale) for k,v in errors.items()}})
        labels.extend({'record':folder.name,'frequency':hz,'case':meta['case'],'controller':meta['controller'],'step':i,'time_seconds':i*.02,
                       'initial_state':x[i].tolist(),'executed_thrust':u[i].tolist()} for i in range(len(x)))
        # Check the recurrence directly on archived successive physics samples.
        start=p[:-1];end=p[1:];last_theta=np.r_[start[0,4],start[:-1,4]]
        T=np.repeat(u.sum(1),N);diff=np.repeat(u[:,1]-u[:,0],N)
        accel=np.column_stack([T/f['mass']*np.sin(last_theta),T/f['mass']*np.cos(last_theta)-f['gravity'],
              .028/f['Iyy']*diff*np.cos(start[:,4]-last_theta)])
        observed=(end[:,[1,3,5]]-start[:,[1,3,5]])/h
        pos_err=end[:,[0,2,4]]-start[:,[0,2,4]]-h*end[:,[1,3,5]]
        current=np.column_stack([T/f['mass']*np.sin(start[:,4]),T/f['mass']*np.cos(start[:,4])-f['gravity'],.028/f['Iyy']*diff])
        inferred_angle=np.arctan2(observed[:,0],observed[:,1]+f['gravity'])
        thrust_projection=np.column_stack([T*np.sin(last_theta),T*np.cos(last_theta)])
        design=np.zeros((2*len(T),2));rhs=observed[:,:2].reshape(-1)
        design[0::2,0]=thrust_projection[:,0];design[1::2,0]=thrust_projection[:,1];design[1::2,1]=-1
        inverse_mass,gravity=np.linalg.lstsq(design,rhs,rcond=None)[0]
        lever_inertia=diff*np.cos(start[:,4]-last_theta)
        fitted_ratio=float(np.dot(lever_inertia,observed[:,2])/np.dot(lever_inertia,lever_inertia))
        physics_audits.append({'record':folder.name,'frequency':hz,'physics_intervals':len(start),
            'cached_frame_acceleration_residual_by_velocity':np.max(np.abs(observed-accel),axis=0).tolist(),
            'current_frame_acceleration_residual_by_velocity':np.max(np.abs(observed-current),axis=0).tolist(),
            'semiimplicit_position_residual_by_position':np.max(np.abs(pos_err),axis=0).tolist(),
            'inferred_force_angle_previous_sample_error':float(np.max(np.abs(inferred_angle-last_theta))),
            'inferred_force_angle_current_sample_error':float(np.max(np.abs(inferred_angle-start[:,4]))),
            'mass_fit_kg':float(1/inverse_mass),'gravity_fit_m_s2':float(gravity),
            'lever_over_Iyy_fit':fitted_ratio,'source_lever_over_Iyy':.028/f['Iyy'],
            'coordinate_note':'Pitch is Euler Y; archived angular rate is world Y. Agreement is conditional on these recorded planar cases.'})
    joined={k:np.concatenate(v) for k,v in all_errors.items()};np.savez_compressed(output/'signed_error_components.npz',**joined)
    for component in range(6):
        index=int(np.argmax(np.abs(joined['native_total'][:,component])))
        worst.append({'state_component':f['state_labels'][component],'unit':f['state_units'][component],**labels[index],
            'signed_native_error':float(joined['native_total'][index,component]),
            'signed_contributions':{k:float(v[index,component]) for k,v in joined.items() if k not in ['native_total','geometry_aligned_continuous_error','geometry_and_discretization_error']}})
    result={'state_labels':f['state_labels'],'state_units':f['state_units'],'state_scale':scale.tolist(),
        'records':individual,'physics_records':physics_audits,'control_intervals':len(labels),
        'physics_intervals':sum(x['physics_intervals'] for x in physics_audits),
        'global_components':{k:component_metrics(v,scale) for k,v in joined.items()},'worst_by_component':worst,
        'native_RK4_reconstruction_maximum_error':native_max,'quadrature_32_64_maximum_error':quad_max,
        'additive_decomposition_closure_error':closure_max,'executed_motor_thrust_reconstruction_error':motor_max,
        'symbolic_lever_m':f['arm_length']/np.sqrt(2),'URDF_lever_m':.028,
        'geometry_relative_difference':(f['arm_length']/np.sqrt(2)-.028)/.028,
        'no_new_physics_steps':True,'no_sequential_closed_loop_rollouts':True,
        'interpretation':'Signed attribution is order-dependent; component maxima are not additive causal percentages. The original model/tolerance remains unchanged.'}
    write(output/'transition_diagnosis.json',result)
    return result


def problem_graph(opti):
    import casadi as c
    fn=c.Function('program',[opti.x,opti.p],[opti.f,opti.g,opti.lbg,opti.ubg])
    return hashlib.sha256(fn.serialize().encode()).hexdigest()


def toy_problem(kind,q):
    import casadi as c
    o=c.Opti(kind) if kind else c.Opti();x=o.variable(2)
    H=c.DM([[4,1],[1,2]]);G=c.DM([[-1,0],[0,-1],[1,0],[0,1],[1,1]])
    b=c.DM([0,0,.8,.8,1]);o.minimize(.5*c.mtimes([x.T,H,x])+c.dot(c.DM(q),x));o.subject_to(G@x<=b)
    return o,x


def standalone_solver_audit(output):
    import casadi as c
    start=time.perf_counter();plugins=c.CasadiMeta_plugins().split(';')
    available={'conic_qrqp':c.has_conic('qrqp'),'conic_ipopt':c.has_conic('ipopt'),'nlpsol_ipopt':c.has_nlpsol('ipopt')}
    bad,x=toy_problem('conic',[-5,-3]);bad.solver('ipopt',{'expand':True,'ipopt.max_iter':100})
    try:bad.solve();raise AssertionError('Expected native conic/IPOPT failure')
    except RuntimeError as e:native_failure=str(e)
    cases=[([-5,-3],[F(3,4),F(1,4)],F(-25,8)),([-1,-1],[F(1,7),F(3,7)],F(-2,7)),([2,-3],[F(0),F(4,5)],F(-44,25))]
    results=[]
    for repeat in range(2):
        for q,expected,cost in cases:
            values={}
            for backend in ['adapter_qrqp','reference_nlp_ipopt']:
                opti,x=toy_problem('conic' if backend=='adapter_qrqp' else '',q);before=problem_graph(opti)
                if backend=='adapter_qrqp':dispatch=configure_linear_conic_qrqp_v1(opti,'qrqp','ipopt',{'expand':True,'ipopt.max_iter':100})
                else:opti.solver('ipopt',{'expand':True,'ipopt.max_iter':100});dispatch=None
                after=problem_graph(opti);assert before==after
                sol=opti.solve();z=np.array(sol.value(x));expected_np=np.array([float(v) for v in expected]);err=float(np.max(np.abs(z-expected_np)))
                assert err<1e-6
                G=np.array([[-1,0],[0,-1],[1,0],[0,1],[1,1]]);b=np.array([0,0,.8,.8,1]);lam=np.array(sol.value(opti.lam_g))
                grad=np.array([[4,1],[1,2]])@z+np.array(q)
                values[backend]={'x':z.tolist(),'objective':float(sol.value(opti.f)),'solution_error':err,
                    'objective_error':abs(float(sol.value(opti.f))-float(cost)),
                    'primal_excess':float(max(0,np.max(G@z-b))),'stationarity_maximum':float(np.max(np.abs(grad+G.T@lam))),
                    'complementarity_maximum':float(np.max(np.abs(lam*(G@z-b)))),
                    'solver_success':bool(opti.stats()['success']),'status':opti.stats()['return_status'],
                    'iterations':int(opti.stats()['iter_count']),'program_graph_unchanged':before==after,'dispatch':dispatch}
            results.append({'repeat':repeat,'q':q,'exact_solution':[str(v) for v in expected],'exact_objective':str(cost),
                            'solver_comparison_maximum':float(np.max(np.abs(np.array(values['adapter_qrqp']['x'])-values['reference_nlp_ipopt']['x']))),'solvers':values})
    impossible=c.Opti('conic');z=impossible.variable();impossible.minimize(z*z);impossible.subject_to(z>=2);impossible.subject_to(z<=1)
    configure_linear_conic_qrqp_v1(impossible,'qrqp','ipopt',{'expand':True,'ipopt.max_iter':100})
    try:impossible.solve();raise AssertionError('Infeasible toy problem accepted')
    except RuntimeError as e:infeasible={'correctly_rejected':True,'status':impossible.return_status(),'message':str(e)}
    out={'casadi_version':c.__version__,'registered_plugins':plugins,'availability':available,'native_conic_failure':native_failure,
         'toy_cases':results,'infeasible_control':infeasible,'seconds':time.perf_counter()-start,
         'scope':'Only two-variable QPs and one scalar infeasible QP; no controller, environment, physics or episode instantiated'}
    write(Path(output)/'solver_diagnosis.json',out);return out
