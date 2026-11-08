"""Scoped integration of the accepted QRQP v1 adapter; old sources stay intact."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import hashlib
import resource
import numpy as np

from src import quadrotor_pilot as pilot
from src.quadrotor_compatibility_v1 import configure_linear_conic_qrqp_v1, ADAPTER_VERSION
from src.quadrotor_diagnostic import problem_graph
from src.quadrotor_linear_verification import independent_program,parameter_terms,kkt_metrics,require_valid_solution


def prepare_revalidation(root,upstream,out):
    """Freeze inputs only after checking the two accepted E12 evidence seals."""
    root,out,upstream=Path(root),Path(out),Path(upstream);out.mkdir(exist_ok=True,parents=True)
    if (out/'preservation_start.json').exists():return pilot.read(out/'preservation_start.json')
    pilot_root=root/'results/benchmark_selection/technical_pilot'
    diag_root=root/'results/benchmark_selection/compatibility_diagnostic';markers={}
    for folder,marker in [(pilot_root,'review_integrity.json'),(diag_root,'completion.json')]:
        seal=pilot.read(folder/marker);markers[str(folder/marker)]=pilot.sha(folder/marker)
        for name,h in seal['artifact_sha256'].items():assert pilot.sha(folder/name)==h,name
        for name,h in seal['source_sha256'].items():assert pilot.sha(root/name)==h,name
    old=pilot.read(pilot_root/'preservation_before.json');f=pilot.read(pilot_root/'frozen.json')
    for name,h in old['files'].items():assert pilot.sha(root/name)==h,name
    for name,h in old['authority_files'].items():assert pilot.sha(name)==h,name
    for name,h in f['upstream_files'].items():assert pilot.sha(upstream/name)==h,name
    record={'accepted_markers':markers,'historical_files':len(old['files']),'authority_files':len(old['authority_files']),
            'upstream_files':len(f['upstream_files'])}
    for name in ['frozen.json','reference.npy','environment.lock.txt']:
        if (out/name).exists():assert pilot.sha(out/name)==pilot.sha(pilot_root/name)
        else:(out/name).write_bytes((pilot_root/name).read_bytes())
    pilot.write(out/'revalidation_protocol.json',{'adapter':'linear_conic_qrqp_v1','output_label':'linear40_qrqp_v1',
        'cases':['T0','T1','T2','T3'],'frequency':1000,'horizon':40,'repetitions':['original','repeat'],
        'native_QRQP_primal_dual_tolerance':1e-8,'KKT_roundoff_allowance':1e-10,'graph_reconstruction_tolerance':1e-10,
        'original_symbolic_model_gate':'FAILED; threshold remains 0.001','run_nonlinear':False,'run_screening':False,
        'run_action_probes':False,'train_RL':False})
    pilot.write(out/'preservation_start.json',record);return record


def finite_error(a,b):
    a,b=np.asarray(a),np.asarray(b)
    assert a.shape==b.shape and np.array_equal(np.isposinf(a),np.isposinf(b)) and np.array_equal(np.isneginf(a),np.isneginf(b))
    good=np.isfinite(a)&np.isfinite(b)
    return float(np.max(np.abs(a[good]-b[good]),initial=0))


def inspect_graph(opti,controller,f,folder,before,dispatch):
    import casadi as c
    folder=Path(folder);folder.mkdir(exist_ok=True,parents=True)
    after=problem_graph(opti);assert before==after
    assert controller.T==40 and not controller.soft_constraints and not controller.use_lqr_gain_and_terminal_cost
    assert controller.constraint_tol==1e-8 and controller.warmstart
    assert np.array_equal(controller.X_EQ,np.zeros(6))
    assert np.array_equal(controller.U_EQ,np.array(f['hover_input']))
    assert np.array_equal(controller.Q,np.array(f['Q'])) and np.array_equal(controller.R,np.array(f['R']))
    H=c.hessian(opti.f,opti.x)[0];A=c.jacobian(opti.g,opti.x);grad=c.gradient(opti.f,opti.x)
    # MX derivative call nodes retain structural input dependencies even for a
    # constant derivative. Expand exactly into SX before testing constancy.
    expanded=c.Function('derivative_structure',[opti.x,opti.p],[H,A]).expand()
    sx=c.SX.sym('z',opti.nx);sp=c.SX.sym('p',opti.np);hs,am=expanded(sx,sp)
    assert not c.depends_on(hs,sx) and not c.depends_on(hs,sp)
    assert not c.depends_on(am,sx) and not c.depends_on(am,sp)
    H=c.DM(hs);A=c.DM(am)
    zero=c.DM.zeros(opti.nx);g0=c.substitute(opti.g,opti.x,zero)
    q=c.substitute(grad,opti.x,zero);constant=c.substitute(opti.f,opti.x,zero)
    coefficients=c.Function('coefficients',[opti.p],[H,A,q,opti.lbg-g0,opti.ubg-g0,constant])
    terms=c.Function('terms',[opti.p],[q,opti.lbg-g0,opti.ubg-g0,constant])
    expected=independent_program(f);eq=expected['equality_rows']
    assert opti.nx==326 and opti.np==252 and opti.ng==898
    sample=np.r_[f['cases']['T0'],np.zeros(246)]
    values=coefficients(sample);actual_H=np.array(values[0]);actual_A=np.array(values[1])
    errors={'H':finite_error(actual_H,np.diag(expected['Hdiag'])),'A':finite_error(actual_A,expected['A'].toarray())}
    for case in ['T0','T1','T2','T3']:
        for step in [0,149,299]:
            # The caller provides the frozen full reference on the controller.
            ref=controller.traj[:,np.minimum(step+np.arange(41),300)]
            params=np.r_[f['cases'][case],ref.reshape(-1,order='F')]
            native=terms(params);wanted=parameter_terms(expected,f,params)
            for name,x,y in zip(['q','lower','upper','constant'],native,wanted):
                errors[name]=max(errors.get(name,0),finite_error(np.array(x).reshape(-1),np.asarray(y).reshape(-1)))
    assert max(errors.values())<1e-10,errors
    eig=np.linalg.eigvalsh(actual_H);rank=int(np.linalg.matrix_rank(actual_A[eq]))
    assert eig.min()>0 and rank==246
    np.savez_compressed(folder/'native_graph_coefficients.npz',H=actual_H,A=actual_A)
    (folder/'native_program.casadi').write_text(c.Function('program',[opti.x,opti.p],[opti.f,opti.g,opti.lbg,opti.ubg]).serialize())
    report={'adapter_version':ADAPTER_VERSION,'graph_sha256_before':before,'graph_sha256_after':after,
            'program_graph_unchanged':True,'variables':opti.nx,'parameters':opti.np,'constraint_rows':opti.ng,
            'equality_rows':246,'equality_rank':rank,'H_min_eigenvalue':float(eig.min()),'H_max_eigenvalue':float(eig.max()),
            'H_condition_number':float(eig.max()/eig.min()),'independent_reconstruction_errors':errors,
            'horizon':controller.T,'constraint_tol':controller.constraint_tol,'warmstart':controller.warmstart,'dispatch':dispatch,
            'interpretation':'SPD objective and full-rank shooting equalities; numerical KKT checks support QP optimality, not plant safety.'}
    pilot.write(folder/'graph_audit.json',report)
    return expected,terms,report


def create_adapted_controller(upstream,cfg,environment_factory,f,graph_folder):
    import casadi as c
    real_solver=c.Opti.solver;captured=[]
    def intercept(opti,backend,options,*args):
        assert not args
        before=problem_graph(opti)
        # Proxy exposes only the original solver-selection method. The accepted
        # adapter cannot touch any expression or initial guess through this proxy.
        proxy=SimpleNamespace(solver=lambda requested,opts:real_solver(opti,requested,opts))
        dispatch=configure_linear_conic_qrqp_v1(proxy,'qrqp',backend,options)
        captured.append((opti,before,dispatch))
    with patch.object(c.Opti,'solver',intercept):
        ctrl=pilot.make_controller(upstream,cfg,environment_factory,'linear40')
    assert len(captured)==1
    opti,before,dispatch=captured[0];assert opti is ctrl.opti_dict['opti']
    program,terms,report=inspect_graph(opti,ctrl,f,graph_folder,before,dispatch)
    return ctrl,program,terms,report


def preflight(upstream,out):
    out=Path(out);f=pilot.read(out/'frozen.json');pilot.activate(upstream,f['config'])
    factory=lambda:pilot.make_environment(upstream,f['config'],np.array(f['cases']['T0']),1000)
    ctrl,_,_,report=create_adapted_controller(upstream,f['config'],factory,f,out/'preflight')
    ctrl.close();pilot.write(out/'preflight.json',report)
    return report


def episode(upstream,out,case,phase):
    import casadi as c
    out=Path(out);f=pilot.read(out/'frozen.json');folder=out/'runs'/f'{case}_linear40_1000_{phase}'
    if (folder/'revalidation_complete.json').exists():return pilot.read(folder/'completion.json')
    if folder.exists():raise RuntimeError('Refusing to overwrite partial revalidation '+str(folder))
    real_factory=pilot.make_controller;real_solve=c.Opti.solve;active={};solutions=[];solve_events=[]
    def factory(upstream_arg,cfg,env_factory,key):
        assert key=='linear40'
        # Avoid recursively calling this scoped factory from the reusable creator.
        with patch.object(pilot,'make_controller',real_factory):
            ctrl,program,terms,report=create_adapted_controller(upstream_arg,cfg,env_factory,f,out/'graphs'/f'{case}_{phase}')
        active.update(ctrl=ctrl,program=program,terms=terms)
        return ctrl
    def solve(opti,*args,**kwargs):
        if opti is not active['ctrl'].opti_dict['opti']:return real_solve(opti,*args,**kwargs)
        try:
            sol=real_solve(opti,*args,**kwargs);stats=opti.stats()
            p=np.asarray(sol.value(opti.p)).reshape(-1);z=np.asarray(sol.value(opti.x)).reshape(-1)
            lam=np.asarray(sol.value(opti.lam_g)).reshape(-1);objective=float(sol.value(opti.f))
            metrics=kkt_metrics(active['program'],f,p,z,lam)
            require_valid_solution(bool(stats.get('success')),bool(np.isfinite(np.r_[p,z,lam,objective]).all()),metrics)
            native=active['terms'](p);expected=parameter_terms(active['program'],f,p)
            reconstruction=max(finite_error(np.array(x).reshape(-1),np.asarray(y).reshape(-1)) for x,y in zip(native,expected))
            assert reconstruction<1e-10
            assert abs(metrics['independent_objective']-objective)<1e-10
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss < f['config']['maximum_memory_bytes']
            solutions.append((p,z,lam,objective))
            solve_events.append({'success':True,'status':stats['return_status'],'iterations':int(stats['iter_count']),
                                 'metrics':metrics,'coefficient_reconstruction_error':reconstruction})
            return sol
        except Exception as exc:
            solve_events.append({'success':False,'kind':type(exc).__name__,'message':str(exc)})
            raise
    with patch.object(pilot,'make_controller',factory),patch.object(c.Opti,'solve',solve):
        metadata=pilot.run_episode(Path.cwd(),upstream,out,case,'linear40',phase,1000)
    if solutions:
        p,z,lam,objective=map(np.array,zip(*solutions))
    else:p,z,lam,objective=np.empty((0,252)),np.empty((0,326)),np.empty((0,898)),np.empty(0)
    np.savez_compressed(folder/'kkt_solutions.npz',parameters=p,primal=z,dual=lam,objective=objective)
    metadata.update(controller_variant='linear40_qrqp_v1',adapter_version=ADAPTER_VERSION,
                    kkt_solutions_sha256=pilot.sha(folder/'kkt_solutions.npz'))
    pilot.write(folder/'completion.json',metadata);pilot.write(folder/'solve_events.json',solve_events)
    pilot.write(folder/'revalidation_complete.json',{'completion_sha256':pilot.sha(folder/'completion.json'),'completed':metadata['complete']})
    return metadata


def replay_executed_motor_commands(upstream,out,folder):
    """Use recorded RPMs directly; independently check native conversion too."""
    out,folder=Path(out),Path(folder);f=pilot.read(out/'frozen.json');m=pilot.read(folder/'completion.json')
    pilot.activate(upstream,f['config']);a=np.load(folder/'trajectory.npz')
    env=pilot.make_environment(upstream,f['config'],np.array(f['cases'][m['case']]),1000)
    states=[];rewards=[];index=0;original=env._preprocess_control;motor_equal=[]
    def recorded(action):
        nonlocal index
        candidate=original(action);rpm=a['rpms'][index].copy();motor_equal.append(candidate.tobytes()==rpm.tobytes())
        assert motor_equal[-1];index+=1;return rpm
    try:
        env.reset();states.append(pilot.raw_state(env));env._preprocess_control=recorded
        with pilot.capture_physics(env) as physics:
            for requested in a['actions'][:m['transitions']]:
                _,reward,_,_=env.step(requested);states.append(pilot.raw_state(env));rewards.append(reward)
        result={'control_states_bitwise':np.asarray(states).tobytes()==a['states'].tobytes(),
                'rewards_bitwise':np.asarray(rewards).tobytes()==a['rewards'].tobytes(),
                'physics_states_bitwise':np.asarray(physics['states']).tobytes()==a['physics_states'].tobytes(),
                'contacts_bitwise':np.asarray(physics['contacts']).tobytes()==a['physics_contacts'].tobytes(),
                'executed_motor_inputs_bitwise':all(motor_equal),'instrumented':True,'steps':len(rewards),
                'mode':'Recorded four-motor RPMs drive physics; requested commands reconstruct native cost bookkeeping only'}
        assert all(result[k] for k in ['control_states_bitwise','rewards_bitwise','physics_states_bitwise','contacts_bitwise','executed_motor_inputs_bitwise'])
        pilot.write(folder/'replay_instrumented.json',result);return result
    finally:env.close()
