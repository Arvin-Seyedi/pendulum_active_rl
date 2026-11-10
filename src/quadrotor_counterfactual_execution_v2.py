"""Execution wiring for Pilot B v2; scientific execution PAUSED pending fresh approval.

Accepted preflight/controller/plant sources stay immutable. Every native query
is independently guarded, then archive parity precedes detached substitution.
No executable acquisition/menu-search mode exists.
"""
from contextlib import contextmanager,ExitStack
from pathlib import Path
from unittest.mock import patch
import copy,json,resource,time,sys,platform,importlib.metadata,subprocess
import numpy as np
from src import quadrotor_pilot as p
from src import quadrotor_mpc80_execution_v2 as live
from src import quadrotor_mpc80_execution_v1 as checks
from src import quadrotor_counterfactual_v1 as cf
from src.quadrotor_counterfactual_context_v1 import prepare_selector
from src.quadrotor_h2_requalification_v2 import IndependentCandidateGuard
from src.quadrotor_qualification_telemetry_v2 import atomic_arrays
from src.quadrotor_mmaq_verification_v1 import independent_nlp_program
from src.quadrotor_mismatch_contract_v1 import source_prediction
from src.quadrotor_mmaq_v1 import safe_write

PREF=Path('results/benchmark_selection/counterfactual_v1_preflight')
ACCEPTED=Path('results/benchmark_selection/mpc80_feas_v2_study')
FILES=['src/quadrotor_counterfactual_execution_v2.py','src/quadrotor_counterfactual_dispatch_v2.py','experiments/12_counterfactual_pilot_b_v2.py','tests/test_quadrotor_counterfactual_execution_v2.py',
       'configs/experiment_12_counterfactual_pilot_b_v2.json','docs/experiment_12_counterfactual_pilot_b_v2_protocol.md']
CONTEXT_KEYS=['base_position','base_quaternion','base_linear_velocity','base_angular_velocity',
              'joint_states','contacts','native_cached_arrays','control_counter','physics_counter','environment_RNG','numpy_RNG']
COMPARE_FIELDS=['states','actions','clipped_actions','rpms','rewards','dones','physics_states','physics_contacts',
    'predicted_states','predicted_inputs','native_model_predictions','current_steps','solver_success','solver_iterations',
    'warm_u_error','warm_x_error']


def checkpoint(root,upstream,out=None):
    root=Path(root);pref=root/PREF;seal=p.read(pref/'final_protocol_seal.json')
    assert p.sha(pref/'final_protocol_seal.json')=='facb3a19264f111290c33982bfd959f237bf6bae60ad0ddaee470206ecec934d'
    for n,h in seal['source_sha256'].items():assert p.sha(root/n)==h,n
    for n,h in seal['artifact_sha256'].items():assert p.sha(pref/n)==h,n
    live.checkpoint(root,upstream,root/'results/benchmark_selection/mpc80_feas_v2_preflight',root/ACCEPTED)
    runtime_identity(root,upstream)
    assert seal['static_gate_passed'] and seal['tests_passed']==154
    if out is not None:
        out=Path(out);binding=p.read(out/'execution_binding.json')
        for n,h in binding['source_sha256'].items():assert p.sha(root/n)==h,n
        for n,h in binding['artifact_sha256'].items():assert p.sha(out/n)==h,n


def runtime_identity(root,upstream):
    model=p.read(Path(root)/ACCEPTED/'frozen.json');ident=p.read(Path(root)/'results/benchmark_selection/mpc80_feas_v2_preflight/native_dependency_identity.json')
    if sys.version!=model['python'] or platform.platform()!=model['platform'] or p.sha(sys.executable)!=ident['python_executable_sha256']:
        raise RuntimeError('Pinned interpreter/platform identity mismatch')
    for name,item in ident['package_RECORD_identities'].items():
        dist=importlib.metadata.distribution(name);record=next(x for x in dist.files if str(x).endswith('.dist-info/RECORD'))
        if p.sha(dist.locate_file(record))!=item['RECORD_sha256']:raise RuntimeError('Pinned package RECORD differs: '+name)
    if subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=upstream,text=True).strip():
        raise RuntimeError('Pinned checkout not clean')
    return {'passed':True,'interpreter_sha256':ident['python_executable_sha256'],'package_RECORDs':len(ident['package_RECORD_identities'])}


def compare_archived_nominal(folder,archive):
    a=np.load(Path(folder)/'trajectory.npz');b=np.load(Path(archive)/'trajectory.npz')
    equal={n:a[n].tobytes()==b[n].tobytes() for n in COMPARE_FIELDS}
    if not all(equal.values()):raise ValueError('Nominal scientific arrays not bitwise: '+repr(equal))
    for k in range(300):
        x=np.load(Path(folder)/'attempts'/f'{k:04d}'/'candidate.npz');y=np.load(Path(archive)/'attempts'/f'{k:04d}'/'candidate.npz')
        for n in ['primal','parameters','dual']:
            if x[n].tobytes()!=y[n].tobytes():raise ValueError('Nominal archived candidate not bitwise')
    return {'arrays_bitwise':equal,'all_300_candidate_vectors_bitwise':True}


def context_equal(left,right):
    """Used numerical reset context; unused Python RNG/internal error text is metadata."""
    return all(exact_context(left[k],right[k]) for k in CONTEXT_KEYS)


def verify_first_submissions(rows,model,menu,root):
    if menu!=3:return {'alternative':False}
    expected=p.read(Path(root)/PREF/'frozen_alternative.json')['commands']
    for chunk,field in [(slice(3,5),'requested'),(slice(7,9),'clipped'),(slice(13,17),'rpm'),
                        (slice(17,29),'forces'),(slice(41,44),'torque')]:
        values=np.asarray(expected[field],dtype=np.float64).reshape(-1)
        if rows[:20,chunk].astype('<f8').tobytes()!=np.repeat(values[None,:],20,axis=0).astype('<f8').tobytes():
            raise ValueError('Frozen alternative submission differs: '+field)
    forces=rows[0,17:29].reshape(4,3);grouped=np.array([forces[0,2]+forces[3,2],forces[1,2]+forces[2,2]])
    if grouped.tobytes()!=np.asarray(expected['submitted_grouped_thrust']).tobytes():raise ValueError('Grouped submitted thrust differs')
    baseline=np.fromfile(Path(root)/ACCEPTED/'runs/0_nonlinear80_1000_original/submitted_scientific.f64',dtype='<f8').reshape(-1,56)[0,17:29].reshape(4,3)
    original=np.array([baseline[0,2]+baseline[3,2],baseline[1,2]+baseline[2,2]])
    separation=cf.separation(grouped,original,model)
    if not separation['at_least_five_percent']:raise ValueError('Original 5-percent separation failed')
    return {'alternative':True,'twenty_first_interval_submissions_match_frozen':True,
        'direct_submitted_grouped_thrust':grouped.tolist(),'normalized_separation':separation,
        'telemetry_scope':'native API arguments, not sensor outputs'}



def exact_context(a,b):
    if isinstance(a,(float,np.floating)) or isinstance(b,(float,np.floating)):
        return isinstance(a,(float,np.floating)) and isinstance(b,(float,np.floating)) and np.isfinite(a) and np.isfinite(b) and np.float64(a).tobytes()==np.float64(b).tobytes()
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(exact_context(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return len(a)==len(b) and all(exact_context(x,y) for x,y in zip(a,b))
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):return cf.same(a,b) and np.isfinite(np.asarray(a)).all()
    return type(a)==type(b) and a==b


def programs(ctrl,model):
    import casadi as c
    from src.quadrotor_diagnostic import problem_graph
    own,_=independent_nlp_program(ctrl,model);opti=ctrl.opti_dict['opti']
    return own,problem_graph(opti),c.gradient(opti.f+c.dot(opti.g,opti.lam_g),opti.x)


def check_submission_row(row,model,menu,root):
    """Called after command journaling and BEFORE physics integration."""
    checks.require_executed(row[13:17],model)
    force=row[17:29].reshape(4,3);grouped=np.array([force[0,2]+force[3,2],force[1,2]+force[2,2]])
    if np.any(grouped<model['action_low']) or np.any(grouped>model['action_high']):raise ValueError('Negative direct submitted-input margin')
    if menu==3 and row[0]<20:
        expected=p.read(Path(root)/PREF/'frozen_alternative.json')['commands']
        for part,name in [(slice(3,5),'requested'),(slice(7,9),'clipped'),(slice(13,17),'rpm'),(slice(17,29),'forces'),(slice(41,44),'torque')]:
            if row[part].astype('<f8').tobytes()!=np.asarray(expected[name],dtype='<f8').reshape(-1).tobytes():raise ValueError('Pre-physics frozen command mismatch: '+name)
        baseline=np.fromfile(Path(root)/ACCEPTED/'runs/0_nonlinear80_1000_original/submitted_scientific.f64',dtype='<f8',count=56)[17:29].reshape(4,3)
        original=np.array([baseline[0,2]+baseline[3,2],baseline[1,2]+baseline[2,2]])
        if not cf.separation(grouped,original,model)['at_least_five_percent']:raise ValueError('Pre-physics action separation failed')


@contextmanager
def frozen_stream_scope(model,menu,root):
    parent=live.SubmissionStream
    class BoundaryStream(parent):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.boundary_failures=set()
        def emit(self,packet):
            super().emit(packet) # retain submissions before a check may reject them
            idx=packet['application_index']
            if packet['force_application_returned'] and not packet['physics_returned'] and idx not in self.boundary_failures:
                try:check_submission_row(live.flatten_packet(packet),model,menu,root)
                except BaseException as exc:
                    self.boundary_failures.add(idx);self.errors+=1
                    safe_write(self.folder/(self.label+f'_boundary_failure_{idx}.json'),{'error':repr(exc),'packet':packet,'checked_before_physics':True})
                    raise
    with patch.object(live,'SubmissionStream',BoundaryStream):yield


def episode(root,upstream,out,menu,phase):
    import casadi as c
    import pybullet as bullet
    from src.quadrotor_diagnostic import problem_graph
    root,out=Path(root),Path(out);branch=out/'branches'/f'menu{menu}';model=p.read(branch/'frozen.json')
    folder=branch/'runs'/f'0_nonlinear80_1000_{phase}';archive=root/ACCEPTED/'runs/0_nonlinear80_1000_original'
    if folder.exists():raise FileExistsError('Partial/completed attempt retained; no retry')
    old=np.load(archive/'trajectory.npz');native_factory=p.make_controller;native_env_factory=p.make_environment;hooks=ExitStack()
    active={};captured={'native_capture':p.capture_physics};guards=[];predictions=[];applied=[];calls=0
    expected_states=expected_commands=None
    if menu==4:
        expected_states=old['physics_states'];expected_commands=np.fromfile(archive/'submitted_scientific.f64',dtype='<f8').reshape(-1,56)
    elif phase=='repeat':
        original=branch/'runs/0_nonlinear80_1000_original';expected_states=np.load(original/'trajectory.npz')['physics_states']
        expected_commands=np.fromfile(original/'submitted_scientific.f64',dtype='<f8').reshape(-1,56)
    @contextmanager
    def capture(env):
        with frozen_stream_scope(model,menu,root):
            with live.checked_capture(env,folder,'submitted_scientific',model,captured,predictions,expected_states,expected_commands) as data:yield data
    def env_factory(*args,**kwargs):
        env=native_env_factory(*args,**kwargs)
        if 'plant' not in active:
            active['plant']=env;native_reset=env.reset
            def reset(*args,**kwargs):
                value=native_reset(*args,**kwargs);x=p.raw_state(env)
                snap=cf.passive_reset_state(env,bullet)
                safe_write(folder/'reset_context.json',snap)
                if x.tobytes()!=old['states'][0].tobytes() or np.max(np.abs(x-model['cases']['0']))>1e-14:raise ValueError('Exact native reset mismatch')
                if value[1]['current_step']!=0 or env.ctrl_step_counter!=0 or np.any(env.last_clipped_action!=0):raise ValueError('Initial plant context mismatch')
                baseline=out/'branches/menu4/runs/0_nonlinear80_1000_original/reset_context.json'
                if baseline.exists() and baseline.resolve()!=(folder/'reset_context.json').resolve():
                    # JSON round-trip makes tuples lists; compare equally represented copies.
                    if not context_equal(p.read(baseline),p.read(folder/'reset_context.json')):raise ValueError('Numerical reset context differs across branches/repeats')
                return value
            env.reset=reset
        return env
    def factory(*args,**kwargs):
        ctrl=native_factory(*args,**kwargs);native_select=ctrl.select_action;opti=ctrl.opti_dict['opti']
        own,graph,gradient=programs(ctrl,model)
        assert graph==p.read(root/'results/benchmark_selection/mpc80_feas_v1_preflight/preflight.json')['graph']['native_program_sha256']
        assert ctrl.T==80 and ctrl.warmstart and ctrl.solver=='ipopt' and ctrl.constraint_tol==1e-6
        assert not ctrl.soft_constraints and not ctrl.use_lqr_gain_and_terminal_cost and ctrl.x_prev is None and ctrl.u_prev is None
        active.update(ctrl=ctrl,opti=opti,gradient=gradient)
        original_instance_solve=opti.solve
        hooks.enter_context(patch.object(opti,'solve',lambda *a,**kw:solve(opti,original_instance_solve,*a,**kw)))
        def opti_state():
            r={'initial_guess':np.asarray(opti.debug.value(opti.x,opti.initial())).copy()}
            try:r['parameters']=np.asarray(opti.debug.value(opti.p)).copy();r['parameters_available']=True
            except RuntimeError:r['parameters']=None;r['parameters_available']=False
            return r
        def query(obs,info):
            g=active['guard']
            return g.select(lambda:native_select(obs,info),lambda:g.current_candidate,
                lambda:{'physics_states':np.asarray(captured['data']['states']),'physics_contacts':np.asarray(captured['data']['contacts']),
                    'verified_actions':np.asarray(applied),'applied_full_intervals':np.array((len(captured['data']['states'])-1)//20)})
        def contract(request,k):
            data=captured['data'];x=np.asarray(data['states'][-1]).copy();cache=data['states'][-2][4] if len(data['states'])>1 else x[4]
            b=cf.candidate_contract(x,request,cache,model)
            record={'step':k,'initial':x,'cache':cache,'requested':request.copy(),
                'source_prediction':source_prediction(x,b['executed_thrust'],cache,model),**b}
            predictions.append(record);safe_write(folder/'causal_contract.json',predictions);return record
        def sink(record):safe_write(folder/'intervention'/f"{record['step']:04d}.json",record)
        km=active['plant'].KM
        selector=prepare_selector(ctrl,query,lambda:{'candidate':active['guard'].current_candidate,'warmstart':active['warmstart']},
            archive,old,model,km,menu,opti_state=opti_state,contract=contract,sink=sink)
        def select(obs,info):
            begun=time.perf_counter();k=len(guards);data=captured['data'];x=np.asarray(data['states'][-1]).copy();cache=data['states'][-2][4] if len(data['states'])>1 else x[4]
            guard=IndependentCandidateGuard(folder/'attempts'/f'{k:04d}',{'state_id':0,'menu_index':menu,'phase':phase,
                'control_step':k,'native_reference_step':info['current_step'],'current_raw_state':x,'cache':cache,
                'program_sha256':graph,'controller_observation':np.asarray(obs),'initial_memory_empty':ctrl.u_prev is None},readout=own)
            guards.append(guard);active['guard']=guard
            active['previous_x']=np.empty(0) if ctrl.x_prev is None else ctrl.x_prev.copy()
            active['previous_u']=np.empty(0) if ctrl.u_prev is None else ctrl.u_prev.copy()
            try:
                action=selector.select(obs,info);checks.resource_check();applied.append(action.copy());return action
            except BaseException as exc:
                guard.state.update(status='failed_or_incomplete',supervision_exception=repr(exc))
                safe_write(folder/'supervision_failure.json',{'kind':type(exc).__name__,'message':str(exc),'control_step':k,
                    'native_solve_attempts':calls,'physics_steps':len(data['states'])-1,'fallback_action_applied':False});raise
            finally:
                guard.state['timings']=guard.timing.records;guard.state['timing_summary']=guard.timing.summary()
                guard.state['observed_full_wrapper_wall_at_checkpoint']=time.perf_counter()-begun
                safe_write(guard.folder/'status.json',guard.state)
        ctrl.select_action=select;return ctrl
    def solve(opti,original_instance_solve,*args,**kwargs):
        nonlocal calls
        if opti is not active.get('opti') or calls>=300:raise RuntimeError('Unexpected or excessive solver call')
        guard=active['guard']
        with guard.timing.span('resource_accounting'):
            calls+=1;safe_write(folder/'progress.json',{'solver_attempts':calls,'physics_steps':len(captured['data']['states'])-1})
        with guard.timing.span('warm_start_capture'):
            active['warmstart']={'initial_guess_not_solver_candidate':np.asarray(opti.debug.value(opti.x,opti.initial())),
                'previous_x':active['previous_x'],'previous_u':active['previous_u']}
            atomic_arrays(guard.folder/'warmstart.npz',active['warmstart'])
        return guard.native_solve_hook(opti,lambda:original_instance_solve(*args,**kwargs),active['gradient'])
    try:
        with hooks,patch.object(p,'make_environment',env_factory),patch.object(p,'make_controller',factory),patch.object(p,'capture_physics',capture),patch.object(p,'write',safe_write):
            result=p.run_episode(root,upstream,branch,'0','nonlinear80',phase,1000)
    except BaseException as exc:
        folder.mkdir(parents=True,exist_ok=True)
        safe_write(folder/'startup_or_execution_failure.json',{'kind':type(exc).__name__,'message':str(exc),
            'native_solve_attempts':calls,'physics_steps':max(0,len(captured.get('data',{}).get('states',[]))-1),
            'fallback_action_applied':False,'complete':False})
        if captured.get('data'):
            atomic_arrays(folder/'exception_partial_trajectory.npz',{'states':np.asarray(captured['data']['states']),
                'contacts':np.asarray(captured['data']['contacts']),'available_actions':np.asarray(applied)})
        raise
    result.update(version='E12_MPC80_COUNTERFACTUAL_PILOT_B_v2',state_id=0,menu_index=menu,
        actual_native_solve_attempts=calls,actual_physics_steps=max(0,len(captured.get('data',{}).get('states',[]))-1))
    safe_write(folder/'completion.json',result)
    if not result['complete']:raise RuntimeError('Incomplete scientific run; first-failure stop')
    rows=live.verify_submissions(folder,'submitted_scientific',np.load(folder/'trajectory.npz'))
    submission=verify_first_submissions(rows,model,menu,root)
    if menu==4:safe_write(folder/'archived_nominal_fidelity.json',compare_archived_nominal(folder,archive))
    check=checks.verify_episode(branch,folder,compare_repeat=True)
    check.update(menu_index=menu,submission=submission,native_post_query_memory_preserved=True)
    safe_write(folder/'independent_episode_check.json',check)
    return check


def replay(upstream,out,menu,phase,kind):
    with frozen_stream_scope(p.read(Path(out)/'branches'/f'menu{menu}'/'frozen.json'),menu,Path(__file__).resolve().parents[1]):
        return live.replay(upstream,Path(out)/'branches'/f'menu{menu}',0,phase,kind)
