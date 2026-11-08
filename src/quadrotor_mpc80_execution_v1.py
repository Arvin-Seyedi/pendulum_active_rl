"""Execution wiring for the reviewed E12_MPC80_FEAS_v1 protocol.

The accepted preflight and controller sources are immutable. This module adds
only dispatch accounting, observation/capture, unchanged v2 verification, and a
stop-on-first-failure supervisor. No controller fallback is introduced.
"""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import copy
import hashlib
import importlib.metadata
import json
import resource
import subprocess
import time
import numpy as np
from src import quadrotor_pilot as p
from src import quadrotor_mpc80_feasibility_v1 as f
from src.quadrotor_mmaq_v1 import safe_write, immutable
from src.quadrotor_h2_requalification_v2 import IndependentCandidateGuard
from src.quadrotor_mmaq_verification_v1 import independent_nlp_program, exact_majorant
from src.quadrotor_mismatch_contract_v1 import bound_components, source_prediction
from src.quadrotor_verification import rk4, audit_record
from src.quadrotor_kkt_integrity_v2 import nlp_readout, canonical_kkt, require_kkt, matching_program_values
from src.quadrotor_qualification_telemetry_v2 import atomic_arrays
from src.quadrotor_finalization import physics_reconstruction
from fractions import Fraction

SOURCES=['src/quadrotor_mpc80_execution_v1.py','experiments/12_mpc80_feasibility_study.py',
         'tests/test_quadrotor_mpc80_execution_v1.py']


def require_raw_state(x,contacts,model):
    x=np.asarray(x)
    if x.shape!=(6,) or not np.isfinite(x).all():raise ValueError('Nonfinite/malformed raw state')
    lower=x-model['state_low'];upper=np.asarray(model['state_high'])-x
    if np.any(lower<0) or np.any(upper<0):raise ValueError('Negative raw state safety margin')
    if contacts:raise ValueError('Observed contact')
    return lower,upper


def require_executed(rpm,model):
    rpm=np.asarray(rpm)
    if rpm.shape!=(4,) or not np.isfinite(rpm).all():raise ValueError('Nonfinite/malformed executed RPM')
    u=model['KF']*np.array([rpm[0]**2+rpm[3]**2,rpm[1]**2+rpm[2]**2])
    if np.any(u<model['action_low']) or np.any(u>model['action_high']):raise ValueError('Negative executed input margin')
    return u


def checkpoint(root,upstream,pref,out=None):
    """Exact source/configuration checks immediately before each dispatch."""
    root,upstream,pref=map(Path,(root,upstream,pref));f.check_seal(root,pref)
    config=p.read(pref/'protocol.json');states,_=f.validate_cohort(root,config)
    contract=p.read(pref/'controller_contract.json')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()==contract['upstream_commit']
    for name,h in contract['upstream_sha256'].items():assert p.sha(upstream/name)==h,name
    for name,version in contract['packages'].items():assert importlib.metadata.version(name)==version,(name,version)
    identity=p.read(pref/'native_dependency_identity.json')
    for name,item in identity['core_native_binary_identities'].items():
        package='pybullet' if name.startswith('pybullet') else name.split('/')[0]
        assert p.sha(importlib.metadata.distribution(package).locate_file(name))==item['sha256'],name
    if out is not None:
        out=Path(out);seal=p.read(out/'execution_binding.json')
        assert seal['accepted_preflight_seal_sha256']==p.sha(pref/'protocol_seal.json')
        for name,h in seal['source_sha256'].items():assert p.sha(root/name)==h,name
        for name,h in seal['artifact_sha256'].items():assert p.sha(out/name)==h,name
        m=p.read(out/'frozen.json')
        for i,x in enumerate(states):assert np.asarray(m['cases'][str(i)]).tobytes()==x.tobytes()
    return states


def prepare(root,upstream,pref,out,request):
    root,upstream,pref,out,request=map(Path,(root,upstream,pref,out,request))
    if out.exists():raise FileExistsError('Existing study retained; no automatic restart or overwrite')
    states=checkpoint(root,upstream,pref)
    evidence=p.read(pref/'evidence_integrity.json')
    for name,h in evidence['artifacts_sha256'].items():assert p.sha(pref/name)==h,name
    assert p.read(pref/'preflight.json')['passed'] and p.read(pref/'repeatability.json')['passed']
    previous=p.read(pref/'resource_ledger.json')
    assert not previous['reservations'] and previous['charged']=={'solver_calls':0,'physics_steps':0,'cpu_seconds':58.,'wall_seconds':95.2}
    preservation=f.capture_preservation(root,upstream,out)
    out.mkdir(parents=True)
    immutable(out/'preservation_start.json',preservation)
    model=p.read(pref/'accepted_frozen.json')
    model['cases']={str(i):x.tolist() for i,x in enumerate(states)}
    model['initial_binary64_hex']={str(i):[float(v).hex() for v in x] for i,x in enumerate(states)}
    immutable(out/'frozen.json',model)
    (out/'reference.npy').write_bytes((pref/'accepted_reference.npy').read_bytes())
    (out/'protocol.json').write_bytes((pref/'protocol.json').read_bytes())
    (out/'accepted_preflight_ledger.json').write_bytes((pref/'resource_ledger.json').read_bytes())
    (out/'execution_approval_request.txt').write_bytes(request.read_bytes())
    approval={'execution_approved':True,'seal_sha256':p.sha(pref/'protocol_seal.json'),
              'preflight_report_sha256':p.sha(pref/'preflight.json'),
              'review_reference':'Explicit user execution approval; request SHA-256 '+p.sha(request),
              'scope':'Exact E12_MPC80_FEAS_v1 jobs only; stop on first failed acceptance check'}
    immutable(out/'approval.json',approval)
    ledger=copy.deepcopy(previous);ledger['execution_approval_present']=True
    ledger['accepted_preflight_ledger_sha256']=p.sha(pref/'resource_ledger.json')
    f.durable_json(out/'resource_ledger.json',ledger)
    artifacts=['preservation_start.json','frozen.json','reference.npy','protocol.json',
               'accepted_preflight_ledger.json','execution_approval_request.txt','approval.json']
    immutable(out/'execution_binding.json',{'version':'E12_MPC80_FEAS_v1_execution_binding_1',
        'accepted_preflight_seal_sha256':p.sha(pref/'protocol_seal.json'),
        'accepted_preflight_evidence_integrity_sha256':p.sha(pref/'evidence_integrity.json'),
        'source_sha256':{n:p.sha(root/n) for n in SOURCES},
        'artifact_sha256':{n:p.sha(out/n) for n in artifacts},
        'original_overall_qualification':'FAILED','original_0001_agreement':'FAILED',
        'preflight_CPU_58s_and_wall_95_2s_already_charged':True})
    for name in SOURCES:
        target=out/'execution_source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((root/name).read_bytes())


def resource_check():
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>4294967296:raise MemoryError('4-GiB memory cap exceeded')


def episode(root,upstream,pref,out,state_id,phase):
    """Unchanged native episode loop, observed by fail-closed supervision."""
    import casadi as c
    import pybullet as bullet
    from src.quadrotor_diagnostic import problem_graph
    out=Path(out);model=p.read(out/'frozen.json');case=str(state_id)
    folder=out/'runs'/f'{case}_nonlinear80_1000_{phase}'
    if folder.exists():raise FileExistsError('Attempt retained; no automatic retry')
    original_factory=p.make_controller;original_capture=p.capture_physics;original_solve=c.Opti.solve
    active={};captured={};predictions=[];verified=[];guards=[];calls=0

    def retain_failure(exc,where,extra=None):
        safe_write(folder/'supervision_failure.json',{'kind':type(exc).__name__,'message':str(exc),
            'where':where,'extra':extra,'solver_attempts':calls,
            'physics_steps':max(0,len(captured.get('data',{}).get('states',[]))-1),
            'fallback_action_applied':False})

    @contextmanager
    def capture(env):
        with original_capture(env) as data:
            captured['data']=data;step=bullet.stepSimulation;data['stream_write_seconds']=[]
            with (folder/'partial_physics.f64').open('xb',buffering=0) as stream:
                def persist():
                    start=time.perf_counter()
                    stream.write(np.r_[data['states'][-1],data['contacts'][-1]].astype('<f8').tobytes())
                    data['stream_write_seconds'].append(time.perf_counter()-start)
                persist();require_raw_state(data['states'][-1],data['contacts'][-1],model)
                def observed(*args,**kwargs):
                    count=len(data['states']);value=step(*args,**kwargs)
                    if len(data['states'])==count:return value
                    persist();n=len(data['states'])-1
                    try:
                        if n>6000:raise RuntimeError('Per-execution physics cap exceeded')
                        require_raw_state(data['states'][-1],data['contacts'][-1],model)
                        require_executed(env.last_clipped_action,model)
                        if n%20==0:
                            b=predictions[n//20-1];x=data['states'][-1]
                            err=np.abs(np.asarray(b['prediction'])-x)
                            close=float(np.max(np.abs(np.asarray(b['source_prediction'])-x)))
                            safe_write(folder/'intervals'/f'{n//20-1:04d}.json',{
                                'prediction_error':err,'source_closure':close,'bound':b['bound'],
                                'reserve':b['reserve'],'physics_end_index':n})
                            if np.any(err>np.asarray(b['bound'])+1e-10):raise ValueError('Prediction discrepancy contract failed')
                            if close>1e-10:raise ValueError('Source-recurrence reconstruction failed')
                        resource_check()
                    except Exception as exc:
                        retain_failure(exc,'physics_sample',{'raw_state':data['states'][-1],
                            'contacts':data['contacts'][-1],'physics_index':n});raise
                    return value
                with patch.object(bullet,'stepSimulation',observed):yield data

    def factory(*args,**kwargs):
        ctrl=original_factory(*args,**kwargs);native_select=ctrl.select_action
        own,_=independent_nlp_program(ctrl,model);opti=ctrl.opti_dict['opti'];graph=problem_graph(opti)
        expected=p.read(Path(pref)/'preflight.json')['graph']['native_program_sha256']
        if graph!=expected:raise ValueError('Native MPC80 graph changed; no solve permitted')
        assert ctrl.T==80 and ctrl.solver=='ipopt' and ctrl.warmstart and ctrl.constraint_tol==1e-6
        assert ctrl.x_prev is None and ctrl.u_prev is None
        gradient=c.gradient(opti.f+c.dot(opti.g,opti.lam_g),opti.x)
        active.update(ctrl=ctrl,opti=opti,gradient=gradient)
        def select(obs,info):
            data=captured['data'];k=len(guards);x=np.asarray(data['states'][-1]).copy()
            cache=data['states'][-2][4] if len(data['states'])>1 else x[4]
            guard=IndependentCandidateGuard(folder/'attempts'/f'{k:04d}',{
                'state_id':state_id,'phase':phase,'control_step':k,'native_reference_step':info['current_step'],
                'current_raw_state':x,'cache':cache,'program_sha256':graph,
                'controller_observation':np.asarray(obs),'initial_memory_empty':ctrl.u_prev is None},readout=own)
            guards.append(guard);active['guard']=guard
            active['previous_x']=np.empty(0) if ctrl.x_prev is None else ctrl.x_prev.copy()
            active['previous_u']=np.empty(0) if ctrl.u_prev is None else ctrl.u_prev.copy()
            def snapshot():return {'physics_states':np.asarray(data['states']),
                'physics_contacts':np.asarray(data['contacts']),'verified_actions':np.asarray(verified),
                'applied_full_intervals':np.array((len(data['states'])-1)//20)}
            try:
                action=guard.select(lambda:native_select(obs,info),lambda:guard.current_candidate,snapshot)
                with guard.timing.span('prediction_diagnostics'):
                    b=bound_components(x,action,cache,model)
                    predicted=rk4(x,b['model_thrust'],model)
                    source=source_prediction(x,b['executed_thrust'],cache,model)
                    reserve=np.minimum(predicted-model['state_low'],np.asarray(model['state_high'])-predicted)-b['bound']
                    record={'step':k,'initial':x,'cache':cache,'requested':action.copy(),
                            'prediction':predicted,'source_prediction':source,'reserve':reserve,**b}
                predictions.append(record);safe_write(folder/'causal_contract.json',predictions)
                if not np.isfinite(reserve).all() or np.any(reserve<=0):raise ValueError('Nonpositive discrepancy-aware reserve')
                guard.state['timings']=guard.timing.records;guard.state['timing_summary']=guard.timing.summary()
                safe_write(guard.folder/'status.json',guard.state)
                resource_check();verified.append(action.copy());return action
            except Exception as exc:retain_failure(exc,'action_selection');raise
        ctrl.select_action=select;return ctrl

    def solve(opti,*args,**kwargs):
        nonlocal calls
        if opti is not active.get('opti') or calls>=300:raise RuntimeError('Unexpected or excess native solve')
        guard=active['guard']
        with guard.timing.span('resource_accounting'):
            calls+=1
            safe_write(folder/'progress.json',{'solver_attempts':calls,'physics_steps':len(captured['data']['states'])-1})
        with guard.timing.span('warm_start_capture'):
            atomic_arrays(guard.folder/'warmstart.npz',{'initial_guess_not_solver_candidate':np.asarray(opti.debug.value(opti.x,opti.initial())),
                'previous_x':active['previous_x'],'previous_u':active['previous_u']})
        return guard.native_solve_hook(opti,lambda:original_solve(opti,*args,**kwargs),active['gradient'])

    with patch.object(p,'make_controller',factory),patch.object(p,'capture_physics',capture),\
         patch.object(p,'write',safe_write),patch.object(c.Opti,'solve',solve):
        result=p.run_episode(root,upstream,out,case,'nonlinear80',phase,1000)
    result.update(state_id=state_id,actual_native_solve_attempts=calls,
                  actual_physics_steps=max(0,len(captured.get('data',{}).get('states',[]))-1))
    safe_write(folder/'completion.json',result)
    if not result['complete']:raise RuntimeError('Incomplete scientific execution; campaign must stop')
    return verify_episode(out,folder,compare_repeat=True)


def compare_repetition(out,folder):
    m=p.read(folder/'completion.json')
    if m['phase']!='repeat':return {'required':False}
    previous=Path(out)/'runs'/f"{m['case']}_nonlinear80_1000_original"
    x=np.load(previous/'trajectory.npz');y=np.load(folder/'trajectory.npz')
    fields=['states','actions','clipped_actions','rpms','rewards','physics_states','physics_contacts',
            'predicted_states','predicted_inputs','native_model_predictions','current_steps','solver_success','solver_iterations']
    equal={name:x[name].tobytes()==y[name].tobytes() for name in fields}
    for k in range(300):
        a=np.load(previous/'attempts'/f'{k:04d}'/'candidate.npz');b=np.load(folder/'attempts'/f'{k:04d}'/'candidate.npz')
        for name in ['primal','parameters','dual']:
            if a[name].tobytes()!=b[name].tobytes():raise ValueError('Candidate repetition not bitwise')
    if not all(equal.values()):raise ValueError('Scientific repetition not bitwise: '+str(equal))
    return {'required':True,'fields':equal,'all_300_candidate_vectors_bitwise':True}


def verify_episode(out,folder,compare_repeat=False):
    """Independent saved-record check before authorizing the next replay/job."""
    out,folder=Path(out),Path(folder);m=p.read(folder/'completion.json');model=p.read(out/'frozen.json')
    if not m['complete'] or m['transitions']!=300:raise ValueError('Full scientific record missing')
    assert p.sha(folder/'trajectory.npz')==m['trajectory_sha256']
    a=np.load(folder/'trajectory.npz');reference=np.load(out/'reference.npy')
    assert a['states'].shape==(301,6) and a['physics_states'].shape==(6001,6)
    assert a['physics_states'][::20].tobytes()==a['states'].tobytes()
    assert np.max(np.abs(a['states'][0]-model['cases'][m['case']]))<=1e-14
    low=a['physics_states']-model['state_low'];high=np.asarray(model['state_high'])-a['physics_states']
    assert np.isfinite(a['physics_states']).all() and np.min(low)>=0 and np.min(high)>=0
    assert not np.any(a['physics_contacts']) and np.all(a['solver_success'])
    assert np.array_equal(a['current_steps'],np.arange(300))
    executed=np.array([require_executed(rpm,model) for rpm in a['rpms']])
    error=a['states'][1:]-reference[1:]
    state_cost=.5*np.einsum('ni,ij,nj->n',error,np.asarray(model['Q']),error)
    du=a['clipped_actions']-model['hover_input'];input_cost=.5*np.einsum('ni,ij,nj->n',du,np.asarray(model['R']),du)
    costs=state_cost+input_cost;assert np.max(np.abs(costs+a['rewards']))<=1e-10
    assert np.max(np.abs(executed-a['clipped_actions']))<=1e-12
    physics=physics_reconstruction(a,model,1000);causal=p.read(folder/'causal_contract.json');assert len(causal)==300
    readout=nlp_readout(model,80);maxima={};reserves=[];timing=[];max_close=max_error=0.;max_excess=-np.inf
    for k in range(300):
        directory=folder/'attempts'/f'{k:04d}';status=p.read(directory/'status.json');saved=np.load(directory/'candidate.npz')
        assert status['status']=='verified_action_not_yet_applied' and not status['missing_fields'] and not status['fallback_action_applied']
        assert p.sha(directory/'candidate.npz')==status['candidate_sha256']
        assert (directory/'warmstart.npz').exists()
        z,params,lam=[saved[n].reshape(-1) for n in ['primal','parameters','dual']]
        own=readout(z,params,lam)
        matching_program_values(own,[saved[n] for n in ['objective','constraint_values','lower','upper','lagrangian_gradient']])
        d=canonical_kkt(own[1],own[2],own[3],lam,own[4]);require_kkt(d,m['solver_statuses'][k])
        for key,value in d['metrics'].items():maxima[key]=max(maxima.get(key,0.),value)
        expected=reference[np.minimum(k+np.arange(81),300)].T.reshape(-1,order='F')
        assert np.array_equal(params[6:],expected) and np.max(np.abs(params[:6]-a['states'][k]))<1e-14
        record=causal[k];cache=a['physics_states'][max(0,20*k-1),4]
        assert record['cache']==cache and np.array_equal(record['initial'],a['states'][k])
        assert np.array_equal(record['requested'],a['actions'][k])
        bound=bound_components(a['states'][k],a['actions'][k],cache,model)
        exact=exact_majorant(a['states'][k],bound['model_thrust'],bound['executed_thrust'],cache,model)
        assert all(Fraction(float(v))>=e for v,e in zip(bound['bound'],exact))
        predicted=rk4(a['states'][k],bound['model_thrust'],model)
        reserve=np.minimum(predicted-model['state_low'],np.asarray(model['state_high'])-predicted)-bound['bound']
        assert np.all(reserve>0);reserves.append(reserve)
        max_error=max(max_error,float(np.max(np.abs(predicted-a['states'][k+1]))))
        max_excess=max(max_excess,float(np.max(np.abs(predicted-a['states'][k+1])-bound['bound'])))
        close=float(np.max(np.abs(source_prediction(a['states'][k],bound['executed_thrust'],cache,model)-a['states'][k+1])))
        max_close=max(max_close,close);assert close<=1e-10 and max_excess<=1e-10
        spans=status['timings'];solver=[s for s in spans if s['category']=='native_solver'];assert len(solver)==1
        timing.append({'step':k,'solver_only':solver[0]['wall_inclusive'],
            'native_action':status['timing_summary']['native_action_selection_wall'],
            'supervised_action':float(a['solver_wall'][k]),'exclusive_categories':status['timing_summary']['exclusive_categories']})
    assert np.max(a['warm_u_error'])<=1e-12 and np.max(a['warm_x_error'])<=1e-12
    reserve=np.asarray(reserves);margins=np.minimum(low,high);input_margins=np.minimum(executed-model['action_low'],np.asarray(model['action_high'])-executed)
    atomic_arrays(folder/'independent_diagnostics.npz',{'executed_thrusts':executed,'requested_thrusts':a['actions'],
        'clipped_thrusts':a['clipped_actions'],'raw_lower_margins':low,'raw_upper_margins':high,
        'input_margins':input_margins,'state_cost':state_cost,'input_cost':input_cost,'tracking_errors':error,'reserves':reserve})
    result={'passed':True,'state_id':int(m['case']),'phase':m['phase'],'complete':True,
        'controls':300,'raw_states':6001,'return':-float(costs.sum()),'cost':float(costs.sum()),
        'discounted_return':-float(np.dot(.99**np.arange(300),costs)),
        'position_tracking_rmse':float(np.sqrt(np.mean(error[:,[0,2]]**2))),
        'tracking_rmse_by_state':np.sqrt(np.mean(error**2,axis=0)).tolist(),
        'terminal_tracking_error':error[-1].tolist(),'absolute_effort':float(np.sum(executed**2)/50),
        'hover_relative_effort':float(np.sum((executed-model['hover_input'])**2)/50),
        'thrust_rms':np.sqrt(np.mean(executed**2,axis=0)).tolist(),'peak_thrust':executed.max(0).tolist(),
        'saturation_fraction':np.mean((np.abs(executed-model['action_low'])<1e-7)|(np.abs(executed-model['action_high'])<1e-7),axis=0).tolist(),
        'total_variation':float(np.abs(np.diff(executed,axis=0)).sum()),
        'minimum_raw_lower_margins':low.min(0).tolist(),'minimum_raw_upper_margins':high.min(0).tolist(),
        'minimum_raw_margins':margins.min(0).tolist(),'minimum_normalized_raw_margin':float(np.min(margins/(np.asarray(model['state_high'])-model['state_low']))),
        'minimum_input_margins':input_margins.min(0).tolist(),'minimum_reserves':reserve.min(0).tolist(),
        'state_violation_samples':0,'input_violations':0,'contacts':0,'first_crossing_seconds':None,
        'independent_candidate_checks':300,'KKT_maxima':maxima,'source_closure_maximum':max_close,
        'maximum_prediction_discrepancy':max_error,'maximum_contract_excess':max_excess,
        'original_0001_test_passed_on_this_record':max_error<=.001,
        'original_overall_qualification':'FAILED','original_global_0001_gate':'FAILED',
        'maximum_solver_iterations':int(a['solver_iterations'].max()),'native_statuses':sorted(set(m['solver_statuses'])),
        'physics_equation_reconstruction':physics,'timings':timing}
    if compare_repeat:result['repetition']=compare_repetition(out,folder)
    safe_write(folder/'independent_episode_check.json',result)
    return result


def replay(upstream,out,state_id,phase,kind):
    """Reuse accepted replay implementations, adding fail-fast record checks."""
    import casadi as c
    import pybullet as bullet
    from src.quadrotor_linear_revalidation import replay_executed_motor_commands
    out=Path(out);folder=out/'runs'/f'{state_id}_nonlinear80_1000_{phase}'
    original_capture=p.capture_physics;original_factory=p.make_environment
    a=np.load(folder/'trajectory.npz');model=p.read(out/'frozen.json');live={}
    journal=folder/(kind+'_partial.npz')
    def forbidden(*args,**kwargs):raise RuntimeError('Replay cannot call an optimizer')
    def factory(*args,**kwargs):
        env=original_factory(*args,**kwargs);step=env.step;reset=env.reset;states=[];rewards=[]
        def reset_checked(*args,**kwargs):
            result=reset(*args,**kwargs);states.append(p.raw_state(env));live['controls']=0
            atomic_arrays(journal,{'states':states,'rewards':rewards})
            if states[-1].tobytes()!=a['states'][0].tobytes():raise ValueError('Replay reset is not bitwise')
            return result
        def step_checked(action):
            result=step(action);k=len(rewards);states.append(p.raw_state(env));rewards.append(result[1]);live['controls']=len(rewards)
            atomic_arrays(journal,{'states':states,'rewards':rewards})
            if np.asarray(states[-1]).tobytes()!=a['states'][k+1].tobytes():raise ValueError('Replay control state is not bitwise')
            if np.asarray(result[1]).tobytes()!=np.asarray(a['rewards'][k]).tobytes():raise ValueError('Replay reward is not bitwise')
            require_executed(env.last_clipped_action,model);return result
        env.reset=reset_checked;env.step=step_checked;return env
    @contextmanager
    def capture(env):
        with original_capture(env) as data:
            native=bullet.stepSimulation
            with (folder/(kind+'_physics.f64')).open('xb',buffering=0) as stream:
                stream.write(np.r_[data['states'][0],data['contacts'][0]].astype('<f8').tobytes())
                def check(*args,**kwargs):
                    old=len(data['states']);r=native(*args,**kwargs)
                    if len(data['states'])>old:
                        k=len(data['states'])-1;stream.write(np.r_[data['states'][-1],data['contacts'][-1]].astype('<f8').tobytes())
                        live['physics_steps']=k;require_raw_state(data['states'][-1],data['contacts'][-1],model)
                        if np.asarray(data['states'][-1]).tobytes()!=a['physics_states'][k].tobytes():raise ValueError('Motor replay physics is not bitwise')
                    return r
                with patch.object(bullet,'stepSimulation',check):yield data
    try:
        with patch.object(p,'make_environment',factory),patch.object(p,'capture_physics',capture),\
             patch.object(p,'write',safe_write),patch.object(c.Opti,'solve',forbidden),patch.object(c.Opti,'solve_limited',forbidden):
            if kind=='native_replay':result=p.replay(upstream,out,folder,False)
            else:result=replay_executed_motor_commands(upstream,out,folder)
        assert result['steps']==300
        for name,value in result.items():
            if name.endswith('bitwise'):assert value,name
        if kind=='motor_replay':
            independent,_=audit_record(folder,model,np.load(out/'reference.npy'))
            assert independent['safe_complete_no_contact']
            assert abs(independent['return']-p.read(folder/'independent_episode_check.json')['return'])<=1e-10
        return {'passed':True,'controls':300,'actual_physics_steps':6000,'replay':result}
    except Exception as exc:
        safe_write(folder/(kind+'_failure.json'),{'kind':type(exc).__name__,'message':str(exc),'partial':live,
            'status':'failed_or_incomplete','fallback_action_applied':False});raise
