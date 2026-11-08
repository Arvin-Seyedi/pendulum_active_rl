"""Fresh E12_MPC80_FEAS_v2 execution, using the sealed boundary observer.

No v1 source or accepted ledger is modified. Native controllers, plant and
verification equations are reused; only observer/dispatch wiring is new.
"""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import copy
import importlib.metadata
import resource
import subprocess
import time
import numpy as np
from src import quadrotor_pilot as p
from src import quadrotor_mpc80_feasibility_v1 as ledger_api
from src import quadrotor_mpc80_feasibility_v2 as preflight
from src import quadrotor_mpc80_execution_v1 as verified
from src.quadrotor_applied_action_v2 import AppliedActionObserver
from src.quadrotor_mmaq_v1 import immutable,safe_write
from src.quadrotor_h2_requalification_v2 import IndependentCandidateGuard
from src.quadrotor_mmaq_verification_v1 import independent_nlp_program
from src.quadrotor_mismatch_contract_v1 import bound_components,source_prediction
from src.quadrotor_verification import rk4
from src.quadrotor_qualification_telemetry_v2 import atomic_arrays

SOURCES=['src/quadrotor_mpc80_execution_v2.py','experiments/12_mpc80_feasibility_v2_study.py',
         'tests/test_quadrotor_mpc80_execution_v2.py']
REUSED=['src/quadrotor_mpc80_execution_v1.py','src/quadrotor_h2_requalification_v2.py',
        'src/quadrotor_qualification_telemetry_v2.py','src/quadrotor_mpc80_feasibility_v1.py']
FIELD_WIDTH=56
PHYSICAL_COLUMNS=slice(0,54)  # Last two columns identify transient body/client IDs.


def flatten_packet(packet):
    """Direct API argument values; never read the late last_clipped_action cache."""
    control=packet['preprocess_record'];forces=[e for e in packet['submissions'] if e['kind']=='force']
    torques=[e for e in packet['submissions'] if e['kind']=='torque']
    if len(forces)!=4 or len(torques)!=1:raise ValueError('Missing submitted force/torque arguments')
    torque=torques[0]
    row=np.r_[packet['application_index'],packet['control_index'],packet['physics_index'],
        control['requested_argument'],control['native_physical_action'],control['native_clipped_action'],
        control['native_returned_rpm'],packet['direct_rpm_argument'],
        np.asarray([e['kwargs']['forceObj'] for e in forces]).reshape(-1),
        np.asarray([e['kwargs']['posObj'] for e in forces]).reshape(-1),torque['kwargs']['torqueObj'],
        [e['args'][1] for e in forces],[e['kwargs']['flags'] for e in forces],
        torque['args'][1],torque['kwargs']['flags'],forces[0]['args'][0],forces[0]['kwargs']['physicsClientId']]
    if row.shape!=(FIELD_WIDTH,) or not np.isfinite(row).all():raise ValueError('Invalid direct application record')
    return row.astype('<f8',copy=False)


class SubmissionStream:
    """Journal before integration, mark completion separately; preserve failures."""
    def __init__(self,folder,label,expected=None):
        self.folder=Path(folder);self.label=label;self.expected=expected
        self.count=0;self.completed=0;self.errors=0
        self.app=(self.folder/(label+'.f64')).open('xb',buffering=0)
        self.done=(self.folder/(label+'_completed.u1')).open('xb',buffering=0)
        immutable(self.folder/(label+'_layout.json'),{'dtype':'<f8','columns':56,
            'fields':{'indices':[0,3],'requested_thrust':[3,5],'native_physical_thrust':[5,7],
              'native_clipped_thrust':[7,9],'preprocessed_RPM':[9,13],'direct_physics_RPM':[13,17],
              'direct_submitted_force_vectors':[17,29],'direct_submitted_application_positions':[29,41],
              'direct_submitted_torque':[41,44],'motor_links':[44,48],'motor_frames':[48,52],
              'torque_link_and_frame':[52,54],'runtime_body_and_client':[54,56]},
            'provenance':'Direct native call arguments, not physical sensor measurements; thrust from RPM is derived',
            'comparison_ignores_only_transient_runtime_body_and_client_IDs':True})
    def emit(self,packet):
        idx=packet['application_index']
        if packet['force_application_returned'] and not packet['physics_returned'] and idx==self.count:
            try:
                row=flatten_packet(packet);self.app.write(row.tobytes());self.count+=1
                if self.expected is not None:
                    if idx>=len(self.expected) or row[PHYSICAL_COLUMNS].tobytes()!=self.expected[idx,PHYSICAL_COLUMNS].tobytes():
                        raise ValueError('Submitted motor/force commands disagree bitwise')
            except BaseException as exc:
                safe_write(self.folder/(self.label+f'_failure_{self.errors}.json'),{'error':repr(exc),'packet':packet})
                self.errors+=1;raise
        if packet['physics_returned'] and idx==self.completed:
            self.done.write(b'\x01');self.completed+=1
        if any(k in packet for k in ['application_error','step_or_callback_error','context_exit_with_unconsumed_application']):
            safe_write(self.folder/(self.label+f'_failure_{self.errors}.json'),packet);self.errors+=1
    def close(self):
        self.app.close();self.done.close()
        safe_write(self.folder/(self.label+'_counts.json'),{'applications':self.count,'completed_physics_steps':self.completed,
            'error_records':self.errors,'complete_6000':self.count==self.completed==6000})


def checkpoint(root,upstream,pref,out=None):
    root,upstream,pref=map(Path,(root,upstream,pref));preflight.check_seal(root,pref)
    preflight.verify_parent(root,upstream)
    config=p.read(pref/'protocol.json');preflight.validate_inheritance(root,config)
    states,_=ledger_api.validate_cohort(root,config)
    closure=p.read(pref/'supplemental_parent_source_checks.json')
    for n,h in closure['parent_source_and_report_sha256'].items():assert p.sha(root/n)==h,n
    if out is not None:
        out=Path(out);binding=p.read(out/'execution_binding.json')
        assert binding['accepted_preflight_seal_sha256']==p.sha(pref/'protocol_seal.json')
        for n,h in binding['source_sha256'].items():assert p.sha(root/n)==h,n
        for n,h in binding['artifact_sha256'].items():assert p.sha(out/n)==h,n
        model=p.read(out/'frozen.json')
        for i,x in enumerate(states):assert np.asarray(model['cases'][str(i)]).tobytes()==x.tobytes()
    return states


def prepare(root,upstream,pref,out,request):
    root,upstream,pref,out,request=map(Path,(root,upstream,pref,out,request))
    if out.exists():raise FileExistsError('Existing v2 execution retained; no automatic retry')
    states=checkpoint(root,upstream,pref)
    integrity=p.read(pref/'evidence_integrity.json')
    for n,h in integrity['artifact_sha256'].items():assert p.sha(pref/n)==h,n
    previous=p.read(pref/'resource_ledger.json')
    assert previous['charged']=={'solver_calls':0,'physics_steps':0,'cpu_seconds':158.63,'wall_seconds':270.86899999999997}
    assert previous['preflight_passed'] and not previous['halted'] and not previous['dispatched_jobs']
    preserved=ledger_api.capture_preservation(root,upstream,out)
    out.mkdir(parents=True);immutable(out/'preservation_start.json',preserved)
    model=p.read(pref/'accepted_frozen.json');model['cases']={str(i):x.tolist() for i,x in enumerate(states)}
    model['initial_binary64_hex']={str(i):[float(v).hex() for v in x] for i,x in enumerate(states)}
    immutable(out/'frozen.json',model)
    (out/'reference.npy').write_bytes((pref/'accepted_reference.npy').read_bytes())
    (out/'protocol.json').write_bytes((pref/'protocol.json').read_bytes())
    (out/'accepted_preflight_ledger.json').write_bytes((pref/'resource_ledger.json').read_bytes())
    (out/'execution_approval_request.txt').write_bytes(request.read_bytes())
    approval={'execution_approved':True,'seal_sha256':p.sha(pref/'protocol_seal.json'),
        'preflight_report_sha256':p.sha(pref/'preflight.json'),
        'review_reference':'Explicit v2 execution approval; request SHA-256 '+p.sha(request)}
    immutable(out/'approval.json',approval)
    ledger={'version':'E12_MPC80_FEAS_v2','seal_sha256':approval['seal_sha256'],
        'preflight_report_sha256':approval['preflight_report_sha256'],'preflight_passed':True,
        'execution_approval_present':True,'halted':False,'jobs':ledger_api.job_plan(),
        'caps':p.read(pref/'protocol.json')['budget'],'reservations':[],
        'charged':copy.deepcopy(previous['charged']),
        'carried_CPU_and_wall_from_v1_and_both_preflights':True,
        'prior_failed_v1_reservations_not_released':{'solver_calls':300,'physics_steps':6000}}
    ledger_api.durable_json(out/'resource_ledger.json',ledger)
    artifacts=['preservation_start.json','frozen.json','reference.npy','protocol.json',
               'accepted_preflight_ledger.json','execution_approval_request.txt','approval.json']
    immutable(out/'execution_binding.json',{'version':'E12_MPC80_FEAS_v2_execution_binding_1',
        'accepted_preflight_seal_sha256':p.sha(pref/'protocol_seal.json'),
        'source_sha256':{n:p.sha(root/n) for n in SOURCES+REUSED},
        'artifact_sha256':{n:p.sha(out/n) for n in artifacts},
        'v1_failed_execution_preserved':True,'original_E12_qualification':'FAILED','original_0001_gate':'FAILED'})
    for n in SOURCES:
        target=out/'execution_source_snapshot'/n;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((root/n).read_bytes())


@contextmanager
def checked_capture(env,folder,label,model,captured,predictions=None,expected_states=None,expected_commands=None):
    """Observer is INSIDE the native raw capture, as sealed in v2 preflight."""
    import pybullet as bullet
    native_capture=captured['native_capture']
    with native_capture(env) as data:
        captured['data']=data;data['stream_write_seconds']=[]
        rawname='partial_physics.f64' if label=='submitted_scientific' else label+'_physics.f64'
        stream=(folder/rawname).open('xb',buffering=0)
        telemetry=SubmissionStream(folder,label,expected_commands)
        try:
            stream.write(np.r_[data['states'][0],data['contacts'][0]].astype('<f8').tobytes())
            verified.require_raw_state(data['states'][0],data['contacts'][0],model)
            def after(packet):
                start=time.perf_counter();x=np.asarray(data['states'][-1]);n=len(data['states'])-1
                stream.write(np.r_[x,data['contacts'][-1]].astype('<f8').tobytes())
                data['stream_write_seconds'].append(time.perf_counter()-start)
                try:
                    assert n==packet['physics_index']+1 and n<=6000
                    verified.require_raw_state(x,data['contacts'][-1],model)
                    derived=verified.require_executed(packet['direct_rpm_argument'],model)
                    force_sum=packet['grouped_submitted_force_argument_sums']
                    if np.any(force_sum<model['action_low']) or np.any(force_sum>model['action_high']):raise ValueError('Negative submitted-force input margin')
                    assert np.max(np.abs(derived-force_sum))<=1e-12
                    if expected_states is not None and x.tobytes()!=expected_states[n].tobytes():raise ValueError('Scientific raw state not bitwise reproducible')
                    if predictions is not None:
                        b=predictions[(n-1)//20]
                        assert np.array_equal(packet['preprocess_record']['requested_argument'],b['requested'])
                        if n%20==0:
                            error=np.abs(np.asarray(b['prediction'])-x)
                            close=float(np.max(np.abs(np.asarray(b['source_prediction'])-x)))
                            safe_write(folder/'intervals'/f'{n//20-1:04d}.json',{'prediction_error':error,
                                'source_closure':close,'bound':b['bound'],'reserve':b['reserve'],'physics_end_index':n})
                            if close>1e-10 or np.any(error>np.asarray(b['bound'])+1e-10):raise ValueError('Source or discrepancy contract failed')
                    verified.resource_check()
                except BaseException as exc:
                    safe_write(folder/(label+'_live_failure.json'),{'error':repr(exc),'packet':packet,
                        'physics_index':n,'raw_state':x,'contacts':data['contacts'][-1]});raise
            with AppliedActionObserver(env,bullet,after_step=after,sink=telemetry.emit) as observer:
                captured['observer']=observer;yield data
        finally:
            stream.close();telemetry.close()


def verify_submissions(folder,label,trajectory):
    counts=p.read(folder/(label+'_counts.json'))
    assert counts['complete_6000'] and counts['error_records']==0
    rows=np.fromfile(folder/(label+'.f64'),dtype='<f8').reshape(-1,FIELD_WIDTH)
    done=np.fromfile(folder/(label+'_completed.u1'),dtype='u1');assert rows.shape==(6000,FIELD_WIDTH) and np.all(done==1)
    assert np.array_equal(rows[:,0],np.arange(6000)) and np.array_equal(rows[:,2],np.arange(6000))
    assert np.array_equal(rows[:,1],np.repeat(np.arange(300),20))
    np.testing.assert_array_equal(rows[:,3:5],np.repeat(trajectory['actions'],20,axis=0))
    np.testing.assert_array_equal(rows[:,7:9],np.repeat(trajectory['clipped_actions'],20,axis=0))
    np.testing.assert_array_equal(rows[:,13:17],np.repeat(trajectory['rpms'],20,axis=0))
    return rows


def episode(root,upstream,pref,out,state_id,phase):
    import casadi as c
    from src.quadrotor_diagnostic import problem_graph
    out=Path(out);model=p.read(out/'frozen.json');case=str(state_id)
    folder=out/'runs'/f'{case}_nonlinear80_1000_{phase}'
    if folder.exists():raise FileExistsError('Attempt retained; no retry')
    native_factory=p.make_controller;native_solve=c.Opti.solve
    captured={'native_capture':p.capture_physics};active={};guards=[];predictions=[];verified_actions=[];calls=0
    expected_states=expected_commands=None
    if phase=='repeat':
        original=out/'runs'/f'{case}_nonlinear80_1000_original'
        expected_states=np.load(original/'trajectory.npz')['physics_states']
        expected_commands=np.fromfile(original/'submitted_scientific.f64',dtype='<f8').reshape(-1,FIELD_WIDTH)
    @contextmanager
    def capture(env):
        with checked_capture(env,folder,'submitted_scientific',model,captured,predictions,
                             expected_states,expected_commands) as data:yield data
    def factory(*args,**kwargs):
        ctrl=native_factory(*args,**kwargs);native_select=ctrl.select_action;opti=ctrl.opti_dict['opti']
        own,_=independent_nlp_program(ctrl,model);graph=problem_graph(opti)
        expected=p.read(Path(pref)/'preflight.json')['checks']
        # Sealed original MPC80 program remains the native program reference.
        v1pref=Path(root)/'results/benchmark_selection/mpc80_feas_v1_preflight'
        assert graph==p.read(v1pref/'preflight.json')['graph']['native_program_sha256']
        assert ctrl.T==80 and ctrl.warmstart and ctrl.solver=='ipopt' and ctrl.constraint_tol==1e-6
        assert not ctrl.soft_constraints and not ctrl.use_lqr_gain_and_terminal_cost
        assert ctrl.x_prev is None and ctrl.u_prev is None
        active.update(ctrl=ctrl,opti=opti,gradient=c.gradient(opti.f+c.dot(opti.g,opti.lam_g),opti.x))
        def select(obs,info):
            begun=time.perf_counter();k=len(guards);data=captured['data'];x=np.asarray(data['states'][-1]).copy()
            cache=data['states'][-2][4] if len(data['states'])>1 else x[4]
            guard=IndependentCandidateGuard(folder/'attempts'/f'{k:04d}',{
                'state_id':state_id,'phase':phase,'control_step':k,'native_reference_step':info['current_step'],
                'current_raw_state':x,'cache':cache,'program_sha256':graph,
                'controller_observation':np.asarray(obs),'initial_memory_empty':ctrl.u_prev is None},readout=own)
            guards.append(guard);active['guard']=guard
            active['previous_x']=np.empty(0) if ctrl.x_prev is None else ctrl.x_prev.copy()
            active['previous_u']=np.empty(0) if ctrl.u_prev is None else ctrl.u_prev.copy()
            snapshot=lambda:{'physics_states':np.asarray(data['states']),'physics_contacts':np.asarray(data['contacts']),
                'verified_actions':np.asarray(verified_actions),'applied_full_intervals':np.array((len(data['states'])-1)//20)}
            try:
                action=guard.select(lambda:native_select(obs,info),lambda:guard.current_candidate,snapshot)
                with guard.timing.span('prediction_diagnostics'):
                    b=bound_components(x,action,cache,model);prediction=rk4(x,b['model_thrust'],model)
                    source=source_prediction(x,b['executed_thrust'],cache,model)
                    reserve=np.minimum(prediction-model['state_low'],np.asarray(model['state_high'])-prediction)-b['bound']
                    record={'step':k,'initial':x,'cache':cache,'requested':action.copy(),
                            'prediction':prediction,'source_prediction':source,'reserve':reserve,**b}
                predictions.append(record);safe_write(folder/'causal_contract.json',predictions)
                if not np.isfinite(reserve).all() or np.any(reserve<=0):raise ValueError('Nonpositive numerical reserve')
                verified.resource_check();verified_actions.append(action.copy());return action
            except BaseException as exc:
                guard.state.update(status='failed_or_incomplete',supervision_exception=repr(exc))
                safe_write(folder/'supervision_failure.json',{'error':repr(exc),'control_step':k,
                    'native_solve_attempts':calls,'physics_steps':len(data['states'])-1});raise
            finally:
                guard.state['timings']=guard.timing.records;guard.state['timing_summary']=guard.timing.summary()
                guard.state['observed_full_wrapper_wall_at_checkpoint']=time.perf_counter()-begun
                safe_write(guard.folder/'status.json',guard.state)
        ctrl.select_action=select;return ctrl
    def solve(opti,*args,**kwargs):
        nonlocal calls
        if opti is not active.get('opti') or calls>=300:raise RuntimeError('Unexpected or excessive native solve')
        guard=active['guard']
        with guard.timing.span('resource_accounting'):
            calls+=1;safe_write(folder/'progress.json',{'solver_attempts':calls,'physics_steps':len(captured['data']['states'])-1})
        with guard.timing.span('warm_start_capture'):
            atomic_arrays(guard.folder/'warmstart.npz',{'initial_guess_not_solver_candidate':np.asarray(opti.debug.value(opti.x,opti.initial())),
                'previous_x':active['previous_x'],'previous_u':active['previous_u']})
        return guard.native_solve_hook(opti,lambda:native_solve(opti,*args,**kwargs),active['gradient'])
    with patch.object(p,'make_controller',factory),patch.object(p,'capture_physics',capture),\
         patch.object(p,'write',safe_write),patch.object(c.Opti,'solve',solve):
        result=p.run_episode(root,upstream,out,case,'nonlinear80',phase,1000)
    result.update(version='E12_MPC80_FEAS_v2',state_id=state_id,actual_native_solve_attempts=calls,
                  actual_physics_steps=max(0,len(captured.get('data',{}).get('states',[]))-1))
    safe_write(folder/'completion.json',result)
    if not result['complete']:raise RuntimeError('Incomplete scientific execution; campaign stopped')
    rows=verify_submissions(folder,'submitted_scientific',np.load(folder/'trajectory.npz'))
    if expected_commands is not None:assert rows[:,PHYSICAL_COLUMNS].tobytes()==expected_commands[:,PHYSICAL_COLUMNS].tobytes()
    check=verified.verify_episode(out,folder,compare_repeat=True)
    check.update(direct_application_rows=6000,submitted_command_timing_verified=True,
        thrust_quantities_are_derived_not_sensor_outputs=True)
    safe_write(folder/'independent_episode_check.json',check);return check


def replay(upstream,out,state_id,phase,kind):
    """Native replay stays without physics instrumentation; motor replay observes submissions."""
    out=Path(out);folder=out/'runs'/f'{state_id}_nonlinear80_1000_{phase}';model=p.read(out/'frozen.json')
    a=np.load(folder/'trajectory.npz');native_capture=p.capture_physics;native_factory=p.make_environment
    expected=np.fromfile(folder/'submitted_scientific.f64',dtype='<f8').reshape(-1,FIELD_WIDTH)
    if kind=='native_replay':
        cache_records=[]
        def factory(*args,**kwargs):
            env=native_factory(*args,**kwargs);step=env.step
            def checked(action):
                r=step(action);k=len(cache_records);cache=env.last_clipped_action.copy();cache_records.append(cache)
                atomic_arrays(folder/'native_replay_end_interval_RPM.npz',{'end_interval_cache_RPM':cache_records})
                if cache.tobytes()!=a['rpms'][k].tobytes():raise ValueError('Native replay submitted RPM mismatch at completed interval')
                return r
            env.step=checked;return env
        with patch.object(p,'make_environment',factory):return verified.replay(upstream,out,state_id,phase,kind)
    captured={'native_capture':native_capture}
    @contextmanager
    def capture(env):
        with checked_capture(env,folder,'submitted_motor_replay',model,captured,
                             expected_states=a['physics_states'],expected_commands=expected) as data:yield data
    with patch.object(p,'capture_physics',capture):result=verified.replay(upstream,out,state_id,phase,kind)
    rows=verify_submissions(folder,'submitted_motor_replay',a)
    assert rows[:,PHYSICAL_COLUMNS].tobytes()==expected[:,PHYSICAL_COLUMNS].tobytes()
    result['submitted_motor_and_force_commands_bitwise']=True;return result
