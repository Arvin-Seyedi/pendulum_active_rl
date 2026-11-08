"""Only frozen H2/MPC20, v2 evidence capture and independent verification.

All native programs, options, warm starts and environment stepping are reused.
The v1 runner and accepted v2 verifier/telemetry sources remain immutable.
"""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace
import hashlib
import importlib.metadata
import json
import resource
import time
import numpy as np
from src import quadrotor_pilot as pilot
from src.quadrotor_mmaq_v1 import safe_write,immutable,preservation
from src.quadrotor_kkt_integrity_v2 import canonical_kkt,require_kkt,matching_program_values,nlp_readout
from src.quadrotor_qualification_telemetry_v2 import CandidateGuard,TimingLedger,native_opti_snapshot
from src.quadrotor_mismatch_contract_v1 import bound_components,source_prediction
from src.quadrotor_verification import rk4

VERSION='E12_H2_MPC20_requalification_v2'
BASE=Path('results/benchmark_selection')
SOURCES=['configs/experiment_12_h2_requalification_v2.json','src/quadrotor_h2_requalification_v2.py',
         'src/quadrotor_h2_verification_v2.py','experiments/12_h2_requalification_v2.py',
         'tests/test_quadrotor_h2_requalification_v2.py']


class IndependentCandidateGuard(CandidateGuard):
    """Persist native candidates first; check independent FG and derivatives."""
    def __init__(self,*args,readout,**kwargs):
        super().__init__(*args,**kwargs);self.readout=readout;self.current_candidate=None

    def preserve_candidate(self,candidate):
        if candidate is self.current_candidate:
            return self.state['missing_fields']
        # The accepted base writes raw NPZ, native status and diagnostics before
        # any independent readout/reconstruction check below can fail.
        missing=super().preserve_candidate(candidate);self.current_candidate=candidate
        required=['primal','parameters','dual','objective','constraint_values','lower','upper','lagrangian_gradient']
        if missing or not all(np.isfinite(np.asarray(candidate[k])).all() for k in ['primal','parameters','dual']):
            return missing
        with self.timing.span('independent_verification'):
            own=self.readout(candidate['primal'],candidate['parameters'],candidate['dual'])
            obj,g,lo,hi,grad=[np.asarray(x).reshape(-1) for x in own]
            diagnostic=canonical_kkt(g,lo,hi,candidate['dual'],grad)
        # Persist even a nonfinite/mismatched independent diagnostic BEFORE any
        # assertion. A failure never discards its rejected candidate again.
        with self.timing.span('independent_diagnostic_persistence'):
            safe_write(self.folder/'independent_constraint_diagnostics.json',diagnostic)
            self.state['independent_residuals']=diagnostic['metrics']
            self.state['independent_objective']=obj
            safe_write(self.folder/'status.json',self.state)
        with self.timing.span('independent_verification'):
            error=matching_program_values([obj,g,lo,hi,grad],
                [candidate['objective'],candidate['constraint_values'],candidate['lower'],candidate['upper'],candidate['lagrangian_gradient']])
            if not np.isfinite(obj).all():raise AssertionError('Nonfinite independent objective')
        self.state['independent_reconstruction_error']=error
        self.last_diagnostic=diagnostic
        safe_write(self.folder/'status.json',self.state)
        return missing


def check_seal(root,out):
    root,out=Path(root),Path(out);seal=pilot.read(out/'protocol_seal.json')
    for name,h in seal['source_sha256'].items():assert pilot.sha(root/name)==h,name
    for name,h in seal['artifact_sha256'].items():assert pilot.sha(out/name)==h,name


def protect(root,upstream):
    root=Path(root);prior=preservation(root,upstream);seals={}
    for sub in ['mmaq_v1','qualification_integrity_v2']:
        folder=root/BASE/sub;seal=pilot.read(folder/'completion.json')
        for name,h in seal['artifact_sha256'].items():assert pilot.sha(folder/name)==h,name
        for key in ['execution_source_sha256','postrun_source_sha256','source_sha256']:
            for name,h in seal.get(key,{}).items():assert pilot.sha(root/name)==h,name
        seals[sub]=pilot.sha(folder/'completion.json')
    iq=pilot.read(root/BASE/'technical_pilot/iql_preservation.json')['sha256']
    for name,h in iq.items():assert pilot.sha(root/name)==h,name
    return {'previous':prior,'accepted_seals':seals,'iql_sha256':iq,'all_nine_iql_unchanged':True}


def prepare(root,upstream,out):
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
    if (out/'protocol_seal.json').exists():check_seal(root,out);return
    old=root/BASE/'mmaq_v1';record=protect(root,upstream)
    f=pilot.read(old/'frozen.json');c=pilot.read(root/'configs/experiment_12_h2_requalification_v2.json')
    assert c['case']=='H2' and c['controller']=='nonlinear20' and c['maximum_additional_solver_calls']==600
    x=np.load(old/'heldout_states.npy')[2]
    assert x.tobytes()==np.asarray(f['cases']['H2']).tobytes()
    immutable(out/'frozen.json',f);immutable(out/'protocol.json',c)
    (out/'reference.npy').write_bytes((old/'reference.npy').read_bytes())
    (out/'environment.lock.txt').write_bytes((old/'environment.lock.txt').read_bytes())
    immutable(out/'initial_state.json',{'source':'mmaq_v1/heldout_states.npy row 2',
        'values':x.tolist(),'binary64_hex':[float(v).hex() for v in x],'little_endian_bytes':x.astype('<f8').tobytes().hex()})
    immutable(out/'preservation_start.json',record)
    previous=pilot.read(root/BASE/'qualification_integrity_v2/budget_audit.json')
    immutable(out/'prior_remaining_budget.json',previous['remaining_conservative_original_budget'])
    assert previous['remaining_conservative_original_budget']['solver_calls']>=600
    assert previous['remaining_conservative_original_budget']['physics_steps']>=36000
    artifact=['frozen.json','protocol.json','reference.npy','environment.lock.txt','initial_state.json',
              'preservation_start.json','prior_remaining_budget.json']
    immutable(out/'protocol_seal.json',{'version':VERSION,
        'source_sha256':{name:pilot.sha(root/name) for name in SOURCES},
        'artifact_sha256':{name:pilot.sha(out/name) for name in artifact},
        'scope':'H2 MPC20 original/repeat only; no new outcomes before preflight pass'})


def preflight(root,upstream,out):
    import casadi as c
    import pybullet as bullet
    from src.quadrotor_mmaq_verification_v1 import independent_nlp_program
    from src.quadrotor_diagnostic import problem_graph
    from src.quadrotor_qualification_integrity_audit import retained_kkt
    root,out=Path(root),Path(out);check_seal(root,out)
    if (out/'preflight.json').exists():return pilot.read(out/'preflight.json')
    cfg=pilot.read(out/'frozen.json');config=cfg['config'];pilot.activate(upstream,config)
    for name,version in cfg['packages'].items():
        assert importlib.metadata.version(name)==version,(name,version)
    def forbidden(*args,**kwargs):raise AssertionError('No solve or physics step is authorized during preflight')
    folder=out/'preflight';folder.mkdir(exist_ok=True)
    with patch.object(c.Opti,'solve',forbidden),patch.object(c.Opti,'solve_limited',forbidden),patch.object(bullet,'stepSimulation',forbidden):
        retained=retained_kkt(root/BASE/'mmaq_v1',folder)
        factory=lambda:pilot.make_environment(upstream,config,np.asarray(cfg['cases']['H2']),1000)
        ctrl=pilot.make_controller(upstream,config,factory,'nonlinear20')
        try:
            assert ctrl.T==20 and ctrl.warmstart and not ctrl.soft_constraints and not ctrl.use_lqr_gain_and_terminal_cost
            assert ctrl.constraint_tol==1e-6
            np.testing.assert_array_equal(ctrl.Q,cfg['Q']);np.testing.assert_array_equal(ctrl.R,cfg['R'])
            opti=ctrl.opti_dict['opti'];before=problem_graph(opti)
            native_function=c.Function('archived_native_values',[opti.x,opti.p,opti.lam_g],
                [opti.f,opti.g,opti.lbg,opti.ubg,c.gradient(opti.f+c.dot(opti.g,opti.lam_g),opti.x)])
            own,_=independent_nlp_program(ctrl,cfg)
            history=np.load(root/BASE/'mmaq_v1/runs/H2_nonlinear20_1000_original/nlp_solutions.npz')
            errors=[]
            for z,params,lam in zip(history['primal'],history['parameters'],history['dual']):
                left=own(z,params,lam);right=native_function(z,params,lam)
                errors.append(matching_program_values(left,right))
                result=canonical_kkt(left[1],left[2],left[3],lam,left[4]);require_kkt(result,'Solve_Succeeded')
            # Test the real native graph/extractor API using an archived vector,
            # not a new optimizer call or fabricated historical rejected iterate.
            z,params,lam=history['primal'][-1],history['parameters'][-1],history['dual'][-1]
            def evaluate(expression):
                fn=c.Function('archived_expression',[opti.x,opti.p,opti.lam_g],[expression])
                return fn(z,params,lam)
            fake_solution=SimpleNamespace(value=evaluate)
            with patch.object(c.Opti,'stats',lambda self:{'success':True,'return_status':'Solve_Succeeded','provenance':'archived_substitution_not_new_solve'}):
                candidate=native_opti_snapshot(opti,fake_solution)
            assert not candidate['missing_fields'],candidate['capture_errors']
            guard=IndependentCandidateGuard(folder/'native_graph_capture',{'kind':'static_archived_substitution'},readout=own)
            guard.select(lambda:np.array([.1,.1]),lambda:candidate,lambda:{'states':np.zeros((1,6))})
            after=problem_graph(opti);assert before==after
            graph={'native_program_sha256':before,'native_graph_unchanged':True,'horizon':ctrl.T,
                'archived_H2_candidates_checked':len(errors),'maximum_native_independent_error':max(errors),
                'native_candidate_extractor_fields_complete':True,'original_IPOPT_options_unchanged':True}
            safe_write(folder/'graph_and_capture.json',graph)
        finally:ctrl.close()
    tests=pilot.read(out/'preflight_tests.json');assert tests['passed']
    result={'passed':True,'retained_historical_candidates_rechecked':retained['retained_candidates'],
        'graph_and_capture':graph,'tests_passed':True,'new_solver_calls':0,'new_physics_steps':0,
        'historical_outcomes_modified':False,'original_model_agreement_gate':'FAILED',
        'evidence_sha256':{str(p.relative_to(out)):pilot.sha(p) for p in folder.rglob('*') if p.is_file()}}
    immutable(out/'preflight.json',result);return result


def episode(root,upstream,out,phase):
    import casadi as c
    from src.quadrotor_diagnostic import problem_graph
    from src.quadrotor_mmaq_verification_v1 import independent_nlp_program
    out=Path(out);check_seal(root,out);assert pilot.read(out/'preflight.json')['passed']
    if phase not in ['original','repeat']:raise ValueError('Only two approved attempts')
    folder=out/'runs'/f'H2_nonlinear20_1000_{phase}'
    if (folder/'v2_execution_complete.json').exists():return pilot.read(folder/'completion.json')
    if folder.exists():raise RuntimeError('Partial execution retained; no automatic rerun')
    f=pilot.read(out/'frozen.json');native_factory=pilot.make_controller;native_capture=pilot.capture_physics;native_solve=c.Opti.solve
    active={};captures={};guards=[];verified_actions=[];predictions=[];native_calls=0

    @contextmanager
    def capture(env):
        import pybullet as bullet
        with native_capture(env) as data:
            captures['data']=data;step=bullet.stepSimulation
            data['stream_write_seconds']=[]
            with (folder/'partial_physics.f64').open('xb',buffering=0) as stream:
                stream.write(np.r_[data['states'][0],data['contacts'][0]].astype('<f8').tobytes())
                def instrument(*args,**kwargs):
                    count=len(data['states'])
                    value=step(*args,**kwargs)
                    if len(data['states'])>count:
                        start=time.perf_counter()
                        stream.write(np.r_[data['states'][-1],data['contacts'][-1]].astype('<f8').tobytes())
                        data['stream_write_seconds'].append(time.perf_counter()-start)
                    return value
                with patch.object(bullet,'stepSimulation',instrument):yield data

    def factory(*args,**kwargs):
        ctrl=native_factory(*args,**kwargs);native_select=ctrl.select_action
        readout,native_program=independent_nlp_program(ctrl,f);opti=ctrl.opti_dict['opti']
        graph=problem_graph(opti);assert graph==pilot.read(out/'preflight.json')['graph_and_capture']['native_program_sha256']
        gradient=c.gradient(opti.f+c.dot(opti.g,opti.lam_g),opti.x)
        active.update(ctrl=ctrl,opti=opti,gradient=gradient)
        def select(obs,info):
            k=len(guards);data=captures['data'];x=np.asarray(data['states'][-1]).copy()
            cache=data['states'][-2][4] if len(data['states'])>1 else x[4]
            guard=IndependentCandidateGuard(folder/'attempts'/f'{k:04d}',{
                'case':'H2','controller':'nonlinear20','phase':phase,'control_step':k,
                'native_reference_step':info['current_step'],'current_raw_state':x,'cache':cache,
                'protocol_sha256':pilot.sha(out/'protocol_seal.json'),'program_sha256':graph},readout=readout)
            guards.append(guard);active['guard']=guard
            def snapshot():
                return {'physics_states':np.asarray(data['states']),'physics_contacts':np.asarray(data['contacts']),
                        'verified_actions':np.asarray(verified_actions),'applied_controls':np.array((len(data['states'])-1)//20)}
            action=guard.select(lambda:native_select(obs,info),lambda:guard.current_candidate,snapshot)
            with guard.timing.span('prediction_diagnostics'):
                b=bound_components(x,action,cache,f)
                predicted=rk4(x,b['model_thrust'],f);source=source_prediction(x,b['executed_thrust'],cache,f)
                reserve=np.minimum(predicted-f['state_low'],np.asarray(f['state_high'])-predicted)-b['bound']
                predictions.append({'step':k,'initial':x,'cache':cache,'prediction':predicted,'source_prediction':source,
                                    'reserve':reserve,**b})
            guard.state['timings']=guard.timing.records;guard.state['timing_summary']=guard.timing.summary()
            safe_write(guard.folder/'status.json',guard.state)
            safe_write(folder/'causal_contract.json',predictions)
            verified_actions.append(action.copy())
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>4294967296:raise MemoryError('Memory cap exceeded')
            return action
        ctrl.select_action=select;return ctrl

    def solve(opti,*args,**kwargs):
        nonlocal native_calls
        if opti is not active.get('opti'):raise AssertionError('Unexpected solver instance')
        if native_calls>=300:raise RuntimeError('Per-execution solver cap exhausted')
        with active['guard'].timing.span('resource_accounting'):
            native_calls+=1
            safe_write(folder/'progress.json',{'solver_attempts':native_calls,'physics_steps':len(captures['data']['states'])-1})
        return active['guard'].native_solve_hook(opti,lambda:native_solve(opti,*args,**kwargs),active['gradient'])

    with patch.object(pilot,'make_controller',factory),patch.object(pilot,'capture_physics',capture),\
         patch.object(pilot,'write',safe_write),patch.object(c.Opti,'solve',solve):
        result=pilot.run_episode(root,upstream,out,'H2','nonlinear20',phase,1000)
    result.update(version=VERSION,actual_native_solve_attempts=native_calls,
                  historical_v1_outcome='unresolved verifier rejection; unchanged')
    safe_write(folder/'completion.json',result)
    safe_write(folder/'v2_execution_complete.json',{'complete':result['complete'],'completion_sha256':pilot.sha(folder/'completion.json')})
    return result
