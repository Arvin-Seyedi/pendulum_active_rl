"""E12_MPC80_FEAS_v1 static preflight. No cohort executor exists here.

Native graph construction and archived-vector evaluation are permitted; native
optimization, action selection, environment stepping and Bullet integration are
blocked. Historical implementations and evidence are read-only inputs.
"""
from contextlib import contextmanager, ExitStack
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
import numpy as np
from src.quadrotor_finalization import BASE, IQL, preservation, physics_reconstruction
from src.quadrotor_pilot import read, sha
from src.quadrotor_mmaq_v1 import immutable, safe_write, tagged
from src.quadrotor_kkt_integrity_v2 import canonical_kkt, require_kkt, matching_program_values
from src.quadrotor_mmaq_verification_v1 import independent_nlp_program, exact_majorant
from src.quadrotor_h2_requalification_v2 import IndependentCandidateGuard
from src.quadrotor_qualification_telemetry_v2 import QualificationFailure, native_opti_snapshot
from src.quadrotor_mismatch_contract_v1 import bound_components, source_prediction
from src.quadrotor_verification import audit_record, rk4

VERSION = 'E12_MPC80_FEAS_v1'
CONFIG = 'configs/experiment_12_mpc80_feas_v1.json'
SOURCES = [CONFIG, 'src/quadrotor_mpc80_feasibility_v1.py',
           'experiments/12_mpc80_feasibility_preflight.py',
           'tests/test_quadrotor_mpc80_feasibility_v1.py',
           'docs/experiment_12_mpc80_feas_v1_protocol.md', 'requirements_experiment_12.txt']


def validate_cohort(root, config):
    """Load originals; independently check raw bits, never write/redraw inputs."""
    root=Path(root);c=config['cohort'];mp=root/c['manifest'];ap=root/c['array']
    if sha(mp)!=c['manifest_sha256'] or sha(ap)!=c['array_sha256']:
        raise AssertionError('Original cohort identity mismatch')
    manifest=read(mp);a=np.load(ap,allow_pickle=False)
    assert a.shape==(16,6) and a.dtype==np.dtype('float64') and np.isfinite(a).all()
    assert manifest['seed']==120001 and manifest['bit_generator']=='PCG64'
    assert [s['id'] for s in manifest['states']]==list(range(16))
    assert len(manifest['draws'])==16 and all(d['accepted'] for d in manifest['draws'])
    assert manifest['draw_order']==c['draw_order'] and manifest['evaluated'] is False
    # A check against independent raw-bit conversion, not a replacement dataset.
    bits=np.random.PCG64(120001).random_raw((16,6))
    unit=(bits>>np.uint64(11)).astype(np.float64)*2.**-53
    checked=np.asarray(c['low'])+(np.asarray(c['high'])-c['low'])*unit
    assert a.tobytes()==checked.tobytes()
    for i,row in enumerate(manifest['states']):
        assert np.asarray(row['coordinates'],dtype='<f8').tobytes()==a[i].astype('<f8').tobytes()
        assert row['binary64_hex']==[float(v).hex() for v in a[i]]
        assert row['little_endian_bytes_hex']==a[i].astype('<f8').tobytes().hex()
        assert manifest['draws'][i]['draw']==i
        assert manifest['draws'][i]['unit_hex']==[float(v).hex() for v in unit[i]]
    assert len({x.tobytes() for x in a})==16
    return a, {'passed':True,'manifest_sha256':sha(mp),'array_sha256':sha(ap),
               'coordinate_count':96,'ids':list(range(16)), 'draw_order':c['draw_order'],
               'verification_only_raw_bit_reconstruction':True,'authoritative_files_rewritten':False,
               'cohort_role':c['role'],'independent_learning_evaluation_defined':False}


def require_distinct_evaluation(evaluation, qualification):
    """Coordinate separation is necessary, not sufficient for an independent set."""
    old={np.asarray(x,dtype='<f8').tobytes() for x in qualification}
    if any(np.asarray(x,dtype='<f8').tobytes() in old for x in evaluation):
        raise ValueError('Qualification coordinates cannot be independent learned-policy evaluation')


def job_plan():
    jobs=[]
    for state_id in range(16):
        for phase in ['original','repeat']:
            for kind in ['episode','native_replay','motor_replay']:
                jobs.append({'id':f'{state_id:02d}_{phase}_{kind}','state_id':state_id,
                             'phase':phase,'kind':kind,'solver_calls':300 if kind=='episode' else 0,
                             'physics_steps':6000})
    assert len(jobs)==96 and sum(j['solver_calls'] for j in jobs)==9600
    assert sum(j['physics_steps'] for j in jobs)==576000
    return jobs


def durable_json(path, value):
    """File fsync and atomic replacement precede a prospective dispatch receipt."""
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w') as stream:
        json.dump(tagged(value),stream,indent=2,allow_nan=False);stream.write('\n')
        stream.flush();os.fsync(stream.fileno())
    tmp.replace(path)


def reserve_job(path, job_id, approval):
    """Account ONLY; never dispatch. Requires a separate reviewed approval record.

    Failed reservations cannot be released or retried. A future executor must
    receive this function's persisted receipt before performing any work.
    No such approval is created by preflight or this repository entry point.
    """
    path=Path(path);s=read(path)
    if not s['preflight_passed'] or not isinstance(approval,dict):raise PermissionError('Execution not approved')
    if not (approval.get('execution_approved') is True and approval.get('seal_sha256')==s['seal_sha256']
            and approval.get('preflight_report_sha256')==s['preflight_report_sha256']
            and approval.get('review_reference')):raise PermissionError('Approval is not bound to this preflight')
    if s['halted']:raise RuntimeError('Failed/interrupted job retained; campaign halted')
    n=len(s['reservations']);jobs=s['jobs']
    if n>=len(jobs) or jobs[n]['id']!=job_id:raise RuntimeError('Out of order or repeated job')
    if any(v['status']!='completed' for v in s['reservations']):raise RuntimeError('Previous reserved job incomplete')
    j=jobs[n];usage=s['charged'];cap=s['caps']
    if not np.isfinite([usage['cpu_seconds'],usage['wall_seconds']]).all():raise RuntimeError('Nonfinite resource accounting')
    if usage['cpu_seconds']>=cap['aggregate_cpu_seconds'] or usage['wall_seconds']>=cap['wall_seconds']:
        raise RuntimeError('Overhead consumed execution budget')
    for k in ['solver_calls','physics_steps']:
        if usage[k]+j[k]>cap[k]:raise RuntimeError('Resource cap exceeded')
        usage[k]+=j[k]
    receipt={**j,'status':'reserved','approval_review_reference':approval['review_reference']}
    s['reservations'].append(receipt);durable_json(path,s)
    return receipt


def finish_job(path, job_id, success):
    s=read(path)
    if not s['reservations'] or s['reservations'][-1]['id']!=job_id:
        raise RuntimeError('No matching reservation')
    row=s['reservations'][-1]
    if row['status']!='reserved':raise RuntimeError('Completion already recorded')
    row['status']='completed' if success else 'failed';s['halted']=s['halted'] or not success
    durable_json(path,s)  # No resource refund, including zero-transition failures.


def charge_overhead(path, cpu, wall):
    if not np.isfinite([cpu,wall]).all() or min(cpu,wall)<0:raise ValueError('Invalid overhead')
    s=read(path);s['charged']['cpu_seconds']+=cpu;s['charged']['wall_seconds']+=wall
    exhausted=(s['charged']['cpu_seconds']>s['caps']['aggregate_cpu_seconds'] or
               s['charged']['wall_seconds']>s['caps']['wall_seconds'])
    if exhausted:s['halted']=True
    durable_json(path,s)
    if exhausted:raise RuntimeError('Resource budget exhausted; recorded before stopping')


@contextmanager
def zero_operations(controller_apis=True):
    """Block integration/optimization/selection, including native API entry points."""
    import casadi as c
    import pybullet as b
    counts={}
    def forbidden(name):
        def fail(*args,**kwargs):
            counts[name]=counts.get(name,0)+1
            raise RuntimeError('Forbidden during zero-solve preflight: '+name)
        return fail
    with ExitStack() as stack:
        for obj,name,label in [(c.Opti,'solve','Opti.solve'),(c.Opti,'solve_limited','Opti.solve_limited'),
                               (c,'nlpsol','nlpsol'),(c,'qpsol','qpsol'),(c,'conic','conic'),
                               (b,'stepSimulation','stepSimulation')]:
            stack.enter_context(patch.object(obj,name,forbidden(label)))
        if controller_apis:
            from safe_control_gym.controllers.mpc.mpc import MPC
            from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor
            stack.enter_context(patch.object(MPC,'select_action',forbidden('MPC.select_action')))
            stack.enter_context(patch.object(Quadrotor,'step',forbidden('Quadrotor.step')))
        yield counts


def check_seal(root,out):
    seal=read(Path(out)/'protocol_seal.json')
    assert seal['version']==VERSION and seal['authorization']=='preflight only'
    for n,h in seal['source_sha256'].items():assert sha(Path(root)/n)==h,n
    for n,h in seal['artifact_sha256'].items():assert sha(Path(out)/n)==h,n
    return seal


def capture_preservation(root,upstream,out):
    root,out=Path(root),Path(out).resolve()
    original=preservation(root,root,upstream)
    names=subprocess.check_output(['git','ls-files'],cwd=root,text=True).splitlines()
    records={}
    for sub in ['results','data']:
        for p in (root/sub).rglob('*'):
            if p.is_file() and not p.resolve().is_relative_to(out):records[str(p.relative_to(root))]=sha(p)
    return {'original_verified':original,'tracked_sha256':{n:sha(root/n) for n in names},
            'all_prior_result_and_data_sha256':records,
            'project_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()}


def verify_preservation(root,upstream,snapshot):
    root=Path(root)
    for group in ['tracked_sha256','all_prior_result_and_data_sha256']:
        for n,h in snapshot[group].items():assert sha(root/n)==h,n
    again=preservation(root,root,upstream)
    assert again==snapshot['original_verified']
    return {'passed':True,'tracked_files':len(snapshot['tracked_sha256']),
            'prior_result_and_data_files':len(snapshot['all_prior_result_and_data_sha256']),
            'authority_files':again['authority_files_checked'],'upstream_files':again['upstream_files_checked'],
            'IQL_files':len(IQL),'accepted_artifacts':again['accepted_artifacts_checked']}


def prepare(root,upstream,out):
    root,out,upstream=map(Path,(root,out,upstream))
    if (out/'protocol_seal.json').exists():check_seal(root,out);return
    if out.exists():raise FileExistsError('Refusing partial/existing preflight output: '+str(out))
    # Do not allow any accepted evidence directory as a destination.
    from src.quadrotor_finalization import destination
    destination(out/'protocol_seal.json',root)
    c=read(root/CONFIG);assert c['version']==VERSION and not c['execution_authorized']
    _,cohort=validate_cohort(root,c)
    assert platform.python_version()==c['python_version']
    previous=root/BASE/'mmaq_v1';f=read(previous/'frozen.json')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()==c['upstream_commit']
    assert not subprocess.check_output(['git','diff','HEAD','--name-only'],cwd=upstream,text=True).strip()
    for name,version in f['packages'].items():assert importlib.metadata.version(name)==version,(name,version)
    before=capture_preservation(root,upstream,out)
    out.mkdir(parents=True)
    immutable(out/'protocol.json',c);immutable(out/'cohort_validation.json',cohort)
    immutable(out/'preservation_start.json',before);immutable(out/'jobs_plan.json',job_plan())
    for name in ['frozen.json','reference.npy','environment.lock.txt']:
        (out/('accepted_'+name)).write_bytes((previous/name).read_bytes())
    usage=read(root/BASE/'H2_MPC20_requalification_v2/resource_accounting.json')
    assert usage['remaining_conservative_original_budget']['solver_calls']==12970
    assert usage['remaining_conservative_original_budget']['physics_steps']==948000
    immutable(out/'prior_budget.json',{'source_sha256':sha(root/BASE/'H2_MPC20_requalification_v2/resource_accounting.json'),
        'accepted_prior_balance':usage['remaining_conservative_original_budget'],
        'CPU_balance_note':'historical ledger balance, not an exact balance after all later finalization overhead',
        'prospective_remaining_after_full_plan':{'solver_calls':3370,'physics_steps':372000}})
    # Explicitly freeze native source, all model/vehicle inputs and method semantics.
    native=['safe_control_gym/controllers/mpc/mpc.py','safe_control_gym/controllers/mpc/mpc_utils.py',
            f['config']['task_config'],f['config']['nonlinear_config']]
    contract={'upstream_commit':c['upstream_commit'],'upstream_sha256':f['upstream_files'],
              'native_configuration_sha256':{n:sha(upstream/n) for n in native},
              'archived_frozen_sha256':sha(previous/'frozen.json'), 'packages':f['packages'],
              'plant_and_task':{k:f[k] for k in ['mass','Iyy','arm_length','gravity','KF','PWM2RPM_SCALE',
                  'PWM2RPM_CONST','MIN_PWM','MAX_PWM','state_low','state_high','action_low','action_high','Q','R','hover_input']},
              'solver':'ipopt','solver_options':{'expand':True,'error_on_fail':False},
              'horizon':80,'warmstart':True,'soft_constraints':False,'constraint_tol':1e-6,
              'use_lqr_gain_and_terminal_cost':False,'initial_x_prev':None,'initial_u_prev':None,
              'cohort_role':'qualification/development; not independent learned-policy evaluation',
              'future_learning_evaluation':'not defined; must be prospectively independent',
              'original_model_agreement':'FAILED at unchanged 0.001','overall_E12_qualification':'FAILED'}
    immutable(out/'controller_contract.json',contract)
    artifacts=[p for p in out.iterdir() if p.is_file()]
    immutable(out/'protocol_seal.json',{'version':VERSION,'authorization':'preflight only',
        'source_sha256':{n:sha(root/n) for n in SOURCES},
        'artifact_sha256':{p.name:sha(p) for p in artifacts}})
    durable_json(out/'resource_ledger.json',{'version':VERSION,'seal_sha256':sha(out/'protocol_seal.json'),
        'preflight_passed':False,'preflight_report_sha256':None,'execution_approval_present':False,
        'halted':False,'jobs':job_plan(),'caps':c['budget'],'reservations':[],
        'earmarked':{'solver_calls':9600,'physics_steps':576000},
        'charged':{'solver_calls':0,'physics_steps':0,'cpu_seconds':0.,'wall_seconds':0.}})


def graph_and_capture(root,upstream,out,evidence):
    """Construct native graph, evaluate archived candidates; no solve/step/select."""
    import casadi as c
    from src import quadrotor_pilot as p
    from src.quadrotor_diagnostic import problem_graph
    out,evidence=Path(out),Path(evidence);evidence.mkdir(exist_ok=False)
    f=read(out/'accepted_frozen.json');config=f['config'];p.activate(upstream,config)
    factory=lambda:p.make_environment(upstream,config,np.asarray(f['cases']['H0']),1000)
    dispatch=[];native_solver=c.Opti.solver
    def spy(opti,solver,options):
        dispatch.append({'solver':solver,'options':options.copy()});return native_solver(opti,solver,options)
    with patch.object(c.Opti,'solver',spy):ctrl=p.make_controller(upstream,config,factory,'nonlinear80')
    try:
        assert ctrl.T==80 and ctrl.warmstart and ctrl.solver=='ipopt'
        assert not ctrl.soft_constraints and not ctrl.use_lqr_gain_and_terminal_cost and ctrl.constraint_tol==1e-6
        assert ctrl.x_prev is None and ctrl.u_prev is None
        np.testing.assert_array_equal(ctrl.Q,f['Q']);np.testing.assert_array_equal(ctrl.R,f['R'])
        assert dispatch==[{'solver':'ipopt','options':{'expand':True,'error_on_fail':False}}]
        opti=ctrl.opti_dict['opti'];before=problem_graph(opti)
        assert (opti.nx,opti.np,opti.ng)==(646,492,1778)
        own,_=independent_nlp_program(ctrl,f)
        native=c.Function('native_readout',[opti.x,opti.p,opti.lam_g],
            [opti.f,opti.g,opti.lbg,opti.ubg,c.gradient(opti.f+c.dot(opti.g,opti.lam_g),opti.x)])
        reference=np.load(out/'accepted_reference.npy');maximum=0.;maxima={};count=0;representative=None
        for folder in sorted((Path(root)/BASE/'mmaq_v1/runs').glob('*_nonlinear80_*')):
            m=read(folder/'completion.json');data=np.load(folder/'nlp_solutions.npz');a=np.load(folder/'trajectory.npz')
            assert m['complete'] and len(data['primal'])==300
            for k,(z,params,lam) in enumerate(zip(data['primal'],data['parameters'],data['dual'])):
                expected=reference[np.minimum(k+np.arange(81),300)].T.reshape(-1,order='F')
                assert np.array_equal(params[6:],expected)
                assert np.max(np.abs(params[:6]-a['states'][k]))<1e-14
                left=own(z,params,lam);right=native(z,params,lam)
                maximum=max(maximum,matching_program_values(left,right))
                d=canonical_kkt(left[1],left[2],left[3],lam,left[4]);require_kkt(d,m['solver_statuses'][k])
                for key,value in d['metrics'].items():maxima[key]=max(maxima.get(key,0.),value)
                count+=1
                if representative is None:representative=(z,params,lam,a,m,folder)
        assert count==2400 and before==problem_graph(opti)
        native.save(str(evidence/'native_mpc80_readout.casadi'))
        # Exercise real extractor on a known archived vector, not a live solution.
        z,params,lam,a,m,folder=representative
        def evaluate(expression):return c.Function('archived_value',[opti.x,opti.p,opti.lam_g],[expression])(z,params,lam)
        stats={'success':True,'return_status':m['solver_statuses'][0],'provenance':'archived substitution; no new solve'}
        with patch.object(c.Opti,'stats',lambda self:stats):candidate=native_opti_snapshot(opti,SimpleNamespace(value=evaluate))
        assert not candidate['missing_fields']
        candidate['provenance']='archived_native_graph_substitution_not_new_solver_result'
        snapshot=lambda:{'physics_states':a['physics_states'][:21], 'physics_contacts':a['physics_contacts'][:21]}
        good=IndependentCandidateGuard(evidence/'capture_static_valid',{'source':str(folder)},readout=own)
        action=good.select(lambda:a['actions'][0].copy(),lambda:candidate,snapshot)
        assert np.array_equal(action,a['actions'][0])
        bad=copy.deepcopy(candidate);bad['primal']=bad['primal'].copy();bad['primal'].flat[0]=np.nan
        bad['objective']=np.array([np.inf]);bad['solver_status']='unknown'
        bad['constraint_values']=bad['constraint_values'].copy();bad['constraint_values'].flat[0]=np.nan
        rejected=IndependentCandidateGuard(evidence/'capture_static_failure',{'source':'injected nonfinite static fixture'},readout=own)
        def fail():raise RuntimeError('Static synthetic failure, not a controller invocation')
        try:rejected.select(fail,lambda:bad,snapshot)
        except QualificationFailure:pass
        else:raise AssertionError('Invalid candidate was accepted')
        saved=np.load(rejected.folder/'candidate.npz');status=read(rejected.folder/'status.json')
        assert np.isnan(saved['primal'].flat[0]) and np.isposinf(saved['objective']).all()
        assert not status['fallback_action_applied'] and status['status']=='failed_or_incomplete'
        assert status['residuals']['primal_inf']=={'nonfinite_float':'inf'}
        assert (rejected.folder/'prefix_after.npz').exists() and status['timing_summary']['native_action_selection_measured']
        return {'passed':True,'native_program_sha256':before,'native_readout_sha256':sha(evidence/'native_mpc80_readout.casadi'),
                'dimensions':{'variables':646,'parameters':492,'constraint_rows':1778},'dispatch':dispatch,
                'archived_candidates':count,'maximum_program_reconstruction_error':maximum,'KKT_maxima':maxima,
                'native_graph_unchanged':True,'static_native_extractor_and_failure_capture_passed':True,
                'new_solver_calls':0,'new_physics_steps':0,
                'setup_note':'One disposable native graph/environment at archived H0; no selected action or integrated state'}
    finally:ctrl.close()


def archived_contract(root,out):
    """Recheck every archived MPC80 interval, including both physics resolutions."""
    root,out=Path(root),Path(out);f=read(out/'accepted_frozen.json');ref=np.load(out/'accepted_reference.npy')
    rows=[];total=0;exact_count=0;max_closure=0.;max_excess=-np.inf;min_reserve=np.full(6,np.inf)
    for sub in ['technical_pilot','mmaq_v1']:
        for folder in sorted((root/BASE/sub/'runs').glob('*_nonlinear80_*')):
            m=read(folder/'completion.json');r,a=audit_record(folder,read(root/BASE/sub/'frozen.json'),ref)
            assert m['complete'] and r['safe_complete_no_contact']
            physics_reconstruction(a,f,m['frequency'])
            hz=m['frequency'];stride=hz//50;reserves=[];closure=0.;excess=-np.inf
            for k in range(300):
                x=a['states'][k];cache=a['physics_states'][max(0,k*stride-1),4]
                b=bound_components(x,a['actions'][k],cache,f,frequency=hz)
                exact=exact_majorant(x,b['model_thrust'],b['executed_thrust'],cache,f,frequency=hz)
                assert all(Fraction(float(v))>=e for v,e in zip(b['bound'],exact));exact_count+=6
                predicted=rk4(x,b['model_thrust'],f)
                err=np.abs(predicted-a['states'][k+1]);excess=max(excess,float(np.max(err-b['bound'])))
                close=float(np.max(np.abs(source_prediction(x,b['executed_thrust'],cache,f,hz)-a['states'][k+1])))
                closure=max(closure,close)
                reserve=np.minimum(predicted-f['state_low'],np.asarray(f['state_high'])-predicted)-b['bound']
                reserves.append(reserve);total+=1
            reserves=np.asarray(reserves);assert excess<=1e-10 and closure<=1e-10
            assert np.all(reserves>0) and np.max(np.abs(a['physics_states'][:,4]))<np.pi/2
            min_reserve=np.minimum(min_reserve,reserves.min(0));max_closure=max(max_closure,closure);max_excess=max(max_excess,excess)
            rows.append({'source':str(folder.relative_to(root)),'controls':300,'frequency':hz,
                         'original_0001_passed':r['one_step_tolerance_passed'],
                         'maximum_original_discrepancy':r['symbolic_pybullet_one_step_max_scaled'],
                         'minimum_reserve_by_component':reserves.min(0).tolist(),'closure':closure,'excess':excess})
    assert len(rows)==24 and total==7200
    return {'passed':True,'records':rows,'archived_intervals':total,'exact_component_checks':exact_count,
            'source_closure_maximum':max_closure,'maximum_bound_excess':max_excess,
            'minimum_reserve_by_component':min_reserve.tolist(),
            'original_0001_gate':'FAILED','overall_E12_qualification':'FAILED',
            'contract_scope':'declared contact-free planar recurrence; empirical Bullet closure, not formal robust safety'}


def validate(root,upstream,out,evidence):
    root,out=Path(root),Path(out);check_seal(root,out)
    _,cohort=validate_cohort(root,read(out/'protocol.json'))
    heldout=np.load(root/BASE/'mmaq_v1/heldout_states.npy')
    qualifications=np.load(root/read(out/'protocol.json')['cohort']['array'])
    require_distinct_evaluation(heldout,qualifications)
    # This proves separation of old cohorts, not future evaluation independence.
    from src.quadrotor_pilot import activate
    activate(upstream,read(out/'accepted_frozen.json')['config'])
    with zero_operations() as blocked:
        graph=graph_and_capture(root,upstream,out,evidence)
        contract=archived_contract(root,out)
    assert not blocked,blocked
    preserved=verify_preservation(root,upstream,read(out/'preservation_start.json'))
    return {'version':VERSION,'passed':True,'cohort':cohort,'graph':graph,'contract':contract,
            'preservation':preserved,'forbidden_native_calls':blocked,
            'new_solver_calls':0,'new_physics_steps':0,'new_episodes':0,'cohort_states_evaluated':0,
            'execution_authorized':False,'ready_for_separate_execution_review':True,
            'original_overall_qualification':'FAILED','original_0001_agreement':'FAILED',
            'future_learning_evaluation':'not defined; qualification cohort reuse prohibited'}
