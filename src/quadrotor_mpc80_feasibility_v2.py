"""Separate v2 static instrumentation preflight; never executes a controller."""
from pathlib import Path
from contextlib import contextmanager,ExitStack
from unittest.mock import patch
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import numpy as np
from src import quadrotor_mpc80_feasibility_v1 as old
from src import quadrotor_pilot as p
from src.quadrotor_mmaq_v1 import immutable,safe_write
from src.quadrotor_applied_action_v2 import AppliedActionObserver
from src.quadrotor_action_mock_v2 import native_harness,trace_digest
from src.quadrotor_kkt_integrity_v2 import nlp_readout,canonical_kkt,require_kkt,matching_program_values

VERSION='E12_MPC80_FEAS_v2'
CONFIG='configs/experiment_12_mpc80_feas_v2.json'
SOURCES=[CONFIG,'src/quadrotor_applied_action_v2.py','src/quadrotor_action_mock_v2.py',
         'src/quadrotor_mpc80_feasibility_v2.py','experiments/12_mpc80_feasibility_v2_preflight.py',
         'tests/test_quadrotor_applied_action_v2.py','tests/test_quadrotor_mpc80_feas_v2.py',
         'docs/experiment_12_mpc80_feas_v2_protocol.md','requirements_experiment_12.txt']


def verify_parent(root,upstream):
    root,upstream=Path(root),Path(upstream);base=root/'results/benchmark_selection'
    pref=base/'mpc80_feas_v1_preflight';run=base/'mpc80_feas_v1_study'
    old.check_seal(root,pref)
    for folder,marker,key in [(pref,'evidence_integrity.json','artifacts_sha256'),
                              (run,'completion_integrity.json','artifacts_sha256')]:
        seal=p.read(folder/marker)
        for n,h in seal[key].items():assert p.sha(folder/n)==h,(folder,n)
    stopped=p.read(run/'scientific_summary.json');assert stopped['status']=='STOPPED_ON_FIRST_CHECK'
    assert stopped['actual_solver_calls']==1 and stopped['actual_physics_steps']==1
    ledger=p.read(run/'resource_ledger.json')
    assert ledger['halted'] and len(ledger['reservations'])==1 and ledger['reservations'][0]['status']=='failed'
    assert ledger['charged']['solver_calls']==300 and ledger['charged']['physics_steps']==6000
    contract=p.read(pref/'controller_contract.json')
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip()==contract['upstream_commit']
    for n,h in contract['upstream_sha256'].items():assert p.sha(upstream/n)==h,n
    for n,v in contract['packages'].items():assert importlib.metadata.version(n)==v,(n,v)
    binaries=p.read(pref/'native_dependency_identity.json')
    for n,item in binaries['core_native_binary_identities'].items():
        package='pybullet' if n.startswith('pybullet') else n.split('/')[0]
        assert p.sha(importlib.metadata.distribution(package).locate_file(n))==item['sha256']
    return pref,run,ledger


def validate_inheritance(root,c):
    root=Path(root);parent=p.read(root/c['parent_protocol']['path'])
    assert p.sha(root/c['parent_protocol']['path'])==c['parent_protocol']['sha256']
    for key in ['cohort','acceptance','stop','horizon','seed','phases','jobs_per_phase','ordering',
                'task_seconds','controller_frequency','physics_frequency','controls_per_execution',
                'physics_states_per_complete_execution','warm_start','controller_information']:
        assert c[key]==parent[key],key
    for key in ['solver_calls','physics_steps','aggregate_cpu_seconds','wall_seconds','memory_bytes',
                'workers','action_watchdog_seconds','reserve_before_dispatch','release_failed_reservations','automatic_retries']:
        assert c['budget'][key]==parent['budget'][key],key
    assert not c['execution_authorized'] and c['version']==VERSION
    assert c['original_overall_qualification']=='FAILED' and c['original_model_agreement']=={'threshold':.001,'status':'FAILED'}


@contextmanager
def zero_real_operations(upstream):
    """Mocks may execute native Python control flow; all real engine APIs blocked."""
    import sys
    sys.path.insert(0,str(upstream))
    import pybullet as b
    denied={}
    def fail(name):
        def call(*args,**kwargs):denied[name]=denied.get(name,0)+1;raise RuntimeError('Forbidden real engine call: '+name)
        return call
    with old.zero_operations() as blocked:
        with ExitStack() as stack:
            for n in ['connect','resetSimulation','loadURDF','applyExternalForce','applyExternalTorque']:
                stack.enter_context(patch.object(b,n,fail(n)))
            yield blocked,denied


def prepare(root,upstream,out):
    root,upstream,out=map(Path,(root,upstream,out))
    if out.exists():raise FileExistsError('V2 static output already exists; no overwrite')
    c=p.read(root/CONFIG);validate_inheritance(root,c)
    assert platform.python_version()=='3.10.19'
    pref,run,previous=verify_parent(root,upstream)
    assert p.sha(root/c['resource_carry']['prior_v1_ledger'])==c['resource_carry']['prior_v1_ledger_sha256']
    _,cohort=old.validate_cohort(root,c)
    preserved=old.capture_preservation(root,upstream,out)
    out.mkdir(parents=True)
    immutable(out/'protocol.json',c);immutable(out/'cohort_validation.json',cohort)
    immutable(out/'preservation_start.json',preserved);immutable(out/'jobs_plan.json',old.job_plan())
    for name in ['accepted_frozen.json','accepted_reference.npy','accepted_environment.lock.txt',
                 'controller_contract.json','native_dependency_identity.json']:
        (out/name).write_bytes((pref/name).read_bytes())
    (out/'prior_v1_ledger.json').write_bytes((run/'resource_ledger.json').read_bytes())
    immutable(out/'prior_v1_status.json',{'status':'FAILED execution, preserved without repair',
        'actual_solver_calls':1,'actual_physics_steps':1,'charged_solves':300,'charged_physics_steps':6000,
        'remaining_original_solves':12670,'remaining_original_physics_steps':942000,
        'state0_previously_partially_attempted':True,'never_fresh_learning_evaluation':True})
    immutable(out/'protocol_seal.json',{'version':VERSION,'authorization':'zero-solve preflight only',
        'source_sha256':{n:p.sha(root/n) for n in SOURCES},
        'artifact_sha256':{x.name:p.sha(x) for x in out.iterdir() if x.is_file()}})
    old.durable_json(out/'resource_ledger.json',{'version':VERSION,'execution_authorized':False,
        'preflight_passed':False,'halted':False,
        'prior_failed_reservations_preserved':{'solver_calls':300,'physics_steps':6000},
        'carried_CPU_seconds':previous['charged']['cpu_seconds'],'carried_active_wall_seconds':previous['charged']['wall_seconds'],
        'charged':{'solver_calls':0,'physics_steps':0,'cpu_seconds':previous['charged']['cpu_seconds'],
                   'wall_seconds':previous['charged']['wall_seconds']},
        'caps':c['budget'],'prospective_plan':{'solver_calls':9600,'physics_steps':576000},
        'remaining_original':{'solver_calls':12670,'physics_steps':942000},
        'after_prospective_plan':{'solver_calls':3070,'physics_steps':366000},'dispatched_jobs':[]})


def check_seal(root,out):
    s=p.read(Path(out)/'protocol_seal.json');assert s['version']==VERSION
    for n,h in s['source_sha256'].items():assert p.sha(Path(root)/n)==h,n
    for n,h in s['artifact_sha256'].items():assert p.sha(Path(out)/n)==h,n
    return s


def archive_and_mock_checks(root,upstream,out,evidence):
    """Pinned native force/control flow on scripted archives, not simulated states."""
    root,out,evidence=Path(root),Path(out),Path(evidence);evidence.mkdir(exist_ok=False)
    model=p.read(out/'accepted_frozen.json');base=root/'results/benchmark_selection';rows=[]
    native_path=base/'mpc80_feas_v1_preflight/checks_first/native_mpc80_readout.casadi'
    import casadi as c
    native=c.Function.load(str(native_path));own=nlp_readout(model,80)
    graph_error=0.;candidate_count=0;closure=0.;mock_steps=0;max_force_difference=0.
    for folder in sorted((base/'mmaq_v1/runs').glob('*_nonlinear80_*')):
        points=np.load(folder/'nlp_solutions.npz');m=p.read(folder/'completion.json')
        for k,(z,params,lam) in enumerate(zip(points['primal'],points['parameters'],points['dual'])):
            values=own(z,params,lam);graph_error=max(graph_error,matching_program_values(values,native(z,params,lam)))
            d=canonical_kkt(values[1],values[2],values[3],lam,values[4]);require_kkt(d,m['solver_statuses'][k]);candidate_count+=1
    assert candidate_count==2400
    for state in ['H0','H1','H2','H3']:
        folder=base/'mmaq_v1/runs'/f'{state}_nonlinear80_1000_original';a=np.load(folder/'trajectory.npz')
        commands=a['actions'][:2].copy();states=a['physics_states'][:41].copy();expected=a['rpms'][:2].copy()
        observations=[];traces=[];finals=[]
        # Scripted, unmodified-native-control-flow fixtures, not scientific runs.
        for observed in [False,True]:
            with native_harness(upstream,model,states) as (env,b):
                if observed:
                    def after(packet):
                        k=packet['physics_index'];u=packet['grouped_submitted_force_argument_sums']
                        rpm=packet['direct_rpm_argument'];np.testing.assert_array_equal(rpm,expected[k//20])
                        x,y=states[k],states[k+1];cache=states[max(0,k-1),4];h=.001
                        v=x[[1,3,5]]+h*np.array([u.sum()/model['mass']*np.sin(cache),
                            u.sum()/model['mass']*np.cos(cache)-model['gravity'],
                            .028/model['Iyy']*(u[1]-u[0])*np.cos(x[4]-cache)])
                        prediction=x.copy();prediction[[1,3,5]]=v;prediction[[0,2,4]]=x[[0,2,4]]+h*v
                        error=float(np.max(np.abs(prediction-y)))
                        observations.append({'physics_index':k,'direct_mock_boundary_rpm':rpm,
                            'grouped_submitted_force_argument_sums':u,'source_equation_error':error,
                            'provenance':'mocked native application of archived commands; no live telemetry generated'})
                    with AppliedActionObserver(env,b,after_step=after) as observer:
                        for command in commands:env._advance_simulation(env._preprocess_control(command))
                        assert len(observer.steps)==40
                        for k,packet in enumerate(observer.steps):
                            np.testing.assert_array_equal(packet['preprocess_record']['requested_argument'],commands[k//20])
                        max_force_difference=max(max_force_difference,max(float(np.max(np.abs(pkt['grouped_submitted_force_argument_sums']-
                            pkt['derived_grouped_thrust_from_direct_rpm']))) for pkt in observer.steps))
                else:
                    for command in commands:env._advance_simulation(env._preprocess_control(command))
                traces.append(trace_digest(b.trace));finals.append((env.last_clipped_action.tobytes(),env.update_count,b.states.tobytes()))
        assert traces[0]==traces[1] and finals[0]==finals[1]
        current=max(r['source_equation_error'] for r in observations);assert current<=1e-10
        closure=max(closure,current);mock_steps+=len(observations)
        safe_write(evidence/(state+'_mock_records.json'),observations)
        rows.append({'archived_state':state,'matched_mock_boundaries':40,'native_trace_bitwise_same':True,
            'scripted_state_bytes_same':True,'controller_commands_preserved':True,
            'trace_sha256':traces[0],'maximum_source_equation_error':current})
    # Reuse unchanged accepted source-based discrepancy machinery; no new states.
    mismatch=old.archived_contract(root,out)
    return {'passed':True,'native_control_flow_mock_rows':rows,'mock_boundary_count':mock_steps,
        'actual_new_PyBullet_steps':0,'archived_MPC80_candidates':candidate_count,
        'native_independent_graph_error':graph_error,'maximum_mock_force_source_recurrence_error':closure,
        'maximum_direct_force_sum_vs_KF_derived_group_difference':max_force_difference,
        'archived_discrepancy_contract':mismatch,
        'new_live_trajectory_invariance_measured':False,
        'invariance_scope':'exact native call forwarding, native mock traces and archived scripted state bytes; not new empirical physics replay',
        'direct_v1_missing_live_RPM_recovered':False}


def validate(root,upstream,out,evidence):
    root,out=Path(root),Path(out);check_seal(root,out);c=p.read(out/'protocol.json');validate_inheritance(root,c)
    verify_parent(root,upstream);_,cohort=old.validate_cohort(root,c)
    with zero_real_operations(upstream) as (blocked,denied):
        checks=archive_and_mock_checks(root,upstream,out,evidence)
    assert not blocked and not denied
    preserved=old.verify_preservation(root,upstream,p.read(out/'preservation_start.json'))
    return {'version':VERSION,'zero_solve_preflight_passed':True,'cohort':cohort,'checks':checks,
        'preservation':preserved,'forbidden_optimizer_or_step_calls':blocked,'forbidden_engine_api_calls':denied,
        'execution_authorized':False,'v2_states_executed':0,'new_solves':0,'new_physics_steps':0,
        'v1_execution':'FAILED, unchanged; no retry or overwrite',
        'original_overall_E12_qualification':'FAILED','original_0001_agreement':'FAILED',
        'observability':'native motor and submitted-force API boundaries are directly observable; internal engine/hardware force sensing not claimed',
        'live_validation_limit':'new live scientific-trajectory bitwise invariance not measured under zero-simulation authorization',
        'ready_for_separate_execution_review':True}
