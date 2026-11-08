"""Read-only mathematical audit of the stopped MPC80 feasibility attempt.

Never creates a simulator/controller or calls an optimizer. The stale-cache
finding is kept separate from source-derived commands and recorded raw states.
No absent RPM/reward/replay evidence is manufactured or inserted into the run.
"""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np
from src import quadrotor_pilot as p
from src import quadrotor_mpc80_feasibility_v1 as f
from src.quadrotor_kkt_integrity_v2 import canonical_kkt,require_kkt,matching_program_values,nlp_readout
from src.quadrotor_mismatch_contract_v1 import executed_thrust
from src.quadrotor_mmaq_verification_v1 import exact_majorant
from fractions import Fraction


def method_source(path,name):
    text=Path(path).read_text();tree=ast.parse(text)
    node=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name==name)
    return {'path':str(path),'start_line':node.lineno,'end_line':node.end_lineno,
            'text':ast.get_source_segment(text,node),'file_sha256':p.sha(path)}


def audit(root,upstream,pref,out):
    import casadi as c
    root,upstream,pref,out=map(Path,(root,upstream,pref,out))
    from src.quadrotor_mpc80_execution_v1 import checkpoint
    checkpoint(root,upstream,pref,out)
    evidence=p.read(pref/'evidence_integrity.json')
    for n,h in evidence['artifacts_sha256'].items():assert p.sha(pref/n)==h,n
    assert p.read(out/'campaign_stop.json')['first_failure_stops_campaign']
    ledger=p.read(out/'resource_ledger.json');assert ledger['halted']
    assert len(ledger['reservations'])==1 and ledger['reservations'][0]['status']=='failed'
    assert ledger['charged']['solver_calls']==300 and ledger['charged']['physics_steps']==6000
    folder=out/'runs/0_nonlinear80_1000_original';m=p.read(folder/'completion.json')
    assert not m['complete'] and m['transitions']==0 and m['actual_native_solve_attempts']==1 and m['actual_physics_steps']==1
    assert p.sha(folder/'trajectory.npz')==m['trajectory_sha256']
    model=p.read(out/'frozen.json');a=np.load(folder/'trajectory.npz');status=p.read(folder/'attempts/0000/status.json')
    candidate=np.load(folder/'attempts/0000/candidate.npz')
    assert p.sha(folder/'attempts/0000/candidate.npz')==status['candidate_sha256']
    assert not status['missing_fields'] and status['candidate_metadata']['solver_status']=='Solve_Succeeded'
    assert not status['fallback_action_applied'] and a['physics_states'].shape==(2,6)
    assert a['actions'].shape==(1,2) and len(a['rewards'])==0 and len(a['rpms'])==0
    stream=np.fromfile(folder/'partial_physics.f64',dtype='<f8').reshape(-1,7)
    assert stream.shape==(2,7) and stream[:,:6].tobytes()==a['physics_states'].tobytes()
    assert np.array_equal(stream[:,6],a['physics_contacts'])
    z,params,lam=[candidate[n].reshape(-1) for n in ['primal','parameters','dual']]
    independent=nlp_readout(model,80)
    native_path=pref/'checks_first/native_mpc80_readout.casadi'
    assert p.sha(native_path)==p.read(pref/'preflight.json')['graph']['native_readout_sha256']
    native=c.Function.load(str(native_path));own=independent(z,params,lam)
    reconstruction=matching_program_values(own,native(z,params,lam))
    matching_program_values(own,[candidate[n] for n in ['objective','constraint_values','lower','upper','lagrangian_gradient']])
    kkt=canonical_kkt(own[1],own[2],own[3],lam,own[4]);require_kkt(kkt,'Solve_Succeeded')
    assert np.isfinite(np.r_[z,params,lam]).all()
    causal=p.read(folder/'causal_contract.json');assert len(causal)==1
    b=causal[0];u=a['actions'][0]
    assert np.array_equal(z[486:488],u)
    assert np.array_equal(params[:6],a['physics_states'][0])
    clipped,expected,rpm=executed_thrust(u,model)
    assert np.array_equal(expected,b['executed_thrust']) and np.array_equal(rpm,b['rpm'])
    exact=exact_majorant(a['physics_states'][0],b['model_thrust'],b['executed_thrust'],b['cache'],model)
    assert all(Fraction(float(v))>=e for v,e in zip(b['bound'],exact))
    assert np.all(np.asarray(b['reserve'])>0)
    # One matched-state equation, NOT integration of another trajectory.
    x0,x1=a['physics_states'];h=.001;total=expected.sum()/model['mass']
    velocity=x0[[1,3,5]]+h*np.array([total*np.sin(x0[4]),total*np.cos(x0[4])-model['gravity'],
                                                 .028/model['Iyy']*(expected[1]-expected[0])])
    predicted=x0.copy();predicted[[1,3,5]]=velocity;predicted[[0,2,4]]=x0[[0,2,4]]+h*velocity
    closure=np.abs(predicted-x1);assert np.max(closure)<=1e-10
    zero_velocity=x0[[1,3,5]]+h*np.array([0.,-model['gravity'],0.])
    zero_difference=np.abs(zero_velocity-x1[[1,3,5]])
    low=a['physics_states']-model['state_low'];high=np.asarray(model['state_high'])-a['physics_states']
    assert np.all(low>=0) and np.all(high>=0) and not np.any(a['physics_contacts'])
    base=upstream/'safe_control_gym/envs/gym_pybullet_drones/base_aviary.py'
    quad=upstream/'safe_control_gym/envs/gym_pybullet_drones/quadrotor.py'
    advance=method_source(base,'_advance_simulation');preprocess=method_source(quad,'_preprocess_control')
    text=base.read_text();lines=text.splitlines()
    init_line=next(i+1 for i,s in enumerate(lines) if 'self.last_clipped_action = np.zeros(4)' in s)
    step_line=next(i+1 for i,s in enumerate(lines) if 'p.stepSimulation(physicsClientId=self.PYB_CLIENT)' in s)
    update_line=next(i+1 for i,s in enumerate(lines) if 'self.last_clipped_action = clipped_action' in s)
    assert init_line<step_line<update_line
    assert 'self._physics(clipped_action)' in advance['text']
    assert advance['text'].index('self._physics(clipped_action)')<advance['text'].index('p.stepSimulation(')
    source=(root/'src/quadrotor_mpc80_execution_v1.py').read_text()
    assert 'require_executed(env.last_clipped_action,model)' in source
    assert not any(out.glob('runs/*repeat'))
    assert not (folder/'replay_native.json').exists() and not (folder/'replay_instrumented.json').exists()
    native_seconds=status['timing_summary']['native_action_selection_wall']
    solver=[r for r in status['timings'] if r['category']=='native_solver'];assert len(solver)==1
    times={'solver_only':solver[0]['wall_inclusive'],'native_action':native_seconds,
           'supervised_callback':float(a['solver_wall'][0])}
    provenance=f.verify_preservation(root,upstream,p.read(out/'preservation_start.json'))
    source_truth={'classification':'new execution-instrumentation defect; not a demonstrated applied-input bound violation',
        'cached_field_name':'last_clipped_action','initialization_line':init_line,
        'physics_step_call_line':step_line,'cache_assignment_after_step_line':update_line,
        'cached_RPM_at_failure_source_inferred':[0.,0.,0.,0.],
        'direct_per_physics_applied_RPM_record_available':False,
        'requested_thrust':u.tolist(),'source_derived_executed_thrust':expected.tolist(),
        'source_derived_RPM':rpm.tolist(),
        'source_derived_lower_input_margins':(expected-model['action_low']).tolist(),
        'source_derived_upper_input_margins':(np.asarray(model['action_high'])-expected).tolist(),
        'invalid_cached_zero_thrust_margin':(-np.asarray(model['action_low'])).tolist(),
        'first_physics_equation_closure_by_state':closure.tolist(),
        'zero_RPM_velocity_discrepancy_by_component':zero_difference.tolist(),
        'missing_direct_RPM_not_retrospectively_inserted':True,'original_failure_artifacts_unchanged':True}
    rows=[{'state_id':i,'original':'incomplete after 1 physics step' if i==0 else 'not run',
           'repeat':'not run','return':None,'cost':None,'discounted_return':None,
           'full_horizon_tracking_RMSE':None,'full_horizon_effort':None} for i in range(16)]
    report={'audit_version':'E12_MPC80_FEAS_v1_stopped_record_audit_1','execution_gate_passed':False,
        'failure_cause':source_truth,'complete_state_pairs':0,'state_denominator':16,
        'complete_scientific_executions':0,'execution_denominator':32,'attempted_states':1,
        'incomplete_states':1,'unattempted_states':15,'repetitions_executed':0,'replays_executed':0,
        'states':rows,'full_cohort_mean_return':None,'full_cohort_worst_return':None,
        'full_cohort_performance_unavailable':True,'completed_control_transitions':0,
        'actual_solver_calls':1,'actual_physics_steps':1,'raw_physics_states':2,
        'prefix_state_violations':0,'prefix_contacts':0,
        'prefix_minimum_lower_margins':low.min(0).tolist(),'prefix_minimum_upper_margins':high.min(0).tolist(),
        'prefix_minimum_raw_margins':np.minimum(low,high).min(0).tolist(),
        'prefix_minimum_normalized_margin':float(np.min(np.minimum(low,high)/(np.asarray(model['state_high'])-model['state_low']))),
        'prospective_first_control_interval_reserves':b['reserve'],
        'full_20ms_discrepancy_validation_available':False,'candidate_native_status':'Solve_Succeeded',
        'candidate_IPOPT_iterations':m['last_native_solver_stats']['iter_count'],
        'candidate_v2_KKT':kkt['metrics'],'native_independent_graph_error':reconstruction,
        'candidate_fields_complete':True,'partial_stream_matches_NPZ_bitwise':True,
        'latency_seconds':{k:{'observations':1,'median':v,'p95':v,'p99':v,'maximum':v,
                                  'over_20ms':int(v>.02)} for k,v in times.items()},
        'preserved_exclusive_timing_categories':status['timing_summary']['exclusive_categories'],
        'cold_start_included':True,'latency_population_inference_justified':False,
        'recorded_peak_worker_RSS_bytes':m['peak_rss_bytes'],
        'aggregate_parent_peak_RSS_not_retained':True,
        'conservative_charged_solves':300,'conservative_charged_physics_steps':6000,
        'failed_reservations_released':False,
        'remaining_original_solver_allowance':12970-300,'remaining_original_physics_allowance':948000-6000,
        'original_overall_qualification':'FAILED','original_0001_agreement':'FAILED',
        'reproducibility_of_new_trajectory':'not tested: stop prohibited repetition/replays',
        'preservation':provenance,'new_audit_solver_calls':0,'new_audit_physics_steps':0,
        'execution_sources_modified_after_stop':False,'thresholds_modified':False}
    return report,{'advance':advance,'preprocess':preprocess}
