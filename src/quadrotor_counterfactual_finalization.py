"""Independent saved-record Pilot B verification. No new solves or integration.

The original numerical audit is reused with explicit data/code roots and fresh
outputs. Historical evidence and ledgers are never rewritten.
"""
from pathlib import Path
from fractions import Fraction
import hashlib,json,subprocess
import numpy as np
from src import quadrotor_pilot as p
from src import quadrotor_counterfactual_execution_v2 as wiring
from src.quadrotor_counterfactual_context_v1 import require_archived_query
from src.quadrotor_kkt_integrity_v2 import nlp_readout,canonical_kkt,require_kkt,matching_program_values
from src.quadrotor_mmaq_v1 import safe_write
from src.quadrotor_mpc80_feasibility_v2 import zero_real_operations
BASE=Path('results/benchmark_selection')
CONTEXT={'CURRENT_STATUS.md','RESEARCH_CONTEXT.md','docs/RESEARCH_DECISION_LOG.md'}


def preservation(data,code):
    """Old scientific bytes stay fixed; prove historical context bytes using Git."""
    data,code=Path(data),Path(code);base=data/BASE
    def match(root,values):
        for name,h in values.items():assert p.sha(Path(root)/name)==h,name
        return len(values)
    accepted=base/'mpc80_feas_v2_study';seal=p.read(accepted/'read_only_audit/final_integrity.json')
    count=match(accepted,seal['artifact_sha256']);match(code,seal['additional_reporting_source_sha256'])
    snap=p.read(accepted/'preservation_start.json');changes=[]
    for group in ['tracked_sha256','all_prior_result_and_data_sha256']:
        for n,h in snap[group].items():
            if p.sha(data/n)==h:continue
            if n not in CONTEXT:raise AssertionError('Historical scientific bytes changed: '+n)
            original=subprocess.check_output(['git','show',snap['project_head']+':'+n],cwd=data)
            assert hashlib.sha256(original).hexdigest()==h,n;changes.append(n)
    technical=p.read(base/'technical_pilot/preservation_before.json');authority=match(data,technical['authority_files'])
    iql=match(data,p.read(base/'technical_pilot/iql_preservation.json')['sha256'])
    for folder in ['runtime_restoration_v1','counterfactual_pilot_B_v1']:
        ident=p.read(base/folder/'integrity.json');match(base/folder,ident['artifact_sha256'])
    stopped=p.read(base/'counterfactual_pilot_B_v1/integrity.json')
    assert p.sha(data/'src/quadrotor_counterfactual_execution_v1.py')==stopped['unexecuted_draft_source_sha256']
    assert p.sha(code/'docs/experiment_12_counterfactual_pilot_B_report.md')==stopped['additional_report_sha256']
    return {'passed':True,'accepted_MPC80_files':count,'prior_result_data_files':len(snap['all_prior_result_and_data_sha256']),'authority_files':authority,'unchanged_IQL_files':iql,'authorized_context_changes':sorted(set(changes))}


def verify(data,code,upstream,output,*,full_preservation=False):
    ROOT,code,upstream,output=map(lambda x:Path(x).resolve(),(data,code,upstream,output))
    OUT=ROOT/BASE/'counterfactual_pilot_B_v2'
    if output.exists():raise FileExistsError(output)
    accepted_base=(ROOT/BASE).resolve()
    if output.is_relative_to(accepted_base) and not output.is_relative_to((ROOT/BASE/'counterfactual_pilot_B_finalization').resolve()):
        raise ValueError('Do not write inside accepted evidence')
    integrity=p.read(OUT/'completion_integrity.json')
    for n,h in integrity['artifact_sha256'].items():assert p.sha(OUT/n)==h,n
    for n,h in integrity['authorization_artifact_sha256'].items():assert p.sha(ROOT/BASE/'counterfactual_pilot_B_v2_authorization'/n)==h,n
    assert p.sha(code/'docs/experiment_12_counterfactual_pilot_b_v2_report.md')==integrity['additional_report_sha256']
    review=ROOT/BASE/'counterfactual_wiring_review_v2';seal=p.read(review/'review_seal.json')
    assert p.sha(review/'review_seal.json')=='cfb6a56fd40158f26602d536bab80dd9636467f92b5912aed3721605874f8126'
    for n,h in seal['source_sha256'].items():assert p.sha(code/n)==h,n
    for n,h in seal['artifact_sha256'].items():assert p.sha(review/n)==h,n
    preservation_result=preservation(ROOT,code) if full_preservation else {'scope':'Pilot B/review seals; full historical preservation checked separately'}
    model=p.read(OUT/'branches/menu4/frozen.json');ref=np.load(OUT/'branches/menu4/reference.npy');archive=ROOT/wiring.ACCEPTED/'runs/0_nonlinear80_1000_original'
    low=np.asarray(model['state_low']);high=np.asarray(model['state_high']);alow=np.asarray(model['action_low']);ahigh=np.asarray(model['action_high']);Q=np.asarray(model['Q']);R=np.asarray(model['R']);hover=np.asarray(model['hover_input']);rows=[];checks=0
    ledger=p.read(OUT/'resource_ledger.json');result={'version':'Pilot_B_v2_independent_read_only_audit','campaign_complete':(OUT/'dispatch_complete.json').exists(),'campaign_halted':ledger['halted'],'records':[],'original_qualification':'FAILED','original_0001_model_agreement':'FAILED','new_solves':0,'new_physics_steps':0}
    def stats(values):
     a=np.asarray(values,dtype=float);b=a[np.isfinite(a)]
     return {'samples':len(a),'missing_or_nonfinite':len(a)-len(b),'median':float(np.median(b)) if len(b) else None,'p95':float(np.quantile(b,.95)) if len(b) else None,'p99':float(np.quantile(b,.99)) if len(b) else None,'max':float(np.max(b)) if len(b) else None,'over_20ms':int(np.sum(b>.02))}
    with zero_real_operations(upstream) as (blocked,engine):
     own=nlp_readout(model,80)
     for menu in [4,3]:
      for phase in ['original','repeat']:
       f=OUT/'branches'/f'menu{menu}'/'runs'/f'0_nonlinear80_1000_{phase}'
       if not (f/'completion.json').exists():
        result['records'].append({'menu_index':menu,'phase':phase,'status':'NOT_EXECUTED_OR_NO_COMPLETION_RECORD','progress':p.read(f/'progress.json') if (f/'progress.json').exists() else None});continue
       meta=p.read(f/'completion.json')
       if not meta['complete']:
        result['records'].append({'menu_index':menu,'phase':phase,'status':'FAILED_OR_INCOMPLETE','completion':meta});continue
       a=dict(np.load(f/'trajectory.npz'));assert p.sha(f/'trajectory.npz')==meta['trajectory_sha256']
       assert a['states'].shape==(301,6) and a['physics_states'].shape==(6001,6);assert a['physics_states'][::20].tobytes()==a['states'].tobytes()
       margin=np.minimum(a['physics_states']-low,high-a['physics_states']);assert np.isfinite(margin).all() and margin.min()>=0
       assert not np.any(a['physics_contacts']);assert np.array_equal(a['current_steps'],np.arange(300))
       submitted=np.fromfile(f/'submitted_scientific.f64',dtype='<f8').reshape(6000,56)
       assert np.fromfile(f/'submitted_scientific_completed.u1',dtype='u1').tobytes()==bytes([1])*6000
       assert np.isfinite(submitted).all();forces=submitted[:,17:29].reshape(6000,4,3);executed=np.c_[forces[:,0,2]+forces[:,3,2],forces[:,1,2]+forces[:,2,2]]
       assert np.min(executed-alow)>=0 and np.min(ahigh-executed)>=0
       assert submitted[:,3:5].tobytes()==np.repeat(a['actions'],20,axis=0).tobytes()
       assert submitted[:,13:17].tobytes()==np.repeat(a['rpms'],20,axis=0).tobytes()
       direct=executed[::20];derived=model['KF']*np.c_[a['rpms'][:,0]**2+a['rpms'][:,3]**2,a['rpms'][:,1]**2+a['rpms'][:,2]**2]
       assert np.max(np.abs(derived-direct))<=1e-12
       error=a['states'][1:]-ref[1:];du=a['clipped_actions']-hover
       costs=.5*np.einsum('ni,ij,nj->n',error,Q,error)+.5*np.einsum('ni,ij,nj->n',du,R,du)
       assert np.max(np.abs(costs+a['rewards']))<=1e-10
       if menu==4:
        wiring.compare_archived_nominal(f,archive)
        old=np.fromfile(archive/'submitted_scientific.f64',dtype='<f8').reshape(6000,56)
        assert submitted[:,:54].tobytes()==old[:,:54].tobytes()
       if phase=='repeat':
        original=OUT/'branches'/f'menu{menu}'/'runs/0_nonlinear80_1000_original';old=np.load(original/'trajectory.npz')
        for key in wiring.COMPARE_FIELDS:assert a[key].tobytes()==old[key].tobytes(),key
        prior=np.fromfile(original/'submitted_scientific.f64',dtype='<f8').reshape(6000,56);assert submitted[:,:54].tobytes()==prior[:,:54].tobytes()
       reserve=[];solver=[];native=[];supervised=[];kktmax={}
       causal=p.read(f/'causal_contract.json');assert len(causal)==300
       for k in range(300):
        folder=f/'attempts'/f'{k:04d}';z=dict(np.load(folder/'candidate.npz'));w=dict(np.load(folder/'warmstart.npz'));status=p.read(folder/'status.json')
        assert status['status']=='verified_action_not_yet_applied' and not status['missing_fields'] and not status['fallback_action_applied'];assert p.sha(folder/'candidate.npz')==status['candidate_sha256']
        native_values=own(z['primal'],z['parameters'],z['dual']);matching_program_values(native_values,[z[n] for n in ['objective','constraint_values','lower','upper','lagrangian_gradient']])
        d=canonical_kkt(native_values[1],native_values[2],native_values[3],z['dual'],native_values[4]);require_kkt(d,meta['solver_statuses'][k]);checks+=1
        for n,v in d['metrics'].items():kktmax[n]=max(kktmax.get(n,0.),v)
        params=z['parameters'].reshape(-1);preview=ref[np.minimum(k+np.arange(81),300)].T
        assert params[:6].tobytes()==a['states'][k].tobytes();assert params[6:].reshape(6,81,order='F').tobytes()==preview.tobytes()
        if menu==4 or k==0:require_archived_query(archive,k,a['states'][k],preview,z,w)
        if k:
         assert w['previous_x'].tobytes()==a['predicted_states'][k-1].tobytes();assert w['previous_u'].tobytes()==a['predicted_inputs'][k-1].tobytes()
         xx=w['previous_x'].copy();uu=w['previous_u'].copy();xx[:,:-1]=xx[:,1:];uu[:-1]=uu[1:]
         assert w['initial_guess_not_solver_candidate'].tobytes()==np.r_[xx.reshape(-1,order='F'),uu.reshape(-1,order='F')].tobytes()
        if phase=='repeat':
         priorz=np.load(OUT/'branches'/f'menu{menu}'/'runs/0_nonlinear80_1000_original/attempts'/f'{k:04d}'/'candidate.npz')
         for key in ['primal','parameters','dual']:assert z[key].tobytes()==priorz[key].tobytes()
        intervention=p.read(f/'intervention'/f'{k:04d}.json');assert intervention['post_intervention_history_unchanged'] and intervention['status']=='ready_for_native_step'
        cr=causal[k];assert np.asarray(cr['requested']).tobytes()==a['actions'][k].tobytes();rr=np.asarray(cr['reserve']);assert np.isfinite(rr).all() and np.min(rr)>0;reserve.append(rr)
        expected_reserve=np.minimum(np.asarray(cr['prediction'])-low,high-np.asarray(cr['prediction']))-cr['bound'];assert np.asarray(rr).tobytes()==expected_reserve.tobytes()
        spans=status['timings'];st=[r['wall_inclusive'] for r in spans if r['category']=='native_solver'];assert len(st)==1
        solver.append(st[0]);native.append(sum(r['wall_exclusive'] for r in spans if r['category'] in ['native_action_selection','native_solver']));supervised.append(a['solver_wall'][k])
       for name in ['replay_native.json','replay_instrumented.json']:
        rr=p.read(f/name);assert rr['steps']==300 and rr['control_states_bitwise'] and rr['rewards_bitwise']
        if name=='replay_instrumented.json':assert rr['physics_states_bitwise'] and rr['contacts_bitwise']
       motor=np.fromfile(f/'submitted_motor_replay.f64',dtype='<f8').reshape(6000,56);assert motor[:,:54].tobytes()==submitted[:,:54].tobytes()
       assert p.read(f/'submitted_motor_replay_counts.json')['complete_6000']
       normalized=margin/(high-low);reserve=np.asarray(reserve)
       record={'menu_index':menu,'phase':phase,'status':'COMPLETE_INDEPENDENTLY_VERIFIED','controls':300,'raw_states':6001,'checked_candidates':300,'return':-float(costs.sum()),'discounted_return':-float(np.dot(.99**np.arange(300),costs)),
        'position_tracking_RMSE':float(np.sqrt(np.mean(error[:,[0,2]]**2))),'terminal_error':error[-1].tolist(),'terminal_position_error_norm':float(np.linalg.norm(error[-1,[0,2]])),
        'absolute_squared_thrust_effort':float(np.sum(direct**2)/50),'hover_relative_squared_thrust_effort':float(np.sum((direct-hover)**2)/50),
        'minimum_raw_margins':margin.min(0).tolist(),'minimum_normalized_margins':normalized.min(0).tolist(),'minimum_raw_input_margins':np.minimum(executed-alow,ahigh-executed).min(0).tolist(),
        'minimum_reserves':reserve.min(0).tolist(),'state_violation_samples':int(np.any(margin<0,axis=1).sum()),'input_violation_physics_samples':int(np.any((executed<alow)|(executed>ahigh),axis=1).sum()),'contacts':int(np.count_nonzero(a['physics_contacts'])),
        'direct_vs_RPM_derived_thrust_max_abs':float(np.max(np.abs(derived-direct))),'maximum_solver_iterations':int(a['solver_iterations'].max()),'KKT_maxima':kktmax,
        'latency_seconds':{'solver_only':stats(solver),'native_compute':stats(native),'supervised':stats(supervised)},'fullrate_physics_timing_separately_identifiable':False}
       if menu==3:record['submitted_action_separation']=wiring.verify_first_submissions(submitted,model,menu,ROOT)
       result['records'].append(record)
     result['forbidden_solver_calls']=blocked;result['forbidden_engine_calls']=engine;assert not blocked and not engine
    result['independently_checked_candidates']=checks
    if result['campaign_complete']:
     assert checks==1200 and len(result['records'])==4
     bymenu={x['menu_index']:x for x in result['records'] if x['phase']=='original'}
     result['matched_alternative_minus_nominal']={n:(np.asarray(bymenu[3][n])-np.asarray(bymenu[4][n])).tolist() for n in ['return','discounted_return','position_tracking_RMSE','terminal_error','terminal_position_error_norm','absolute_squared_thrust_effort','hover_relative_squared_thrust_effort','minimum_raw_margins','minimum_normalized_margins','minimum_raw_input_margins','minimum_reserves']}
     result['passed']=True
    else:result['passed']=False
    original=p.read(OUT/'independent_scientific_audit.json')
    assert result==original,'Canonical independent scientific results differ'
    summary=p.read(OUT/'scientific_summary.json');assert summary['status']=='PILOT_B_PASS' and summary['verified_candidates']==1200
    labels=p.read(OUT/'all_36_label_accounting.json');assert len(labels)==36 and sum(not x['original_source_predicted_admissible'] for x in labels)==9
    assert sum(x['Pilot_B_status']=='empirically_safe_in_Pilot_B' for x in labels)==2
    encoded=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    final={'passed':True,'scientific_results':result,'canonical_scientific_sha256':hashlib.sha256(encoded).hexdigest(),'sealed_Pilot_B_files':len(integrity['artifact_sha256']),
        'preservation':preservation_result,'new_solves':0,'new_physics_steps':0}
    safe_write(output,final)
    return final
