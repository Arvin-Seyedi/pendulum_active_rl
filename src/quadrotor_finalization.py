"""Read-only E12 milestone verification, including clean source exports.

No controller/environment is instantiated. Numerical replay means checking
retained physics transitions and accepted action-replay records, never executing
additional simulator episodes. Reviewed historical sources/results are immutable.
"""
from pathlib import Path
from fractions import Fraction
import hashlib
import json
import subprocess
import time
import numpy as np
from src.quadrotor_verification import audit_record
from src.quadrotor_kkt_integrity_v2 import canonical_kkt,require_kkt,nlp_readout,matching_program_values
from src.quadrotor_linear_verification import independent_program,parameter_terms,linear_transition
from src.quadrotor_mmaq_verification_v1 import nlp_metrics
from src.quadrotor_mismatch_contract_v1 import bound_components

BASE=Path('results/benchmark_selection')
STAGES={'technical_pilot':'review_integrity.json','compatibility_diagnostic':'completion.json',
        'linear_qrqp_revalidation':'completion.json','mmaq_v1':'completion.json',
        'qualification_integrity_v2':'completion.json','H2_MPC20_requalification_v2':'completion.json'}
CONTEXT={'CURRENT_STATUS.md','RESEARCH_CONTEXT.md','docs/RESEARCH_DECISION_LOG.md'}
IQL={'configs/experiment_06.json','docs/experiment_06_pilot_report.md','experiments/06_offline_rl.py',
     'requirements_experiment_06.txt','src/iql.py','src/offline_data.py','src/offline_evaluation.py',
     'src/offline_experiment.py','tests/test_offline_rl.py'}


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def destination(output,data_root):
    output,data_root=Path(output).resolve(),Path(data_root).resolve()
    if output.exists():raise FileExistsError('Verification output already exists: '+str(output))
    for sub in STAGES:
        if output.is_relative_to((data_root/BASE/sub).resolve()):
            raise ValueError('Output may not be inside accepted evidence')
    return output


def historical_context(root,code,name,expected,commit):
    """Allow only the three authorized context documents, with proven old bytes."""
    path=Path(code)/name
    if sha(path)==expected:return False
    if name not in CONTEXT:raise AssertionError('Historical source changed: '+name)
    previous=subprocess.check_output(['git','show',commit+':'+name],cwd=root)
    if hashlib.sha256(previous).hexdigest()!=expected:raise AssertionError('Unproven context exception: '+name)
    return True


def preserved_path(data,code,name):
    if name in IQL or name.startswith(('results/','data/')):return Path(data)/name
    return Path(code)/name


def preservation(data,code,upstream):
    data,code,upstream=map(Path,(data,code,upstream));seals={};sources={};artifacts=0
    for sub,marker in STAGES.items():
        folder=data/BASE/sub;seal=read(folder/marker);seals[sub]=sha(folder/marker)
        for name,h in seal['artifact_sha256'].items():
            path=(folder/name).resolve()
            if not path.is_relative_to(folder.resolve()):raise ValueError('Artifact escapes archive')
            assert sha(path)==h,(sub,name);artifacts+=1
        for key in ['source_sha256','execution_source_sha256','postrun_source_sha256']:
            for name,h in seal.get(key,{}).items():assert sha(code/name)==h,name;sources[name]=h
    old=read(data/BASE/'technical_pilot/preservation_before.json');changed=[]
    for name,h in old['files'].items():
        if name in CONTEXT:
            if historical_context(data,code,name,h,old['base_commit']):changed.append(name)
        else:assert sha(preserved_path(data,code,name))==h,name
    for name,h in old['authority_files'].items():assert sha(name)==h,name
    f=read(data/BASE/'technical_pilot/frozen.json')
    for name,h in f['upstream_files'].items():assert sha(upstream/name)==h,name
    iq=read(data/BASE/'technical_pilot/iql_preservation.json')['sha256'];assert set(iq)==IQL
    for name,h in iq.items():assert sha(data/name)==h,name
    return {'accepted_seals':seals,'accepted_artifacts_checked':artifacts,'accepted_source_files_checked':len(sources),
        'historical_files_checked':len(old['files']),'authority_files_checked':len(old['authority_files']),
        'upstream_files_checked':len(f['upstream_files']),'all_nine_IQL_sha256':iq,
        'authorized_context_updates':sorted(changed),'historical_context_base_commit':old['base_commit']}


def physics_reconstruction(a,f,frequency):
    """Independent dense matched-state replay of the declared source equations."""
    states=a['physics_states'];n=len(a['rewards']);stride=frequency//50
    assert len(states)==stride*n+1
    if not n:return {'physics_transitions':0,'maximum_equation_error':0.}
    rpm=a['rpms'];u=f['KF']*np.column_stack([rpm[:,0]**2+rpm[:,3]**2,rpm[:,1]**2+rpm[:,2]**2])
    total=np.repeat(u.sum(1)/f['mass'],stride);alpha=np.repeat(.028/f['Iyy']*(u[:,1]-u[:,0]),stride)
    before,after=states[:-1],states[1:];cache=np.r_[states[0,4],states[:-2,4]];h=1/frequency
    velocity=before[:,[1,3,5]]+h*np.column_stack([total*np.sin(cache),total*np.cos(cache)-f['gravity'],
                alpha*np.cos(before[:,4]-cache)])
    pos=before[:,[0,2,4]]+h*after[:,[1,3,5]]
    err=max(float(np.max(np.abs(velocity-after[:,[1,3,5]]))),float(np.max(np.abs(pos-after[:,[0,2,4]]))))
    assert err<1e-10,err
    return {'physics_transitions':len(before),'maximum_equation_error':err}


def same_fields(left,right):
    fields=['states','physics_states','physics_contacts','actions','clipped_actions','rpms','rewards',
            'dones','predicted_states','predicted_inputs','native_model_predictions','current_steps','solver_iterations','solver_success']
    return {name:left[name].tobytes()==right[name].tobytes() for name in fields}


def audit_trajectories(data):
    stages={};count=transitions=replays=0;worst=0.
    for sub in ['technical_pilot','linear_qrqp_revalidation','mmaq_v1','H2_MPC20_requalification_v2']:
        base=Path(data)/BASE/sub;f=read(base/'frozen.json');ref=np.load(base/'reference.npy');rows=[];arrays={};metadata={}
        assert sha(base/'reference.npy')==f['reference_sha256']
        t=np.arange(301)/50;w=2*np.pi/6
        reference=np.column_stack([.75*np.sin(w*t),.75*w*np.cos(w*t),1+.75*np.sin(w*t)*np.cos(w*t),
                                   .75*w*np.cos(2*w*t),np.zeros(301),np.zeros(301)])
        assert np.max(np.abs(ref-reference))<1e-14
        for folder in sorted((base/'runs').iterdir()):
            m=read(folder/'completion.json');result,a=audit_record(folder,f,ref)
            replay=physics_reconstruction(a,f,m['frequency']);transitions+=replay['physics_transitions'];worst=max(worst,replay['maximum_equation_error'])
            for name in ['replay_native.json','replay_instrumented.json']:
                evidence=read(folder/name);assert evidence['steps']==m['transitions']
                for key,value in evidence.items():
                    if key.endswith('bitwise'):assert value,(folder,key)
                if m['complete']:replays+=1
            rows.append({'record':folder.name,'case':m['case'],'controller':m['controller'],'phase':m['phase'],
                'frequency':m['frequency'],'complete':m['complete'],'controls':m['transitions'],
                'safe_complete':result['safe_complete_no_contact'],'return':result['return'],
                'violating_physics_samples':result['violating_physics_samples'],
                'minimum_normalized_margin':result['minimum_normalized_margin'],
                'original_0001_passed':result['one_step_tolerance_passed'],
                'original_error_maximum':result['symbolic_pybullet_one_step_max_scaled'],
                'physics_reconstruction':replay})
            arrays[folder.name]=a;metadata[folder.name]=m;count+=1
        repeat_count=0
        for name,m in metadata.items():
            if m['phase'] not in ['original','refine']:continue
            other=name.rsplit('_',1)[0]+'_repeat'
            assert all(same_fields(arrays[name],arrays[other]).values()),name;repeat_count+=1
        if sub=='technical_pilot':
            for case in f['cases']:
                for controller in ['nonlinear20','nonlinear40','nonlinear80']:
                    r1=next(r for r in rows if r['record']==f'{case}_{controller}_1000_original')
                    r2=next(r for r in rows if r['record']==f'{case}_{controller}_2000_refine')
                    assert r1['safe_complete']==r2['safe_complete']
                    assert abs(r1['return']-r2['return'])/abs(r1['return'])<=.01
        stages[sub]={'records':rows,'attempts':len(rows),'complete':sum(r['complete'] for r in rows),'repeat_pairs':repeat_count}
    return {'stages':stages,'attempts':count,'physics_transitions_reconstructed':transitions,
            'maximum_physics_equation_error':worst,'completed_episode_replay_records_checked':replays}


def audit_candidates(data):
    data=Path(data);f=read(data/BASE/'mmaq_v1/frozen.json');fn={n:nlp_readout(f,n) for n in [20,40,80]};qp=independent_program(f)
    counts={};maxima={};critical=[];prefix=[]
    for sub in ['linear_qrqp_revalidation','mmaq_v1','H2_MPC20_requalification_v2']:
        count=0
        for folder in sorted((data/BASE/sub/'runs').iterdir()):
            m=read(folder/'completion.json');linear=m['controller']=='linear40';points=[]
            trajectory=np.load(folder/'trajectory.npz');reference=np.load(folder.parent.parent/'reference.npy')
            if sub=='linear_qrqp_revalidation':
                graph=folder.parent.parent/'graphs'/f"{m['case']}_{m['phase']}"
                marker=read(graph/'graph_audit.json')
                assert marker['graph_sha256_before']==marker['graph_sha256_after']==sha(graph/'native_program.casadi')
                coefficients=np.load(graph/'native_graph_coefficients.npz')
                assert np.max(np.abs(coefficients['H']-np.diag(qp['Hdiag'])))<1e-10
                assert np.max(np.abs(coefficients['A']-qp['A'].toarray()))<1e-10
            if sub=='H2_MPC20_requalification_v2':
                for k in range(300):
                    attempt=folder/'attempts'/f'{k:04d}';a=np.load(attempt/'candidate.npz');status=read(attempt/'status.json')
                    assert status['status']=='verified_action_not_yet_applied' and not status['missing_fields']
                    assert not status['fallback_action_applied'] and sha(attempt/'candidate.npz')==status['candidate_sha256']
                    points.append(tuple(a[key].reshape(-1) for key in ['primal','parameters','dual']))
            else:
                saved=np.load(folder/('kkt_solutions.npz' if linear else 'nlp_solutions.npz'))
                points=list(zip(saved['primal'],saved['parameters'],saved['dual']))
            assert len(points)==m['transitions']
            for k,(z,params,lam) in enumerate(points):
                assert np.isfinite(np.r_[z,params,lam]).all()
                assert np.max(np.abs(params[:6]-trajectory['states'][k]))<1e-14
                horizon=40 if linear else int(m['controller'].replace('nonlinear',''))
                expected=reference[np.minimum(k+np.arange(horizon+1),300)].T.reshape(-1,order='F')
                assert np.array_equal(params[6:],expected)
                if linear:
                    q,lo,hi,_=parameter_terms(qp,f,params);g=qp['A'].dot(z);grad=qp['Hdiag']*z+q+qp['A'].T.dot(lam)
                else:
                    N=int(m['controller'].replace('nonlinear',''));own=fn[N](z,params,lam);obj,g,lo,hi,grad=own
                result=canonical_kkt(g,lo,hi,lam,grad)
                require_kkt(result,m['solver_statuses'][k],primal_tolerance=1.01e-8 if linear else 1e-4,
                            required_status='success' if linear else 'Solve_Succeeded')
                for key,value in result['metrics'].items():maxima[key]=max(maxima.get(key,0),value)
                count+=1
                if sub=='H2_MPC20_requalification_v2':
                    attempt=folder/'attempts'/f'{k:04d}';a=np.load(attempt/'candidate.npz')
                    matching_program_values(own,[a[key] for key in ['objective','constraint_values','lower','upper','lagrangian_gradient']])
                    if k==54:
                        native=lambda z_,p_:[a[key] for key in ['objective','constraint_values','lower','upper']]
                        old=nlp_metrics(fn[20],native,z,params,lam)
                        assert np.isinf(old['normalized_complementarity_inf'])
                        critical.append({'phase':m['phase'],'step':54,'row':18,'multiplier':float(lam[18]),
                            'v2_dual_sign_inf':result['metrics']['dual_sign_inf'],'legacy_nonfinite_reproduced':True,
                            'evidence':'newly retained v2 point, not recovered historical vector'})
            if sub=='H2_MPC20_requalification_v2':
                previous=data/BASE/'mmaq_v1/runs'/folder.name;oldnp=np.load(previous/'nlp_solutions.npz');newa=np.load(folder/'trajectory.npz');olda=np.load(previous/'trajectory.npz')
                for i,key in enumerate(['primal','parameters','dual']):assert np.asarray([r[i] for r in points[:54]]).tobytes()==oldnp[key].tobytes()
                for key in ['states','physics_states','physics_contacts','actions','clipped_actions','rpms','rewards','predicted_states','predicted_inputs']:
                    assert newa[key][:len(olda[key])].tobytes()==olda[key].tobytes(),key
                prefix.append({'phase':m['phase'],'retained_controls':54,'bitwise':True,'historical_rejected_candidate_still_absent':len(oldnp['primal'])==54})
        counts[sub]=count
    assert counts=={'linear_qrqp_revalidation':2400,'mmaq_v1':7932,'H2_MPC20_requalification_v2':600}
    return {'counts':counts,'total':sum(counts.values()),'corrected_KKT_maxima':maxima,'critical_points':critical,'prefixes':prefix}


def finalize(data,code,upstream,output):
    started=time.process_time();data,code=Path(data),Path(code);output=destination(output,data)
    preserved=preservation(data,code,upstream)
    trajectories=audit_trajectories(data);candidates=audit_candidates(data)
    source=data/BASE/'mmaq_v1';f=read(source/'frozen.json');manifest=read(source/'screening_manifest.json')
    assert len(manifest['states'])==16 and not manifest['evaluated']
    # Recompute the approved discrepancy majorants using archived operands only.
    bounds_checked=0
    discrepancy_maximum=0.
    from src.quadrotor_mmaq_verification_v1 import exact_majorant as rational
    for sub in ['mmaq_v1','H2_MPC20_requalification_v2']:
        for folder in sorted((data/BASE/sub/'runs').iterdir()):
            m=read(folder/'completion.json');a=np.load(folder/'trajectory.npz')
            for k in range(m['transitions']):
                cache=a['physics_states'][max(0,k*20-1),4];b=bound_components(a['states'][k],a['actions'][k],cache,f,m['controller']=='linear40')
                exact=rational(a['states'][k],b['model_thrust'],b['executed_thrust'],cache,f,m['controller']=='linear40')
                assert all(Fraction(float(x))>=y for x,y in zip(b['bound'],exact));bounds_checked+=6
                if m['controller']=='linear40':
                    Ad,Bd=linear_transition(f)
                    prediction=Ad@a['states'][k]+Bd@(a['actions'][k]-f['hover_input'])
                else:prediction=a['native_model_predictions'][k]
                discrepancy=np.abs(prediction-a['states'][k+1])
                assert np.all(discrepancy<=b['bound']+1e-10)
                discrepancy_maximum=max(discrepancy_maximum,float(discrepancy.max()))
    usage=read(data/BASE/'H2_MPC20_requalification_v2/resource_accounting.json')
    remaining=usage['remaining_conservative_original_budget'];assert remaining['solver_calls']==12970 and remaining['physics_steps']==948000
    science={'trajectories':trajectories,'candidates':candidates,'exact_bound_component_checks':bounds_checked,
             'maximum_qualification_prediction_discrepancy':discrepancy_maximum,
             'remaining_solver_calls':12970,'remaining_physics_steps':948000,
             'original_model_agreement_gate':'FAILED at unchanged 0.001','overall_qualification':'FAILED',
             'targeted_H2_v2_qualification':'PASSED','sixteen_state_screening_executed':False,
             'action_probes_executed':False,'new_solver_calls':0,'new_physics_steps':0}
    result={'scientific_results':science,'preservation':preserved,'code_root':str(code.resolve()),
            'data_root':str(data.resolve()),'cpu_seconds':time.process_time()-started,
            'IQL_imports_required':False,'verification_mode':'saved-record numerical/replay-evidence audit; no simulator or controller execution'}
    output.parent.mkdir(exist_ok=True,parents=True);output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result
