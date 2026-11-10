"""Immutable B0 checkpoint/record audit, independent of scientific fitting.

No fitter, MPC controller, environment, optimizer or simulator is executed.
"""
from pathlib import Path
import hashlib,json,subprocess
import numpy as np
import torch
from src.quadrotor_pilot import read,sha
from src.quadrotor_mmaq_v1 import safe_write
from src.quadrotor_mpc80_feasibility_v2 import zero_real_operations
from src.quadrotor_archive_experiment import check_seal
BASE=Path('results/benchmark_selection')
CONTEXT={'CURRENT_STATUS.md','RESEARCH_CONTEXT.md','docs/RESEARCH_DECISION_LOG.md'}


def historical_preservation(data,code,original):
    from src.quadrotor_counterfactual_finalization import preservation
    for name,h in original['tracked_sha256'].items():
        path=code/name
        if sha(path)==h:continue
        if name not in CONTEXT:raise AssertionError('Historical source changed: '+name)
        before=subprocess.check_output(['git','show',original['HEAD']+':'+name],cwd=data)
        assert hashlib.sha256(before).hexdigest()==h,name
    for name,h in original['IQL_sha256'].items():assert sha(data/name)==h,name
    result=preservation(data,code)
    pilot=data/BASE/'counterfactual_pilot_B_v2';seal=read(pilot/'completion_integrity.json')
    for name,h in seal['artifact_sha256'].items():assert sha(pilot/name)==h,name
    for name,h in seal['authorization_artifact_sha256'].items():assert sha(data/BASE/'counterfactual_pilot_B_v2_authorization'/name)==h,name
    return {**result,'B0_previous_tracked_files':len(original['tracked_sha256']),'Pilot_B_files':len(seal['artifact_sha256'])}


def verify(data,code,upstream,output,*,full_preservation=False):
    r,code,upstream,output=map(lambda p:Path(p).resolve(),(data,code,upstream,output));o=r/BASE/'b0_archive_v1'
    if output.exists():raise FileExistsError(output)
    if output.is_relative_to((r/BASE).resolve()) and not output.is_relative_to((r/BASE/'b0_finalization').resolve()):raise ValueError('Output inside accepted evidence')
    seal=read(o/'completion_integrity.json')
    for name,h in seal['artifact_sha256'].items():assert sha(o/name)==h,name
    for name,h in seal['source_sha256'].items():assert sha(code/name)==h,name
    assert sha(code/'docs/experiment_12_b0_report.md')==seal['additional_report_sha256']
    assert sha(o/'protocol_seal.json')=='0ffc97e5731bfde8c22f1f7fac435ebfd4a551421de0d7cf76aad4478a62927e'
    from src.quadrotor_archive_data import runtime_identity
    runtime_identity(r,upstream)
    assert sha(Path(torch.__file__).parent/'lib/libtorch_cpu.dylib')==read(o/'runtime_identity.json')['torch_cpu_library_sha256']
    preservation=historical_preservation(r,code,read(o/'preservation_start.json')) if full_preservation else {'scope':'B0 seal/records; full history checked separately in same finalization'}
    c=check_seal(code,o);a=read(o/'assessment.json');p=np.load(o/'processed.npz');m=read(o/'model_constants.json');lo=np.asarray(m['state_low']);hi=np.asarray(m['state_high']);al=np.asarray(m['action_low']);ah=np.asarray(m['action_high']);w=ah-al
    Y=(p['targets']-al)/w;X=p['features'];frozen=read(o/'data_provenance.json');assert hashlib.sha256(X.tobytes()).hexdigest()==frozen['feature_sha256']
    # Independently assemble all features from the native record and reference.
    for i in range(16):
     folder=r/'results/benchmark_selection/mpc80_feas_v2_study/runs'/f'{i}_nonlinear80_1000_original';raw=np.load(folder/'trajectory.npz');rows=np.fromfile(folder/'submitted_scientific.f64',dtype='<f8').reshape(6000,56);force=rows[::20,17:29].reshape(300,4,3);actual=np.c_[force[:,0,2]+force[:,3,2],force[:,1,2]+force[:,2,2]]
     previous=np.concatenate([np.asarray(m['hover_input'])[None,:],actual[:-1]])
     own=[]
     for k in range(300):
      state=2*(raw['states'][k]-(lo+hi)/2)/(hi-lo);ref=p['reference'][np.minimum(k+np.arange(81),300)].T;preview=2*(ref-(lo+hi)[:,None]/2)/(hi-lo)[:,None]
      prev=2*(previous[k]-(al+ah)/2)/(ah-al);own.append(np.r_[state,preview.reshape(-1),k/300,prev,float(k==0)])
     assert np.asarray(own,dtype=np.float32).tobytes()==X[i].tobytes();assert raw['actions'].tobytes()==p['targets'][i].tobytes()
    checks=0;cross=[];matrices={};training_mae={};val_mae={}
    with zero_real_operations(upstream) as (blocked,engine):
     for n in c['partitions']['subset_sizes']:
      matrix=[];train=[];val=[]
      for seed in c['seeds']:
       f=o/'fits'/f'n{n}_seed{seed}';s=torch.load(f/'final.pt',weights_only=True,map_location='cpu')['model'];pred=np.load(o/f'predictions_n{n}_seed{seed}.npz')['normalized_predictions'];xx=X.reshape(-1,496).astype(np.float64)
       # Independent binary64 affine/tanh/sigmoid readout; no Torch model.forward.
       z=np.tanh(np.einsum('ni,ji->nj',xx,s['network.0.weight'].numpy().astype(float),optimize=False)+s['network.0.bias'].numpy())
       z=np.tanh(np.einsum('ni,ji->nj',z,s['network.2.weight'].numpy().astype(float),optimize=False)+s['network.2.bias'].numpy())
       z=np.einsum('ni,ji->nj',z,s['network.4.weight'].numpy().astype(float),optimize=False)+s['network.4.bias'].numpy();own=(1/(1+np.exp(-z))).reshape(16,300,2)
       delta=float(np.abs(own-pred).max());assert np.isfinite(own).all() and delta<1e-5;cross.append(delta)
       error=np.abs(pred-Y);v=error.mean((1,2));matrix.append(v[c['partitions']['assessment']]);train.append(v[c['partitions']['training_order'][:n]].mean());val.append(v[c['partitions']['validation']].mean())
       saved=next(z for z in a['per_seed_trajectory_metrics'] if z['size']==n and z['seed']==seed)
       for i in range(16):
        rec=saved['trajectories'][str(i)];assert abs(v[i]-rec['normalized_MAE'])<1e-15
        assert abs(float(np.quantile(error[i],.95))-rec['normalized_p95'])<1e-15
        assert abs(float((error[i]*w).mean())-rec['physical_MAE_N'])<1e-15
        for name,(low,high) in c['phases'].items():assert abs(float(error[i,low:high].mean())-rec['phases'][name]['normalized_MAE'])<1e-15
       completion=read(f/'completion.json');assert completion['updates']==5000 and completion['labels']==300*n
       assert np.isfinite(error).all();checks+=1
      matrices[n]=np.array(matrix);curve=a['curves'][str(n)];assert np.asarray(matrix).tobytes()==np.asarray(curve['assessment_seed_trajectory_MAE']).tobytes()
      assert abs(np.mean(train)-curve['training_macro_MAE'])<1e-15 and abs(np.mean(val)-curve['validation_macro_MAE'])<1e-15
     # Independent crossed paired bootstrap, sharing the fixed seed/trajectory indices.
     rng=np.random.Generator(np.random.PCG64(122999));si=rng.integers(5,size=(20000,5));ti=rng.integers(4,size=(20000,4));D=matrices[4]-matrices[10];distribution=D[si[:,:,None],ti[:,None,:]].mean((1,2));ci=np.quantile(distribution,[.025,.975])
     assert np.array_equal(ci,np.asarray(a['decision']['paired_base_minus_large']['95_CI']))
     assert abs(float(D.mean())-a['decision']['paired_base_minus_large']['mean'])<1e-15
     assert not blocked and not engine
    for n,matrix in matrices.items():
        own_ci=np.quantile(matrix[si[:,:,None],ti[:,None,:]].mean((1,2)),[.025,.975])
        assert np.array_equal(own_ci,np.asarray(a['curves'][str(n)]['assessment_summary']['95_CI']))
    # Baseline verification evaluates the saved fixed coefficients, never refits.
    ridge_checks=0;worst={}
    for n in c['partitions']['subset_sizes']:
        ids=c['partitions']['training_order'][:n];x=X[ids].reshape(-1,496).astype(float);y=Y[ids].reshape(-1,2);design=np.c_[np.ones(len(x)),x]
        A=np.einsum('ni,nj->ij',design,design,optimize=False)/len(design);A[1:,1:]+=.001*np.eye(496);b=np.einsum('ni,nj->ij',design,y,optimize=False)/len(design)
        beta=np.load(o/f'ridge_{n}.npz')['coefficients'];residual=np.linalg.norm(np.einsum('ij,jk->ik',A,beta,optimize=False)-b)/(np.linalg.norm(A)*np.linalg.norm(beta)+np.linalg.norm(b));assert residual<=1e-12
        raw=np.einsum('ni,ij->nj',np.c_[np.ones(4800),X.reshape(-1,496).astype(float)],beta,optimize=False).reshape(16,300,2);pred=np.clip(raw,0,1)
        for i in range(16):assert abs(np.abs(pred[i]-Y[i]).mean()-a['ridge'][str(n)]['trajectories'][str(i)]['normalized_MAE'])<1e-15
        maxerr=-1.
        for seed in c['seeds']:
            pred=np.load(o/f'predictions_n{n}_seed{seed}.npz')['normalized_predictions'];err=np.abs(pred[c['partitions']['assessment']]-Y[c['partitions']['assessment']]);maxerr=max(maxerr,float(err.max()))
        worst[n]=maxerr;ridge_checks+=1
    recorded_worst=read(o/'worst_prediction_cases.json')
    for z in recorded_worst:assert worst[z['size']]==z['normalized_absolute_error']
    # Five seeds / four subset sizes, with the same initial weights and uniforms.
    for seed in c['seeds']:
        rows=[read(o/'fits'/f'n{n}_seed{seed}'/'completion.json') for n in c['partitions']['subset_sizes']]
        assert len({v['initial_parameters_sha256'] for v in rows})==1 and len({v['uniform_schedule_sha256'] for v in rows})==1
        uniforms=np.random.Generator(np.random.PCG64(seed+1000000)).random((5000,256));assert hashlib.sha256(uniforms.tobytes()).hexdigest()==rows[0]['uniform_schedule_sha256']
        for n,z in zip(c['partitions']['subset_sizes'],rows):assert hashlib.sha256(np.floor(uniforms*(300*n)).astype(np.int64).tobytes()).hexdigest()==z['index_schedule_sha256']
    first=torch.load(o/'fits/n4_seed122000/final.pt',weights_only=True,map_location='cpu');repeat=torch.load(o/'reproducibility_repeat/final.pt',weights_only=True,map_location='cpu')
    for k in first['model']:assert torch.equal(first['model'][k],repeat['model'][k])
    for i,fields in first['optimizer']['state'].items():
        for k,v in fields.items():assert torch.equal(v,repeat['optimizer']['state'][i][k])
    assert not a['decision']['GO_for_designing_future_acquisition']
    encoded=json.dumps(a,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    result={'passed':True,'scientific_results':a,'canonical_scientific_sha256':hashlib.sha256(encoded).hexdigest(),'preservation':preservation,'sealed_B0_artifacts':len(seal['artifact_sha256']),
        'models_checked':checks,'ridge_checks':ridge_checks,'feature_rows':4800,'paired_CI_reconstructed':True,'deterministic_fit_record_bitwise':True,
        'max_FP64_readout_difference':max(cross),'new_training_fits':0,'new_MPC_solves':0,'new_physics_steps':0}
    safe_write(output,result);return result
