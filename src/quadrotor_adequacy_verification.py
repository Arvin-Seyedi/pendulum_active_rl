"""Read-only independent B1 checkpoint/statistic reconstruction; never refits."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
from src.quadrotor_pilot import read,sha
from src.quadrotor_adequacy_data import BASE,compact
from src.quadrotor_mpc80_feasibility_v2 import zero_real_operations

def verify(data_root,code_root,upstream,study,output,full_preservation=False):
    root,code,upstream,out,output=map(Path,(data_root,code_root,upstream,study,output))
    if output.exists():raise FileExistsError(output)
    seal=read(out/'protocol_seal.json')
    for name,h in seal['sources'].items():assert sha(code/name)==h,name
    for name,h in seal['artifacts'].items():assert sha(out/name)==h,name
    assert sha(code/'docs/experiment_12_b1_adequacy_design.md')==seal['approved_design_sha256']
    snap=read(out/'preservation_start.json')
    for name,h in snap['tracked_sha256'].items():assert sha(code/name)==h,name
    for name,h in snap['untracked_sha256'].items():assert sha(root/name)==h,name
    # Preserve all original B0 scientific artifacts, including the NO-GO report.
    old=root/BASE/'b0_archive_v1';oldseal=read(old/'completion_integrity.json')
    for name,h in oldseal['artifact_sha256'].items():assert sha(old/name)==h,name
    for name,h in oldseal['source_sha256'].items():assert sha(code/name)==h,name
    assert sha(code/'docs/experiment_12_b0_report.md')==oldseal['additional_report_sha256']
    for name,h in read(out/'external_provenance.json')['external_source_sha256'].items():assert sha(root/name)==h,name
    from src.quadrotor_archive_data import runtime_identity
    runtime_identity(root,upstream)
    c=read(out/'protocol.json');d=np.load(out/'processed.npz');a=read(out/'assessment.json');m=read(out/'model_constants.json');width=np.array(m['action_high'])-m['action_low']
    Y=(d['targets']-m['action_low'])/width;H=(d['external_targets']-m['action_low'])/width
    assert compact(d['full']).tobytes()==d['compact'].tobytes()
    assert compact(d['external_full']).tobytes()==d['external_compact'].tobytes()
    assert read(out/'fits_complete.json')['scientific_MLP_fits']==20 and read(out/'fits_complete.json')['total_updates']==420000
    matrices={};max_delta=0.;count=0;maxridge=0.
    def network(x,state):
        x=x.astype(np.float64)
        for layer in [0,2]:
            x=np.tanh(np.einsum('ni,ji->nj',x,state[f'network.{layer}.weight'].numpy().astype(float),optimize=False)+state[f'network.{layer}.bias'].numpy())
        z=np.einsum('ni,ji->nj',x,state['network.4.weight'].numpy().astype(float),optimize=False)+state['network.4.bias'].numpy()
        return 1/(1+np.exp(-z))
    with zero_real_operations(upstream) as (blocked,engine):
        for rep in ['full','compact']:
            X=d[rep];HX=d['external_'+rep];dim=X.shape[-1]
            for n in [4,10]:
                ids=c['partitions']['training_order'][:n]
                for u in [5000,20000]:
                    key=f'{rep}_n{n}_u{u}';matrix=[]
                    for seed,rec in zip(c['seeds'],a['MLP'][key]['per_seed']):
                        f=out/'fits'/f'{rep}_n{n}_seed{seed}';s=torch.load(f/f'update_{u}.pt',weights_only=True)['model'];p=np.load(out/f'predictions_{key}_seed{seed}.npz');hp=p['external'];pred=p['development']
                        for x,record in [(X,pred),(HX,hp)]:
                            own=network(x.reshape(-1,dim),s).reshape(record.shape);delta=float(abs(own-record).max());max_delta=max(max_delta,delta);assert delta<1e-5
                        for part,ix,target,prediction in [('training',ids,Y[ids],pred[ids]),('validation',c['partitions']['validation'],Y[c['partitions']['validation']],pred[c['partitions']['validation']]),
                                ('reused_development',c['partitions']['assessment'],Y[c['partitions']['assessment']],pred[c['partitions']['assessment']]),('external',list(range(4)),H,hp)]:
                            e=abs(prediction-target);own=float(e.mean());record=rec[part]
                            assert abs(own-record['normalized_MAE'])<1e-15
                            assert abs(float(np.mean(e**2))-record['normalized_MSE'])<1e-15
                            assert abs(float(np.mean(e*width))-record['physical_MAE_N'])<1e-15
                            assert abs(float(np.max(e*width))-record['worst_action_error_N'])<1e-15
                            assert abs(float(np.quantile(e,.95))-record['normalized_p95'])<1e-15
                            for name,(lo,hi) in c['phases'].items():assert abs(float(e[:,lo:hi].mean())-record['phases'][name]['normalized_MAE'])<1e-15
                        matrix.append(abs(hp-H).mean((1,2)));count+=1
                    matrices[key]=np.array(matrix)
                    assert matrices[key].tobytes()==np.array(a['MLP'][key]['external_summary']['seed_trajectory_matrix']).tobytes()
                # Independently reconstruct the regularized normal equations.
                x=X[ids].reshape(-1,dim).astype(float);design=np.c_[np.ones(len(x)),x];y=Y[ids].reshape(-1,2)
                A=np.einsum('ni,nj->ij',design,design,optimize=False)/len(x);A[1:,1:]+=.001*np.eye(dim);b=np.einsum('ni,nj->ij',design,y,optimize=False)/len(x)
                beta=np.load(out/f'ridge_{rep}_n{n}.npz')['coefficients'];res=np.linalg.norm(np.einsum('ij,jk->ik',A,beta,optimize=False)-b)/(np.linalg.norm(A)*np.linalg.norm(beta)+np.linalg.norm(b));assert res<1e-12;maxridge=max(maxridge,float(res))
                raw=np.einsum('ni,ij->nj',np.c_[np.ones(1200),HX.reshape(-1,dim).astype(float)],beta,optimize=False).reshape(4,300,2);p=np.load(out/f'predictions_ridge_{rep}_n{n}.npz');assert raw.tobytes()==p['raw_external'].tobytes();assert np.clip(raw,0,1).tobytes()==p['external'].tobytes()
        # Independent crossed seed/trajectory bootstrap; never transition resampling.
        rng=np.random.Generator(np.random.PCG64(123999));si=rng.integers(5,size=(20000,5));ti=rng.integers(4,size=(20000,4))
        pairs={'optimization':('full_n10_u5000','full_n10_u20000'),'parameterization':('full_n10_u20000','compact_n10_u20000'),'data':('compact_n4_u20000','compact_n10_u20000')}
        for name,(left,right) in pairs.items():
            D=matrices[left]-matrices[right];v=D[si[:,:,None],ti[:,None,:]].mean((1,2));rec=a['decision']['contrasts'][name]
            assert abs(D.mean()-rec['mean'])<1e-15
            assert np.array_equal(np.quantile(v,[.025,.975]),rec['95_CI'])
            assert np.array_equal(np.quantile(v,[.05/6,1-.05/6]),rec['98.333_CI'])
            if name!='data':
                good=(D.mean()/matrices[left].mean()>=.25 and D.mean()>=.001 and rec['98.333_CI'][0]>0 and np.max(-D.mean(0))<=.005)
                if name=='optimization':good=good and a['training_MSE'][right]<a['training_MSE'][left]
                assert bool(good)==rec['practical_optimization_or_parameterization_benefit']
        key='compact_n10_u20000';gap=matrices[key].mean();ridge=a['ridge']['compact_n10']['external']['normalized_MAE']
        go=(a['training_MAE'][key]<=.005 and gap>=.01 and gap<=1.1*ridge and
            a['decision']['contrasts']['data']['relative_reduction']>=.25 and a['decision']['contrasts']['data']['98.333_CI'][0]>0)
        assert bool(go)==a['decision']['GO_for_reviewing_future_acquisition_design']
        # Exact moments and histories from prescribed deterministic repeat.
        first=out/'fits/compact_n4_seed123000';repeat=out/'deterministic_repeat'
        for u in [5000,20000]:
            x=torch.load(first/f'update_{u}.pt',weights_only=True);y=torch.load(repeat/f'update_{u}.pt',weights_only=True)
            for k,v in x['model'].items():assert torch.equal(v,y['model'][k])
            for i,fields in x['optimizer']['state'].items():
                for k,v in fields.items():assert torch.equal(v,y['optimizer']['state'][i][k])
        assert read(first/'history.json')==read(repeat/'history.json')
        # Prescribed schedule bytes and initialization pairing, no retraining.
        for seed in c['seeds']:
            uniforms=np.random.Generator(np.random.PCG64(seed+1000000)).random((20000,256));uh=hashlib.sha256(uniforms.tobytes()).hexdigest()
            records=[]
            for rep in ['full','compact']:
                weights=[]
                for n in [4,10]:
                    f=out/'fits'/f'{rep}_n{n}_seed{seed}';rec=read(f/'completion.json');assert rec['updates']==20000 and rec['uniform_schedule_sha256']==uh
                    assert rec['index_schedule_sha256']==hashlib.sha256(np.floor(uniforms*300*n).astype(np.int64).tobytes()).hexdigest()
                    weights.append(torch.load(f/'initial.pt',weights_only=True)['model'])
                for k in weights[0]:assert torch.equal(weights[0][k],weights[1][k])
                records.append(weights[0])
            for k in ['network.2.weight','network.2.bias','network.4.weight','network.4.bias']:assert torch.equal(records[0][k],records[1][k])
        assert not blocked and not engine
    preservation={}
    if full_preservation:
        from src.quadrotor_archive_finalization import historical_preservation
        preservation=historical_preservation(root,code,read(old/'preservation_start.json'))
    result={'passed':True,'MLP_checkpoints':count,'independent_FP64_max_prediction_difference':max_delta,'ridge_systems':4,
        'max_ridge_backward_error':maxridge,'all_three_adjusted_CIs_independently_reconstructed':True,'repeat_parameters_moments_histories_bitwise':True,
        'tracked_source_files_preserved':len(snap['tracked_sha256']),'prior_untracked_files_preserved':len(snap['untracked_sha256']),
        'historical_preservation':preservation,'new_training_fits':0,'new_MPC_solves':0,'new_physics_steps':0}
    output.write_text(json.dumps(result,indent=2)+'\n');return result

def coverage_diagnostics(study):
    """Descriptive same-phase nearest-context distance, no independent-step p tests."""
    from scipy.stats import spearmanr
    out=Path(study);d=np.load(out/'processed.npz');c=read(out/'protocol.json');rows={}
    for n in [4,10]:
        x=d['compact'][c['partitions']['training_order'][:n]].astype(float);h=d['external_compact'].astype(float)
        # At identical phase, reference/reset terms are equal. This is a causal
        # context distance, not state-only or physical feasible-set distance.
        delta=h[:,None,:,:]-x[None,:,:,:];nearest=np.sqrt(np.mean(delta**2,axis=-1)).min(axis=1)
        m=read(out/'model_constants.json');target=(d['external_targets']-m['action_low'])/(np.array(m['action_high'])-m['action_low'])
        err=np.mean([abs(np.load(out/f'predictions_compact_n{n}_u20000_seed{s}.npz')['external']-target).mean(axis=-1) for s in c['seeds']],axis=0)
        rows[n]={'distance_definition':'RMS normalized 16-input difference to nearest training context at same reference index; source-reconstructed history',
            'phases':{name:{'distance_mean':float(nearest[:,lo:hi].mean()),'distance_p95':float(np.quantile(nearest[:,lo:hi],.95)),
                'MAE_mean':float(err[:,lo:hi].mean()),'descriptive_Spearman_distance_error':float(spearmanr(nearest[:,lo:hi].reshape(-1),err[:,lo:hi].reshape(-1)).statistic)} for name,(lo,hi) in c['phases'].items()},
            'no_transition_independence_or_causality_claim':True}
    return rows
