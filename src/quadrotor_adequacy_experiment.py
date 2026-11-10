"""Sealed, zero-solve B1 stages. External predictions unlock after all fits."""
from pathlib import Path
import hashlib,json,subprocess,time,tempfile
import numpy as np
import torch
from src.quadrotor_pilot import read,sha
from src.quadrotor_mmaq_v1 import safe_write
from src.quadrotor_archive_experiment import RunLimits
from src.quadrotor_mpc80_feasibility_v2 import zero_real_operations
from src.quadrotor_adequacy_data import BASE,CONFIG,audit,compact,same,rpm_commands
from src import quadrotor_adequacy_learner as learner
from src import quadrotor_adequacy_assessment as assess

SOURCES=[CONFIG,'src/quadrotor_adequacy_data.py','src/quadrotor_adequacy_learner.py','src/quadrotor_adequacy_assessment.py',
    'src/quadrotor_adequacy_experiment.py','experiments/12_b1_archive.py','tests/test_quadrotor_adequacy_b1.py',
    'docs/experiment_12_b1_protocol.md']

def immutable(path,value):
    if Path(path).exists():raise FileExistsError(path)
    safe_write(path,value)

def check_seal(code,out):
    s=read(Path(out)/'protocol_seal.json')
    for n,h in s['sources'].items():assert sha(Path(code)/n)==h,n
    for n,h in s['artifacts'].items():assert sha(Path(out)/n)==h,n
    assert read(Path(out)/'preflight.json')['passed']
    return s

def preflight(root,upstream,out):
    root,upstream,out=map(Path,(root,upstream,out));c=read(root/CONFIG)
    if (out/'protocol_seal.json').exists():raise FileExistsError('No overwrite')
    limits=RunLimits(out);start=time.process_time();wall=time.perf_counter()
    from src.quadrotor_archive_finalization import verify
    # Full independent B0 metrics, moments, checkpoints, seeds, sources and history.
    # Original verifier forbids writing to any new benchmark evidence subtree.
    # Keep that guard unchanged; copy its returned read-only audit to B1.
    with tempfile.TemporaryDirectory(prefix='b1_read_only_') as temp:
        baseline=verify(root,root,upstream,Path(temp)/'audit.json',full_preservation=True)
    immutable(out/'b0_read_only_verification.json',baseline)
    with zero_real_operations(upstream) as (blocked,engine):
        provenance,data,model=audit(root,upstream)
        from src.quadrotor_action_mock_v2 import native_harness
        from src.quadrotor_applied_action_v2 import AppliedActionObserver
        mock_count=0
        for i in range(4):
            a=np.load(root/BASE/'mmaq_v1/runs'/f'H{i}_nonlinear80_1000_original/trajectory.npz')
            with native_harness(upstream,model,a['physics_states'][:41]) as (env,b):
                with AppliedActionObserver(env,b) as observer:
                    for command in a['actions'][:2]:env._advance_simulation(env._preprocess_control(command))
                    for k,p in enumerate(observer.steps):
                        assert same(p['direct_rpm_argument'],a['rpms'][k//20])
                        assert same(p['grouped_submitted_force_argument_sums'],rpm_commands(a['rpms'],model)[k//20])
                    mock_count+=len(observer.steps)
        assert not blocked and not engine
    limits()
    assert time.process_time()-start<60 and time.perf_counter()-wall<120,'Preflight subcap'
    assert c['partitions']['training_order']==[0,4,8,12,5,6,13,14,9,1] and c['partitions']['validation']==[2,10]
    assert c['partitions']['assessment']==[3,7,11,15]
    np.savez_compressed(out/'processed.npz',**data);immutable(out/'model_constants.json',model);immutable(out/'protocol.json',c)
    immutable(out/'external_provenance.json',provenance)
    result={'passed':True,'B0_verified':baseline['passed'],'B0_feature_rows':4800,'external_labels':1200,'native_mock_boundaries':mock_count,
        'new_MPC_solves':0,'new_physics_steps':0,'external_learner_predictions':0,'CPU_seconds':time.process_time()-start,
        'external_history':'source-reconstructed command, not direct execution measurement','missing_MPC_history_limitation':True}
    immutable(out/'preflight.json',result)
    immutable(out/'protocol_seal.json',{'version':c['version'],'sources':{n:sha(root/n) for n in SOURCES},
        'artifacts':{n:sha(out/n) for n in ['protocol.json','processed.npz','model_constants.json','external_provenance.json','preflight.json','preservation_start.json']},
        'approved_design_sha256':sha(root/'docs/experiment_12_b1_adequacy_design.md')})
    return result

def normalized(data,model,key):return (data[key]-model['action_low'])/(np.array(model['action_high'])-model['action_low'])

def train(root,upstream,out):
    root,upstream,out=map(Path,(root,upstream,out));check_seal(root,out);c=read(out/'protocol.json');d=np.load(out/'processed.npz');m=read(out/'model_constants.json');y=normalized(d,m,'targets');check=RunLimits(out)
    if (out/'fits').exists():raise FileExistsError('No automatic retry/resume')
    with zero_real_operations(upstream) as (blocked,engine):
        completions=[]
        for rep in ['full','compact']:
            for n in [4,10]:
                ids=c['partitions']['training_order'][:n];x=d[rep][ids].reshape(-1,d[rep].shape[-1]);yy=y[ids].reshape(-1,2)
                for seed in c['seeds']:
                    v=learner.fit(x,yy,seed,c,out/'fits'/f'{rep}_n{n}_seed{seed}',check);completions.append(v)
                    print(rep,n,seed,v['CPU_seconds'],flush=True)
                check();beta,diag=learner.ridge_fit(x,yy,.001);np.savez_compressed(out/f'ridge_{rep}_n{n}.npz',coefficients=beta)
                immutable(out/f'ridge_{rep}_n{n}.json',diag)
        ids=c['partitions']['training_order'][:4]
        learner.fit(d['compact'][ids].reshape(-1,16),y[ids].reshape(-1,2),123000,c,out/'deterministic_repeat',check)
        original=out/'fits/compact_n4_seed123000';repeat=out/'deterministic_repeat'
        for u in c['checkpoints']:
            a=torch.load(original/f'update_{u}.pt',weights_only=True);b=torch.load(repeat/f'update_{u}.pt',weights_only=True)
            for key,v in a['model'].items():assert torch.equal(v,b['model'][key])
            for key,fields in a['optimizer']['state'].items():
                for name,v in fields.items():assert torch.equal(v,b['optimizer']['state'][key][name])
        assert read(original/'history.json')==read(repeat/'history.json')
        for rep in ['full','compact']:
            for seed in c['seeds']:
                a=read(out/'fits'/f'{rep}_n4_seed{seed}'/'completion.json');b=read(out/'fits'/f'{rep}_n10_seed{seed}'/'completion.json')
                for name in ['initial_parameters_sha256','uniform_schedule_sha256']:assert a[name]==b[name]
        assert not blocked and not engine
    immutable(out/'fits_complete.json',{'scientific_MLP_fits':20,'scientific_checkpoints':40,'repeated_updates':20000,'total_updates':420000,'ridge_fits':4,
        'repeat_bitwise':True,'external_predictions_not_yet_computed':True,'completions':completions})

def evaluate(root,upstream,out):
    root,upstream,out=map(Path,(root,upstream,out));check_seal(root,out);assert read(out/'fits_complete.json')['repeat_bitwise']
    immutable(out/'assessment_unlocked.json',{'fits_complete_sha256':sha(out/'fits_complete.json'),'all_prespecified_fits_completed':True})
    c=read(out/'protocol.json');d=np.load(out/'processed.npz');m=read(out/'model_constants.json');width=np.array(m['action_high'])-m['action_low']
    Y=normalized(d,m,'targets');H=normalized(d,m,'external_targets');rows={};matrices={};training_mse={};training_mae={};ridge={};check=RunLimits(out)
    with zero_real_operations(upstream) as (blocked,engine):
        for rep in ['full','compact']:
            for n in [4,10]:
                ids=c['partitions']['training_order'][:n];X=d[rep];HX=d['external_'+rep];dim=X.shape[-1]
                for u in c['checkpoints']:
                    key=f'{rep}_n{n}_u{u}';rec=[];matrix=[];tmse=[];tmae=[]
                    for seed in c['seeds']:
                        check();net=learner.AdequacyPolicy(dim,seed);ck=out/'fits'/f'{rep}_n{n}_seed{seed}'/f'update_{u}.pt';net.load_state_dict(torch.load(ck,weights_only=True)['model'])
                        pred=learner.predict(net,X.reshape(-1,dim)).reshape(16,300,2);hp=learner.predict(net,HX.reshape(-1,dim)).reshape(4,300,2)
                        np.savez_compressed(out/f'predictions_{key}_seed{seed}.npz',development=pred,external=hp)
                        a=assess.metrics(pred[ids],Y[ids],width,c['phases']);v=assess.metrics(pred[c['partitions']['validation']],Y[c['partitions']['validation']],width,c['phases']);h=assess.metrics(hp,H,width,c['phases'])
                        dev=assess.metrics(pred[c['partitions']['assessment']],Y[c['partitions']['assessment']],width,c['phases'])
                        rec.append({'seed':seed,'training':a,'validation':v,'external':h,'reused_development':dev});matrix.append([z['normalized_MAE'] for z in h['trajectories']]);tmse.append(a['normalized_MSE']);tmae.append(a['normalized_MAE'])
                    matrices[key]=np.array(matrix);training_mse[key]=float(np.mean(tmse));training_mae[key]=float(np.mean(tmae));rows[key]={'per_seed':rec,'external_summary':assess.bootstrap(matrices[key],c)}
                beta=np.load(out/f'ridge_{rep}_n{n}.npz')['coefficients'];p,raw=learner.ridge_predict(beta,X.reshape(-1,dim));hp,hraw=learner.ridge_predict(beta,HX.reshape(-1,dim));p=p.reshape(16,300,2);hp=hp.reshape(4,300,2)
                np.savez_compressed(out/f'predictions_ridge_{rep}_n{n}.npz',development=p,external=hp,raw_development=raw.reshape(16,300,2),raw_external=hraw.reshape(4,300,2))
                ridge[f'{rep}_n{n}']={'training':assess.metrics(p[ids],Y[ids],width,c['phases']),'validation':assess.metrics(p[c['partitions']['validation']],Y[c['partitions']['validation']],width,c['phases']),
                    'external':assess.metrics(hp,H,width,c['phases']),'raw_external':assess.metrics(hraw.reshape(4,300,2),H,width,c['phases'])}
        assert not blocked and not engine
    decision=assess.decide(matrices,training_mse,training_mae,ridge['compact_n10']['external']['normalized_MAE'],c)
    result={'MLP':rows,'ridge':ridge,'decision':decision,'training_MSE':training_mse,'training_MAE':training_mae,'width_N':width.tolist(),
        'external_prior_exposure':'expert diagnostics only, not globally untouched','closed_loop_tested':False}
    immutable(out/'assessment.json',result);return result
