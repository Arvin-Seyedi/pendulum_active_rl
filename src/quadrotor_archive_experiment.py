"""Sealed B0 archive-only preflight, fixed fits and supervised assessment."""
from pathlib import Path
import hashlib,json,resource,time,subprocess
import numpy as np
from src.quadrotor_pilot import read,sha
from src.quadrotor_mmaq_v1 import safe_write
from src import quadrotor_archive_data as data_api

CONFIG='configs/experiment_12_b0_archive_v1.json'
SOURCES=[CONFIG,'src/quadrotor_archive_data.py','src/quadrotor_archive_learner.py','src/quadrotor_archive_assessment.py','src/quadrotor_archive_experiment.py',
         'experiments/12_b0_archive.py','tests/test_quadrotor_archive_b0.py','docs/experiment_12_b0_protocol.md']

def immutable(path,value):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    safe_write(path,value)

class RunLimits:
    def __init__(self,out):
        s=read(Path(out)/'resource_ledger.json');self.base=s['charged_cumulative'];self.caps=s['caps'];self.cpu=time.process_time();self.wall=time.perf_counter()
    def __call__(self):
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>4294967296:raise MemoryError('4 GiB memory cap')
        if self.base['cpu_seconds']+time.process_time()-self.cpu>=self.caps['cpu_seconds']-2:raise TimeoutError('B0 CPU cap; reserve failure persistence')
        if self.base['wall_seconds']+time.perf_counter()-self.wall>=self.caps['wall_seconds']-2:raise TimeoutError('B0 active-wall cap')

def seal(root,out):
    root,out=Path(root),Path(out);config=read(root/CONFIG);subsets=data_api.validate_partitions(config)
    immutable(out/'protocol.json',config)
    immutable(out/'protocol_seal.json',{'version':config['version'],'source_sha256':{n:sha(root/n) for n in SOURCES},'authorization':'archive-only B0; no expert/physics/RL/IQL',
        'training_not_started':True,'historical_OVERALL':'FAILED','historical_0001':'FAILED'})
    immutable(out/'split_manifest.json',{'training':config['partitions']['training_order'],'validation':config['partitions']['validation'],'assessment':config['partitions']['assessment'],
        'nested_subsets':subsets,'independent_units':'16 complete original trajectories; repeats excluded','assessment_status':'partition of reviewed archive, not fresh independent policy evaluation'})

def check_seal(root,out):
    root,out=Path(root),Path(out);s=read(out/'protocol_seal.json')
    for n,h in s['source_sha256'].items():assert sha(root/n)==h,n
    assert read(root/CONFIG)==read(out/'protocol.json')
    return read(out/'protocol.json')

def load_processed(out):
    a=np.load(Path(out)/'processed.npz');provenance=read(Path(out)/'data_provenance.json')
    for name,key in [('features','feature_sha256'),('targets','target_sha256')]:assert hashlib.sha256(a[name].tobytes()).hexdigest()==provenance[key]
    return dict(a),read(Path(out)/'model_constants.json')

def preflight(root,upstream,out):
    root,out=Path(root),Path(out);config=check_seal(root,out);limits=RunLimits(out)
    from src.quadrotor_mpc80_feasibility_v2 import zero_real_operations
    with zero_real_operations(upstream) as (blocked,engine):
        identity=data_api.runtime_identity(root,upstream);data=data_api.load_archive(root);limits()
        import torch
        assert torch.__version__=='2.8.0';torch.set_num_threads(1);torch.set_num_interop_threads(1)
        lib=Path(torch.__file__).parent/'lib/libtorch_cpu.dylib'
        identity.update(torch_version=torch.__version__,torch_git_version=torch.version.git_version,torch_cpu_library_sha256=sha(lib),torch_num_threads=1)
        immutable(out/'runtime_identity.json',identity);immutable(out/'data_provenance.json',data['provenance']);immutable(out/'model_constants.json',data['model'])
        np.savez_compressed(out/'processed.npz',features=data['features'],targets=data['targets'],states=data['states'],submitted=data['submitted'],reference=data['reference'])
        # Every asserted sample is observational data, never a newly integrated state.
        import sys,unittest
        sys.path.insert(0,str(root/'tests'));suite=unittest.defaultTestLoader.loadTestsFromName('test_quadrotor_archive_b0')
        with (out/'preflight_tests.log').open('x') as f:tests=unittest.TextTestRunner(stream=f,verbosity=2).run(suite)
        if not tests.wasSuccessful():raise RuntimeError('Preflight tests failed; scientific training forbidden')
        assert not blocked and not engine
    immutable(out/'preflight.json',{'passed':True,'tests':tests.testsRun,'original_trajectories':16,'original_records':4800,'deterministic_repeat_records_excluded':4800,
        'candidate_parameter_reference_action_checks':4800,'no_optimizer_or_engine_calls':not blocked and not engine,'training_started':False,
        'processed_sha256':sha(out/'processed.npz'),'split_sha256':sha(out/'split_manifest.json'),'runtime_identity_sha256':sha(out/'runtime_identity.json')})

def train(root,upstream,out):
    root,out=Path(root),Path(out);config=check_seal(root,out);p=read(out/'preflight.json');assert p['passed'] and sha(out/'processed.npz')==p['processed_sha256']
    if (out/'training_started.json').exists():raise FileExistsError('No implicit training retry or resume')
    from src.quadrotor_mpc80_feasibility_v2 import zero_real_operations
    from src import quadrotor_archive_learner as learn
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    data,model=load_processed(out);x=data['features'];y=data_api.normalized_targets(data['targets'],model);limits=RunLimits(out)
    immutable(out/'training_started.json',{'fits':20,'reproducibility_repeats':1,'baseline_fits':4,'updates_per_fit':5000,'assessment_locked':True})
    nonassessment=config['partitions']['training_order']+config['partitions']['validation'];jobs=[];initial={};schedule={}
    with zero_real_operations(upstream) as (blocked,engine):
        data_api.runtime_identity(root,upstream)
        for n in config['partitions']['subset_sizes']:
            ids=config['partitions']['training_order'][:n];xx=x[ids].reshape(-1,496);yy=y[ids].reshape(-1,2)
            beta,diagnostic=learn.ridge_fit(xx,yy,config['baseline']['regularization']);np.savez_compressed(out/f'ridge_{n}.npz',coefficients=beta);immutable(out/f'ridge_{n}.json',diagnostic)
            for seed in config['seeds']:
                check_seal(root,out);limits();folder=out/'fits'/f'n{n}_seed{seed}'
                net,result=learn.fit_fixed(xx,yy,seed,config,folder,limits)
                if seed in initial:assert result['initial_parameters_sha256']==initial[seed] and result['uniform_schedule_sha256']==schedule[seed]
                else:initial[seed]=result['initial_parameters_sha256'];schedule[seed]=result['uniform_schedule_sha256']
                # No assessment feature/label enters fit or these diagnostics.
                pred=learn.predict(net,x[nonassessment].reshape(-1,496));np.savez_compressed(folder/'nonassessment_predictions.npz',predictions=pred,trajectory_ids=nonassessment)
                jobs.append({'size':n,'seed':seed,'folder':str(folder.relative_to(out)),'result':result})
                safe_write(out/'training_progress.json',{'complete_scientific_fits':len(jobs),'planned':20,'assessment_still_locked':True})
        repeat=config['reproducibility_repeat'];n=repeat['subset_size'];seed=repeat['seed'];ids=config['partitions']['training_order'][:n]
        net,again=learn.fit_fixed(x[ids].reshape(-1,496),y[ids].reshape(-1,2),seed,config,out/'reproducibility_repeat',limits)
        old=read(out/'fits'/f'n{n}_seed{seed}'/'completion.json')
        for k in ['initial_parameters_sha256','final_parameters_sha256','uniform_schedule_sha256','index_schedule_sha256','history']:assert old[k]==again[k],k
        # Optimizer tensor moments and parameter restoration, not container-file timestamps.
        first=torch.load(out/'fits'/f'n{n}_seed{seed}'/'final.pt',weights_only=True,map_location='cpu');second=torch.load(out/'reproducibility_repeat/final.pt',weights_only=True,map_location='cpu')
        for k in first['model']:assert torch.equal(first['model'][k],second['model'][k])
        for i,values in first['optimizer']['state'].items():
            for k,v in values.items():assert torch.equal(v,second['optimizer']['state'][i][k])
        assert not blocked and not engine
    immutable(out/'training_complete.json',{'passed':True,'scientific_fits':20,'baseline_fits':4,'repeated_fit_bitwise':True,'optimizer_state_bitwise':True,'source_seal_passed':True,'new_MPC_solves':0,'new_physics_steps':0})
    immutable(out/'assessment_unlocked.json',{'after_all_scientific_fits_and_reproducibility_checks':True,'final_checkpoints':{str(p.relative_to(out)):sha(p) for p in sorted((out/'fits').glob('*/final.pt'))}})

def assess(root,upstream,out):
    root,out=Path(root),Path(out);config=check_seal(root,out);assert read(out/'training_complete.json')['passed']
    if (out/'assessment.json').exists():raise FileExistsError('Assessment is immutable; no re-selection')
    from src import quadrotor_archive_learner as learn
    from src import quadrotor_archive_assessment as a
    from src.quadrotor_mpc80_feasibility_v2 import zero_real_operations
    import torch
    torch.set_num_threads(1);data,model=load_processed(out);y=data_api.normalized_targets(data['targets'],model);width=np.asarray(model['action_high'])-model['action_low'];xx=data['features'].reshape(-1,496)
    test=config['partitions']['assessment'];val=config['partitions']['validation'];curves={};baseline={};all_metrics=[];limits=RunLimits(out)
    with zero_real_operations(upstream) as (blocked,engine):
        for n in config['partitions']['subset_sizes']:
            train_ids=config['partitions']['training_order'][:n];size_metrics=[];seed_traj=[];step_errors=[];phase={k:[] for k in config['phases']}
            for seed in config['seeds']:
                limits();folder=out/'fits'/f'n{n}_seed{seed}';checkpoint=torch.load(folder/'final.pt',weights_only=True,map_location='cpu');net=learn.ArchivePolicy();net.load_state_dict(checkpoint['model'])
                assert learn.parameters_sha(net)==read(folder/'completion.json')['final_parameters_sha256']
                pred=learn.predict(net,xx).reshape(16,300,2);assert np.isfinite(pred).all()
                # Restoration predictions must repeat bitwise, with fixed evaluation batching.
                restored=learn.ArchivePolicy();restored.load_state_dict(torch.load(folder/'final.pt',weights_only=True,map_location='cpu')['model']);assert pred.tobytes()==learn.predict(restored,xx).reshape(16,300,2).tobytes()
                np.savez_compressed(out/f'predictions_n{n}_seed{seed}.npz',normalized_predictions=pred)
                metrics={str(i):a.trajectory_metrics(pred[i],y[i],width,config['phases']) for i in range(16)};size_metrics.append(metrics)
                seed_traj.append([metrics[str(i)]['normalized_MAE'] for i in test]);step_errors.append(np.abs(pred[test]-y[test]).mean(-1))
                for name in phase:phase[name].append(np.mean([metrics[str(i)]['phases'][name]['normalized_MAE'] for i in test]))
                all_metrics.append({'size':n,'seed':seed,'trajectories':metrics})
            matrix=np.asarray(seed_traj);curve={'labels':n*300,'training_ids':train_ids,'assessment_seed_trajectory_MAE':matrix.tolist(),
                'assessment_summary':a.bootstrap_means(matrix,config['uncertainty']['bootstrap_seed'],config['uncertainty']['replicates']),
                'training_macro_MAE':float(np.mean([m[str(i)]['normalized_MAE'] for m in size_metrics for i in train_ids])),
                'validation_macro_MAE':float(np.mean([m[str(i)]['normalized_MAE'] for m in size_metrics for i in val])),
                'assessment_phase_macro_MAE':{k:float(np.mean(v)) for k,v in phase.items()},
                'assessment_per_trajectory':{str(i):{'mean_MAE':float(matrix[:,j].mean()),'seed_SD':float(matrix[:,j].std(ddof=1)),
                    'mean_p95':float(np.mean([m[str(i)]['normalized_p95'] for m in size_metrics])),'maximum_error':float(max(m[str(i)]['normalized_max'] for m in size_metrics))} for j,i in enumerate(test)}}
            curve['coverage']=a.coverage({'states':data['states'],'model':model},train_ids,test,np.mean(step_errors,axis=0));curves[str(n)]=curve
            beta=np.load(out/f'ridge_{n}.npz')['coefficients'];bp,raw=learn.ridge_predict(beta,xx);bp=bp.reshape(16,300,2);raw=raw.reshape(16,300,2)
            bm={str(i):a.trajectory_metrics(bp[i],y[i],width,config['phases']) for i in range(16)}
            baseline[str(n)]={'assessment_macro_MAE':float(np.mean([bm[str(i)]['normalized_MAE'] for i in test])),
                'assessment_raw_macro_MAE':float(np.abs(raw[test]-y[test]).mean()),'assessment_clipped_fraction':float(np.mean((raw[test]<0)|(raw[test]>1))),
                'training_macro_MAE':float(np.mean([bm[str(i)]['normalized_MAE'] for i in train_ids])),'validation_macro_MAE':float(np.mean([bm[str(i)]['normalized_MAE'] for i in val])),
                'trajectories':bm}
        assert not blocked and not engine
    verdict=a.decision(curves,baseline,config)
    immutable(out/'assessment.json',{'version':config['version'],'curves':curves,'ridge':baseline,'per_seed_trajectory_metrics':all_metrics,'decision':verdict,
        'action_width_N':width.tolist(),'assessment_trajectory_ids':test,'uncertainty':'crossed seed/trajectory resampling; conditional reviewed archive; no transition-level independence',
        'all_checkpoint_predictions_restored_bitwise':True,'new_solves':0,'new_physics_steps':0,'closed_loop_return_or_safety_evaluated':False})

