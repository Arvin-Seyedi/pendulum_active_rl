"""B1 paired whole-trajectory assessment; no learner selection or fitting."""
import numpy as np

def metrics(prediction,target,width,phases):
    p,y=np.asarray(prediction),np.asarray(target);e=np.abs(p-y)
    if p.shape!=y.shape or not np.isfinite(e).all():raise ValueError('Invalid predictions')
    physical=e*np.asarray(width);where=np.unravel_index(np.argmax(physical),physical.shape)
    near=(y<=.01)|(y>=.99)
    return {'normalized_MAE':float(e.mean()),'normalized_MSE':float(np.mean(e**2)),
        'physical_MAE_N':float(physical.mean()),'physical_MSE_N2':float(np.mean(physical**2)),
        'physical_RMSE_N':float(np.sqrt(np.mean(physical**2))),'normalized_p95':float(np.quantile(e,.95)),
        'physical_p95_N':float(np.quantile(physical,.95)),'worst_action_error_N':float(physical.max()),
        'worst_action_index':list(map(int,where)),'near_bound_count':int(near.sum()),
        'near_bound_MAE':float(e[near].mean()) if near.any() else None,
        'trajectories':[{'normalized_MAE':float(a.mean()),'normalized_MSE':float(np.mean(a**2)),
            'physical_MAE_N':float(b.mean()),'physical_MSE_N2':float(np.mean(b**2)),
            'worst_action_error_N':float(b.max()),'normalized_p95':float(np.quantile(a,.95))} for a,b in zip(e,physical)],
        'phases':{name:{'normalized_MAE':float(e[:,lo:hi].mean()),'normalized_MSE':float(np.mean(e[:,lo:hi]**2)),
            'physical_MAE_N':float(physical[:,lo:hi].mean()),'physical_MSE_N2':float(np.mean(physical[:,lo:hi]**2))} for name,(lo,hi) in phases.items()}}

def bootstrap(matrix,c):
    v=np.asarray(matrix)
    if v.shape!=(5,4) or not np.isfinite(v).all():raise ValueError('Five seeds by four independent starts required')
    rng=np.random.Generator(np.random.PCG64(c['uncertainty']['bootstrap_seed']));n=c['uncertainty']['replicates']
    si=rng.integers(5,size=(n,5));ti=rng.integers(4,size=(n,4));dist=v[si[:,:,None],ti[:,None,:]].mean((1,2))
    return {'mean':float(v.mean()),'95_CI':np.quantile(dist,[.025,.975]).tolist(),
        '98.333_CI':np.quantile(dist,c['uncertainty']['primary_interval']).tolist(),
        'seed_means_SD':float(v.mean(1).std(ddof=1)),'seed_mean_range':[float(v.mean(1).min()),float(v.mean(1).max())],
        'seed_trajectory_matrix':v.tolist()}

def decide(matrices,training_mse,training_mae,ridge,c):
    pairs={'optimization':('full_n10_u5000','full_n10_u20000'),
        'parameterization':('full_n10_u20000','compact_n10_u20000'),
        'data':('compact_n4_u20000','compact_n10_u20000')}
    results={}
    for name,(a,b) in pairs.items():
        d=matrices[a]-matrices[b];v=bootstrap(d,c);relative=float(d.mean()/matrices[a].mean())
        v.update({'relative_reduction':relative,'maximum_start_mean_deterioration':float(np.max(-d.mean(0)))})
        practical=relative>=.25 and d.mean()>=.001 and v['98.333_CI'][0]>0 and v['maximum_start_mean_deterioration']<=.005
        if name=='optimization':practical=bool(practical and training_mse[b]<training_mse[a])
        v['practical_optimization_or_parameterization_benefit']=bool(practical) if name!='data' else None
        results[name]=v
    key='compact_n10_u20000';go={'training_MAE_at_most_.005':training_mae[key]<=.005,
        'external_MAE_at_least_.01':matrices[key].mean()>=.01,
        'external_MAE_within_1.10_ridge':matrices[key].mean()<=1.1*ridge,
        'data_relative_reduction_at_least_.25':results['data']['relative_reduction']>=.25,
        'data_adjusted_CI_positive':results['data']['98.333_CI'][0]>0}
    return {'contrasts':results,'acquisition_design_gates':{k:bool(v) for k,v in go.items()},
        'GO_for_reviewing_future_acquisition_design':bool(all(go.values())),
        'historical_B0_NO_GO_preserved':True,'closed_loop_performance_or_safety_tested':False}
