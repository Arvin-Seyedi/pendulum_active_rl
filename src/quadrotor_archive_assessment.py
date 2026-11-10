"""Whole-trajectory metrics, paired uncertainty and fixed B0 decision rules."""
import numpy as np
from scipy.stats import spearmanr

def trajectory_metrics(pred,target,action_width,phases):
    diff=np.asarray(pred)-np.asarray(target);physical=diff*np.asarray(action_width)
    def metrics(e,p):
        return {'normalized_MAE':float(np.abs(e).mean()),'normalized_RMSE':float(np.sqrt(np.square(e).mean())),
                'normalized_p95':float(np.quantile(np.abs(e),.95)),'normalized_max':float(np.abs(e).max()),
                'physical_MAE_N':float(np.abs(p).mean()),'physical_RMSE_N':float(np.sqrt(np.square(p).mean())),
                'physical_p95_N':float(np.quantile(np.abs(p),.95)),'physical_max_N':float(np.abs(p).max()),
                'physical_MAE_by_channel_N':np.abs(p).mean(0).tolist()}
    result=metrics(diff,physical);result['phases']={name:metrics(diff[lo:hi],physical[lo:hi]) for name,(lo,hi) in phases.items()};return result

def bootstrap_means(values,seed=122999,replicates=20000):
    """Crossed seed/trajectory bootstrap; no adjacent transitions resampled."""
    v=np.asarray(values,dtype=np.float64)
    if v.ndim!=2:raise ValueError('Expected training_seed by independent_trajectory')
    rng=np.random.Generator(np.random.PCG64(seed));s=rng.integers(v.shape[0],size=(replicates,v.shape[0]));t=rng.integers(v.shape[1],size=(replicates,v.shape[1]))
    means=v[s[:,:,None],t[:,None,:]].mean((1,2))
    return {'mean':float(v.mean()),'95_CI':np.quantile(means,[.025,.975]).tolist(),'seed_mean_sample_SD':float(np.std(v.mean(1),ddof=1)) if len(v)>1 else None,
            'seed_mean_range':[float(v.mean(1).min()),float(v.mean(1).max())],'worst_trajectory_mean':float(v.mean(0).max()),'worst_seed_trajectory':float(v.max())}

def coverage(data,train_ids,assessment_ids,per_step_errors):
    state=data['states'][:,:300];width=np.asarray(data['model']['state_high'])-data['model']['state_low']
    # Distances are conditioned on equal phase, not pooled across the moving task.
    d=np.sqrt(np.mean(((state[assessment_ids,:,None,:]-state[train_ids].transpose(1,0,2)[None,:,:,:])/width)**2,axis=-1)).min(-1)
    def stats(a):return {'mean':float(np.mean(a)),'median':float(np.median(a)),'p95':float(np.quantile(a,.95)),'maximum':float(np.max(a))}
    corr=spearmanr(d.reshape(-1),np.asarray(per_step_errors).reshape(-1))
    return {'definition':'same-phase nearest training physical state RMS / full component constraint ranges; partial context descriptor, not safety distance',
            'overall':stats(d),'early':stats(d[:,:50]),'late':stats(d[:,200:]),'distance_error_Spearman':float(corr.statistic),
            'Spearman_inference':'descriptive only; adjacent transitions are dependent; no transition-level p-value'}

def decision(curves,baseline,config):
    p=config['partitions'];c=config['go_criteria'];base=curves[str(p['primary_base_size'])];large=curves[str(p['primary_large_size'])]
    b=np.asarray(base['assessment_seed_trajectory_MAE']);l=np.asarray(large['assessment_seed_trajectory_MAE']);improvement=bootstrap_means(b-l,config['uncertainty']['bootstrap_seed'],config['uncertainty']['replicates'])
    ratio=float(l.mean()/b.mean()) if b.mean()>0 else float('inf')
    criteria={'meaningful_base_gap':bool(b.mean()>=c['meaningful_base_macro_normalized_MAE_at_least'] and base['assessment_summary']['95_CI'][0]>=c['base_hierarchical_CI_lower_at_least']),
      'at_least_25_percent_reduction':bool(ratio<=c['large_to_base_macro_MAE_ratio_at_most']),
      'positive_paired_improvement_CI':bool(improvement['95_CI'][0]>0),
      'no_large_trajectory_deterioration':bool(np.max(l.mean(0)-b.mean(0))<=c['maximum_per_trajectory_macro_MAE_deterioration']),
      'early_phase_does_not_worsen':bool(large['assessment_phase_macro_MAE']['early']<=base['assessment_phase_macro_MAE']['early']),
      'not_clear_underfit_relative_to_ridge':bool(l.mean()<=c['large_MLP_MAE_at_most_ridge_multiple']*baseline[str(p['primary_large_size'])]['assessment_macro_MAE'])}
    return {'GO_for_designing_future_acquisition':bool(all(criteria.values())),'criteria':criteria,'paired_base_minus_large':improvement,'large_to_base_ratio':ratio,
            'interpretation':'GO is design only, never acquisition/closed-loop/safety approval; a failed gap criterion does not prove no other learner can benefit'}
