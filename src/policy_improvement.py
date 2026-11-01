"""Frozen E10 selection, reward-derived LQR and independent outcome criteria."""

from pathlib import Path
import time

import numpy as np
import scipy
from scipy.linalg import solve_discrete_are

from src.active_query import reset_at
from src.counterfactual import episode_metrics
from src.dataset import get_state
from src.horizon_experiment import immutable_json, read_json, sha256, write_json
from src.recoverability import validate_execution
from src.recoverable_pd import GAINS, case_identity, evaluate_gain, load_benchmark
from src.rollout_horizon import MeteredEnvironment
from src.safety import safety_margin

PILOT_IDS = [3280, 2956, 5383, 2568, 6194]
OPTIMIZATION_IDS = PILOT_IDS[1:]
OUTPUT = 'results/safe_policy_improvement/pilot'


def freeze_selection(root, config):
    """No simulator or controller call; serialize all 64 choices before evaluation."""
    root = Path(root).resolve(); output = root/OUTPUT
    benchmark = load_benchmark(root/config['benchmark'])
    source = root/config['baseline_archive']/'baseline_summary.json'
    rows = read_json(source)
    safe = [r for r in rows if not r['identity']['archived_unsafe']]
    assert len(safe) == 3331 and all(not r['metrics']['violated'] for r in safe)
    nonzero = sorted([r for r in safe if r['state_id'] != 3280],
                     key=lambda r: (-r['metrics']['returns'], r['state_id']))
    assert len(nonzero) == 3330
    rng = np.random.Generator(np.random.PCG64(101001))
    strata = []; selected = []; pilot = [3280]
    for group, indices in enumerate(np.array_split(np.arange(3330), 4), start=1):
        chosen = sorted(int(v) for v in rng.choice(indices, 16, replace=False))
        records = []
        for rank in chosen:
            old = nonzero[rank]; identity = case_identity(benchmark, old['state_id'])
            record = dict(identity, stratum=group, global_PD_cost_rank=rank,
                          PD_cost=-old['metrics']['returns'], PD_raw_minimum_margin=old['metrics']['minimum_margins'])
            records.append(record); selected.append(record)
        pilot.append(min(records, key=lambda r:(r['PD_cost'],r['state_id']))['state_id'])
        strata.append({'stratum':group, 'population_count':len(indices), 'selected':records})
    assert pilot == config['pilot_ids'] == PILOT_IDS
    assert len({r['grid_id'] for r in selected}) == len(selected) == 64
    manifest = {'seed':101001, 'generator':'PCG64', 'source_sha256':sha256(source),
                'benchmark_sha256':benchmark['metadata']['sha256'],
                'population_ids':[r['state_id'] for r in safe],
                'strata':strata, 'optimization_ids_sorted':sorted(r['grid_id'] for r in selected),
                'pilot_ids':pilot, 'remaining_optimization_ids':sorted(r['grid_id'] for r in selected if r['grid_id'] not in OPTIMIZATION_IDS)}
    assert len(manifest['remaining_optimization_ids']) == 60
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output/'selection.json', manifest)
    immutable_json(output/'selection_frozen.json', {'passed':True,
                   'selection_sha256':sha256(output/'selection.json'),
                   'new_controller_evaluations_before_freeze':0,
                   'basis':'Exact archived baseline data only; no new simulation, controller evaluation, or optimization.'})
    return manifest


def derive_lqr(config):
    c=config['lqr']; assert scipy.__version__ == c['scipy_version']
    a,b,q,r=(np.array(c[key],dtype=np.float64) for key in ('A','B','Q','R'))
    p=solve_discrete_are(a,b,q,r,balanced=c['balanced'])
    gain=np.linalg.solve(r+b.T@p@b,b.T@p@a)
    residual=a.T@p@a-p-(a.T@p@b)@gain+q
    poles=np.linalg.eigvals(a-b@gain)
    assert float(np.max(np.abs(residual))) <= c['maximum_absolute_DARE_residual']
    assert float(np.max(np.abs(poles))) < 1
    assert np.min(np.linalg.eigvalsh(p)) > 0
    repeated=solve_discrete_are(a,b,q,r,balanced=c['balanced'])
    assert p.tobytes() == repeated.tobytes()
    # Independent finite-horizon Riccati iteration checks the stabilizing branch.
    iterate=q.copy()
    for iteration in range(10000):
        k=np.linalg.solve(r+b.T@iterate@b,b.T@iterate@a)
        successor=q+a.T@iterate@a-(a.T@iterate@b)@k
        if np.max(np.abs(successor-iterate)) < 1e-13:
            iterate=successor; break
        iterate=successor
    else: raise AssertionError('Independent Riccati iteration failed to converge')
    assert np.max(np.abs(iterate-p)) < 1e-10
    return {'A':a.tolist(),'B':b.tolist(),'Q':q.tolist(),'R':r.tolist(),'P':p.tolist(),
            'K':gain.tolist(),'max_absolute_DARE_residual':float(np.max(np.abs(residual))),
            'frobenius_DARE_residual':float(np.linalg.norm(residual)),
            'closed_loop_poles':[[float(z.real),float(z.imag)] for z in poles],
            'spectral_radius':float(np.max(np.abs(poles))),
            'P_symmetric_error':float(np.max(np.abs(p-p.T))),
            'minimum_P_eigenvalue':float(np.min(np.linalg.eigvalsh(p))),
            'repeated_DARE_solution_bitwise':True,'independent_Riccati_iterations':iteration+1,
            'independent_Riccati_max_difference':float(np.max(np.abs(iterate-p))),
            'scipy_version':scipy.__version__,
            'scope':'Unconstrained local linear infinite-horizon design; no nonlinear global optimality or safety guarantee.'}


def controller_bank(lqr):
    return [{'name':f'PD_{kp:g}_{kd:g}','kp':kp,'kd':kd,'kind':'PD'} for kp,kd in GAINS] + [
            {'name':'LQR','kp':float(lqr['K'][0][0]),'kd':float(lqr['K'][0][1]),'kind':'LQR'}]


def evaluate_fixed(initial, seed, controller):
    if controller['kind']=='PD':
        return evaluate_gain(initial,seed,controller['kp'],controller['kd'])
    assert controller['kind']=='LQR' and controller['name']=='LQR'
    initial=np.asarray(initial,dtype=np.float64); horizon=200
    raw=np.empty((1,201,2)); states=np.empty_like(raw); margins=np.empty((1,201))
    actions=np.empty((1,200,1),dtype=np.float32); rewards=np.empty((1,200))
    terminated=np.zeros((1,200),bool); truncated=np.zeros_like(terminated)
    inputs=[]; commands=[]; observations=[]
    env=MeteredEnvironment(horizon); started=time.perf_counter()
    try:
        obs=reset_at(env,initial,seed)
        assert env.unwrapped.state.tobytes()==initial.tobytes()
        raw[0,0]=env.unwrapped.state; states[0,0]=get_state(obs); margins[0,0]=safety_margin(*raw[0,0])
        observations.append(str(obs.dtype))
        for t in range(horizon):
            theta,omega=get_state(obs)
            inputs.append([str(np.asarray(theta).dtype),str(np.asarray(omega).dtype)])
            action=np.clip(-controller['kp']*theta-controller['kd']*omega,-2.,2.)
            commands.append(str(np.asarray(action).dtype)); actions[0,t,0]=action
            obs,reward,term,trunc,_=env.step(actions[0,t])
            raw[0,t+1]=env.unwrapped.state; states[0,t+1]=get_state(obs)
            margins[0,t+1]=safety_margin(*raw[0,t+1]); rewards[0,t]=reward
            terminated[0,t]=term; truncated[0,t]=trunc; observations.append(str(obs.dtype))
            if term or (trunc and t!=199): raise RuntimeError('Unexpected simulator termination')
        costs={key:value.tolist() for key,value in env.costs().items()}
    finally: env.close()
    costs['wall_seconds']=time.perf_counter()-started
    arrays={'initial_conditions':initial[None].copy(),'states':states,'raw_states':raw,'margins':margins,
            'actions':actions,'rewards':rewards,'terminated':terminated,'truncated':truncated,
            'episode_returns':rewards.sum(axis=1),'episode_min_margins':margins.min(axis=1)}
    validate_execution(arrays,initial)
    assert observations==['float64']+['float32']*200
    assert inputs==[['float64']*2]+[['float32']*2]*199
    assert commands==['float64']+['float32']*199
    return arrays,costs,{'gains':[controller['kp'],controller['kd']],
                       'observation_dtypes':observations,'controller_input_dtypes':inputs,
                       'command_dtypes':commands,'initial_reset_bitwise':True,
                       'all_201_raw_states_checked':True,'executed_action_dtype':'float32'}


def measure(arrays, config):
    metrics=episode_metrics(arrays,config['discount'],config['dt'])
    result={key:value[0].item() for key,value in metrics.items()}
    raw=arrays['raw_states'][0]; u=arrays['actions'][0,:,0].astype(np.float64)
    if not result['violated']: result['first_crossing_step']=result['first_crossing_seconds']=None
    reward_angle=(raw[:-1,0]+np.pi)%(2*np.pi)-np.pi
    components={'angle_cost':float(np.sum(reward_angle**2)),
                'speed_cost':float(.1*np.sum(raw[:-1,1]**2)),
                'torque_reward_cost':float(.001*np.sum(u**2))}
    assert abs(sum(components.values())+result['returns']) < 1e-6
    result.update(components)
    result.update(initial_raw_margin=float(arrays['margins'][0,0]),
                  squared_torque_effort=float(config['dt']*np.sum(u**2)),
                  rms_torque=float(np.sqrt(np.mean(u**2))),peak_torque=float(np.max(np.abs(u))),
                  saturation_fraction=float(np.mean(np.abs(u)==2.)),
                  total_variation=float(np.sum(np.abs(np.diff(u)))),
                  initial_torque_jump=float(abs(u[0])),
                  terminal_theta=float(raw[-1,0]),terminal_omega=float(raw[-1,1]),
                  last20_maximum_abs_theta=float(np.max(np.abs(raw[-20:,0]))),
                  last20_maximum_abs_omega=float(np.max(np.abs(raw[-20:,1]))),
                  last20_squared_torque_effort=float(config['dt']*np.sum(u[-20:]**2)),
                  witness=bool(np.all(arrays['margins']>=0)))
    return result


def paired(record, baseline):
    return dict(record,delta_return=record['returns']-baseline['returns'],
                delta_discounted_return=record['discounted_returns']-baseline['discounted_returns'],
                delta_effort=record['squared_torque_effort']-baseline['squared_torque_effort'],
                delta_minimum_margin=record['minimum_margins']-baseline['minimum_margins'])


def assess(records, baselines, config):
    assert len(records)==len(baselines) and records
    c=config['criteria']; delta=np.array([r['returns']-b['returns'] for r,b in zip(records,baselines)])
    dm=np.array([r['minimum_margins']-b['minimum_margins'] for r,b in zip(records,baselines)])
    cost=-float(np.mean([b['returns'] for b in baselines]))
    threshold=max(c['mean_return_gain_floor'],c['relative_PD_mean_cost_gain']*cost)
    base_effort=float(np.mean([b['squared_torque_effort'] for b in baselines]))
    effort=float(np.mean([r['squared_torque_effort'] for r in records]))
    reduction=None if base_effort==0 else (base_effort-effort)/base_effort
    safety=all(r['witness'] for r in records) and float(dm.min()) >= -c['maximum_per_state_margin_loss']
    worst=float(delta.min()) >= c['minimum_per_state_paired_return_difference']
    eligible=np.array([b['initial_raw_margin']-b['minimum_margins'] >= c['pointwise_margin_improvement'] for b in baselines])
    margin_hits=int(np.sum((dm>=c['pointwise_margin_improvement']) & eligible))
    margin_rate=None if not eligible.any() else margin_hits/int(eligible.sum())
    return {'mean_return':float(np.mean([r['returns'] for r in records])),
            'worst_return':min(r['returns'] for r in records),
            'mean_discounted_return':float(np.mean([r['discounted_returns'] for r in records])),
            'mean_squared_torque_effort':effort,'mean_return_difference':float(delta.mean()),
            'minimum_per_state_paired_return_difference':float(delta.min()),
            'minimum_margin_difference':float(dm.min()),'minimum_raw_margin':min(r['minimum_margins'] for r in records),
            'violations':sum(r['violated'] for r in records),'return_threshold':threshold,
            'effort_reduction_fraction':reduction,'safety_and_margin_criterion':safety,
            'margin_eligible_count':int(eligible.sum()),'pointwise_margin_improvements':int(np.sum(dm>=c['pointwise_margin_improvement'])),
            'eligible_margin_improvements':margin_hits,'eligible_margin_improvement_fraction':margin_rate,
            'practical_margin_pass':None if margin_rate is None else bool(safety and worst and margin_rate>=c['minimum_fraction_of_margin_eligible_states_improved']
                                       and float(delta.mean())>=-c['maximum_mean_return_loss_for_secondary_objectives']),
            'worst_paired_return_criterion':worst,
            'practical_reward_pass':bool(safety and worst and float(delta.mean())>=threshold),
            'practical_effort_pass':bool(safety and worst and reduction is not None and reduction>=c['minimum_mean_effort_reduction_fraction']
                                       and float(delta.mean())>=-c['maximum_mean_return_loss_for_secondary_objectives'])}


def safe_oracle(records, baseline, mode):
    candidates=[r for r in records if r.get('phase')=='original' and r.get('mode')==mode
                and r.get('candidate_available') and r.get('witness')]
    best=min(candidates,key=lambda r:(-r['returns'],r['start_index'])) if candidates else None
    genuine=best is not None and best['returns']>baseline['returns']
    result={'mode':mode,'source':'optimized' if genuine else 'PD_fallback',
            'optimized_safe_candidate_available':bool(candidates),
            'optimized_safe_improvement':genuine,
            'selected_start':best['start'] if genuine else None,
            'safe_candidate_count':len(candidates),
            'selected_metrics':best if genuine else baseline}
    result['delta_return']=result['selected_metrics']['returns']-baseline['returns']
    result['gain_above_1e-6']=result['delta_return']>1e-6
    result['gain_above_0.01']=result['delta_return']>.01
    result['gain_above_0.1']=result['delta_return']>.1
    return result
