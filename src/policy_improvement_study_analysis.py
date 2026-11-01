"""Independent full E10 reconstruction, oracle accounting and reporting."""

from collections import Counter
import csv
import json
from pathlib import Path
import time

import numpy as np

from src.horizon_experiment import _assert_hashes,load_npz,read_json,sha256,write_json
from src.policy_improvement import assess


def describe(values):
    a=np.asarray(list(values),dtype=float)
    if not len(a):return {'count':0,'mean':None,'min':None,'max':None,'median':None}
    return {'count':len(a),'mean':float(a.mean()),'min':float(a.min()),'max':float(a.max()),'median':float(np.median(a))}


def write_table(path,rows):
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=keys);writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(list,dict)) else v for k,v in row.items()})


def compare_oracle(oracle,lqr):
    selected=oracle['selected_metrics']
    return {'selected_source':oracle['source'],'optimized_safe_improvement':oracle['optimized_safe_improvement'],
            'selected_start':oracle['selected_start'],'safe_candidate_count':oracle['safe_candidate_count'],
            'selected_return':selected['returns'],'selected_discounted_return':selected['discounted_returns'],
            'selected_effort':selected['squared_torque_effort'],'selected_margin':selected['minimum_margins'],
            'delta_vs_PD':oracle['delta_return'],'delta_vs_LQR':selected['returns']-lqr['returns'],
            'LQR_safe':lqr['witness'],'selected_safe':selected['witness'],
            'genuine_optimized_gain_beyond_LQR_gt_1e-6':bool(oracle['source']=='optimized' and selected['returns']-lqr['returns']>1e-6),
            'genuine_optimized_gain_beyond_LQR_gt_0_01':bool(oracle['source']=='optimized' and selected['returns']-lqr['returns']>.01),
            'genuine_optimized_gain_beyond_LQR_gt_0_1':bool(oracle['source']=='optimized' and selected['returns']-lqr['returns']>.1)}


def _assert_arrays(a,b):
    assert a.keys()==b.keys()
    for name in a:
        assert a[name].dtype==b[name].dtype and a[name].shape==b[name].shape
        assert a[name].tobytes()==b[name].tobytes(),name


def _audit_metrics(a,k,record):
    raw=a['raw_states'][k];u=a['actions'][k,:,0]
    assert raw.shape==(201,2) and u.dtype==np.float32 and len(u)==200
    assert np.isfinite(raw).all() and np.max(np.abs(u))<=2
    assert not a['terminated'][k].any() and not a['truncated'][k,:-1].any() and a['truncated'][k,-1]
    m=np.minimum(1-np.abs(raw[:,0]),1-np.abs(raw[:,1])/3)
    assert m.tobytes()==a['margins'][k].tobytes()
    assert record['initial_raw_margin']==float(m[0]) and record['minimum_margins']==float(m.min())
    assert record['witness']==bool(np.all(m>=0)) and record['violated']==bool(np.any(m<0))
    cross=np.flatnonzero(m<0);first=None if not len(cross) else int(cross[0])
    assert record['first_crossing_step']==first
    assert record['first_crossing_seconds']==(None if first is None else first*.05)
    t=(raw[:-1,0]+np.pi)%(2*np.pi)-np.pi;v=u.astype(float)
    ideal=-(t**2+.1*raw[:-1,1]**2+.001*v**2)
    error=float(np.max(np.abs(ideal-a['rewards'][k])));assert error<1e-7
    assert record['returns']==float(a['rewards'][k].sum())
    discounted=sum(float(r)*(.99**n) for n,r in enumerate(a['rewards'][k]))
    assert abs(discounted-record['discounted_returns'])<1e-10
    assert abs(record['squared_torque_effort']-.05*float(v@v))<1e-12
    assert abs(record['rms_torque']-float(np.sqrt((v@v)/200)))<1e-12
    assert record['peak_torque']==float(np.max(np.abs(v)))
    assert record['saturation_fraction']==float(np.mean(np.abs(v)==2))
    assert abs(record['total_variation']-sum(abs(float(x)-float(y)) for x,y in zip(v[1:],v[:-1])))<1e-11
    assert record['initial_torque_jump']==abs(float(v[0]))
    assert record['terminal_theta']==float(raw[-1,0]) and record['terminal_omega']==float(raw[-1,1])
    assert record['last20_maximum_abs_theta']==float(np.max(np.abs(raw[-20:,0])))
    assert record['last20_maximum_abs_omega']==float(np.max(np.abs(raw[-20:,1])))
    assert abs(record['last20_squared_torque_effort']-.05*float(v[-20:]@v[-20:]))<1e-12
    components=record['angle_cost']+record['speed_cost']+record['torque_reward_cost']
    assert abs(components+record['returns'])<1e-6
    return error


def independent_audit(root,output,fixed,optimizations):
    started=time.perf_counter();selection=read_json(output/'selection.json')
    archive=load_npz(root/'results/expert_safety_map/expert_safety_map.npz')
    eligible=np.flatnonzero((archive['margin_map']>=0).ravel()).tolist()
    assert selection['population_ids']==eligible and len(eligible)==3331
    controllers=read_json(output/'controllers.json');byname={c['name']:c for c in controllers}
    fixed_by={(r['controller'],r['state_id']):r for r in fixed}
    assert len(fixed_by)==33310
    pairs=0;reward_error=0.;fallbacks=0;retained={name:{} for name in ('PD_8_2','LQR')}
    for controller in controllers:
        ids_seen=[]
        for folder in sorted((output/'fixed'/controller['name']).glob('chunk_*')):
            a=load_npz(folder/'original.npz');b=load_npz(folder/'replay.npz');_assert_arrays(a,b)
            ids=load_npz(folder/'grid_ids.npz')['grid_ids']
            for k,s in enumerate(ids):
                sid=int(s);ids_seen.append(sid);i,j=divmod(sid,81)
                initial=np.array([archive['theta_values'][j],archive['omega_values'][i]])
                assert a['raw_states'][k,0].tobytes()==initial.tobytes()
                r=fixed_by[controller['name'],sid];reward_error=max(reward_error,_audit_metrics(a,k,r))
                raw=a['raw_states'][k];angles=np.arctan2(np.sin(raw[:-1,0]).astype(np.float32),np.cos(raw[:-1,0]).astype(np.float32))
                speeds=raw[:-1,1].astype(np.float32)
                expected=np.clip(-controller['kp']*angles-controller['kd']*speeds,-2.,2.).astype(np.float32)
                theta=np.arctan2(np.sin(initial[0]),np.cos(initial[0]))
                expected[0]=np.clip(-controller['kp']*theta-controller['kd']*initial[1],-2.,2.)
                if expected.tobytes()!=a['actions'][k,:,0].tobytes():
                    fallbacks+=1
                    for t in range(200):
                        obs=np.array([np.cos(raw[t,0]),np.sin(raw[t,0]),raw[t,1]],dtype=np.float64 if t==0 else np.float32)
                        angle,w=np.arctan2(obs[1],obs[0]),obs[2]
                        expected[t]=np.clip(-controller['kp']*angle-controller['kd']*w,-2.,2.)
                assert expected.tobytes()==a['actions'][k,:,0].tobytes(),(controller['name'],sid)
                if controller['name'] in retained and sid in selection['optimization_ids_sorted']:
                    retained[controller['name']][sid]={key:value[[k]] for key,value in a.items()}
                pairs+=1
        assert ids_seen==eligible
    optimization_pairs=0;plan_error={'open_loop':np.zeros(2),'tracking':np.zeros(2)};residuals=[]
    for state in optimizations:
        sid=state['state_id'];initial=retained['PD_8_2'][sid]['raw_states'][0,0]
        for r in state['records']:
            if not r['candidate_available']:continue
            folder=output/f'state_{sid:04d}/optimization'/r['start']/'original'
            path=folder/r['mode'];a=load_npz(path/'rollout.npz');_assert_arrays(a,load_npz(path/'replay.npz'))
            assert a['raw_states'][0,0].tobytes()==initial.tobytes()
            reward_error=max(reward_error,_audit_metrics(a,0,r));optimization_pairs+=1
            candidate=folder/'candidate.npz'
            if not candidate.exists():candidate=folder/'last_iterate.npz'
            vector=load_npz(candidate)['decision'];planned=np.vstack([initial,vector[:400].reshape(200,2)])
            command=vector[400:].copy()
            if r['mode']=='tracking':
                delta=a['raw_states'][0,:-1]-planned[:-1]
                command-=8.*delta[:,0]+2.*delta[:,1]
            assert np.clip(command,-2,2).astype(np.float32).tobytes()==a['actions'][0,:,0].tobytes()
            plan_error[r['mode']]=np.maximum(plan_error[r['mode']],np.max(np.abs(a['raw_states'][0]-planned),axis=0))
            residuals.append(r['maximum_dynamics_residual'])
        baseline=fixed_by['PD_8_2',sid]
        for o in state['oracles']:
            candidates=[r for r in state['records'] if r.get('mode')==o['mode'] and r.get('phase')=='original'
                        and r.get('candidate_available') and r.get('witness')]
            best=min(candidates,key=lambda r:(-r['returns'],r['start_index'])) if candidates else None
            genuine=best is not None and best['returns']>baseline['returns']
            assert genuine==o['optimized_safe_improvement']
            assert o['source']==('optimized' if genuine else 'PD_fallback')
            if not genuine:assert o['delta_return']==0
            else:assert o['selected_start']==best['start']
    assert {o['state_id'] for o in optimizations}==set(selection['optimization_ids_sorted'])
    return {'passed':True,'fixed_recorded_action_replay_pairs':pairs,
            'optimization_recorded_action_replay_pairs':optimization_pairs,
            'raw_state_samples':(pairs+optimization_pairs)*201,
            'all_fixed_native_action_sequences_reconstructed':True,
            'scalar_dispatch_fallback_episodes':fallbacks,'maximum_reward_formula_difference':reward_error,
            'maximum_plan_execution_discrepancy':{k:v.tolist() for k,v in plan_error.items()},
            'NLP_residual_range':describe(residuals),'extra_simulator_steps':0,'extra_solver_executions':0,
            'wall_seconds':time.perf_counter()-started},retained


def cost_summary(output,optimizations):
    fixedcost=[read_json(p) for p in (output/'fixed').glob('*/chunk_*/costs.json')]
    new_ids={s['state_id'] for s in optimizations if not s['reused_pilot']}
    solver=[];attempts=[];sim=[]
    for sid in new_ids:
        folder=output/f'state_{sid:04d}/optimization'
        solver.extend(read_json(p) for p in folder.glob('*/original/solver_result.json'))
        attempts.extend(read_json(p) for p in folder.glob('*/original/attempt.json'))
        sim.extend(read_json(p) for p in folder.rglob('*_costs.json'))
    return {'fixed_new_steps':sum(c['new_steps'] for c in fixedcost),
            'fixed_new_resets':sum(c['new_resets'] for c in fixedcost),
            'fixed_new_wall_seconds':sum(c['new_wall_seconds'] for c in fixedcost),
            'reused_E09_PD_conditions':sum(c['reused_E09_PD_conditions'] for c in fixedcost),
            'reused_pilot_nonbaseline_conditions':sum(c['reused_pilot_conditions'] for c in fixedcost),
            'new_solver_executions':len(attempts),'new_solver_seconds':sum(c['solver_seconds'] for c in solver),
            'new_worker_process_seconds':sum(c['process_wall_seconds'] for c in attempts),
            'new_solver_iterations':sum(c['stats'].get('iter_count',0) for c in solver),
            'new_optimization_steps':sum(int(c['step_attempts']) for c in sim),
            'new_optimization_resets':sum(int(c['reset_attempts']) for c in sim),
            'new_optimization_wall_seconds':sum(float(c['wall_seconds']) for c in sim),
            'new_watchdog_timeouts':sum(c['watchdog_timeout'] for c in attempts),
            'new_technical_worker_failures':sum(c['technical_failure'] for c in attempts),
            'reused_original_pilot_solves':12,'retained_historical_pilot_repeat_solves':12,
            'scientific_original_solves':192}


def _oracle_arrays(output,row,retained):
    sid=row['state_id']
    if row['selected_source']=='PD_fallback':return retained['PD_8_2'][sid]
    return load_npz(output/f'state_{sid:04d}/optimization'/row['selected_start']/'original'/row['mode']/'rollout.npz')


def scientific_report(summary,verification,costs):
    primary=summary['fixed_assessments']['LQR'];opt=summary['optimization_aggregate']
    lines=['# Experiment 10 — Safe Policy Improvement Feasibility: full study','',
           f'Primary LQR-versus-PD result: mean return gain {primary["mean_return_difference"]:+.9f}, threshold {primary["return_threshold"]:.12g}; violations {primary["violations"]}/3331; minimum paired return difference {primary["minimum_per_state_paired_return_difference"]:+.9f}; maximum observed per-state margin loss {-primary["minimum_margin_difference"]:.12g}. Frozen practical reward criterion passed: **{primary["practical_reward_pass"]}**.', '',
           '## Protocol and provenance','',
           'All 3,331 original archived PD-safe initial states were evaluated with ten frozen controllers, identical original observation/action dtypes, torque bounds ±2, 200 transitions and all 201 raw states. PD uses the verified E09 baseline archive; accepted pilot controller evidence and 12 original pilot solves are reused. The exact 64-state optimization subset, four cost strata, PCG64 seed 101001, grid-ID tie-breaking, LQR gain, objective, constraints, three starts, random streams, solver/time limits and practical criteria are unchanged. Remaining 60 states used 180 new original solves; no new solver-repeat starts or training occurred.', '',
           'Worst-case deterioration means the minimum per-state paired return difference (required >=−0.10). A practical reward success also needs mean gain >=max(0.10, 10% of PD mean cost), zero violations, and no per-state margin loss >0.01. No negative margin is accepted. Effort and margin criteria remain separate and unchanged. All conditions, including unsafe executions and unsuccessful optimizations, remain represented.', '',
           '## Complete fixed-controller cohort','',
           '| Controller | Mean return | Worst return | Mean discounted return | Mean gain vs PD | Minimum paired gain | Mean effort | Minimum raw margin | Violations | Reward criterion |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for name,a in summary['fixed_assessments'].items():
        lines.append(f'| {name} | {a["mean_return"]:.9f} | {a["worst_return"]:.9f} | {a["mean_discounted_return"]:.9f} | {a["mean_return_difference"]:+.9f} | {a["minimum_per_state_paired_return_difference"]:+.9f} | {a["mean_squared_torque_effort"]:.9f} | {a["minimum_raw_margin"]:.9f} | {a["violations"]}/3331 | {a["practical_reward_pass"]} |')
    lines+=['', '| Controller | Mean RMS torque | Max peak torque | Mean saturation | Mean total variation | Effort reduction vs PD | Worst margin change | Effort criterion | Margin criterion |',
            '|---|---:|---:|---:|---:|---:|---:|---|---|']
    for name,a in summary['fixed_assessments'].items():
        lines.append(f'| {name} | {a["mean_RMS_torque"]:.9f} | {a["maximum_peak_torque"]:.6f} | {a["mean_saturation_fraction"]:.6f} | {a["mean_total_variation"]:.9f} | {a["effort_reduction_fraction"]:.6f} | {a["minimum_margin_difference"]:+.9f} | {a["practical_effort_pass"]} | {a["practical_margin_pass"]} |')
    lines+=['', f'Margin-eligible baseline states: {primary["margin_eligible_count"]}/3331; the identical initial state caps attainable episode minimum margin. Per-state paired differences, crossing events, effort, reward components and terminal diagnostics are retained in fixed_metrics.csv. Total variation excludes the separately recorded initial-action jump; squared torque is an effort proxy, not physical energy.', '',
            '## Prespecified 64-state optimization sample','',
            f'Original scientific solver statuses (one status per solve): {summary["original_solver_status_counts"]}. Solver success is distinct from a safe executed trajectory. Open-loop and tracking are independently executed/replayed with actual float32 torques. All 192 original scientific starts remain in the record; the accepted 12 pilot repeat solves remain separate historical technical evidence.', '',
            '| Oracle mode | Safe original executions | Genuine optimized improvements over PD | PD fallbacks | Mean gain vs PD | Minimum gain vs PD | Mean gain vs LQR | Minimum gain vs LQR | Optimized gains vs LQR >1e−6 / >0.01 / >0.1 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for mode,a in opt.items():
        lines.append(f'| {mode} | {a["safe_original_executions"]}/{a["finite_original_executions"]} | {a["genuine_optimized_PD_improvements"]}/64 | {a["PD_fallbacks"]}/64 | {a["delta_vs_PD"]["mean"]:+.9f} | {a["delta_vs_PD"]["min"]:+.9f} | {a["delta_vs_LQR"]["mean"]:+.9f} | {a["delta_vs_LQR"]["min"]:+.9f} | {a["beyond_LQR_gt_1e-6"]} / {a["beyond_LQR_gt_0_01"]} / {a["beyond_LQR_gt_0_1"]} |')
    lines+=['', 'Oracle outcomes use the best safely executed original candidate only when it improves PD return; otherwise the source is explicitly PD fallback, which is not an optimization success. These are state-specific outcome-informed comparisons, not results for a deployable universal controller. Comparisons against LQR use the same 64 states and preserve LQR safety flags. Missing finite candidates are unresolved numerical cases, never evidence of optimality. All-start results and oracles are in optimization_metrics.csv and oracle_metrics.csv.', '',
            f'Pilot-only diagnostic before scaling found tracking-minus-LQR gains between −1.02116e−9 and +1.34021e−6, with maximum torque differences up to 0.0130317 N·m. It did not motivate any method or threshold change. Full-sample return and executed-torque difference distributions are recorded in the oracle table and summary.json.', '',
            '## Verification and computational costs','',
            f'Verification passed: {verification["passed"]}; protected pre-existing files: {verification["protected_files"]}. Independent fixed replay pairs: {verification["independent_audit"]["fixed_recorded_action_replay_pairs"]}; optimization replay pairs: {verification["independent_audit"]["optimization_recorded_action_replay_pairs"]}; raw samples reconstructed: {verification["independent_audit"]["raw_state_samples"]}. Rewards, discounted sums, effort, torque sequences, raw margins and crossing events were independently reconstructed. Technical tests: {verification["technical_tests"]}. All accepted pilot evidence, full scientific inputs and prior results passed preservation/source hashes.', '',
            f'New execution costs: {costs}. These exclude historical reused executions, saved-array audit and file handling. Timings are descriptive, not new independent observations. Numerical execution diagnostics: {verification["independent_audit"]["maximum_plan_execution_discrepancy"]}; NLP residual range: {verification["independent_audit"]["NLP_residual_range"]}. Any timeouts, technical failures, missing candidates and failed checks are retained.', '',
            '## Scientific interpretation and limitations','',
            'Judge LQR against every frozen practical criterion rather than mean return alone. Positive state-specific optimization gains do not imply deployable controller improvement, and PD fallback must not be counted as optimized success. Local solver convergence and small residuals cannot prove global optimality; open-loop failures can reflect numerical sensitivity. The selected 64 points are a prespecified stratified sample for diagnostics, not a proof over the continuous domain or a globally optimal reference.', '',
            'The 3,331-point cohort has empirical PD safety, not a certified continuous viable region. All claims concern this fixed nominal simulator, exact initial states and 200-step sampled horizon. No formal robustness, continuous-time, model-error, hardware or infinite-horizon guarantee is established. Reward, actuator effort and safety margins can trade off. Deterministic repeats are verification, not independent statistical samples. No favorable seed selection, tuning, reward change, new controller, RL training, IQL work, commit or push occurred. Any future learning or acquisition design requires review of these results first.', '']
    return '\n'.join(lines)


def finalize(root,output,pilot,scientific,control,fixed,optimizations,before,hashes):
    selection=read_json(output/'selection.json');ids=selection['population_ids']
    by={(r['controller'],r['state_id']):r for r in fixed};baselines=[by['PD_8_2',sid] for sid in ids]
    names=[c['name'] for c in read_json(output/'controllers.json')];assessments={}
    for name in names:
        rows=[by[name,sid] for sid in ids];a=assess(rows,baselines,scientific)
        a.update(mean_RMS_torque=float(np.mean([r['rms_torque'] for r in rows])),
                 maximum_peak_torque=max(r['peak_torque'] for r in rows),
                 mean_saturation_fraction=float(np.mean([r['saturation_fraction'] for r in rows])),
                 mean_total_variation=float(np.mean([r['total_variation'] for r in rows])),
                 maximum_terminal_abs_theta=max(abs(r['terminal_theta']) for r in rows),
                 maximum_terminal_abs_omega=max(abs(r['terminal_omega']) for r in rows),
                 violation_ids=[r['state_id'] for r in rows if r['violated']],
                 worst_paired_return_state=min(rows,key=lambda r:r['delta_return'])['state_id'],
                 worst_margin_change_state=min(rows,key=lambda r:r['delta_minimum_margin'])['state_id'],
                 paired_return_distribution=describe(r['delta_return'] for r in rows))
        assessments[name]=a
    assert assessments['LQR']['return_threshold']==read_json(output/'practical_thresholds_frozen.json')['full_benchmark_return_threshold']
    audit,retained=independent_audit(root,output,fixed,optimizations)
    write_json(output/'independent_audit.json',audit)
    oracles=[]
    for state in optimizations:
        sid=state['state_id'];lqr=by['LQR',sid]
        for oracle in state['oracles']:
            row=dict(state_id=sid,mode=oracle['mode'],**compare_oracle(oracle,lqr))
            a=_oracle_arrays(output,row,retained);b=retained['LQR'][sid]
            du=a['actions'][0,:,0].astype(float)-b['actions'][0,:,0].astype(float)
            row.update(max_abs_torque_difference_vs_LQR=float(np.max(np.abs(du))),
                       RMS_torque_difference_vs_LQR=float(np.sqrt(np.mean(du**2))),
                       different_executed_actions_vs_LQR=int(np.sum(a['actions']!=b['actions'])))
            oracles.append(row)
    optrecords=[r for s in optimizations for r in s['records']]
    optaggregate={}
    for mode in scientific['candidate_modes']:
        rows=[r for r in oracles if r['mode']==mode]
        actual=[r for r in optrecords if r.get('mode')==mode and r.get('candidate_available')]
        optaggregate[mode]={'safe_original_executions':sum(r['witness'] for r in actual),
              'finite_original_executions':len(actual),'genuine_optimized_PD_improvements':sum(r['optimized_safe_improvement'] for r in rows),
              'PD_fallbacks':sum(r['selected_source']=='PD_fallback' for r in rows),
              'delta_vs_PD':describe(r['delta_vs_PD'] for r in rows),
              'delta_vs_LQR':describe(r['delta_vs_LQR'] for r in rows),
              'selected_returns':describe(r['selected_return'] for r in rows),
              'selected_discounted_returns':describe(r['selected_discounted_return'] for r in rows),
              'selected_effort':describe(r['selected_effort'] for r in rows),
              'selected_margins':describe(r['selected_margin'] for r in rows),
              'beyond_LQR_gt_1e-6':sum(r['genuine_optimized_gain_beyond_LQR_gt_1e-6'] for r in rows),
              'beyond_LQR_gt_0_01':sum(r['genuine_optimized_gain_beyond_LQR_gt_0_01'] for r in rows),
              'beyond_LQR_gt_0_1':sum(r['genuine_optimized_gain_beyond_LQR_gt_0_1'] for r in rows),
              'torque_RMS_difference_vs_LQR':describe(r['RMS_torque_difference_vs_LQR'] for r in rows),
              'maximum_torque_difference_vs_LQR':describe(r['max_abs_torque_difference_vs_LQR'] for r in rows)}
    statuses=[];missing=[];feasible_unsafe={'open_loop':0,'tracking':0}
    for state in optimizations:
        for name in scientific['starts']:
            folder=output/f'state_{state["state_id"]:04d}/optimization'/name/'original'
            path=folder/'solver_result.json'
            statuses.append(read_json(path)['stats']['return_status'] if path.exists() else 'missing_solver_result')
        missing.extend(r for r in state['records'] if not r['candidate_available'])
    for r in optrecords:
        if r.get('candidate_available') and r['maximum_dynamics_residual']<=1e-8 and r['maximum_bound_violation']<=1e-8 and r['predicted_minimum_raw_margin']>=0 and not r['witness']:
            feasible_unsafe[r['mode']]+=1
    summary={'population_count':3331,'fixed_condition_count':len(fixed),'optimization_state_count':64,
             'fixed_assessments':assessments,'optimization_aggregate':optaggregate,
             'original_solver_status_counts':dict(Counter(statuses)),
             'missing_candidate_records':missing,'nominally_feasible_but_executed_unsafe':feasible_unsafe,
             'optimization_sample_interpretation':'Prespecified 64-state stratified diagnostic sample, not global optimality proof',
             'controller_and_method_changes':False}
    costs=cost_summary(output,optimizations)
    assert costs['new_solver_executions']==180 and costs['fixed_new_steps']==11973600
    assert costs['reused_E09_PD_conditions']==3331 and costs['reused_pilot_nonbaseline_conditions']==45
    _assert_hashes(root,before['sha256']);_assert_hashes(root,hashes)
    _assert_hashes(pilot,read_json(pilot/'completion.json')['artifacts_sha256'])
    verification={'passed':True,'protected_files':len(before['sha256']),
                  'technical_tests':read_json(output/'technical_tests.json'),'independent_audit':audit,
                  'accepted_pilot_preserved_bitwise':True,'selection_and_controllers_byte_identical_to_pilot':True,
                  'new_optimization_states':60,'reused_original_pilot_solves':12,'scientific_original_solves':192,
                  'new_technical_worker_failures':costs['new_technical_worker_failures'],
                  'missing_candidates':len(missing),'failed_checks':[]}
    if costs['new_technical_worker_failures']:
        verification['passed']=False;verification['failed_checks'].append('technical_worker_failure')
    write_json(output/'summary.json',summary);write_json(output/'verification.json',verification)
    write_json(output/'computational_costs.json',costs)
    write_table(output/'fixed_metrics.csv',fixed);write_table(output/'optimization_metrics.csv',optrecords)
    write_table(output/'oracle_metrics.csv',oracles)
    text=scientific_report(summary,verification,costs);(output/'scientific_report.md').write_text(text)
    doc=root/'docs/experiment_10_report.md'
    if doc.exists():assert doc.read_text()==text
    else:doc.write_text(text)
    assert verification['passed'],'Technical failure retained; inspect full records'
    artifacts={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*'))
               if p.is_file() and p.name!='.lock' and p!=output/'completion.json'}
    write_json(output/'completion.json',{'passed':True,'population_count':3331,'optimization_state_count':64,
               'artifacts_sha256':artifacts,'report_sha256':sha256(doc)})
    print('Full E10 study complete and verified. No tuning, training, commit or push.',flush=True)
