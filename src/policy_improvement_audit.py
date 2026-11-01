"""Independent saved-array E10 verification and concise pilot reporting."""

from collections import Counter
import json
from pathlib import Path
import time

import casadi as ca
import numpy as np

from src.horizon_experiment import _assert_hashes,load_npz,read_json,sha256,write_json
from src.recoverability import symbolic_dynamics


def audit(root,output,config,summary):
    started=time.perf_counter();ids=[3280,2956,5383,2568,6194]
    assert summary['pilot_ids']==ids and not summary['full_benchmark_executed']
    assert not summary['remaining_60_optimization_states_executed']
    selection=read_json(output/'selection.json')
    rows=read_json(root/config['baseline_archive']/'baseline_summary.json')
    ordered=sorted([r for r in rows if not r['identity']['archived_unsafe'] and r['state_id']!=3280],
                   key=lambda r:(-r['metrics']['returns'],r['state_id']))
    rng=np.random.Generator(np.random.PCG64(101001));chosen_ids=[];pilot=[3280]
    for group,indices in enumerate(np.array_split(np.arange(len(ordered)),4),1):
        sample=sorted(int(v) for v in rng.choice(indices,16,replace=False))
        selected=[ordered[i] for i in sample];chosen_ids.extend(r['state_id'] for r in selected)
        expected=selection['strata'][group-1]
        assert expected['population_count']==len(indices)
        assert [r['grid_id'] for r in expected['selected']]==[r['state_id'] for r in selected]
        pilot.append(min(selected,key=lambda r:(-r['metrics']['returns'],r['state_id']))['state_id'])
    assert sorted(chosen_ids)==selection['optimization_ids_sorted'] and pilot==ids
    assert len(set(chosen_ids))==64 and len(selection['remaining_optimization_ids'])==60
    assert read_json(output/'selection_frozen.json')['selection_sha256']==sha256(output/'selection.json')
    lqr=summary['lqr'];a=np.array(lqr['A']);b=np.array(lqr['B']);p=np.array(lqr['P'])
    q=np.diag([1.,.1]);r=np.array([[.001]]);k=np.array(lqr['K'])
    k2=np.linalg.inv(r+b.T@p@b)@(b.T@p@a)
    residual=q+a.T@p@a-(a.T@p@b)@k-p
    assert np.max(np.abs(k-k2))<1e-12 and np.max(np.abs(residual))<1e-10
    assert np.max(np.abs(np.linalg.eigvals(a-b@k)))<1
    x=ca.SX.sym('x',2);u=ca.SX.sym('u')
    y=symbolic_dynamics()(x,u)
    jac=ca.Function('E10_independent_linearization',[x,u],[ca.jacobian(y,x),ca.jacobian(y,u)])
    actual_a,actual_b=jac([0,0],0)
    assert np.max(np.abs(np.array(actual_a)-a))<1e-15
    assert np.max(np.abs(np.array(actual_b)-b))<1e-15
    controllers=read_json(output/'controllers.json')
    assert len(controllers)==10
    assert {(c['kp'],c['kd']) for c in controllers if c['kind']=='PD'}=={(kp,kd) for kp in (6.,8.,12.) for kd in (1.,2.,4.)}
    archived=load_npz(root/config['benchmark'])
    comparisons=0;policy_repeats=0;reward_error=0.;decomposition_error=0.
    records={(r['state_id'],r['controller']):r for r in summary['fixed_records']}
    optrecords={(r['state_id'],r['start'],r['phase'],r.get('mode')):r
                for o in summary['optimizations'] for r in o['records'] if r['candidate_available']}
    rollouts=list(output.glob('state_*/**/rollout.npz'))
    for path in rollouts:
        arrays=load_npz(path);repeat=load_npz(path.parent/'replay.npz')
        for name in arrays:
            assert arrays[name].dtype==repeat[name].dtype and arrays[name].shape==repeat[name].shape
            assert arrays[name].tobytes()==repeat[name].tobytes(),(path,name)
        comparisons+=1
        sid=int(path.relative_to(output).parts[0][6:]);i,j=divmod(sid,81)
        initial=np.array([archived['theta_values'][j],archived['omega_values'][i]])
        raw=arrays['raw_states'][0];torque=arrays['actions'][0,:,0]
        assert raw.shape==(201,2) and torque.dtype==np.float32 and len(torque)==200
        assert raw[0].tobytes()==initial.tobytes()
        assert np.isfinite(raw).all() and np.max(np.abs(torque))<=2
        margin=np.minimum(1-np.abs(raw[:,0]),1-np.abs(raw[:,1])/3)
        assert margin.tobytes()==arrays['margins'][0].tobytes()
        assert not arrays['terminated'].any() and not arrays['truncated'][0,:-1].any() and arrays['truncated'][0,-1]
        if path.parent.parent.name=='controllers':
            name=path.parent.name;record=records[sid,name]
            control=next(c for c in controllers if c['name']==name)
            policy=load_npz(path.parent/'policy_repeat.npz')
            for key in arrays:assert arrays[key].tobytes()==policy[key].tobytes()
            policy_repeats+=1
            for t in range(200):
                obs=np.array([np.cos(raw[t,0]),np.sin(raw[t,0]),raw[t,1]],dtype=np.float64 if t==0 else np.float32)
                theta,omega=np.arctan2(obs[1],obs[0]),obs[2]
                expected=np.float32(np.clip(-control['kp']*theta-control['kd']*omega,-2.,2.))
                assert expected.tobytes()==torque[t].tobytes(),(sid,name,t)
        else:
            # .../optimization/start/phase/mode/rollout.npz
            mode=path.parent.name;phase=path.parent.parent.name;start=path.parent.parent.parent.name
            record=optrecords[sid,start,phase,mode]
            vector=load_npz(path.parent.parent/'candidate.npz')['decision'] if (path.parent.parent/'candidate.npz').exists() else load_npz(path.parent.parent/'last_iterate.npz')['decision']
            planned=np.vstack([initial,vector[:400].reshape(200,2)])
            command=vector[400:].copy()
            if mode=='tracking':
                error=raw[:-1]-planned[:-1]
                command-=8.*error[:,0]+2.*error[:,1]
            assert np.clip(command,-2,2).astype(np.float32).tobytes()==torque.tobytes()
        numeric=torque.astype(np.float64)
        angle=(raw[:-1,0]+np.pi)%(2*np.pi)-np.pi
        ideal=-(angle**2+.1*raw[:-1,1]**2+.001*numeric**2)
        reward_error=max(reward_error,float(np.max(np.abs(ideal-arrays['rewards'][0]))))
        assert reward_error<1e-7
        assert record['returns']==float(arrays['rewards'].sum())
        discounted=sum(float(v)*(.99**t) for t,v in enumerate(arrays['rewards'][0]))
        assert abs(record['discounted_returns']-discounted)<1e-10
        assert record['witness']==bool(np.all(margin>=0))
        assert record['violated']==bool(np.any(margin<0))
        assert record['minimum_margins']==float(margin.min())
        cross=np.flatnonzero(margin<0);step=None if len(cross)==0 else int(cross[0])
        assert record['first_crossing_step']==step
        assert record['first_crossing_seconds']==(None if step is None else step*.05)
        assert abs(record['squared_torque_effort']-.05*float(np.dot(numeric,numeric)))<1e-12
        assert abs(record['rms_torque']-float(np.sqrt(np.dot(numeric,numeric)/200)))<1e-12
        assert record['peak_torque']==float(np.max(np.abs(numeric)))
        assert record['saturation_fraction']==float(np.mean(np.abs(numeric)==2))
        assert abs(record['total_variation']-sum(abs(float(v)-float(w)) for v,w in zip(numeric[1:],numeric[:-1])))<1e-12
        assert record['initial_torque_jump']==abs(float(numeric[0]))
        assert record['terminal_theta']==float(raw[-1,0]) and record['terminal_omega']==float(raw[-1,1])
        assert record['last20_maximum_abs_theta']==float(np.max(np.abs(raw[-20:,0])))
        assert record['last20_maximum_abs_omega']==float(np.max(np.abs(raw[-20:,1])))
        components=record['angle_cost']+record['speed_cost']+record['torque_reward_cost']
        decomposition_error=max(decomposition_error,abs(components+record['returns']))
        assert decomposition_error<1e-6
    assert policy_repeats==50
    assert len(summary['fixed_records'])==50
    assert len(summary['optimizations'])==4
    repetitions=[r for o in summary['optimizations'] for r in o['repetitions']]
    assert len(repetitions)==12 and all(r['passed'] for r in repetitions)
    for controller in controllers:
        selected=[records[sid,controller['name']] for sid in ids]
        baselines=[summary['baselines'][str(sid)] for sid in ids]
        deltas=[r['returns']-b['returns'] for r,b in zip(selected,baselines)]
        a=summary['fixed_assessments'][controller['name']]
        assert a['minimum_per_state_paired_return_difference']==min(deltas)
        assert a['return_threshold']==max(.1,.1*-float(np.mean([r['returns'] for r in baselines])))
    fallback=0;genuine=0
    for o in summary['optimizations']:
        baseline=summary['baselines'][str(o['state_id'])]
        for oracle in o['oracles']:
            candidates=[r for r in o['records'] if r.get('phase')=='original' and r.get('mode')==oracle['mode']
                        and r.get('candidate_available') and r.get('witness')]
            best=min(candidates,key=lambda r:(-r['returns'],r['start_index'])) if candidates else None
            improved=best is not None and best['returns']>baseline['returns']
            assert oracle['optimized_safe_improvement']==improved
            assert oracle['source']==('optimized' if improved else 'PD_fallback')
            if improved: genuine+=1;assert oracle['selected_start']==best['start']
            else: fallback+=1;assert oracle['delta_return']==0
    assert sorted(int(p.name[6:]) for p in output.glob('state_*') if p.is_dir())==sorted(ids)
    attempted=[p for p in output.glob('state_*/optimization/*/*/attempt.json')]
    assert len(attempted)==24
    assert {int(p.relative_to(output).parts[0][6:]) for p in attempted}==set(ids[1:])
    return {'passed':True,'all_64_selection_identities_verified':True,'pilot_ids_verified':ids,
            'policy_repeats_bitwise':policy_repeats,'recorded_action_replay_pairs':comparisons,
            'all_finite_original_and_repeat_candidates_executed':True,'solver_execution_count':len(attempted),
            'paired_solver_reproducibility_checks':len(repetitions),'optimized_oracle_improvements':genuine,
            'PD_fallback_oracles':fallback,'maximum_reward_formula_difference':reward_error,
            'maximum_reward_decomposition_difference':decomposition_error,'all_201_raw_states_checked':True,
            'extra_simulator_steps':0,'extra_solver_attempts':0,'wall_seconds':time.perf_counter()-started}


def costs(output):
    c=[r for p in output.rglob('*_costs.json') if 'step_attempts' in (r:=read_json(p))]
    attempts=[read_json(p) for p in output.glob('state_*/optimization/*/*/attempt.json')]
    solvers=[read_json(p) for p in output.glob('state_*/optimization/*/*/solver_result.json')]
    return {'simulator_steps':sum(int(r['step_attempts']) for r in c),
            'simulator_resets':sum(int(r['reset_attempts']) for r in c),
            'simulator_wall_seconds':sum(float(r['wall_seconds']) for r in c),
            'solver_executions':len(attempts),'solver_seconds':sum(r['solver_seconds'] for r in solvers),
            'worker_process_seconds':sum(r['process_wall_seconds'] for r in attempts),
            'solver_iterations':sum(r['stats'].get('iter_count',0) for r in solvers),
            'watchdog_timeouts':sum(r['watchdog_timeout'] for r in attempts),
            'technical_worker_failures':sum(r['technical_failure'] for r in attempts)}


def report(summary,verification,execution_costs,selection,config):
    lqr=summary['lqr']
    lines=['# Experiment 10 — Safe Policy Improvement Feasibility: technical pilot','',
           'Only grid IDs 3280, 2956, 5383, 2568 and 6194 were evaluated. The complete 64-state optimization subset was frozen from archived PD costs using PCG64 seed 101001, four cost strata and grid-ID tie-breaking before new controller evaluations. The remaining 60 optimization states and the full 3,331-state controller benchmark were not run. No RL or data collection for training occurred.', '',
           '## LQR and fixed protocol','',
           f'Clipped LQR gain K = {lqr["K"][0]}. Maximum absolute discrete Riccati residual {lqr["max_absolute_DARE_residual"]:.6e}; spectral radius {lqr["spectral_radius"]:.12g}; poles {lqr["closed_loop_poles"]}. Repeated solve bitwise: {lqr["repeated_DARE_solution_bitwise"]}; independent Riccati iteration discrepancy {lqr["independent_Riccati_max_difference"]:.6e}. This is a local unconstrained linear design, not nonlinear optimality or safety certification.', '',
           'All controllers use identical exact archived initial states, original native observation scalar conventions and float32 executed torques. Horizon 200 steps with all 201 raw states checked; |u|<=2, |theta|<=1, |omega|<=3. Fixed PD gains are Kp in {6,8,12}, Kd in {1,2,4}; LQR uses the approved linearization, Q=diag(1,0.1), R=0.001 without tuning.', '',
           '## Fixed-controller results across five pilot states','',
           '| Controller | Mean return | Mean paired return gain | Minimum paired return difference | Worst return | Mean squared-torque effort | Minimum raw margin | Violations | Practical reward criterion |',
           '|---|---:|---:|---:|---:|---:|---:|---:|---|']
    for name,a in summary['fixed_assessments'].items():
        lines.append(f'| {name} | {a["mean_return"]:.9f} | {a["mean_return_difference"]:+.9f} | {a["minimum_per_state_paired_return_difference"]:+.9f} | {a["worst_return"]:.9f} | {a["mean_squared_torque_effort"]:.9f} | {a["minimum_raw_margin"]:.9f} | {a["violations"]}/5 | {a["practical_reward_pass"]} |')
    threshold=summary['fixed_assessments']['LQR']['return_threshold']
    lines+=['',f'The pilot mean-return threshold is max(0.10, 10% of pilot PD mean cost) = {threshold:.9f}. Worst-case deterioration is the **minimum per-state paired return difference**, with threshold -0.10; it is not the difference between independently minimized episode returns. The safety/margin criterion requires zero violations and no per-state margin loss >0.01. Effort success requires >=10% mean reduction with mean return loss <=0.01 and the same worst-paired-return and safety conditions. Every raw margin must be nonnegative; the comparison allowance is never a safety tolerance.', '',
            'Full per-state undiscounted and discounted returns, paired differences, reward components, squared-torque effort, RMS/peak torque, saturation, total variation, raw margins, violations and terminal/last-20-step diagnostics are in fixed_metrics.csv. Effort is dt*sum(u^2), a torque-effort proxy rather than physical energy. Total variation excludes the initial-to-first-action jump, which is retained separately.', '',
            f'The corresponding full-benchmark mean-return threshold, frozen from existing E09 data before evaluation, is {summary["practical_thresholds"]["full_benchmark_return_threshold"]:.12g}; it is not a full-benchmark pilot result. Margin-eligible pilot states: {summary["fixed_assessments"]["LQR"]["margin_eligible_count"]}; a margin-improvement criterion with no eligible states is not applicable. Controllers meeting the pilot effort criterion: {[name for name,a in summary["fixed_assessments"].items() if a["practical_effort_pass"]]}.', '',
            '| Controller | Mean discounted return | Mean RMS torque | Max peak torque | Mean saturation | Mean total variation | Max terminal abs(theta) | Max terminal abs(omega) |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in summary['fixed_assessments']:
        records=[r for r in summary['fixed_records'] if r['controller']==name]
        lines.append(f'| {name} | {np.mean([r["discounted_returns"] for r in records]):.9f} | '
                     f'{np.mean([r["rms_torque"] for r in records]):.9f} | {max(r["peak_torque"] for r in records):.9f} | '
                     f'{np.mean([r["saturation_fraction"] for r in records]):.6f} | {np.mean([r["total_variation"] for r in records]):.9f} | '
                     f'{max(abs(r["terminal_theta"]) for r in records):.6e} | {max(abs(r["terminal_omega"]) for r in records):.6e} |')
    lines+=['',
            '## Optimization and safe-oracle results','',
            'The four non-equilibrium pilot states each use PD, zero and seeded 20-step random-block starts (SeedSequence([101004, grid_id])). Every start was solved twice with unchanged Experiment 07 equations, objective, constraints, IPOPT options and 30-second solver/60-second watchdog limits. Every finite candidate from both original and repeated solves was independently executed and replayed under both open-loop and existing tracking modes. Original solves alone enter the oracle; repeat solves are reproducibility checks. Solver feasibility remains separate from execution safety.', '',
            '| Grid | Mode | Safe original candidates | Selected source | Selected start | Paired return gain | Gain >0.01 | Gain >0.1 |',
            '|---:|---|---:|---|---|---:|---|---|']
    for o in summary['optimizations']:
        for r in o['oracles']:
            lines.append(f'| {o["state_id"]} | {r["mode"]} | {r["safe_candidate_count"]}/3 | {r["source"]} | {r["selected_start"]} | {r["delta_return"]:+.9f} | {r["gain_above_0.01"]} | {r["gain_above_0.1"]} |')
    original=[r for o in summary['optimizations'] for r in o['records'] if r.get('phase')=='original' and r['candidate_available']]
    lines+=['',f'Original candidate solver statuses by execution mode: {dict(Counter(r["solver_status"] for r in original))}. Safe executions: open-loop {sum(r["witness"] for r in original if r["mode"]=="open_loop")}/12; tracking {sum(r["witness"] for r in original if r["mode"]=="tracking")}/12. All unsuccessful and unsafe records remain included in optimization_metrics.csv. A PD fallback is explicitly not an optimized improvement, even when PD itself is safe. Best-of-start results are state-specific oracle comparisons, not performance of one deployable controller.', '',
            '## Verification, costs and scientific scope','',
            f'Verification passed: {verification["passed"]}. Technical tests: {verification["technical_tests"]}. Independent audit: {verification["independent_audit"]}. Protected pre-existing files unchanged: {verification["protected_files"]}. Failed checks: {verification["failed_checks"]}.', '',
            f'Executed costs: {execution_costs}. Unit tests make no simulator/solver calls; the independent audit uses saved arrays only. Numerical reward decomposition permits float32 arithmetic-order discrepancies without relaxing any safety threshold.', '',
            'The five states are technical pilot cases selected from baseline cost strata, not an independent estimate of full-grid improvement. Practical criteria are displayed diagnostically and do not authorize full-benchmark claims or tuning. The PD-safe cohort has empirical finite-horizon simulator evidence, not a certified continuous viable region. Clipped LQR and local optimization cannot establish global optimality or its absence. Small NLP residuals do not imply safe float32 open-loop execution. No continuous-time, robustness, physical-system or infinite-horizon safety guarantee is claimed.', '',
            'All prior experiments, datasets, Research Decision Log and unfinished IQL files are preserved. The pilot does not establish that diagnostic trajectories are suitable offline-RL training data. No remaining-state evaluations, learning, commit or push was performed.', '',
            'Reproduce with requirements_experiment_10.txt and the exact E01 archive plus E09 baseline archive. First run experiments/10_safe_policy_improvement.py --freeze-subset; then --pilot --resume. Existing completed output can be checked with --verify without regeneration. Scientific outputs remain under results/safe_policy_improvement/pilot/.', '']
    return '\n'.join(lines)


def finalize(root,output,config,summary,before,hashes):
    checked=audit(root,output,config,summary);write_json(output/'independent_audit.json',checked)
    execution_costs=costs(output);assert execution_costs['solver_executions']==24
    write_json(output/'execution_costs.json',execution_costs)
    _assert_hashes(root,before['sha256']);_assert_hashes(root,hashes)
    failed=[]
    if not all(o['passed'] for o in summary['optimizations']):failed.append('solver_reproducibility')
    if execution_costs['technical_worker_failures']:failed.append('technical_worker_failure')
    verification={'passed':not failed,'failed_checks':failed,'technical_tests':read_json(output/'technical_tests.json'),
                  'independent_audit':checked,'protected_files':len(before['sha256']),
                  'full_benchmark_executed':False,'remaining_60_executed':False,
                  'baseline_PD_E09_parity_cases':5,'subset_frozen_before_evaluation':True}
    write_json(output/'verification.json',verification)
    text=report(summary,verification,execution_costs,read_json(output/'selection.json'),config)
    (output/'scientific_report.md').write_text(text)
    doc=root/'docs/experiment_10_pilot_report.md'
    if doc.exists():assert doc.read_text()==text,'Existing reviewed report differs; generated report retained separately'
    else:doc.write_text(text)
    assert verification['passed'],'Technical verification failed; results preserved'
    artifact_hashes={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*'))
                     if p.is_file() and p.name!='.lock' and p!=output/'completion.json'}
    write_json(output/'completion.json',{'passed':True,'pilot_ids':summary['pilot_ids'],
               'artifacts_sha256':artifact_hashes,'report_sha256':sha256(doc)})
    print('E10 five-case pilot complete and verified; full benchmark and remaining 60 states not run.',flush=True)
