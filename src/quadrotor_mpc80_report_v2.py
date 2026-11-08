"""Saved-record scientific audit/report for E12_MPC80_FEAS_v2.

No environments or controllers are constructed, no solver/physics calls occur,
and experimental records are never overwritten. Output belongs in a new audit
directory. Original v1 and original E12 failed gates stay separate.
"""
from pathlib import Path
import json
import numpy as np
from src import quadrotor_pilot as p
from src import quadrotor_mpc80_execution_v2 as execution
from src import quadrotor_mpc80_feasibility_v1 as ledger_api
from src.quadrotor_verification import audit_record
from src.quadrotor_finalization import physics_reconstruction
from src.quadrotor_kkt_integrity_v2 import nlp_readout,canonical_kkt,require_kkt,matching_program_values
from src.quadrotor_h2_verification_v2 import timing_stats


def read(a):return p.read(a)


def markdown(result,resource):
    """Concise report; partial outcomes never become survivor-only cohort means."""
    passed=result['new_expert_feasibility_gate_passed']
    lines=['# E12_MPC80_FEAS_v2 — MPC80-only expert feasibility', '',
        '**'+('Scoped expert-feasibility gate PASSED.' if passed else 'Campaign STOPPED; scoped expert-feasibility gate did not pass.')+'**',
        f"Completion: {result['complete_initial_state_pairs']}/16 initial-state pairs; "
        f"{result['complete_scientific_rollouts']}/32 complete scientific rollouts; "
        f"{result['qualified_scientific_executions']}/32 fully verified executions.", '',
        'V1 remains FAILED. The original FAILED overall E12 qualification and original',
        'FAILED 0.001 symbolic/PyBullet agreement criterion are unchanged. This study',
        'qualifies only the frozen MPC80 expert scope; it is not a new four-controller benchmark.', '',
        '## Frozen setup and verification', '',
        'The original sixteen PCG64-120001 initial states and exact binary64 values/IDs',
        'were loaded directly. Each state used original/repeat, then native uninstrumented',
        'and independent recorded-motor replay, in the frozen order. All native controller,',
        'model, URDF, Q/R, bounds, IPOPT, horizon-80, warm-start, reference, seed-1337,',
        'six-second, 50-Hz control and 1,000-Hz physics definitions were preserved.', '',
        'The sealed v2 observer records current `_physics(rpm)` arguments and actual',
        'submitted force/torque API arguments; it does not use the stale command cache',
        'inside physics callbacks. Grouped thrust/effort are derived from these commands,',
        'not measured hardware outputs. Comparisons ignore only transient world/body IDs,',
        'not motor commands, forces, frames, application points or control/physics indices.', '',
        '## Per-state original-run performance', '',
        '| ID | Qualified original/repeat | Return | Discounted return | Position RMSE, m | Absolute effort, N²s | Hover-relative effort, N²s | Min normalized margin |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for state in range(16):
        rows=[r for r in result['records'] if r['state_id']==state]
        original=next(r for r in rows if r['phase']=='original');m=original.get('metrics')
        status='/'.join('pass' if r['qualified'] else r['status'] for r in rows)
        values=[f'{m[k]:.9f}' for k in ['return','discounted_return','position_tracking_rmse','absolute_effort','hover_relative_effort','minimum_normalized_raw_margin']] if m else ['N/A']*6
        lines.append('| '+str(state)+' | '+status+' | '+' | '.join(values)+' |')
    s=result['full_cohort_original_statistics']
    lines+=['','## Full-cohort statistics','']
    if s is None:
        lines+=['Full-cohort mean/worst return, cost, tracking error, terminal behavior,',
                'effort and violation incidence are unavailable because the frozen full',
                'execution/replay requirements did not complete. Per-state available values',
                'above are partial evidence, not survivor-only full-cohort estimates.']
    else:
        lines+=['Statistics below use sixteen originals, not thirty-two independent samples.',
                'Returns are negative unscaled native cumulative quadratic cost; tracking',
                'RMSE uses the native control-boundary reference. Raw safety uses every',
                'physics state including initial and terminal samples.','',
                '| Metric | Mean over 16 | Worst over 16 |','|---|---:|---:|']
        for name,label,worst in [('return','Return','minimum'),('cost','Cost','maximum'),
            ('discounted_return','Discounted return, gamma=0.99','minimum'),
            ('position_tracking_rmse','Position RMSE, m','maximum'),('absolute_effort','Absolute squared-thrust effort, N²s','maximum'),
            ('hover_relative_effort','Hover-relative squared-thrust effort, N²s','maximum'),('total_variation','Thrust total variation, N','maximum')]:
            lines.append(f"| {label} | {s[name]['mean']:.9f} | {s[name][worst]:.9f} |")
        lines+=['',f"Mean near-bound saturation fractions by channel: {s['mean_saturation_fraction_by_channel']}.",
            f"Mean absolute terminal error `(x,vx,z,vz,theta,omega)`: {s['mean_absolute_terminal_error_by_state']}.",
            f"Worst absolute terminal error by component: {s['worst_absolute_terminal_error_by_state']}.",
            '', '| Component | Minimum lower raw margin | Minimum upper raw margin | Minimum one-step reserve |',
            '|---|---:|---:|---:|']
        for j,label in enumerate(['x, m','vx, m/s','z, m','vz, m/s','theta, rad','omega, rad/s']):
            lines.append(f"| {label} | {s['minimum_raw_lower_margins'][j]:.12g} | {s['minimum_raw_upper_margins'][j]:.12g} | {s['minimum_reserves'][j]:.12g} |")
        lines += ['',f"Minimum normalized raw margin: {s['minimum_normalized_raw_margin']:.12g}.",
            f"Minimum derived input margins, N: {s['minimum_input_margins']}.",
            'Zero raw state/input violations and contacts were observed in all qualified executions.',
            'No negative margin was rounded into safety. Derived effort is not electrical energy.']
    lines+=['','## Numerical reliability, model discrepancy and latency','',
        f"Independently accepted retained candidates: {result['independently_accepted_candidates']}.",
        f"Maximum independent/native graph reconstruction difference: {result['maximum_program_reconstruction_error']:.12g}.",
        f"KKT residual maxima: `{result['KKT_maxima']}`.",
        'Native success, finite candidates, frozen v2 numerical tolerances and positive',
        'one-step reserves are separate checks. KKT conditions do not imply global NLP optimality.', '',
        '| Timing scope | Measured calls | Median ms | p95 ms | p99 ms | Max ms | >20 ms | Missing |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,t in result['timing_summary'].items():
        if t['calls']:
            lines.append('| '+name+' | '+str(t['calls'])+' | '+' | '.join(f"{1000*t[k]:.6f}" for k in ['median','p95','p99','maximum'])+f" | {t['over_20ms']} | {t['missing_observations']} |")
        else:lines.append(f"| {name} | 0 | N/A | N/A | N/A | N/A | N/A | {t['missing_observations']} |")
    metrics=[r['metrics'] for r in result['records'] if 'metrics' in r]
    if metrics:
        lines+=['',f"Maximum retained complete-record source closure: {max(m['source_closure_maximum'] for m in metrics):.12g}.",
            f"Maximum symbolic/plant one-control-step disagreement: {max(m['maximum_prediction_discrepancy'] for m in metrics):.12g}.",
            f"Maximum discrepancy-contract excess: {max(m['maximum_contract_excess'] for m in metrics):.12g}.",
            f"Maximum recorded native solver iteration count: {max(m['maximum_solver_iterations'] for m in metrics)}."]
    lines+=['','Cold starts and every 20-ms overrun are retained. Native action latency is',
        'an adjusted exclusive-span estimate; supervised latency includes qualification',
        'work. Solver time is the native solve boundary, not just printed IPOPT kernel',
        'time. Fixed-step offline simulation waits for computation; hard real-time',
        'readiness is not established. All individual overrun records and timing categories',
        'are retained in the machine-readable audit.','',
        '## Resource accounting and preservation','',
        f"Actual recorded native solver attempts: {result['actual_solver_attempts']}.",
        f"Actual physics steps: {result['actual_physics_steps']} (recorded lower bound {result['recorded_physics_step_lower_bound']}).",
        f"Conservative reservations: {result['reserved_solver_calls']} solves / {result['reserved_physics_steps']} steps.",
        f"Remaining original allowance: {result['remaining_original_solver_allowance']} solves / {result['remaining_original_physics_allowance']} steps.",
        f"Shared study CPU/wall charges after this audit: {resource['cpu_seconds']:.6f} / {resource['wall_seconds']:.6f} s.",
        f"Maximum recorded worker peak RSS: {result['maximum_recorded_worker_RSS_bytes']} bytes.",
        'These carry v1 and both preflights; no cap or failed reservation was reset or refunded.',
        'One worker and the original 1,800-CPU-second, 3,600-active-wall-second, 4-GiB',
        'and one-second action-watchdog limits remain. Scientific-worker peak RSS is',
        'retained per episode; aggregate parent peak is not separately reconstructed.',
        '',f"Preservation checks: `{result['preservation']}`.",
        'Accepted protocols, source/configuration hashes, v1 failure, historical reports,',
        'drilling authority, decision history and all nine IQL files remain unchanged.', '',
        '## Scientific interpretation and limitations','']
    if passed:
        lines+=['The unchanged MPC80 controller generated complete, reproducible, empirically',
            'constraint-satisfying demonstrations on the sixteen frozen initial states.',
            'Native replays and independently driven motor replays support live instrumentation',
            'nonintrusiveness on this cohort, including submitted force/motor command equality.',
            'This supports a finite-cohort offline simulation-expert role. It does not establish',
            'a certified operating region, formal robust/continuous-time/hardware safety,',
            'reward optimality, safe action diversity, learning headroom or acquisition benefit.']
    else:
        lines+=['The first-failure protocol was preserved. No failed case was retried or replaced.',
            'The retained failure must be reviewed before any correction or further execution.',
            'Incomplete or missing evidence cannot be counted as a safe demonstration.']
    lines+=['','This is a deterministic fixed-cohort study; repetitions are reproducibility',
        'checks, not independent statistical replicates. The sixteen states remain',
        'qualification/development data. Any future learned-policy evaluation requires',
        'a fresh independent cohort. Safe-action diversity, equal-budget active acquisition',
        'and beyond-expert learning each require separate evidence and approval.',
        'No action probes, counterfactual continuations, additional baselines, RL/IQL work,',
        'controller tuning, commits or pushes were performed. Stop for scientific review.','']
    lines+=['## Frozen identities','', '```json',
        json.dumps(result['frozen_identities'],indent=2,sort_keys=True),'```','']
    return '\n'.join(lines)


def summarize(root,upstream,pref,out):
    root,out,pref=Path(root),Path(out),Path(pref)
    if not ((out/'execution_complete.json').exists() or (out/'campaign_stop.json').exists()):
        raise RuntimeError('Campaign still active; no parallel verification worker permitted')
    execution.checkpoint(root,upstream,pref,out)
    model=read(out/'frozen.json');reference=np.load(out/'reference.npy');ledger=read(out/'resource_ledger.json')
    nlp=nlp_readout(model,80);records=[];times=[];candidates=[];candidate_count=0
    max_graph=0.;maxima={};actual_solves=0;physics_lower=0;physics_unknown=False
    for state_id in range(16):
        for phase in ['original','repeat']:
            folder=out/'runs'/f'{state_id}_nonlinear80_1000_{phase}'
            if not folder.exists():
                records.append({'state_id':state_id,'phase':phase,'status':'not_run','complete':False,'qualified':False});continue
            completion=read(folder/'completion.json') if (folder/'completion.json').exists() else None
            a=dict(np.load(folder/'trajectory.npz')) if (folder/'trajectory.npz').exists() else None
            full=bool(completion and completion['complete']);qualified=False
            row={'state_id':state_id,'phase':phase,'status':'complete' if full else 'incomplete','complete':full,'qualified':False}
            if completion:
                row['failure']=completion['failure'];row['controls']=completion['transitions']
                row['recorded_peak_RSS_bytes']=completion['peak_rss_bytes']
                actual_solves+=completion.get('actual_native_solve_attempts',completion['solver_calls'])
                physics_lower+=completion.get('actual_physics_steps',max(0,len(a.get('physics_states',[]))-1) if a else 0)
            else:row['failure']='completion metadata unavailable'
            if full:
                assert a is not None and len(a['physics_states'])==6001
                check=read(folder/'independent_episode_check.json') if (folder/'independent_episode_check.json').exists() else None
                both_replays=all((folder/name).exists() for name in ['replay_native.json','replay_instrumented.json'])
                if check:row['metrics']={k:v for k,v in check.items() if k!='timings'}
                if check and both_replays:
                    own,_=audit_record(folder,model,reference)
                    assert own['safe_complete_no_contact'] and own['physics_samples']==6001
                    assert abs(own['return']-check['return'])<=1e-10
                    assert abs(own['absolute_thrust_squared_integral']-check['absolute_effort'])<=1e-10
                    physics_reconstruction(a,model,1000)
                    stream=np.fromfile(folder/'partial_physics.f64',dtype='<f8').reshape(-1,7)
                    assert stream[:,:6].tobytes()==a['physics_states'].tobytes()
                    assert np.array_equal(stream[:,6],a['physics_contacts'])
                    direct=execution.verify_submissions(folder,'submitted_scientific',a)
                    motor=execution.verify_submissions(folder,'submitted_motor_replay',a)
                    assert direct[:,execution.PHYSICAL_COLUMNS].tobytes()==motor[:,execution.PHYSICAL_COLUMNS].tobytes()
                    native_rpm=np.load(folder/'native_replay_end_interval_RPM.npz')['end_interval_cache_RPM']
                    assert native_rpm.tobytes()==a['rpms'].tobytes()
                    # Independently reconstruct thrust from ACTUAL submitted force arguments.
                    forces=direct[:,17:29].reshape(-1,4,3)
                    force_groups=np.column_stack([forces[:,0,2]+forces[:,3,2],forces[:,1,2]+forces[:,2,2]])
                    assert np.isfinite(direct).all() and np.min(force_groups-model['action_low'])>=0
                    assert np.min(np.asarray(model['action_high'])-force_groups)>=0
                    direct_effort=float(np.sum(force_groups**2)/1000)
                    assert abs(direct_effort-check['absolute_effort'])<=1e-10
                    np.testing.assert_array_equal(direct[:,13:17],np.repeat(a['rpms'],20,axis=0))
                    resource_records=read(folder/'submitted_scientific_counts.json');assert resource_records['complete_6000']
                    if phase=='repeat':
                        original=out/'runs'/f'{state_id}_nonlinear80_1000_original'
                        orig=np.fromfile(original/'submitted_scientific.f64',dtype='<f8').reshape(-1,56)
                        assert orig[:,:54].tobytes()==direct[:,:54].tobytes()
                        from src.quadrotor_mpc80_execution_v1 import compare_repetition
                        assert compare_repetition(out,folder)['all_300_candidate_vectors_bitwise']
                    row.update(qualified=True,live_instrumentation_replay_passed=True,
                        motor_submissions_bitwise=True,force_argument_effort=direct_effort,
                        direct_derived_sensor_distinction_preserved=True)
                    qualified=True
            if a is not None and 'physics_states' in a:
                x=a['physics_states'];low=x-model['state_low'];high=np.asarray(model['state_high'])-x
                row['recorded_raw_states']=len(x)
                row['prefix_minimum_raw_margins']=np.minimum(low,high).min(0).tolist()
                row['prefix_state_violation_samples']=int(np.any((low<0)|(high<0),axis=1).sum())
                row['prefix_contacts']=int(np.count_nonzero(a['physics_contacts']))
            for directory in sorted((folder/'attempts').glob('*')) if (folder/'attempts').exists() else []:
                if not (directory/'status.json').exists():continue
                s=read(directory/'status.json');k=int(directory.name)
                candidate_status={'state_id':state_id,'phase':phase,'step':k,'retained':(directory/'candidate.npz').exists()}
                if candidate_status['retained']:
                    z=np.load(directory/'candidate.npz')
                    if all(n in z.files for n in ['primal','parameters','dual']):
                        try:
                            own=nlp(z['primal'],z['parameters'],z['dual'])
                            max_graph=max(max_graph,matching_program_values(own,[z[n] for n in ['objective','constraint_values','lower','upper','lagrangian_gradient']]))
                            d=canonical_kkt(own[1],own[2],own[3],z['dual'],own[4]);candidate_status['metrics']=d['metrics']
                            native=s.get('candidate_metadata',{}).get('solver_status')
                            candidate_status['native_status']=native;require_kkt(d,native)
                            candidate_status['independently_accepted']=True;candidate_count+=1
                            for key,value in d['metrics'].items():maxima[key]=max(maxima.get(key,0.),value)
                        except Exception as exc:
                            candidate_status.update(independently_accepted=False,error=repr(exc))
                            if qualified:raise
                candidates.append(candidate_status)
                spans=s.get('timings',[]);solvers=[r for r in spans if r['category']=='native_solver']
                timing={'state_id':state_id,'phase':phase,'step':k,
                    'solver_only':solvers[0]['wall_inclusive'] if len(solvers)==1 else None,
                    'native_action':s.get('timing_summary',{}).get('native_action_selection_wall'),
                    'supervised_action':float(a['solver_wall'][k]) if a is not None and k<len(a.get('solver_wall',[])) else None,
                    'failed_wrapper_checkpoint':s.get('observed_full_wrapper_wall_at_checkpoint'),
                    'exclusive_categories':s.get('timing_summary',{}).get('exclusive_categories',{})}
                times.append(timing)
            # Count replay work only when its real success marker exists. Partial
            # motor replay is directly counted; partial uninstrumented intervals
            # do not establish an exact engine-step count.
            for kind,marker in [('native_replay','replay_native.json'),('motor_replay','replay_instrumented.json')]:
                if (folder/marker).exists():
                    r=read(folder/marker)
                    assert all(v for k,v in r.items() if k.endswith('bitwise'))
                    physics_lower+=r['steps']*20
                elif (folder/(kind+'_failure.json')).exists():
                    if kind=='motor_replay' and (folder/'submitted_motor_replay_completed.u1').exists():
                        physics_lower+=(folder/'submitted_motor_replay_completed.u1').stat().st_size
                    else:physics_unknown=True
            records.append(row)
    original=[r for r in records if r['phase']=='original'];all_qualified=all(r['qualified'] for r in records)
    pairs=sum(all(r['qualified'] for r in records if r['state_id']==i) for i in range(16))
    passed=all_qualified and pairs==16 and not ledger['halted'] and len(ledger['reservations'])==96
    if passed:
        assert candidate_count==actual_solves==9600
        assert physics_lower==576000 and not physics_unknown
        assert all(r['status']=='completed' for r in ledger['reservations'])
    summary=None
    if passed:
        metrics=[r['metrics'] for r in original]
        fields=['return','cost','discounted_return','position_tracking_rmse','absolute_effort','hover_relative_effort','total_variation']
        summary={name:{'mean':float(np.mean([m[name] for m in metrics])),
                       'minimum':float(np.min([m[name] for m in metrics])),
                       'maximum':float(np.max([m[name] for m in metrics]))} for name in fields}
        for name in ['minimum_raw_margins','minimum_raw_lower_margins','minimum_raw_upper_margins','minimum_input_margins','minimum_reserves']:
            summary[name]=np.min([m[name] for m in metrics],axis=0).tolist()
        summary['minimum_normalized_raw_margin']=min(m['minimum_normalized_raw_margin'] for m in metrics)
        summary['mean_saturation_fraction_by_channel']=np.mean([m['saturation_fraction'] for m in metrics],axis=0).tolist()
        summary['mean_absolute_terminal_error_by_state']=np.mean(np.abs([m['terminal_tracking_error'] for m in metrics]),axis=0).tolist()
        summary['worst_absolute_terminal_error_by_state']=np.max(np.abs([m['terminal_tracking_error'] for m in metrics]),axis=0).tolist()
    timings={}
    for name in ['solver_only','native_action','supervised_action']:
        valid=[r[name] for r in times if r[name] is not None]
        timings[name]={**timing_stats(valid),'missing_observations':sum(r[name] is None for r in times),
            'overrun_records':[{k:r[k] for k in ['state_id','phase','step',name]} for r in times if r[name] is not None and r[name]>.02]}
    preserved=ledger_api.verify_preservation(root,upstream,read(out/'preservation_start.json'))
    return {'version':'E12_MPC80_FEAS_v2','new_expert_feasibility_gate_passed':passed,
        'complete_initial_state_pairs':pairs,'state_denominator':16,
        'complete_scientific_rollouts':sum(r['complete'] for r in records),
        'qualified_scientific_executions':sum(r['qualified'] for r in records),'execution_denominator':32,
        'records':records,'full_cohort_original_statistics':summary,'survivor_only_full_cohort_statistics_prohibited':True,
        'timing_summary':timings,'all_action_timing_records':times,'candidate_checks':candidates,
        'independently_accepted_candidates':candidate_count,'maximum_program_reconstruction_error':max_graph,
        'KKT_maxima':maxima,'actual_solver_attempts':actual_solves,
        'maximum_recorded_worker_RSS_bytes':max(
            (read(n).get('peak_rss_bytes',0) for n in (out/'job_results').glob('*.json')),default=0),
        'frozen_identities':{
            'accepted_preflight_seal_sha256':p.sha(pref/'protocol_seal.json'),
            'screening_manifest_sha256':p.sha(root/'results/benchmark_selection/mmaq_v1/screening_manifest.json'),
            'screening_states_npy_sha256':p.sha(root/'results/benchmark_selection/mmaq_v1/screening_states.npy'),
            'execution_binding_sha256':p.sha(out/'execution_binding.json'),
            'source_sha256':read(out/'execution_binding.json')['source_sha256'],
            'frozen_model_sha256':p.sha(out/'frozen.json'),
            'frozen_reference_sha256':p.sha(out/'reference.npy'),
            'accepted_config_sha256':p.sha(root/'configs/experiment_12_mpc80_feas_v2.json')},
        'actual_physics_steps':None if physics_unknown else physics_lower,'recorded_physics_step_lower_bound':physics_lower,
        'reserved_solver_calls':ledger['charged']['solver_calls'],'reserved_physics_steps':ledger['charged']['physics_steps'],
        'remaining_original_solver_allowance':12670-ledger['charged']['solver_calls'],
        'remaining_original_physics_allowance':942000-ledger['charged']['physics_steps'],
        'shared_resource_accounting_at_audit_start':ledger['charged'],'preservation':preserved,
        'v1_status':'FAILED, unchanged','original_overall_E12_qualification':'FAILED',
        'original_0001_agreement':'FAILED; unchanged historical threshold',
        'safety_scope':'finite-horizon empirical sampled simulator evidence; no formal robust or physical guarantee',
        'hard_real_time_ready':False,'new_audit_controller_calls':0,'new_audit_physics_steps':0}
