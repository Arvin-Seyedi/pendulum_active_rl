"""Aggregate, independently audit, and report the unchanged Experiment 07 study."""

from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from src.counterfactual import assert_bitwise_equal
from src.horizon_experiment import _assert_hashes, load_npz, read_json, save_npz, sha256, write_json
from src.recoverability import validate_execution
from src.recoverability_experiment import collect_costs


def describe(values):
    x = np.asarray(values, dtype=float)
    if not len(x):
        return {'count': 0, 'mean': None, 'std': None, 'median': None, 'min': None, 'max': None, 'p95': None}
    return dict(count=len(x), mean=float(x.mean()), std=float(x.std(ddof=1)) if len(x)>1 else 0.,
                median=float(np.median(x)), min=float(x.min()), max=float(x.max()), p95=float(np.quantile(x,.95)))


def choose_oracle(candidates):
    available = [r for r in candidates if r['candidate_available']]
    return max(available, key=lambda r: (r['minimum_margins'], r['returns'], -r['start_index'])) if available else None


def aggregate(records, config):
    candidates = [r for record in records for r in record['candidates']]
    attempts = [r for record in records for r in record['attempts']]
    states = []
    for record in records:
        case, pd = record['case'], record['PD']
        state = {**case, 'PD_return': pd['returns'], 'PD_discounted_return': pd['discounted_returns'],
                 'PD_minimum_margin': pd['minimum_margins'], 'PD_first_crossing_seconds': pd['first_crossing_seconds']}
        for mode in ('open_loop', 'tracking'):
            rows = [r for r in record['candidates'] if r['mode'] == mode]
            state[mode + '_witness_count'] = sum(r['witness'] for r in rows)
            state[mode + '_any_witness'] = any(r['witness'] for r in rows)
            state[mode + '_oracle'] = choose_oracle(rows)
            for margin in config['margin_buffers']:
                state[f'{mode}_margin_at_least_{margin}'] = any(r['candidate_available'] and r['minimum_margins']>=margin for r in rows)
        state['any_witness'] = state['open_loop_any_witness'] or state['tracking_any_witness']
        state['solver_statuses'] = {r['start']:r['solver_status'] for r in record['attempts']}
        states.append(state)
    unsafe = [r for r in states if r['PD_unsafe']]
    primary = {'denominator': len(unsafe),
               **{mode + '_witness_ids': [r['state_id'] for r in unsafe if r[mode + '_any_witness']] for mode in ('open_loop','tracking')},
               'any_witness_ids': [r['state_id'] for r in unsafe if r['any_witness']],
               'no_witness_found_ids': [r['state_id'] for r in unsafe if not r['any_witness']],
               'tracking_only_ids': [r['state_id'] for r in unsafe if r['tracking_any_witness'] and not r['open_loop_any_witness']],
               'both_modes_ids': [r['state_id'] for r in unsafe if r['tracking_any_witness'] and r['open_loop_any_witness']]}
    groups = {}
    for name in ('PD_unsafe', 'PD_safe_references', 'nominal_safe', 'challenging_safe', 'equilibrium', 'all_non_equilibrium'):
        selected = [r for r in states if (r['group']==name or name=='PD_safe_references' and r['group'] in ('nominal_safe','challenging_safe') or name=='all_non_equilibrium' and r['group']!='equilibrium')]
        ids = {r['state_id'] for r in selected}
        g = {'states':len(selected), 'state_ids':sorted(ids), 'methods':{}}
        for mode in ('PD', 'open_loop', 'tracking'):
            if mode == 'PD':
                values=[record['PD'] for record in records if record['case']['state_id'] in ids]
            else:
                values=[r[mode+'_oracle'] for r in selected if r[mode+'_oracle'] is not None]
            g['methods'][mode] = {'evaluated_states':len(values), 'missing_states':len(selected)-len(values),
                'safe_states':sum(v['witness'] for v in values),
                'violation_rate':float(np.mean([v['violated'] for v in values])) if values else None,
                'returns':describe([v['returns'] for v in values]),
                'discounted_returns':describe([v['discounted_returns'] for v in values]),
                'minimum_raw_margins':describe([v['minimum_margins'] for v in values]),
                'observed_first_crossing_seconds':describe([v['first_crossing_seconds'] for v in values if v['first_crossing_seconds'] is not None]),
                'unsafe_state_fractions':describe([v['unsafe_state_fraction'] for v in values])}
            if mode != 'PD':
                g['methods'][mode]['paired_return_minus_PD'] = describe([v['delta_return'] for v in values])
                g['methods'][mode]['paired_discounted_minus_PD'] = describe([v['delta_discounted_return'] for v in values])
        rows = [r for r in candidates if r['state_id'] in ids]
        g['candidate_modes'] = {}
        for mode in ('open_loop','tracking'):
            entries=[r for r in rows if r['mode']==mode]
            available=[r for r in entries if r['candidate_available']]
            successful=[r for r in available if r['solver_success']]
            nominal=[r for r in available if r['nominally_feasible_at_solver_tolerance']]
            g['candidate_modes'][mode] = {'attempted_candidates':len(entries), 'available':len(available),
                'safe_witnesses':sum(r['witness'] for r in available), 'solver_success_candidates':len(successful),
                'solver_success_but_unsafe':sum(not r['witness'] for r in successful),
                'nominally_feasible_candidates':len(nominal), 'nominally_feasible_but_unsafe':sum(not r['witness'] for r in nominal),
                'safe_despite_nonconvergence':sum(r['witness'] and not r['solver_success'] for r in available),
                'maximum_angle_deviation':describe([r['maximum_plan_replay_state_discrepancy'][0] for r in available]),
                'maximum_velocity_deviation':describe([r['maximum_plan_replay_state_discrepancy'][1] for r in available]),
                'predicted_minimum_margin':describe([r['predicted_minimum_raw_margin'] for r in available]),
                'executed_minimum_margin':describe([r['minimum_margins'] for r in available])}
        subset=[r for r in attempts if r['state_id'] in ids]
        g['solver_status_counts']=dict(Counter(r['solver_status'] for r in subset))
        g['residuals']=describe([r['diagnostics']['maximum_dynamics_residual'] for r in subset if r['candidate_available']])
        groups[name]=g
    return {'primary':primary, 'groups':groups, 'states':states,
            'original_solver_status_counts':dict(Counter(r['solver_status'] for r in attempts)),
            'original_attempts':len(attempts), 'candidate_execution_count':sum(r['candidate_available'] for r in candidates),
            'watchdog_timeouts':sum(r['watchdog_timeout'] for r in attempts),
            'solver_timeouts':sum('Time' in r['solver_status'] for r in attempts),
            'scope':'Finite-horizon numerical witnesses at fixed sampled states; absence of a witness is inconclusive.',
            'oracle_rule':'Maximum executed minimum raw margin, then return, then earlier start; not deployable controller performance.'}


def write_table(path, rows):
    fields=sorted(set().union(*(row.keys() for row in rows)))
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def audit_saved_data(root, output, records, config):
    """Independent scalar Gym equations, reward, masks, controls, and replay checks."""
    count, max_reward_error, max_residual_difference = 0, 0., 0.
    error_traces, trace_ids, trace_modes, trace_starts, success_flags = [], [], [], [], []
    time_indices=np.array([0,1,20,50,100,150,200])
    for record in records:
        case=record['case']; sid=case['state_id']; initial=np.array(case['initial'])
        folder=output/f'state_{sid:03d}'
        baseline=load_npz(folder/'PD.npz')
        assert_bitwise_equal(baseline,load_npz(folder/'PD_replay.npz'))
        validate_execution(baseline,initial)
        rows=[(None,baseline,record['PD'],None,None)]
        for a in record['attempts']:
            if not a['candidate_available']:continue
            start=a['start']; original=folder/start/'original'
            vector=load_npz(original/a['candidate_source'])['decision']
            states=np.vstack([initial,vector[:400].reshape(200,2)]); controls=vector[400:]
            omega=states[:-1,1]+.05*(15*np.sin(states[:-1,0])+3*controls)
            predicted=np.column_stack([states[:-1,0]+.05*omega,omega])
            residual=float(np.abs(states[1:]-predicted).max())
            delta=abs(residual-a['diagnostics']['maximum_dynamics_residual'])
            max_residual_difference=max(max_residual_difference,delta)
            assert delta<1e-12
            for mode in ('open_loop','tracking'):
                arrays=load_npz(folder/start/mode/'rollout.npz')
                assert_bitwise_equal(arrays,load_npz(folder/start/mode/'replay.npz'))
                validate_execution(arrays,initial)
                metric=next(r for r in record['candidates'] if r['start']==start and r['mode']==mode)
                rows.append((mode,arrays,metric,states,controls))
                trace=np.abs(arrays['raw_states'][0]-states)
                error_traces.append(trace);trace_ids.append(sid);trace_modes.append(mode);trace_starts.append(start);success_flags.append(a['solver_success'])
                np.testing.assert_allclose(trace.max(axis=0),metric['maximum_plan_replay_state_discrepancy'],rtol=0,atol=0)
        for mode,arrays,metric,planned,controls in rows:
            raw=arrays['raw_states'][0];act=arrays['actions'][0,:,0]
            if mode:
                requested=controls.copy()
                if mode=='tracking':
                    error=raw[:-1]-planned[:-1]
                    requested-=8*error[:,0]+2*error[:,1]
                np.testing.assert_array_equal(act,np.clip(requested,-2,2).astype(np.float32))
            margins=np.minimum(1-np.abs(raw[:,0]),1-np.abs(raw[:,1])/3)
            np.testing.assert_array_equal(margins,arrays['margins'][0])
            assert metric['witness']==bool(margins.min()>=0)
            assert metric['minimum_margins']==float(margins.min())
            crossing=np.flatnonzero(margins<0)
            assert metric['first_crossing_step']==(int(crossing[0]) if len(crossing) else None)
            assert abs(metric['returns']-float(arrays['rewards'].sum()))<1e-12
            assert abs(metric['discounted_returns']-float(np.sum(arrays['rewards'][0]*.99**np.arange(200))))<1e-9
            for t in range(200):
                theta,omega=raw[t]; u=act[t]
                velocity=np.clip(omega+(15*np.sin(theta)+3.0*u)*.05,-8,8)
                np.testing.assert_array_equal(raw[t+1],[theta+.05*velocity,velocity])
                reward=-(((theta+np.pi)%(2*np.pi)-np.pi)**2+.1*omega**2+.001*(u**2))
                max_reward_error=max(max_reward_error,abs(float(reward)-arrays['rewards'][0,t]))
            count+=1
    assert max_reward_error<1e-12
    errors=np.array(error_traces); modes=np.array(trace_modes); successful=np.array(success_flags)
    save_npz(output/'plan_execution_discrepancies.npz',dict(absolute_errors=errors,state_ids=np.array(trace_ids),modes=modes,
             starts=np.array(trace_starts),solver_success=successful))
    sensitivity={}
    for mode in ('open_loop','tracking'):
        selected=errors[(modes==mode)&successful]
        sensitivity[mode]={'solver_success_candidates':len(selected),'time_points_seconds':(time_indices*.05).tolist(),
            'angle_error_at_times':[describe(selected[:,i,0]) for i in time_indices],
            'velocity_error_at_times':[describe(selected[:,i,1]) for i in time_indices]}
    write_json(output/'numerical_sensitivity.json',sensitivity)
    return {'passed':True,'trajectories_and_replays_checked':count,'extra_simulator_calls':0,
            'maximum_reward_error':max_reward_error,'maximum_residual_recomputation_difference':max_residual_difference,
            'raw_states_and_float32_torques_verified':True,'solver_status_not_used_for_witness_acceptance':True}


def costs_with_provenance(output, pilot):
    # Aggregate files are not individual simulator cost records. Reuse a
    # completed aggregate if report generation was interrupted afterwards.
    saved = output / 'interaction_costs.json'
    if saved.exists():
        return read_json(saved)
    combined=collect_costs(output)
    old=read_json(pilot/'interaction_costs.json')
    new={key:combined[key]-old[key] for key in ('total_simulator_steps','total_resets','solver_attempts','solver_process_seconds','solver_seconds','watchdog_timeouts','ipopt_iterations_total')}
    new['simulator_categories']={name:{key:value-old['simulator_categories'].get(name,{}).get(key,0) for key,value in values.items()}
                                 for name,values in combined['simulator_categories'].items()}
    return {'combined_including_reused_pilot':combined,'new_full_study_execution':new,'reused_pilot':old,
            'note':'Pilot cases, repeated pilot solves, and model tests are reused, not re-executed. Solver attempts include the twelve pilot repeats; scientific starts total 171.'}


def make_plots(output, summary, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,mode in zip(axes,('open_loop','tracking')):
        for group,marker in [('PD_unsafe','^'),('nominal_safe','o'),('challenging_safe','s'),('equilibrium','*')]:
            rows=[r for r in summary['states'] if r['group']==group]
            for hit,color in [(False,'tab:red'),(True,'tab:blue')]:
                chosen=[r for r in rows if r[mode+'_any_witness']==hit]
                if chosen:ax.scatter([r['initial'][0] for r in chosen],[r['initial'][1] for r in chosen],c=color,marker=marker,label=f'{group}: '+('witness' if hit else 'none found'))
        ax.set(xlabel='Initial raw angle',ylabel='Initial angular velocity',title=mode)
        ax.legend(fontsize=7)
    fig.savefig(output/'witness_map.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for ax,mode in zip(axes,('open_loop','tracking')):
        rows=[r for record in records for r in record['candidates'] if r['mode']==mode and r['candidate_available']]
        for success,color in [(True,'tab:blue'),(False,'tab:red')]:
            chosen=[r for r in rows if r['solver_success']==success]
            ax.scatter([r['predicted_minimum_raw_margin'] for r in chosen],[r['minimum_margins'] for r in chosen],s=16,alpha=.6,c=color,label='solver success' if success else 'solver nonconvergence')
        ax.axhline(0,color='black',linewidth=.7);ax.set(xlabel='Predicted minimum raw margin',ylabel='Executed minimum raw margin',title=mode)
        ax.legend(fontsize=8)
    fig.savefig(output/'predicted_executed_margins.png',dpi=160);plt.close(fig)


def report_text(summary, verification, costs):
    p=summary['primary'];groups=summary['groups']; c=costs['combined_including_reused_pilot'];new=costs['new_full_study_execution']
    lines=['# Experiment 07 — Safety Recoverability Analysis', '',
           f"Primary outcome: **{len(p['any_witness_ids'])}/{p['denominator']} PD-unsafe initial states** admitted at least one independently replayed safe sequence. Open-loop: {len(p['open_loop_witness_ids'])}/40; feedback tracking: {len(p['tracking_witness_ids'])}/40. These are finite-horizon numerical witnesses, not formal recoverability or controller guarantees.", '',
           '## Frozen protocol and verification', '',
           'The accepted 57-state manifest contains all 40 PD-unsafe states, eight nominal PD-safe references, eight challenging PD-safe references, and equilibrium. The four accepted pilot cases were reused byte-for-byte; 53 additional cases were executed. The scientific configuration, model, objective, solver, gains, and safety acceptance criteria are unchanged from the pilot.', '',
           'CasADi 3.7.2, IPOPT 3.14.11, MUMPS 5.4.1; 600-variable, 400-equality direct multiple shooting; three fixed starts (PD, zero, seeded 20-step random blocks). Each start has 1000 iterations, 30 s solver limit, 60 s watchdog; tolerances 1e-8, exact derivatives, no bound relaxation or acceptable-termination heuristic. Hard constraints: |theta|≤1, |omega|≤3, |u|≤2. Horizon 200 steps (10 s), discount 0.99. Safety is checked on all 201 raw states with no tolerance for negative margin.', '',
           'Every available candidate is executed independently in Gymnasium as float32 torques, once open-loop and separately with fixed feedback tracking. Each actual action sequence is then independently replayed. Unsuccessful and timed-out candidates are retained and evaluated when finite. No training dataset is changed and no RL controller is trained.', '',
           f"All checks passed: {verification['passed']}. {verification['independent_audit']['trajectories_and_replays_checked']} evaluated trajectories and their replays passed the independent audit; the original PD traces match Experiment 06A bitwise. {verification['protected_files']} pre-existing files are unchanged, including the accepted pilot and unfinished IQL files. The twelve paired pilot solves remain the solver reproducibility check; no extra starts were introduced.", '',
           '## Safe-witness counts', '',
           '| Group | States | PD-safe | Open-loop witness | Tracking witness | Either mode |',
           '|---|---:|---:|---:|---:|---:|']
    for name in ('PD_unsafe','nominal_safe','challenging_safe','equilibrium'):
        g=groups[name]; selected=[r for r in summary['states'] if r['group']==name]
        lines.append(f"| {name} | {g['states']} | {g['methods']['PD']['safe_states']} | {g['methods']['open_loop']['safe_states']} | {g['methods']['tracking']['safe_states']} | {sum(r['any_witness'] for r in selected)} |")
    lines+=['',f"PD-unsafe states with an open-loop witness: {p['open_loop_witness_ids']}.",
            f"PD-unsafe states with a tracking witness: {p['tracking_witness_ids']}.",
            f"PD-unsafe states where neither mode found a safe witness: {p['no_witness_found_ids']}.", '',
            'The complete state IDs, initial coordinates, individual start outcomes, and missing-candidate flags are recorded in state_outcomes.csv and candidate_metrics.csv. No-witness states are not classified as unrecoverable.', '',
            '## Return and safety outcomes', '',
            'For optimized modes, each state uses the prespecified oracle choice: greatest executed minimum margin, then greatest return, then earliest start. This is outcome-informed selection among three candidates, not deployable policy performance. Failed and unsafe candidates remain included; the tables do not condition on safe outcomes.', '',
            '| Group | Method | Mean return | Worst return | Mean discounted return | Violation rate | Minimum raw margin |',
            '|---|---|---:|---:|---:|---:|---:|']
    for name in ('PD_unsafe','PD_safe_references','all_non_equilibrium'):
        for mode in ('PD','open_loop','tracking'):
            x=groups[name]['methods'][mode]
            if x['evaluated_states']:
                lines.append(f"| {name} | {mode} | {x['returns']['mean']:.6f} | {x['returns']['min']:.6f} | {x['discounted_returns']['mean']:.6f} | {100*x['violation_rate']:.2f}% | {x['minimum_raw_margins']['min']:.6f} |")
    lines+=['','## Solver outcomes and numerical sensitivity','',
            f"The 171 scientific starts produced these statuses: {summary['original_solver_status_counts']}. External watchdog timeouts: {summary['watchdog_timeouts']}; solver time-limit terminations: {summary['solver_timeouts']}.",'',
            '| Group / mode | Solver-success candidates | Success but executed unsafe | Nominally feasible but executed unsafe | Largest angle deviation | Largest velocity deviation |',
            '|---|---:|---:|---:|---:|---:|']
    for name in ('PD_unsafe','PD_safe_references','equilibrium'):
        for mode in ('open_loop','tracking'):
            x=groups[name]['candidate_modes'][mode]
            if x['available']:
                lines.append(f"| {name} / {mode} | {x['solver_success_candidates']} | {x['solver_success_but_unsafe']} | {x['nominally_feasible_but_unsafe']} | {x['maximum_angle_deviation']['max']:.6g} | {x['maximum_velocity_deviation']['max']:.6g} |")
    lines+=['', 'Nominal feasibility here requires small dynamics and bound residuals (≤1e-8) and nonnegative predicted raw margins. Merely satisfying state bounds at shooting nodes is insufficient. Residuals, predicted versus executed margins, terminal deviations, first-crossing times, and constraint statuses are retained for every candidate.', '',
            'The discrepancy traces and time summaries use already saved plans and executions; no precision, solver, dynamics, control, or acceptance settings were changed. Large open-loop divergence despite small local numerical errors is consistent with amplification in unstable dynamics. This is a sensitivity observation, not a controlled causal decomposition. Feedback tracking results are reported separately and remain empirical.', '',
            '## Computation and preservation', '',
            f"New execution: {new['solver_attempts']} solver attempts, {new['solver_seconds']:.3f} solver seconds, {new['solver_process_seconds']:.3f} worker-process seconds, {new['total_simulator_steps']} simulator steps, {new['total_resets']} resets.",
            f"Including reused pilot work: {c['solver_attempts']} solver attempts (171 scientific starts plus 12 pilot repeats), {c['ipopt_iterations_total']} IPOPT iterations, {c['solver_seconds']:.3f} solver seconds, {c['solver_process_seconds']:.3f} worker-process seconds, {c['total_simulator_steps']} simulator steps, {c['total_resets']} resets. These totals include initial guesses, evaluation, replay, and the pilot's 1031 one-step model checks.", '',
            'Solver logs, defaults/options, initial guesses, last iterates, unsuccessful candidates, and timeout records are retained. interaction_costs.json separates new and reused work and gives simulator categories and NLP evaluation counts. Source, configuration, case, state-result, and final-output hashes protect reproducibility. No previous files, datasets, or checkpoints were overwritten.', '',
            '## Scientific scope', '',
            'A verified witness establishes an observed admissible 200-step sequence for an exact sampled state in the nominal simulator. It does not establish infinite-horizon viability, robustness, safety between discrete samples, or a certified backup controller. A tracking-generated witness is not evidence that its fixed feedback law is safe from other states.', '',
            'A local solver reporting infeasibility, exhausting its budget, or failing to produce a safe replay does not prove physical unrecoverability. The 40 PD-unsafe cases were conditioned on prior PD failure; the 16 reference states and equilibrium are controls, not independent population-level validation. Counts are descriptive for this fixed sample. These results can inform later backup-controller research but do not justify deployment or alter the reviewed IQL conclusions.', '']
    return '\n'.join(lines)


def finish_analysis(root, output, records, config, before, hashes):
    assert len(records)==57 and len({r['case']['state_id'] for r in records})==57
    assert sum(r['case']['PD_unsafe'] for r in records)==40
    summary=aggregate(records,config)
    assert summary['original_attempts']==171
    audit=audit_saved_data(root,output,records,config)
    pilot=root/'results/recoverability/pilot'
    # Reused pilot files must remain byte-identical in both locations.
    for sid in read_json(pilot/'state_manifest.json')['pilot_ids_in_order']:
        origin=pilot/f'state_{sid:03d}'; copied=output/f'state_{sid:03d}'
        for p in origin.rglob('*'):
            if p.is_file():assert sha256(p)==sha256(copied/p.relative_to(origin)),str(p)
    costs=costs_with_provenance(output,pilot)
    _assert_hashes(root,before['sha256']);_assert_hashes(root,hashes)
    verification={'passed':True,'independent_audit':audit,'protected_files':len(before['sha256']),
                  'scientific_configuration_matches_pilot':True,'pilot_copies_bitwise':True,
                  'states':57,'PD_unsafe_states':40,'scientific_starts':171,
                  'technical_tests':read_json(output/'full_technical_tests.json'),
                  'pilot_reproducibility':read_json(pilot/'verification.json')['solver_repetitions'],
                  'datasets_changed':False,'RL_training':False}
    write_json(output/'summary.json',summary);write_json(output/'verification.json',verification)
    write_json(output/'interaction_costs.json',costs)
    write_table(output/'state_outcomes.csv',summary['states'])
    write_table(output/'candidate_metrics.csv',[r for record in records for r in record['candidates']])
    write_table(output/'solver_attempts.csv',[r for record in records for r in record['attempts']])
    missing=[r for r in summary['states'] if r['PD_unsafe'] and not r['any_witness']]
    if missing:write_table(output/'PD_unsafe_no_witness.csv',missing)
    make_plots(output,summary,records)
    report=report_text(summary,verification,costs)
    (output/'scientific_report.md').write_text(report)
    report_path=root/'docs/experiment_07_report.md'
    if report_path.exists() and report_path.read_text()!=report:
        raise FileExistsError('Refusing to overwrite a different existing report.')
    report_path.write_text(report)
    artifact_hashes={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name not in ('.lock','completion.json','execution.log')}
    write_json(output/'completion.json',{'passed':True,'states':57,'primary':summary['primary'],
               'artifacts_sha256':artifact_hashes,'report_sha256':sha256(report_path)})
