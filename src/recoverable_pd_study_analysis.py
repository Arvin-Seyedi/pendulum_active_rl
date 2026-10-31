"""Read-only scientific aggregation and independent saved-array audit for E09."""

from collections import Counter
import csv
import json
from pathlib import Path
import statistics
import time

import numpy as np

from src.counterfactual import write_csv
from src.horizon_experiment import _assert_hashes, load_npz, read_json, sha256, write_json
from src.viability_checker import check_extremal_certificate


def describe(values):
    values=list(values)
    return {'count':len(values),'mean':statistics.fmean(values) if values else None,
            'min':min(values) if values else None,'max':max(values) if values else None}


def aggregate_baseline(records):
    result={}
    groups={'all_grid':records,
            'archived_PD_unsafe':[r for r in records if r['identity']['archived_unsafe']],
            'archived_PD_safe':[r for r in records if not r['identity']['archived_unsafe']]}
    for name, rows in groups.items():
        result[name]={'count':len(rows),'raw_violation_count':sum(r['metrics']['violated'] for r in rows),
                      'raw_violation_rate':sum(r['metrics']['violated'] for r in rows)/len(rows) if rows else None,
                      'returns':describe(r['metrics']['returns'] for r in rows),
                      'discounted_returns':describe(r['metrics']['discounted_returns'] for r in rows),
                      'minimum_raw_margins':describe(r['metrics']['minimum_margins'] for r in rows),
                      'raw_first_crossing_seconds':describe(r['metrics']['first_crossing_seconds'] for r in rows
                                                          if r['metrics']['first_crossing_seconds'] is not None)}
    discrepancy_fields=['archived_vs_legacy_label_disagreement','archived_vs_raw_label_disagreement',
                        'legacy_vs_raw_label_disagreement','archived_vs_legacy_crossing_disagreement']
    result['discrepancy_ids']={k:[r['state_id'] for r in records if r['comparison'][k]] for k in discrepancy_fields}
    result['maximum_absolute_legacy_margin_difference']=max(abs(r['comparison']['legacy_margin_difference']) for r in records)
    result['raw_vs_legacy_first_crossing_difference_ids']=[r['state_id'] for r in records
           if r['comparison']['remeasured_raw_first_crossing_seconds'] != r['comparison']['remeasured_legacy_first_crossing_seconds']]
    return result


def independent_baseline_audit(output, baseline):
    """No rollouts: reconstruct native commands, safety, rewards and replay equality."""
    by_id={r['state_id']:r for r in baseline}
    count=0; scalar_fallbacks=0; reward_error=0.; discounted_error=0.
    started=time.perf_counter()
    for folder in sorted((output/'baseline').glob('chunk_*')):
        a=load_npz(folder/'original.npz'); b=load_npz(folder/'replay.npz')
        assert a.keys()==b.keys()
        for key in a:
            assert a[key].dtype==b[key].dtype and a[key].shape==b[key].shape
            assert a[key].tobytes()==b[key].tobytes(), (folder,key)
        info=load_npz(folder/'identity_and_legacy.npz')
        for k,sid_value in enumerate(info['grid_ids']):
            sid=int(sid_value); r=by_id[sid]; raw=a['raw_states'][k]; actions=a['actions'][k,:,0]
            assert raw.shape==(201,2) and actions.dtype==np.float32 and len(actions)==200
            assert raw[0].tobytes().hex()==r['identity']['initial_raw_bytes_hex']
            assert np.isfinite(raw).all() and np.isfinite(actions).all() and np.max(np.abs(actions))<=2
            margins=np.minimum(1-np.abs(raw[:,0]),1-np.abs(raw[:,1])/3)
            assert margins.tobytes()==a['margins'][k].tobytes()
            assert float(margins.min())==r['metrics']['minimum_margins']
            assert bool(np.any(margins<0))==r['metrics']['violated']
            crossing=np.flatnonzero(margins<0); first=int(crossing[0]) if len(crossing) else None
            assert r['metrics']['first_crossing_step']==first
            assert r['metrics']['first_crossing_seconds']==(None if first is None else first*.05)
            angles=np.arctan2(np.sin(raw[:-1,0]).astype(np.float32),np.cos(raw[:-1,0]).astype(np.float32))
            speeds=raw[:-1,1].astype(np.float32)
            expected=np.clip(-8.*angles-2.*speeds,-2.,2.).astype(np.float32)
            t0=np.arctan2(np.sin(raw[0,0]),np.cos(raw[0,0])); w0=raw[0,1]
            expected[0]=np.clip(-8.*t0-2.*w0,-2.,2.)
            legacy=np.minimum(1.-np.abs(angles),1.-np.abs(speeds)/3.).astype(np.float64)
            legacy[0]=min(1.-abs(t0),1.-abs(w0)/3.)
            if expected.tobytes()!=actions.tobytes() or legacy.tobytes()!=info['legacy_margins'][k].tobytes():
                # Scalar libm/ufunc dispatch can differ from vector dispatch;
                # resolve with the actual prescribed scalar observation path.
                scalar_fallbacks+=1
                for t in range(200):
                    obs=np.array([np.cos(raw[t,0]),np.sin(raw[t,0]),raw[t,1]],
                                 dtype=np.float64 if t==0 else np.float32)
                    angle,speed=np.arctan2(obs[1],obs[0]),obs[2]
                    expected[t]=np.clip(-8.*angle-2.*speed,-2.,2.)
                    legacy[t]=min(1.-abs(angle),1.-abs(speed)/3.)
            assert expected.tobytes()==actions.tobytes(),sid
            assert legacy.tobytes()==info['legacy_margins'][k].tobytes(),sid
            old=r['identity']['archived_unsafe']; new=bool(np.any(legacy<0))
            assert r['comparison']['archived_vs_legacy_label_disagreement']==(old!=new)
            assert r['comparison']['archived_vs_raw_label_disagreement']==(old!=bool(np.any(margins<0)))
            legacy_cross=np.flatnonzero(legacy<0)
            legacy_time=None if not len(legacy_cross) else int(legacy_cross[0])*.05
            assert r['comparison']['remeasured_legacy_first_crossing_seconds']==legacy_time
            assert r['comparison']['legacy_margin_difference']==float(legacy.min())-r['identity']['archived_minimum_margin']
            assert not a['terminated'][k].any() and not a['truncated'][k,:-1].any() and a['truncated'][k,-1]
            angle=(raw[:-1,0]+np.pi)%(2*np.pi)-np.pi
            ideal=-(angle**2+.1*raw[:-1,1]**2+.001*actions.astype(np.float64)**2)
            difference=float(np.max(np.abs(ideal-a['rewards'][k])))
            reward_error=max(reward_error,difference); assert difference<1e-7
            assert float(a['rewards'][k].sum())==r['metrics']['returns']
            discounted=sum(float(v)*(.99**t) for t,v in enumerate(a['rewards'][k]))
            difference=abs(discounted-r['metrics']['discounted_returns'])
            discounted_error=max(discounted_error,difference); assert difference<1e-10
            count+=1
    assert count==6561
    return {'passed':True,'episodes':count,'recorded_action_replay_pairs':count,
            'raw_state_samples':count*201,'native_action_sequences_verified':count,
            'scalar_dispatch_fallback_episodes':scalar_fallbacks,
            'maximum_reward_formula_difference':reward_error,'maximum_discounted_sum_difference':discounted_error,
            'extra_simulator_steps':0,'wall_seconds':time.perf_counter()-started}


def certificate_rows(screening,primary):
    states=[]; directions=[]; primary=set(primary)
    for sid,r in sorted(screening.items()):
        certs=[]
        for trial in r['trials']:
            final=trial['final']
            if final is None: continue
            d=final['diagnostics']; checked=final['checker']
            row={'state_id':sid,'primary_cohort':sid in primary,'action':trial['action'],
                 'status':final['status'],'certified_nonviable':checked['certified'],
                 'crossing_direction':('upper' if trial['action']==-2 else 'lower') if checked['certified'] else None,
                 'crossing_component':final['crossing_component'],'crossing_step':final['decisive_step'],
                 'crossing_seconds':None if final['decisive_step'] is None else final['decisive_step']/20,
                 'separation_lower_exact':json.dumps(d['crossing_separation_lower_exact']),
                 'separation_lower_display':d['crossing_separation_lower_approx'],
                 'max_theta_width':d['maximum_width_approx']['theta'],
                 'max_omega_width':d['maximum_width_approx']['omega'],
                 'safe_prefix_margin_lower':d['safe_prefix_minimum_margin_lower_approx'],
                 'precision_bits':final['precision_bits'],'reused_pilot':r['reused_pilot']}
            directions.append(row)
            if checked['certified']: certs.append(row)
        first=min(certs,key=lambda a:(a['crossing_step'],a['action'])) if certs else None
        states.append({'state_id':sid,'primary_cohort':sid in primary,
                       'theta_hex':r['identity']['initial_hex'][0],'omega_hex':r['identity']['initial_hex'][1],
                       'certified_nonviable':r['certified_nonviable'],
                       'mathematical_status':'certified_nonviable' if certs else 'exact_equilibrium_viable' if sid==3280 else 'unresolved_model',
                       'certifying_action':None if first is None else first['action'],
                       'crossing_direction':None if first is None else first['crossing_direction'],
                       'crossing_component':None if first is None else first['crossing_component'],
                       'crossing_step':None if first is None else first['crossing_step'],
                       'crossing_seconds':None if first is None else first['crossing_seconds'],
                       'separation_lower_display':None if first is None else first['separation_lower_display'],
                       'separation_lower_exact':None if first is None else first['separation_lower_exact'],
                       'max_theta_width':None if first is None else first['max_theta_width'],
                       'max_omega_width':None if first is None else first['max_omega_width'],
                       'precision_bits':None if first is None else first['precision_bits'],
                       'reused_pilot':r['reused_pilot'],'real_model_only':True})
    return states,directions


def independent_certificate_audit(output,screening):
    count=0; steps=0; started=time.perf_counter()
    for sid,r in sorted(screening.items()):
        for trial in r['trials']:
            name='u_minus2' if trial['action']==-2 else 'u_plus2'
            for attempt in trial['attempts']:
                if 'checker' not in attempt: continue
                folder=output/'certificates'/f'state_{sid:04d}'/name/f'precision_{attempt["precision_bits"]}'
                certificate=read_json(folder/'certificate.json')
                checked=check_extremal_certificate(certificate,sid,r['identity']['initial_hex'])
                assert checked==attempt['checker'] and checked['passed']
                count+=1; steps+=checked['check_count']
    return {'passed':True,'records_rechecked':count,'rational_transition_rechecks':steps,
            'extra_simulator_steps':0,'wall_seconds':time.perf_counter()-started}


def report_text(summary,verification,costs):
    p=summary['primary']; b=summary['baseline']; c=summary['certificates']
    lines=['# Experiment 09 — Identifying recoverable PD failures','',
           f'Primary result: **{p["certified_nonviable"]}/3230 archived PD-unsafe states have independently verified mathematical nonviability certificates**. Verified simulator-safe alternatives found: {p["verified_simulator_safe_alternatives"]}/3230; unresolved model cases: {p["unresolved"]}/3230.', '',
           '## Frozen protocol and provenance','',
           'The exact Experiment 01 81×81 axes, original masks, binary coordinates and grid IDs were reused. The PD action path preserves the original reconstructed float64 initial observation followed by native float32 Gym observations. All 6,561 baseline episodes were recorded for 200 steps and independently replayed, with safety checked on all 201 raw states. Five accepted pilot baselines and ten pilot interval records were reused unchanged. The primary denominator remains the original 3,230 failures; reference controls never enter it.', '',
           'The certificate method is unchanged from Experiment 08: omega_next = omega + (15 sin(theta)+3u)/20 and theta_next = theta + omega_next/20, exact rational parameters, exact archived binary initial states, |u|<=2, |theta|<=1, |omega|<=3. On the safe strip the map is monotone; speed clipping is inactive on steps beginning in the safe set. A safe extremal prefix followed by a strict upper crossing under -2 or lower crossing under +2 rules out every safe admissible sequence. Opposite exits never certify. Directed MPFR rounding and the independent exact-rational/Taylor checker validate every premise. Precision escalation remains 128→256→512 only for boundary ambiguity.', '',
           '## Outcomes and discrepancies','',
           f'Certificate directions: {c["direction_counts"]}; components: {c["component_counts"]}. Crossing steps: {c["crossing_steps"]["min"]}–{c["crossing_steps"]["max"]}, times {c["crossing_seconds"]["min"]}–{c["crossing_seconds"]["max"]} s. Minimum certified crossing separation (decimal display) {c["separations"]["min"]:.12g}. Final precision counts: {c["precision_counts"]}. Exact endpoints and separation lower bounds remain authoritative.', '',
           f'There were {summary["controls"]["false_nonviability_count"]} nonviability certificates on 32 non-equilibrium PD-safe reference controls. Equilibrium has an exact zero-input witness; other reference states retain empirical PD safety separately from mathematical viability.', '',
           f'Archive/reproduction discrepancy IDs: {b["discrepancy_ids"]}. Raw-versus-legacy first-crossing difference IDs: {b["raw_vs_legacy_first_crossing_difference_ids"]}. Maximum absolute legacy-margin discrepancy: {b["maximum_absolute_legacy_margin_difference"]:.12g}. Raw unwrapped margins can differ from archived wrapped-angle minima even when labels and crossing times agree. No archive values were overwritten.', '',
           '| PD cohort | Episodes | Mean return | Worst return | Mean discounted return | Violating episodes | Minimum raw margin |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for label,key in [('Whole original grid','all_grid'),('Archived PD-unsafe','archived_PD_unsafe'),('Archived PD-safe','archived_PD_safe')]:
        r=b[key]
        lines.append(f'| {label} | {r["count"]} | {r["returns"]["mean"]:.9f} | {r["returns"]["min"]:.9f} | {r["discounted_returns"]["mean"]:.9f} | {r["raw_violation_count"]} ({100*r["raw_violation_rate"]:.6f}%) | {r["minimum_raw_margins"]["min"]:.12g} |')
    lines+=['', 'All state identities, archived/legacy/raw labels, crossings and margins are in baseline_results.csv. primary_state_results.csv retains all 3,230 certificate outcomes, directions, steps, separations, widths and precisions. direction_results.csv retains every reference and primary extremal attempt, including noncertifying results.', '',
            '## Gated controller search and optimization','',
            'All primary states received valid certificates, so the approved protocol skipped every full-study alternative-controller and optimization evaluation. For each of the eight frozen alternative gain pairs, the number evaluated in the full-study search is zero (not a measured zero success rate). The optimizer subset is empty and full-study scientific solver attempts are zero. No new outcome-selected controller or global viability grid was introduced. The accepted pilot controller/optimization findings remain historical technical evidence and are not extrapolated into full-grid recovery rates.', '',
            '## Verification and computation','',
            f'Verification passed: {verification["passed"]}; protected pre-existing files: {verification["protected_files"]}. Baseline action replay pairs: {verification["baseline_audit"]["recorded_action_replay_pairs"]}; raw-state samples audited: {verification["baseline_audit"]["raw_state_samples"]}. Independent certificate archive rechecks: {verification["certificate_audit"]["records_rechecked"]}. Every interval generation repeated exactly; source, pilot, case and result hashes are preserved. Technical tests: {verification["technical_tests"]}. No model/simulator conflict was detected among evaluated trajectories.', '',
            f'New baseline simulator interactions: {costs["baseline"]["new_steps"]} steps, {costs["baseline"]["new_resets"]} resets, {costs["baseline"]["new_wall_seconds"]:.6f} simulator seconds. Reused pilot baseline evidence corresponds to 2,000 historical steps and ten resets; combined baseline audit evidence is 2,624,400 steps and 13,122 resets. No new extremal Gym rollouts were needed for mathematical certificates. New interval generation/repeat/checker updates: {costs["interval"]["new_updates"]}/{costs["interval"]["new_updates"]}/{costs["interval"]["new_checks"]}; generator/repeat/checker seconds: {costs["interval"]["new_generation_seconds"]:.6f}/{costs["interval"]["new_repeat_seconds"]:.6f}/{costs["interval"]["new_checker_seconds"]:.6f}. Independent final recheck and saved-array audit costs are recorded separately; no scientific training or full-study solver cost occurred.', '',
            '## Interpretation and limitations','',
            'For these exact 3,230 initial points under the declared real-valued map and constraints, another admissible controller cannot maintain sampled-time safety for 200 steps. This conclusion follows from checked comparison certificates rather than failed search or local optimization. It does not assert universal impossibility for floating-point Gymnasium executions: no implementation-wide error envelope is established. The PD traces agree with the certificate outcomes but do not close that numerical-model gap.', '',
            'This is a census of the fixed benchmark, not an estimate for all continuous initial conditions. No continuous-time, uncertain-state, model-error, real-world, or infinite-horizon claim is made. The archived PD-safe region is not thereby proved viable or robust. Safe RL research on this benchmark should distinguish infeasible initial states from controller mistakes and investigate a defensible feasible operating domain before seeking safety rescue. Any changed domain, filter, data collection or new experiment requires separate review. Experiments 01–08, the Research Decision Log, all prior results and unfinished IQL files remain unchanged. No commit or push was performed.', '']
    return '\n'.join(lines)


def finalize(root,output,pilot,baseline,screening,primary,controls,before,hashes):
    from src.recoverable_pd_study import select_controls
    baseline_summary=aggregate_baseline(baseline)
    states,directions=certificate_rows(screening,primary)
    primary_rows=[r for r in states if r['primary_cohort']]
    assert len(primary_rows)==3230 and all(r['certified_nonviable'] for r in primary_rows)
    for sid in primary:
        assert baseline[sid]['state_id']==sid
        assert baseline[sid]['metrics']['violated'], f'Model/simulator conflict or archive discrepancy at {sid}; investigate'
    assert all(not baseline[sid]['metrics']['violated'] for sid in controls)
    assert not any(screening[sid]['certified_nonviable'] for sid in controls)
    discrepancies=baseline_summary['discrepancy_ids']
    if any(discrepancies.values()) or baseline_summary['raw_vs_legacy_first_crossing_difference_ids']:
        write_json(output/'discrepancy_investigation_required.json',baseline_summary)
        raise RuntimeError('Baseline discrepancies retained; investigation required before scientific completion.')
    baseline_audit=independent_baseline_audit(output,baseline)
    write_json(output/'baseline_independent_audit.json',baseline_audit)
    certificate_audit=independent_certificate_audit(output,screening)
    write_json(output/'certificate_independent_audit.json',certificate_audit)
    cert_directions=[r for r in directions if r['primary_cohort'] and r['certified_nonviable']]
    summary={'primary':{'denominator':3230,'certified_nonviable':3230,'verified_simulator_safe_alternatives':0,
                        'unresolved':0,'unresolved_ids':[],'real_model_only':True},
             'baseline':baseline_summary,
             'certificates':{'direction_counts':dict(Counter(r['crossing_direction'] for r in cert_directions)),
                 'component_counts':dict(Counter(r['crossing_component'] for r in cert_directions)),
                 'crossing_steps':describe(r['crossing_step'] for r in cert_directions),
                 'crossing_seconds':describe(r['crossing_seconds'] for r in cert_directions),
                 'separations':describe(r['separation_lower_display'] for r in cert_directions),
                 'theta_widths':describe(r['max_theta_width'] for r in cert_directions),
                 'omega_widths':describe(r['max_omega_width'] for r in cert_directions),
                 'precision_counts':dict(Counter(str(r['precision_bits']) for r in directions)),
                 'direction_status_counts':dict(Counter(r['status'] for r in directions)),
                 'primary_unresolved_ids':[]},
             'controls':{'ids':controls,'count':33,'non_equilibrium_count':32,'false_nonviability_count':0,
                         'raw_PD_safe_count':33,'mathematical_viability_proved_ids':[3280]},
             'stages':{'search_required':False,'controller_evaluations':0,'optimization_scientific_solves':0,
                       'reason':'All 3230 original PD failures have independently verified model certificates'},
             'individual_controller_bank':[{'kp':kp,'kd':kd,'full_study_evaluations':0,'success_rate':None,
                         'status':'skipped_by_predefined_certificate_gate'}
                         for kp in (6.,8.,12.) for kd in (1.,2.,4.) if (kp,kd)!=(8.,2.)]}
    basecost=[read_json(p) for p in (output/'baseline').glob('chunk_*/costs.json')]
    attempts=[(r,a) for r in screening.values() for t in r['trials'] for a in t['attempts'] if 'checker' in a]
    new=[a for r,a in attempts if not r['reused_pilot']]
    old=[a for r,a in attempts if r['reused_pilot']]
    costs={'baseline':{'new_steps':sum(c['new_steps'] for c in basecost),
                       'new_resets':sum(c['new_resets'] for c in basecost),
                       'new_wall_seconds':sum(c['new_wall_seconds'] for c in basecost),
                       'reused_pilot_ids':sorted(sid for c in basecost for sid in c['reused_pilot_ids'])},
           'interval':{'new_updates':sum(a['diagnostics']['interval_steps'] for a in new),
                       'new_checks':sum(a['checker']['check_count'] for a in new),
                       'reused_pilot_updates':sum(a['diagnostics']['interval_steps'] for a in old),
                       'new_generation_seconds':sum(a['generation_seconds'] for a in new),
                       'new_repeat_seconds':sum(a['generation_repeat_seconds'] for a in new),
                       'new_checker_seconds':sum(a['checker_seconds'] for a in new)},
           'final_audits':{'baseline_seconds':baseline_audit['wall_seconds'],
                           'certificate_recheck_seconds':certificate_audit['wall_seconds'],
                           'certificate_transition_rechecks':certificate_audit['rational_transition_rechecks']},
           'full_study_solver_attempts':0,'new_controller_search_steps':0}
    assert costs['baseline']['new_steps']==2622400 and costs['baseline']['new_resets']==13112
    assert costs['baseline']['reused_pilot_ids']==[0,44,767,3280,6560]
    _assert_hashes(root,before['sha256']); _assert_hashes(root,hashes)
    _assert_hashes(pilot,read_json(pilot/'completion.json')['artifacts_sha256'])
    verification={'passed':True,'protected_files':len(before['sha256']),
                  'technical_tests':read_json(output/'technical_tests.json'),
                  'baseline_audit':baseline_audit,'certificate_audit':certificate_audit,
                  'pilot_preserved_bitwise':True,'all_generation_repeats_exact':True,
                  'failed_checks':[],'model_simulator_conflict_ids':[],
                  'baseline_discrepancy_ids':discrepancies,'real_model_only':True,
                  'implementation_error_bound_established':False,'full_study_search_skipped_as_prespecified':True}
    write_json(output/'summary.json',summary); write_json(output/'verification.json',verification)
    write_json(output/'computational_costs.json',costs)
    baseline_rows=[dict(state_id=r['state_id'],theta_hex=r['identity']['initial_hex'][0],
                   omega_hex=r['identity']['initial_hex'][1],**{k:v for k,v in r['comparison'].items() if k!='grid_id'},
                   **r['metrics']) for r in baseline]
    write_csv(output/'baseline_results.csv',baseline_rows)
    write_csv(output/'primary_state_results.csv',primary_rows)
    write_csv(output/'reference_state_results.csv',[r for r in states if not r['primary_cohort']])
    write_csv(output/'direction_results.csv',directions)
    report=report_text(summary,verification,costs)
    (output/'scientific_report.md').write_text(report)
    doc=root/'docs/experiment_09_report.md'
    if doc.exists(): assert doc.read_text()==report
    else: doc.write_text(report)
    artifacts={str(p.relative_to(output)):sha256(p) for p in sorted(output.rglob('*'))
               if p.is_file() and p.name!='.lock' and p!=output/'completion.json'}
    write_json(output/'completion.json',{'passed':True,'primary_denominator':3230,
               'artifacts_sha256':artifacts,'report_sha256':sha256(doc),'primary':summary['primary']})
    print('Full E09 study complete: 3230/3230 certified. Search and optimization skipped by frozen gate.',flush=True)
