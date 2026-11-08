"""Read-only verification of targeted v2 records; historical v1 stays unchanged."""
from pathlib import Path
from fractions import Fraction
import numpy as np
from src import quadrotor_pilot as p
from src.quadrotor_mmaq_v1 import safe_write
from src.quadrotor_h2_requalification_v2 import check_seal,protect
from src.quadrotor_kkt_integrity_v2 import canonical_kkt,require_kkt,matching_program_values,nlp_readout
from src.quadrotor_mmaq_verification_v1 import nlp_metrics as legacy_metrics,require_nlp as legacy_require
from src.quadrotor_mmaq_verification_v1 import exact_majorant
from src.quadrotor_verification import audit_record


def timing_stats(values):
    a=np.asarray(values)
    return {'calls':len(a),'median':float(np.median(a)),'p95':float(np.percentile(a,95)),
            'p99':float(np.percentile(a,99)),'maximum':float(a.max()),
            'over_20ms':int(np.count_nonzero(a>.02))} if len(a) else {'calls':0}


def verify(root,upstream,out):
    root,out=Path(root),Path(out);check_seal(root,out);f=p.read(out/'frozen.json');ref=np.load(out/'reference.npy')
    old=root/'results/benchmark_selection/mmaq_v1';fn=nlp_readout(f,20)
    rows=[];legacy_rows=[];timing_rows=[];candidate_sets={};prefix_comparisons=[]
    for phase in ['original','repeat']:
        folder=out/'runs'/f'H2_nonlinear20_1000_{phase}';m=p.read(folder/'completion.json')
        if not m['complete']:
            # Retain the failure and raw prefix rather than manufacturing a
            # full-horizon return or assigning a truncated episode safe status.
            row={'phase':phase,'complete':False,'safe_complete_no_contact':False,'failure':m['failure'],
                 'transitions':m['transitions'],'actual_solver_attempts':m['actual_native_solve_attempts']}
            if m['transitions']:
                prefix,_=audit_record(folder,f,ref);row['prefix_audit']=prefix
            rows.append(row);continue
        row,a=audit_record(folder,f,ref);assert len(a['physics_states'])==6001
        for marker in ['replay_native.json','replay_instrumented.json']:
            replay=p.read(folder/marker)
            assert all(value for key,value in replay.items() if key.endswith('bitwise'))
        stream=np.fromfile(folder/'partial_physics.f64',dtype='<f8').reshape(-1,7)
        assert stream.shape==(6001,7)
        assert stream[:,:6].tobytes()==a['physics_states'].tobytes()
        assert np.array_equal(stream[:,6],a['physics_contacts'])
        rpm=a['rpms'];executed=f['KF']*np.column_stack([rpm[:,0]**2+rpm[:,3]**2,rpm[:,1]**2+rpm[:,2]**2])
        input_margin=np.minimum(executed-f['action_low'],np.asarray(f['action_high'])-executed)
        row['minimum_input_margins']=input_margin.min(0).tolist();row['input_violations']=int(np.count_nonzero(input_margin<0))
        row['safe_complete_no_contact']=row['safe_complete_no_contact'] and row['input_violations']==0
        candidates=[];maxima={};causal=p.read(folder/'causal_contract.json');reserves=[];closures=[]
        for k in range(300):
            attempt=folder/'attempts'/f'{k:04d}';status=p.read(attempt/'status.json');data=np.load(attempt/'candidate.npz')
            assert status['status']=='verified_action_not_yet_applied' and not status['fallback_action_applied']
            assert not status['missing_fields']
            assert p.sha(attempt/'candidate.npz')==status['candidate_sha256']
            z,params,lam=[data[name].reshape(-1) for name in ['primal','parameters','dual']]
            assert np.max(np.abs(params[:6]-a['states'][k]))<1e-14
            expected=ref[np.minimum(k+np.arange(21),300)].T.reshape(-1,order='F')
            np.testing.assert_array_equal(params[6:],expected)
            own=[np.asarray(v).reshape(-1) for v in fn(z,params,lam)]
            matching_program_values(own,[data[name] for name in ['objective','constraint_values','lower','upper','lagrangian_gradient']])
            diagnostic=canonical_kkt(own[1],own[2],own[3],lam,own[4])
            native_meta=status['candidate_metadata'];require_kkt(diagnostic,native_meta['solver_status'])
            assert native_meta['native_stats']['success']
            for key,value in diagnostic['metrics'].items():maxima[key]=max(maxima.get(key,0),value)
            saved=p.read(attempt/'independent_constraint_diagnostics.json')
            for key,value in diagnostic['metrics'].items():assert abs(saved['metrics'][key]-value)<1e-12
            candidates.append((z,params,lam))
            # Evaluate frozen v1 mathematics ONLY on this newly retained point.
            # This never fills a missing historical vector or changes v1 status.
            native=lambda z_,p_: [data[name].reshape(-1) for name in ['objective','constraint_values','lower','upper']]
            oldmetric=legacy_metrics(fn,native,z,params,lam)
            try:legacy_require(oldmetric,native_meta['solver_status']);old_pass=True;error=None
            except Exception as exc:old_pass=False;error=str(exc)
            if not old_pass:
                mask=np.isneginf(own[2])&np.isfinite(own[3])&(lam<0)
                legacy_rows.append({'phase':phase,'step':k,'v1_would_accept_new_point':False,
                    'new_v2_metrics':diagnostic['metrics'],'legacy_metrics':oldmetric,'legacy_error':error,
                    'negative_upper_only_rows':np.flatnonzero(mask).tolist(),'negative_upper_only_multipliers':lam[mask].tolist(),
                    'evidence_origin':'new v2 candidate, not recovered historical rejected candidate'})
            b=causal[k];cache=a['physics_states'][max(0,20*k-1),4]
            assert b['cache']==cache and np.array_equal(b['initial'],a['states'][k])
            exact=exact_majorant(a['states'][k],b['model_thrust'],b['executed_thrust'],cache,f)
            assert all(Fraction(float(value))>=bound for value,bound in zip(b['bound'],exact))
            closures.append(float(np.max(np.abs(np.asarray(b['source_prediction'])-a['states'][k+1]))))
            assert np.all(np.abs(np.asarray(b['prediction'])-a['states'][k+1])<=np.asarray(b['bound'])+1e-10)
            reserves.append(b['reserve'])
            spans=status['timings'];kernels=[r for r in spans if r['category']=='native_solver'];assert len(kernels)==1
            native_time=status['timing_summary']['native_action_selection_wall'];kernel=kernels[0]['wall_inclusive']
            assert native_time>=kernel-1e-9
            timing_rows.append({'phase':phase,'step':k,'solver_only_wall':kernel,'native_action_wall':native_time,
                'supervised_action_wall':float(a['solver_wall'][k]),'categories':status['timing_summary']['exclusive_categories'],
                'solver_overrun':kernel>.02,'native_action_overrun':native_time>.02,'supervised_overrun':a['solver_wall'][k]>.02})
        assert max(closures)<1e-10
        row.update(v2_KKT_passed=True,KKT_maxima=maxima,source_closure_maximum=max(closures),
                   minimum_reserve_by_component=np.min(reserves,0).tolist(),
                   physics_stream_write_seconds=float(a['physics_stream_write_seconds'].sum()))
        candidate_sets[phase]=candidates
        old_folder=old/'runs'/f'H2_nonlinear20_1000_{phase}'
        hist=np.load(old_folder/'trajectory.npz');oldc=np.load(old_folder/'nlp_solutions.npz')
        fields={name:a[name][:len(hist[name])].tobytes()==hist[name].tobytes() for name in
            ['states','physics_states','physics_contacts','actions','clipped_actions','rpms','rewards','predicted_states','predicted_inputs']}
        vectors={name:np.asarray([point[i] for point in candidates[:54]]).tobytes()==oldc[name].tobytes()
                 for i,name in enumerate(['primal','parameters','dual'])}
        prefix_comparisons.append({'phase':phase,'historical_applied_controls':54,
            'historical_fields_bitwise':fields,'historical_retained_candidates_bitwise':vectors,
            'historical_v1_outcome':'unresolved verifier rejection; unchanged'})
        rows.append(row)
    repeats={}
    if all(r['complete'] for r in rows):
        x=np.load(out/'runs/H2_nonlinear20_1000_original/trajectory.npz');y=np.load(out/'runs/H2_nonlinear20_1000_repeat/trajectory.npz')
        for name in ['states','actions','clipped_actions','rpms','rewards','physics_states','physics_contacts','predicted_states','predicted_inputs','solver_iterations']:
            repeats[name]=x[name].tobytes()==y[name].tobytes()
        repeats['primal_parameters_duals']=all(all(left.tobytes()==right.tobytes() for left,right in zip(a,b))
                    for a,b in zip(candidate_sets['original'],candidate_sets['repeat']))
        assert all(repeats.values())
    summary={key:timing_stats([r[field] for r in timing_rows]) for key,field in [
        ('solver_only','solver_only_wall'),('native_action','native_action_wall'),('supervised_action','supervised_action_wall')]}
    passed=all(r['complete'] and r['safe_complete_no_contact'] and r.get('v2_KKT_passed') for r in rows) and bool(repeats) and all(repeats.values())
    result={'version':'E12_H2_MPC20_requalification_v2','passed':bool(passed),'records':rows,
        'repeats_bitwise':repeats,'historical_prefix_comparisons':prefix_comparisons,
        'legacy_would_reject_new_points':legacy_rows,'timings':timing_rows,'timing_summary':summary,
        'historical_v1_results_unchanged':True,'overall_benchmark_qualification':'FAILED; not superseded by this targeted result',
        'original_symbolic_agreement_gate':'FAILED at unchanged 0.001','new_original_gate_passed':all(r.get('one_step_tolerance_passed',False) for r in rows),
        'no_screening_or_probes_or_training':True}
    safe_write(out/'verification.json',result);safe_write(out/'preservation_end.json',protect(root,upstream))
    return {'passed':result['passed'],'complete_executions':sum(r['complete'] for r in rows),
        'returns':[r.get('return') for r in rows],'legacy_rejections_of_new_points':len(legacy_rows),
        'historical_prefix_bitwise':all(all(r['historical_fields_bitwise'].values()) and all(r['historical_retained_candidates_bitwise'].values()) for r in prefix_comparisons),
        'timings':summary}
