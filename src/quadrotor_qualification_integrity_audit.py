"""Static E12 integrity audit. No plant, controller evaluation or solve calls."""
from pathlib import Path
import re
import numpy as np
from scipy import linalg
from src import quadrotor_pilot as p
from src.quadrotor_mmaq_v1 import safe_write
from src.quadrotor_kkt_integrity_v2 import canonical_kkt,require_kkt,nlp_readout
from src.quadrotor_linear_verification import independent_program,parameter_terms


def prepare_audit(root,source,out,upstream):
    """Read-only verification of all accepted seals before a fresh audit."""
    import time
    from src.quadrotor_mmaq_v1 import check_seal,preservation
    root,source,out=Path(root),Path(source),Path(out)
    if (out/'preservation_start.json').exists():
        saved=p.read(out/'preservation_start.json')
        assert saved['accepted_mmaq_completion_sha256']==p.sha(source/'completion.json')
        return saved
    start=time.process_time();seal=p.read(source/'completion.json')
    for name,h in seal['artifact_sha256'].items():assert p.sha(source/name)==h,name
    for label in ['execution_source_sha256','postrun_source_sha256']:
        for name,h in seal[label].items():assert p.sha(root/name)==h,name
    check_seal(root,source);old=preservation(root,upstream)
    iq=p.read(root/'results/benchmark_selection/technical_pilot/iql_preservation.json')['sha256']
    for name,h in iq.items():assert p.sha(root/name)==h,name
    record={'accepted_mmaq_completion_sha256':p.sha(source/'completion.json'),
        'accepted_mmaq_files_verified':len(seal['artifact_sha256']),'previous_preservation':old,
        'all_nine_iql_verified':True,'iql_sha256':iq,'cpu_seconds':time.process_time()-start,
        'charged_separately':False,'new_solver_calls':0,'new_physics_steps':0,'new_episodes':0}
    safe_write(out/'preservation_start.json',record);return record


def scalar_native_reports(text):
    keys=['Dual infeasibility','Constraint violation','Complementarity','Overall NLP error']
    result={}
    for line in text.splitlines():
        for key in keys:
            if line.startswith(key):
                vals=re.findall(r'[-+]?\d+\.\d+e[-+]\d+',line)
                if len(vals)==2:result[key]={'scaled':float(vals[0]),'unscaled':float(vals[1])}
    return result


def retained_kkt(source,out):
    """Recompute from retained candidates only; never fabricate rejected vectors."""
    source,out=Path(source),Path(out);f=p.read(source/'frozen.json')
    readouts={N:nlp_readout(f,N) for N in [20,40,80]};qp=independent_program(f)
    records=[];missing=[];arrays={};all_max={}
    for folder in sorted((source/'runs').iterdir()):
        m=p.read(folder/'completion.json');linear=m['controller']=='linear40'
        file=folder/('kkt_solutions.npz' if linear else 'nlp_solutions.npz');data=np.load(file)
        residual=[];invalid=[];wrong_sign=0
        for index,(z,params,lam) in enumerate(zip(data['primal'],data['parameters'],data['dual'])):
            if linear:
                q,lo,hi,c=parameter_terms(qp,f,params)
                values=qp['A'].dot(z);grad=qp['Hdiag']*z+q+qp['A'].T.dot(lam)
            else:
                N=int(m['controller'].replace('nonlinear',''))
                objective,values,lo,hi,grad=readouts[N](z,params,lam)
            result=canonical_kkt(values,lo,hi,lam,grad)
            require_kkt(result,m['solver_statuses'][index],
                        primal_tolerance=1.01e-8 if linear else 1e-4,
                        required_status='success' if linear else 'Solve_Succeeded')
            residual.append(result['metrics']);wrong_sign+=int(np.count_nonzero(result['rows']['dual_sign_residual']))
        maxima={key:max(r[key] for r in residual) for key in residual[0]} if residual else {}
        record={'record':folder.name,'retained_candidates':len(data['primal']),
                'native_attempts':m['solver_calls'],'new_KKT_passes':len(residual),
                'maxima':maxima,'rows_with_nonzero_dual_sign_residual':wrong_sign,
                'historical_complete':m['complete'],'historical_classification_changed':False}
        records.append(record)
        arrays[folder.name]=np.asarray([[r[key] for key in residual[0]] for r in residual])
        if not linear and not m['complete']:
            # Inventory is evidence. The candidate rejected before list.append
            # does not exist in these NPZ/JSON records and cannot be recovered.
            checks=p.read(folder/'nlp_checks.json');log=source/f"{m['case']}_{m['controller']}_{m['phase']}_episode.log"
            missing.append({'record':folder.name,'attempts':m['solver_calls'],
                'retained_candidate_count':len(data['primal']),'retained_check_count':len(checks),
                'missing_candidate_step_zero_based':len(data['primal']),
                'native_last_status':m['last_native_solver_stats'],
                'native_printed_last_residuals':scalar_native_reports(log.read_text()),
                'candidate_files':sorted(x.name for x in folder.iterdir()),
                'classification':'UNRESOLVED verifier rejection; no rejected primal/dual/diagnostic vector retained',
                'no_reconstruction_attempted':True})
    np.savez_compressed(out/'retained_kkt_residuals.npz',**arrays)
    result={'records':records,'missing_rejected_candidates':missing,
            'retained_candidates':sum(r['retained_candidates'] for r in records),
            'NLP_candidates':sum(r['retained_candidates'] for r in records if 'nonlinear' in r['record']),
            'QP_candidates':sum(r['retained_candidates'] for r in records if '_linear' in r['record']),
            'all_retained_candidates_pass_corrected_KKT':True,'new_optimization_calls':0,
            'historical_classifications_changed':False}
    safe_write(out/'retained_kkt_audit.json',result);return result


TRACE=re.compile(r'^\s*(\d+)\s+(\d+)\s+([-+0-9.e]+)\s+([-+0-9.e]+)\s+(-?\d+)\s+([-+0-9.e]+)\s+(-?\d+)\s+([-+0-9.e]+)\s+(-?\d+)\s+([-+0-9.e]+)\s*(.*)$')


def qrqp_groups(text):
    groups=[]
    for line in text.splitlines():
        m=TRACE.match(line)
        if not m:continue
        fields=m.groups();iteration=int(fields[0])
        if iteration==0:groups.append([])
        if not groups:raise AssertionError('Trace does not start at iteration zero')
        groups[-1].append(dict(zip(['iteration','singular','objective_printed','primal_printed','primal_index',
                                  'dual_printed','dual_index','min_R_printed','min_R_index','last_tau','note'],
            [iteration,int(fields[1]),float(fields[2]),float(fields[3]),int(fields[4]),
             float(fields[5]),int(fields[6]),float(fields[7]),int(fields[8]),float(fields[9]),fields[10]])))
    return groups


def printed_vector(text,name):
    match=re.search(r'^Input \d+ \('+re.escape(name)+r'\): \[([^\n]*)\]',text,re.M)
    if match is None:return None
    return np.array([float(v.strip()) for v in match.group(1).split(',')])


def row_label(combined):
    """QRQP z=[decision x; A*x], in the pinned sparse shooting order."""
    if combined<326:
        if combined<246:return {'decision_state_step':combined//6,'state_component':combined%6}
        return {'decision_action_step':(combined-246)//2,'action_component':(combined-246)%2}
    row=combined-326
    if row>=880:return {'constraint_row':row,'type':'terminal_state_or_initial_equality'}
    stage,offset=divmod(row,22)
    if offset<6:kind='shooting_equality';component=offset
    elif offset<18:kind='state_lower' if offset<12 else 'state_upper';component=(offset-6)%6
    else:kind='input_lower' if offset<20 else 'input_upper';component=(offset-18)%2
    return {'constraint_row':row,'stage':stage,'type':kind,'component':component}


def singular_values(matrix):
    s=linalg.svdvals(matrix,check_finite=True)
    tol=max(matrix.shape)*np.finfo(float).eps*s[0]
    rank=int(np.count_nonzero(s>tol))
    return {'shape':list(matrix.shape),'rank':rank,'standard_rank_tolerance':float(tol),
            'largest_singular_value':float(s[0]),'smallest_singular_value':float(s[-1]),
            'condition_2':float(s[0]/s[-1]) if s[-1]>0 else np.inf,
            'sensitivity_rank_relative_1e_12':int(np.count_nonzero(s>s[0]*1e-12)),
            'coordinate_dependent':True}


def qrqp_audit(source,out):
    source,out=Path(source),Path(out);f=p.read(source/'frozen.json');ref=np.load(source/'reference.npy')
    qp=independent_program(f);A=qp['A'].toarray();H=np.diag(qp['Hdiag']);eq=qp['equality_rows']
    eqmask=np.zeros(898,dtype=bool);eqmask[eq]=True
    common={'H_eigenvalue_min':float(qp['Hdiag'].min()),'H_eigenvalue_max':float(qp['Hdiag'].max()),
            'H_condition_2':float(qp['Hdiag'].max()/qp['Hdiag'].min()),
            'equality_matrix':singular_values(A[eq]),'full_constraint_matrix':singular_values(A),
            'note':'Raw-coordinate matrix conditions are not a physical error or feasibility certificate.'}
    records=[];max_success=0
    # Parse every linear run, including successful prefixes omitted by v1's
    # complete-record iteration maxima.
    for folder in sorted((source/'runs').glob('*_linear40_*')):
        m=p.read(folder/'completion.json');text=(source/f"{m['case']}_linear40_{m['phase']}_episode.log").read_text()
        groups=qrqp_groups(text);assert len(groups)==m['solver_calls']
        successes=[g[-1]['iteration'] for g in groups if 'Converged' in g[-1]['note']]
        max_success=max([max_success,*successes])
        if m['complete']:continue
        coefficients=np.load(source/'graphs'/f"{m['case']}_{m['phase']}"/'native_graph_coefficients.npz')
        coefficient_error=max(float(np.max(np.abs(coefficients['H']-H))),float(np.max(np.abs(coefficients['A']-A))))
        assert coefficient_error<1e-10
        n=m['transitions'];history=np.load(folder/'trajectory.npz');solutions=np.load(folder/'kkt_solutions.npz')
        last_z=solutions['primal'][-1];last_lam=solutions['dual'][-1];oldp=solutions['parameters'][-1]
        failedp=np.r_[history['states'][-1],ref[np.minimum(n+np.arange(41),300)].T.reshape(-1,order='F')]
        q,lo,hi,constant=parameter_terms(qp,f,failedp)
        warm=last_z.copy();warm[246:248]+=f['hover_input']
        printed=printed_vector(text,'x0');assert printed is not None and len(printed)==326
        printed_error=float(np.max(np.abs(printed-warm)))
        # Text dumps are rounded; do not promote them to exact saved vectors.
        assert printed_error<1e-4
        g=qp['A'].dot(last_z);oldq,oldlo,oldhi,_=parameter_terms(qp,f,oldp)
        close=(~eqmask)&(np.minimum(np.abs(g-oldlo),np.abs(oldhi-g))<=1e-7)
        active=np.flatnonzero(close);C=A[np.r_[eq,active]]
        K=np.block([[H,C.T],[C,np.zeros((len(C),len(C)))]])
        eigen=linalg.eigvalsh(K,check_finite=True)
        Kcondition=float(np.max(np.abs(eigen))/np.min(np.abs(eigen)))
        last_group=groups[-1];last=last_group[-1]
        traced=[]
        for row in last_group:
            match=re.search(r'i=(\d+)',row['note'])
            if match:traced.append(int(match.group(1)))
        warm_values=qp['A'].dot(warm);warm_violation=np.maximum(lo-warm_values,warm_values-hi)
        sortedbad=np.argsort(warm_violation)[-6:][::-1]
        candidate={'record':folder.name,'solver_calls':len(groups),'last_trace':last,
            'classification':'iteration-limit cycling' if 'Max iter' in last['note'] else 'numerical failure with singular active-set trace before iteration limit',
            'native_status':'unknown','final_iterate_available':False,'final_duals_available':False,
            'printed_primal_residual_is_rounded_not_independent':True,
            'trace_tail':last_group[-8:],'traced_constraint_labels':{str(i):row_label(i) for i in sorted(set(traced))},
            'last_successful_candidate_active_bounds':len(active),'last_successful_active_constraint_matrix':singular_values(C),
            'last_successful_near_active_set_note':'Diagnostic slack threshold 1e-7; not a reconstruction of the failed solver working set.',
            'last_successful_KKT_condition_2':Kcondition,
            'archived_native_problem_matrix_reconstruction_error':coefficient_error,
            'last_successful_dual_abs_max':float(np.max(np.abs(last_lam))),
            'current_QP_parameters_reconstructed_from_recorded_state_and_reference':True,
            'last_successful_primal_is_not_failed_iterate':True,
            'native_warm_start_alias_verified_to_print_precision':True,'printed_x0_max_difference':printed_error,
            'first_input_deviations_before_alias':last_z[246:248].tolist(),
            'first_input_warm_start_after_alias':warm[246:248].tolist(),
            'input_deviation_upper':(np.asarray(f['action_high'])-f['hover_input']-1e-8).tolist(),
            'warm_start_maximum_constraint_violation':float(np.max(warm_violation)),
            'warm_start_most_violated_rows':[{'row':int(i),'label':row_label(326+int(i)),
                'violation':float(warm_violation[i])} for i in sortedbad],
            'input_dump':{'x0':printed.tolist(),'lam_x0_all_zero':bool(np.all(printed_vector(text,'lam_x0')==0)),
                          'lam_a0_all_zero':bool(np.all(printed_vector(text,'lam_a0')==0)),
                          'note':'x0 and lam_*0 are initial guesses, NOT final failed primal/dual evidence.'},
            'no_infeasibility_certificate_found':True,'no_alternative_solver_or_iteration_limit_used':True}
        records.append(candidate)
    result={'common':common,'failed_records':records,'maximum_successful_iterations_including_prefixes':max_success,
            'H2_historical_iteration_limit_label_is_incorrect':True,'new_solver_calls':0,
            'interpretation':'Warm-start inconsistency and singular traces are verified; causal attribution remains unproven without a controlled approved comparison.'}
    safe_write(out/'qrqp_audit.json',result);return result


def native_solver_wall_times(text):
    """Parse rounded IPOPT kernel wall times, not full MPC action latency."""
    result=[]
    for line in text.splitlines():
        if not re.match(r'^\s*total\s*\|',line):continue
        tokens=re.findall(r'([0-9.]+)\s*(us|ms|s)',line.split('|',1)[1])
        if len(tokens)!=4:raise AssertionError('Unexpected native timing row: '+line)
        number,unit=tokens[2]  # CPU, mean CPU, WALL, mean WALL.
        result.append(float(number)*{'us':1e-6,'ms':1e-3,'s':1}[unit])
    return np.asarray(result)


def latency_audit(source,out):
    source,out=Path(source),Path(out);records=[];aggregates={};timings={}
    for folder in sorted((source/'runs').iterdir()):
        m=p.read(folder/'completion.json');data=np.load(folder/'trajectory.npz')
        text=(source/f"{m['case']}_{m['controller']}_{m['phase']}_episode.log").read_text()
        outer=data['solver_wall'];native=native_solver_wall_times(text) if m['controller']!='linear40' else np.array([])
        if len(native):assert len(native)==m['solver_calls']
        records.append({'record':folder.name,'attempts':m['solver_calls'],
             'measured_supervised_calls':len(outer),'missing_supervised_failed_calls':m['solver_calls']-len(outer),
             'supervised_overruns':int(np.count_nonzero(outer>.02)),
             'native_kernel_times_printed':len(native), 'native_kernel_overruns_printed':int(np.count_nonzero(native>.02)),
             'native_kernel_printed_exactly_20ms_ambiguous':int(np.count_nonzero(native==.02)),
             'actual_exclusive_MPC_action_selection_latency':'not recorded; cannot recover by subtracting aggregate timings',
             'failed_native_kernel_time_printed':float(native[-1]) if len(native)>len(outer) else None})
        timings[folder.name+'_supervised']=outer;timings[folder.name+'_kernel']=native
        if len(native):
            timings[folder.name+'_unallocated_remainder']=outer-native[:len(outer)]
    for controller in ['linear40','nonlinear20','nonlinear40','nonlinear80']:
        selected=[r for r in records if f'_{controller}_' in r['record']]
        kernel=np.concatenate([timings[r['record']+'_kernel'] for r in selected])
        outer=np.concatenate([timings[r['record']+'_supervised'] for r in selected])
        def stats(a):
            return dict(zip(['median','p95','p99','maximum'],map(float,[*np.percentile(a,[50,95,99]),max(a)]))) if len(a) else None
        aggregates[controller]={'supervised_seconds':stats(outer),'native_kernel_seconds_rounded':stats(kernel),
            'supervised_overruns':int(np.count_nonzero(outer>.02)),'native_kernel_overruns_printed':int(np.count_nonzero(kernel>.02)),
            'native_kernel_printed_exactly_20ms_ambiguous':int(np.count_nonzero(kernel==.02)),
            'measured_supervised_calls':len(outer),'native_kernel_calls':len(kernel),
            'missing_supervised_failed_calls':sum(r['missing_supervised_failed_calls'] for r in selected)}
        if len(kernel):
            remainder=np.concatenate([timings[r['record']+'_unallocated_remainder'] for r in selected])
            aggregates[controller]['unallocated_remainder_seconds']=stats(remainder)
    np.savez_compressed(out/'latency_components.npz',**timings)
    result={'records':records,'aggregates':aggregates,'original_native_linear_failure_preserved':True,
            'accepted_linear_revalidation_overruns_preserved':{'calls':2400,'overruns':2,'maximum_seconds':.080281417},
            'interpretation':'IPOPT printed solver-kernel times exclude warm-start/setup/extraction. They are not measured complete action-selection times. Exact historical exclusive overhead separation is unavailable.',
            'future_timing':'CandidateGuard/TimingLedger v2 separates native selection, extraction, persistence, verification and partial snapshots, including failed calls.'}
    safe_write(out/'latency_audit.json',result);return result


def budget_audit(source,out,audit_cpu_seconds):
    source,out=Path(source),Path(out)
    prior=p.read(source/'prior_usage.json');charged=p.read(source/'usage.json');accepted=p.read(source/'study_accounting.json')
    remaining={'solver_calls':40000-prior['solver_calls']-charged['solver_calls'],
               'physics_steps':3000000-prior['physics_steps']-charged['physics_steps'],
               'cpu_seconds':14400-prior['cpu_seconds']-charged['cpu_seconds']-audit_cpu_seconds}
    actual={'solver_calls':40000-prior['solver_calls']-accepted['actual_new_solver_attempts'],
            'physics_steps':3000000-prior['physics_steps']-accepted['actual_new_physics_steps_including_replays']}
    scenarios=[]
    for name,episodes in [('targeted_H2_MPC20_two_repeats_only',2),
                           ('both_failed_linear_cases_two_repeats_only',4),
                           ('complete_four_state_four_controller_qualification_twice',32),
                           ('original_sixteen_state_four_controller_screen_once',64)]:
        calls=300*episodes;physics=6000*episodes*3
        scenarios.append({'scenario':name,'scientific_episodes':episodes,'maximum_solver_calls':calls,
            'physics_steps_including_two_replays':physics,'fits_remaining_charged_budget':calls<=remaining['solver_calls'] and physics<=remaining['physics_steps'],
            'solve_shortfall':max(0,calls-remaining['solver_calls']),
            'physics_shortfall':max(0,physics-remaining['physics_steps']),
            'authorized':False})
    result={'remaining_conservative_original_budget':remaining,'remaining_from_actual_counts_not_released':actual,
            'audit_cpu_seconds_charged_separately':audit_cpu_seconds,'new_audit_solver_calls':0,'new_audit_physics_steps':0,
            'future_scenarios':scenarios,'unused_failure_reservations_released':False,
            'additional_allocation_authorized_or_consumed':False,
            'CPU_projection':'Targeted two-episode check is on the order of tens of seconds including startup/replays; a full four-state redo is on the order of 5-10 minutes based on prior measured costs, not a guarantee. One-second/action and remaining CPU cap still apply.',
            'scope_note':'Minimum allocations assume full 300-control completion, no additional controller repeats beyond those shown, and two recorded-action replays. Do not budget by assuming failures.'}
    safe_write(out/'budget_audit.json',result);return result
