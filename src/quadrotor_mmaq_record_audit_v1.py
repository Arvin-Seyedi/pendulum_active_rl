"""Supplemental read-only audit of incomplete qualification records.

Added after encountering held-out QRQP failures. This changes no execution code,
contract, acceptance condition or trajectory. It makes unmeasured failed-call
latencies and finite prefixes explicit instead of dropping failed cases.
"""
from pathlib import Path
import re
import numpy as np


def latency_accounting(metadata, measured):
    measured=np.asarray(measured,dtype=float)
    attempts=int(metadata['solver_calls']);missing=attempts-len(measured)
    if missing<0:raise AssertionError('More measured calls than attempts')
    return {'attempted_calls':attempts,'measured_calls':len(measured),
            'unmeasured_failed_calls':missing,
            'observed_deadline_overruns':int(np.count_nonzero(measured>.02)),
            'complete_latency_accounting':missing==0,
            'limitation':None if not missing else
              'The frozen runner records elapsed action time only after select_action returns. Failed-call latency is unknown; no zero or successful latency is substituted.'}


def one_sided_complementarity(g,lo,hi,lam):
    """Read-only diagnostic: finite constraint side, independent of dual sign.

    A tiny wrong-sign upper-bound multiplier must produce a small sign residual,
    not an infinite complementarity value from multiplying the absent lower bound.
    This function is not wired into the frozen execution checker.
    """
    g,lo,hi,lam=map(np.asarray,[g,lo,hi,lam]);equal=lo==hi
    upper_only=np.isfinite(hi)&~np.isfinite(lo)
    lower_only=np.isfinite(lo)&~np.isfinite(hi)
    active=np.where(upper_only,hi,np.where(lower_only,lo,np.where(lam>=0,hi,lo)))
    mask=~equal
    product=lam[mask]*(g[mask]-active[mask])
    sign=np.where(upper_only,np.maximum(-lam,0),np.where(lower_only,np.maximum(lam,0),0))
    return {'normalized_complementarity':float(np.max(np.abs(product)/(1+np.abs(lam[mask])),initial=0)),
            'dual_sign_residual':float(np.max(sign,initial=0))}


def audit_partial_records(out,upstream):
    from src import quadrotor_pilot as p
    from src.quadrotor_mmaq_v1 import safe_write
    from src.quadrotor_verification import audit_record
    from src.quadrotor_linear_verification import independent_program,kkt_metrics,require_valid_solution
    from src.quadrotor_mmaq_verification_v1 import exact_majorant
    from src.quadrotor_mmaq_verification_v1 import independent_nlp_program,nlp_metrics,require_nlp
    from fractions import Fraction
    out=Path(out);f=p.read(out/'frozen.json');reference=np.load(out/'reference.npy')
    partial=[];latencies=[];program=independent_program(f)
    p.activate(upstream,f['config'])
    for folder in sorted((out/'runs').iterdir()):
        m=p.read(folder/'completion.json');a=np.load(folder/'trajectory.npz')
        timing=latency_accounting(m,a['solver_wall']);timing['record']=folder.name;latencies.append(timing)
        if m['complete']:continue
        n=m['transitions'];record={'record':folder.name,'complete':False,'safe_episode':False,
             'full_horizon_return':None,'transitions':n,'failure':m['failure'],'latency':timing}
        if n:
            check,_=audit_record(folder,f,reference)
            assert not check['safe_complete_no_contact']
            causal=p.read(folder/'causal_contract.json')
            for k in range(n):
                row=causal[k];assert np.array_equal(row['initial'],a['states'][k])
                cache=a['physics_states'][max(0,k*20-1),4];assert row['cache']==cache
                exact=exact_majorant(a['states'][k],row['model_thrust'],row['executed_thrust'],cache,f,m['controller']=='linear40')
                assert all(Fraction(float(v))>=e for v,e in zip(row['bound'],exact))
                assert np.max(np.abs(np.asarray(row['source_prediction'])-a['states'][k+1]))<1e-10
                assert np.all(np.abs(np.asarray(row['prediction'])-a['states'][k+1])<=np.asarray(row['bound'])+1e-10)
            if m['controller']=='linear40':
                q=np.load(folder/'kkt_solutions.npz')
                assert len(q['primal'])==n
                for par,z,lam in zip(q['parameters'],q['primal'],q['dual']):
                    require_valid_solution(True,True,kkt_metrics(program,f,par,z,lam))
            else:
                factory=lambda:p.make_environment(upstream,f['config'],np.asarray(f['cases'][m['case']]),1000)
                ctrl=p.make_controller(upstream,f['config'],factory,m['controller'])
                try:
                    functions=independent_nlp_program(ctrl,f);q=np.load(folder/'nlp_solutions.npz')
                    assert len(q['primal'])==n
                    for z,par,lam in zip(q['primal'],q['parameters'],q['dual']):
                        require_nlp(nlp_metrics(*functions,z,par,lam),'Solve_Succeeded')
                finally:ctrl.close()
            stream=np.fromfile(folder/'partial_physics.f64',dtype='<f8').reshape(-1,7)
            assert stream.shape==(20*n+1,7)
            assert stream[:,:6].tobytes()==a['physics_states'].tobytes()
            for marker in ['replay_native.json','replay_instrumented.json']:
                replay=p.read(folder/marker)
                assert all(v for k,v in replay.items() if k.endswith('bitwise'))
            record.update(prefix_independently_verified=True,physics_states=20*n+1,
                minimum_raw_margins=check['minimum_margins_by_state'],
                prefix_violations=check['violating_physics_samples'],prefix_contacts=check['contact_physics_samples'],
                partial_return=check['partial_return'],cost_reconstruction_error=check['cost_reconstruction_error'])
        else:
            record.update(prefix_independently_verified=False,physics_states=len(a['physics_states']),
                          reason='No applied transition; initial state only, never a safe episode')
        log=out/f"{m['case']}_{m['controller']}_{m['phase']}_episode.log"
        text=log.read_text()
        maxiter=[int(v) for v in re.findall(r'^\s*(\d+)\s+.*Max iter\s*$',text,re.M)]
        record.update(log_iteration_limits=maxiter,
                      interpretation='Solver iteration-limit/unknown status does not prove MPC infeasibility.' if maxiter else 'Execution failure; no infeasibility inference.')
        partial.append(record)
    demo=one_sided_complementarity(np.array([-.1]),np.array([-np.inf]),np.array([0.]),np.array([-1e-30]))
    result={'partial_records':partial,'latency_accounting':latencies,
            'unmeasured_failed_calls':sum(r['unmeasured_failed_calls'] for r in latencies),
            'all_partial_records_retained':True,'no_additional_executions':True,
            'execution_code_modified':False,'one_sided_multiplier_counterexample':demo,
            'checker_limitation':'Frozen nlp_metrics selects an absent infinite bound for a negative upper-only multiplier. This is a verified checker defect. The rejected step-54 primal/dual diagnostic vector was not saved, so this specific cause for its nonfinite diagnostic remains unconfirmed.',
            'failed_diagnostic_values_fully_retained':False}
    safe_write(out/'partial_record_verification.json',result)
    return result


def audit_all_records(root,upstream,out):
    """Adapt only the legacy verifier's (metrics, arrays) return-value API.

    The frozen post-run aggregator incorrectly expects a metrics dict. Preserve
    its source and failed attempt, and unwrap the existing read-only API here.
    No metric, threshold, model, controller, data or classification is changed.
    """
    from unittest.mock import patch
    from src import quadrotor_verification as legacy
    from src.quadrotor_mmaq_verification_v1 import verify_qualification
    from src.quadrotor_mmaq_v1 import safe_write
    native=legacy.audit_record
    def metrics_only(*args,**kwargs):return native(*args,**kwargs)[0]
    with patch.object(legacy,'audit_record',metrics_only):
        result=verify_qualification(root,upstream,out)
    partial=audit_partial_records(out,upstream)
    safe_write(Path(out)/'postrun_verification_provenance.json',{
        'version':'read_only_record_audit_v1','frozen_execution_sources_modified':False,
        'legacy_return_adapter':'Unpack (metrics, arrays) -> metrics for the frozen aggregator only.',
        'initial_aggregation_failure_log':'independent_verification.log',
        'failed_diagnostic_values_fully_retained':False,
        'new_episodes':0,'new_solver_calls':0,'new_physics_steps':0})
    return result,partial
