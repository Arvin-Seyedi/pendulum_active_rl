"""Additive E12_MMAQ_v1 orchestration. Historical producers remain unchanged."""
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import math
import resource
import shutil
import sys
import time
import numpy as np

from src import quadrotor_pilot as pilot
from src.quadrotor_mismatch_contract_v1 import (
    draw_states, state_manifest, action_menu, executed_thrust, bound_components,
    source_prediction, linear_prediction)
from src.quadrotor_verification import rk4

VERSION = 'E12_MMAQ_v1'
BASE = Path('results/benchmark_selection')
NEW_SOURCES = ['configs/experiment_12_mmaq_v1.json', 'src/quadrotor_mismatch_contract_v1.py',
               'src/quadrotor_mmaq_v1.py', 'src/quadrotor_mmaq_verification_v1.py',
               'experiments/12_mmaq_qualification.py', 'tests/test_quadrotor_mmaq_v1.py',
               'docs/experiment_12_mmaq_v1_protocol.md']


def tagged(value):
    """Lossless explicit nonfinite diagnostics; JSON itself stays standards-valid."""
    if isinstance(value, np.ndarray): return tagged(value.tolist())
    if isinstance(value, np.generic): return tagged(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return {'nonfinite_float': value.hex()}
    if isinstance(value, dict): return {str(k): tagged(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [tagged(v) for v in value]
    return value


def safe_write(path, value):
    path = Path(path); path.parent.mkdir(exist_ok=True, parents=True)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(tagged(value), indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def immutable(path, value):
    path = Path(path)
    if path.exists():
        if pilot.read(path) != tagged(value): raise RuntimeError('Frozen record changed: '+str(path))
    else: safe_write(path, value)


def historical_usage(root):
    root = Path(root)/BASE
    a = pilot.read(root/'technical_pilot/usage.json')
    b = pilot.read(root/'linear_qrqp_revalidation/usage.json')
    c = pilot.read(root/'compatibility_diagnostic/solver_performance.json')
    d = pilot.read(root/'compatibility_diagnostic/transition_performance.json')
    return {'solver_calls': a['solver_calls']+b['solver_calls']+14,
            'physics_steps': a['physics_steps']+b['physics_steps'],
            'cpu_seconds': a['cpu_seconds']+b['cpu_seconds']+c['cpu_seconds']+d['cpu_seconds']}


def preservation(root, upstream):
    root, upstream = Path(root), Path(upstream)
    old = pilot.read(root/BASE/'technical_pilot/preservation_before.json')
    count = 0
    for name,h in old['files'].items():
        if pilot.sha(root/name) != h: raise AssertionError('Historical file changed: '+name)
        count += 1
    for name,h in old['authority_files'].items():
        if pilot.sha(name) != h: raise AssertionError('Authority changed: '+name)
    accepted = {}
    for sub,marker in [('technical_pilot','review_integrity.json'),
                       ('compatibility_diagnostic','completion.json'),
                       ('linear_qrqp_revalidation','completion.json')]:
        folder = root/BASE/sub; seal = pilot.read(folder/marker)
        for name,h in seal['artifact_sha256'].items():
            if pilot.sha(folder/name) != h: raise AssertionError('Accepted artifact changed: '+str(folder/name))
        for name,h in seal['source_sha256'].items():
            if pilot.sha(root/name) != h: raise AssertionError('Accepted source changed: '+name)
        accepted[sub] = pilot.sha(folder/marker)
    f = pilot.read(root/BASE/'technical_pilot/frozen.json')
    for name,h in f['upstream_files'].items():
        if pilot.sha(upstream/name) != h: raise AssertionError('Upstream changed: '+name)
    iq = pilot.read(root/BASE/'technical_pilot/iql_preservation.json')
    return {'historical_files':count, 'authority_files':len(old['authority_files']),
            'upstream_files':len(f['upstream_files']), 'accepted_seals':accepted,
            'iql_preservation_record_sha256':pilot.sha(root/BASE/'technical_pilot/iql_preservation.json'),
            'iql_record':iq}


def prepare(root, upstream, out, design):
    root, out = Path(root), Path(out); out.mkdir(exist_ok=True, parents=True)
    if (out/'phase_a_freeze.json').exists():
        check_seal(root,out); return pilot.read(out/'phase_a_freeze.json')
    saved = preservation(root,upstream)
    text = Path(design).read_text()
    for required in ['120001', '0, 4, 8 and 12', '0.05', 'PCG64']:
        if required not in text: raise AssertionError('Original design provenance missing: '+required)
    approved = out/'recovered_approved_design.md'
    if approved.exists(): assert approved.read_text()==text
    else: approved.write_text(text)
    config = pilot.read(root/'configs/experiment_12_mmaq_v1.json')
    f = pilot.read(root/BASE/'technical_pilot/frozen.json')
    assert config['upstream_commit']==f['config']['upstream_commit']
    assert config['original_symbolic_gate']=={'threshold':.001,'historical_status':'FAILED'}
    screening, draws = draw_states(config['screening_seed'],16,config['low'],config['high'])
    manifest = state_manifest(screening,draws,config['screening_seed'])
    manifest.update(probe_indices=config['probe_indices'],action_offsets_N=action_menu(f),
                    draw_order=config['draw_order'],menu_order=config['menu_order'],evaluated=False,
                    ordering_note='The original design specified seed/ranges but no draw loop. Row-major physical-state order is frozen here before any new outcome.')
    immutable(out/'screening_manifest.json',manifest)
    np.save(out/'screening_states.npy',screening)
    shutil.copyfile(root/BASE/'technical_pilot/reference.npy',out/'reference.npy')
    shutil.copyfile(root/BASE/'technical_pilot/environment.lock.txt',out/'environment.lock.txt')
    immutable(out/'accepted_frozen.json',f)
    immutable(out/'protocol.json',config)
    immutable(out/'preservation_start.json',saved)
    immutable(out/'prior_usage.json',historical_usage(root))
    freeze = {'version':VERSION,'source_sha256':{name:pilot.sha(root/name) for name in NEW_SOURCES},
              'artifact_sha256':{name:pilot.sha(out/name) for name in ['protocol.json','accepted_frozen.json',
                'screening_manifest.json','screening_states.npy','reference.npy','environment.lock.txt',
                'recovered_approved_design.md','prior_usage.json','preservation_start.json']},
              'original_agreement_gate':'FAILED', 'heldout_outcomes_generated':False,
              'run_screening':False,'run_probes':False,'additional_allocation_authorized':False}
    immutable(out/'phase_a_freeze.json',freeze)
    return freeze


def check_seal(root,out):
    root,out = Path(root),Path(out); seal=pilot.read(out/'phase_a_freeze.json')
    for name,h in seal['source_sha256'].items(): assert pilot.sha(root/name)==h,name
    for name,h in seal['artifact_sha256'].items(): assert pilot.sha(out/name)==h,name


def archived_validation(root,out):
    """Only historical transitions. All inputs to a bound precede its successor."""
    root,out = Path(root),Path(out); check_seal(root,out)
    f=pilot.read(out/'accepted_frozen.json'); tolerance=1e-10
    records=[]; all_bound=[]; max_closure=0.; max_motor=0.; failures=[]
    folders=[]
    for sub in ['technical_pilot','linear_qrqp_revalidation']:
        for folder in sorted((root/BASE/sub/'runs').iterdir()):
            m=pilot.read(folder/'completion.json')
            if m['complete'] and (m['phase']=='original' or (m['phase']=='refine' and m['frequency']==2000)):
                folders.append(folder)
    for folder in folders:
        m=pilot.read(folder/'completion.json'); a=np.load(folder/'trajectory.npz'); N=m['frequency']//50
        linear=m['controller']=='linear40'; bounds=[]; actual=[]; closure=[]; reserves=[]
        for k in range(m['transitions']):
            x=a['states'][k]; u=a['actions'][k]
            # p[k*N-1] is past information; at reset the cache is the reset angle.
            cache=a['physics_states'][max(0,k*N-1),4]
            b=bound_components(x,u,cache,f,linear,m['frequency'])
            known=b['executed_thrust']; rpm=a['rpms'][k]
            measured=f['KF']*np.array([rpm[0]**2+rpm[3]**2,rpm[1]**2+rpm[2]**2])
            motor=float(np.max(np.abs(known-measured))); max_motor=max(max_motor,motor)
            predicted=linear_prediction(x,b['model_thrust'],f) if linear else rk4(x,b['model_thrust'],f)
            y=source_prediction(x,known,cache,f,m['frequency'])
            discrepancy=np.abs(predicted-a['states'][k+1]); error=float(np.max(np.abs(y-a['states'][k+1])))
            max_closure=max(max_closure,error)
            if motor>1e-12 or error>tolerance or np.any(discrepancy>b['bound']+tolerance):
                failures.append({'record':folder.name,'step':k,'closure':error,'motor':motor,
                                 'excess':(discrepancy-b['bound']).tolist()})
            if linear:
                Ad_prediction=a['predicted_states'][k,:,1]
                # Native linear predictions are deviations about X_EQ, not raw states.
                # X_EQ is zero in this model; independently check graph records separately.
                if np.max(np.abs(predicted-Ad_prediction))>1e-10:
                    failures.append({'record':folder.name,'step':k,'linear_prediction_mismatch':float(np.max(np.abs(predicted-Ad_prediction)))})
            margin=np.minimum(predicted-f['state_low'],np.asarray(f['state_high'])-predicted)
            reserves.append(margin-b['bound']); bounds.append(b['bound']); actual.append(discrepancy);closure.append(error)
        bounds=np.asarray(bounds); actual=np.asarray(actual); reserves=np.asarray(reserves)
        all_bound.append(bounds)
        records.append({'record':folder.name,'intervals':m['transitions'],'frequency':m['frequency'],
            'bound_maximum_by_component':bounds.max(0).tolist(),'error_maximum_by_component':actual.max(0).tolist(),
            'minimum_reserve_by_component':reserves.min(0).tolist(),'source_closure_maximum':max(closure)})
    report={'passed':not failures,'records':records,'intervals':sum(r['intervals'] for r in records),
            'maximum_source_closure':max_closure,'maximum_motor_error':max_motor,'failures':failures,
            'bound_maximum_by_component':np.concatenate(all_bound).max(0).tolist(),
            'new_episodes':0,'new_physics_steps':0,'new_solver_calls':0,
            'proof_scope':'Analytic real-recurrence majorant; numerical closure to Bullet empirically checked only.',
            'original_agreement_gate':'FAILED'}
    immutable(out/'archive_contract_validation.json',report)
    return report


def graph_preflight(root,upstream,out):
    """Construct mathematical graphs and evaluate archived primals; zero solves."""
    from src.quadrotor_linear_revalidation import create_adapted_controller
    from src.quadrotor_mmaq_verification_v1 import independent_nlp_program,nlp_metrics
    out=Path(out);f=pilot.read(out/'accepted_frozen.json');reference=np.load(out/'reference.npy')
    pilot.activate(upstream,f['config']);reports=[]
    factory=lambda:pilot.make_environment(upstream,f['config'],np.asarray(f['cases']['T0']),1000)
    ctrl,_,_,linear_report=create_adapted_controller(upstream,f['config'],factory,f,out/'preflight_linear_graph')
    ctrl.close();reports.append({'controller':'linear40','passed':True,'graph':linear_report})
    for key in ['nonlinear20','nonlinear40','nonlinear80']:
        ctrl=pilot.make_controller(upstream,f['config'],factory,key)
        try:
            fn,native=independent_nlp_program(ctrl,f);errors=[]
            for case in ['T0','T1','T2','T3']:
                folder=Path(root)/BASE/'technical_pilot/runs'/f'{case}_{key}_1000_original'
                a=np.load(folder/'trajectory.npz')
                for k in [0,149,299]:
                    z=np.r_[a['predicted_states'][k].reshape(-1,order='F'),a['predicted_inputs'][k].reshape(-1,order='F')]
                    ref=reference[np.minimum(k+np.arange(ctrl.T+1),300)].T
                    p=np.r_[a['states'][k],ref.reshape(-1,order='F')]
                    m=nlp_metrics(fn,native,z,p,np.zeros(ctrl.opti_dict['opti'].ng))
                    errors.append(m['program_reconstruction_error'])
            reports.append({'controller':key,'passed':max(errors)<1e-10,
                            'archived_primal_checks':len(errors),'maximum_reconstruction_error':max(errors)})
        finally:ctrl.close()
    report={'passed':all(r['passed'] for r in reports),'controllers':reports,'new_solver_calls':0,'new_episodes':0}
    safe_write(out/'graph_preflight.json',report)
    return report


def preflight(root,upstream,out):
    from src.quadrotor_mmaq_verification_v1 import verify_archived_bounds
    out=Path(out);check_seal(root,out)
    if (out/'preflight.json').exists():return pilot.read(out/'preflight.json')
    archive=archived_validation(root,out)
    exact=verify_archived_bounds(root,out);immutable(out/'independent_rational_validation.json',exact)
    graphs=graph_preflight(root,upstream,out)
    c=pilot.read(out/'protocol.json');manifest=pilot.read(out/'screening_manifest.json')
    bitgen=np.random.PCG64(c['screening_seed']);raw=bitgen.random_raw(16*6)
    unit=(raw>>np.uint64(11)).astype(float)*(2.**-53)
    independent=np.asarray(c['low'])+(np.asarray(c['high'])-c['low'])*unit.reshape(16,6)
    sampling=independent.tobytes()==np.load(out/'screening_states.npy').tobytes()
    test=pilot.read(out/'failure_logging_tests.json')
    passed=archive['passed'] and exact['passed'] and graphs['passed'] and sampling and test['passed']
    budget_available(root,out,9600,576000)
    report={'passed':bool(passed),'archive_contract':archive['passed'],'independent_exact_bounds':exact['passed'],
            'graph_reconstruction':graphs['passed'],'sampling_independent_raw_bits':sampling,
            'failure_logging_tests':test['passed'],'screening_cases_sealed':len(manifest['states']),
            'screening_cases_evaluated':0,'new_episodes':0,'new_solver_calls':0,'new_physics_steps':0,
            'original_gate':'FAILED','authorization':'Only H0-H3, four controllers, original/repeat if passed.',
            'preflight_evidence_sha256':{name:pilot.sha(out/name) for name in ['archive_contract_validation.json',
                'independent_rational_validation.json','graph_preflight.json','failure_logging_tests.json']}}
    immutable(out/'preflight.json',report)
    return report


def generate_heldout(root,out):
    check_seal(root,out); out=Path(out)
    pre=pilot.read(out/'preflight.json')
    if not pre['passed']: raise RuntimeError('Phase A did not pass; no heldout generation/execution')
    c=pilot.read(out/'protocol.json'); f=pilot.read(out/'accepted_frozen.json')
    screen=np.load(out/'screening_states.npy')
    x,draws=draw_states(c['heldout_seed'],4,c['low'],c['high'],[*screen,*f['cases'].values()])
    immutable(out/'heldout_manifest.json',state_manifest(x,draws,c['heldout_seed']))
    if (out/'heldout_states.npy').exists(): assert np.array_equal(np.load(out/'heldout_states.npy'),x)
    else: np.save(out/'heldout_states.npy',x)
    f['cases'].update({f'H{i}':row.tolist() for i,row in enumerate(x)})
    immutable(out/'frozen.json',f)
    immutable(out/'heldout_seal.json',{'phase_a_sha256':pilot.sha(out/'phase_a_freeze.json'),
              'preflight_sha256':pilot.sha(out/'preflight.json'),
              'sha256':{name:pilot.sha(out/name) for name in ['heldout_manifest.json','heldout_states.npy','frozen.json']}})
    return x


def budget_available(root,out,solves=0,physics=0):
    out=Path(out); c=pilot.read(out/'protocol.json')['budget']; old=pilot.read(out/'prior_usage.json')
    new=pilot.read(out/'usage.json') if (out/'usage.json').exists() else {'solver_calls':0,'physics_steps':0,'cpu_seconds':0.}
    for key,extra,limit in [('solver_calls',solves,'maximum_cumulative_solver_calls'),
                            ('physics_steps',physics,'maximum_cumulative_physics_steps')]:
        if old[key]+new[key]+extra>c[limit]: raise RuntimeError('Original remaining budget exhausted: '+key)
    if old['cpu_seconds']+new['cpu_seconds']>=c['maximum_cumulative_cpu_seconds']:
        raise RuntimeError('Original CPU budget exhausted')
    return old,new


@contextmanager
def supervised_episode(root,upstream,out,case,controller,phase):
    """Scoped wrappers only. The upstream mathematical programs stay intact."""
    from src import quadrotor_linear_revalidation as linear
    out=Path(out); f=pilot.read(out/'frozen.json')
    folder=out/'runs'/f'{case}_{controller}_1000_{phase}'
    real_run=pilot.run_episode; real_capture=pilot.capture_physics
    captures={}; evaluations=[]

    @contextmanager
    def capture(env):
        import pybullet as bullet
        with real_capture(env) as data:
            captures['data']=data
            original_step=bullet.stepSimulation
            # Append-only raw prefix survives a process-level interruption. This
            # wrapper delegates to the accepted read-only capture first; it never
            # refreshes caches or changes Bullet forces/timing.
            with (folder/'partial_physics.f64').open('xb',buffering=0) as stream:
                stream.write(np.r_[data['states'][0],data['contacts'][0]].astype('<f8').tobytes())
                def step(*a,**kw):
                    result=original_step(*a,**kw)
                    stream.write(np.r_[data['states'][-1],data['contacts'][-1]].astype('<f8').tobytes())
                    return result
                with patch.object(bullet,'stepSimulation',step):yield data

    def wrapped_run(*args,**kwargs):
        creator=pilot.make_controller
        import casadi as cas
        real_solve=cas.Opti.solve; programs={};solutions=[];metrics_history=[]
        def create(*a,**kw):
            ctrl=creator(*a,**kw); native_select=ctrl.select_action
            if controller!='linear40':
                from src.quadrotor_mmaq_verification_v1 import independent_nlp_program
                programs['ctrl']=ctrl
                programs['functions']=independent_nlp_program(ctrl,f)
            def select(obs,info):
                k=len(evaluations); data=captures['data']; x=np.asarray(data['states'][-1]).copy()
                cache=data['states'][-2][4] if len(data['states'])>1 else x[4]
                safe_write(folder/'progress.json',{'status':'running','solver_attempts':k+1,
                    'completed_physics_steps':len(data['states'])-1,'applied_transitions':(len(data['states'])-1)//20})
                action=np.asarray(native_select(obs,info)).copy()
                stats=ctrl.opti_dict['opti'].stats()
                if not stats.get('success') or not np.isfinite(action).all():
                    safe_write(folder/'rejected_action.json',{'status':'failed','step':k,'action':action,'stats':stats})
                    raise RuntimeError('Nonfinite/unsuccessful output; no fallback action applied')
                if controller!='linear40' and stats.get('return_status')!='Solve_Succeeded':
                    raise RuntimeError('NLP did not satisfy frozen strict termination criterion')
                b=bound_components(x,action,cache,f,controller=='linear40')
                predicted=linear_prediction(x,b['model_thrust'],f) if controller=='linear40' else rk4(x,b['model_thrust'],f)
                source=source_prediction(x,b['executed_thrust'],cache,f)
                reserve=np.minimum(predicted-f['state_low'],np.asarray(f['state_high'])-predicted)-b['bound']
                evaluations.append({'step':k,'initial':x,'cache':cache,'requested':action,'prediction':predicted,
                    'source_prediction':source,'reserve':reserve,**b})
                safe_write(folder/'causal_contract.json',evaluations)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>f['config']['maximum_memory_bytes']:
                    raise MemoryError('Original RSS budget exceeded')
                return action
            ctrl.select_action=select
            return ctrl
        def solve(opti,*a,**kw):
            sol=real_solve(opti,*a,**kw)
            if programs and opti is programs['ctrl'].opti_dict['opti']:
                from src.quadrotor_mmaq_verification_v1 import nlp_metrics,require_nlp
                z=np.asarray(sol.value(opti.x)).reshape(-1);p=np.asarray(sol.value(opti.p)).reshape(-1)
                lam=np.asarray(sol.value(opti.lam_g)).reshape(-1)
                m=nlp_metrics(*programs['functions'],z,p,lam)
                require_nlp(m,opti.stats()['return_status'])
                solutions.append((z,p,lam));metrics_history.append(m)
                safe_write(folder/'nlp_checks.json',metrics_history)
            return sol
        with patch.object(pilot,'make_controller',create),patch.object(pilot,'capture_physics',capture),patch.object(cas.Opti,'solve',solve):
            result=real_run(*args,**kwargs)
        if solutions:
            z,p,lam=map(np.asarray,zip(*solutions))
            np.savez_compressed(folder/'nlp_solutions.npz',primal=z,parameters=p,dual=lam)
        return result
    with patch.object(pilot,'write',safe_write),patch.object(pilot,'run_episode',wrapped_run):
        yield linear


def episode(root,upstream,out,case,controller,phase):
    out=Path(out);check_seal(root,out)
    if case not in [f'H{i}' for i in range(4)]: raise ValueError('Only four heldout states permitted')
    if controller not in ['linear40','nonlinear20','nonlinear40','nonlinear80']: raise ValueError(controller)
    if phase not in ['original','repeat']: raise ValueError(phase)
    folder=out/'runs'/f'{case}_{controller}_1000_{phase}'
    if (folder/'qualification_execution.json').exists(): return pilot.read(folder/'completion.json')
    if folder.exists(): raise RuntimeError('Partial run retained; automatic overwrite prohibited')
    with supervised_episode(root,upstream,out,case,controller,phase) as linear:
        if controller=='linear40': result=linear.episode(upstream,out,case,phase)
        else: result=pilot.run_episode(root,upstream,out,case,controller,phase,1000)
    result.update(controller_variant='linear40_qrqp_v1' if controller=='linear40' else controller,
                  execution_status='complete' if result['complete'] else 'failed_or_incomplete')
    safe_write(folder/'completion.json',result)
    if controller=='linear40':
        safe_write(folder/'revalidation_complete.json',{'completion_sha256':pilot.sha(folder/'completion.json'),
                                                       'completed':result['complete']})
    safe_write(folder/'qualification_execution.json',{'complete':result['complete'],
                'completion_sha256':pilot.sha(folder/'completion.json'),'no_fallback_applied':True})
    return result
