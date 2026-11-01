"""E10 pilot orchestration, strictly bounded to five states and 24 solves."""

import io
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time
import traceback
import unittest

import numpy as np
import scipy

from src.counterfactual import assert_bitwise_equal, write_csv
from src.horizon_experiment import (
    _assert_hashes, immutable_json, load_npz, read_json, save_npz, sha256, write_json,
)
from src.horizon_study import exclusive
from src.policy_improvement import (
    PILOT_IDS, OPTIMIZATION_IDS, OUTPUT, assess, controller_bank, derive_lqr,
    evaluate_fixed, freeze_selection, measure, paired, safe_oracle,
)
from src.recoverable_pd import case_identity, compare_archive, load_benchmark
from src.recoverability import (
    candidate_diagnostics, evaluate_controls, pack, replay_and_measure,
    solve_attempt, unpack, validate_execution,
)
from src.recoverability_experiment import cached_evaluation, protected_manifest

CONFIG='configs/experiment_10_pilot.json'
SOURCES=[CONFIG,'requirements_experiment_10.txt','src/policy_improvement.py',
         'src/policy_improvement_experiment.py','src/policy_improvement_audit.py',
         'experiments/10_safe_policy_improvement.py','tests/test_policy_improvement.py']
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def settings(root):
    config=read_json(root/CONFIG)
    assert config['authorized_phase']=='five_case_technical_pilot_only'
    assert config['pilot_ids']==PILOT_IDS and config['optimization_pilot_ids']==OPTIMIZATION_IDS
    assert config['output']==OUTPUT and config['horizon']==200 and config['dt']==.05
    assert config['discount']==.99 and config['torque_limit']==2
    assert config['starts']==['PD','zero','random_blocks']
    assert config['solve_repetitions']==['original','repeat']
    assert config['candidate_modes']==['open_loop','tracking']
    assert config['seeds']=={'environment':101005,'random_start':101004}
    assert config['evaluate_every_finite_original_and_repeat_candidate']
    assert not any(config[k] for k in ('full_controller_benchmark_authorized','remaining_optimization_states_authorized','training_data_use'))
    return config,root/OUTPUT


def worker(root,folder):
    root=Path(root).resolve(); folder=Path(folder).resolve()
    config,output=settings(root); request=read_json(folder/'request.json')
    sid=request['state_id']
    assert sid in OPTIMIZATION_IDS and request['authorized_ids']==OPTIMIZATION_IDS
    assert folder.parent.parent==output/f'state_{sid:04d}'/'optimization'
    assert folder.name in ('original','repeat') and folder.parent.name in config['starts']
    assert request['start']==folder.parent.name and request['phase']==folder.name
    assert request['selection_sha256']==sha256(output/'selection.json')
    assert request['E10_config_sha256']==sha256(root/CONFIG)
    assert request['E07_config_sha256']==sha256(root/config['optimization_config'])
    benchmark=load_benchmark(root/config['benchmark']); identity=case_identity(benchmark,sid)
    initial=np.array(request['initial'],dtype=np.float64)
    assert initial.tobytes().hex()==identity['initial_raw_bytes_hex']==request['initial_raw_bytes_hex']
    assert request['initial_hex']==identity['initial_hex'] and not identity['archived_unsafe']
    selection=read_json(output/'selection.json')
    assert sid in selection['optimization_ids_sorted'] and selection['pilot_ids']==PILOT_IDS
    guess=load_npz(folder/'initial_guess.npz')['decision']
    assert guess.shape==(600,) and np.isfinite(guess).all()
    solve_attempt(folder,read_json(root/config['optimization_config']),initial,guess)


def run_attempt(root,folder,request,guess,config):
    folder.mkdir(parents=True,exist_ok=True)
    immutable_json(folder/'request.json',request)
    if (folder/'initial_guess.npz').exists():
        assert_bitwise_equal(load_npz(folder/'initial_guess.npz'),{'decision':guess})
    else:save_npz(folder/'initial_guess.npz',{'decision':guess})
    old=read_json(root/config['optimization_config'])
    if not (folder/'attempt.json').exists():
        if (folder/'solver.log').exists(): raise RuntimeError(f'Interrupted solver output retained; inspect before resume: {folder}')
        env=dict(os.environ)
        for name in THREADS:env[name]='1'
        started=time.perf_counter()
        with (folder/'solver.log').open('w') as stream:
            process=subprocess.Popen([sys.executable,str(root/'experiments/10_safe_policy_improvement.py'),
                                      '--worker',str(folder)],cwd=folder,env=env,stdout=stream,
                                     stderr=subprocess.STDOUT,start_new_session=True)
            timeout=False
            try:process.wait(timeout=old['watchdog_seconds'])
            except subprocess.TimeoutExpired:
                timeout=True;os.killpg(process.pid,signal.SIGKILL);process.wait()
        write_json(folder/'attempt.json',{'watchdog_timeout':timeout,'exit_code':process.returncode,
                   'process_wall_seconds':time.perf_counter()-started,
                   'technical_failure':bool(process.returncode!=0 and not timeout),
                   'thread_environment':{name:env[name] for name in THREADS}})
    attempt=read_json(folder/'attempt.json')
    target=folder/'candidate.npz'
    if not target.exists():target=folder/'last_iterate.npz'
    vector=load_npz(target)['decision'] if target.exists() else None
    available=vector is not None and vector.shape==(600,) and bool(np.isfinite(vector).all())
    result=dict(attempt,candidate_available=available,candidate_source=target.name if target.exists() else None)
    if (folder/'solver_result.json').exists():result['solver']=read_json(folder/'solver_result.json')
    if available:result['diagnostics']=candidate_diagnostics(vector,request['initial'])
    immutable_json(folder/'candidate_review.json',result)
    return vector if available else None,result


def cached_controller(folder,initial,seed,controller,config):
    folder.mkdir(parents=True,exist_ok=True)
    immutable_json(folder/'controller.json',controller)
    for phase in ('original','policy_repeat'):
        npz='rollout.npz' if phase=='original' else 'policy_repeat.npz'
        if not (folder/npz).exists():
            a,c,d=evaluate_fixed(initial,seed,controller)
            save_npz(folder/npz,a);write_json(folder/f'{phase}_costs.json',c)
            write_json(folder/f'{phase}_diagnostics.json',d)
    a=load_npz(folder/'rollout.npz');validate_execution(a,initial)
    assert_bitwise_equal(a,load_npz(folder/'policy_repeat.npz'))
    assert read_json(folder/'original_diagnostics.json')==read_json(folder/'policy_repeat_diagnostics.json')
    if not (folder/'replay.npz').exists():
        replay,c,metrics=replay_and_measure(a,initial,seed,config)
        save_npz(folder/'replay.npz',replay);write_json(folder/'replay_costs.json',c)
        write_json(folder/'metrics.json',dict(metrics,**measure(a,config)))
    assert_bitwise_equal(a,load_npz(folder/'replay.npz'))
    return a,read_json(folder/'metrics.json')


def original_baseline(root,sid,config):
    start=(sid//64)*64;end=min(start+63,6560)
    folder=root/config['baseline_archive']/f'baseline/chunk_{start:04d}_{end:04d}'
    ids=load_npz(folder/'identity_and_legacy.npz')['grid_ids']
    k=int(np.flatnonzero(ids==sid)[0])
    return {name:value[[k]] for name,value in load_npz(folder/'original.npz').items()}


def optimization_case(root,output,sid,initial,seed,pd,config):
    assert sid in OPTIMIZATION_IDS
    old=read_json(root/config['optimization_config'])
    folder=output/f'state_{sid:04d}'/'optimization';folder.mkdir(exist_ok=True)
    immutable_json(folder/'unchanged_E07_configuration.json',old)
    baseline=measure(pd,config);records=[];repetitions=[]
    for start_index,name in enumerate(config['starts']):
        guessfile=folder/f'initialization_{name}.npz'
        if not guessfile.exists():
            if name=='PD':a=pd
            else:
                u=np.zeros(200) if name=='zero' else np.repeat(np.random.default_rng(
                    np.random.SeedSequence([101004,sid])).uniform(-2,2,size=10),20)
                a,c=evaluate_controls(initial,seed,u)
                write_json(folder/f'initialization_{name}_costs.json',c)
            save_npz(guessfile,a)
        guessarrays=load_npz(guessfile);validate_execution(guessarrays,initial)
        if name=='PD':assert_bitwise_equal(guessarrays,pd)
        guess=pack(guessarrays['raw_states'][0],guessarrays['actions'][0,:,0])
        vectors=[];attempts=[]
        for phase in config['solve_repetitions']:
            target=folder/name/phase
            request={'state_id':sid,'authorized_ids':OPTIMIZATION_IDS,'initial':initial.tolist(),
                     'initial_hex':[float(v).hex() for v in initial],'initial_raw_bytes_hex':initial.tobytes().hex(),
                     'start':name,'phase':phase,'selection_sha256':sha256(output/'selection.json'),
                     'E10_config_sha256':sha256(root/CONFIG),'E07_config_sha256':sha256(root/config['optimization_config'])}
            vector,attempt=run_attempt(root,target,request,guess,config)
            vectors.append(vector);attempts.append(attempt)
            status=attempt.get('solver',{}).get('stats',{}).get('return_status','missing_solver_result')
            if vector is None:
                records.append({'state_id':sid,'start':name,'start_index':start_index,'phase':phase,
                                'candidate_available':False,'solver_status':status})
                continue
            states,actions=unpack(vector,initial,200)
            for mode in config['candidate_modes']:
                arrays,basic=cached_evaluation(target/mode,initial,seed,actions,old,
                                                states if mode=='tracking' else None)
                metrics=measure(arrays,config)
                record=paired({'state_id':sid,'start':name,'start_index':start_index,'phase':phase,'mode':mode,
                            'candidate_available':True,'solver_status':status,
                            'solver_success':bool(attempt.get('solver',{}).get('stats',{}).get('success',False)),
                            **attempt['diagnostics'],**basic,**metrics},baseline)
                write_json(target/mode/'extended_metrics.json',record);records.append(record)
        status=[a.get('solver',{}).get('stats',{}).get('return_status','missing_solver_result') for a in attempts]
        same_available=(vectors[0] is None)==(vectors[1] is None)
        identical=same_available and (vectors[0] is None or vectors[0].tobytes()==vectors[1].tobytes())
        check={'state_id':sid,'start':name,'same_status':status[0]==status[1],
               'decision_bitwise':identical,'same_candidate_availability':same_available,
               'maximum_decision_difference':None if vectors[0] is None or vectors[1] is None else float(np.max(np.abs(vectors[0]-vectors[1]))),
               'timeouts':sum(a['watchdog_timeout'] for a in attempts),
               'technical_failures':sum(a['technical_failure'] for a in attempts)}
        check['passed']=bool(check['same_status'] and identical and not check['timeouts'] and not check['technical_failures'])
        immutable_json(folder/name/'reproducibility.json',check);repetitions.append(check)
        if identical and vectors[0] is not None:
            for mode in config['candidate_modes']:
                assert_bitwise_equal(load_npz(folder/name/'original'/mode/'rollout.npz'),
                                     load_npz(folder/name/'repeat'/mode/'rollout.npz'))
        print(f'Grid {sid}, {name}: {status}; paired decision bitwise={identical}',flush=True)
    return {'state_id':sid,'records':records,'repetitions':repetitions,
            'oracles':[safe_oracle(records,baseline,mode) for mode in config['candidate_modes']],
            'passed':all(r['passed'] for r in repetitions)}


def verify_completed(root):
    root=Path(root).resolve();_,output=settings(root)
    done=read_json(output/'completion.json')
    assert done['passed'] and done['pilot_ids']==PILOT_IDS
    _assert_hashes(output,done['artifacts_sha256'])
    _assert_hashes(root,read_json(output/'source_hashes.json'))
    _assert_hashes(root,read_json(output/'preservation_before.json')['sha256'])
    assert sha256(root/'docs/experiment_10_pilot_report.md')==done['report_sha256']
    print('Completed E10 pilot verified; no simulations or solves rerun.',flush=True)


def run_pilot(root,resume=False,snapshot=None):
    root=Path(root).resolve();config,output=settings(root)
    if output.exists() and not resume:raise FileExistsError('Frozen selection/pilot exists; inspect and use --resume.')
    output.mkdir(parents=True,exist_ok=True)
    with exclusive(output):
        if (output/'completion.json').exists():verify_completed(root);return
        try:_run(root,output,config,snapshot)
        except BaseException as exc:
            folder=output/'failed_checks';folder.mkdir(exist_ok=True)
            write_json(folder/f'failure_{len(list(folder.glob("*.json")))+1:03d}.json',
                       {'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
            raise


def _run(root,output,config,snapshot):
    selection=freeze_selection(root,config)
    assert read_json(output/'selection_frozen.json')['selection_sha256']==sha256(output/'selection.json')
    immutable_json(output/'protocol.json',config)
    if not (output/'preservation_before.json').exists():
        write_json(output/'preservation_before.json',read_json(snapshot) if snapshot else protected_manifest(root,output))
    before=read_json(output/'preservation_before.json');_assert_hashes(root,before['sha256'])
    prior=list(read_json(root/config['baseline_archive']/'source_hashes.json'))
    hashes={name:sha256(root/name) for name in sorted(set(prior+SOURCES))}
    immutable_json(output/'source_hashes.json',hashes)
    for name in hashes:
        p=output/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.write_bytes((root/name).read_bytes())
        assert sha256(p)==hashes[name]
    immutable_json(output/'metadata.json',{'python':sys.version,'executable':sys.executable,
                   'platform':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,
                   'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                   'subset_frozen_before_evaluations':True,'full_benchmark_executed':False})
    old_baselines=read_json(root/config['baseline_archive']/'baseline_summary.json')
    full_cost=-float(np.mean([r['metrics']['returns'] for r in old_baselines if not r['identity']['archived_unsafe']]))
    pilot_cost=-float(np.mean([old_baselines[sid]['metrics']['returns'] for sid in PILOT_IDS]))
    immutable_json(output/'practical_thresholds_frozen.json',{
                   'basis':'Existing E09 PD data only, before new controller evaluations',
                   'formula':'max(0.10, 0.10 * mean PD cost on the comparison set)',
                   'full_benchmark_mean_PD_cost':full_cost,'full_benchmark_return_threshold':max(.1,.1*full_cost),
                   'pilot_mean_PD_cost':pilot_cost,'pilot_return_threshold':max(.1,.1*pilot_cost),
                   'worst_case_basis':'minimum per-state paired return difference',
                   'remaining_criteria':config['criteria']})
    lqr=derive_lqr(config);immutable_json(output/'lqr.json',lqr)
    controllers=controller_bank(lqr);immutable_json(output/'controllers.json',controllers)
    if not (output/'technical_tests.json').exists():
        suite=unittest.TestSuite()
        for pattern in ('test_policy_improvement.py','test_recoverability*.py'):
            suite.addTests(unittest.defaultTestLoader.discover(str(root/'tests'),pattern=pattern))
        stream=io.StringIO();result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        (output/'technical_tests.txt').write_text(stream.getvalue())
        write_json(output/'technical_tests.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,
                   'failures':len(result.failures),'errors':len(result.errors),'simulator_steps':0,'solver_attempts':0})
    assert read_json(output/'technical_tests.json')['passed']
    benchmark=load_benchmark(root/config['benchmark']);fixed=[];baselines={};optimizations=[]
    for sid in PILOT_IDS:
        folder=output/f'state_{sid:04d}';folder.mkdir(exist_ok=True)
        identity=case_identity(benchmark,sid);assert not identity['archived_unsafe']
        immutable_json(folder/'identity.json',identity)
        initial=np.array(identity['initial'],dtype=np.float64);seed=101005+sid
        state_records=[];pd=None;pd_diagnostics=None
        for controller in controllers:
            a,m=cached_controller(folder/'controllers'/controller['name'],initial,seed,controller,config)
            state_records.append(dict(state_id=sid,controller=controller['name'],**m))
            if controller['name']=='PD_8_2':
                pd=a;pd_diagnostics=read_json(folder/'controllers/PD_8_2/original_diagnostics.json')
                assert_bitwise_equal(pd,original_baseline(root,sid,config))
        baseline=next(r for r in state_records if r['controller']=='PD_8_2');baselines[str(sid)]=baseline
        comparison=compare_archive(identity,pd,pd_diagnostics)
        assert not comparison['archived_vs_legacy_label_disagreement'] and not comparison['archived_vs_raw_label_disagreement']
        assert comparison['legacy_margin_difference']==0
        write_json(folder/'PD_parity.json',dict(comparison,E09_saved_arrays_bitwise=True))
        fixed.extend(paired(r,baseline) for r in state_records)
        print(f'Grid {sid}: evaluated all ten fixed controllers; PD matches E09 bitwise.',flush=True)
        if sid in OPTIMIZATION_IDS:
            optimizations.append(optimization_case(root,output,sid,initial,seed,pd,config))
    summary={'pilot_ids':PILOT_IDS,'optimization_pilot_ids':OPTIMIZATION_IDS,'lqr':lqr,
             'practical_thresholds':read_json(output/'practical_thresholds_frozen.json'),
             'fixed_records':fixed,'baselines':baselines,'optimizations':optimizations,
             'fixed_assessments':{c['name']:assess([next(r for r in fixed if r['state_id']==sid and r['controller']==c['name'])
                                                  for sid in PILOT_IDS],[baselines[str(sid)] for sid in PILOT_IDS],config)
                                  for c in controllers},
             'full_benchmark_executed':False,'remaining_60_optimization_states_executed':False}
    write_json(output/'summary.json',summary)
    write_csv(output/'fixed_metrics.csv',fixed)
    optrecords=[r for o in optimizations for r in o['records']]
    # Missing finite candidates remain explicit and do not disappear from output.
    fields=list(dict.fromkeys(k for r in optrecords for k in r))
    import csv
    with (output/'optimization_metrics.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(optrecords)
    from src.policy_improvement_audit import finalize
    finalize(root,output,config,summary,before,hashes)
