"""Additive E10 full study; all accepted pilot methods and evidence are reused."""

import io
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time
import traceback
import unittest

import numpy as np

from src.counterfactual import assert_bitwise_equal
from src.horizon_experiment import (
    _assert_hashes,immutable_json,load_npz,read_json,save_npz,sha256,write_json,
)
from src.horizon_study import exclusive
from src.policy_improvement import assess,controller_bank,derive_lqr,evaluate_fixed,measure,paired,safe_oracle
from src.policy_improvement_experiment import original_baseline,verify_completed as verify_pilot
from src.recoverability import candidate_diagnostics,evaluate_controls,pack,replay_and_measure,solve_attempt,unpack,validate_execution
from src.recoverability_experiment import cached_evaluation,protected_manifest
from src.recoverable_pd import case_identity,load_benchmark

CONFIG='configs/experiment_10_full_study.json'
ADDITIONS=[CONFIG,'experiments/10_full_study.py','src/policy_improvement_study.py',
           'src/policy_improvement_study_analysis.py','tests/test_policy_improvement_study.py']
THREADS=('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS')


def settings(root):
    control=read_json(root/CONFIG);scientific=read_json(root/control['scientific_config'])
    pilot=root/control['accepted_pilot'];output=root/control['output']
    assert control['authorized_phase']=='full_study_after_reviewed_five_case_pilot'
    assert scientific==read_json(pilot/'protocol.json')
    assert control['new_optimization_states']==60 and control['new_optimization_solves']==180
    assert control['reuse_original_pilot_optimization_solves']==12
    assert control['total_scientific_optimization_solves']==192 and not control['new_solver_repeats']
    assert not control['controller_or_method_changes'] and not control['training_data_use']
    assert control['workers']==1 and control['maximum_workers']==2
    return scientific,control,pilot,output


def _freeze(folder):
    marker=folder/'completion.json'
    hashes={str(p.relative_to(folder)):sha256(p) for p in sorted(folder.rglob('*')) if p.is_file() and p!=marker}
    write_json(marker,{'passed':True,'artifacts_sha256':hashes})


def _load_PD_population(root,scientific,ids):
    """Read each E09 baseline chunk once; never call a simulator here."""
    result={};wanted=set(ids)
    source=root/scientific['baseline_archive']/'baseline'
    for folder in sorted(source.glob('chunk_*')):
        grid=load_npz(folder/'identity_and_legacy.npz')['grid_ids']
        eligible=[(k,int(sid)) for k,sid in enumerate(grid) if int(sid) in wanted]
        if not eligible:continue
        arrays=load_npz(folder/'original.npz');replay=load_npz(folder/'replay.npz')
        for k,sid in eligible:
            a={name:value[[k]] for name,value in arrays.items()}
            b={name:value[[k]] for name,value in replay.items()}
            assert_bitwise_equal(a,b);result[sid]=a
    assert set(result)==wanted
    return result


def fixed_chunk(output,pilot,ids,controller,PD,benchmark,scientific):
    folder=output/'fixed'/controller['name']/f'chunk_{ids[0]:04d}_{ids[-1]:04d}'
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'completion.json').exists():
        _assert_hashes(folder,read_json(folder/'completion.json')['artifacts_sha256'])
        return read_json(folder/'records.json')
    originals=[];replays=[];records=[]
    costs={'new_steps':0,'new_resets':0,'new_wall_seconds':0.,'reused_pilot_conditions':0,'reused_E09_PD_conditions':0}
    pilot_ids=read_json(pilot/'selection.json')['pilot_ids']
    for sid in ids:
        initial=np.array(case_identity(benchmark,sid)['initial'],dtype=np.float64)
        if controller['name']=='PD_8_2':
            a=PD[sid];b=a;costs['reused_E09_PD_conditions']+=1
        elif sid in pilot_ids:
            source=pilot/f'state_{sid:04d}/controllers'/controller['name']
            a=load_npz(source/'rollout.npz');b=load_npz(source/'replay.npz')
            costs['reused_pilot_conditions']+=1
        else:
            a,c,diagnostics=evaluate_fixed(initial,101005+sid,controller)
            b,d,_=replay_and_measure(a,initial,101005+sid,scientific)
            assert diagnostics['initial_reset_bitwise'] and diagnostics['all_201_raw_states_checked']
            costs['new_steps']+=int(c['step_attempts'])+int(d['step_attempts'])
            costs['new_resets']+=int(c['reset_attempts'])+int(d['reset_attempts'])
            costs['new_wall_seconds']+=float(c['wall_seconds'])+float(d['wall_seconds'])
        validate_execution(a,initial);assert_bitwise_equal(a,b)
        metric=paired(measure(a,scientific),measure(PD[sid],scientific))
        records.append(dict(state_id=sid,controller=controller['name'],**metric))
        originals.append(a);replays.append(b)
    save_npz(folder/'original.npz',{name:np.concatenate([a[name] for a in originals]) for name in originals[0]})
    save_npz(folder/'replay.npz',{name:np.concatenate([a[name] for a in replays]) for name in replays[0]})
    save_npz(folder/'grid_ids.npz',{'grid_ids':np.array(ids,dtype=np.int64)})
    write_json(folder/'records.json',records);write_json(folder/'costs.json',costs);_freeze(folder)
    return records


def worker(root,folder):
    root=Path(root).resolve();folder=Path(folder).resolve()
    scientific,_,_,output=settings(root);selection=read_json(output/'selection.json')
    request=read_json(folder/'request.json');sid=request['state_id']
    assert sid in selection['remaining_optimization_ids'] and len(selection['remaining_optimization_ids'])==60
    assert folder.parent.parent==output/f'state_{sid:04d}'/'optimization'
    assert folder.name=='original' and folder.parent.name in scientific['starts']
    assert request['start']==folder.parent.name and request['phase']=='original'
    assert request['selection_sha256']==sha256(output/'selection.json')
    assert request['scientific_config_sha256']==sha256(root/'configs/experiment_10_pilot.json')
    assert request['E07_config_sha256']==sha256(root/scientific['optimization_config'])
    identity=case_identity(load_benchmark(root/scientific['benchmark']),sid)
    initial=np.array(request['initial'],dtype=np.float64)
    assert initial.tobytes().hex()==identity['initial_raw_bytes_hex']==request['initial_raw_bytes_hex']
    assert request['initial_hex']==identity['initial_hex'] and not identity['archived_unsafe']
    guess=load_npz(folder/'initial_guess.npz')['decision']
    assert guess.shape==(600,) and np.isfinite(guess).all()
    solve_attempt(folder,read_json(root/scientific['optimization_config']),initial,guess)


def attempt(root,folder,request,guess,scientific):
    folder.mkdir(parents=True,exist_ok=True);immutable_json(folder/'request.json',request)
    if (folder/'initial_guess.npz').exists():assert_bitwise_equal(load_npz(folder/'initial_guess.npz'),{'decision':guess})
    else:save_npz(folder/'initial_guess.npz',{'decision':guess})
    old=read_json(root/scientific['optimization_config'])
    if not (folder/'attempt.json').exists():
        if (folder/'solver.log').exists():raise RuntimeError(f'Interrupted output preserved; inspect before resume: {folder}')
        env=dict(os.environ)
        for name in THREADS:env[name]='1'
        started=time.perf_counter()
        with (folder/'solver.log').open('w') as stream:
            process=subprocess.Popen([sys.executable,str(root/'experiments/10_full_study.py'),'--worker',str(folder)],
                                     cwd=folder,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            timeout=False
            try:process.wait(timeout=old['watchdog_seconds'])
            except subprocess.TimeoutExpired:timeout=True;os.killpg(process.pid,signal.SIGKILL);process.wait()
        write_json(folder/'attempt.json',{'watchdog_timeout':timeout,'exit_code':process.returncode,
                   'process_wall_seconds':time.perf_counter()-started,
                   'technical_failure':bool(process.returncode!=0 and not timeout),
                   'thread_environment':{name:env[name] for name in THREADS}})
    record=read_json(folder/'attempt.json');path=folder/'candidate.npz'
    if not path.exists():path=folder/'last_iterate.npz'
    vector=load_npz(path)['decision'] if path.exists() else None
    valid=vector is not None and vector.shape==(600,) and bool(np.isfinite(vector).all())
    review=dict(record,candidate_available=valid,candidate_source=path.name if path.exists() else None)
    if (folder/'solver_result.json').exists():review['solver']=read_json(folder/'solver_result.json')
    if valid:review['diagnostics']=candidate_diagnostics(vector,request['initial'])
    immutable_json(folder/'candidate_review.json',review)
    return vector if valid else None,review


def optimization_state(root,output,pilot,sid,PD,scientific,selection):
    folder=output/f'state_{sid:04d}';folder.mkdir(exist_ok=True)
    if (folder/'completion.json').exists():
        _assert_hashes(folder,read_json(folder/'completion.json')['artifacts_sha256'])
        return read_json(folder/'summary.json')
    initial=PD[sid]['raw_states'][0,0].copy();baseline=measure(PD[sid],scientific)
    if sid in scientific['optimization_pilot_ids']:
        source=pilot/f'state_{sid:04d}'/'optimization'
        if not (folder/'optimization').exists():shutil.copytree(source,folder/'optimization')
        original=next(o for o in read_json(pilot/'summary.json')['optimizations'] if o['state_id']==sid)
        records=[r for r in original['records'] if r['phase']=='original']
        reused=True
        for p in source.rglob('*'):
            if p.is_file():assert sha256(p)==sha256(folder/'optimization'/p.relative_to(source))
    else:
        assert sid in selection['remaining_optimization_ids'];reused=False;records=[]
        opt=folder/'optimization';opt.mkdir(exist_ok=True)
        old=read_json(root/scientific['optimization_config'])
        immutable_json(opt/'unchanged_E07_configuration.json',old)
        for index,name in enumerate(scientific['starts']):
            init=opt/f'initialization_{name}.npz'
            if not init.exists():
                if name=='PD':a=PD[sid]
                else:
                    u=np.zeros(200) if name=='zero' else np.repeat(np.random.default_rng(
                        np.random.SeedSequence([101004,sid])).uniform(-2,2,size=10),20)
                    a,c=evaluate_controls(initial,101005+sid,u)
                    write_json(opt/f'initialization_{name}_costs.json',c)
                save_npz(init,a)
            a=load_npz(init);guess=pack(a['raw_states'][0],a['actions'][0,:,0])
            if name=='PD':assert_bitwise_equal(a,PD[sid])
            request={'state_id':sid,'initial':initial.tolist(),'initial_hex':[float(v).hex() for v in initial],
                     'initial_raw_bytes_hex':initial.tobytes().hex(),'start':name,'phase':'original',
                     'selection_sha256':sha256(output/'selection.json'),
                     'scientific_config_sha256':sha256(root/'configs/experiment_10_pilot.json'),
                     'E07_config_sha256':sha256(root/scientific['optimization_config'])}
            target=opt/name/'original';vector,review=attempt(root,target,request,guess,scientific)
            status=review.get('solver',{}).get('stats',{}).get('return_status','missing_solver_result')
            if vector is None:
                records.append({'state_id':sid,'start':name,'start_index':index,'phase':'original',
                                'candidate_available':False,'solver_status':status})
                continue
            states,actions=unpack(vector,initial,200)
            for mode in scientific['candidate_modes']:
                arrays,basic=cached_evaluation(target/mode,initial,101005+sid,actions,old,
                                                states if mode=='tracking' else None)
                record=paired({'state_id':sid,'start':name,'start_index':index,'phase':'original','mode':mode,
                               'candidate_available':True,'solver_status':status,
                               'solver_success':bool(review.get('solver',{}).get('stats',{}).get('success',False)),
                               **review['diagnostics'],**basic,**measure(arrays,scientific)},baseline)
                difference=np.abs(arrays['raw_states'][0]-states)
                record['maximum_plan_execution_discrepancy']=difference.max(axis=0).tolist()
                write_json(target/mode/'extended_metrics.json',record);records.append(record)
    result={'state_id':sid,'reused_pilot':reused,'records':records,
            'oracles':[safe_oracle(records,baseline,mode) for mode in scientific['candidate_modes']]}
    write_json(folder/'summary.json',result);_freeze(folder)
    return result


def verify_completed(root):
    root=Path(root).resolve();_,_,_,output=settings(root);verify_pilot(root)
    done=read_json(output/'completion.json');assert done['passed']
    _assert_hashes(output,done['artifacts_sha256'])
    _assert_hashes(root,read_json(output/'source_hashes.json'))
    _assert_hashes(root,read_json(output/'preservation_before.json')['sha256'])
    assert sha256(root/'docs/experiment_10_report.md')==done['report_sha256']
    print('Completed full E10 study verified; no simulation or optimization rerun.',flush=True)


def run_study(root,resume=False,snapshot=None):
    root=Path(root).resolve();scientific,control,pilot,output=settings(root)
    if output.exists() and not resume:raise FileExistsError('Study exists; inspect and use --resume or --verify.')
    output.mkdir(parents=True,exist_ok=True)
    with exclusive(output):
        if (output/'completion.json').exists():verify_completed(root);return
        try:_run(root,output,pilot,scientific,control,snapshot)
        except BaseException as exc:
            p=output/'failed_checks';p.mkdir(exist_ok=True)
            write_json(p/f'failure_{len(list(p.glob("*.json")))+1:03d}.json',
                       {'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
            raise


def _run(root,output,pilot,scientific,control,snapshot):
    verify_pilot(root)
    immutable_json(output/'scientific_protocol.json',scientific);immutable_json(output/'authorization.json',control)
    if not (output/'preservation_before.json').exists():
        write_json(output/'preservation_before.json',read_json(snapshot) if snapshot else protected_manifest(root,output))
    before=read_json(output/'preservation_before.json');_assert_hashes(root,before['sha256'])
    hashes={name:sha256(root/name) for name in sorted(set(list(read_json(pilot/'source_hashes.json'))+ADDITIONS))}
    immutable_json(output/'source_hashes.json',hashes)
    for name,h in hashes.items():
        p=output/'source_snapshot'/name;p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists():p.write_bytes((root/name).read_bytes())
        assert sha256(p)==h
    for name in ['selection.json','selection_frozen.json','lqr.json','controllers.json','practical_thresholds_frozen.json']:
        if not (output/name).exists():shutil.copy2(pilot/name,output/name)
        assert sha256(output/name)==sha256(pilot/name)
    assert derive_lqr(scientific)==read_json(pilot/'lqr.json')
    immutable_json(output/'metadata.json',{'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                   'numpy':np.__version__,'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
                   'pilot_tracking_vs_LQR_diagnostic_sha256':sha256(root/'results/safe_policy_improvement/pilot_tracking_vs_LQR.json')})
    if not (output/'technical_tests.json').exists():
        stream=io.StringIO();suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_policy_improvement_study.py')
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        (output/'technical_tests.txt').write_text(stream.getvalue())
        write_json(output/'technical_tests.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,
                   'failures':len(result.failures),'errors':len(result.errors),'simulator_steps':0,'solver_executions':0})
    assert read_json(output/'technical_tests.json')['passed']
    selection=read_json(output/'selection.json');ids=selection['population_ids']
    assert len(ids)==3331 and len(set(ids))==3331
    benchmark=load_benchmark(root/scientific['benchmark']);PD=_load_PD_population(root,scientific,ids)
    controllers=read_json(output/'controllers.json');fixed=[]
    for controller in controllers:
        collected=[]
        for start in range(0,len(ids),control['chunk_size']):
            batch=ids[start:start+control['chunk_size']]
            collected.extend(fixed_chunk(output,pilot,batch,controller,PD,benchmark,scientific))
            if start%(control['chunk_size']*16)==0:print(f'{controller["name"]}: {min(start+len(batch),3331)}/3331',flush=True)
        fixed.extend(collected)
        print(f'Completed controller {controller["name"]}: 3331/3331',flush=True)
    write_json(output/'fixed_records.json',fixed)
    optimizations=[]
    for index,sid in enumerate(selection['optimization_ids_sorted']):
        optimizations.append(optimization_state(root,output,pilot,sid,PD,scientific,selection))
        print(f'Optimization state {index+1}/64: grid {sid}, reused pilot={optimizations[-1]["reused_pilot"]}',flush=True)
    write_json(output/'optimization_records.json',optimizations)
    from src.policy_improvement_study_analysis import finalize
    finalize(root,output,pilot,scientific,control,fixed,optimizations,before,hashes)
