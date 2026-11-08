"""Execute ONLY the explicitly approved frozen E12_MPC80_FEAS_v2 jobs.

No screening regeneration, action probes, other baselines, training or automatic
retry. Every worker requires an already-persisted job reservation. Accepted
preflight evidence and its resource ledger remain read-only.
"""
from pathlib import Path
import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
START=time.perf_counter()
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['MPLCONFIGDIR']='/private/tmp/e12_matplotlib'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))
from src import quadrotor_mpc80_execution_v2 as e
from src import quadrotor_mpc80_feasibility_v1 as f
from src import quadrotor_pilot as p
from src.quadrotor_mmaq_v1 import safe_write


def children_cpu():
    r=resource.getrusage(resource.RUSAGE_CHILDREN);return r.ru_utime+r.ru_stime


def run(a):
    out=a.output;ledger_path=out/'resource_ledger.json';last_cpu=0.;last_wall=START
    def account():
        nonlocal last_cpu,last_wall
        now=time.process_time()+children_cpu();wall=time.perf_counter()
        cpu_charge=math.ceil(max(0,now-last_cpu)*1000)/1000
        wall_charge=math.ceil(max(0,wall-last_wall)*1000)/1000
        last_cpu,last_wall=now,wall
        f.charge_overhead(ledger_path,cpu_charge,wall_charge)
    if out.exists():raise FileExistsError('Completed/partial campaign retained. No automatic retry/resume.')
    e.prepare(ROOT,a.upstream,a.preflight,out,a.request)
    try:
        # A single static test child; no solver/physics execution in this child.
        with (out/'tests_stdout.log').open('x') as stream:
            tested=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream',str(a.upstream),
                '--preflight',str(a.preflight),'--output',str(out),'--tests-worker'],
                stdout=stream,stderr=subprocess.STDOUT,timeout=300)
        account()
        if tested.returncode or not p.read(out/'tests.json')['passed']:raise RuntimeError('Execution-wiring static tests failed')
        approval=p.read(out/'approval.json')
        for job in f.job_plan():
            e.checkpoint(ROOT,a.upstream,a.preflight,out);account()
            state=p.read(ledger_path);assert not state['halted']
            cpu=state['caps']['aggregate_cpu_seconds']-state['charged']['cpu_seconds']
            wall=state['caps']['wall_seconds']-state['charged']['wall_seconds']
            if min(cpu,wall)<=0:raise RuntimeError('No remaining study budget')
            receipt=f.reserve_job(ledger_path,job['id'],approval)
            with (out/(job['id']+'.log')).open('x') as stream:
                try:
                    result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream',str(a.upstream),
                        '--preflight',str(a.preflight),'--output',str(out),'--worker',receipt['id'],
                        '--cpu-limit',str(max(1,int(cpu)))],stdout=stream,stderr=subprocess.STDOUT,
                        timeout=min(600,wall))
                    ok=result.returncode==0 and p.read(out/'job_results'/f"{job['id']}.json")['passed']
                except BaseException:
                    f.finish_job(ledger_path,job['id'],False);account();raise
            f.finish_job(ledger_path,job['id'],ok);account()
            print(job['id'],'PASS' if ok else 'FAIL','charged',p.read(ledger_path)['charged'],flush=True)
            if not ok:raise RuntimeError('FIRST failed job; campaign stopped: '+job['id'])
        safe_write(out/'execution_complete.json',{'complete_jobs':96,'scientific_executions':32,
            'complete_state_pairs':16,'new_solver_reservations':9600,'physics_step_reservations':576000,
            'original_overall_qualification':'FAILED','original_0001_agreement':'FAILED'})
    except BaseException as exc:
        state=p.read(ledger_path);state['halted']=True;f.durable_json(ledger_path,state)
        safe_write(out/'campaign_stop.json',{'kind':type(exc).__name__,'message':str(exc),
            'first_failure_stops_campaign':True,'no_retries':True,'full_state_denominator':16,
            'full_execution_denominator':32})
        print('CAMPAIGN STOPPED:',type(exc).__name__,str(exc),flush=True)
    finally:
        account()


def worker(a):
    e.checkpoint(ROOT,a.upstream,a.preflight,a.output)
    state=p.read(a.output/'resource_ledger.json');job=state['reservations'][-1]
    assert job['id']==a.worker and job['status']=='reserved' and not state['halted']
    resource.setrlimit(resource.RLIMIT_CPU,(a.cpu_limit,a.cpu_limit+1))
    if job['kind']=='episode':result=e.episode(ROOT,a.upstream,a.preflight,a.output,job['state_id'],job['phase'])
    else:result=e.replay(a.upstream,a.output,job['state_id'],job['phase'],job['kind'])
    safe_write(a.output/'job_results'/f"{job['id']}.json",{'passed':True,'job':job,
        'actual_solver_calls':300 if job['kind']=='episode' else 0,'actual_physics_steps':6000,
        'result_summary':{k:v for k,v in result.items() if k!='timings'},
        'worker_cpu_seconds':time.process_time(),'worker_wall_seconds':time.perf_counter()-START,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})


def tests_worker(a):
    import unittest
    p.activate(a.upstream,p.read(a.preflight/'accepted_frozen.json')['config'])
    from src.quadrotor_mpc80_feasibility_v2 import zero_real_operations
    with zero_real_operations(a.upstream) as (forbidden,denied):
        suite=unittest.TestSuite()
        for path in sorted((ROOT/'tests').glob('test_quadrotor*.py')):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromName(path.stem))
        with (a.output/'tests.log').open('x') as stream:
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    safe_write(a.output/'tests.json',{'passed':result.wasSuccessful() and not forbidden and not denied,
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),
        'skipped':len(result.skipped),'forbidden_native_calls':forbidden,'forbidden_real_engine_calls':denied,'new_solves':0,'new_physics_steps':0})
    if not result.wasSuccessful() or forbidden or denied:raise SystemExit(1)


def main():
    q=argparse.ArgumentParser(__doc__);q.add_argument('--upstream',type=Path,required=True)
    q.add_argument('--preflight',type=Path,default=ROOT/'results/benchmark_selection/mpc80_feas_v2_preflight')
    q.add_argument('--output',type=Path,default=ROOT/'results/benchmark_selection/mpc80_feas_v2_study')
    q.add_argument('--request',type=Path);q.add_argument('--run',action='store_true')
    q.add_argument('--worker');q.add_argument('--cpu-limit',type=int)
    q.add_argument('--tests-worker',action='store_true');a=q.parse_args()
    a.output=a.output.resolve();a.preflight=a.preflight.resolve();a.upstream=a.upstream.resolve()
    if a.tests_worker:tests_worker(a)
    elif a.worker:
        try:worker(a)
        except BaseException as exc:
            safe_write(a.output/'job_results'/f'{a.worker}_failure.json',{'kind':type(exc).__name__,
                'message':str(exc),'status':'failed_or_incomplete','fallback_action_applied':False,
                'worker_cpu_seconds':time.process_time(),'worker_wall_seconds':time.perf_counter()-START})
            raise
    elif a.run:
        if a.request is None:q.error('--run requires the explicit execution approval request')
        run(a)
    else:q.error('Specify the authorized --run mode; workers require existing reservations')


if __name__=='__main__':main()
