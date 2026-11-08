"""E12_MPC80_FEAS_v2 static instrumentation preflight only; no execution mode."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import resource
import signal
import subprocess
import sys
import time
import unittest
START=time.perf_counter()
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[n]='1'
os.environ['E12_UPSTREAM']='/private/tmp/e12_safe_control_gym'
os.environ['MPLCONFIGDIR']='/private/tmp/e12_matplotlib'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))


def tests_worker(a):
    from src import quadrotor_mpc80_feasibility_v2 as v
    os.environ['E12_UPSTREAM']=str(a.upstream)
    with v.zero_real_operations(a.upstream) as (blocked,denied):
        suite=unittest.TestSuite()
        for p in sorted((ROOT/'tests').glob('test_quadrotor*.py')):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromName(p.stem))
        with (a.output/'tests.log').open('x') as stream:
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    v.immutable(a.output/'tests.json',{'passed':result.wasSuccessful() and not blocked and not denied,
        'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
        'forbidden_optimizer_or_step_calls':blocked,'forbidden_real_engine_calls':denied,
        'mock_calls_are_not_real_physics_steps':True,'new_solves':0,'new_physics_steps':0})
    if not result.wasSuccessful() or blocked or denied:raise SystemExit(1)


def main():
    q=argparse.ArgumentParser(__doc__);q.add_argument('--upstream',type=Path,required=True)
    q.add_argument('--output',type=Path,default=ROOT/'results/benchmark_selection/mpc80_feas_v2_preflight')
    g=q.add_mutually_exclusive_group(required=True)
    g.add_argument('--preflight',action='store_true');g.add_argument('--verify-static',action='store_true')
    g.add_argument('--tests-worker',action='store_true',help=argparse.SUPPRESS)
    a=q.parse_args();a.output=a.output.resolve();a.upstream=a.upstream.resolve()
    from src import quadrotor_mpc80_feasibility_v2 as v
    from src.quadrotor_mpc80_feasibility_v1 import charge_overhead,durable_json
    if a.tests_worker:tests_worker(a);return
    if a.preflight:v.prepare(ROOT,a.upstream,a.output)
    else:v.check_seal(ROOT,a.output);assert v.p.read(a.output/'preflight.json')['zero_solve_preflight_passed']
    out=a.output;ledger=v.p.read(out/'resource_ledger.json')
    left_cpu=1800-ledger['charged']['cpu_seconds'];left_wall=3600-ledger['charged']['wall_seconds']-(time.perf_counter()-START)
    if min(left_cpu,left_wall)<=0:raise RuntimeError('Shared v1/v2 budget exhausted')
    resource.setrlimit(resource.RLIMIT_CPU,(max(1,int(left_cpu)),max(2,int(left_cpu)+1)))
    def timeout(*args):raise TimeoutError('Shared active wall limit exceeded')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,left_wall)
    label='first' if a.preflight else 'repeat';report=out/('preflight.json' if a.preflight else 'repeat_verification.json')
    if report.exists():raise FileExistsError('Existing verification preserved')
    try:
        if a.preflight:
            with (out/'tests_stdout.log').open('x') as stream:
                t=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream',str(a.upstream),
                    '--output',str(out),'--tests-worker'],stdout=stream,stderr=subprocess.STDOUT,timeout=min(300,left_wall))
            if t.returncode:raise RuntimeError('Zero-solve unit tests failed; no execution permitted')
        assert v.p.read(out/'tests.json')['passed']
        children=resource.getrusage(resource.RUSAGE_CHILDREN)
        parent_limit=max(1,int(left_cpu-children.ru_utime-children.ru_stime))
        resource.setrlimit(resource.RLIMIT_CPU,(parent_limit,parent_limit+1))
        r=v.validate(ROOT,a.upstream,out,out/('checks_'+label))
        peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        child_peak=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        assert peak+child_peak<=4294967296
        assert ledger['charged']['cpu_seconds']+time.process_time()+children.ru_utime+children.ru_stime+.1<=1800
        assert ledger['charged']['wall_seconds']+time.perf_counter()-START+.1<=3600
        v.immutable(report,r)
        digest=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
        if a.verify_static:
            assert r==v.p.read(out/'preflight.json'),'Fresh-process static results differ'
            v.immutable(out/'repeatability.json',{'passed':True,'scientific_sha256':digest,'new_solves':0,'new_physics_steps':0})
        state=v.p.read(out/'resource_ledger.json');state['preflight_passed']=True
        durable_json(out/'resource_ledger.json',state)
        v.immutable(out/('resource_record_'+label+'.json'),{'peak_RSS_bytes':peak,
            'child_peak_RSS_bytes':child_peak,'conservative_sum_of_process_peaks':peak+child_peak,'scientific_sha256':digest,
            'self_CPU_seconds_at_checkpoint':time.process_time(),'child_CPU_seconds':
            resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
            'wall_seconds_at_checkpoint':time.perf_counter()-START,'new_solves':0,'new_physics_steps':0})
        print({'zero_solve_preflight_passed':True,'real_steps':0,'optimizer_solves':0,'execution_authorized':False,
               'scientific_sha256':digest,'mock_application_boundaries':r['checks']['mock_boundary_count']})
    except BaseException as exc:
        state=v.p.read(out/'resource_ledger.json');state['preflight_passed']=False;state['halted']=True
        durable_json(out/'resource_ledger.json',state)
        v.safe_write(out/('failure_'+label+'.json'),{'status':'failed_or_incomplete','kind':type(exc).__name__,
            'message':str(exc),'execution_authorized':False,'new_solves':0,'new_physics_steps':0});raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        children=resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu=math.ceil((time.process_time()+children.ru_utime+children.ru_stime+.1)*10)/10
        wall=math.ceil((time.perf_counter()-START+.1)*10)/10
        charge_overhead(out/'resource_ledger.json',cpu,wall)


if __name__=='__main__':main()
