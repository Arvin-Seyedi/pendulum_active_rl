"""E12_MPC80_FEAS_v1 ZERO-SOLVE PREFLIGHT ONLY; no execution/replay mode."""
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

START_WALL=time.perf_counter()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['MPLCONFIGDIR']='/private/tmp/e12_matplotlib'
ROOT=Path(__file__).resolve().parents[1]
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'tests'))


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def tests_worker(upstream,out):
    from src import quadrotor_mpc80_feasibility_v1 as f
    from src.quadrotor_pilot import activate
    activate(upstream,f.read(ROOT/'configs/experiment_12.json'))
    with f.zero_operations() as blocked:
        suite=unittest.TestSuite()
        for path in sorted((ROOT/'tests').glob('test_quadrotor*.py')):
            suite.addTests(unittest.defaultTestLoader.loadTestsFromName(path.stem))
        with (out/'tests.log').open('x') as stream:
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    report={'passed':result.wasSuccessful() and not blocked,'tests':result.testsRun,
            'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
            'forbidden_production_calls':blocked,'actual_new_solves':0,'actual_new_physics_steps':0}
    f.immutable(out/'tests.json',report)
    print(report)
    if not report['passed']:raise SystemExit(1)


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'results/benchmark_selection/mpc80_feas_v1_preflight')
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--preflight',action='store_true')
    g.add_argument('--verify-archived',action='store_true')
    g.add_argument('--tests-worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args();out=a.output.resolve()
    # No command can select an action, execute/replay a cohort, or authorize it.
    from src import quadrotor_mpc80_feasibility_v1 as f
    if a.tests_worker:
        tests_worker(a.upstream,out);return
    if a.preflight and (out/'preflight.json').exists():
        f.check_seal(ROOT,out)
        print('Completed preflight preserved. Use --verify-archived for a new static repeat.');return
    if a.preflight:f.prepare(ROOT,a.upstream,out)
    else:
        f.check_seal(ROOT,out)
        if not f.read(out/'preflight.json')['passed']:raise RuntimeError('No passed preflight to repeat')
    ledger=f.read(out/'resource_ledger.json');caps=ledger['caps'];prior=ledger['charged']
    cpu_left=caps['aggregate_cpu_seconds']-prior['cpu_seconds']
    wall_left=caps['wall_seconds']-prior['wall_seconds']-(time.perf_counter()-START_WALL)
    if min(cpu_left,wall_left)<=0:raise RuntimeError('Preflight budget exhausted')
    resource.setrlimit(resource.RLIMIT_CPU,(max(1,int(cpu_left)),max(2,int(cpu_left)+1)))
    def timeout(signum,frame):raise TimeoutError('Aggregate active wall budget exhausted')
    signal.signal(signal.SIGALRM,timeout);signal.setitimer(signal.ITIMER_REAL,wall_left)
    label='first' if a.preflight else 'repeat'
    evidence=out/('checks_'+label)
    report_path=out/('preflight.json' if a.preflight else 'repeat_verification.json')
    if report_path.exists() or evidence.exists():raise FileExistsError('Static attempt already exists; no overwrite')
    try:
        if a.preflight:
            with (out/'tests_worker_stdout.log').open('x') as stream:
                tested=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream',str(a.upstream),
                    '--output',str(out),'--tests-worker'],stdout=stream,stderr=subprocess.STDOUT,
                    timeout=min(300,wall_left))
            if tested.returncode or not f.read(out/'tests.json')['passed']:
                raise RuntimeError('Zero-solve regression did not pass; inspect tests.log')
        tests=f.read(out/'tests.json');assert tests['passed']
        report=f.validate(ROOT,a.upstream,out,evidence)
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes=rss if sys.platform=='darwin' else rss*1024
        assert rss_bytes<=caps['memory_bytes'],'Peak RSS exceeds 4 GiB'
        f.immutable(report_path,report)
        digest=hashlib.sha256(canonical(report)).hexdigest()
        if a.verify_archived:
            old=f.read(out/'preflight.json')
            assert canonical(report)==canonical(old),'Independent static repeat differs'
            f.immutable(out/'repeatability.json',{'passed':True,'canonical_scientific_sha256':digest,
                'new_solves':0,'new_physics_steps':0,'cohort_executions':0})
        else:
            ledger=f.read(out/'resource_ledger.json');ledger['preflight_passed']=True
            ledger['preflight_report_sha256']=f.sha(report_path);f.durable_json(out/'resource_ledger.json',ledger)
        f.immutable(out/('resources_'+label+'.json'),{'peak_rss_bytes':rss_bytes,
            'process_cpu_seconds_at_checkpoint':time.process_time(),
            'child_cpu_seconds':resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+
                                resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime,
            'wall_seconds_at_checkpoint':time.perf_counter()-START_WALL,
            'source_seal_sha256':f.sha(out/'protocol_seal.json'),
            'scientific_sha256':digest,'new_solves':0,'new_physics_steps':0})
        print({'preflight_passed':True,'cohort_states_executed':0,'execution_authorized':False,
               'archived_MPC80_candidates':report['graph']['archived_candidates'],
               'archived_contract_intervals':report['contract']['archived_intervals'],
               'canonical_scientific_sha256':digest})
    except BaseException as exc:
        ledger=f.read(out/'resource_ledger.json');ledger['preflight_passed']=False;ledger['halted']=True
        f.durable_json(out/'resource_ledger.json',ledger)
        f.safe_write(out/('failure_'+label+'.json'),{'kind':type(exc).__name__,'message':str(exc),
            'status':'failed_or_incomplete','new_solver_calls':0,'new_physics_steps':0,
            'execution_authorized':False,'partial_evidence_preserved':True})
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        child=resource.getrusage(resource.RUSAGE_CHILDREN)
        # Include startup, tests, checking and persistence; conservatively reserve
        # another 0.1 s for the final accounting write, rounded upward to 0.1 s.
        cpu=math.ceil((time.process_time()+child.ru_utime+child.ru_stime+.1)*10)/10
        wall=math.ceil((time.perf_counter()-START_WALL+.1)*10)/10
        f.charge_overhead(out/'resource_ledger.json',cpu,wall)


if __name__=='__main__':main()
