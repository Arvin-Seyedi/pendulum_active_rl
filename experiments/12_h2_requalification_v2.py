"""Only approved H2/MPC20 original/repeat, after zero-solve preflight."""
from pathlib import Path
import argparse
import os
import resource
import subprocess
import sys
import time
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[name]='1'
os.environ['MPLCONFIGDIR']='/private/tmp/e12_matplotlib';os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import quadrotor_pilot as p
from src import quadrotor_h2_requalification_v2 as h
from src.quadrotor_mmaq_v1 import safe_write


def run(upstream,out):
    h.check_seal(ROOT,out);assert p.read(out/'preflight.json')['passed']
    budget=p.read(out/'protocol.json');usage=p.read(out/'usage.json') if (out/'usage.json').exists() else {'solver_calls':0,'physics_steps':0,'cpu_seconds':0.}
    ledger=p.read(out/'jobs.json') if (out/'jobs.json').exists() else {}
    for phase in ['original','repeat']:
        folder=out/'runs'/f'H2_nonlinear20_1000_{phase}'
        for job,marker in [('episode','v2_execution_complete.json'),('native_replay','replay_native.json'),('motor_replay','replay_instrumented.json')]:
            if (folder/marker).exists():continue
            key=f'{phase}_{job}'
            if key in ledger:raise RuntimeError('Interrupted attempt retained; no automatic restart')
            calls=300 if job=='episode' else 0
            assert usage['solver_calls']+calls<=600 and usage['physics_steps']+6000<=36000
            assert usage['cpu_seconds']<p.read(out/'prior_remaining_budget.json')['cpu_seconds']
            ledger[key]={'status':'reserved','solver_calls':calls,'physics_steps':6000}
            usage['solver_calls']+=calls;usage['physics_steps']+=6000
            safe_write(out/'jobs.json',ledger);safe_write(out/'usage.json',usage)
            before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.perf_counter()
            left=p.read(out/'prior_remaining_budget.json')['cpu_seconds']-usage['cpu_seconds']-time.process_time()
            with (out/f'{key}.log').open('x') as stream:
                result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream',str(upstream),
                    '--output',str(out),'--worker',job,'--phase',phase,'--cpu-limit',str(max(1,int(left)))],
                    stdout=stream,stderr=subprocess.STDOUT,timeout=min(600,left))
            after=resource.getrusage(resource.RUSAGE_CHILDREN)
            cpu=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
            usage['cpu_seconds']+=cpu;safe_write(out/'usage.json',usage)
            ledger[key].update(status='completed' if result.returncode==0 and (folder/marker).exists() else 'failed',
                               cpu_seconds=cpu,wall_seconds=time.perf_counter()-start)
            safe_write(out/'jobs.json',ledger)
            if ledger[key]['status']=='failed':raise RuntimeError('Worker failed; retained all outputs: '+key)
        print(phase,p.read(folder/'completion.json')['complete'],flush=True)
    safe_write(out/'execution_complete.json',{'scientific_executions_attempted':2,'new_solver_cap':600,'new_physics_cap':36000})


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'results/benchmark_selection/H2_MPC20_requalification_v2')
    parser.add_argument('--preflight',action='store_true');parser.add_argument('--run',action='store_true');parser.add_argument('--verify',action='store_true')
    parser.add_argument('--worker',choices=['episode','native_replay','motor_replay']);parser.add_argument('--phase',choices=['original','repeat'])
    parser.add_argument('--cpu-limit',type=int);args=parser.parse_args();out=args.output.resolve();out.mkdir(exist_ok=True,parents=True)
    if args.worker:
        h.check_seal(ROOT,out);assert p.read(out/'preflight.json')['passed']
        if args.cpu_limit:resource.setrlimit(resource.RLIMIT_CPU,(args.cpu_limit,args.cpu_limit+1))
        folder=out/'runs'/f'H2_nonlinear20_1000_{args.phase}'
        if args.worker=='episode':result=h.episode(ROOT,args.upstream,out,args.phase)
        elif args.worker=='native_replay':result=p.replay(args.upstream,out,folder,False)
        else:
            from src.quadrotor_linear_revalidation import replay_executed_motor_commands
            result=replay_executed_motor_commands(args.upstream,out,folder)
        print(result);return
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        if args.preflight:
            h.prepare(ROOT,args.upstream,out)
            with (out/'preflight_tests.log').open('x') as stream:
                result=subprocess.run([sys.executable,'-m','pytest','-q',*sorted(str(x) for x in (ROOT/'tests').glob('test_quadrotor*.py'))],
                    stdout=stream,stderr=subprocess.STDOUT,timeout=300)
            h.immutable(out/'preflight_tests.json',{'passed':result.returncode==0,'log_sha256':p.sha(out/'preflight_tests.log')})
            print(h.preflight(ROOT,args.upstream,out))
        if args.run:run(args.upstream,out)
        if args.verify:
            from src.quadrotor_h2_verification_v2 import verify
            print(verify(ROOT,args.upstream,out))
    except Exception as exc:
        safe_write(out/'last_failure.json',{'kind':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        if (out/'protocol_seal.json').exists():
            usage=p.read(out/'usage.json') if (out/'usage.json').exists() else {'solver_calls':0,'physics_steps':0,'cpu_seconds':0.}
            after=resource.getrusage(resource.RUSAGE_CHILDREN)
            child=0 if args.run else after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
            usage['cpu_seconds']+=time.process_time()+child;safe_write(out/'usage.json',usage)


if __name__=='__main__':main()
