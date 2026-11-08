"""E12_MMAQ_v1: Phase A, then ONLY four heldout states if preflight passes."""
from pathlib import Path
import argparse
import os
import resource
import subprocess
import sys
import time

for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:
    os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['MPLCONFIGDIR']='/private/tmp/e12_matplotlib'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import quadrotor_pilot as pilot
from src import quadrotor_mmaq_v1 as q


def run_plan(upstream,out):
    q.check_seal(ROOT,out)
    if not pilot.read(out/'preflight.json')['passed']:raise RuntimeError('Phase A failed')
    q.generate_heldout(ROOT,out)
    for case in ['H0','H1','H2','H3']:
        for controller in ['linear40','nonlinear20','nonlinear40','nonlinear80']:
            for phase in ['original','repeat']:
                folder=out/'runs'/f'{case}_{controller}_1000_{phase}'
                for job,marker in [('episode','qualification_execution.json'),('native_replay','replay_native.json'),('motor_replay','replay_instrumented.json')]:
                    if (folder/marker).exists():continue
                    key=f'{case}_{controller}_{phase}_{job}'
                    old,new=q.budget_available(ROOT,out,300 if job=='episode' else 0,6000)
                    ledger=pilot.read(out/'jobs.json') if (out/'jobs.json').exists() else {}
                    if key in ledger:raise RuntimeError('Interrupted job retained; no automatic rerun: '+key)
                    ledger[key]={'status':'reserved','solver_calls_charged':300 if job=='episode' else 0,'physics_steps_charged':6000}
                    q.safe_write(out/'jobs.json',ledger)
                    # Charge a conservative full reservation before dispatch. A killed
                    # worker never makes consumed resources disappear from accounting.
                    new['solver_calls']+=ledger[key]['solver_calls_charged'];new['physics_steps']+=6000
                    q.safe_write(out/'usage.json',new)
                    before=resource.getrusage(resource.RUSAGE_CHILDREN);start=time.monotonic()
                    left=14400-old['cpu_seconds']-new['cpu_seconds']-time.process_time()
                    if left<=0:raise RuntimeError('Original CPU budget exhausted')
                    log=out/f'{key}.log'
                    try:
                        with log.open('x') as stream:
                            r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream',str(upstream),
                                '--output',str(out),'--worker',job,'--case',case,'--controller',controller,'--phase',phase,
                                '--cpu-limit',str(max(1,int(left)))],stdout=stream,stderr=subprocess.STDOUT,timeout=min(600,left))
                        status='completed' if r.returncode==0 and (folder/marker).exists() else 'failed'
                    except Exception as exc:
                        status='failed';ledger[key]['exception']={'kind':type(exc).__name__,'message':str(exc)}
                    after=resource.getrusage(resource.RUSAGE_CHILDREN)
                    used=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime
                    new['cpu_seconds']+=used;q.safe_write(out/'usage.json',new)
                    ledger[key].update(status=status,cpu_seconds=used,wall_seconds=time.monotonic()-start)
                    if (folder/'completion.json').exists():
                        m=pilot.read(folder/'completion.json')
                        ledger[key]['actual_solver_calls']=m['solver_calls'] if job=='episode' else 0
                        ledger[key]['actual_physics_steps']=20*m['transitions']
                        # Reservations remain charged for incomplete jobs; report actuals separately.
                        if m['complete'] and not m['failure']:
                            assert ledger[key]['actual_physics_steps']==6000
                    q.safe_write(out/'jobs.json',ledger)
                    if status!='completed':raise RuntimeError('Worker failed; evidence retained: '+key)
                print(case,controller,phase,pilot.read(folder/'completion.json')['execution_status'],flush=True)
    q.safe_write(out/'execution_complete.json',{'scientific_executions':32,'screening_executions':0,
                'action_probes':0,'training':False,'original_symbolic_gate':'FAILED'})


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'results/benchmark_selection/mmaq_v1')
    p.add_argument('--approved-design',type=Path)
    p.add_argument('--phase-a',action='store_true');p.add_argument('--run-heldout',action='store_true');p.add_argument('--verify',action='store_true')
    p.add_argument('--worker',choices=['episode','native_replay','motor_replay'])
    p.add_argument('--case',choices=['H0','H1','H2','H3'])
    p.add_argument('--controller',choices=['linear40','nonlinear20','nonlinear40','nonlinear80'])
    p.add_argument('--phase',choices=['original','repeat']);p.add_argument('--cpu-limit',type=int)
    args=p.parse_args();out=args.output.resolve();out.mkdir(exist_ok=True,parents=True)
    if args.worker:
        q.check_seal(ROOT,out)
        if not pilot.read(out/'preflight.json')['passed']:raise RuntimeError('Phase A failed')
        seal=pilot.read(out/'heldout_seal.json')
        for name,h in seal['sha256'].items():assert pilot.sha(out/name)==h
        if args.cpu_limit:resource.setrlimit(resource.RLIMIT_CPU,(args.cpu_limit,args.cpu_limit+1))
        folder=out/'runs'/f'{args.case}_{args.controller}_1000_{args.phase}'
        if args.worker=='episode':result=q.episode(ROOT,args.upstream,out,args.case,args.controller,args.phase)
        elif args.worker=='native_replay':result=pilot.replay(args.upstream,out,folder,False)
        else:
            from src.quadrotor_linear_revalidation import replay_executed_motor_commands
            result=replay_executed_motor_commands(args.upstream,out,folder)
        print(result);return
    start=time.monotonic()
    child_start=resource.getrusage(resource.RUSAGE_CHILDREN)
    try:
        if args.phase_a:
            if not args.approved_design:raise ValueError('--approved-design required')
            q.prepare(ROOT,args.upstream,out,args.approved_design)
            with (out/'preflight_tests.log').open('x') as stream:
                r=subprocess.run([sys.executable,'-m','pytest','-q','tests/test_quadrotor_pilot.py',
                    'tests/test_quadrotor_diagnostic.py','tests/test_quadrotor_linear_revalidation.py',
                    'tests/test_quadrotor_mmaq_v1.py'],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,timeout=300)
            q.immutable(out/'failure_logging_tests.json',{'passed':r.returncode==0,'log_sha256':pilot.sha(out/'preflight_tests.log')})
            result=q.preflight(ROOT,args.upstream,out);print(result)
            if not result['passed']:raise RuntimeError('Phase A failed; Phase B prohibited')
        if args.run_heldout:run_plan(args.upstream,out)
        if args.verify:
            from src.quadrotor_mmaq_verification_v1 import verify_qualification
            print(verify_qualification(ROOT,args.upstream,out))
    except Exception as exc:
        q.safe_write(out/'latest_failure.json',{'status':'failed_or_incomplete','kind':type(exc).__name__,
                     'message':str(exc),'heldout_execution_complete':(out/'execution_complete.json').exists()})
        raise
    finally:
        if (out/'prior_usage.json').exists():
            usage=pilot.read(out/'usage.json') if (out/'usage.json').exists() else {'solver_calls':0,'physics_steps':0,'cpu_seconds':0.}
            # run_plan charges child workers itself; other CLI phases charge all children here.
            children=resource.getrusage(resource.RUSAGE_CHILDREN)
            child_cpu=0 if args.run_heldout else children.ru_utime+children.ru_stime-child_start.ru_utime-child_start.ru_stime
            charge=time.process_time()+child_cpu
            usage['cpu_seconds']+=charge;q.safe_write(out/'usage.json',usage)
            entry={'arguments':sys.argv[1:],'wall_seconds':time.monotonic()-start,'cpu_seconds_charged_here':charge}
            with (out/'cli_resource_log.jsonl').open('a') as stream:stream.write(__import__('json').dumps(entry)+'\n')


if __name__=='__main__':main()
