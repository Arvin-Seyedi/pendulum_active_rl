"""Only the approved four cases, linear QRQP adapter, and 1000 Hz physics."""
import argparse
from pathlib import Path
import os
import resource
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import quadrotor_pilot as pilot
from src.quadrotor_linear_revalidation import prepare_revalidation,preflight,episode,replay_executed_motor_commands


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--upstream',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'results/benchmark_selection/linear_qrqp_revalidation')
    p.add_argument('--prepare',action='store_true')
    p.add_argument('--preflight',action='store_true');p.add_argument('--run',action='store_true');p.add_argument('--verify',action='store_true')
    p.add_argument('--worker',choices=['episode','native_replay','motor_replay'])
    p.add_argument('--case',choices=['T0','T1','T2','T3']);p.add_argument('--phase',choices=['original','repeat'])
    a=p.parse_args();out=a.output.resolve()
    if a.prepare:print(prepare_revalidation(ROOT,a.upstream,out))
    assert (out/'preservation_start.json').exists()
    if a.preflight:
        if (out/'preflight.json').exists():print(pilot.read(out/'preflight.json'))
        else:print(preflight(a.upstream,out))
    if a.worker:
        folder=out/'runs'/f'{a.case}_linear40_1000_{a.phase}'
        if a.worker=='episode':r=episode(a.upstream,out,a.case,a.phase)
        elif a.worker=='native_replay':r=pilot.replay(a.upstream,out,folder,False)
        else:r=replay_executed_motor_commands(a.upstream,out,folder)
        print(r);return
    if a.run:
        assert pilot.read(out/'preflight.json')['program_graph_unchanged']
        cfg=pilot.read(out/'frozen.json')['config'];prior=pilot.read(ROOT/'results/benchmark_selection/technical_pilot/usage.json')
        usage=pilot.read(out/'usage.json') if (out/'usage.json').exists() else {'cpu_seconds':0.,'solver_calls':0,'physics_steps':0}
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',MKL_NUM_THREADS='1',MPLCONFIGDIR='/private/tmp/e12_matplotlib')
        for case in ['T0','T1','T2','T3']:
            for phase in ['original','repeat']:
                folder=out/'runs'/f'{case}_linear40_1000_{phase}'
                for job,marker in [('episode','revalidation_complete.json'),('native_replay','replay_native.json'),('motor_replay','replay_instrumented.json')]:
                    if (folder/marker).exists():continue
                    assert prior['cpu_seconds']+usage['cpu_seconds']<cfg['maximum_cpu_seconds']
                    assert prior['solver_calls']+usage['solver_calls']+300<=cfg['maximum_solver_calls']
                    assert prior['physics_steps']+usage['physics_steps']+6000<=cfg['maximum_physics_steps']
                    start=resource.getrusage(resource.RUSAGE_CHILDREN)
                    log=out/f'{case}_{phase}_{job}.log'
                    with log.open('w') as stream:
                        r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--upstream',str(a.upstream),'--output',str(out),
                            '--worker',job,'--case',case,'--phase',phase],env=env,stdout=stream,stderr=subprocess.STDOUT,timeout=600)
                    end=resource.getrusage(resource.RUSAGE_CHILDREN)
                    usage['cpu_seconds']+=end.ru_utime+end.ru_stime-start.ru_utime-start.ru_stime
                    pilot.write(out/'usage.json',usage)
                    if r.returncode:raise RuntimeError('Worker failed; retained log '+str(log))
                    m=pilot.read(folder/'completion.json')
                    usage['solver_calls']+=m['solver_calls'] if job=='episode' else 0
                    usage['physics_steps']+=20*m['transitions'];pilot.write(out/'usage.json',usage)
                print(case,phase,pilot.read(folder/'completion.json')['complete'],flush=True)
        pilot.write(out/'execution_complete.json',{'episodes_attempted':8,'usage':usage,'nonlinear_runs_executed':0,'frequency':1000,
            'screening_executed':False,'action_probes_executed':False,'RL_training_executed':False})
    if a.verify:
        from src.quadrotor_linear_verification import verify_revalidation
        r=verify_revalidation(ROOT,a.upstream,out);print({'records':len(r['records']),'original_model_gate':r['original_model_agreement_gate']})


if __name__=='__main__':main()
