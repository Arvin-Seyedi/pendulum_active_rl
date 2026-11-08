"""Only the approved E12 four-case pilot; no screening/acquisition entry point."""
import argparse
from pathlib import Path
import os
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.quadrotor_pilot import read, write, freeze, run_episode, replay, sha


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--upstream', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT/'results/benchmark_selection/technical_pilot')
    p.add_argument('--prepare', action='store_true')
    p.add_argument('--run', action='store_true')
    p.add_argument('--verify', action='store_true')
    p.add_argument('--worker', choices=['episode', 'replay'])
    p.add_argument('--case', choices=['T0', 'T1', 'T2', 'T3'])
    p.add_argument('--controller', choices=['linear40', 'nonlinear20', 'nonlinear40', 'nonlinear80'])
    p.add_argument('--phase', choices=['original', 'repeat', 'refine'])
    p.add_argument('--frequency', type=int, choices=[1000, 2000], default=1000)
    p.add_argument('--folder', type=Path)
    p.add_argument('--instrument', action='store_true')
    a = p.parse_args(); out = a.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    cfg = read(ROOT/'configs/experiment_12.json')
    if a.worker:
        if a.worker == 'episode':
            result = run_episode(ROOT, a.upstream, out, a.case, a.controller, a.phase, a.frequency)
        else:
            result = replay(a.upstream, out, a.folder, a.instrument)
        print(result)
        return
    if a.prepare:
        frozen = freeze(ROOT, a.upstream, out, cfg)
        print({'prepared_cases': frozen['cases'], 'commit': cfg['upstream_commit']})
    if a.run:
        if not (out/'frozen.json').exists():
            raise SystemExit('Run --prepare before executing')
        if read(out/'frozen.json')['config'] != cfg:
            raise SystemExit('Frozen protocol changed')
        usage = {'cpu_seconds': 0., 'solver_calls': 0, 'physics_steps': 0}
        if (out/'usage.json').exists():
            usage = read(out/'usage.json')
        plan = [(k, c, 1000, phase) for k in cfg['cases'] for c in cfg['controllers'] for phase in cfg['repetitions']]
        plan += [(k, c, 2000, phase) for k in cfg['cases'] for c in cfg['controllers'] for phase in ('refine', 'repeat')]
        env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1',
                   MKL_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/e12_matplotlib', PYTHONDONTWRITEBYTECODE='1')
        def child(args, log):
            if usage['cpu_seconds'] >= cfg['maximum_cpu_seconds'] or usage['solver_calls'] >= cfg['maximum_solver_calls'] or usage['physics_steps'] >= cfg['maximum_physics_steps']:
                raise RuntimeError('Approved computational budget exhausted')
            before = resource.getrusage(resource.RUSAGE_CHILDREN)
            with log.open('w') as f:
                process = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--upstream', str(a.upstream),
                    '--output', str(out), *args], env=env, stdout=f, stderr=subprocess.STDOUT, timeout=600)
            after = resource.getrusage(resource.RUSAGE_CHILDREN)
            usage['cpu_seconds'] += after.ru_utime + after.ru_stime - before.ru_utime - before.ru_stime
            write(out/'usage.json', usage)
            if process.returncode:
                raise RuntimeError('Worker failed; preserve log ' + str(log))
        for case, controller, frequency, phase in plan:
            folder = out/'runs'/f'{case}_{controller}_{frequency}_{phase}'
            if not (folder/'completion.json').exists():
                log = out/f'{folder.name}.log'
                child(['--worker', 'episode', '--case', case, '--controller', controller,
                       '--frequency', str(frequency), '--phase', phase], log)
                record = read(folder/'completion.json')
                usage['solver_calls'] += record['solver_calls']
                usage['physics_steps'] += record['transitions'] * (frequency//50)
                write(out/'usage.json', usage)
                print(folder.name, record['complete'], record['failure'], flush=True)
            for instrument in (False, True):
                name = 'replay_instrumented' if instrument else 'replay_native'
                if not (folder/(name+'.json')).exists():
                    extra = ['--instrument'] if instrument else []
                    child(['--worker', 'replay', '--folder', str(folder), *extra], out/f'{folder.name}_{name}.log')
                    usage['physics_steps'] += read(folder/'completion.json')['transitions'] * (frequency//50)
                    write(out/'usage.json', usage)
        write(out/'execution_complete.json', {'conditions': len(plan), 'usage': usage, 'screening_executed': False,
                                             'counterfactuals_executed': False, 'training_executed': False})
    if a.verify:
        from src.quadrotor_verification import verify
        print(verify(ROOT, a.upstream, out))


if __name__ == '__main__':
    main()
