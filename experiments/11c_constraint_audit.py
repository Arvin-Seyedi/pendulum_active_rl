"""Run only the four-snapshot static E11C pilot, with one numerical worker."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args();cfg=json.loads((ROOT/'configs/experiment_11c.json').read_text())
    out=(args.output or ROOT/cfg['output']).resolve()
    for key in ('accepted_11b','native_11a'):
        if out.is_relative_to((ROOT/cfg[key]).resolve()):raise ValueError('Output inside prior evidence')
    if out.exists():raise FileExistsError('Refusing to overwrite completed or failed evidence')
    if args.worker:
        import traceback
        try:
            from src.drilling_constraint_audit import run,preserve
            from src.drilling_constraint_verification import verify
            from src.drilling_static_audit import enforce_limits
            run(ROOT,out);enforce_limits(cfg);verify(ROOT,out)
            preserve(ROOT,out,json.loads((out/'preservation_before.json').read_text()))
            import signal
            signal.setitimer(signal.ITIMER_REAL,0)
        except BaseException:
            out.mkdir(parents=True,exist_ok=True)
            (out/'failure.txt').write_text(traceback.format_exc())
            raise
        return
    out.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
             MKL_NUM_THREADS='1',VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    started=time.monotonic();timeout=False
    logfile=out.parent/(out.name+'_execution.log')
    with logfile.open('x') as log:
        child=subprocess.Popen([sys.executable,__file__,'--worker','--output',str(out)],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        while child.poll() is None:
            if time.monotonic()-started>2*cfg['cpu_seconds_cap']:timeout=True;child.kill()
            time.sleep(.25)
        code=child.wait()
    out.mkdir(parents=True,exist_ok=True)
    execution={'returncode':code,'wall_timeout':timeout,'wall_seconds':time.monotonic()-started,
        'workers':1,'cpu_seconds_cap':cfg['cpu_seconds_cap'],'rss_bytes_cap':cfg['rss_bytes_cap'],
        'resource_enforcement':'RLIMIT_CPU and 0.1-second in-process high-water RSS guard',
        'new_trajectories':0,'controller_calls':0,'log':str(logfile)}
    (out/'execution.json').write_text(json.dumps(execution,indent=2)+'\n');print(json.dumps(execution,indent=2))
    if code or timeout:raise SystemExit(code or 1)


if __name__=='__main__':main()
