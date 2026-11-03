"""One-worker static audit supervisor with CPU/RSS limits; no MATLAB launch."""
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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--resume',action='store_true',help='Resume saved precision passes; never overwrite completed evidence')
    args=parser.parse_args()
    cfg=json.loads((ROOT/'configs/experiment_11b.json').read_text())
    out=(args.output or ROOT/cfg['output']).resolve()
    if out.is_relative_to((ROOT/cfg['accepted_evidence']).resolve()):
        raise ValueError('Cannot write into accepted E11A evidence')
    if args.worker:
        from src.drilling_static_audit import run
        run(ROOT,out,resume=args.resume)
        return
    if out.exists() and not args.resume:raise FileExistsError('Refusing to overwrite completed or partial evidence')
    if args.resume and (not out.exists() or (out/'completion.json').exists()):
        raise ValueError('Resume requires incomplete evidence')
    out.parent.mkdir(parents=True,exist_ok=True)
    env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
             VECLIB_MAXIMUM_THREADS='1',NUMEXPR_NUM_THREADS='1')
    start=time.monotonic();peak=0;stopped=None
    logpath=out.parent/(out.name+('_resume' if args.resume else '')+'_execution.log')
    with logpath.open('x') as log:
        p=subprocess.Popen([sys.executable,__file__,'--worker','--output',str(out)]+(['--resume'] if args.resume else []),
                           cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        while p.poll() is None:
            # A generous wall guard additionally prevents blocked I/O from hanging.
            if time.monotonic()-start>2*cfg['cpu_seconds_cap']:
                stopped='wall guard';p.kill()
            time.sleep(.25)
        code=p.wait()
    if (out/'summary.json').exists():
        peak=json.loads((out/'summary.json').read_text())['peak_rss_bytes']
    meta={'returncode':code,'stop_reason':stopped,'peak_sampled_rss_bytes':peak,
          'rss_sampling_seconds':.1,'rss_monitor':'Worker high-water RSS signal guard',
          'wall_seconds':time.monotonic()-start,
          'workers':1,'cpu_hard_limit_seconds':cfg['cpu_seconds_cap'],
          'rss_limit_bytes':cfg['rss_bytes_cap'],'new_simulations':0,'log':str(logpath)}
    out.mkdir(parents=True,exist_ok=True)
    (out/'execution_limits.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(meta,indent=2))
    if code or stopped:raise SystemExit(code or 1)


if __name__=='__main__':main()
