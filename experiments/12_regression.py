"""Run one E12 finalization regression group without controller optimization.

Run quadrotor with requirements_experiment_12.txt and prior with the established
prior-experiment environment. Unfinished Experiment 06 is deliberately excluded.
"""
from pathlib import Path
import argparse
import json
import os
import resource
import sys
import time
import unittest
from unittest.mock import patch

for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
    os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/'tests'))


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--group',choices=['quadrotor','prior'],required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import casadi as c
    def forbidden(*a,**kw):raise AssertionError('Controller solves forbidden in milestone regression')
    started=time.process_time()
    with patch.object(c.Opti,'solve',forbidden),patch.object(c.Opti,'solve_limited',forbidden):
        suite=unittest.TestSuite()
        for path in sorted((ROOT/'tests').glob('test_*.py')):
            if path.name=='test_offline_rl.py':continue
            if path.name.startswith('test_quadrotor')!=(args.group=='quadrotor'):continue
            suite.addTests(unittest.defaultTestLoader.loadTestsFromName(path.stem))
        args.output.parent.mkdir(parents=True,exist_ok=True)
        log=args.output.with_suffix('.log')
        if log.exists():raise FileExistsError(log)
        with log.open('w') as stream:
            result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    evidence={'passed':result.wasSuccessful(),'group':args.group,'tests':result.testsRun,
              'failures':len(result.failures),'errors':len(result.errors),'skipped':len(result.skipped),
              'python':sys.version,'cpu_seconds':time.process_time()-started,
              'peak_rss_native':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
              'IQL_excluded':True,'MPC_solves_forbidden':True,'PyBullet_imported':'pybullet' in sys.modules}
    args.output.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence))
    if not result.wasSuccessful():raise SystemExit(1)


if __name__=='__main__':main()
