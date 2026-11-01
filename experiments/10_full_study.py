"""Run/resume the frozen full Experiment 10 study, or verify saved outputs."""

import argparse
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    modes=parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--run',action='store_true')
    modes.add_argument('--verify',action='store_true')
    modes.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preservation-snapshot',type=Path)
    args=parser.parse_args()
    from src.policy_improvement_study import run_study,verify_completed,worker
    if args.worker:worker(ROOT,args.worker)
    elif args.verify:
        if args.resume or args.preservation_snapshot:parser.error('--verify accepts no execution options')
        verify_completed(ROOT)
    else:run_study(ROOT,args.resume,args.preservation_snapshot)
