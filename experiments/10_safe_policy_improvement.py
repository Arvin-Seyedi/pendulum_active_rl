"""Freeze selection, execute only the approved E10 pilot, or verify its archive."""

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
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-subset',action='store_true')
    mode.add_argument('--pilot',action='store_true')
    mode.add_argument('--verify',action='store_true')
    mode.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--preservation-snapshot',type=Path)
    args=parser.parse_args()
    if args.freeze_subset:
        from src.horizon_experiment import read_json
        from src.policy_improvement import freeze_selection
        result=freeze_selection(ROOT,read_json(ROOT/'configs/experiment_10_pilot.json'))
        print('Frozen 64-state subset; pilot IDs:',result['pilot_ids'])
    else:
        from src.policy_improvement_experiment import run_pilot,verify_completed,worker
        if args.worker: worker(ROOT,args.worker)
        elif args.verify:
            if args.resume or args.preservation_snapshot: parser.error('--verify accepts no execution options')
            verify_completed(ROOT)
        else:run_pilot(ROOT,args.resume,args.preservation_snapshot)
