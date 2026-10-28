"""Run/resume the approved full Experiment 07; reuse the accepted pilot."""

import argparse
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.recoverability_full_study import full_worker, run_full

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--run', action='store_true')
    group.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preservation-snapshot', type=Path)
    args = parser.parse_args()
    if args.worker: full_worker(ROOT, args.worker)
    else: run_full(ROOT, args.resume, args.preservation_snapshot)
