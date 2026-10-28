"""Run only the approved Experiment 07 technical pilot.

.venv/experiment07/bin/python experiments/07_recoverability.py --pilot [--resume]
"""

import argparse
import os
from pathlib import Path
import sys

for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.recoverability_experiment import run_pilot, worker

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--pilot', action='store_true')
    modes.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preservation-snapshot', type=Path)
    args = parser.parse_args()
    if args.worker:
        worker(ROOT, args.worker)
    else:
        run_pilot(ROOT, args.resume, args.preservation_snapshot)
