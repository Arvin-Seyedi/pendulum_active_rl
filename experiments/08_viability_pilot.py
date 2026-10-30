"""Run only the approved five-case viability pilot, or verify saved results."""

import argparse
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.viability_experiment import run_pilot, verify_completed

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pilot', action='store_true')
    mode.add_argument('--verify', action='store_true', help='Verify completed hashes only; never compute trajectories.')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preservation-snapshot', type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.resume or args.preservation_snapshot:
            parser.error('--verify does not accept execution options')
        verify_completed(ROOT)
    else:
        run_pilot(ROOT, args.resume, args.preservation_snapshot)
