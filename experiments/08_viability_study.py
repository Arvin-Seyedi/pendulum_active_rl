"""Run the approved remaining-state analysis or verify completed results."""

import argparse
import os
from pathlib import Path
import sys

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.viability_study import run_study, verify_study

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run', action='store_true')
    mode.add_argument('--verify', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preservation-snapshot', type=Path)
    args = parser.parse_args()
    if args.verify:
        if args.resume or args.preservation_snapshot: parser.error('--verify accepts no execution options')
        verify_study(ROOT)
    else:
        run_study(ROOT, args.resume, args.preservation_snapshot)
