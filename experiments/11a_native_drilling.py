"""Execute only the approved six-case, two-repeat, native ideal-input pilot."""
import argparse
import os
from pathlib import Path
import sys
os.environ.setdefault('MPLCONFIGDIR','/private/tmp/pendulum_11a_mpl')
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--run',action='store_true')
    g.add_argument('--audit',action='store_true');args=p.parse_args()
    if args.audit:
        from src.drilling_numerical_audit import audit
        audit(ROOT)
    else:
        from src.drilling_reproduction import freeze,run
        (run if args.run else freeze)(ROOT)
