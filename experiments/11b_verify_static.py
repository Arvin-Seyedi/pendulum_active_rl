"""Recheck accepted E11B static evidence without writing to its archive."""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root',type=Path,default=ROOT,help='Explicit accepted scientific-data/Git root')
    parser.add_argument('--output',type=Path,required=True,help='New JSON outside accepted archives')
    args=parser.parse_args()
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[key]='1'
    from src.drilling_static_finalization import finalize
    result=finalize(args.project_root,ROOT,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='independent_verification'},indent=2))


if __name__=='__main__':main()
