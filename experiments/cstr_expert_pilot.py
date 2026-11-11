"""Frozen expert-only CSTR technical pilot: zero-solve preflight then gated run."""
from pathlib import Path
import argparse,json,sys,os
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[n]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ.setdefault('MPLCONFIGDIR','/private/tmp/cstr_mpl')
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
def main():
    a=argparse.ArgumentParser(__doc__);a.add_argument('stage',choices=['plan','preflight','run']);a.add_argument('--output',type=Path,required=True);args=a.parse_args()
    from src import cstr_expert_pilot as p
    if args.stage=='plan':result=p.jobs(p.read(ROOT/p.CONFIG))
    elif args.stage=='preflight':
        from src.cstr_pilot_preflight import preflight
        result=preflight(ROOT,args.output)
    else:result=p.run(ROOT,args.output)
    print(json.dumps(result))
if __name__=='__main__':main()
