"""Read-only accepted Pilot B audit; no controller or physics execution."""
from pathlib import Path
import argparse,os,sys
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--full-preservation',action='store_true');a=p.parse_args()
    from src.quadrotor_counterfactual_finalization import verify
    result=verify(a.data_root,ROOT,a.upstream,a.output,full_preservation=a.full_preservation)
    assert not any(n in sys.modules for n in ['src.iql','src.offline_data','src.offline_experiment'])
    print({k:v for k,v in result.items() if k!='scientific_results'})

if __name__=='__main__':main()
