"""Fresh-output, read-only verification of accepted B0 records/checkpoints."""
from pathlib import Path
from unittest.mock import patch
import argparse,os,sys
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--data-root',type=Path,required=True);p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--full-preservation',action='store_true');a=p.parse_args()
    from src import quadrotor_archive_learner as learner
    from src.quadrotor_archive_finalization import verify
    def forbidden(*a,**k):raise AssertionError('No new scientific fitting in finalization')
    with patch.object(learner,'fit_fixed',forbidden),patch.object(learner,'ridge_fit',forbidden):r=verify(a.data_root,ROOT,a.upstream,a.output,full_preservation=a.full_preservation)
    assert not any(n in sys.modules for n in ['src.iql','src.offline_experiment','src.offline_data'])
    print({k:v for k,v in r.items() if k!='scientific_results'})

if __name__=='__main__':main()
