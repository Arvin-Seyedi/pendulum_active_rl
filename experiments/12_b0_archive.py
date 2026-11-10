"""Archive-only fixed BC gap assessment; never dispatches MPC or physics."""
from pathlib import Path
import argparse,os,sys
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['MPLCONFIGDIR']='/private/tmp/e12_matplotlib';sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))

def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--phase',choices=['seal','preflight','train','assess'],required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--upstream',type=Path,default=Path('/private/tmp/e12_safe_control_gym'));a=p.parse_args()
    from src import quadrotor_archive_experiment as b
    try:
        if a.phase=='seal':b.seal(ROOT,a.output)
        else:getattr(b,a.phase)(ROOT,a.upstream,a.output)
    except BaseException as exc:
        from src.quadrotor_mmaq_v1 import safe_write
        safe_write(a.output/(a.phase+'_failure.json'),{'kind':type(exc).__name__,'message':str(exc),'stage':a.phase,'MPC_solves':0,'physics_steps':0,'automatic_retry':False});raise

if __name__=='__main__':main()
