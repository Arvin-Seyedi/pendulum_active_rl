"""Verify accepted MPC80 records; never instantiate a controller or simulator."""
from pathlib import Path
import argparse
import os
import sys
from unittest.mock import patch
for name in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[name]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import casadi as c
    from src.quadrotor_mpc80_finalization import verify
    def forbidden(*args,**kwargs):raise AssertionError('New optimization forbidden in finalization')
    with patch.object(c.Opti,'solve',forbidden),patch.object(c.Opti,'solve_limited',forbidden),\
         patch.object(c,'nlpsol',forbidden),patch.object(c,'qpsol',forbidden),patch.object(c,'conic',forbidden):
        r=verify(a.data_root,ROOT,a.upstream,a.output)
    assert 'pybullet' not in sys.modules
    assert not any(name in sys.modules for name in ['src.iql','src.offline_data','src.offline_experiment'])
    print({k:v for k,v in r.items() if k!='scientific_results'})


if __name__=='__main__':main()
