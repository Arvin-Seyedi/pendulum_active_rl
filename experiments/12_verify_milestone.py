"""Immutable saved-record verification of the complete E12 milestone."""
from pathlib import Path
import argparse
import os
import sys
from unittest.mock import patch
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--data-root',type=Path,required=True)
    p.add_argument('--upstream',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import casadi as c
    from src.quadrotor_finalization import finalize
    def forbidden(*args,**kwargs):raise AssertionError('New optimization is forbidden during finalization')
    with patch.object(c.Opti,'solve',forbidden),patch.object(c.Opti,'solve_limited',forbidden):
        result=finalize(a.data_root,ROOT,a.upstream,a.output)
    assert 'pybullet' not in sys.modules
    s=result['scientific_results']
    print({'attempts':s['trajectories']['attempts'],'retained_candidates':s['candidates']['total'],
           'H2_v2':s['targeted_H2_v2_qualification'],'overall':s['overall_qualification'],
           'new_solves':0,'new_physics_steps':0})


if __name__=='__main__':main()
