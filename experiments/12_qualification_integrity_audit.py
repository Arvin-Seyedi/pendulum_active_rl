"""Static integrity audit ONLY; optimizer solves and PyBullet are forbidden."""
import os
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import argparse
import resource
import sys
import time
from unittest.mock import patch
import casadi as c
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import quadrotor_pilot as p
from src.quadrotor_mmaq_v1 import safe_write
from src.quadrotor_qualification_integrity_audit import retained_kkt,qrqp_audit,latency_audit,prepare_audit


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'results/benchmark_selection/mmaq_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'results/benchmark_selection/qualification_integrity_v2')
    parser.add_argument('--repeat-label',default='original')
    parser.add_argument('--upstream',type=Path,default=Path('/private/tmp/e12_safe_control_gym'))
    a=parser.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    if (a.output/'completion.json').exists():raise RuntimeError('Sealed audit is immutable; choose a fresh --output')
    start=time.perf_counter()
    def forbidden(*args,**kwargs):raise AssertionError('Optimization calls are not authorized in this audit')
    try:
        prepare_audit(ROOT,a.source,a.output,a.upstream)
        with patch.object(c.Opti,'solve',forbidden),patch.object(c.Opti,'solve_limited',forbidden):
            kkt=retained_kkt(a.source,a.output);qr=qrqp_audit(a.source,a.output);lat=latency_audit(a.source,a.output)
        assert 'pybullet' not in sys.modules,'Audit imported a simulator unexpectedly'
        print({'retained_candidates_checked':kkt['retained_candidates'],
               'MPC20_unresolved_records':len(kkt['missing_rejected_candidates']),
               'QRQP_failures':[(r['record'],r['last_trace']['iteration']) for r in qr['failed_records']],
               'maximum_successful_QRQP_iterations_including_prefixes':qr['maximum_successful_iterations_including_prefixes'],
               'new_episodes':0,'new_solver_calls':0,'new_physics_steps':0})
    finally:
        safe_write(a.output/f'audit_resources_{a.repeat_label}.json',{
            'cpu_seconds':time.process_time(),'wall_seconds':time.perf_counter()-start,
            'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            'new_solver_calls':0,'new_physics_steps':0,'new_episodes':0})


if __name__=='__main__':main()
