"""Read-only audit of a completed/stopped v2 campaign; never runs/retries jobs."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import resource
import signal
import sys
import time
from unittest.mock import patch
START=time.perf_counter()
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[n]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    q=argparse.ArgumentParser(__doc__);q.add_argument('--upstream',type=Path,required=True)
    q.add_argument('--preflight',type=Path,default=ROOT/'results/benchmark_selection/mpc80_feas_v2_preflight')
    q.add_argument('--study',type=Path,default=ROOT/'results/benchmark_selection/mpc80_feas_v2_study')
    a=q.parse_args();audit=a.study/'read_only_audit'
    if audit.exists():raise FileExistsError('Existing audit preserved')
    if not ((a.study/'campaign_stop.json').exists() or (a.study/'execution_complete.json').exists()):
        raise RuntimeError('Scientific worker still active; no second computational worker')
    import casadi as c
    from src.quadrotor_mpc80_report_v2 import summarize,markdown
    from src.quadrotor_mpc80_feasibility_v1 import durable_json,charge_overhead
    from src.quadrotor_mmaq_v1 import tagged
    ledger=json.loads((a.study/'resource_ledger.json').read_text())
    cpu_left=1800-ledger['charged']['cpu_seconds'];wall_left=3600-ledger['charged']['wall_seconds']
    if min(cpu_left,wall_left)<=0:raise RuntimeError('No remaining shared verification budget')
    resource.setrlimit(resource.RLIMIT_CPU,(max(1,int(cpu_left)),max(2,int(cpu_left)+1)))
    def expired(*args):raise TimeoutError('Shared active-wall limit reached during read-only audit')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,wall_left)
    def forbidden(*args,**kwargs):raise RuntimeError('New optimization forbidden in report audit')
    with patch.object(c.Opti,'solve',forbidden),patch.object(c.Opti,'solve_limited',forbidden),\
         patch.object(c,'nlpsol',forbidden),patch.object(c,'qpsol',forbidden),patch.object(c,'conic',forbidden):
        result=summarize(ROOT,a.upstream,a.preflight,a.study)
    assert 'pybullet' not in sys.modules
    audit.mkdir();durable_json(audit/'scientific_results.json',result)
    digest=hashlib.sha256(json.dumps(tagged(result),sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    cpu=math.ceil((time.process_time()+.1)*10)/10;wall=math.ceil((time.perf_counter()-START+.1)*10)/10
    charge_overhead(a.study/'resource_ledger.json',cpu,wall)
    signal.setitimer(signal.ITIMER_REAL,0)
    durable_json(audit/'audit_resources.json',{'CPU_seconds_charged':cpu,'active_wall_seconds_charged':wall,
        'scientific_sha256':digest,'new_solves':0,'new_physics_steps':0})
    resource_after=json.loads((a.study/'resource_ledger.json').read_text())['charged']
    report=markdown(result,resource_after)
    doc=ROOT/'docs/experiment_12_mpc80_feas_v2_report.md'
    if doc.exists():raise FileExistsError('Existing scientific report preserved')
    doc.write_text(report);(audit/'scientific_report.md').write_text(report)
    print({'expert_gate_passed':result['new_expert_feasibility_gate_passed'],
        'complete_state_pairs':result['complete_initial_state_pairs'],
        'complete_executions':result['complete_scientific_rollouts'],
        'qualified_executions':result['qualified_scientific_executions'],
        'accepted_candidates':result['independently_accepted_candidates'],'sha256':digest})


if __name__=='__main__':main()
