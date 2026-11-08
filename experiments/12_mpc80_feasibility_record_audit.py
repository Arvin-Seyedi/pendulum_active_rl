"""Read-only audit after the FIRST stopped MPC80 feasibility job. No replay."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import time
import sys
from unittest.mock import patch
START=time.perf_counter()
for n in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS','MKL_NUM_THREADS']:os.environ[n]='1'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))


def main():
    q=argparse.ArgumentParser(__doc__);q.add_argument('--upstream',type=Path,required=True)
    q.add_argument('--study',type=Path,default=ROOT/'results/benchmark_selection/mpc80_feas_v1_study')
    q.add_argument('--preflight',type=Path,default=ROOT/'results/benchmark_selection/mpc80_feas_v1_preflight')
    q.add_argument('--label',choices=['first','repeat'],required=True);a=q.parse_args()
    from src.quadrotor_mpc80_stopped_audit_v1 import audit
    from src.quadrotor_mpc80_feasibility_v1 import durable_json,charge_overhead
    import casadi as c
    out=a.study/'readonly_audit_v1';out.mkdir(exist_ok=True)
    target=out/(a.label+'.json')
    if target.exists():raise FileExistsError('Completed audit preserved')
    forbidden=lambda *args,**kwargs:(_ for _ in ()).throw(RuntimeError('New optimization forbidden after stop'))
    original_files={str(p.relative_to(a.study)):hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in a.study.rglob('*') if p.is_file() and not p.is_relative_to(out)
                    and p.name!='resource_ledger.json'}
    with patch.object(c.Opti,'solve',forbidden),patch.object(c.Opti,'solve_limited',forbidden),\
         patch.object(c,'nlpsol',forbidden),patch.object(c,'qpsol',forbidden),patch.object(c,'conic',forbidden):
        result,sources=audit(ROOT,a.upstream,a.preflight,a.study)
    assert 'pybullet' not in sys.modules
    encoded=json.dumps(result,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    digest=hashlib.sha256(encoded).hexdigest()
    durable_json(target,result)
    if a.label=='first':durable_json(out/'source_evidence.json',sources)
    else:
        previous=json.loads((out/'first.json').read_text())
        assert result==previous,'Read-only audit repeat differs'
        durable_json(out/'repeatability.json',{'passed':True,'canonical_scientific_sha256':digest,
                                             'new_solves':0,'new_physics_steps':0})
    for name,h in original_files.items():assert hashlib.sha256((a.study/name).read_bytes()).hexdigest()==h,name
    durable_json(out/('integrity_'+a.label+'.json'),{'passed':True,'scientific_sha256':digest,
        'original_study_files_sha256':original_files,'read_only_source_sha256':{
            n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in
            ['src/quadrotor_mpc80_stopped_audit_v1.py','experiments/12_mpc80_feasibility_record_audit.py']}})
    cpu=math.ceil((time.process_time()+.1)*10)/10;wall=math.ceil((time.perf_counter()-START+.1)*10)/10
    charge_overhead(a.study/'resource_ledger.json',cpu,wall)
    durable_json(out/('resources_'+a.label+'.json'),{'charged_cpu_seconds':cpu,'charged_wall_seconds':wall,
        'remaining':json.loads((a.study/'resource_ledger.json').read_text())['charged']})
    print({'read_only_audit_passed':True,'complete_states':0,'complete_executions':0,
           'new_controller_solves':0,'new_physics_steps':0,'canonical_scientific_sha256':digest})


if __name__=='__main__':main()
