"""Static saved-record investigation and standalone optimization tests only."""
import argparse
from pathlib import Path
import json
import resource
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.quadrotor_diagnostic import saved_transition_audit,standalone_solver_audit


def main():
    p=argparse.ArgumentParser(__doc__)
    p.add_argument('--pilot',type=Path,default=ROOT/'results/benchmark_selection/technical_pilot')
    p.add_argument('--output',type=Path,default=ROOT/'results/benchmark_selection/compatibility_diagnostic')
    p.add_argument('--solver',action='store_true');p.add_argument('--transitions',action='store_true')
    a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
    if not (a.output/'start_integrity.json').exists():raise RuntimeError('Freeze accepted evidence identities first')
    resource.setrlimit(resource.RLIMIT_CPU,(600,600))
    cpu=time.process_time();wall=time.perf_counter()
    if a.solver:
        if (a.output/'solver_diagnosis.json').exists():raise RuntimeError('Refusing to overwrite completed solver investigation')
        r=standalone_solver_audit(a.output)
        print({'availability':r['availability'],'toy_comparisons':len(r['toy_cases'])},flush=True)
    if a.transitions:
        if (a.output/'transition_diagnosis.json').exists():raise RuntimeError('Refusing to overwrite completed transition investigation')
        r=saved_transition_audit(a.pilot,a.output)
        print({'archived_control_intervals':r['control_intervals'],'archived_physics_intervals':r['physics_intervals'],
               'native_error':r['global_components']['native_total'],'recurrence_error':r['global_components']['remaining_recurrence_residual']},flush=True)
    performance={'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-wall,
                 'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'new_physics_steps':0,'quadrotor_episodes':0}
    (a.output/('solver_performance.json' if a.solver else 'transition_performance.json')).write_text(json.dumps(performance,indent=2)+'\n')


if __name__=='__main__':main()
