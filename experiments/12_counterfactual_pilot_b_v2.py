"""Exact Pilot B v2 entrypoint. Dry dispatch is inert; live needs NEW approval."""
from pathlib import Path
import argparse,json,os,resource,subprocess,sys,time
PROCESS_START_WALL=time.perf_counter()
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['MPLCONFIGDIR']='/private/tmp/e12_matplotlib'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src import quadrotor_counterfactual_dispatch_v2 as d
from src import quadrotor_pilot as p
from src.quadrotor_mmaq_v1 import safe_write
from src import quadrotor_mpc80_feasibility_v1 as budget
REVIEW=ROOT/'results/benchmark_selection/counterfactual_wiring_review_v2'


def prepare(upstream,out,approval_path):
    """Only after fresh approval; never make an approval from an old prompt."""
    if approval_path is None or os.environ.get('E12_FORCE_ZERO_SOLVE')=='1':raise PermissionError('Fresh execution approval required')
    if out.exists():raise FileExistsError('Existing attempt retained')
    a=p.read(approval_path);seal=p.read(REVIEW/'review_seal.json');prior=p.read(REVIEW/'resource_ledger.json')
    if not (a.get('execution_approved') is True and a.get('purpose')=='FRESH_PILOT_B_EXECUTION' and a.get('dry_dispatch') is False
            and a.get('seal_sha256')==p.sha(REVIEW/'review_seal.json') and a.get('preflight_report_sha256')==p.sha(REVIEW/'scientific_report.md') and a.get('review_reference')):
        raise PermissionError('Approval not bound to reviewed wiring')
    proposal=prior['proposed_execution_caps']
    if (a.get('cumulative_CPU_ceiling'),a.get('cumulative_wall_ceiling'))!=(proposal['cpu_seconds'],proposal['wall_seconds']):raise PermissionError('Resource extension requires explicit approval')
    for n,h in seal['source_sha256'].items():
        if p.sha(ROOT/n)!=h:raise ValueError('Reviewed source changed: '+n)
    for n,h in seal['artifact_sha256'].items():
        if p.sha(REVIEW/n)!=h:raise ValueError('Reviewed evidence changed: '+n)
    from src import quadrotor_counterfactual_execution_v2 as e
    e.checkpoint(ROOT,upstream)
    out.mkdir(parents=True)
    for menu in [4,3]:
        b=out/'branches'/f'menu{menu}';b.mkdir(parents=True)
        for n in ['frozen.json','reference.npy']:(b/n).write_bytes((ROOT/e.ACCEPTED/n).read_bytes())
    safe_write(out/'approval.json',a)
    state=d.ledger(ROOT,dry=False,carry=prior['charged_cumulative'],seal_sha=a['seal_sha256'],report_sha=a['preflight_report_sha256'],
        cpu_cap=a['cumulative_CPU_ceiling'],wall_cap=a['cumulative_wall_ceiling'])
    budget.durable_json(out/'resource_ledger.json',state)
    files=['approval.json']+[f'branches/menu{m}/{n}' for m in [4,3] for n in ['frozen.json','reference.npy']]
    safe_write(out/'execution_binding.json',{'review_seal_sha256':a['seal_sha256'],'review_report_sha256':a['preflight_report_sha256'],
        'source_sha256':seal['source_sha256'],'artifact_sha256':{n:p.sha(out/n) for n in files}})


def dry(out,fail_at=None,gate_fail=False):
    if out.exists():raise FileExistsError('Dry evidence retained; no implicit overwrite')
    out.mkdir(parents=True)
    # Simulation of reservation accounting ONLY; no real scientific budget changes.
    state=d.ledger(ROOT,dry=True,carry={'cpu_seconds':0.,'wall_seconds':0.},seal_sha='mock-seal',report_sha='mock-report',cpu_cap=300,wall_cap=600)
    budget.durable_json(out/'resource_ledger.json',state)
    approval={'execution_approved':True,'dry_dispatch':True,'seal_sha256':'mock-seal','preflight_report_sha256':'mock-report','review_reference':'INERT dispatcher fixture; no scientific permission'}
    trace=[]
    def launch(job):
        trace.append(job['id']);safe_write(out/'dry_trace.json',trace)
        if job['id']==fail_at:raise RuntimeError('Injected dry worker failure')
        return {'passed':True,'mock':True,'job':job,'new_actual_solves':0,'new_actual_physics_steps':0}
    from src.quadrotor_mpc80_feasibility_v2 import zero_real_operations
    with zero_real_operations(Path('/private/tmp/e12_safe_control_gym')) as (forbidden,engine):
        try:d.dispatch(ROOT,out,approval,launch,gate=lambda path:not gate_fail)
        finally:safe_write(out/'zero_operations.json',{'forbidden_calls':forbidden,'engine_calls':engine,'actual_solves':0,'actual_physics_steps':0})
        assert not forbidden and not engine


def worker(upstream,out,job_id):
    from src import quadrotor_counterfactual_execution_v2 as e
    e.checkpoint(ROOT,upstream,out)
    ledger=p.read(out/'resource_ledger.json');receipt=ledger['reservations'][-1]
    if receipt['id']!=job_id or receipt['status']!='reserved' or ledger['halted'] or ledger['dry_dispatch']:raise PermissionError('No live reservation')
    # Approval is checked again in every child, not inferred from a prior prompt.
    approval=p.read(out/'approval.json');validate_approval(out,approval)
    left=ledger['caps']['aggregate_cpu_seconds']-ledger['charged']['cpu_seconds'];resource.setrlimit(resource.RLIMIT_CPU,(max(1,int(left)),max(2,int(left)+1)))
    if receipt['kind']=='scientific':result=e.episode(ROOT,upstream,out,receipt['menu_index'],receipt['phase'])
    else:result=e.replay(upstream,out,receipt['menu_index'],receipt['phase'],receipt['kind'])
    if result.get('passed') is not True:raise RuntimeError('Scientific/replay verification did not pass')
    safe_write(out/'worker_results'/f'{job_id}.json',{'passed':True,'mock':False,'job':receipt,'result':result})


def validate_approval(out,approval):
    if os.environ.get('E12_FORCE_ZERO_SOLVE')=='1':raise PermissionError('Zero-solve test process forbids live dispatch')
    if not approval or approval.get('execution_approved') is not True or approval.get('purpose')!='FRESH_PILOT_B_EXECUTION' or approval.get('dry_dispatch') is not False:
        raise PermissionError('Scientific execution PAUSED; fresh explicit approval required')
    binding=p.read(out/'execution_binding.json')
    if approval.get('seal_sha256')!=binding['review_seal_sha256'] or approval.get('preflight_report_sha256')!=binding['review_report_sha256'] or not approval.get('review_reference'):
        raise PermissionError('Approval not bound to reviewed wiring')
    state=p.read(out/'resource_ledger.json');d.require_scope(ROOT,state)
    if state['dry_dispatch'] or state['caps']['aggregate_cpu_seconds']!=approval.get('cumulative_CPU_ceiling') or state['caps']['wall_seconds']!=approval.get('cumulative_wall_ceiling'):
        raise PermissionError('Approved cumulative resource caps differ')


def execute(upstream,out,approval_path):
    # Preparation is explicit and immutable; the paused review creates NO live
    # output tree or approval. A fresh authorized preparation must supply them.
    if approval_path is None:raise PermissionError('Scientific execution PAUSED; fresh approval required')
    approval=p.read(approval_path);validate_approval(out,approval)
    if (out/'campaign_stop.json').exists() or (out/'dispatch_complete.json').exists():raise RuntimeError('Existing completed/stopped attempt cannot resume')
    from src import quadrotor_counterfactual_execution_v2 as e
    # Include startup/imports, approval checks and preparation; no free clock reset.
    last_cpu=0.;last_wall=PROCESS_START_WALL
    def account():
        nonlocal last_cpu,last_wall
        now=time.process_time()+_children_cpu();wall=time.perf_counter()
        budget.charge_overhead(out/'resource_ledger.json',max(0,now-last_cpu),max(0,wall-last_wall));last_cpu,last_wall=now,wall
    def check():e.checkpoint(ROOT,upstream,out);account()
    def launch(job):
        state=p.read(out/'resource_ledger.json');timeout=state['caps']['wall_seconds']-state['charged']['wall_seconds']
        try:
            with (out/(job['id']+'.log')).open('x') as stream:
                r=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--mode','worker','--upstream',str(upstream),'--output',str(out),'--job-id',job['id']],stdout=stream,stderr=subprocess.STDOUT,timeout=timeout)
            if r.returncode:raise RuntimeError('Scientific/replay worker failed; inspect retained records')
            return p.read(out/'worker_results'/f"{job['id']}.json")
        finally:account()
    try:d.dispatch(ROOT,out,approval,launch,checkpoint=check)
    finally:account()


def _children_cpu():
    r=resource.getrusage(resource.RUSAGE_CHILDREN);return r.ru_utime+r.ru_stime


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--mode',choices=['dry-dispatch','execute','worker'],required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--upstream',type=Path,default=Path('/private/tmp/e12_safe_control_gym'))
    parser.add_argument('--approval',type=Path);parser.add_argument('--job-id');parser.add_argument('--fail-at');parser.add_argument('--gate-fail',action='store_true');a=parser.parse_args()
    if a.mode=='dry-dispatch':dry(a.output.resolve(),a.fail_at,a.gate_fail)
    elif a.mode=='execute':
        if not a.output.exists():prepare(a.upstream.resolve(),a.output.resolve(),a.approval)
        execute(a.upstream.resolve(),a.output.resolve(),a.approval)
    else:
        try:worker(a.upstream.resolve(),a.output.resolve(),a.job_id)
        except BaseException as exc:
            safe_write(a.output/'worker_results'/f'{a.job_id}_failure.json',{'kind':type(exc).__name__,'message':str(exc),'complete':False,'fallback_action_applied':False});raise


if __name__=='__main__':main()
