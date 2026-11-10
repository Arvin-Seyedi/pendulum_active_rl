"""Pilot B reservation/ordering/stop supervisor. No controller code here."""
from pathlib import Path
from src import quadrotor_pilot as p
from src import quadrotor_mpc80_feasibility_v1 as accounting
from src.quadrotor_mmaq_v1 import safe_write


def frozen_plan(root):
    jobs=p.read(Path(root)/'results/benchmark_selection/counterfactual_v1_preflight/pilot_B_jobs.json')
    assert len(jobs)==12 and sum(j['kind']=='scientific' for j in jobs)==4
    assert sum(j['solver_calls'] for j in jobs)==1200 and sum(j['physics_steps'] for j in jobs)==72000
    assert all(j['state_id']==0 and j['menu_index'] in [4,3] for j in jobs)
    return jobs


def ledger(root,*,dry,carry,seal_sha,report_sha,cpu_cap,wall_cap):
    return {'version':'E12_COUNTERFACTUAL_DISPATCH_v2','dry_dispatch':bool(dry),'preflight_passed':True,
        'seal_sha256':seal_sha,'preflight_report_sha256':report_sha,'halted':False,'reservations':[],
        'jobs':frozen_plan(root),'caps':{'solver_calls':1200,'physics_steps':72000,'aggregate_cpu_seconds':cpu_cap,'wall_seconds':wall_cap},
        'charged':{'solver_calls':0,'physics_steps':0,'cpu_seconds':carry['cpu_seconds'],'wall_seconds':carry['wall_seconds']},
        'original_starting_allowance':{'solver_calls':3070,'physics_steps':366000}}


def require_scope(root,state):
    if state['jobs']!=frozen_plan(root):raise ValueError('Pilot B plan altered, reordered, duplicated or expanded')
    if state['caps']['solver_calls']!=1200 or state['caps']['physics_steps']!=72000:raise ValueError('Pilot B count caps altered')
    if state['halted']:raise RuntimeError('Stopped campaign retained; no retry')


def nominal_gate(out):
    for phase in ['original','repeat']:
        folder=Path(out)/'branches/menu4/runs'/f'0_nonlinear80_1000_{phase}'
        meta=p.read(folder/'completion.json');check=p.read(folder/'independent_episode_check.json');fidelity=p.read(folder/'archived_nominal_fidelity.json')
        assert meta['complete'] and meta['transitions']==300 and check['passed'] and check['raw_states']==6001
        assert all(fidelity['arrays_bitwise'].values()) and fidelity['all_300_candidate_vectors_bitwise']
        for name in ['replay_native.json','replay_instrumented.json']:
            record=p.read(folder/name);assert record['steps']==300
            assert record['control_states_bitwise'] and record['rewards_bitwise']
            if name=='replay_instrumented.json':assert record['physics_states_bitwise'] and record['contacts_bitwise']
        count=p.read(folder/'submitted_motor_replay_counts.json');assert count['complete_6000'] and count['error_records']==0
    return True


def dispatch(root,out,approval,launch,*,checkpoint=lambda:None,gate=nominal_gate):
    """Exactly this loop is used by live dispatch and isolated dry-dispatch tests."""
    out=Path(out);path=out/'resource_ledger.json';state=p.read(path);require_scope(root,state)
    mode=state['dry_dispatch']
    if approval.get('dry_dispatch') is not mode:raise PermissionError('Approval mode mismatch')
    if not mode and approval.get('purpose')!='FRESH_PILOT_B_EXECUTION':raise PermissionError('Fresh scientific approval missing')
    try:
        for job in state['jobs']:
            checkpoint();current=p.read(path);require_scope(root,current)
            if job['id']==state['jobs'][6]['id']:
                if not gate(out):raise RuntimeError('Nominal gate failed before alternative')
                safe_write(out/'nominal_gate.json',{'passed':True,'dry_dispatch':mode})
            receipt=accounting.reserve_job(path,job['id'],approval)
            try:
                result=launch(receipt)
                if not isinstance(result,dict) or result.get('passed') is not True:raise RuntimeError('Worker failed or incomplete')
                if result.get('mock') is not mode:raise RuntimeError('Mock/live worker result mismatch')
                safe_write(out/'job_results'/f"{job['id']}.json",result)
            except BaseException as exc:
                accounting.finish_job(path,job['id'],False)
                safe_write(out/'job_results'/f"{job['id']}_failure.json",{'kind':type(exc).__name__,'message':str(exc),'job':receipt,
                    'fallback_action_applied':False,'dry_dispatch':mode})
                raise
            accounting.finish_job(path,job['id'],True)
        safe_write(out/'dispatch_complete.json',{'jobs':12,'scientific_runs':4,'replays':8,'dry_dispatch':mode})
    except BaseException as exc:
        stopped=p.read(path);stopped['halted']=True;accounting.durable_json(path,stopped)
        safe_write(out/'campaign_stop.json',{'kind':type(exc).__name__,'message':str(exc),'first_failure_stops':True,'dry_dispatch':mode,
            'scientific_denominator':4,'replay_denominator':8,'no_retry_or_fallback':True})
        raise
