"""Read-only independent reconstruction of the separately sealed v2 pilot."""
from pathlib import Path
from contextlib import ExitStack
from unittest.mock import patch
import json
import numpy as np
import casadi as ca
import casadi.tools as ct
from src import cstr_verification as v
from src import cstr_expert_pilot_v2 as p
from src.cstr_pilot_review import blocked,distribution


def verify(root,*,full_preservation=False):
    root=Path(root).resolve();out=root/p.OUT;c=p.check_seal(root,out);p.verify_identities(root,c)
    ledger=p.read(out/'resource_ledger.json');plan=p.base.jobs(c)
    completed=p.read(out/'study_completion.json')
    assert completed['complete'] and completed['jobs']==40
    assert len(ledger['jobs'])==40 and all(j['status']=='complete' for j in ledger['jobs'])
    assert ledger['actual']==ledger['reserved']=={'solves':816,'intervals':2400,'monitors':24000}
    assert not ledger['halted'] and not (out/'first_failure.json').exists()
    load=lambda path:dict(np.load(path,allow_pickle=False))
    program=v.independent_program();metrics_max={key:0. for key in ['primal_scaled_inf','scaled_stationarity_inf',
        'scaled_complementarity_inf','dual_sign_inf','objective_reconstruction_error','constraint_reconstruction_error']}
    cases={};contexts=[];backend_times=[];native_times=[];iterations=[];count=0;pure_solver_times=[]
    for job in plan:
        f=out/'runs'/job['id'];result=p.read(f/'completion.json');assert result['complete']
        if job['mode'] in ['scientific','context']:
            attempts=sorted((f/'attempts').glob('[0-9][0-9][0-9][0-9]'))
            assert len(attempts)==(100 if job['mode']=='scientific' else 1)
            trajectory=load(f/'trajectory.npz') if job['mode']=='scientific' else None
            for k,attempt in enumerate(attempts):
                a=load(attempt/'candidate.npz');submitted=load(attempt/'submitted.npz');stats=p.read(attempt/'native_stats.json')
                for key in submitted:assert a[key].tobytes()==submitted[key].tobytes()
                m=v.candidate_metrics(a,program);v.require_candidate(m,stats,c);count+=1
                for key in metrics_max:metrics_max[key]=max(metrics_max[key],m[key])
                options=p.read(attempt/'effective_ipopt_options.json');assert options['passed'] and options['observed']==p.expected_options()
                command=load(attempt/'returned_command.npz');u=command['command']
                assert u.tobytes()==np.asarray(m['command']).tobytes();v.require_input(u)
                assert command['lower_margin'].tobytes()==(u-v.ULO).tobytes()
                assert command['upper_margin'].tobytes()==(v.UHI-u).tobytes()
                before=load(attempt/'prequery_memory.npz');after=load(attempt/'postquery_memory.npz')
                assert before['previous_u'].tobytes()==a['p'][4:].tobytes()
                assert after['previous_u'].tobytes()==u.tobytes();assert after['primal'].tobytes()==a['z'].tobytes()
                assert after['lam_x_num'].tobytes()==a['lam_x'].tobytes();assert after['lam_g_num'].tobytes()==a['lam_g'].tobytes()
                bound=load(attempt/'bound_residuals.npz');physical=a['z'][336:].reshape(20,2)*v.SU
                for key,expected in [('scaled_lower',a['z']-a['lbx']),('scaled_upper',a['ubx']-a['z']),
                    ('physical_predicted_commands',physical),('input_lower',physical-v.ULO),('input_upper',v.UHI-physical)]:
                    assert bound[key].tobytes()==expected.tobytes()
                backend_times.append(p.read(attempt/'solver_timing.json')['solver_wall'])
                timing=p.read(attempt/'native_timing.json');assert timing['native_query_returned'] and timing['command_available']
                native_times.append(timing['native_action_selection_including_verification_wall']);iterations.append(stats['iter_count'])
                if 't_wall_total' in stats:pure_solver_times.append(stats['t_wall_total'])
                if trajectory is not None:
                    assert a['p'][:4].tobytes()==trajectory['states'][k].tobytes()
                    assert u.tobytes()==trajectory['commands'][k].tobytes()
                    previous=np.asarray(c['previous_input']) if k==0 else trajectory['commands'][k-1]
                    assert a['p'][4:].tobytes()==previous.tobytes()
                    assert abs(float(before['native_time'][0])-k*c['dt'])<1e-14
            if job['mode']=='context':
                original=out/'runs'/f"{job['case']}_original_scientific"/'attempts'/f"{job['step']:04d}"
                old=load(original/'candidate.npz');assert a['p'].tobytes()==old['p'].tobytes()
                if job['initialization']=='native_prequery_primal_duals':
                    for key in ['z','p','lam_x','lam_g','f','g']:assert a[key].tobytes()==old[key].tobytes()
                diff=u-old['z'][336:338]*v.SU
                assert np.array_equal(diff,np.asarray(result['difference_from_original_command']))
                contexts.append({'case':job['case'],'step':job['step'],'initialization':job['initialization'],
                    'difference':diff.tolist(),'normalized_difference_inf':float(np.max(abs(diff)/v.UWIDTH)),
                    'objective_difference':float(a['f'][0]-old['f'][0]),'iterations':stats['iter_count']})
                continue
        if job['mode']!='scientific':continue
        x,u=trajectory['states'],trajectory['commands'];assert x.shape==(101,4) and u.shape==(100,2)
        expected=next(row['values'] for row in c['cases'] if row['id']==job['case'])
        assert x[0].tobytes()==np.asarray(expected).tobytes();v.require_state(x,c)
        du=u-np.vstack([np.asarray(c['previous_input']),u[:-1]])
        tracking=np.square(x[:-1,:2]-[.7,.6]).sum(axis=1)
        movement=.1*np.square(du[:,0]/100)+.001*np.square(du[:,1]/2000)
        terminal=float(np.square(x[-1,:2]-[.7,.6]).sum());cost=float(tracking.sum()+movement.sum()+terminal)
        assert abs(result['costs']['total_cost']-cost)<1e-14
        assert np.allclose(result['costs']['concentration_RMSE'],np.sqrt(np.square(x[:-1,:2]-[.7,.6]).mean(0)),rtol=0,atol=1e-15)
        assert abs(result['costs']['feed_volume_L']-10.01*.005*u[:,0].sum())<1e-12
        assert abs(result['costs']['cooling_energy_kJ']+.005*u[:,1].sum())<1e-12
        raw=load(f/'raw_margins.npz')
        for key,want in [('state_lower',x-v.LOW),('state_upper',v.HIGH-x),('input_lower',u-v.ULO),('input_upper',v.UHI-u)]:assert np.array_equal(raw[key],want)
        native=out/'runs'/f"{job['case']}_{job['phase']}_native_replay"
        assert p.read(native/'completion.json')['complete'];p.base.compare_npz(f/'trajectory.npz',native/'trajectory.npz')
        for folder in [f,native]:
            for k in range(100):
                packet=load(folder/'integrations'/f'{k:04d}_submitted.npz');end=load(folder/'integrations'/f'{k:04d}_result.npz')
                assert packet['p'][:2].tobytes()==u[k].tobytes();assert packet['x0'].reshape(-1).tobytes()==x[k].tobytes()
                assert end['xf'].reshape(-1).tobytes()==x[k+1].tobytes()
        independent=out/'runs'/f"{job['case']}_{job['phase']}_independent_replay"
        assert p.read(independent/'completion.json')['complete']
        a=load(independent/'monitoring_partial.npz');assert a['refinement'].shape==(100,21,4)
        assert np.array_equal(a['primary'],a['refinement'][:,::2]);v.require_state(a['refinement'],c)
        for k in range(100):assert p.read(independent/f'integration_{k:04d}.json')['success']
        dense=load(independent/'dense_raw_margins.npz')
        assert np.array_equal(dense['lower'],a['refinement']-v.LOW);assert np.array_equal(dense['upper'],v.HIGH-a['refinement'])
        err=float(np.max(abs(a['states']-x)/v.WIDTH));assert err<c['normalized_replay_tolerance']
        margins=v.state_margins(a['refinement']).min(axis=(0,1));reserve=float(np.min(margins/v.WIDTH)-c['normalized_reserve'])
        if job['phase']=='original':
            cases[job['case']]={'costs':result['costs'],'raw_boundary_lower_margins':(x-v.LOW).min(0).tolist(),
                'raw_boundary_upper_margins':(v.HIGH-x).min(0).tolist(),'dense_lower_margins':dense['lower'].min(axis=(0,1)).tolist(),
                'dense_upper_margins':dense['upper'].min(axis=(0,1)).tolist(),'dense_min_margins':margins.tolist(),
                'input_lower_margins':(u-v.ULO).min(0).tolist(),'input_upper_margins':(v.UHI-u).min(0).tolist(),
                'minimum_normalized_reserve':reserve,'maximum_normalized_native_independent_difference':err,
                'state_violations':0,'applied_input_violations':0}
        else:
            original=out/'runs'/f"{job['case']}_original_scientific"
            p.base.compare_npz(original/'trajectory.npz',f/'trajectory.npz')
            for k in range(100):
                for name in ['submitted.npz','candidate.npz','prequery_memory.npz','postquery_memory.npz','returned_command.npz','bound_residuals.npz']:
                    p.base.compare_npz(original/'attempts'/f'{k:04d}'/name,f/'attempts'/f'{k:04d}'/name)
                s,t=[p.read(g/'attempts'/f'{k:04d}'/'native_stats.json') for g in [original,f]]
                assert all(s[key]==t[key] for key in ['success','return_status','iter_count'])
    assert count==816 and len(cases)==4 and len(contexts)==16
    returns=[x['costs']['return'] for x in cases.values()];cold=[x for x in contexts if x['initialization']=='constant_x_previous_u_zero_duals']
    result={'independent_verification_passed':True,'pilot_qualification_passed':True,'new_solves':0,'new_integrations':0,
        'scientific_executions':8,'native_replays':8,'independent_replays':8,'candidates':count,'case_count':4,
        'cases':cases,'contexts':contexts,'complete_cohort_mean_return':float(np.mean(returns)),'complete_cohort_worst_return':float(min(returns)),
        'complete_cohort_best_return':float(max(returns)),'maximum_cold_initialization_normalized_action_difference':max(x['normalized_difference_inf'] for x in cold),
        'maximum_cold_initialization_absolute_action_difference':np.max(np.abs([x['difference'] for x in cold]),axis=0).tolist(),
        'candidate_metric_maxima':metrics_max,'backend_envelope_including_option_getter_seconds':distribution(backend_times),
        'native_action_selection_including_verification_seconds':distribution(native_times),'solver_iterations':distribution(iterations),
        'native_18_second_deadline_overruns':int(np.sum(np.asarray(native_times)>18.)),
        'native_t_wall_total_observations':len(pure_solver_times),'missing_returned_command_or_postquery_memory':0,
        'peak_RSS_recorded':False,'memory_limit_watchdog_GiB':4,'source_seal_sha256':p.sha(out/'protocol_seal.json')}
    if pure_solver_times:result['native_t_wall_total_seconds']=distribution(pure_solver_times)
    if full_preservation:
        from src.quadrotor_archive_finalization import historical_preservation
        result['historical_preservation']=historical_preservation(root,root,p.read(root/'results/benchmark_selection/b0_archive_v1/preservation_start.json'))
        extra={}
        for name,filename in [('b0_archive_v1','completion_integrity.json'),('b0_finalization','finalization_integrity.json'),('b1_archive_v1','completion_integrity.json'),('b1_finalization','completion_integrity.json')]:
            b=root/'results/benchmark_selection'/name;s=p.read(b/filename)
            for n,h in s['artifact_sha256'].items():assert p.sha(b/n)==h,n
            extra[name]=len(s['artifact_sha256'])
        result['additional_preservation']=extra
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--full-preservation',action='store_true');args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    with ExitStack() as stack:
        for module in [ca,ct]:
            for name in ['nlpsol','integrator']:stack.enter_context(patch.object(module,name,blocked))
        for name in ['solve','solve_limited']:stack.enter_context(patch.object(ca.Opti,name,blocked))
        stack.enter_context(patch('scipy.integrate.solve_ivp',blocked))
        result=verify(Path.cwd(),full_preservation=args.full_preservation)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))

if __name__=='__main__':main()
