"""E12 step-zero counterfactual preflight ONLY; deliberately no --run mode."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import argparse,ast,hashlib,json,math,os,resource,signal,sys,time,unittest
START=time.perf_counter()
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1';os.environ['MPLCONFIGDIR']='/private/tmp/e12_matplotlib'
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'tests')]


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--upstream',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'results/benchmark_selection/counterfactual_v1_preflight')
    a=parser.parse_args();out=a.output.resolve();up=a.upstream.resolve()
    if out.exists():raise FileExistsError('Existing preflight evidence retained; use a fresh path')
    import numpy as np
    import casadi as c
    from src import quadrotor_pilot as p
    from src import quadrotor_counterfactual_v1 as cf
    from src import quadrotor_mpc80_execution_v2 as accepted
    from src.quadrotor_mpc80_feasibility_v2 import zero_real_operations
    from src.quadrotor_mpc80_finalization import verify_snapshot
    from src.quadrotor_mmaq_v1 import safe_write,tagged
    from src.quadrotor_counterfactual_mock_v1 import controller
    from src.quadrotor_action_mock_v2 import native_harness
    from src.quadrotor_applied_action_v2 import AppliedActionObserver
    from src.quadrotor_kkt_integrity_v2 import nlp_readout,canonical_kkt,require_kkt,matching_program_values
    import xml.etree.ElementTree as ET
    base=ROOT/'results/benchmark_selection';study=base/'mpc80_feas_v2_study';pref=base/'mpc80_feas_v2_preflight'
    carry=p.read(base/'mpc80_feas_finalization/finalization_resource_accounting.json')
    cpu_before=carry['conservative_combined_CPU_seconds'];wall_before=carry['conservative_combined_active_wall_seconds']
    # Five CPU/twenty wall seconds conservatively cover shell/source preparation.
    resource.setrlimit(resource.RLIMIT_CPU,(int(1800-cpu_before-5),int(1800-cpu_before-5)+1))
    def expired(*args):raise TimeoutError('Old cumulative active-wall ceiling reached')
    signal.signal(signal.SIGALRM,expired);signal.setitimer(signal.ITIMER_REAL,3600-wall_before-20)
    tracked={n:p.sha(ROOT/n) for n in __import__('subprocess').check_output(['git','ls-files'],cwd=ROOT,text=True).splitlines()}
    frozen=p.read(study/'frozen.json');reference=np.load(study/'reference.npy')
    config=p.read(ROOT/'configs/experiment_12_counterfactual_v1.json')
    sources=['configs/experiment_12_counterfactual_v1.json','src/quadrotor_counterfactual_v1.py',
        'src/quadrotor_counterfactual_mock_v1.py','tests/test_quadrotor_counterfactual_v1.py',
        'experiments/12_counterfactual_preflight.py','docs/experiment_12_counterfactual_v1_protocol.md']
    out.mkdir(parents=True);safe_write(out/'tracked_before.json',tracked)
    original_ledger_hash=p.sha(study/'resource_ledger.json')
    try:
        accepted.checkpoint(ROOT,up,pref,study)
        assert config['original_gates']=={'overall':'FAILED','model_agreement_0_001':'FAILED'}
        assert not config['execution_authorized'] and config['probe_ids']==[0,4,8,12]
        source_hashes={n:p.sha(ROOT/n) for n in sources}
        safe_write(out/'protocol.json',config)
        # Freeze all mappings and both proposed job plans BEFORE any tests; no
        # simulation/optimizer is available in this entire preflight.
        km=float(ET.parse(up/'safe_control_gym/envs/gym_pybullet_drones/assets/cf2x.urdf').getroot().find('properties').attrib['km'])
        archives={i:dict(np.load(study/'runs'/f'{i}_nonlinear80_1000_original/trajectory.npz')) for i in cf.PROBES}
        rows=cf.labels(frozen,{i:v['actions'][0] for i,v in archives.items()},km);chosen=cf.select_alternative(rows)
        assert (chosen['state_id'],chosen['menu_index'])==(0,3)
        safe_write(out/'all_36_labels.json',rows);safe_write(out/'frozen_alternative.json',chosen)
        safe_write(out/'pilot_A_jobs.json',cf.job_plan('A'));safe_write(out/'pilot_B_jobs.json',cf.job_plan('B'))
        safe_write(out/'source_seal.json',{'source_sha256':source_hashes,'accepted_protocol_seal_sha256':p.sha(pref/'protocol_seal.json'),
            'screening_manifest_sha256':p.sha(base/'mmaq_v1/screening_manifest.json'),
            'accepted_v2_integrity_sha256':p.sha(study/'read_only_audit/final_integrity.json'),
            'accepted_ledger_sha256':original_ledger_hash,'execution_approval_present':False,
            'source_predicted_command_bytes':{'RPM':chosen['commands']['rpm'].astype('<f8').tobytes().hex(),
                'forces':chosen['commands']['forces'].astype('<f8').tobytes().hex(),'torque':chosen['commands']['torque'].astype('<f8').tobytes().hex()}})
        p.activate(up,frozen['config'])
        with zero_real_operations(up) as (forbidden,engine):
            def forbidden_solver(*args,**kwargs):raise RuntimeError('Optimization constructor forbidden in zero-solve preflight')
            with patch.object(c,'nlpsol',forbidden_solver),patch.object(c,'qpsol',forbidden_solver),patch.object(c,'conic',forbidden_solver):
                suite=unittest.TestSuite()
                for file in sorted((ROOT/'tests').glob('test_quadrotor*.py')):suite.addTests(unittest.defaultTestLoader.loadTestsFromName(file.stem))
                with (out/'tests.log').open('x') as stream:result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
                safe_write(out/'tests.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,'failures':len(result.failures),
                    'errors':len(result.errors),'skipped':len(result.skipped),'forbidden_real_calls':forbidden,'forbidden_engine_calls':engine,
                    'new_solves':0,'new_physics_steps':0})
                if not result.wasSuccessful():raise RuntimeError('Static regression failed')
                # Independent native source mapping and directly observed MOCK
                # motor submissions; scripted states are NOT counterfactual states.
                utility=up/'safe_control_gym/envs/gym_pybullet_drones/quadrotor_utils.py'
                nodes=[n for n in ast.parse(utility.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['cmd2pwm','pwm2rpm']]
                namespace={'np':np};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(utility),'exec'),namespace)
                native_matches=0
                for row in rows:
                    cmds=row['commands'];request=cmds['requested'];thrust=np.clip(request,frozen['action_low'],frozen['action_high'])
                    pwm=namespace['cmd2pwm'](thrust,frozen['PWM2RPM_SCALE'],frozen['PWM2RPM_CONST'],frozen['KF'],frozen['MIN_PWM'],frozen['MAX_PWM'])
                    rpm=namespace['pwm2rpm'](pwm,frozen['PWM2RPM_SCALE'],frozen['PWM2RPM_CONST'])
                    assert rpm.tobytes()==cmds['rpm'].tobytes()
                    with native_harness(up,frozen,archives[row['state_id']]['physics_states'][:21]) as (env,backend):
                        with AppliedActionObserver(env,backend) as observer:
                            env._advance_simulation(env._preprocess_control(request))
                            for packet in observer.steps:
                                assert packet['direct_rpm_argument'].tobytes()==rpm.tobytes()
                                assert packet['grouped_submitted_force_argument_sums'].tobytes()==cmds['submitted_grouped_thrust'].tobytes()
                                forces=np.array([e['kwargs']['forceObj'] for e in packet['submissions'] if e['kind']=='force'])
                                torque=next(e['kwargs']['torqueObj'] for e in packet['submissions'] if e['kind']=='torque')
                                assert np.asarray(forces,dtype='<f8').tobytes()==cmds['forces'].tobytes()
                                assert np.asarray(torque,dtype='<f8').tobytes()==cmds['torque'].tobytes()
                                native_matches+=1
                readout=nlp_readout(frozen,80);kkt=[];guess_checks=0
                for state,archive in archives.items():
                    ctrl,opt=controller(up,archive,reference)
                    for k in range(300):
                        ctrl.select_action(archive['states'][k],{'current_step':k})  # SCRIPTED lookup, never MPC solve
                        warm=np.load(study/'runs'/f'{state}_nonlinear80_1000_original/attempts'/f'{k:04d}'/'warmstart.npz')
                        guess=np.r_[opt.guesses['x'].reshape(-1,order='F'),opt.guesses['u'].reshape(-1,order='F')]
                        assert guess.tobytes()==warm['initial_guess_not_solver_candidate'].reshape(-1).tobytes();guess_checks+=1
                        if k in [0,1,220,221,299]:
                            candidate=np.load(study/'runs'/f'{state}_nonlinear80_1000_original/attempts'/f'{k:04d}'/'candidate.npz')
                            own=readout(candidate['primal'],candidate['parameters'],candidate['dual'])
                            error=matching_program_values(own,[candidate[n] for n in ['objective','constraint_values','lower','upper','lagrangian_gradient']])
                            d=canonical_kkt(own[1],own[2],own[3],candidate['dual'],own[4]);require_kkt(d,'Solve_Succeeded')
                            pars=candidate['parameters'].reshape(-1)
                            assert pars[:6].tobytes()==archive['states'][k].tobytes()
                            assert pars[6:].reshape((6,81),order='F').tobytes()==ctrl.get_references(k).tobytes()
                            kkt.append({'state_id':state,'step':k,'graph_error':error,'KKT':d['metrics']})
                safe_write(out/'archived_checks.json',{'passed':True,'archived_initial_guess_checks':guess_checks,
                    'candidate_and_reference_checks':kkt,'native_mock_application_boundaries':native_matches,
                    'live_context_equivalence_tested':False})
            assert not forbidden and not engine
        # Immutable accepted evidence and historical data; no repeated full
        # numerical training/trajectory audit is necessary for this static task.
        integrity=p.read(study/'read_only_audit/final_integrity.json')
        for n,h in integrity['artifact_sha256'].items():assert p.sha(study/n)==h,n
        preserved,context=verify_snapshot(ROOT,ROOT,up,p.read(study/'preservation_start.json'))
        for n,h in tracked.items():assert p.sha(ROOT/n)==h,n
        for folder in ['mpc80_feas_finalization','safe_action_diversity_design']:
            if folder=='safe_action_diversity_design':
                manifest=p.read(base/folder/'integrity_v2.json')
                for n,h in manifest['files_sha256'].items():assert p.sha(base/folder/n)==h,n
        assert p.sha(study/'resource_ledger.json')==original_ledger_hash
        for n,h in source_hashes.items():assert p.sha(ROOT/n)==h,n
        safe_write(out/'preservation.json',{'passed':True,'prior_milestone':preserved,'v2_artifacts':len(integrity['artifact_sha256']),
            'current_tracked_files':len(tracked),'authorized_prior_context_updates':context,'accepted_ledger_unchanged':True})
        safe_write(out/'preflight.json',{'passed':True,'new_solves':0,'new_physics_steps':0,'denominator':36,
            'input_admissible':sum(r['commands']['admissible'] for r in rows),'input_inadmissible':sum(not r['commands']['admissible'] for r in rows),
            'frozen_selection':[0,3],'tests':result.testsRun,'execution_approved':False,
            'live_context_equivalence_established':False,'recommended_pilot':'B',
            'conditions_before_execution':['separate explicit approval','incremental CPU envelope and carried ledger','fresh nominal live equivalence and direct telemetry checks'],
            'original_gates':config['original_gates']})
    except BaseException as exc:
        safe_write(out/'preflight_failure.json',{'kind':type(exc).__name__,'message':str(exc),'new_solves':0,'new_physics_steps':0});raise
    finally:
        cpu=math.ceil((time.process_time()+.1)*10)/10+5;wall=math.ceil((time.perf_counter()-START+.1)*10)/10+20
        signal.setitimer(signal.ITIMER_REAL,0)
        safe_write(out/'resource_accounting.json',{'carried_CPU_seconds':cpu_before,'carried_wall_seconds':wall_before,
            'preflight_CPU_charge':cpu,'preflight_wall_charge':wall,'combined_CPU_seconds':cpu_before+cpu,'combined_wall_seconds':wall_before+wall,
            'new_solves':0,'new_physics_steps':0,'original_remaining_solves':3070,'original_remaining_physics_steps':366000,
            'within_old_1800_3600_ceilings':cpu_before+cpu<=1800 and wall_before+wall<=3600,
            'peak_RSS_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'memory_cap_bytes':4294967296,'workers':1})
    if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>4294967296:raise MemoryError('Preflight RAM cap')
    print(p.read(out/'preflight.json'));print(p.read(out/'resource_accounting.json'))


if __name__=='__main__':main()
