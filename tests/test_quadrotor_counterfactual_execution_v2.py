"""Inert exact-entrypoint/wiring regressions; no optimization or integration."""
from pathlib import Path
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch
import importlib.util,json,os,subprocess,sys,tempfile,unittest
import numpy as np
from src import quadrotor_counterfactual_dispatch_v2 as d
from src import quadrotor_counterfactual_execution_v2 as e
from src import quadrotor_counterfactual_v1 as cf
from src import quadrotor_pilot as p
from src import quadrotor_mpc80_feasibility_v1 as budget
from src.quadrotor_mmaq_v1 import safe_write
ROOT=Path(__file__).resolve().parents[1];UP=Path('/private/tmp/e12_safe_control_gym')
ARCH=ROOT/e.ACCEPTED/'runs/0_nonlinear80_1000_original'
ENTRY=ROOT/'experiments/12_counterfactual_pilot_b_v2.py'


def setup_ledger(out):
    budget.durable_json(out/'resource_ledger.json',d.ledger(ROOT,dry=True,carry={'cpu_seconds':1830.740281,'wall_seconds':2517.051233},
        seal_sha='mock',report_sha='mock',cpu_cap=2130.740281,wall_cap=3117.051233))
    return {'execution_approved':True,'dry_dispatch':True,'seal_sha256':'mock','preflight_report_sha256':'mock','review_reference':'INERT UNIT FIXTURE'}


class DispatchTests(unittest.TestCase):
    def test_exact_twelve_jobs_and_totals(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);approval=setup_ledger(out);seen=[]
            d.dispatch(ROOT,out,approval,lambda j:seen.append(j['id']) or {'passed':True,'mock':True},gate=lambda _:True)
            s=p.read(out/'resource_ledger.json')
            self.assertEqual(seen,[j['id'] for j in d.frozen_plan(ROOT)])
            self.assertEqual((s['charged']['solver_calls'],s['charged']['physics_steps']),(1200,72000))
            self.assertEqual(s['charged']['cpu_seconds'],1830.740281)
            self.assertTrue(all(j['status']=='completed' for j in s['reservations']))
    def test_every_worker_failure_stops_and_keeps_reservation(self):
        for fail in range(12):
            with self.subTest(index=fail),tempfile.TemporaryDirectory() as t:
                out=Path(t);approval=setup_ledger(out);seen=[]
                def launch(j):
                    seen.append(j['id'])
                    if len(seen)==fail+1:raise RuntimeError('mock failure')
                    return {'passed':True,'mock':True}
                with self.assertRaisesRegex(RuntimeError,'mock failure'):d.dispatch(ROOT,out,approval,launch,gate=lambda _:True)
                s=p.read(out/'resource_ledger.json');self.assertEqual(len(seen),fail+1);self.assertTrue(s['halted'])
                self.assertEqual(s['reservations'][-1]['status'],'failed')
                self.assertEqual(s['charged']['physics_steps'],6000*(fail+1))
                self.assertEqual(s['charged']['solver_calls'],sum(j['solver_calls'] for j in d.frozen_plan(ROOT)[:fail+1]))
                with self.assertRaises(RuntimeError):d.dispatch(ROOT,out,approval,launch,gate=lambda _:True)
    def test_nominal_gate_precedes_any_alternative_reservation(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);approval=setup_ledger(out);seen=[]
            with self.assertRaisesRegex(RuntimeError,'Nominal gate'):
                d.dispatch(ROOT,out,approval,lambda j:seen.append(j) or {'passed':True,'mock':True},gate=lambda _:False)
            self.assertEqual(len(seen),6);self.assertTrue(all(j['menu_index']==4 for j in seen))
            self.assertEqual(len(p.read(out/'resource_ledger.json')['reservations']),6)
    def test_actual_nominal_gate_on_accepted_archives(self):
        with tempfile.TemporaryDirectory() as t:
            out=Path(t)
            for phase in ['original','repeat']:
                a=ROOT/e.ACCEPTED/'runs'/f'0_nonlinear80_1000_{phase}';b=out/'branches/menu4/runs'/a.name;b.mkdir(parents=True)
                for n in ['completion.json','independent_episode_check.json','replay_native.json','replay_instrumented.json','submitted_motor_replay_counts.json']:(b/n).write_bytes((a/n).read_bytes())
                safe_write(b/'archived_nominal_fidelity.json',{'arrays_bitwise':{'inert_fixture':True},'all_300_candidate_vectors_bitwise':True})
            self.assertTrue(d.nominal_gate(out))
            safe_write(b/'replay_native.json',{'steps':299})
            with self.assertRaises((AssertionError,KeyError)):d.nominal_gate(out)
    def test_altered_plan_or_mock_live_mismatch_rejected(self):
        for mutation in ['order','id','repeat','cap','result']:
            with self.subTest(mutation=mutation),tempfile.TemporaryDirectory() as t:
                out=Path(t);approval=setup_ledger(out);s=p.read(out/'resource_ledger.json');seen=[]
                if mutation=='order':s['jobs'][0],s['jobs'][1]=s['jobs'][1],s['jobs'][0]
                if mutation=='id':s['jobs'][0]['state_id']=4
                if mutation=='repeat':s['jobs'].append(s['jobs'][0])
                if mutation=='cap':s['caps']['solver_calls']=1201
                budget.durable_json(out/'resource_ledger.json',s)
                with self.assertRaises((ValueError,RuntimeError)):
                    d.dispatch(ROOT,out,approval,lambda j:seen.append(j) or {'passed':True,'mock':False},gate=lambda _:True)
                self.assertEqual(len(seen),int(mutation=='result'))
    def test_budget_exhaustion_never_dispatches(self):
        for key in ['cpu_seconds','wall_seconds']:
            with tempfile.TemporaryDirectory() as t:
                out=Path(t);a=setup_ledger(out);s=p.read(out/'resource_ledger.json');s['charged'][key]=10000;budget.durable_json(out/'resource_ledger.json',s)
                seen=[]
                with self.assertRaises(RuntimeError):d.dispatch(ROOT,out,a,lambda j:seen.append(j),gate=lambda _:True)
                self.assertEqual(seen,[])
    def test_exact_cli_dry_success_failure_and_no_live_permission(self):
        for mode in ['success','failure','paused']:
            with self.subTest(mode=mode),tempfile.TemporaryDirectory() as t:
                out=Path(t)/'job';args=[sys.executable,str(ENTRY),'--output',str(out),'--mode','execute' if mode=='paused' else 'dry-dispatch']
                if mode=='failure':args+=['--fail-at',d.frozen_plan(ROOT)[3]['id']]
                r=subprocess.run(args,env={**os.environ,'E12_FORCE_ZERO_SOLVE':'1'},capture_output=True,text=True,timeout=35)
                self.assertEqual(r.returncode==0,mode=='success',r.stderr[-1500:])
                if mode=='paused':self.assertFalse(out.exists());continue
                z=p.read(out/'zero_operations.json');self.assertFalse(z['forbidden_calls']);self.assertFalse(z['engine_calls'])
                trace=p.read(out/'dry_trace.json');self.assertEqual(len(trace),12 if mode=='success' else 4)
    def test_worker_routing_and_reservation_rechecked(self):
        spec=importlib.util.spec_from_file_location('pilot_entry_test',ENTRY);entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        for j in d.frozen_plan(ROOT):
            with tempfile.TemporaryDirectory() as t:
                out=Path(t);a=setup_ledger(out);s=p.read(out/'resource_ledger.json');s['dry_dispatch']=False;s['reservations']=[{**j,'status':'reserved'}];budget.durable_json(out/'resource_ledger.json',s);safe_write(out/'approval.json',a)
                with patch.object(entry,'validate_approval'),patch.object(e,'checkpoint'),patch.object(entry.resource,'setrlimit'),patch.object(e,'episode',return_value={'passed':True,'mock':True}) as science,patch.object(e,'replay',return_value={'passed':True,'mock':True}) as replay:
                    entry.worker(UP,out,j['id'])
                    if j['kind']=='scientific':science.assert_called_once_with(ROOT,UP,out,j['menu_index'],j['phase']);replay.assert_not_called()
                    else:replay.assert_called_once_with(UP,out,j['menu_index'],j['phase'],j['kind']);science.assert_not_called()
                    with self.assertRaises(PermissionError):entry.worker(UP,out,'wrong-id')
    def test_authorized_preparation_and_exact_execute_subprocess_route_are_inert(self):
        spec=importlib.util.spec_from_file_location('pilot_preparation_test',ENTRY);entry=importlib.util.module_from_spec(spec);spec.loader.exec_module(entry)
        with tempfile.TemporaryDirectory() as t:
            parent=Path(t);review=parent/'review';review.mkdir();out=parent/'execution'
            safe_write(review/'review_seal.json',{'source_sha256':{},'artifact_sha256':{}})
            (review/'scientific_report.md').write_text('INERT review fixture; never scientific permission')
            safe_write(review/'resource_ledger.json',{'charged_cumulative':{'cpu_seconds':1830.740281,'wall_seconds':2517.051233},'proposed_execution_caps':{'cpu_seconds':2130.740281,'wall_seconds':3117.051233}})
            a={'execution_approved':True,'purpose':'FRESH_PILOT_B_EXECUTION','dry_dispatch':False,
               'seal_sha256':p.sha(review/'review_seal.json'),'preflight_report_sha256':p.sha(review/'scientific_report.md'),
               'review_reference':'INERT fixture under patched subprocess; NOT an authorization',
               'cumulative_CPU_ceiling':2130.740281,'cumulative_wall_ceiling':3117.051233}
            approval=parent/'inert_approval.json';safe_write(approval,a);calls=[]
            def failed_child(command,**kw):calls.append((command,kw));return SimpleNamespace(returncode=7)
            with patch.object(entry,'REVIEW',review),patch.object(e,'checkpoint'),patch.dict(os.environ,{'E12_FORCE_ZERO_SOLVE':'0'}),patch.object(entry.subprocess,'run',side_effect=failed_child):
                entry.prepare(UP,out,approval)
                self.assertEqual((out/'branches/menu4/frozen.json').read_bytes(),(ROOT/e.ACCEPTED/'frozen.json').read_bytes())
                with self.assertRaises(FileExistsError):entry.prepare(UP,out,approval)
                with self.assertRaisesRegex(RuntimeError,'worker failed'):entry.execute(UP,out,approval)
            self.assertEqual(len(calls),1);command,kw=calls[0]
            self.assertEqual(command[:4],[sys.executable,str(ENTRY),'--mode','worker'])
            self.assertEqual(command[-1],d.frozen_plan(ROOT)[0]['id']);self.assertGreater(kw['timeout'],0)
            s=p.read(out/'resource_ledger.json');self.assertTrue(s['halted']);self.assertEqual(s['charged']['solver_calls'],300)
            self.assertEqual(s['charged']['physics_steps'],6000);self.assertGreater(s['charged']['cpu_seconds'],1830.740281)
    def test_restored_runtime_identity_is_rechecked(self):
        self.assertTrue(e.runtime_identity(ROOT,UP)['passed'])
        with patch.object(e.sys,'version','different interpreter'):
            with self.assertRaisesRegex(RuntimeError,'interpreter'):e.runtime_identity(ROOT,UP)


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.model=p.read(ROOT/e.ACCEPTED/'frozen.json')
    def frozen_row(self):
        row=np.fromfile(ARCH/'submitted_scientific.f64',dtype='<f8',count=56).copy()
        c=p.read(ROOT/e.PREF/'frozen_alternative.json')['commands']
        for part,key in [(slice(3,5),'requested'),(slice(5,7),'requested'),(slice(7,9),'clipped'),(slice(9,13),'rpm'),(slice(13,17),'rpm'),(slice(17,29),'forces'),(slice(41,44),'torque')]:row[part]=np.asarray(c[key]).reshape(-1)
        return row
    def test_exact_frozen_binary64_and_separation(self):
        row=self.frozen_row();e.check_submission_row(row,self.model,3,ROOT)
        rows=np.repeat(row[None,:],20,axis=0);r=e.verify_first_submissions(rows,self.model,3,ROOT)
        self.assertTrue(r['normalized_separation']['at_least_five_percent'])
    def test_first_and_last_step_of_first_interval_are_checked(self):
        for index in [0,19]:
            row=self.frozen_row();row[0]=index;row[3]=np.nextafter(row[3],np.inf)
            with self.assertRaisesRegex(ValueError,'Pre-physics frozen'):e.check_submission_row(row,self.model,3,ROOT)
        row=self.frozen_row();row[0]=20;row[3]=.2;e.check_submission_row(row,self.model,3,ROOT)
    def test_signed_zero_changed_force_is_not_exact(self):
        row=self.frozen_row();row[17]=-0.
        with self.assertRaisesRegex(ValueError,'forces'):e.check_submission_row(row,self.model,3,ROOT)
        self.assertTrue(cf.same({'a':0.},{'a':-0.}))
        self.assertFalse(e.exact_context({'a':0.},{'a':-0.}))
    def test_nonfinite_and_out_of_bounds_fail(self):
        for change in ['rpm','force']:
            row=self.frozen_row()
            if change=='rpm':row[13]=np.nan
            else:row[19]=-1.
            with self.assertRaises(ValueError):e.check_submission_row(row,self.model,3,ROOT)
    def test_submission_journal_precedes_rejection_and_no_completion(self):
        row=self.frozen_row();row[3]+=1e-9
        packet={'application_index':0,'force_application_returned':True,'physics_returned':False}
        with tempfile.TemporaryDirectory() as t,patch.object(e.live,'flatten_packet',return_value=row):
            with e.frozen_stream_scope(self.model,3,ROOT):
                stream=e.live.SubmissionStream(Path(t),'inert')
                with self.assertRaises(ValueError):stream.emit(packet)
                # Observer may emit a failure packet again; preserve it without a recursive rejection.
                stream.emit({**packet,'step_or_callback_error':'fixture'});stream.close()
            self.assertEqual((Path(t)/'inert.f64').stat().st_size,56*8)
            self.assertEqual((Path(t)/'inert_completed.u1').stat().st_size,0)
            self.assertTrue((Path(t)/'inert_boundary_failure_0.json').exists())
    def test_replay_wrapper_exact_mode_and_boundary_observer(self):
        original=e.live.SubmissionStream
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);b=out/'branches/menu3';b.mkdir(parents=True);safe_write(b/'frozen.json',self.model)
            def backend(*args):self.assertIsNot(e.live.SubmissionStream,original);return {'inert':True}
            with patch.object(e.live,'replay',side_effect=backend) as call:
                for kind in ['native_replay','motor_replay']:
                    e.replay(UP,out,3,'repeat',kind);self.assertEqual(call.call_args.args,(UP,b,0,'repeat',kind))
            self.assertIs(e.live.SubmissionStream,original)


class EpisodeWiringTests(unittest.TestCase):
    def test_actual_episode_wiring_with_300_archived_queries_each_branch(self):
        """Drive the real wrapper with native Python MPC flow and scripted arrays.

        No counterfactual successor is predicted here. Both paths are deliberately
        fed nominal archived states; this is an ordering test, never a safety run.
        """
        from src.quadrotor_counterfactual_mock_v1 import controller
        import xml.etree.ElementTree as ET
        archive=dict(np.load(ARCH/'trajectory.npz'));reference=np.load(ROOT/e.ACCEPTED/'reference.npy');model=p.read(ROOT/e.ACCEPTED/'frozen.json')
        km=float(ET.parse(UP/'safe_control_gym/envs/gym_pybullet_drones/assets/cf2x.urdf').getroot().find('properties').attrib['km'])
        graph=p.read(ROOT/'results/benchmark_selection/mpc80_feas_v1_preflight/preflight.json')['graph']['native_program_sha256']
        for menu in [4,3]:
            with self.subTest(menu=menu),tempfile.TemporaryDirectory() as t:
                out=Path(t);b=out/'branches'/f'menu{menu}';b.mkdir(parents=True);safe_write(b/'frozen.json',model)
                ctrl,op=controller(UP,archive,reference);ctrl.soft_constraints=False;ctrl.constraint_tol=1e-6
                op.x='primal';op.p='parameters';op.initial=lambda:'initial fixture'
                def debug_value(key,initial=None):
                    if key=='primal':return np.r_[op.guesses['x'].reshape(-1,order='F'),op.guesses['u'].reshape(-1,order='F')]
                    if not op.values:raise RuntimeError('parameters not set')
                    return np.r_[op.values['initial'].reshape(-1),op.values['reference'].reshape(-1,order='F')]
                op.debug=SimpleNamespace(value=debug_value)
                log=[];current={'state':archive['states'][0].copy()}
                env=SimpleNamespace(KM=km,ctrl_step_counter=0,last_clipped_action=np.zeros(4))
                def reset():log.append('plant.reset');return archive['states'][0].copy(),{'current_step':0}
                env.reset=reset
                def env_factory(*args,**kwargs):log.append('plant.construct');return env
                def controller_factory(*args,**kwargs):log.append('controller.native.construct/reset');return ctrl
                @contextmanager
                def scripted_capture(env,folder,label,model,captured,predictions,*args):
                    data={'states':[archive['physics_states'][0].copy()],'contacts':[0]};captured['data']=data
                    yield data
                class Timing:
                    records=[]
                    @contextmanager
                    def span(self,*args):yield
                    def summary(self):return {'inert':True}
                class Guard:
                    def __init__(self,folder,metadata,readout=None):self.folder=folder;folder.mkdir(parents=True);self.state={};self.timing=Timing();self.step=metadata['control_step']
                    def select(self,query,*args):return query()
                    def native_solve_hook(self,opti,query,gradient):
                        result=query();self.current_candidate=dict(np.load(ARCH/'attempts'/f'{self.step:04d}'/'candidate.npz'));return result
                def driver(*args):
                    plant=p.make_environment();controller=p.make_controller();obs,info=plant.reset()
                    with p.capture_physics(plant) as data:
                        for k in range(300):
                            request=controller.select_action(archive['states'][k].copy(),{'current_step':k})
                            expected=archive['actions'][k].copy()
                            if k==0 and menu==3:expected+=cf.menu_offsets(model)[3]
                            self.assertEqual(request.tobytes(),expected.tobytes());self.assertFalse(np.shares_memory(request,ctrl.u_prev))
                            self.assertEqual(ctrl.u_prev.tobytes(),archive['predicted_inputs'][k].tobytes())
                            self.assertEqual(ctrl.x_prev.tobytes(),archive['predicted_states'][k].tobytes())
                            self.assertEqual(op.i,k+1) # exactly one native query, no replacement solve
                            guess=np.r_[op.calls[-1]['guesses']['x'].reshape(-1,order='F'),op.calls[-1]['guesses']['u'].reshape(-1,order='F')]
                            with np.load(ARCH/'attempts'/f'{k:04d}'/'warmstart.npz') as w:self.assertEqual(guess.tobytes(),w['initial_guess_not_solver_candidate'].tobytes())
                            data['states'].extend(archive['physics_states'][20*k+1:20*k+21]);data['contacts'].extend([0]*20)
                            current['state']=archive['states'][k+1].copy()
                    raise RuntimeError('END OF INERT DRIVER; not a scientific execution')
                def contract(x,request,cache,model):return {'executed_thrust':request.copy(),'reserve':np.ones(6),'provenance':'INERT contract spy; no prediction'}
                context={key:[] for key in e.CONTEXT_KEYS}
                with patch.object(p,'make_environment',side_effect=env_factory),patch.object(p,'make_controller',side_effect=controller_factory),patch.object(p,'run_episode',side_effect=driver),\
                     patch.object(p,'raw_state',side_effect=lambda _:current['state'].copy()),patch.object(cf,'passive_reset_state',return_value=context),\
                     patch.object(e,'programs',return_value=(None,graph,None)),patch.object(e,'IndependentCandidateGuard',Guard),\
                     patch.object(e.live,'checked_capture',side_effect=scripted_capture),patch.object(cf,'candidate_contract',side_effect=contract),patch.object(e,'source_prediction',side_effect=lambda x,u,c,m:x.copy()):
                    with self.assertRaisesRegex(RuntimeError,'END OF INERT DRIVER'):e.episode(ROOT,UP,out,menu,'original')
                self.assertEqual(log,['plant.construct','controller.native.construct/reset','plant.reset']);self.assertEqual(op.i,300)
                f=b/'runs/0_nonlinear80_1000_original';r=p.read(f/'startup_or_execution_failure.json')
                self.assertEqual((r['native_solve_attempts'],r['physics_steps']),(300,6000)) # mock counts, no real engine operations
                for k in [0,1,220,221,299]:
                    record=p.read(f/'intervention'/f'{k:04d}.json');self.assertTrue(record['post_intervention_history_unchanged'])
                    self.assertEqual(record['status'],'ready_for_native_step')
                self.assertTrue(np.load(f/'exception_partial_trajectory.npz')['states'].shape==(6001,6))

    def test_constructor_exception_is_retained_without_fallback(self):
        model=p.read(ROOT/e.ACCEPTED/'frozen.json')
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);b=out/'branches/menu4';b.mkdir(parents=True);safe_write(b/'frozen.json',model)
            with patch.object(p,'run_episode',side_effect=RuntimeError('fixture startup')):
                with self.assertRaisesRegex(RuntimeError,'fixture startup'):e.episode(ROOT,UP,out,4,'original')
            r=p.read(b/'runs/0_nonlinear80_1000_original/startup_or_execution_failure.json')
            self.assertFalse(r['complete']);self.assertFalse(r['fallback_action_applied']);self.assertEqual(r['native_solve_attempts'],0)
    def test_incomplete_run_never_reaches_independent_acceptance(self):
        model=p.read(ROOT/e.ACCEPTED/'frozen.json')
        with tempfile.TemporaryDirectory() as t:
            out=Path(t);b=out/'branches/menu3';b.mkdir(parents=True);safe_write(b/'frozen.json',model)
            folder=b/'runs/0_nonlinear80_1000_original'
            def failed(*args):folder.mkdir(parents=True);return {'complete':False,'transitions':0}
            with patch.object(p,'run_episode',side_effect=failed),patch.object(e.checks,'verify_episode') as verifier:
                with self.assertRaisesRegex(RuntimeError,'Incomplete scientific'):e.episode(ROOT,UP,out,3,'original')
                verifier.assert_not_called()
            self.assertFalse(p.read(folder/'completion.json')['complete'])


if __name__=='__main__':unittest.main()
