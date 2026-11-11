"""Static, mocked and analytical tests. No MPC solve or integration."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest,tempfile,json,subprocess,sys
import numpy as np
import casadi as ca
from src import cstr_expert_pilot as p
from src import cstr_verification as v
from src.cstr_pilot_preflight import zero_operations
from src.quadrotor_kkt_integrity_v2 import canonical_kkt

ROOT=Path(__file__).resolve().parents[1]

def ledger(path):
    p.safe_write(Path(path)/'resource_ledger.json',{'charged_cumulative':{'cpu_seconds':0.,'wall_seconds':0.},'caps':{'cpu_seconds':600.,'wall_seconds':1200.},
        'reserved':dict.fromkeys(['solves','intervals','monitors'],0),'actual':dict.fromkeys(['solves','intervals','monitors'],0),
        'limits':{'solves':816,'intervals':2400,'monitors':24000},'jobs':[],'halted':False})

class CSTRPilotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.c=p.read(ROOT/p.CONFIG)
    def native(self):
        m,mpc,_,_=p.native_templates(self.c);ctrl=mpc(m(),silence_solver=True);ctrl.x0=np.array(self.c['cases'][0]['values']).reshape(4,1);ctrl.u0=np.array([5.,0.]).reshape(2,1);ctrl.set_initial_guess();return ctrl
    def test_original_binary_initial_states(self):
        expected=[[.8,.5,134.14,130],[.6,.4,130,125],[1,.7,136,135],[.8,.5,138,134]]
        for row,x in zip(self.c['cases'],expected):self.assertEqual(np.array(x,dtype='<f8').tobytes().hex(),row['little_endian_bytes_hex'])
    def test_exact_job_budget_and_diagnostics(self):
        jobs=p.jobs(self.c);self.assertEqual(len(jobs),40);self.assertEqual(tuple(sum(x[k] for x in jobs) for k in ['solves','intervals','monitors']),(816,2400,24000));self.assertEqual(sum(x['mode']=='context' for x in jobs),16)
    def test_prespecified_scientific_replay_order(self):
        jobs=p.jobs(self.c);self.assertEqual([j['mode'] for j in jobs[:6]],['scientific','native_replay','independent_replay']*2);self.assertEqual(jobs[6]['case'],'C1')
    def test_raw_negative_input_is_not_repaired(self):
        self.assertEqual(v.require_input([5.,0.]).min(),0.)
        with self.assertRaisesRegex(AssertionError,'Negative raw'):v.require_input([np.nextafter(5.,-np.inf),0.])
    def test_raw_state_and_positive_reserve(self):
        with self.assertRaisesRegex(AssertionError,'Negative raw'):v.require_state([.8,.5,np.nextafter(140.,np.inf),130.],self.c)
        with self.assertRaisesRegex(AssertionError,'reserve'):v.require_state([.1+1e-8,.5,134.,130.],self.c)
    def test_state_nonfinite_rejected(self):
        with self.assertRaises(ValueError):v.state_margins([.8,np.nan,134.,130.])
    def test_correct_bound_dual_signs(self):
        good=canonical_kkt([1.],[1.],[np.inf],[-1.],[0.]);bad=canonical_kkt([1.],[1.],[np.inf],[1.],[0.]);self.assertEqual(good['metrics']['dual_sign_inf'],0.);self.assertEqual(bad['metrics']['dual_sign_inf'],1.)
    def test_genuine_complementarity_and_free_equality(self):
        bad=canonical_kkt([.5],[0.],[1.],[1.],[0.]);eq=canonical_kkt([.5],[.5],[.5],[-99.],[0.]);self.assertEqual(bad['metrics']['complementarity_inf'],.5);self.assertEqual(eq['metrics']['dual_sign_inf'],0.)
    def test_nonfinite_kkt_rejected(self):self.assertFalse(canonical_kkt([0.],[-1.],[1.],[np.nan],[0.])['valid_arithmetic'])
    def test_independent_bound_locations(self):
        lo,hi,gl,gu=v.graph_bounds();self.assertFalse(np.isfinite(lo[:16]).any());self.assertFalse(np.isfinite(lo[320:336]).any());self.assertTrue(np.isfinite(lo[16:320]).all());self.assertTrue(np.array_equal(gl,gu))
    def test_independent_rhs_matches_archived_native_readout(self):
        f=ca.Function.load(str(ROOT/'results/cstr_source_audit/zero_solve_v1/checks_v2/native_rhs.casadi'))
        for row in self.c['cases']:np.testing.assert_allclose(v.rhs(0,np.array(row['values']),[5.,0.]),np.array(f(row['values'],[5.,0.])).reshape(-1),rtol=1e-13,atol=1e-12)
    def test_tracking_and_scaled_movement_cost(self):
        x=np.tile([.8,.5,134.,130.],(3,1));u=np.array([[15.,-2000.],[25.,-4000.]]);cost=v.costs(x,u,[5.,0.]);self.assertAlmostEqual(cost['tracking_sum'],.04);self.assertAlmostEqual(cost['movement_sum'],.004);self.assertAlmostEqual(cost['total_cost'],.064)
    def test_fresh_native_object_independence(self):
        with zero_operations() as counts:
            a,sa,_=p.initialize_objects(self.c,self.c['cases'][0]['values']);b,sb,_=p.initialize_objects(self.c,self.c['cases'][0]['values']);a.flags['initial_run']=True
            self.assertIsNot(a,b);self.assertIsNot(sa,sb);self.assertFalse(b.flags['initial_run']);self.assertEqual(p.memory(a)['primal'].tobytes(),p.memory(b)['primal'].tobytes());self.assertEqual(counts['real_operations'],0)
    def test_candidate_failure_preserves_vectors_before_rejection(self):
        with tempfile.TemporaryDirectory() as t,zero_operations():
            ctrl=self.native();z=p.array(ctrl.opt_x_num.cat).reshape(-1);params=np.r_[self.c['cases'][0]['values'],[5.,0.]];program=v.independent_program();f,g,*_=program(z,params,np.zeros(324))
            class Backend:
                def __call__(self,**kwargs):return {'x':z,'lam_x':np.zeros(376),'lam_g':np.zeros(324),'f':f,'g':g}
                def stats(self):return {'success':True,'return_status':'Solve_Succeeded','iter_count':1}
            charged=[];observer=p.SolverObserver(Backend(),ctrl,t,self.c,SimpleNamespace(charge=lambda name:charged.append(name)),lambda:None);lo,hi,gl,gu=v.graph_bounds()
            with self.assertRaises(AssertionError):observer(x0=z,p=params,lbx=lo,ubx=hi,lbg=gl,ubg=gu)
            self.assertEqual(charged,['solves']);self.assertTrue((Path(t)/'attempts/0000/candidate.npz').exists());self.assertTrue((Path(t)/'attempts/0000/native_stats.json').exists());self.assertTrue((Path(t)/'attempts/0000/independent_metrics.json').exists())
    def test_backend_exception_does_not_invent_candidate(self):
        with tempfile.TemporaryDirectory() as t,zero_operations():
            ctrl=self.native()
            def fail(**kwargs):raise RuntimeError('injected backend failure')
            observer=p.SolverObserver(fail,ctrl,t,self.c,SimpleNamespace(charge=lambda n:None),lambda:None)
            with self.assertRaises(RuntimeError):observer(x0=np.zeros(376),p=np.zeros(6))
            self.assertTrue((Path(t)/'attempts/0000/submitted.npz').exists());self.assertTrue((Path(t)/'attempts/0000/exception.json').exists());self.assertFalse((Path(t)/'attempts/0000/candidate.npz').exists())
    def test_current_command_telemetry_not_stale_cache(self):
        with tempfile.TemporaryDirectory() as t:
            submitted=np.array([7.,-100.]);sim=SimpleNamespace(sim_p_num={'_u':submitted},u0=np.array([0.,0.]))
            observer=p.IntegratorObserver(lambda **kw:{'xf':np.array([.8,.5,134.,130.])},sim,t,SimpleNamespace(charge=lambda n:None),lambda:None)
            observer(x0=np.array([.8,.5,134.,130.]),p=submitted);self.assertTrue((Path(t)/'integrations/0000_submitted.npz').exists())
    def test_reservation_cannot_dispatch_twice(self):
        with tempfile.TemporaryDirectory() as t:
            ledger(t);a=p.Accounting(t);job=p.jobs(self.c)[0];a.reserve(job)
            with self.assertRaisesRegex(RuntimeError,'restart'):a.reserve(job)
    def test_first_failure_stops_replays_and_next_case(self):
        with tempfile.TemporaryDirectory() as t:
            ledger(t)
            def fail(root,out,c,job,*args):
                (Path(out)/'runs'/job['id']).mkdir(parents=True);raise AssertionError('injected mandatory failure')
            with patch.object(p,'check_seal',return_value=self.c),patch.object(p,'verify_identities',return_value=True),patch.object(p,'scientific',side_effect=fail) as science,patch.object(p,'replay') as replay,patch.object(p,'diagnostic') as diag:
                result=p.run(ROOT,t);self.assertFalse(result['complete']);self.assertEqual(science.call_count,1);self.assertFalse(replay.called);self.assertFalse(diag.called);self.assertTrue(p.read(Path(t)/'resource_ledger.json')['halted'])
    def test_run_requires_preflight_before_native_factory(self):
        with tempfile.TemporaryDirectory() as t,patch.object(p,'initialize_objects') as factory:
            with self.assertRaises(FileNotFoundError):p.run(ROOT,t)
            self.assertFalse(factory.called)
    def test_exact_runner_plan_entrypoint_zero_solve(self):
        result=subprocess.run([sys.executable,str(ROOT/'experiments/cstr_expert_pilot.py'),'plan','--output','/private/tmp/cstr_plan_only'],text=True,capture_output=True,check=True);plan=json.loads(result.stdout);self.assertEqual(len(plan),40)
