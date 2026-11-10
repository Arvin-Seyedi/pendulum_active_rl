"""Zero-solve mocked/archived counterfactual pathway tests; no new trajectories."""
from pathlib import Path
from copy import deepcopy
from types import SimpleNamespace
import json,tempfile,unittest
import numpy as np
from src.quadrotor_counterfactual_v1 import (commands,labels,select_alternative,separation,CounterfactualPath,
    same,job_plan,history,initialize_native_context,passive_reset_state,classify_branch,failure_sink,candidate_contract)
from src.quadrotor_counterfactual_mock_v1 import controller
from src.quadrotor_action_mock_v2 import native_harness,trace_digest
from src.quadrotor_applied_action_v2 import AppliedActionObserver
ROOT=Path(__file__).resolve().parents[1]
UP=Path('/private/tmp/e12_safe_control_gym')
BASE=ROOT/'results/benchmark_selection/mpc80_feas_v2_study'


class CounterfactualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model=json.loads((BASE/'frozen.json').read_text());cls.ref=np.load(BASE/'reference.npy')
        import xml.etree.ElementTree as E
        cls.km=float(E.parse(UP/'safe_control_gym/envs/gym_pybullet_drones/assets/cf2x.urdf').getroot().find('properties').attrib['km'])
        cls.archives={i:dict(np.load(BASE/'runs'/f'{i}_nonlinear80_1000_original/trajectory.npz')) for i in [0,4,8,12]}
    def fixture(self,index=4,state=0,contract=None):
        a=self.archives[state];ctrl,op=controller(UP,a,self.ref);log=[]
        path=CounterfactualPath(ctrl,ctrl.select_action,self.model,self.km,index,a,opti_state=op.snapshot,
            contract=contract or (lambda action,k:{'request':action.copy(),'reserve':np.ones(6),'provenance':'inert fixture; not a model prediction'}),sink=log.append)
        return a,ctrl,op,path,log
    def test_all_36_labels_and_exact_nine_input_failures(self):
        rows=labels(self.model,{i:a['actions'][0] for i,a in self.archives.items()},self.km)
        self.assertEqual(len(rows),36)
        bad={(r['state_id'],r['menu_index']) for r in rows if not r['commands']['admissible']}
        self.assertEqual(bad,{(0,0),(0,1),(0,2),(4,0),(4,3),(4,6),(8,0),(8,1),(8,2)})
        for r in rows:
            if not r['commands']['admissible']:self.assertEqual(min(r['commands']['lower_margins']),-1.3877787807814457e-17)
    def test_frozen_alternative_is_id0_index3_and_five_percent(self):
        rows=labels(self.model,{i:a['actions'][0] for i,a in self.archives.items()},self.km);r=select_alternative(rows)
        self.assertEqual((r['state_id'],r['menu_index']),(0,3));self.assertTrue(r['separation_from_nominal']['at_least_five_percent'])
    def test_no_duplicate_force_signatures_within_each_original_menu(self):
        rows=labels(self.model,{i:a['actions'][0] for i,a in self.archives.items()},self.km)
        for i in [0,4,8,12]:self.assertEqual(len({r['commands']['submitted_signature'] for r in rows if r['state_id']==i}),9)
        self.assertTrue(all(r['duplicate_of'] is None for r in rows))
    def test_zero_offset_native_300_query_inputs_and_memories_all_four_archives(self):
        for i in [0,4,8,12]:
            a,c,o,path,log=self.fixture(state=i);plain,po=controller(UP,a,self.ref)
            for k in range(300):
                expected=plain.select_action(a['states'][k],{'current_step':k})
                actual=path.select(a['states'][k],{'current_step':k})
                self.assertTrue(same(actual,expected));self.assertTrue(same(history(c,o.snapshot),history(plain,po.snapshot)))
                self.assertFalse(np.shares_memory(actual,c.u_prev));self.assertTrue(same(o.calls[-1],po.calls[-1]))
                self.assertTrue(same(c.x_prev,a['predicted_states'][k]));self.assertTrue(same(c.u_prev,a['predicted_inputs'][k]))
            self.assertEqual(o.i,300)
    def test_alternative_changes_only_first_detached_return_and_preserves_nominal_memory(self):
        a,c,o,path,log=self.fixture(index=3);plain,po=controller(UP,a,self.ref)
        for k in range(2):
            nominal=plain.select_action(a['states'][k],{'current_step':k}).copy();action=path.select(a['states'][k],{'current_step':k})
            self.assertEqual(same(action,nominal),k!=0);self.assertTrue(same(history(c,o.snapshot),history(plain,po.snapshot)))
        self.assertTrue(same(log[0]['causal_contract']['request'],log[0]['application_request']))
        self.assertFalse(same(log[0]['application_request'],log[0]['nominal_request']))
    def test_native_input_row_shift_is_not_replaced_by_column_shift(self):
        a,c,o,path,log=self.fixture()
        path.select(a['states'][0],{'current_step':0});u=c.u_prev.copy()
        path.select(a['states'][1],{'current_step':1});expected=u.copy();expected[:-1]=expected[1:]
        self.assertTrue(same(o.calls[1]['guesses']['u'],expected))
        wrong=u.copy();wrong[:,:-1]=wrong[:,1:];self.assertFalse(same(expected,wrong))
    def test_reference_preview_and_terminal_padding_native(self):
        a,c,o,path,log=self.fixture()
        for k in [0,220,221,299,300]:
            actual=c.get_references(k);expected=np.concatenate([self.ref[k:min(301,k+81)],np.repeat(self.ref[-1][None,:],max(0,k+81-301),axis=0)]).T
            self.assertTrue(same(actual,expected));self.assertEqual(actual.shape,(6,81))
    def test_bad_phase_or_nonempty_initial_memory_is_rejected_before_query(self):
        a,c,o,path,log=self.fixture()
        with self.assertRaises(ValueError):path.select(a['states'][0],{'current_step':1})
        self.assertEqual(o.i,0)
        c.u_prev=np.zeros((2,80))
        with self.assertRaises(ValueError):path.select(a['states'][0],{'current_step':0})
        self.assertEqual(o.i,0)
    def test_negative_input_label_cannot_be_applied_and_has_no_fallback(self):
        a,c,o,path,log=self.fixture(index=0)
        with self.assertRaises(ValueError):path.select(a['states'][0],{'current_step':0})
        self.assertEqual(log[-1]['status'],'input_inadmissible');self.assertFalse(log[-1]['fallback_applied']);self.assertEqual(path.step,0)
    def test_candidate_contract_uses_candidate_not_nominal_and_nonfinite_fails_closed(self):
        seen=[]
        def callback(action,k):seen.append(action.copy());return {'reserve':np.full(6,np.nan)}
        a,c,o,path,log=self.fixture(index=3,contract=callback)
        with self.assertRaises(ValueError):path.select(a['states'][0],{'current_step':0})
        self.assertFalse(same(seen[0],a['actions'][0]));self.assertTrue(np.isnan(log[-1]['causal_contract']['reserve']).all())
    def test_contract_predictor_receives_altered_action_without_running_a_model(self):
        a,c,o,path,log=self.fixture(index=3);request=a['actions'][0]+np.array([0.,-.05*(self.model['action_high'][1]-self.model['action_low'][1])]);seen=[]
        def spy(x,u,f):seen.append(u.copy());return a['native_model_predictions'][0].copy()
        b=candidate_contract(a['states'][0],request,a['states'][0,4],self.model,predict=spy)
        np.testing.assert_array_equal(seen[0],b['model_thrust']);self.assertTrue(same(b['request'],request))
    def test_observer_sees_current_substituted_motor_and_force_commands_in_native_mock_order(self):
        for index in [4,3]:
            a,c,o,path,log=self.fixture(index=index)
            with native_harness(UP,self.model,a['physics_states'][:41]) as (env,backend):
                with AppliedActionObserver(env,backend) as observer:
                    for k in range(2):
                        action=path.select(a['states'][k],{'current_step':k});env._advance_simulation(env._preprocess_control(action))
                    self.assertEqual(len(observer.steps),40)
                    first=observer.steps[0];self.assertTrue(np.array_equal(first['stale_cached_rpm_debug_only'],np.zeros(4)))
                    np.testing.assert_array_equal(first['direct_rpm_argument'],log[0]['source_predicted_commands']['rpm'])
                    np.testing.assert_array_equal(first['grouped_submitted_force_argument_sums'],log[0]['source_predicted_commands']['submitted_grouped_thrust'])
                    np.testing.assert_array_equal(observer.steps[20]['stale_cached_rpm_debug_only'],first['direct_rpm_argument'])
                    np.testing.assert_array_equal(observer.steps[20]['direct_rpm_argument'],a['rpms'][1])
    def test_zero_offset_preserves_native_mock_application_trace(self):
        a=self.archives[0];traces=[]
        for wrapped in [False,True]:
            _,c,o,path,log=self.fixture()
            with native_harness(UP,self.model,a['physics_states'][:41]) as (env,backend):
                for k in range(2):
                    u=path.select(a['states'][k],{'current_step':k}) if wrapped else c.select_action(a['states'][k],{'current_step':k})
                    env._advance_simulation(env._preprocess_control(u))
                traces.append((trace_digest(backend.trace),env.last_clipped_action.tobytes(),backend.states[:41].tobytes()))
        self.assertEqual(traces[0],traces[1])
    def test_passive_snapshot_uses_no_forward_kinematic_refresh(self):
        calls=[]
        def method(name,value):
            def call(*args,**kwargs):calls.append(name);return value
            return call
        backend=SimpleNamespace(getBasePositionAndOrientation=method('base',([0,0,1],[0,0,0,1])),
            getBaseVelocity=method('velocity',([0,0,0],[0,0,0])),getNumJoints=method('num_joints',1),
            getJointStates=method('joints',[(0.,0.,(0.,)*6,0.)]),getContactPoints=method('contacts',[]))
        env=SimpleNamespace(DRONE_ID=11,PYB_CLIENT=17,ctrl_step_counter=0,np_random=np.random.default_rng(1337),last_clipped_action=np.zeros(4))
        before=deepcopy(env.np_random.bit_generator.state);r=passive_reset_state(env,backend)
        self.assertEqual(calls,['base','velocity','num_joints','joints','contacts']);self.assertTrue(same(before,env.np_random.bit_generator.state))
        self.assertIn('not a complete PyBullet checkpoint',r['unobservable_internal_engine_state'])
    def test_native_reset_order_and_no_second_controller_reset(self):
        a=self.archives[0];trace=[];env=SimpleNamespace(last_clipped_action=np.zeros(4),ctrl_step_counter=0)
        env.reset=lambda:(trace.append('plant_reset') or a['states'][0].copy(),{'current_step':0})
        def ef():trace.append('plant_constructor');return env
        def cf(factory):trace.append('native_controller_factory_including_reset');return SimpleNamespace(x_prev=None,u_prev=None)
        result=initialize_native_context(ef,cf,lambda e:a['states'][0],np.array(self.model['cases']['0']),a['states'][0],lambda e:{'scripted':True})
        self.assertEqual(trace,['plant_constructor','native_controller_factory_including_reset','plant_reset']);self.assertIs(result[0],env)
    def test_reset_archive_bit_mismatch_is_not_hidden_by_tolerance(self):
        a=self.archives[0];env=SimpleNamespace(reset=lambda:(a['states'][0],{'current_step':0}),last_clipped_action=np.zeros(4),ctrl_step_counter=0)
        wrong=a['states'][0].copy();wrong[0]=np.nextafter(wrong[0],np.inf)
        with self.assertRaises(ValueError):initialize_native_context(lambda:env,lambda factory:SimpleNamespace(x_prev=None,u_prev=None),lambda e:wrong,wrong,a['states'][0],lambda e:{})
    def test_failure_record_keeps_nonfinite_diagnostics_and_partial_arrays(self):
        with tempfile.TemporaryDirectory() as tmp:
            failure_sink(tmp,{'diagnostic':np.nan,'status':'incomplete'},{'primal':np.array([np.nan]),'partial_states':self.archives[0]['states'][:1]})
            r=json.loads((Path(tmp)/'failure.json').read_text());self.assertIn('nonfinite_float',r['diagnostic'])
            self.assertTrue(np.isnan(np.load(Path(tmp)/'partial_context.npz')['primal']).all());self.assertFalse(r['fallback_applied'])
            with self.assertRaises(FileExistsError):failure_sink(tmp,{}, {})
    def test_status_categories_and_missing_evidence_never_count_as_safe(self):
        self.assertEqual(classify_branch(input_admissible=False),'input_inadmissible')
        self.assertEqual(classify_branch(input_admissible=True,duplicate_of=4),'duplicate')
        self.assertEqual(classify_branch(input_admissible=True),'incomplete_or_unverified')
        self.assertEqual(classify_branch(input_admissible=True,observed_violation=True),'empirically_unsafe')
        self.assertEqual(classify_branch(input_admissible=True,complete=True,numerical_verified=True,replays_verified=True),'empirically_safe')
    def test_pilot_plans_fit_existing_solve_and_step_balance(self):
        for scope,counts in [('A',(30,3000,180000)),('B',(12,1200,72000))]:
            jobs=job_plan(scope);self.assertEqual((len(jobs),sum(j['solver_calls'] for j in jobs),sum(j['physics_steps'] for j in jobs)),counts)
            self.assertLessEqual(counts[1],3070);self.assertLessEqual(counts[2],366000)


if __name__=='__main__':unittest.main()
