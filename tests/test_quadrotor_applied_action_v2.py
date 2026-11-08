"""Deterministic native-control-flow MOCK tests; zero real simulation calls."""
from pathlib import Path
from unittest.mock import patch
import os
import pickle
import unittest
import numpy as np
from src.quadrotor_applied_action_v2 import AppliedActionObserver,ObservabilityError
from src.quadrotor_action_mock_v2 import native_harness,trace_digest
from src.quadrotor_mpc80_execution_v1 import require_executed
from src.quadrotor_pilot import read

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=Path(os.environ.get('E12_UPSTREAM','/private/tmp/e12_safe_control_gym'))


class AppliedActionV2Tests(unittest.TestCase):
    def fixture(self):
        model=read(ROOT/'results/benchmark_selection/mpc80_feas_v1_preflight/accepted_frozen.json')
        a=np.load(ROOT/'results/benchmark_selection/mmaq_v1/runs/H0_nonlinear80_1000_original/trajectory.npz')
        return model,a['actions'][:2].copy(),a['physics_states'][:41].copy(),a['rpms'][:2].copy()

    def test_original_stale_zero_failure_reproduced_with_native_order(self):
        m,u,x,r=self.fixture()
        with native_harness(UPSTREAM,m,x) as (env,b):
            native=b.stepSimulation
            def old_callback(*a,**k):
                result=native(*a,**k);require_executed(env.last_clipped_action,m);return result
            with patch.object(b,'stepSimulation',old_callback):
                rpm=env._preprocess_control(u[0])
                with self.assertRaisesRegex(ValueError,'Negative executed input margin'):env._advance_simulation(rpm)
            self.assertEqual(b.index,1);self.assertTrue(np.array_equal(env.last_clipped_action,np.zeros(4)))
            self.assertEqual(len([t for t in b.trace if t[0]=='force']),4)

    def test_first_step_has_direct_current_rpm_not_reset_cache(self):
        m,u,x,r=self.fixture()
        with native_harness(UPSTREAM,m,x) as (env,b):
            with AppliedActionObserver(env,b,after_step=lambda pkt:require_executed(pkt['direct_rpm_argument'],m)) as observer:
                rpm=env._preprocess_control(u[0]);env._advance_simulation(rpm)
                packet=observer.steps[0]
                np.testing.assert_array_equal(packet['direct_rpm_argument'],r[0])
                np.testing.assert_array_equal(packet['stale_cached_rpm_debug_only'],np.zeros(4))
                self.assertTrue(packet['physics_returned']);self.assertEqual(len(observer.steps),20)

    def test_subsequent_transition_uses_new_command_not_previous_cache(self):
        m,u,x,r=self.fixture()
        with native_harness(UPSTREAM,m,x) as (env,b):
            with AppliedActionObserver(env,b) as observer:
                for action in u:env._advance_simulation(env._preprocess_control(action))
                self.assertEqual(len(observer.steps),40)
                for k,packet in enumerate(observer.steps):np.testing.assert_array_equal(packet['direct_rpm_argument'],r[k//20])
                np.testing.assert_array_equal(observer.steps[20]['stale_cached_rpm_debug_only'],r[0])
                np.testing.assert_array_equal(observer.steps[20]['direct_rpm_argument'],r[1])
                self.assertFalse(np.array_equal(r[0],r[1]))

    def test_native_force_torque_step_trace_bitwise_unchanged(self):
        m,u,x,r=self.fixture();results=[]
        for observed in [False,True]:
            with native_harness(UPSTREAM,m,x) as (env,b):
                if observed:
                    with AppliedActionObserver(env,b):
                        for action in u:env._advance_simulation(env._preprocess_control(action))
                else:
                    for action in u:env._advance_simulation(env._preprocess_control(action))
                results.append((trace_digest(b.trace),env.last_clipped_action.tobytes(),env.update_count,b.states[:b.index+1].tobytes()))
        self.assertEqual(results[0],results[1])

    def test_forwarded_input_and_native_return_object_preserved(self):
        m,u,x,r=self.fixture()
        with native_harness(UPSTREAM,m,x) as (env,b):
            native=env._preprocess_control;seen={}
            def spy(action):
                seen['argument']=action;seen['returned']=native(action);return seen['returned']
            env._preprocess_control=spy
            with AppliedActionObserver(env,b):
                action=u[0];result=env._preprocess_control(action)
                self.assertIs(seen['argument'],action)
                self.assertIs(result,seen['returned'])
                np.testing.assert_array_equal(seen['argument'],u[0])

    def test_hooks_restored_on_normal_exit(self):
        m,u,x,r=self.fixture()
        with native_harness(UPSTREAM,m,x) as (env,b):
            physics=env._physics;preprocess=env._preprocess_control;step=b.stepSimulation
            with AppliedActionObserver(env,b):pass
            self.assertIs(env._physics,physics);self.assertIs(env._preprocess_control,preprocess)
            self.assertEqual(b.stepSimulation,step)

    def test_hooks_restored_on_exception(self):
        m,u,x,r=self.fixture()
        with native_harness(UPSTREAM,m,x) as (env,b):
            physics=env._physics
            with self.assertRaises(RuntimeError):
                with AppliedActionObserver(env,b):raise RuntimeError('fixture')
            self.assertIs(env._physics,physics)

    def test_callback_copies_cannot_change_motor_or_plant_cache(self):
        m,u,x,r=self.fixture()
        with native_harness(UPSTREAM,m,x) as (env,b):
            def mutate(packet):packet['direct_rpm_argument'][:]=0
            with AppliedActionObserver(env,b,after_step=mutate) as observer:
                rpm=env._preprocess_control(u[0]);env._advance_simulation(rpm)
                np.testing.assert_array_equal(rpm,r[0]);np.testing.assert_array_equal(env.last_clipped_action,r[0])
                np.testing.assert_array_equal(observer.steps[0]['direct_rpm_argument'],r[0])

    def test_force_arguments_are_direct_and_grouped_thrust_is_labelled(self):
        m,u,x,r=self.fixture()
        with native_harness(UPSTREAM,m,x) as (env,b):
            with AppliedActionObserver(env,b) as observer:
                env._advance_simulation(env._preprocess_control(u[0]))
                packet=observer.steps[0]
                np.testing.assert_array_equal(packet['direct_submitted_force_z_by_motor'],r[0]**2*m['KF'])
                self.assertIn('derived_grouped_thrust_from_direct_rpm',packet)
                self.assertEqual(len(packet['submissions']),5)

    def test_missing_application_blocks_unknown_step(self):
        m,u,x,r=self.fixture()
        with native_harness(UPSTREAM,m,x) as (env,b):
            with AppliedActionObserver(env,b):
                with self.assertRaises(ObservabilityError):b.stepSimulation(physicsClientId=env.PYB_CLIENT)
            self.assertEqual(b.index,0)

    def test_application_without_preprocess_record_rejected(self):
        m,u,x,r=self.fixture()
        with native_harness(UPSTREAM,m,x) as (env,b):
            with AppliedActionObserver(env,b):
                env._physics(r[0])
                with self.assertRaises(ObservabilityError):b.stepSimulation(physicsClientId=env.PYB_CLIENT)
            self.assertEqual(b.index,0)

    def test_failed_force_call_preserved_without_fake_step(self):
        m,u,x,r=self.fixture()
        with native_harness(UPSTREAM,m,x) as (env,b):
            b.failure='force';events=[]
            with AppliedActionObserver(env,b,sink=events.append) as observer:
                with self.assertRaises(RuntimeError):env._advance_simulation(env._preprocess_control(u[0]))
                self.assertFalse(observer.applications[0]['force_application_returned'])
            self.assertEqual(b.index,0);self.assertTrue(events)

    def test_failed_mock_backend_step_is_not_claimed_as_completed(self):
        m,u,x,r=self.fixture()
        with native_harness(UPSTREAM,m,x) as (env,b):
            b.failure='step'
            with AppliedActionObserver(env,b) as observer:
                with self.assertRaises(RuntimeError):env._advance_simulation(env._preprocess_control(u[0]))
                self.assertFalse(observer.applications[0]['physics_returned'])
            self.assertEqual(b.index,0)

    def test_callback_failure_preserves_already_completed_mock_step(self):
        m,u,x,r=self.fixture()
        with native_harness(UPSTREAM,m,x) as (env,b):
            def fail(packet):raise RuntimeError('synthetic diagnostic rejection')
            with AppliedActionObserver(env,b,after_step=fail) as observer:
                with self.assertRaises(RuntimeError):env._advance_simulation(env._preprocess_control(u[0]))
                self.assertTrue(observer.steps[0]['physics_returned'])
            self.assertEqual(b.index,1)

    def test_nonfinite_motor_argument_fails_closed(self):
        m,u,x,r=self.fixture()
        with native_harness(UPSTREAM,m,x) as (env,b):
            with AppliedActionObserver(env,b):
                env._preprocess_control(u[0]);env._physics(np.full(4,np.nan))
                with self.assertRaises(ObservabilityError):b.stepSimulation(physicsClientId=env.PYB_CLIENT)
            self.assertEqual(b.index,0)

    def test_global_rng_state_unchanged_by_observer(self):
        m,u,x,r=self.fixture()
        with native_harness(UPSTREAM,m,x) as (env,b):
            before=pickle.dumps(np.random.get_state())
            with AppliedActionObserver(env,b):env._advance_simulation(env._preprocess_control(u[0]))
            self.assertEqual(before,pickle.dumps(np.random.get_state()))

    def test_non_PYB_mode_is_not_silently_qualified(self):
        m,u,x,r=self.fixture()
        with native_harness(UPSTREAM,m,x) as (env,b):
            env.PHYSICS='dyn'
            with self.assertRaises(ObservabilityError):
                with AppliedActionObserver(env,b):pass

    def test_disturbances_are_not_silently_qualified(self):
        m,u,x,r=self.fixture()
        with native_harness(UPSTREAM,m,x) as (env,b):
            env.disturbances={'dynamics':'outside contract'}
            with self.assertRaises(ObservabilityError):
                with AppliedActionObserver(env,b):pass
