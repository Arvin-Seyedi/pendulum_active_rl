"""V2 execution wiring fixtures; no live episodes, solves or engine steps."""
from pathlib import Path
import tempfile
import unittest
import numpy as np
from src.quadrotor_mpc80_execution_v2 import flatten_packet,SubmissionStream,FIELD_WIDTH,PHYSICAL_COLUMNS
from src.quadrotor_applied_action_v2 import AppliedActionObserver
from src.quadrotor_action_mock_v2 import native_harness
from src.quadrotor_pilot import read

ROOT=Path(__file__).resolve().parents[1]
UPSTREAM=Path('/private/tmp/e12_safe_control_gym')


class ExecutionV2Tests(unittest.TestCase):
    def fixture(self):
        m=read(ROOT/'results/benchmark_selection/mpc80_feas_v2_preflight/accepted_frozen.json')
        a=np.load(ROOT/'results/benchmark_selection/mmaq_v1/runs/H0_nonlinear80_1000_original/trajectory.npz')
        return m,a['actions'][:2].copy(),a['physics_states'][:41].copy(),a['rpms'][:2].copy()

    def record(self,folder,expected=None):
        m,u,x,rpm=self.fixture();stream=SubmissionStream(folder,'test',expected)
        try:
            with native_harness(UPSTREAM,m,x) as (env,b):
                with AppliedActionObserver(env,b,sink=stream.emit) as observer:
                    for command in u:env._advance_simulation(env._preprocess_control(command))
                    self.assertEqual(len(observer.steps),40)
        finally:stream.close()
        return np.fromfile(Path(folder)/'test.f64',dtype='<f8').reshape(-1,FIELD_WIDTH)

    def test_direct_packet_layout_and_completion_journal(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=self.record(tmp);m,u,x,rpm=self.fixture()
            self.assertEqual(rows.shape,(40,56))
            np.testing.assert_array_equal(rows[:,3:5],np.repeat(u,20,axis=0))
            np.testing.assert_array_equal(rows[:,13:17],np.repeat(rpm,20,axis=0))
            self.assertEqual(np.fromfile(Path(tmp)/'test_completed.u1',dtype='u1').tolist(),[1]*40)
            self.assertEqual(read(Path(tmp)/'test_counts.json')['applications'],40)

    def test_submission_repetition_is_bitwise(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            original=self.record(a);repeat=self.record(b,original)
            self.assertEqual(original.tobytes(),repeat.tobytes())

    def test_only_runtime_world_identifiers_are_ignored(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            original=self.record(a);expected=original.copy();expected[:,54:]+=100
            repeat=self.record(b,expected)
            self.assertEqual(original[:,PHYSICAL_COLUMNS].tobytes(),repeat[:,PHYSICAL_COLUMNS].tobytes())

    def test_changed_force_command_stops_mock_before_integration(self):
        with tempfile.TemporaryDirectory() as a,tempfile.TemporaryDirectory() as b:
            original=self.record(a);original[0,19]+=1e-4
            with self.assertRaisesRegex(ValueError,'Submitted motor/force commands disagree'):self.record(b,original)
            self.assertEqual(read(Path(b)/'test_counts.json')['completed_physics_steps'],0)

    def test_current_RPM_is_recorded_despite_stale_reset_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows=self.record(tmp)
            self.assertTrue(np.all(rows[0,13:17]>0));self.assertEqual(rows[0,2],0)
            self.assertEqual(rows[20,1],1)

    def test_parent_preflight_remains_nonexecuting_and_intact(self):
        p=ROOT/'results/benchmark_selection/mpc80_feas_v2_preflight'
        self.assertFalse(read(p/'preflight.json')['execution_authorized'])
        self.assertEqual(read(p/'resource_ledger.json')['charged']['cpu_seconds'],158.63)
        self.assertEqual(read(p/'resource_ledger.json')['charged']['solver_calls'],0)
