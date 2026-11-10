"""Additional zero-solve archive-parity tests. No control/physics calls."""
from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import unittest,numpy as np
from src.quadrotor_counterfactual_context_v1 import require_archived_query,parity_query
ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'results/benchmark_selection/mpc80_feas_v2_study/runs/0_nonlinear80_1000_original'


class ContextParityTests(unittest.TestCase):
    def fixture(self,k=0):
        z=dict(np.load(FOLDER/'attempts'/f'{k:04d}'/'candidate.npz'));w=dict(np.load(FOLDER/'attempts'/f'{k:04d}'/'warmstart.npz'))
        p=z['parameters'].reshape(-1);return z,w,p[:6].copy(),p[6:].reshape(6,81,order='F').copy()
    def test_archived_first_and_terminal_padding_queries_pass(self):
        for k in [0,1,220,221,299]:
            z,w,x,r=self.fixture(k);self.assertTrue(require_archived_query(FOLDER,k,x,r,z,w))
    def test_changed_dual_is_not_hidden_by_matching_returned_action(self):
        z,w,x,r=self.fixture();z['dual']=z['dual'].copy();z['dual'].reshape(-1)[0]=np.nextafter(z['dual'].reshape(-1)[0],np.inf)
        with self.assertRaisesRegex(ValueError,'dual'):require_archived_query(FOLDER,0,x,r,z,w)
    def test_nonfinite_candidate_is_rejected(self):
        z,w,x,r=self.fixture();z['primal']=z['primal'].copy();z['primal'].reshape(-1)[0]=np.nan
        with self.assertRaises(ValueError):require_archived_query(FOLDER,0,x,r,z,w)
    def test_initial_guess_and_memory_changes_are_rejected(self):
        for field in ['initial_guess_not_solver_candidate','previous_x','previous_u']:
            z,w,x,r=self.fixture(1);w[field]=w[field].copy();w[field].reshape(-1)[0]+=1e-6
            with self.assertRaises(ValueError):require_archived_query(FOLDER,1,x,r,z,w)
    def test_phase_preview_or_observation_mismatch_fails(self):
        z,w,x,r=self.fixture();x[0]=np.nextafter(x[0],np.inf)
        with self.assertRaises(ValueError):require_archived_query(FOLDER,0,x,r,z,w)
        z,w,x,r=self.fixture();r[0,0]+=1e-6
        with self.assertRaises(ValueError):require_archived_query(FOLDER,0,x,r,z,w)
    def test_alternative_checks_first_query_but_does_not_demand_nominal_successors(self):
        z,w,x,r=self.fixture();action=np.array([.1,.2]);ctrl=SimpleNamespace(get_references=lambda k:r)
        q=parity_query(lambda x,i:action,lambda:{'candidate':z,'warmstart':w},ctrl,FOLDER,zero_offset=False)
        self.assertIs(q(x,{'current_step':0}),action)
        self.assertIs(q(np.ones(6),{'current_step':1}),action)
    def test_composed_query_verifies_before_detached_alternative(self):
        import json,xml.etree.ElementTree as ET
        from src.quadrotor_counterfactual_context_v1 import prepare_selector
        from src.quadrotor_counterfactual_mock_v1 import controller
        archive=dict(np.load(FOLDER/'trajectory.npz'));model=json.loads((FOLDER.parents[1]/'frozen.json').read_text());reference=np.load(FOLDER.parents[1]/'reference.npy')
        up=Path('/private/tmp/e12_safe_control_gym');km=float(ET.parse(up/'safe_control_gym/envs/gym_pybullet_drones/assets/cf2x.urdf').getroot().find('properties').attrib['km'])
        ctrl,op=controller(up,archive,reference);z,w,x,r=self.fixture();events=[]
        path=prepare_selector(ctrl,ctrl.select_action,lambda:{'candidate':z,'warmstart':w},FOLDER,archive,model,km,3,
            opti_state=op.snapshot,contract=lambda action,k:{'reserve':np.ones(6),'provenance':'inert fixture'},sink=events.append)
        applied=path.select(x,{'current_step':0})
        self.assertFalse(np.shares_memory(applied,ctrl.u_prev));np.testing.assert_array_equal(ctrl.u_prev,archive['predicted_inputs'][0])
        self.assertEqual(applied[0],archive['actions'][0,0]);self.assertLess(applied[1],archive['actions'][0,1]);self.assertEqual(events[-1]['status'],'ready_for_native_step')
    def test_composed_parity_rejection_returns_no_candidate_action(self):
        import json,xml.etree.ElementTree as ET
        from src.quadrotor_counterfactual_context_v1 import prepare_selector
        from src.quadrotor_counterfactual_mock_v1 import controller
        archive=dict(np.load(FOLDER/'trajectory.npz'));model=json.loads((FOLDER.parents[1]/'frozen.json').read_text());reference=np.load(FOLDER.parents[1]/'reference.npy')
        up=Path('/private/tmp/e12_safe_control_gym');km=float(ET.parse(up/'safe_control_gym/envs/gym_pybullet_drones/assets/cf2x.urdf').getroot().find('properties').attrib['km'])
        ctrl,op=controller(up,archive,reference);z,w,x,r=self.fixture();z['dual']=z['dual']+1e-3;events=[]
        path=prepare_selector(ctrl,ctrl.select_action,lambda:{'candidate':z,'warmstart':w},FOLDER,archive,model,km,3,
            opti_state=op.snapshot,contract=lambda action,k:None,sink=events.append)
        with self.assertRaisesRegex(ValueError,'dual'):path.select(x,{'current_step':0})
        self.assertEqual(path.step,0);self.assertEqual(events[-1]['status'],'failed_or_incomplete')
    def test_original_guard_failure_is_propagated_without_fallback(self):
        def fail(x,i):raise RuntimeError('retained numerical solver failure')
        q=parity_query(fail,lambda:None,SimpleNamespace(),FOLDER,zero_offset=True)
        with self.assertRaisesRegex(RuntimeError,'retained numerical'):q(np.ones(6),{'current_step':0})


if __name__=='__main__':unittest.main()
