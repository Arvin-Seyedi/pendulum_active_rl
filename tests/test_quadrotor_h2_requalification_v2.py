"""Target-path preflight tests; synthetic callbacks only, no real solves."""
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from src.quadrotor_h2_requalification_v2 import IndependentCandidateGuard
from src.quadrotor_qualification_telemetry_v2 import QualificationFailure


class H2Requalification(unittest.TestCase):
    def candidate(self):
        return {'primal':np.array([0.]),'parameters':np.array([1.]),'dual':np.array([-1e-30]),
                'objective':np.array([0.]),'constraint_values':np.array([-.1]),'lower':np.array([-np.inf]),
                'upper':np.array([0.]),'lagrangian_gradient':np.array([-1e-30]),
                'native_stats':{'success':True,'return_status':'Solve_Succeeded'},
                'solver_status':'Solve_Succeeded','provenance':'static_test_candidate'}

    def readout(self,z,p,lam):return [np.array([0.]),np.array([-.1]),np.array([-np.inf]),np.array([0.]),np.array([-1e-30])]
    def snapshot(self):return {'physics_states':np.zeros((21,6)),'physics_contacts':np.zeros(21)}

    def test_corrected_one_sided_point_passes_independent_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            guard=IndependentCandidateGuard(Path(tmp)/'call',{},readout=self.readout)
            action=guard.select(lambda:[.1,.2],self.candidate,self.snapshot)
            np.testing.assert_array_equal(action,[.1,.2])
            state=json.loads((guard.folder/'status.json').read_text())
            self.assertEqual(state['independent_residuals']['dual_sign_inf'],1e-30)
            self.assertTrue((guard.folder/'independent_constraint_diagnostics.json').exists())

    def test_mismatched_native_point_is_saved_before_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate=self.candidate();candidate['objective']=np.array([1.])
            guard=IndependentCandidateGuard(Path(tmp)/'call',{},readout=self.readout)
            with self.assertRaises(QualificationFailure):guard.select(lambda:[.1,.2],lambda:candidate,self.snapshot)
            np.testing.assert_array_equal(np.load(guard.folder/'candidate.npz')['objective'],[1.])
            self.assertTrue((guard.folder/'prefix_after.npz').exists())
            self.assertTrue((guard.folder/'independent_constraint_diagnostics.json').exists())

    def test_nonfinite_candidate_and_native_failure_both_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate=self.candidate();candidate['primal']=np.array([np.nan]);candidate['solver_status']='unknown'
            guard=IndependentCandidateGuard(Path(tmp)/'call',{},readout=self.readout)
            def fail():raise RuntimeError('Synthetic native failure')
            with self.assertRaises(QualificationFailure):guard.select(fail,lambda:candidate,self.snapshot)
            self.assertTrue(np.isnan(np.load(guard.folder/'candidate.npz')['primal']).all())
            state=json.loads((guard.folder/'status.json').read_text())
            self.assertFalse(state['fallback_action_applied'])
            self.assertTrue(state['timing_summary']['native_action_selection_measured'])

    def test_repeated_extraction_does_not_overwrite_first_capture(self):
        with tempfile.TemporaryDirectory() as tmp:
            candidate=self.candidate();guard=IndependentCandidateGuard(Path(tmp)/'call',{},readout=self.readout)
            guard.preserve_candidate(candidate)
            before=(guard.folder/'candidate.npz').read_bytes()
            guard.select(lambda:[.1,.2],lambda:candidate,self.snapshot)
            self.assertEqual(before,(guard.folder/'candidate.npz').read_bytes())

    def test_scope_exactly_H2_and_two_phases(self):
        root=Path(__file__).resolve().parents[1];c=json.loads((root/'configs/experiment_12_h2_requalification_v2.json').read_text())
        self.assertEqual(c['case'],'H2');self.assertEqual(c['controller'],'nonlinear20')
        self.assertEqual(c['maximum_additional_solver_calls'],600);self.assertEqual(c['maximum_additional_physics_steps'],36000)
        self.assertFalse(c['run_linear_MPC']);self.assertFalse(c['run_screening']);self.assertFalse(c['run_probes'])
