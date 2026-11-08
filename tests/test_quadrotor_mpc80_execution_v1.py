"""Static dispatch/capture wiring checks; no new control problems or episodes."""
import unittest
from pathlib import Path
import numpy as np
from src.quadrotor_mpc80_execution_v1 import require_raw_state,require_executed,SOURCES
from src.quadrotor_pilot import read

ROOT=Path(__file__).resolve().parents[1]


class MPC80ExecutionTests(unittest.TestCase):
    def model(self):return read(ROOT/'results/benchmark_selection/mpc80_feas_v1_preflight/accepted_frozen.json')

    def test_raw_boundary_is_inclusive(self):
        m=self.model();x=np.asarray(m['state_low']);lo,hi=require_raw_state(x,0,m)
        self.assertTrue(np.all(lo==0));self.assertTrue(np.all(hi>=0))

    def test_even_one_ulp_negative_margin_is_rejected(self):
        m=self.model();x=np.asarray(m['state_low']).copy();x[0]=np.nextafter(x[0],-np.inf)
        with self.assertRaises(ValueError):require_raw_state(x,0,m)

    def test_contact_is_rejected_without_fallback(self):
        with self.assertRaises(ValueError):require_raw_state(np.array([0,0,1,0,0,0]),1,self.model())

    def test_nonfinite_raw_state_rejected(self):
        for v in [np.nan,np.inf,-np.inf]:
            with self.assertRaises(ValueError):require_raw_state(np.array([v,0,1,0,0,0]),0,self.model())

    def test_hover_mapping_passes_actual_input_check(self):
        m=self.model();rpm=np.full(4,np.sqrt(.1323/(2*m['KF'])))
        np.testing.assert_allclose(require_executed(rpm,m),[.1323,.1323],rtol=0,atol=1e-16)

    def test_out_of_bound_executed_input_is_rejected(self):
        m=self.model()
        for rpm in [np.zeros(4),np.full(4,1e9),np.full(4,np.nan)]:
            with self.assertRaises(ValueError):require_executed(rpm,m)

    def test_immutable_preflight_still_disables_execution(self):
        c=read(ROOT/'configs/experiment_12_mpc80_feas_v1.json')
        self.assertFalse(c['execution_authorized']);self.assertEqual(c['original_overall_qualification'],'FAILED')
        self.assertEqual(c['budget']['solver_calls'],9600);self.assertEqual(c['budget']['physics_steps'],576000)

    def test_only_new_execution_files_in_binding(self):
        self.assertEqual(len(SOURCES),3)
        self.assertNotIn('src/quadrotor_pilot.py',SOURCES)
        self.assertNotIn('src/quadrotor_mpc80_feasibility_v1.py',SOURCES)
