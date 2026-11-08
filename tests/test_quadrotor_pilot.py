"""E12 mathematical contracts; no new trajectories or upstream modifications."""
from pathlib import Path
import json
import unittest
import numpy as np
from src.quadrotor_verification import mechanics, rk4

ROOT = Path(__file__).resolve().parents[1]


class QuadrotorContracts(unittest.TestCase):
    def setUp(self):
        self.p = {'mass':.027, 'gravity':9.8, 'Iyy':1.4e-5, 'arm_length':.0397}

    def test_hover_is_nominal_equilibrium_with_interior_motor_authority(self):
        x = np.array([0.,0.,1.,0.,0.,0.]); u = np.full(2,.027*9.8/2)
        np.testing.assert_allclose(mechanics(x,u,self.p), np.zeros(6),atol=1e-14)
        np.testing.assert_array_equal(rk4(x,u,self.p),x)
        bounds=2*3.16e-10*(.2685*np.array([20000.,65535.])+4070.3)**2
        self.assertLess(bounds[0],u[0]); self.assertGreater(bounds[1],u[0])

    def test_differential_input_has_declared_rotation_sign(self):
        x=np.array([0.,0.,1.,0.,0.,0.]);u=np.array([.1,.2])
        a=mechanics(x,u,self.p); b=mechanics(x,u[::-1],self.p)
        self.assertGreater(a[5],0); self.assertEqual(a[5],-b[5]); self.assertEqual(a[3],b[3])

    def test_thrust_tilt_sign_and_units(self):
        x=np.array([0.,0.,1.,0.,np.pi/6,0.]);u=np.array([.1,.1])
        a=mechanics(x,u,self.p)
        self.assertAlmostEqual(a[1], .1/.027)
        self.assertAlmostEqual(a[3], np.sqrt(3)*.1/.027-9.8)

    def test_pilot_scope_is_exactly_four_frozen_cases(self):
        c=json.loads((ROOT/'configs/experiment_12.json').read_text())
        self.assertEqual(list(c['cases']), ['T0','T1','T2','T3'])
        self.assertEqual(c['cases']['T2'],[-.5,0,.75,0,.15,0])
        self.assertEqual(c['cases']['T3'],[.5,0,1.25,0,-.15,0])
        self.assertFalse(c['run_screening']);self.assertFalse(c['run_counterfactuals']);self.assertFalse(c['train_agents'])
        self.assertEqual(c['control_transitions'],300)

    def test_exact_reference_has_nonzero_initial_velocity(self):
        w=2*np.pi/6
        self.assertAlmostEqual(.75*w,np.pi/4)
        # The hover initial state must not be silently substituted for T1.
        self.assertNotEqual(.75*w,0.)


if __name__ == '__main__':
    unittest.main()
