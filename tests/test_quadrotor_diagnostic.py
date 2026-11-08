"""Mathematical tests only. Do not instantiate PyBullet or controllers."""
import unittest
from pathlib import Path
import json
from fractions import Fraction as F
import numpy as np
from src.quadrotor_diagnostic import continuous_prediction,interval_reconstruction
from src.quadrotor_compatibility_v1 import configure_linear_conic_qrqp_v1


class DiagnosticTests(unittest.TestCase):
    def setUp(self):self.f={'mass':.027,'Iyy':1.4e-5,'gravity':9.8}

    def test_zero_differential_thrust_has_constant_pitch_rate(self):
        x=np.array([[0.,0.,1.,0.,.1,.2]]);u=np.full((1,2),.1323)
        y=continuous_prediction(x,u,self.f,.028)
        self.assertAlmostEqual(y[0,5],.2);self.assertAlmostEqual(y[0,4],.104)

    def test_quadrature_reproduces_constant_orientation_translation(self):
        x=np.array([[0.,.3,1.,-.2,.1,0.]]);u=np.full((1,2),.1323)
        y=continuous_prediction(x,u,self.f,.028)
        acc=.2646/.027*np.array([np.sin(.1),np.cos(.1)])-np.array([0.,9.8])
        np.testing.assert_allclose(y[0,[1,3]],x[0,[1,3]]+.02*acc,atol=1e-15)
        np.testing.assert_allclose(y[0,[0,2]],x[0,[0,2]]+.02*x[0,[1,3]]+.5*.02**2*acc,atol=1e-15)

    def test_semiimplicit_rotation_matches_finite_sum(self):
        x=np.array([[0.,0.,1.,0.,0.,0.]]);u=np.array([[.1,.2]]);alpha=.028*.1/1.4e-5
        y=interval_reconstruction(x,u,x[:,4],self.f,.028,1000,False)
        self.assertAlmostEqual(y[0,5],alpha*.02)
        self.assertAlmostEqual(y[0,4],alpha*.001**2*20*21/2)

    def test_adapter_rejects_unreviewed_dispatch(self):
        with self.assertRaises(ValueError):configure_linear_conic_qrqp_v1(None,'other','ipopt',{'expand':True,'ipopt.max_iter':100})
        with self.assertRaises(ValueError):configure_linear_conic_qrqp_v1(None,'qrqp','ipopt',{'expand':True,'ipopt.max_iter':101})

    def test_exact_QP_KKT_reference(self):
        H=[[F(4),F(1)],[F(1),F(2)]];G=[[-1,0],[0,-1],[1,0],[0,1],[1,1]];b=[F(0),F(0),F(4,5),F(4,5),F(1)]
        for q,x,lam in [([-5,-3],[F(3,4),F(1,4)],[0,0,0,0,F(7,4)]),
                        ([-1,-1],[F(1,7),F(3,7)],[0,0,0,0,0]),
                        ([2,-3],[F(0),F(4,5)],[F(14,5),0,0,F(7,5),0])]:
            grad=[sum(H[i][j]*x[j] for j in range(2))+q[i] for i in range(2)]
            self.assertEqual([grad[i]+sum(G[k][i]*lam[k] for k in range(5)) for i in range(2)],[0,0])
            for row,limit,multiplier in zip(G,b,lam):
                slack=limit-sum(row[i]*x[i] for i in range(2));self.assertGreaterEqual(slack,0)
                self.assertGreaterEqual(multiplier,0);self.assertEqual(slack*multiplier,0)

    def test_saved_angular_attributions_against_closed_forms(self):
        root=Path(__file__).resolve().parents[1];out=root/'results/benchmark_selection/compatibility_diagnostic'
        if not (out/'signed_error_components.npz').exists():self.skipTest('Saved diagnostic evidence is not present')
        d=json.loads((out/'transition_diagnosis.json').read_text());a=np.load(out/'signed_error_components.npz')
        offset=0
        for record in d['records']:
            pilot=np.load(root/'results/benchmark_selection/technical_pilot/runs'/record['record']/'trajectory.npz')
            u=pilot['clipped_actions'];n=len(u);du=u[:,1]-u[:,0];piece=slice(offset,offset+n)
            geometry=.02*(.0397/np.sqrt(2)-.028)*du/1.4e-5
            np.testing.assert_allclose(a['geometry'][piece,5],geometry,rtol=0,atol=3e-15)
            angular_position=-.5*(.028*du/1.4e-5)*.02/record['frequency']
            np.testing.assert_allclose(a['semiimplicit_discretization'][piece,4],angular_position,rtol=0,atol=3e-15)
            offset+=n
        self.assertEqual(offset,7200)

    def test_signed_decomposition_closes_without_hiding_cancellation(self):
        out=Path(__file__).resolve().parents[1]/'results/benchmark_selection/compatibility_diagnostic'
        if not (out/'signed_error_components.npz').exists():self.skipTest('Saved diagnostic evidence is not present')
        a=np.load(out/'signed_error_components.npz')
        total=sum(a[k] for k in ['native_RK4_quadrature','geometry','semiimplicit_discretization','cached_link_timing','remaining_recurrence_residual'])
        np.testing.assert_allclose(total,a['native_total'],rtol=0,atol=1e-14)
        self.assertGreater(np.max(np.abs(a['geometry_aligned_continuous_error'][:,1])),.001)
        self.assertLess(np.max(np.abs(a['remaining_recurrence_residual'])),1e-12)


if __name__=='__main__':unittest.main()
