"""E11A provenance, strict raw safety, fallback and no-overwrite fixtures."""
import unittest
import numpy as np
from src.drilling_reproduction import cases
from src.drilling_numerical_audit import safety_summary,lookup_history,kv

class DrillingReproductionTests(unittest.TestCase):
    def test_case_bindings_preserve_distinct_variants(self):
        c=cases();self.assertEqual([r['case_id'] for r in c],['P0','P1','P2','P3','P4','P5'])
        self.assertNotEqual(c[0]['initial_expression'],c[2]['initial_expression'])
        self.assertIn('0.00056292,4.5930',c[3]['initial_expression'])
        self.assertFalse(c[5]['archived_case']);self.assertEqual(c[5]['initial_expression'],'zeros(20,1)')
    def test_strict_boundary_and_failed_run_are_never_safe(self):
        E=np.zeros((20,2));Vs=np.tile([.001,10.],5)
        good,*_=safety_summary(E,Vs,1.,True);self.assertTrue(good['empirically_strictly_safe'])
        failed,*_=safety_summary(E,Vs,1.,False);self.assertFalse(failed['empirically_strictly_safe'])
        E[19,1]=5.7
        edge,*_=safety_summary(E,Vs,1.,True);self.assertFalse(edge['empirically_strictly_safe'])
    def test_nonreal_or_nonfinite_physical_states_are_invalid(self):
        E=np.zeros((20,2),complex);Vs=np.tile([.001,10.],5);E[0,1]=1j
        result,*_=safety_summary(E,Vs,1.,True);self.assertEqual(result['invalid_physical_state_count'],1)
        self.assertFalse(result['empirically_strictly_safe'])
    def test_lookup_fallback_retains_gain_and_previous_transformed_coordinates(self):
        table=np.empty((39,39),object)
        for i in range(39):
            for j in range(39):
                cell=np.empty((1,3),object);cell[0,2]=np.ones((2,20));table[i,j]=cell
        z=np.zeros((20,3));z[-2:,1:]=1
        a={'n':np.array([[3]]),'cnt1':np.array([[0,20,99]]),'cnt2':np.array([[0,20,20]]),
           'Z_b':z,'Mat_data2':table,'u1':np.array([[0,2,2]]),'u2':np.array([[0,2,2]])}
        r,idx,failed=lookup_history(a);self.assertEqual(r['failures'],1)
        self.assertTrue(r['stale_transformed_memory_exact']);self.assertEqual(idx[0,2],20)
    def test_nominal_reference_is_strictly_inside_cubic_bounds(self):
        lo,up=kv(np.array([0.]));self.assertLess(lo[0],0);self.assertGreater(up[0],0)

if __name__=='__main__':unittest.main()
