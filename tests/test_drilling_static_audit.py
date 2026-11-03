"""Independent exact-fraction fixtures for the diagnostic interval arithmetic."""
from fractions import Fraction as F
import unittest
import numpy as np
from src.drilling_static_interval import Interval,solve,multiply
from src.drilling_static_audit import rational_residuals
from src.drilling_static_audit import archived_points
from unittest.mock import patch


def bounds(x):
    return F(*x.lo.as_integer_ratio()),F(*x.hi.as_integer_ratio())


class StaticAuditTests(unittest.TestCase):
    def assertEncloses(self,x,value):
        lo,hi=bounds(x);self.assertLessEqual(lo,value);self.assertGreaterEqual(hi,value)

    def test_binary64_inputs_are_exact(self):
        for p in (128,256):
            x=Interval(.1,p);self.assertEqual(bounds(x),(F.from_float(.1),)*2)

    def test_signed_rational_arithmetic(self):
        for p in (128,256):
            for a,b in [(3,-7),(-11,5),(1,3),(-1,-3)]:
                x,y=Interval(a,p),Interval(b,p)
                self.assertEncloses(x/y,F(a,b))
                self.assertEncloses(-(x/y),-F(a,b))
                self.assertEncloses((x/y)*y-x,F(0))

    def test_outward_negation_retains_precision(self):
        for p in (128,256):
            x=Interval(1,p)/7
            self.assertEqual(bounds(-x),(-bounds(x)[1],-bounds(x)[0]))

    def test_interval_multiplication_crossing_zero(self):
        x=Interval(-3,128,5)*Interval(-7,128,2)
        self.assertEqual(bounds(x),(F(-35),F(21)))

    def test_interval_solve_against_exact_solution(self):
        for p in (128,256):
            a=[[Interval(x,p) for x in row] for row in [[2,1],[1,3]]]
            b=[[Interval(1,p)],[Interval(2,p)]]
            x=solve(a,b)
            self.assertEncloses(x[0][0],F(1,5));self.assertEncloses(x[1][0],F(3,5))
            r=multiply(a,x)
            self.assertEncloses(r[0][0],F(1));self.assertEncloses(r[1][0],F(2))

    def test_unresolved_domains_fail_closed(self):
        with self.assertRaises(ArithmeticError):Interval(1)/Interval(-1,128,1)
        with self.assertRaises(ArithmeticError):Interval(2).function('sin')
        with self.assertRaises(ValueError):Interval(1,64)

    def test_exact_residuals_distinguish_zero_from_float_cancellation(self):
        a={'K_M':np.eye(10),'D_M':np.eye(10),'IC':np.ones((10,1)),
           'Vs':np.ones((10,1)),'F_M_d':2*np.ones((10,1))}
        r=rational_residuals(a)
        self.assertTrue(r['r0_all_exact_zero']);self.assertFalse(r['KM_Vs_all_exact_zero'])

    def test_transcendental_known_values_and_nested_precision(self):
        for name,expected in [('sin',0),('cos',1),('tan',0),('exp',1),('tanh',0)]:
            self.assertEncloses(Interval(0).function(name),F(expected))
        for name in ('sin','cos','tan','exp','tanh','sqrt'):
            lo,hi=bounds(Interval(.5,128).function(name))
            l2,h2=bounds(Interval(.5,256).function(name))
            self.assertLessEqual(lo,l2);self.assertGreaterEqual(hi,h2)

    def test_native_row_timestamp_shape_in_pointwise_audit(self):
        a={'K_M':np.eye(10),'D_M':np.eye(10),'IC':np.zeros((10,1)),
           'Vs':np.ones((10,1)),'F_M_d':np.ones((10,1)),
           'M1_global':np.eye(20),'angle':np.zeros((1,5)),
           'WOB_d':np.zeros((1,1)),'TOB_d':np.zeros((1,1)),'UU1':np.zeros((20,1))}
        s={'indices':np.array([0,1]),'E':np.zeros((20,2)),'F':np.zeros((20,2)),
           'U':np.zeros((20,2)),'X':np.zeros((20,2)),'t':np.array([[0.,.0001]])}
        with patch('src.drilling_static_audit.loads',return_value={'WOB':0.,'TOB':0.}):
            result=archived_points(a,s,rational_residuals(a))
        self.assertEqual([r['time_seconds'] for r in result],[0.,.0001])
        self.assertTrue(all(r['exact_dyadic_identity'] for r in result))


if __name__=='__main__':unittest.main()
