import unittest
import numpy as np
from src.quadrotor_mmaq_record_audit_v1 import latency_accounting,one_sided_complementarity


class FailedCallAccounting(unittest.TestCase):
    def test_failed_unmeasured_call_is_not_replaced_with_zero(self):
        r=latency_accounting({'solver_calls':3},[.001,.002])
        self.assertEqual(r['unmeasured_failed_calls'],1)
        self.assertFalse(r['complete_latency_accounting'])
        self.assertIsNotNone(r['limitation'])

    def test_complete_calls_preserve_all_observed_overruns(self):
        r=latency_accounting({'solver_calls':3},[.001,.021,.08])
        self.assertEqual(r['observed_deadline_overruns'],2)
        self.assertEqual(r['unmeasured_failed_calls'],0)
        self.assertTrue(r['complete_latency_accounting'])

    def test_inconsistent_attempt_count_rejected(self):
        with self.assertRaises(AssertionError):latency_accounting({'solver_calls':0},[.01])

    def test_tiny_wrong_sign_upper_multiplier_has_finite_residual(self):
        r=one_sided_complementarity(np.array([-.1]),np.array([-np.inf]),np.array([0.]),np.array([-1e-30]))
        self.assertTrue(np.isfinite(list(r.values())).all())
        self.assertAlmostEqual(r['dual_sign_residual'],1e-30,places=40)
        legacy=-1e-30*(-.1-(-np.inf))
        self.assertFalse(np.isfinite(legacy))
