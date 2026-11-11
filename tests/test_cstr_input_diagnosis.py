"""Static arithmetic and archive checks; no optimization or integration."""
from pathlib import Path
from fractions import Fraction
import unittest
import numpy as np
from src.cstr_input_diagnosis import relaxation,blocked
from src.cstr_verification import require_input

P=Path(__file__).resolve().parents[1]/'results/cstr_expert_pilot/technical_v1/runs/C2_original_scientific'

class CSTRInputDiagnosisTests(unittest.TestCase):
    def test_relaxation_uses_NLP_coordinates_and_absolute_cap(self):
        self.assertEqual(relaxation(.05),1e-8)
        self.assertEqual(relaxation(-4.25),float(Fraction.from_float(1e-8)*Fraction(17,4)))
        self.assertEqual(relaxation(1e8),1e-4)

    def test_archived_action_is_below_bound_before_rescaling(self):
        with np.load(P/'attempts/0000/candidate.npz') as a:
            x=float(a['z'][336]);b=float(a['lbx'][336])
            self.assertLess(x,b);self.assertGreater(x,b-relaxation(b))
            self.assertEqual(x*100,float.fromhex('0x1.3ffffdb18aaacp+2'))
            self.assertLess(abs(float(Fraction(x*100)-100*Fraction(x))),1e-15)

    def test_source_tolerance_does_not_override_strict_input_check(self):
        with self.assertRaisesRegex(AssertionError,'Negative raw'):
            require_input([float.fromhex('0x1.3ffffdb18aaacp+2'),-.013781197157147958])

    def test_rejected_command_was_not_integrated(self):
        with np.load(P/'trajectory.npz') as a:
            self.assertEqual(a['states'].shape,(1,4));self.assertEqual(a['commands'].shape,(0,2))
        self.assertFalse((P/'integrations').exists())
        self.assertFalse((P/'attempts/0000/postquery_memory.npz').exists())
        self.assertFalse((P/'attempts/0000/native_timing.json').exists())

    def test_read_only_entrypoint_guard(self):
        with self.assertRaisesRegex(RuntimeError,'prohibits'):
            blocked()

if __name__=='__main__':unittest.main()
