"""Independent checker fixtures: no simulator, optimization, or MPFR producer."""

from copy import deepcopy
from fractions import Fraction
import unittest

from src.viability_checker import MODEL, SCHEMA, check_extremal_certificate, sine_bounds


def interval(lower, upper=None):
    lower = Fraction(lower)
    upper = lower if upper is None else Fraction(upper)
    return {'lo': [str(lower.numerator), str(lower.denominator)],
            'hi': [str(upper.numerator), str(upper.denominator)]}


def fixture(sign=1):
    """At theta=+/-1, zero speed, even opposing torque cannot prevent exit."""
    initial = [float(sign), 0.]
    theta = interval(Fraction(101, 100), Fraction(102, 100)) if sign == 1 else interval(Fraction(-102, 100), Fraction(-101, 100))
    omega = interval(Fraction(3, 10), Fraction(2, 5)) if sign == 1 else interval(Fraction(-2, 5), Fraction(-3, 10))
    return {'schema': SCHEMA, 'model': dict(MODEL), 'state_id': 99,
            'initial_hex': [v.hex() for v in initial], 'precision_bits': 128,
            'action': -2*sign, 'horizon': 200, 'status': 'certified_nonviable',
            'decisive_step': 1, 'crossing_component': 'theta',
            'trace': [{'step': 0, 'theta': interval(sign), 'omega': interval(0)},
                      {'step': 1, 'theta': theta, 'omega': omega}]}


class IndependentCertificateTests(unittest.TestCase):
    def test_both_extremal_certificate_directions(self):
        for sign in (-1, 1):
            record = fixture(sign)
            result = check_extremal_certificate(record, 99, record['initial_hex'])
            self.assertTrue(result['passed'] and result['certified'] and result['model_only'])
            self.assertEqual(result['check_count'], 1)
            self.assertEqual(result['numerical_tolerance'], '0')

    def test_exact_sine_bounds_symmetry_and_nesting(self):
        self.assertEqual(sine_bounds(Fraction(0)), (0, 0))
        lower, upper = sine_bounds(Fraction(1), 6)
        self.assertGreater(lower, Fraction(5, 6))
        self.assertLess(upper, 1)
        refined = sine_bounds(Fraction(1), 12)
        self.assertLessEqual(lower, refined[0])
        self.assertLessEqual(refined[1], upper)
        self.assertEqual(sine_bounds(Fraction(-1), 6), (-upper, -lower))
        with self.assertRaises(ValueError):
            sine_bounds(Fraction(1001, 1000))

    def test_state_identity_and_exact_initial_binary_values(self):
        record = fixture()
        with self.assertRaisesRegex(ValueError, 'State ID'):
            check_extremal_certificate(record, expected_state_id=98)
        with self.assertRaisesRegex(ValueError, 'binary values'):
            check_extremal_certificate(record, expected_initial_hex=[1.0.hex(), (-0.0).hex()])
        record['trace'][0]['theta'] = interval(Fraction(999, 1000), 1)
        with self.assertRaisesRegex(ValueError, 'exact binary initial point'):
            check_extremal_certificate(record)

    def test_unapproved_model_and_nonextremal_action_rejected(self):
        record = fixture()
        record['model']['dt'] = '1/10'
        with self.assertRaisesRegex(ValueError, 'model'):
            check_extremal_certificate(record)
        record = fixture()
        record['action'] = -1
        with self.assertRaisesRegex(ValueError, 'Extremal action'):
            check_extremal_certificate(record)

    def test_shrunken_or_fabricated_transition_is_rejected(self):
        record = fixture()
        record['trace'][1]['theta'] = interval(Fraction(1017, 1000), Fraction(1018, 1000))
        with self.assertRaisesRegex(ValueError, 'independent image'):
            check_extremal_certificate(record)

    def test_no_comparison_after_unsafe_prefix(self):
        record = fixture()
        record['trace'].append(deepcopy(record['trace'][1]))
        record['trace'][2]['step'] = 2
        record['decisive_step'] = 2
        with self.assertRaisesRegex(ValueError, 'prefix'):
            check_extremal_certificate(record)

    def test_opposite_exit_is_not_a_nonviability_certificate(self):
        record = fixture()
        record['action'] = 2
        record['trace'][1]['theta'] = interval(Fraction(104, 100), Fraction(105, 100))
        record['trace'][1]['omega'] = interval(Fraction(9, 10), 1)
        with self.assertRaisesRegex(ValueError, 'correct direction'):
            check_extremal_certificate(record)
        record['status'] = 'opposite_exit'
        result = check_extremal_certificate(record)
        self.assertTrue(result['passed'])
        self.assertFalse(result['certified'])

    def test_boundary_touch_is_not_a_strict_crossing(self):
        record = fixture()
        record['trace'][1]['theta'] = interval(1, Fraction(102, 100))
        with self.assertRaisesRegex(ValueError, 'strict whole-interval'):
            check_extremal_certificate(record)
        record['status'] = 'boundary_unresolved'
        record['crossing_component'] = None
        self.assertFalse(check_extremal_certificate(record)['certified'])

    def test_incomplete_horizon_cannot_claim_safe_horizon(self):
        record = fixture()
        record['status'] = 'horizon_safe_extremal'
        record['decisive_step'] = record['crossing_component'] = None
        with self.assertRaisesRegex(ValueError, 'all 201'):
            check_extremal_certificate(record)

    def test_malformed_endpoints_and_indices_are_rejected(self):
        record = fixture()
        record['trace'][1]['theta']['lo'] = ['1', '0']
        with self.assertRaisesRegex(ValueError, 'positive'):
            check_extremal_certificate(record)
        record = fixture()
        record['trace'][1]['step'] = 2
        with self.assertRaisesRegex(ValueError, 'consecutive'):
            check_extremal_certificate(record)


if __name__ == '__main__':
    unittest.main()
