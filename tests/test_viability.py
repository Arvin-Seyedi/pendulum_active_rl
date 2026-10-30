"""Directed-rounding fixtures; no cohort execution or simulator calls."""

from fractions import Fraction
import unittest

import gmpy2 as g

from src.viability import MODEL, diagnostics, extremal_trajectory, interval_step, rational


class DirectedIntervalTests(unittest.TestCase):
    def test_rational_parameters_and_exact_binary_initials(self):
        self.assertEqual(MODEL['dt'], '1/20')
        initial = [0.1.hex(), (-0.0).hex()]
        r = extremal_trajectory(1, initial, -2)
        self.assertEqual(r['initial_hex'], initial)
        self.assertEqual(rational(r['trace'][0]['theta']['lo']), Fraction.from_float(.1))
        self.assertNotEqual(Fraction.from_float(.1), Fraction(1, 10))

    def test_directional_rounding_encloses_nonbinary_rational(self):
        for p in (128, 256, 512):
            with g.context(precision=p):
                zero = g.mpfr(0)
            box = interval_step(((zero, zero), (zero, zero)), -2, p)
            for pair, exact in zip(box, [Fraction(-3, 200), Fraction(-3, 10)]):
                self.assertLess(Fraction(*pair[0].as_integer_ratio()), exact)
                self.assertGreater(Fraction(*pair[1].as_integer_ratio()), exact)

    def test_upper_and_lower_boundary_certificates(self):
        upper = extremal_trajectory(1, [1.0.hex(), 0.0.hex()], -2)
        lower = extremal_trajectory(2, [(-1.0).hex(), 0.0.hex()], 2)
        for r in (upper, lower):
            self.assertEqual(r['status'], 'certified_nonviable')
            self.assertEqual(r['decisive_step'], 1)
            self.assertEqual(r['crossing_component'], 'theta')
            self.assertGreater(diagnostics(r)['crossing_separation_lower_approx'], 0)
        for a, b in zip(upper['trace'], lower['trace']):
            for name in ('theta', 'omega'):
                self.assertEqual(rational(a[name]['lo']), -rational(b[name]['hi']))
                self.assertEqual(rational(a[name]['hi']), -rational(b[name]['lo']))

    def test_equilibrium_extrema_exit_wrong_side_and_do_not_certify(self):
        for action in (-2, 2):
            r = extremal_trajectory(96, [0.0.hex(), 0.0.hex()], action)
            self.assertEqual(r['status'], 'opposite_exit')
            self.assertEqual(r['trace'][-1]['step'], r['decisive_step'])

    def test_no_propagation_outside_proof_domain(self):
        with self.assertRaises(ValueError):
            interval_step(((g.mpfr(1), g.mpfr('1.01')), (g.mpfr(0), g.mpfr(0))), -2, 128)
        with self.assertRaises(ValueError):
            extremal_trajectory(1, [0.0.hex(), 0.0.hex()], 0)
        with self.assertRaises(ValueError):
            extremal_trajectory(1, [0.0.hex(), 0.0.hex()], 2, 53)

    def test_precision_refinement_and_determinism(self):
        args = (1, [1.0.hex(), 0.0.hex()], -2)
        a = extremal_trajectory(*args, precision=128)
        self.assertEqual(a, extremal_trajectory(*args, precision=128))
        b = extremal_trajectory(*args, precision=256)
        for name in ('theta', 'omega'):
            self.assertGreaterEqual(rational(b['trace'][1][name]['lo']), rational(a['trace'][1][name]['lo']))
            self.assertLessEqual(rational(b['trace'][1][name]['hi']), rational(a['trace'][1][name]['hi']))

    def test_independent_rational_checker_accepts_directed_rounding(self):
        from src.viability_checker import check_extremal_certificate
        for p in (128, 256, 512):
            for angle, action in [(1., -2), (-1., 2), (0., -2), (0., 2)]:
                initial = [angle.hex(), 0.0.hex()]
                record = extremal_trajectory(99, initial, action, p)
                checked = check_extremal_certificate(record, 99, initial)
                self.assertTrue(checked['passed'])
                self.assertEqual(checked['certified'], angle != 0)


if __name__ == '__main__':
    unittest.main()
