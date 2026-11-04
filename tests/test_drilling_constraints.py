"""Small exact mechanical fixtures; no plant trajectories or controllers."""
from fractions import Fraction as F
from pathlib import Path
import json
import unittest
from src.drilling_constraint_math import matrix,constrained_systems,record,exact_solve,ldl_positive
from src.drilling_constraint_verification import bareiss_determinant,directed,fraction_ldl


def fixture(p=None):
    convert=lambda x:matrix(x,p)
    return constrained_systems(convert([[2,0],[0,3]]),convert([[1,0],[0,1]]),convert([[1,0],[0,1]]),
        convert([[1],[1]]),convert([[1,0]]),convert([[-1,1]]),convert([[4],[0]]))


class ConstraintAuditTests(unittest.TestCase):
    def test_exact_reaction_sign_and_retained_effect(self):
        r=fixture()
        self.assertEqual(r['native_retained_acceleration'][0][0],2)
        self.assertEqual(r['virtual_work_acceleration'][0][0],F(4,5))
        self.assertEqual(r['lambda'][0][0],F(12,5))
        self.assertEqual(r['retained_reaction_acceleration'][0][0],F(-6,5))
        self.assertEqual(r['native_constraint_acceleration'][0][0],-2)
        self.assertTrue(all(x['exact_zero'] for x in r['identity_certificates'].values()))

    def test_block_and_virtual_work_agreement_at_both_precisions(self):
        for p in (128,256):
            r=fixture(p)
            self.assertTrue(all(x['contains_zero'] for x in r['identity_certificates'].values()))
            x=r['virtual_work_acceleration'][0][0]
            self.assertLessEqual(F(*x.lo.as_integer_ratio()),F(4,5))
            self.assertGreaterEqual(F(*x.hi.as_integer_ratio()),F(4,5))

    def test_exact_block_uniqueness_independently(self):
        r=fixture();a=[[F(x) for x in row] for row in r['block_matrix']]
        self.assertEqual(bareiss_determinant(a),5)

    def test_bareiss_handles_swap_and_singularity(self):
        self.assertEqual(bareiss_determinant([[F(0),F(1)],[F(2),F(3)]]),-2)
        self.assertEqual(bareiss_determinant([[F(1),F(2)],[F(2),F(4)]]),0)

    def test_singular_exact_solve_fails_closed(self):
        with self.assertRaises(ArithmeticError):exact_solve(matrix([[1,2],[2,4]]),matrix([[1],[1]]))

    def test_fraction_directed_rounding_both_signs(self):
        for p in (128,256):
            for x in [F(1,3),F(-1,7),F(2**300+1,2**200)]:
                self.assertLessEqual(directed(x,p,False),x)
                self.assertGreaterEqual(directed(x,p,True),x)

    def test_independent_positive_mass_certificate(self):
        for p in (128,256):
            a=matrix([[2,1],[1,3]],p)
            self.assertEqual(ldl_positive(a)['status'],'SUPPORTED')
            self.assertTrue(fraction_ldl(record(a),p)['positive'])

    def test_indefinite_matrix_is_not_certified_positive(self):
        a=matrix([[1,2],[2,1]],128)
        self.assertEqual(ldl_positive(a)['status'],'UNRESOLVED')
        self.assertFalse(fraction_ldl(record(a),128)['positive'])

    def test_arithmetic_replay_is_deterministic(self):
        self.assertEqual(record(fixture()['lambda']),record(fixture()['lambda']))
        for p in (128,256):self.assertEqual(record(fixture(p)['lambda']),record(fixture(p)['lambda']))

    def test_only_four_approved_snapshots(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/'configs/experiment_11c.json').read_text())
        self.assertEqual([(x['case'],x['index']) for x in cfg['snapshots']],
                         [('P5',0),('P5',200000),('P0',1000),('P3',1000)])
        self.assertEqual(cfg['precisions_bits'],[128,256])
        self.assertEqual((cfg['workers'],cfg['cpu_seconds_cap'],cfg['rss_bytes_cap']),(1,1800,2**31))


if __name__=='__main__':unittest.main()
