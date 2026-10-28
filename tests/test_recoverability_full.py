"""Full-study reporting checks; no solver calls, simulation, or training."""

import unittest
from src.recoverability_full_analysis import choose_oracle, describe


class FullStudyTests(unittest.TestCase):
    def test_oracle_prioritizes_margin_and_uses_declared_ties(self):
        candidates=[{'candidate_available':True,'minimum_margins':-.1,'returns':10.,'start_index':0},
                    {'candidate_available':True,'minimum_margins':.1,'returns':-5.,'start_index':1},
                    {'candidate_available':True,'minimum_margins':.1,'returns':-4.,'start_index':2}]
        self.assertIs(choose_oracle(candidates),candidates[2])
        candidates[1]['returns']=-4.
        self.assertIs(choose_oracle(candidates),candidates[1])

    def test_missing_candidates_are_not_fabricated_into_safe_results(self):
        self.assertIsNone(choose_oracle([{'candidate_available':False}]))
        self.assertIsNone(describe([])['mean'])

    def test_summary_uses_sample_sd_and_preserves_negative_outcomes(self):
        d=describe([-3.,-1.])
        self.assertEqual(d['mean'],-2.)
        self.assertEqual(d['min'],-3.)
        self.assertAlmostEqual(d['std'],2**.5)


if __name__=='__main__':unittest.main()
