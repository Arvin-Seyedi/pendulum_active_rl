import unittest

import numpy as np

from src.acquisition_analysis import coverage_diagnostics, occupied_bins, seed_summary, tail_diagnostic
from src.active_query import collect_expert


class AnalysisTests(unittest.TestCase):
    def test_mean_above_p95_is_valid_heavy_tail(self):
        values = np.r_[np.zeros(95), np.ones(5)]
        result = tail_diagnostic(values)
        self.assertAlmostEqual(result["distribution"]["mean"], .05)
        self.assertAlmostEqual(result["largest_5_percent_share_of_distance_sum"], 1)
        self.assertEqual(result["mean_reconstructed_from_parts"], .05)

    def test_paired_seed_resampling(self):
        left = np.array([10., 20., 30.])
        right = np.array([9., 19., 29.])
        indices = np.array([[0, 0, 0], [0, 1, 2], [2, 2, 2]])
        result = seed_summary(left - right, indices)
        self.assertEqual(result["bootstrap_95_ci_mean"], [1, 1])
        self.assertEqual(result["sample_std"], 0)
        self.assertEqual(seed_summary(np.array([1., 2., 3.]), indices)["sample_std"], 1)

    def test_trajectory_weights_and_phase_counts(self):
        data = collect_expert(np.array([[.8, 2.], [0., 0.]]), np.arange(2), 202, 41)
        distances = np.r_[np.ones(2), np.zeros(200)]
        result = coverage_diagnostics(data, distances, np.zeros((1, 2)))
        self.assertEqual([x["length"] for x in result["trajectories"]], [2, 200])
        self.assertEqual(result["summary"]["trajectory_mean_distances"]["mean"], .5)
        self.assertAlmostEqual(result["summary"]["tail"]["distribution"]["mean"], 2 / 202)
        self.assertEqual(sum(x["count"] for x in result["by_step"]), 202)
        self.assertEqual(result["summary"]["distance_by_phase"]["after_20"]["count"], 180)
        self.assertEqual(len(occupied_bins([[0, 0], [.001, .001], [1, 3]], .1)), 2)


if __name__ == "__main__":
    unittest.main()
