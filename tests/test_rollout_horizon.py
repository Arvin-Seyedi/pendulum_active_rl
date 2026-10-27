"""Technical fixtures only; no model or acquisition performance tuning."""

import unittest

import numpy as np

from src.active_query import budget_prefix, collect_expert
from src.rollout_horizon import (
    collect_metered, expert_safety_audit, operational_summary,
    selection_diagnostics, shared_prefix_check, summarize_safety_audit,
    validate_horizon_collection,
)

# Observed first crossing at step 22 in the preserved Experiment 04 pilot.
# Used only to test that a safe 20-step cap is not misclassified as 200-step safe.
LATE_CROSSING = [-0.006213953515803361, 1.828041092911897]


class HorizonTests(unittest.TestCase):
    def test_primary_and_interaction_difference_directions(self):
        from src.horizon_experiment import condition_name, paired_contrasts
        metrics = {'D0': {'wide': {'mean_return': -10., 'safety_violation_rate': .5}}}
        for strategy, short, long in [('uncertainty', -4., -9.), ('random', -6., -8.), ('distance', -3., -5.)]:
            for horizon, value in [(20, short), (200, long)]:
                metrics[condition_name(strategy, horizon, 5000)] = {'wide': {'mean_return': value, 'safety_violation_rate': .4 if horizon == 20 else .5}}
        contrasts = paired_contrasts(metrics, 5000)
        primary = contrasts['uncertainty_H20_B5000_minus_uncertainty_H200_B5000']
        self.assertEqual(primary['wide.mean_return'], 5.)
        self.assertAlmostEqual(primary['wide.safety_violation_rate'], -.1)
        self.assertEqual(contrasts['uncertainty_horizon_effect_minus_random_horizon_effect']['wide.mean_return'], 3.)

    def test_exact_prefixes_and_shared_physical_trajectories(self):
        pool = np.array([[.8, 2.], LATE_CROSSING, *[[0., 0.]] * 20])
        streams = {}
        for horizon in (20, 200):
            stream, costs = collect_metered(pool, np.arange(len(pool)), 205, 41, horizon)
            streams[horizon] = stream
            self.assertEqual(int(costs['step_attempts']), 205)
            self.assertEqual(len(costs['reset_seconds']), len(stream['initial_conditions']))
            for budget in (1, 2, 19, 20, 21, 22, 200, 205):
                prefix = budget_prefix(stream, budget)
                validate_horizon_collection(prefix, budget, horizon)
                independent = collect_expert(pool, np.arange(len(pool)), budget, 41, horizon)
                for key in prefix:
                    np.testing.assert_array_equal(prefix[key], independent[key])
            measured = operational_summary(stream, costs, horizon)
            self.assertEqual(measured['transition_equivalent_cost']['20'],
                             205 + 20 * len(stream['initial_conditions']))
        result = shared_prefix_check(streams[20], streams[200])
        self.assertTrue(result['passed'])
        self.assertEqual(result['shared_candidates'], 3)

    def test_cap_at_twenty_is_not_a_terminal_or_safety_violation(self):
        data = collect_expert(np.zeros((2, 2)), np.arange(2), 25, 41, 20)
        self.assertEqual(data['stop_reasons'].tolist(), ['time_limit', 'budget'])
        self.assertFalse(data['terminated'].any())
        self.assertTrue(data['truncated'][19])
        self.assertFalse(data['truncated'][24])
        self.assertTrue(data['budget_cutoff'][24])
        self.assertFalse(data['safety_stop'].any())
        validate_horizon_collection(data, 25, 20)

    def test_audit_detects_late_crossing_and_keeps_training_unchanged(self):
        pool = np.array([LATE_CROSSING, [0., 0.]])
        data = collect_expert(pool, np.arange(2), 40, 41, 20)
        before = {k: v.copy() for k, v in data.items()}
        audit, costs = expert_safety_audit(pool, [1, 0, 0], 41)
        self.assertEqual(audit['audited_candidate_ids'].tolist(), [0, 1])
        self.assertEqual(audit['first_violation_step'].tolist(), [22, -1])
        self.assertEqual(int(costs['step_attempts']), 222)
        summary = summarize_safety_audit(data, audit, 20)
        self.assertEqual(summary['safely_capped_at_20_count'], 2)
        self.assertEqual(summary['subsequent_violation_steps_21_to_200_count'], 1)
        self.assertEqual(summary['subsequent_violation_fraction_among_safely_capped_20'], .5)
        for key in data:
            np.testing.assert_array_equal(data[key], before[key])
        repeated, _ = expert_safety_audit(pool, [0, 1], 41)
        for key in audit:
            np.testing.assert_array_equal(audit[key], repeated[key])

    def test_budget_censored_episode_is_not_counted_as_safe_twenty_step_cap(self):
        pool = np.array([LATE_CROSSING])
        data = collect_expert(pool, [0], 5, 41, 20)
        audit, _ = expert_safety_audit(pool, [0], 41)
        summary = summarize_safety_audit(data, audit, 20)
        self.assertEqual(summary['safely_capped_at_20_count'], 0)
        self.assertIsNone(summary['subsequent_violation_fraction_among_safely_capped_20'])

    def test_selection_scores_are_initial_condition_weighted(self):
        pool = np.array([[.8, 2.], [0., 0.], [.01, 0.], [.02, 0.]])
        rankings = {'uncertainty': np.arange(4), 'distance': np.array([3, 2, 1, 0]),
                    'random': np.array([2, 0, 3, 1])}
        scores = {'uncertainty': np.array([.5, .1, 0., 0.]),
                  'raw_uncertainty': np.array([1., .3, .2, .2]),
                  'support_distance': np.array([.1, .2, .3, .4])}
        data = collect_expert(pool, rankings['uncertainty'], 23, 41, 20)
        arrays, summary = selection_diagnostics(data, rankings, scores, 'uncertainty')
        self.assertEqual(arrays['candidate_ids'].tolist(), [0, 1, 2])
        self.assertEqual(arrays['episode_lengths'].tolist(), [2, 20, 1])
        self.assertEqual(summary['fraction_zero_bounded_disagreement'], 1 / 3)
        self.assertEqual(summary['first_zero_disagreement_query_1based'], 3)
        self.assertEqual(summary['fraction_positive_pool_queried'], 1)
        self.assertEqual(arrays['distance_rank_1based'].tolist(), [4, 3, 2])
        random_data = collect_expert(pool, rankings['random'], 20, 41, 20)
        _, random_summary = selection_diagnostics(random_data, rankings, scores, 'random')
        self.assertIsNone(random_summary['selection_score'])
        self.assertEqual(random_summary['fraction_zero_bounded_disagreement'], 1)


if __name__ == '__main__':
    unittest.main()
