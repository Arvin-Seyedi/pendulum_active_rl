"""Analytic/synthetic Experiment 06A checks: no training or simulator rollouts."""

from pathlib import Path
import unittest

import numpy as np

from src.counterfactual import (
    assert_bitwise_equal, classify_alternatives, compare_PD_reference, episode_metrics,
    evaluate_first_actions, fixed_state_menu, state_bootstrap,
)
from src.expert import pd_controller
from src.horizon_experiment import read_json

CONFIG = read_json(Path(__file__).resolve().parents[1] / 'configs/experiment_06a.json')


class SyntheticEnvironment:
    """Known unsafe raw successor with canonical observation; metered fake."""
    def __init__(self, horizon):
        self.unwrapped = self
        self.horizon = horizon
        self.resets = self.steps = self.t = 0

    def reset(self, seed):
        self.t = 0
        self.resets += 1

    def step(self, action):
        self.t += 1
        self.steps += 1
        self.state = np.array([2 * np.pi, 0.])
        return np.array([1., 0., 0.]), -float(action[0]) ** 2, False, self.t == self.horizon, {}

    def costs(self):
        return {'reset_attempts': self.resets, 'step_attempts': self.steps}

    def close(self):
        pass


class CounterfactualTests(unittest.TestCase):
    def test_sampling_fixed_strata_no_outcome_selection(self):
        a, b = fixed_state_menu(CONFIG), fixed_state_menu(CONFIG)
        assert_bitwise_equal(a, b)
        states = a['initial_conditions']
        self.assertEqual(states.shape, (97, 2))
        self.assertTrue(np.all(np.abs(states[:32]) <= [.25, .5]))
        self.assertTrue(np.all(np.abs(states[32:96]) <= [.8, 2]))
        self.assertFalse(np.any(np.all(np.abs(states[32:96]) <= [.25, .5], axis=1)))
        np.testing.assert_array_equal(states[96], [0., 0.])

    def test_menu_offsets_after_PD_clipping_and_exact_execution_identity(self):
        menu = fixed_state_menu(CONFIG)
        for i in range(97):
            ids = menu['proposal_run_ids'][i]
            np.testing.assert_array_equal(menu['first_actions'][ids], menu['executed_proposals'][i])
            self.assertEqual(len(set(ids)), menu['distinct_action_counts'][i])
            baseline = menu['baseline_run_ids'][i]
            self.assertEqual(menu['first_actions'][baseline], np.float32(pd_controller(*menu['canonical_initial_states'][i])))
            if menu['saturated'][i]:
                self.assertEqual(len(set(ids)), 3)
                self.assertEqual(len(np.unique(menu['first_actions'][ids])), 3)
        self.assertEqual(menu['distinct_action_counts'][-1], 5)

    def test_raw_safety_full_horizon_and_only_first_action_override(self):
        arrays, costs = evaluate_first_actions([[.1, .2], [.1, .2]], [.4, -.4], [8, 8], 3, SyntheticEnvironment)
        self.assertEqual(costs['step_attempts'], 6)
        self.assertEqual(costs['reset_attempts'], 2)
        np.testing.assert_array_equal(arrays['actions'][:, 1:], 0.)
        self.assertTrue(np.all(arrays['margins'][:, 1:] < 0))
        self.assertTrue(np.all(arrays['states'][:, 1:, 0] == 0))
        self.assertTrue(np.all(arrays['raw_states'][:, 1:, 0] == 2 * np.pi))
        self.assertTrue(arrays['truncated'][:, -1].all())

    def test_rewards_discount_crossing_and_censoring(self):
        a = {'rewards': np.array([[-1., -2., -3.], [0., 0., 0.]]),
             'margins': np.array([[.5, .1, -.01, .1], [1., 1., 1., 1.]])}
        m = episode_metrics(a, .5, .05)
        np.testing.assert_array_equal(m['returns'], [-6., 0.])
        np.testing.assert_array_equal(m['discounted_returns'], [-2.75, 0.])
        np.testing.assert_array_equal(m['first_crossing_step'], [2, -1])
        np.testing.assert_array_equal(m['first_crossing_seconds'], [.1, -1])
        np.testing.assert_array_equal(m['unsafe_state_fraction'], [.25, 0.])

    def test_safety_categories_and_joint_threshold_are_not_conflated(self):
        pd_bad = np.array([False, True, True, False, False])
        alt_bad = np.array([False, False, True, True, False])
        flags = classify_alternatives(pd_bad, alt_bad, np.array([.2, .2, .2, .2, .1]),
                                      np.array([.2, .2, .2, .2, .2]), .1)
        np.testing.assert_array_equal(flags['safe_improvement'], [True, True, False, False, False])
        np.testing.assert_array_equal(flags['safety_rescue'], [False, True, False, False, False])
        np.testing.assert_array_equal(flags['return_improvement_without_rescue'], [False, False, True, False, False])
        np.testing.assert_array_equal(flags['introduced_violation'], [False, False, False, True, False])
        self.assertTrue(flags['discounted_only_improvement'][-1])

    def test_state_bootstrap_preserves_stratum_composition(self):
        strata = np.array(['nominal', 'challenging', 'control'])
        interval = state_bootstrap([1., 0.], [0, 1], strata, 606204, 10000)
        self.assertEqual(interval, [.5, .5])
        self.assertIsNone(state_bootstrap([], [], strata, 606204, 10000))

    def test_bitwise_check_detects_signed_zero_and_dtype_change(self):
        with self.assertRaises(AssertionError):
            assert_bitwise_equal({'x': np.array([0.])}, {'x': np.array([-0.])})
        with self.assertRaises(AssertionError):
            assert_bitwise_equal({'x': np.array([0.])}, {'x': np.array([0.], dtype=np.float32)})

    def test_PD_reference_exception_is_only_first_action_signed_zero(self):
        a = {'actions': np.zeros((1, 2, 1), dtype=np.float32)}
        b = {'actions': a['actions'].copy()}
        b['actions'][0, 0, 0] = -0.
        self.assertEqual(compare_PD_reference(a, b), [[0, 0, 0]])
        b['actions'][0, 1, 0] = -0.
        with self.assertRaises(AssertionError):
            compare_PD_reference(a, b)
        b['actions'][0, 1, 0] = np.nextafter(np.float32(0), np.float32(1))
        with self.assertRaises(AssertionError):
            compare_PD_reference(a, b)


if __name__ == '__main__':
    unittest.main()
