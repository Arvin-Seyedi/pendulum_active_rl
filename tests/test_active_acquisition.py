"""Technical invariants; synthetic fixtures are not performance tuning."""

import unittest
from dataclasses import replace

import numpy as np
import torch

from src.active_query import (
    CollectionFailure, budget_prefix, collect_expert, nearest_d0_distance,
    rank_candidates, seed_streams, validate_collection,
)
from src.ensemble import EnsembleConfig, configure_determinism, fit_member, member_seeds, predict_ensemble
from src.policy_evaluation import evaluate_rollouts
from src.ragged_ensemble import episode_transition_indices, fit_ragged_member
from src.safety import safety_margin


class TechnicalTests(unittest.TestCase):
    def setUp(self):
        configure_determinism(41)

    def test_rectangular_equivalence(self):
        rng = np.random.default_rng(41)
        x = rng.normal(size=(12, 4, 2))
        y = np.clip(-8 * x[:, :, :1] - 2 * x[:, :, 1:], -2, 2)
        cfg = replace(EnsembleConfig(), max_epochs=4, seed=41)
        seeds = member_seeds(cfg)[0]
        old, old_record = fit_member(x, y, cfg, seeds)
        new, new_record = fit_ragged_member(x.reshape(-1, 2), y.reshape(-1, 1), np.arange(13) * 4, cfg, seeds)
        for key in old.state_dict():
            self.assertTrue(torch.equal(old.state_dict()[key], new.state_dict()[key]))
        for key, value in old_record.items():
            self.assertEqual(value, new_record[key])

    def test_ragged_bootstrap_multiplicity(self):
        offsets = np.array([0, 2, 7, 8])
        self.assertEqual(episode_transition_indices(offsets, [1, 0, 1]).tolist(),
                         [2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6])

    def test_safety_stop_and_prefix(self):
        pool = np.array([[.8, 2.], [0., 0.], [-.8, -2.], [.01, 0.]])
        data = collect_expert(pool, np.arange(4), 205, 41)
        validate_collection(data, 205)
        self.assertEqual(data["stop_reasons"][0], "safety")
        self.assertLess(data["episode_offsets"][1], 200)
        self.assertTrue(data["safety_stop"][data["episode_offsets"][1] - 1])
        for budget in (1, 2, 17, 202, 205):
            prefix = budget_prefix(data, budget)
            independent = collect_expert(pool, np.arange(4), budget, 41)
            validate_collection(prefix, budget)
            for key in prefix:
                np.testing.assert_array_equal(prefix[key], independent[key], err_msg=key)

    def test_exhaustion_retains_data(self):
        with self.assertRaises(CollectionFailure) as context:
            collect_expert(np.array([[.8, 2.]]), np.array([0]), 200, 41)
        failure = context.exception
        self.assertIn("exhausted", str(failure))
        self.assertGreater(len(failure.partial["states"]), 0)
        self.assertEqual(failure.attempts, len(failure.partial["states"]))

    def test_nonfinite_failure_no_replacement(self):
        import gymnasium as gym
        class BrokenEnv(gym.Wrapper):
            def step(self, action):
                result = list(self.env.step(action))
                if self.env._elapsed_steps == 2:
                    result[1] = float("nan")
                return tuple(result)
        with self.assertRaises(CollectionFailure) as context:
            collect_expert(np.zeros((3, 2)), np.arange(3), 100, 41,
                           env_factory=lambda: BrokenEnv(gym.make("Pendulum-v1")))
        failure = context.exception
        self.assertEqual(len(failure.partial["states"]), 1)
        self.assertEqual(failure.attempts, 2)
        self.assertEqual(failure.partial["stop_reasons"].tolist(), ["technical_failure"])
        self.assertEqual(failure.partial["candidate_ids"].tolist(), [0])

    def test_raw_safety_and_full_horizon_evaluation(self):
        data, metrics = evaluate_rollouts(None, np.array([[.8, 2.], [0., 0.]]))
        self.assertEqual(data["raw_states"].shape, (2, 201, 2))
        self.assertLess(data["margins"][0].min(), 0)
        np.testing.assert_array_equal(data["margins"].ravel(), [safety_margin(*s) for s in data["raw_states"].reshape(-1, 2)])
        self.assertTrue(np.any(np.abs(data["raw_states"] - data["states"]) > 3))
        self.assertEqual(metrics["violating_episodes"], 1)
        repeated, _ = evaluate_rollouts(None, data["initial_conditions"])
        for key in data:
            np.testing.assert_array_equal(data[key], repeated[key])

    def test_distance_and_bounded_mean(self):
        np.testing.assert_array_equal(nearest_d0_distance([[1, 3], [0, 3]], [[0, 0]]), [np.sqrt(2), 1])
        from src.ensemble import PolicyNetwork
        models = [PolicyNetwork() for _ in range(5)]
        for model, torque in zip(models, [-8, 4, 4, 4, 4]):
            for parameter in model.parameters():
                parameter.data.zero_()
            model.network[-1].bias.data.fill_(torque / 2)
        prediction = predict_ensemble(models, np.zeros((10, 2)))
        np.testing.assert_array_equal(prediction["bounded_mean"], np.full(10, 1.2))
        self.assertNotEqual(prediction["bounded_mean"][0], np.clip(prediction["raw_mean"][0], -2, 2))
        rankings, _ = rank_candidates(np.zeros((10, 2)), models, np.zeros((1, 2)), seed_streams(41))
        rankings2, _ = rank_candidates(np.zeros((10, 2)), models, np.zeros((1, 2)), seed_streams(41))
        for name in rankings:
            np.testing.assert_array_equal(rankings[name], rankings2[name])
            self.assertEqual(len(np.unique(rankings[name])), 10)


if __name__ == "__main__":
    unittest.main()
