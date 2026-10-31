"""E09 technical checks; real rollouts are restricted to the five pilot IDs."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src.counterfactual import assert_bitwise_equal
from src.expert import pd_controller
from src.recoverability import evaluate_controls
from src.recoverable_pd import (
    BENCHMARK_SHA256, GAINS, PILOT_IDS, case_identity, compare_archive,
    evaluate_gain, frozen_gain_action, load_benchmark,
)
from src.rollout_horizon import MeteredEnvironment

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'results/expert_safety_map/expert_safety_map.npz'


class LegacyRecorder:
    """Instrument the unchanged E01 function, preserving its native operations."""
    def __init__(self, seed):
        self.env = MeteredEnvironment(200)
        self.seed = seed
        self.actions, self.raw_states, self.rewards = [], [], []

    @property
    def unwrapped(self):
        return self.env.unwrapped

    def reset(self):
        return self.env.reset(seed=self.seed)

    def step(self, action):
        if not self.raw_states:
            self.raw_states.append(self.unwrapped.state.copy())
        self.actions.append(action.copy())
        result = self.env.step(action)
        self.raw_states.append(self.unwrapped.state.copy())
        self.rewards.append(result[1])
        return result


class FinalUnsafeEnvironment:
    """Synthetic final-successor violation, invisible to the old t0..199 check."""
    def __init__(self, horizon):
        self.unwrapped = self
        self.horizon = horizon
        self.t = self.resets = self.steps = 0

    def reset(self, seed):
        self.resets += 1
        self.t = 0

    def step(self, action):
        self.t += 1
        self.steps += 1
        self.state = np.array([2*np.pi if self.t == self.horizon else 0., 0.])
        return np.array([1., 0., 0.], dtype=np.float32), 0., False, self.t == self.horizon, {}

    def costs(self):
        return {'reset_attempts': np.array(self.resets), 'step_attempts': np.array(self.steps)}

    def close(self):
        pass


class RecoverablePDTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmark = load_benchmark(ARCHIVE)

    def test_exact_archive_grid_indexing_and_binary_identities(self):
        benchmark = self.benchmark
        self.assertEqual(benchmark['metadata']['sha256'], BENCHMARK_SHA256)
        self.assertEqual(benchmark['metadata']['archived_unsafe_count'], 3230)
        self.assertEqual(benchmark['metadata']['archived_safe_count'], 3331)
        self.assertFalse(benchmark['metadata']['original_trajectories_available'])
        for sid in PILOT_IDS:
            identity = case_identity(benchmark, sid)
            i, j = divmod(sid, 81)
            expected = np.array([benchmark['arrays']['theta_values'][j], benchmark['arrays']['omega_values'][i]])
            initial = np.array([float.fromhex(value) for value in identity['initial_hex']])
            self.assertEqual(initial.tobytes(), expected.tobytes())
            self.assertEqual(identity['initial_raw_bytes_hex'], expected.tobytes().hex())
            self.assertEqual((identity['omega_index'], identity['theta_index']), (i, j))
        self.assertEqual(case_identity(benchmark, 767)['initial_hex'][0], (-0.040000000000000036).hex())
        self.assertFalse(benchmark['arrays']['theta_values'].flags.writeable)
        self.assertFalse(benchmark['metadata']['theta_bitwise_reflection_symmetric'])
        for sid in (-1, 6561, True, 3.0):
            with self.assertRaises(ValueError): case_identity(benchmark, sid)

    def test_modified_archive_is_rejected_not_silently_regenerated(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'modified.npz'
            path.write_bytes(ARCHIVE.read_bytes()+b'changed')
            with self.assertRaises(ValueError): load_benchmark(path)

    def test_gain_bank_and_native_scalar_arithmetic(self):
        self.assertEqual(GAINS, tuple((p, d) for p in (6., 8., 12.) for d in (1., 2., 4.)))
        self.assertEqual(len(GAINS), 9)
        for scalar in (np.float64, np.float32):
            theta, omega = scalar(.123456789), scalar(-.23456789)
            for kp, kd in GAINS:
                value = frozen_gain_action(theta, omega, kp, kd)
                expected = pd_controller(theta, omega) if (kp, kd) == (8., 2.) else np.clip(-kp*theta-kd*omega, -2., 2.)
                self.assertEqual(np.asarray(value).dtype, np.asarray(expected).dtype)
                self.assertEqual(np.asarray(value).tobytes(), np.asarray(expected).tobytes())
        with self.assertRaises(ValueError): frozen_gain_action(0., 0., 10., 2.)
        self.assertTrue(np.signbit(frozen_gain_action(np.float64(0), np.float64(0), 8., 2.)))

    def test_unchanged_experiment01_function_bitwise_parity_all_five_cases(self):
        spec = importlib.util.spec_from_file_location('reviewed_e01', ROOT/'experiments/01_expert_map.py')
        original = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(original)
        for sid in PILOT_IDS:
            identity = case_identity(self.benchmark, sid)
            initial = np.array(identity['initial'])
            seed = 909100+sid
            arrays, costs, diagnostic = evaluate_gain(initial, seed, 8., 2.)
            recorder = LegacyRecorder(seed)
            try:
                old_margin, old_time = original.run_episode(recorder, *initial)
            finally:
                recorder.env.close()
            assert_bitwise_equal({'raw': arrays['raw_states'][0], 'actions': arrays['actions'][0], 'rewards': arrays['rewards'][0]},
                                 {'raw': np.array(recorder.raw_states), 'actions': np.array(recorder.actions), 'rewards': np.array(recorder.rewards)})
            self.assertEqual(diagnostic['legacy_minimum_margin'], old_margin)
            self.assertEqual(diagnostic['legacy_first_crossing_seconds'], None if np.isnan(old_time) else old_time)
            self.assertEqual(diagnostic['observation_dtypes'], ['float64']+['float32']*200)
            self.assertEqual(diagnostic['controller_input_dtypes'], [['float64']*2]+[['float32']*2]*199)
            self.assertEqual(diagnostic['command_dtypes'], ['float64']+['float32']*199)
            self.assertEqual(costs['step_attempts'], 200)
            self.assertEqual(costs['reset_attempts'], 1)
            compared = compare_archive(identity, arrays, diagnostic)
            self.assertEqual(compared['archived_unsafe'], identity['archived_unsafe'])
            self.assertFalse(compared['historical_trajectory_bitwise_reproduction_claimed'])

    def test_deterministic_gain_repeat_and_independent_recorded_action_replay(self):
        initial = np.array(case_identity(self.benchmark, 44)['initial'])
        first, _, diagnostic = evaluate_gain(initial, 909144, 6., 4.)
        repeated, _, repeated_diagnostic = evaluate_gain(initial, 909144, 6., 4.)
        replay, _ = evaluate_controls(initial, 909144, first['actions'][0, :, 0])
        assert_bitwise_equal(first, repeated)
        assert_bitwise_equal(first, replay)
        self.assertEqual(diagnostic, repeated_diagnostic)
        self.assertEqual(first['raw_states'].shape, (1, 201, 2))
        self.assertEqual(first['actions'].dtype, np.float32)
        self.assertLessEqual(np.abs(first['actions']).max(), 2.)

    def test_final_raw_violation_keeps_archive_and_legacy_labels_separate(self):
        identity = case_identity(self.benchmark, 3280)
        arrays, _, diagnostic = evaluate_gain(identity['initial'], 1, 8., 2., env_factory=FinalUnsafeEnvironment)
        compared = compare_archive(identity, arrays, diagnostic)
        self.assertFalse(compared['archived_unsafe'])
        self.assertFalse(compared['remeasured_legacy_unsafe'])
        self.assertTrue(compared['remeasured_raw_unsafe'])
        self.assertTrue(compared['archived_vs_raw_label_disagreement'])
        self.assertFalse(compared['archived_vs_legacy_label_disagreement'])
        self.assertEqual(diagnostic['raw_first_crossing_step'], 200)
        self.assertEqual(diagnostic['raw_first_crossing_seconds'], 10.)
        self.assertEqual(arrays['raw_states'].shape[1], 201)

    def test_archive_comparison_records_label_time_disagreement_without_repair(self):
        identity = case_identity(self.benchmark, 3280)
        arrays = {'initial_conditions': np.array([identity['initial']])}
        diagnostics = dict(baseline_original_expert=True, legacy_violated=True, raw_violated=True,
                           legacy_first_crossing_seconds=.1, raw_first_crossing_seconds=.1,
                           legacy_minimum_margin=-.01, raw_minimum_margin=-.01)
        comparison = compare_archive(identity, arrays, diagnostics)
        self.assertTrue(comparison['archived_vs_legacy_label_disagreement'])
        self.assertTrue(comparison['archived_vs_legacy_crossing_disagreement'])
        self.assertIsNone(comparison['legacy_crossing_time_difference'])
        self.assertFalse(identity['archived_unsafe'])
        diagnostics['baseline_original_expert'] = False
        with self.assertRaises(ValueError): compare_archive(identity, arrays, diagnostics)


if __name__ == '__main__':
    unittest.main()
