"""Authorization/configuration checks; no solver or simulator execution."""

import copy
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.horizon_experiment import read_json, sha256
from src.recoverability import solve_attempt
from src import recoverable_pd_solver as adapter


ROOT = Path(__file__).resolve().parents[1]


class RecoverablePDSolverTests(unittest.TestCase):
    def setUp(self):
        # Exact binary value displayed by the archived grid; this fixture keeps
        # authorization tests independent of uncommitted generated archives.
        self.initial = np.array([float.fromhex('0x1.47ae147ae1478p-4'), -2.0])
        self.folder = ROOT / adapter.OUTPUT_RELATIVE / 'state_0044/optimization/PD/original'
        self.request = {
            'state_id': 44, 'authorized_pilot_ids': [3280, 0, 6560, 767, 44],
            'start': 'PD', 'config_source_sha256': sha256(ROOT / adapter.CONFIG_SOURCE),
            'random_start_seed': 909004,
        }
        self.set_initial(self.request, self.initial)

    @staticmethod
    def set_initial(request, initial):
        request.update(initial=initial.tolist(),
                       initial_hex=[float(x).hex() for x in initial],
                       initial_raw_bytes_hex=initial.tobytes().hex())

    def test_scientific_configuration_and_solver_are_reused_without_override(self):
        before = sha256(ROOT / adapter.CONFIG_SOURCE)
        self.assertEqual(adapter._configuration(ROOT), read_json(ROOT / adapter.CONFIG_SOURCE))
        self.assertIs(adapter.solve_attempt, solve_attempt)
        self.assertEqual(adapter._configuration(ROOT)['seeds']['random_start'], 707004)
        self.assertEqual(adapter.RANDOM_START_SEED, 909004)
        self.assertEqual(sha256(ROOT / adapter.CONFIG_SOURCE), before)

    def test_request_checks_grid_identity_even_if_changed_bits_are_self_consistent(self):
        with patch.object(adapter, '_reference_initial', return_value=self.initial.copy()):
            actual = adapter._validate_request(ROOT, self.folder, self.request)
            self.assertEqual(actual.tobytes(), self.initial.tobytes())
            invalid_id = copy.deepcopy(self.request)
            invalid_id['state_id'] = 767
            with self.assertRaises(AssertionError):
                adapter._validate_request(ROOT, self.folder, invalid_id)
            altered = self.initial.copy()
            altered[0] = np.nextafter(altered[0], np.inf)
            invalid_bits = copy.deepcopy(self.request)
            self.set_initial(invalid_bits, altered)
            with self.assertRaises(AssertionError):
                adapter._validate_request(ROOT, self.folder, invalid_bits)

    def test_worker_rejects_wrong_scope_or_config_before_calling_solver(self):
        wrong_path = ROOT / adapter.OUTPUT_RELATIVE / 'state_0767/optimization/PD/original'
        invalid_config = copy.deepcopy(self.request)
        invalid_config['config_source_sha256'] = '0' * 64
        invalid_scope = copy.deepcopy(self.request)
        invalid_scope['authorized_pilot_ids'] = [44]
        with patch.object(adapter, 'solve_attempt') as solve:
            for folder, request in [(wrong_path, self.request),
                                    (self.folder, invalid_config),
                                    (self.folder, invalid_scope)]:
                with self.subTest(folder=folder, request=request):
                    with patch.object(adapter, 'read_json', return_value=request):
                        with self.assertRaises(AssertionError):
                            adapter.worker(ROOT, folder)
            solve.assert_not_called()


if __name__ == '__main__':
    unittest.main()
