"""Full-study integrity and inference tests; no performance-based choices."""

from pathlib import Path
import tempfile
import unittest
import numpy as np

from src.horizon_experiment import read_json, write_json
from src.horizon_study import completed, exclusive, frozen_npz, manifest, run_seed
from src.horizon_study_analysis import summarize_records


class StudyIntegrityTests(unittest.TestCase):
    def test_completed_artifacts_are_read_only_and_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            frozen_npz(folder / 'data.npz', {'x': np.arange(5)})
            write_json(folder / 'checks.json', {'all_passed': True})
            write_json(folder / 'artifact_hashes.json', manifest(folder))
            before = {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in folder.iterdir()}
            self.assertTrue(completed(folder))
            self.assertEqual(before, {p.name: (p.stat().st_mtime_ns, p.read_bytes()) for p in folder.iterdir()})
            (folder / 'data.npz').write_bytes(b'corrupt')
            with self.assertRaises(AssertionError): completed(folder)

    def test_partial_completion_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            write_json(folder / 'checks.json', {'all_passed': True})
            self.assertFalse(completed(folder))

    def test_frozen_arrays_refuse_changes_without_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'arrays.npz'
            frozen_npz(path, {'x': np.arange(4)})
            before = path.read_bytes()
            with self.assertRaises(AssertionError): frozen_npz(path, {'x': np.arange(5)})
            self.assertEqual(path.read_bytes(), before)

    def test_exclusive_process_lock_and_pilot_rejection(self):
        with tempfile.TemporaryDirectory() as tmp:
            with exclusive(Path(tmp)):
                with self.assertRaises(BlockingIOError):
                    with exclusive(Path(tmp)): pass
            with self.assertRaises(AssertionError): run_seed(Path(tmp), Path(tmp), 41)

    def test_seed_inference_preserves_pairing_and_missing_strata(self):
        draws = np.array([[0, 1, 2], [0, 0, 0], [2, 2, 2]])
        records = [{'effect': {'return': v}, 'conditional': None if i == 0 else 7}
                   for i, v in enumerate([1., 3., 5.])]
        result = summarize_records(records, draws)
        self.assertEqual(result['effect.return']['per_seed'], [1., 3., 5.])
        self.assertEqual(result['effect.return']['mean'], 3.)
        self.assertEqual(result['effect.return']['sample_std'], 2.)
        self.assertNotIn('conditional', result)


if __name__ == '__main__':
    unittest.main()
