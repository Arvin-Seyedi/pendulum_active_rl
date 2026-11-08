"""Finalization permission boundaries; no optimizer or plant calls."""
from pathlib import Path
import hashlib
import tempfile
import unittest
from unittest.mock import patch
from src.quadrotor_finalization import destination,historical_context,preserved_path,IQL


class FinalizationTests(unittest.TestCase):
    def test_accepted_archive_cannot_be_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            with self.assertRaises(ValueError):destination(root/'results/benchmark_selection/mmaq_v1/new.json',root)
            self.assertEqual(destination(root/'results/benchmark_selection/finalization/new.json',root),(root/'results/benchmark_selection/finalization/new.json').resolve())

    def test_existing_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);f=root/'old.json';f.write_text('keep')
            with self.assertRaises(FileExistsError):destination(f,root)

    def test_context_exception_requires_exact_historical_git_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'CURRENT_STATUS.md').write_text('approved new text')
            expected=hashlib.sha256(b'old verified text').hexdigest()
            with patch('src.quadrotor_finalization.subprocess.check_output',return_value=b'wrong'):
                with self.assertRaises(AssertionError):historical_context(root,root,'CURRENT_STATUS.md',expected,'old')
            with patch('src.quadrotor_finalization.subprocess.check_output',return_value=b'old verified text'):
                self.assertTrue(historical_context(root,root,'CURRENT_STATUS.md',expected,'old'))

    def test_scientific_source_cannot_use_context_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'model.py').write_text('changed')
            with self.assertRaises(AssertionError):historical_context(root,root,'model.py','wrong','old')

    def test_IQL_is_a_preservation_target_not_export_dependency(self):
        self.assertEqual(len(IQL),9)
        self.assertEqual(preserved_path('/original','/export','src/iql.py'),Path('/original/src/iql.py'))
        self.assertEqual(preserved_path('/original','/export','src/quadrotor_pilot.py'),Path('/export/src/quadrotor_pilot.py'))

    def test_data_records_use_explicit_data_root(self):
        self.assertEqual(preserved_path('/original','/export','data/pendulum_D0.npz'),Path('/original/data/pendulum_D0.npz'))
        self.assertEqual(preserved_path('/original','/export','results/old/record.json'),Path('/original/results/old/record.json'))
