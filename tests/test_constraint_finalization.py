"""Finalization distinguishes coordinate enforcement and mechanical balance."""
from pathlib import Path
import tempfile
import hashlib
import unittest
from unittest.mock import patch
from src.drilling_constraint_finalization import geometric_balance,historical_bytes
from src.drilling_constraint_verification import decode
from src.drilling_constraint_math import record
from src.drilling_static_verification import verification_destination
from tests.test_drilling_constraints import fixture


class ConstraintFinalizationTests(unittest.TestCase):
    def test_geometry_can_hold_while_required_force_has_virtual_work(self):
        raw=fixture();data={k:decode(record(raw[k])) for k in
            ('M','T','C','M_H','W','native_retained_acceleration','virtual_work_acceleration')}
        result=geometric_balance(data)
        self.assertTrue(result['geometric_enforcement_exact'])
        self.assertEqual(result['nonzero_required_virtual_work_columns'],[0])

    def test_existing_or_accepted_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'existing.json').write_text('preserve')
            with self.assertRaises(FileExistsError):verification_destination(root/'existing.json',root/'accepted')
            with self.assertRaises(ValueError):verification_destination(root/'accepted/new.json',root/'accepted')

    def test_source_changes_cannot_use_context_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'model.py').write_text('changed')
            with self.assertRaises(AssertionError):historical_bytes(p,'model.py','wrong')

    def test_context_exception_requires_exact_historical_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'CURRENT_STATUS.md').write_text('reviewed update')
            expected=hashlib.sha256(b'accepted old context').hexdigest()
            with patch('src.drilling_constraint_finalization.subprocess.check_output',return_value=b'wrong'):
                with self.assertRaises(AssertionError):historical_bytes(p,'CURRENT_STATUS.md',expected)
            with patch('src.drilling_constraint_finalization.subprocess.check_output',return_value=b'accepted old context'):
                self.assertEqual(historical_bytes(p,'CURRENT_STATUS.md',expected),b'accepted old context')
