"""Finalization rejects archive writes and unjustified historical exceptions."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch
from hashlib import sha256
from src.drilling_static_verification import verification_destination,preserved_project_file


class StaticFinalizationTests(TestCase):
    def test_archive_output_rejected(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp)
            with self.assertRaises(ValueError):verification_destination(p/'accepted/new.json',p/'accepted')

    def test_existing_output_rejected(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'old.json').write_text('preserve')
            with self.assertRaises(FileExistsError):verification_destination(p/'old.json',p/'accepted')

    def test_historical_exception_cannot_cover_source_changes(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'source.py').write_text('changed')
            with self.assertRaises(AssertionError):preserved_project_file(p,'source.py','invalid')

    def test_context_exception_requires_exact_historical_bytes(self):
        with TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'CURRENT_STATUS.md').write_text('reviewed update')
            h=sha256(b'old context').hexdigest()
            with patch('src.drilling_static_verification.subprocess.check_output',return_value=b'wrong context'):
                with self.assertRaises(AssertionError):preserved_project_file(p,'CURRENT_STATUS.md',h)
            with patch('src.drilling_static_verification.subprocess.check_output',return_value=b'old context'):
                preserved_project_file(p,'CURRENT_STATUS.md',h)
