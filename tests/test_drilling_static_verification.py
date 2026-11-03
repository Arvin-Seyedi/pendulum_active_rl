"""Fail-closed evidence tests independent of the interval producer."""
from pathlib import Path
import tempfile
import unittest
from src.drilling_reproduction import write,sha
from src.drilling_static_verification import check_worker_manifest


class StaticEvidenceTests(unittest.TestCase):
    def test_changed_scientific_evidence_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'exact.json').write_text('original')
            write(p/'completion.json',{'artifact_sha256':{'exact.json':sha(p/'exact.json')}})
            (p/'exact.json').write_text('changed')
            with self.assertRaises(AssertionError):check_worker_manifest(p)

    def test_preserved_initial_supervisor_status_is_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write(p/'execution_limits_initial.json',{'returncode':1})
            original=sha(p/'execution_limits_initial.json')
            write(p/'execution_limits.json',{'returncode':0})
            write(p/'completion.json',{'artifact_sha256':{'execution_limits.json':original}})
            self.assertTrue(check_worker_manifest(p))

    def test_unexplained_supervisor_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);write(p/'execution_limits.json',{'returncode':1})
            original=sha(p/'execution_limits.json')
            write(p/'completion.json',{'artifact_sha256':{'execution_limits.json':original}})
            write(p/'execution_limits.json',{'returncode':0})
            with self.assertRaises(AssertionError):check_worker_manifest(p)


if __name__=='__main__':unittest.main()
