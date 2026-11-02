"""Fail-closed archive checks; no simulator, controller or MATLAB process."""
from pathlib import Path
import hashlib
import tempfile
import unittest
import numpy as np
from src.drilling_archive_verification import check_hashes,nested_equal,outside_archive

class DrillingArchiveVerificationTests(unittest.TestCase):
    def test_hash_mismatch_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'native.mat';p.write_bytes(b'original')
            h=hashlib.sha256(p.read_bytes()).hexdigest();check_hashes(folder,{'native.mat':h})
            p.write_bytes(b'changed')
            with self.assertRaises(AssertionError):check_hashes(folder,{'native.mat':h})
    def test_hash_path_cannot_escape_root(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):check_hashes(folder,{'../outside':'invalid'})
    def test_verification_cannot_write_into_accepted_archive(self):
        with tempfile.TemporaryDirectory() as folder:
            archive=Path(folder)/'native_pilot'
            with self.assertRaises(ValueError):outside_archive(archive/'original/audit.json',archive)
            outside_archive(Path(folder)/'finalization/verification.json',archive)
    def test_nested_gain_arrays_are_checked_including_sign_bits(self):
        a=np.empty((1,1),object);a[0,0]=np.array([[0.,1.]])
        b=np.empty((1,1),object);b[0,0]=np.array([[0.,1.]])
        self.assertTrue(nested_equal(a,b));b[0,0][0,0]=-0.
        self.assertFalse(nested_equal(a,b))

if __name__=='__main__':unittest.main()
