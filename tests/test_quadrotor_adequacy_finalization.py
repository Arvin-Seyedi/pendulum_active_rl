"""Output and historical-document guards, no learning or engine calls."""
from pathlib import Path
import tempfile,unittest,hashlib
from unittest.mock import patch
from src.quadrotor_adequacy_finalization import verify,context_versions
ROOT=Path(__file__).resolve().parents[1]
class B1FinalizationTests(unittest.TestCase):
    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'old.json';p.write_text('preserve')
            with self.assertRaises(FileExistsError):verify(ROOT,ROOT,'/private/tmp/e12_safe_control_gym',p)
            self.assertEqual(p.read_text(),'preserve')
    def test_accepted_evidence_output_rejected(self):
        with self.assertRaisesRegex(ValueError,'accepted evidence'):verify(ROOT,ROOT,'/private/tmp/e12_safe_control_gym',ROOT/'results/benchmark_selection/b1_archive_v1/unapproved.json')
    def test_symlinked_evidence_output_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'results').symlink_to(ROOT/'results',target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'accepted evidence'):verify(root,ROOT,'/private/tmp/e12_safe_control_gym',root/'results/benchmark_selection/b1_archive_v1/unapproved.json')
    def test_scientific_changes_cannot_be_masked_as_documentation(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'science.py').write_text('new')
            with self.assertRaisesRegex(AssertionError,'scientific source'):context_versions(root,root,{'HEAD':'old','tracked_sha256':{'science.py':'different'}})
    def test_context_original_must_match_git_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);(root/'CURRENT_STATUS.md').write_bytes(b'new');old=b'original';h=hashlib.sha256(old).hexdigest();snap={'HEAD':'old','tracked_sha256':{'CURRENT_STATUS.md':h}}
            with patch('src.quadrotor_adequacy_finalization.subprocess.check_output',return_value=old):
                versions,changes=context_versions(root,root,snap);self.assertEqual(versions['CURRENT_STATUS.md'],old);self.assertEqual(changes['CURRENT_STATUS.md']['historical_sha256'],h)
            with patch('src.quadrotor_adequacy_finalization.subprocess.check_output',return_value=b'wrong'):
                with self.assertRaises(AssertionError):context_versions(root,root,snap)
