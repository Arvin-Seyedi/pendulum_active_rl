"""Read-only audit output guards; no scientific fitting or engine calls."""
from pathlib import Path
import tempfile,unittest
from src.quadrotor_archive_finalization import verify
ROOT=Path(__file__).resolve().parents[1]

class B0FinalizationTests(unittest.TestCase):
    def test_existing_output_remains_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'old.json';p.write_bytes(b'unchanged')
            with self.assertRaises(FileExistsError):verify(ROOT,ROOT,'/private/tmp/e12_safe_control_gym',p)
            self.assertEqual(p.read_bytes(),b'unchanged')
    def test_archive_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'accepted evidence'):verify(ROOT,ROOT,'/private/tmp/e12_safe_control_gym',ROOT/'results/benchmark_selection/b0_archive_v1/forbidden_output.json')
    def test_parent_evidence_output_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'accepted evidence'):verify(ROOT,ROOT,'/private/tmp/e12_safe_control_gym',ROOT/'results/benchmark_selection/counterfactual_pilot_B_v2/forbidden_output.json')

if __name__=='__main__':unittest.main()
