"""Guard accepted archives from accidental finalization output writes."""
from pathlib import Path
import tempfile,unittest
from src.quadrotor_counterfactual_finalization import verify
ROOT=Path(__file__).resolve().parents[1]

class ReadOnlyFinalizationTests(unittest.TestCase):
    def test_existing_output_is_never_rewritten(self):
        with tempfile.TemporaryDirectory() as t:
            output=Path(t)/'sentinel.json';output.write_bytes(b'original bytes')
            with self.assertRaises(FileExistsError):verify(ROOT,ROOT,'/private/tmp/e12_safe_control_gym',output)
            self.assertEqual(output.read_bytes(),b'original bytes')
    def test_scientific_archive_rejects_fresh_output(self):
        with self.assertRaisesRegex(ValueError,'accepted evidence'):
            verify(ROOT,ROOT,'/private/tmp/e12_safe_control_gym',ROOT/'results/benchmark_selection/counterfactual_pilot_B_v2/forbidden_new_output.json')
    def test_historical_preflight_rejects_fresh_output(self):
        with self.assertRaisesRegex(ValueError,'accepted evidence'):
            verify(ROOT,ROOT,'/private/tmp/e12_safe_control_gym',ROOT/'results/benchmark_selection/counterfactual_wiring_review_v2/forbidden_new_output.json')

if __name__=='__main__':unittest.main()
