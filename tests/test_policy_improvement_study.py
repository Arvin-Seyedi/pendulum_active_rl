"""Full-study authorization and oracle accounting; no simulation or solving."""

from pathlib import Path
import unittest
from unittest.mock import patch

from src.horizon_experiment import read_json
from src.policy_improvement_study import settings,worker
from src.policy_improvement_study_analysis import compare_oracle,describe

ROOT=Path(__file__).resolve().parents[1]


class PolicyImprovementStudyTests(unittest.TestCase):
    def test_scientific_configuration_is_identical_to_reviewed_pilot(self):
        scientific,control,pilot,output=settings(ROOT)
        self.assertEqual(scientific,read_json(pilot/'protocol.json'))
        self.assertEqual(control['new_optimization_solves'],180)
        self.assertEqual(control['total_scientific_optimization_solves'],192)
        self.assertFalse(control['new_solver_repeats'])
        self.assertEqual(scientific['criteria']['minimum_per_state_paired_return_difference'],-.1)

    def test_PD_fallback_is_never_an_optimized_improvement_beyond_LQR(self):
        oracle={'source':'PD_fallback','optimized_safe_improvement':False,'selected_start':None,
                'safe_candidate_count':0,'delta_return':0.,
                'selected_metrics':{'returns':-1.,'discounted_returns':-.9,'squared_torque_effort':1.,'minimum_margins':.5,'witness':True}}
        result=compare_oracle(oracle,{'returns':-2.,'witness':True})
        self.assertEqual(result['delta_vs_LQR'],1.)
        self.assertFalse(result['genuine_optimized_gain_beyond_LQR_gt_0_1'])

    def test_full_worker_does_not_rerun_pilot_or_unsampled_states(self):
        real_read=read_json;selection=real_read(ROOT/'results/safe_policy_improvement/pilot/selection.json')
        unsampled=next(sid for sid in selection['population_ids'] if sid not in selection['optimization_ids_sorted'])
        with patch('src.policy_improvement_study.solve_attempt') as solve:
            for sid in (3280,2956,unsampled):
                def read(path):
                    if Path(path).name=='request.json':return {'state_id':sid}
                    if Path(path).name=='selection.json':return selection
                    return real_read(path)
                with patch('src.policy_improvement_study.read_json',side_effect=read):
                    with self.assertRaises(AssertionError):worker(ROOT,ROOT/f'results/safe_policy_improvement/full_study/state_{sid:04d}/optimization/PD/original')
            solve.assert_not_called()

    def test_missing_numeric_outcomes_remain_missing(self):
        self.assertIsNone(describe([])['mean'])
        self.assertEqual(describe([-2.,-1.])['mean'],-1.5)


if __name__=='__main__':unittest.main()
