"""E10 numerical, selection and claim-accounting fixtures; no Gym/solver runs."""

from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from src.horizon_experiment import read_json
from src.policy_improvement import (
    PILOT_IDS, OPTIMIZATION_IDS, assess, controller_bank, derive_lqr,
    evaluate_fixed, measure, safe_oracle,
)
from src.policy_improvement_experiment import worker

ROOT=Path(__file__).resolve().parents[1]


class PolicyImprovementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config=read_json(ROOT/'configs/experiment_10_pilot.json')

    def test_frozen_subset_and_pilot_cover_four_cost_strata_without_equilibrium_optimization(self):
        s=read_json(ROOT/'results/safe_policy_improvement/pilot/selection.json')
        self.assertEqual(s['pilot_ids'],[3280,2956,5383,2568,6194])
        self.assertEqual(len(s['optimization_ids_sorted']),64)
        self.assertEqual(len(set(s['optimization_ids_sorted'])),64)
        self.assertNotIn(3280,s['optimization_ids_sorted'])
        self.assertEqual(len(s['remaining_optimization_ids']),60)
        self.assertEqual([r['population_count'] for r in s['strata']],[833,833,832,832])
        self.assertTrue(all(len(r['selected'])==16 for r in s['strata']))
        for group,pilot in zip(s['strata'],PILOT_IDS[1:]):
            self.assertEqual(min(group['selected'],key=lambda r:(r['PD_cost'],r['grid_id']))['grid_id'],pilot)

    def test_reward_matched_LQR_is_stabilizing_and_matches_independent_Riccati_iteration(self):
        lqr=derive_lqr(self.config)
        self.assertLess(lqr['max_absolute_DARE_residual'],1e-10)
        self.assertLess(lqr['spectral_radius'],1)
        self.assertLess(lqr['independent_Riccati_max_difference'],1e-10)
        self.assertTrue(lqr['repeated_DARE_solution_bitwise'])
        controllers=controller_bank(lqr)
        self.assertEqual(len(controllers),10)
        self.assertEqual(sum(c['name']=='PD_8_2' for c in controllers),1)
        self.assertEqual(sum(c['kind']=='LQR' for c in controllers),1)

    def test_PD_evaluation_delegates_to_verified_native_dtype_evaluator(self):
        initial=np.array([0.,0.])
        with patch('src.policy_improvement.evaluate_gain',return_value='delegated') as evaluate:
            result=evaluate_fixed(initial,1,{'name':'PD_8_2','kind':'PD','kp':8.,'kd':2.})
            self.assertEqual(result,'delegated')
            evaluate.assert_called_once_with(initial,1,8.,2.)

    def test_effort_RMS_TV_and_reward_components_have_explicit_definitions(self):
        u=np.zeros(200,dtype=np.float32);u[:4]=[1,-1,2,-2]
        arrays={'raw_states':np.zeros((1,201,2)),'actions':u[None,:,None],
                'margins':np.ones((1,201)),'rewards':(-.001*u.astype(np.float64)**2)[None]}
        m=measure(arrays,self.config)
        self.assertAlmostEqual(m['squared_torque_effort'],.5)
        self.assertAlmostEqual(m['rms_torque'],np.sqrt(.05))
        self.assertEqual(m['peak_torque'],2)
        self.assertEqual(m['total_variation'],11)
        self.assertEqual(m['initial_torque_jump'],1)
        self.assertEqual(m['saturation_fraction'],.01)
        self.assertAlmostEqual(m['torque_reward_cost'],-m['returns'])
        self.assertTrue(m['witness'])

    @staticmethod
    def metric(value,margin=1.,effort=1.):
        return {'returns':value,'discounted_returns':value,'minimum_margins':margin,
                'initial_raw_margin':1.,'squared_torque_effort':effort,'witness':margin>=0,'violated':margin<0}

    def test_worst_case_uses_minimum_paired_difference_not_difference_of_minima(self):
        b=[self.metric(-10),self.metric(-1)]
        r=[self.metric(-9),self.metric(-4)]
        result=assess(r,b,self.config)
        self.assertEqual(min(x['returns'] for x in r)-min(x['returns'] for x in b),1)
        self.assertEqual(result['minimum_per_state_paired_return_difference'],-3)
        self.assertFalse(result['worst_paired_return_criterion'])

    def test_mean_reward_threshold_takes_maximum_and_never_relaxes_negative_safety(self):
        b=[self.metric(-3)]
        r=[self.metric(-2.5)]
        a=assess(r,b,self.config)
        self.assertAlmostEqual(a['return_threshold'],.3)
        self.assertTrue(a['practical_reward_pass'])
        a=assess([self.metric(-2.5,margin=-1e-12)],b,self.config)
        self.assertFalse(a['safety_and_margin_criterion'])
        self.assertFalse(a['practical_reward_pass'])
        a=assess([self.metric(-.01)],[self.metric(-.02)],self.config)
        self.assertEqual(a['return_threshold'],.1)

    def test_effort_and_margin_criteria_remain_separate_and_zero_headroom_is_ineligible(self):
        b=[self.metric(-1)]
        r=[self.metric(-.999,effort=.8)]
        a=assess(r,b,self.config)
        self.assertTrue(a['practical_effort_pass'])
        self.assertFalse(a['practical_reward_pass'])
        self.assertEqual(a['margin_eligible_count'],0)
        self.assertIsNone(a['practical_margin_pass'])
        b=[self.metric(-1,margin=.8)]
        r=[self.metric(-1,margin=.83)]
        a=assess(r,b,self.config)
        self.assertTrue(a['practical_margin_pass'])

    def test_oracle_PD_fallback_is_not_an_optimized_success_and_repeats_do_not_add_choices(self):
        baseline=self.metric(-2)
        worse=dict(self.metric(-2.1),phase='original',mode='tracking',candidate_available=True,start='PD',start_index=0)
        repeat=dict(self.metric(-1),phase='repeat',mode='tracking',candidate_available=True,start='PD',start_index=0)
        unsafe=dict(self.metric(-.5,margin=-.1),phase='original',mode='tracking',candidate_available=True,start='zero',start_index=1)
        result=safe_oracle([worse,repeat,unsafe],baseline,'tracking')
        self.assertEqual(result['source'],'PD_fallback')
        self.assertFalse(result['optimized_safe_improvement'])
        self.assertTrue(result['optimized_safe_candidate_available'])
        self.assertEqual(result['delta_return'],0)
        improved=dict(self.metric(-1.8),phase='original',mode='tracking',candidate_available=True,start='random_blocks',start_index=2)
        result=safe_oracle([worse,repeat,unsafe,improved],baseline,'tracking')
        self.assertEqual(result['source'],'optimized')
        self.assertEqual(result['selected_start'],'random_blocks')
        self.assertTrue(result['optimized_safe_improvement'])

    def test_worker_refuses_equilibrium_and_remaining_optimization_states_before_solver(self):
        real_read=read_json
        with patch('src.policy_improvement_experiment.solve_attempt') as solve:
            for sid in (3280,44):
                request={'state_id':sid,'authorized_ids':OPTIMIZATION_IDS}
                def read(path):
                    return request if Path(path).name=='request.json' else real_read(path)
                with patch('src.policy_improvement_experiment.read_json',side_effect=read):
                    with self.assertRaises(AssertionError):
                        worker(ROOT,ROOT/f'results/safe_policy_improvement/pilot/state_{sid:04d}/optimization/PD/original')
            solve.assert_not_called()


if __name__=='__main__':unittest.main()
