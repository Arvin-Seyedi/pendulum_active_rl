"""Small reporting fixtures; no interval propagation or simulator calls."""

from copy import deepcopy
import unittest

from src.viability_study_analysis import summarize


def fixture(sid, group, certified=False):
    trials = []
    for action in (-2, 2):
        hit = certified and action == -2
        final = {'state_id': sid, 'action': action, 'precision_bits': 128,
                 'status': 'certified_nonviable' if hit else 'opposite_exit',
                 'decisive_step': 5, 'crossing_component': 'theta',
                 'checker': {'passed': True, 'model_only': True, 'certified': hit, 'check_count': 5},
                 'generation_repeat_exact': True,
                 'diagnostics': {'interval_steps': 5, 'crossing_separation_lower_exact': ['1','10'],
                     'crossing_separation_lower_approx': .1, 'maximum_width_exact': {'theta':['1','1000'], 'omega':['1','1000']},
                     'maximum_width_approx': {'theta':.001,'omega':.001},
                     'safe_prefix_minimum_margin_lower_exact':['1','10'],
                     'safe_prefix_minimum_margin_lower_approx':.1}}
        simulator = {'first_crossing_step':5,'model_and_simulator_first_crossing_step_match':True,
                     'replay_bitwise':True,'minimum_raw_margin_over_201_states':-.1,
                     'observed_prefix_error_upper_approx':{'theta':1e-15,'omega':1e-15}}
        trials.append({'action':action,'attempts':[final],'final':final,'simulator':simulator})
    return {'state_id':sid,'group':group,'initial_hex':[0.1.hex(),0.0.hex()],
            'mathematical_classification':'certified_nonviable' if certified else 'no_nonviability_certificate',
            'real_model_only':True,'equilibrium_proof':None,'trials':trials,
            'PD_simulator':{'witness':group!='PD_unsafe','violated':group=='PD_unsafe',
                            'returns':-1.,'minimum_margins':-.1 if group=='PD_unsafe' else .1,
                            'first_crossing_step':5 if group=='PD_unsafe' else None,'replay_bitwise':True}}


class ViabilityStudyReportingTests(unittest.TestCase):
    def test_unresolved_case_is_not_silently_nonviable_or_viable(self):
        result = summarize([fixture(1,'PD_unsafe',True),fixture(2,'PD_unsafe'),fixture(3,'nominal_safe')], [1])
        self.assertEqual(result['primary']['denominator'],2)
        self.assertEqual(result['primary']['certified_ids'],[1])
        self.assertEqual(result['primary']['unresolved_ids'],[2])
        self.assertTrue(result['primary']['additional_methods_required_for_unresolved'])
        self.assertEqual(result['controls']['false_nonviability_count'],0)
        self.assertEqual(result['controls']['model_unresolved_ids'],[3])
        self.assertEqual(result['controls']['simulator_safe_count'],1)
        self.assertEqual(result['reused_state_ids'],[1])

    def test_actual_crossing_direction_and_certificate_direction_are_distinct(self):
        result = summarize([fixture(1,'PD_unsafe',True)], [])
        minus, plus = result['directions']
        self.assertEqual(minus['crossing_direction'],'upper')
        self.assertTrue(minus['certified_nonviable'])
        self.assertEqual(plus['crossing_direction'],'upper')
        self.assertFalse(plus['certified_nonviable'])
        self.assertEqual(result['states'][0]['certified_crossing_time_seconds'],.25)
        self.assertFalse(result['primary']['additional_methods_required_for_unresolved'])

    def test_duplicate_identity_or_invalid_checker_is_rejected(self):
        record = fixture(1,'PD_unsafe',True)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            summarize([record,deepcopy(record)],[])
        record['trials'][0]['final']['checker']['passed']=False
        with self.assertRaisesRegex(ValueError,'passed model-only'):
            summarize([record],[])

    def test_equilibrium_exact_witness_is_not_an_extremal_success(self):
        record=fixture(96,'equilibrium')
        record['initial_hex']=[0.0.hex(),0.0.hex()]
        record['mathematical_classification']='exact_equilibrium_viable'
        record['equilibrium_proof']='u=0 leaves the exact origin unchanged'
        result=summarize([record],[96])
        self.assertEqual(result['equilibrium'][0]['mathematical_status'],'exact_equilibrium_viable')
        self.assertEqual(result['numerical_summary']['direction_status_counts'],{'opposite_exit':2})
        self.assertEqual(result['primary']['certified_count'],0)


if __name__=='__main__':
    unittest.main()
