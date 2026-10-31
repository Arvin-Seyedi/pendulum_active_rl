"""Full-study gates and outcome accounting; no simulation or optimization."""

import unittest

from src.recoverable_pd_study import followup_ids
from src.recoverable_pd_study_analysis import aggregate_baseline, describe


class RecoverablePDStudyTests(unittest.TestCase):
    def test_uncertified_states_require_followup_and_are_not_assumed_viable(self):
        records={1:{'certified_nonviable':True},2:{'certified_nonviable':False},3:{'certified_nonviable':False}}
        self.assertEqual(followup_ids([1,2],records),[2])
        self.assertEqual(followup_ids([1],records),[])

    def test_primary_cohort_preserves_archived_labels_and_flags_raw_disagreement(self):
        r={'state_id':1,'identity':{'archived_unsafe':True},
           'comparison':{'archived_vs_legacy_label_disagreement':False,'archived_vs_raw_label_disagreement':True,
                         'legacy_vs_raw_label_disagreement':True,'archived_vs_legacy_crossing_disagreement':False,
                         'legacy_margin_difference':0.,'remeasured_raw_first_crossing_seconds':None,
                         'remeasured_legacy_first_crossing_seconds':.1},
           'metrics':{'violated':False,'returns':-1.,'discounted_returns':-.5,'minimum_margins':.1,'first_crossing_seconds':None}}
        d=aggregate_baseline([r])
        self.assertEqual(d['archived_PD_unsafe']['count'],1)
        self.assertEqual(d['archived_PD_unsafe']['raw_violation_count'],0)
        self.assertEqual(d['discrepancy_ids']['archived_vs_raw_label_disagreement'],[1])
        self.assertEqual(d['raw_vs_legacy_first_crossing_difference_ids'],[1])

    def test_empty_outputs_do_not_manufacture_controller_success_rates(self):
        self.assertEqual(describe([]),{'count':0,'mean':None,'min':None,'max':None})
        self.assertEqual(describe([-3.,-1.])['mean'],-2.)


if __name__=='__main__':unittest.main()
