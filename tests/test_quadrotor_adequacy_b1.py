"""Zero-solve fixtures for the prospective B1 protocol."""
from pathlib import Path
import json,unittest
import numpy as np
import torch
from src.quadrotor_adequacy_data import compact,rpm_commands,same
from src.quadrotor_adequacy_learner import AdequacyPolicy,uniform_schedule
from src.quadrotor_adequacy_assessment import metrics,bootstrap,decide
ROOT=Path(__file__).resolve().parents[1]
class B1Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=json.loads((ROOT/'configs/experiment_12_b1_archive_v1.json').read_text());torch.set_num_threads(1)
    def test_exact_compact_order(self):
        x=np.arange(496,dtype=np.float32);np.testing.assert_array_equal(compact(x),x[self.c['representations']['compact']]);self.assertEqual(compact(x).shape,(16,))
    def test_compact_dimension_reject(self):
        with self.assertRaises(ValueError):compact(np.zeros(16))
    def test_compact_does_not_select_future_physical_state(self):
        x=np.zeros((2,300,496),np.float32);a=compact(x);x[:,100:,0]=4
        self.assertTrue(same(a[:,:100],compact(x)[:,:100]))
    def test_parameter_counts_and_hidden_stream_pairing(self):
        a=AdequacyPolicy(496,123000);b=AdequacyPolicy(16,123000)
        self.assertEqual(sum(p.numel() for p in a.parameters()),36098);self.assertEqual(sum(p.numel() for p in b.parameters()),5378)
        for k in [2,4]:self.assertTrue(torch.equal(a.network[k].weight,b.network[k].weight));self.assertTrue(torch.equal(a.network[k].bias,b.network[k].bias))
    def test_seed_reproducibility(self):
        a=AdequacyPolicy(16,123000);b=AdequacyPolicy(16,123000)
        for k,v in a.state_dict().items():self.assertTrue(torch.equal(v,b.state_dict()[k]))
    def test_schedule_pairing_and_prefix(self):
        a=uniform_schedule(123000,12,4);self.assertTrue(same(a[:3],uniform_schedule(123000,3,4)));self.assertTrue((np.floor(a*1200)<1200).all())
    def test_metrics_units_and_phase(self):
        y=np.zeros((4,300,2));p=y+.1;m=metrics(p,y,[.2,.2],self.c['phases'])
        self.assertAlmostEqual(m['physical_MAE_N'],.02);self.assertAlmostEqual(m['physical_MSE_N2'],.0004);self.assertEqual(len(m['trajectories']),4)
    def test_metrics_reject_nonfinite(self):
        with self.assertRaises(ValueError):metrics(np.full((4,300,2),np.nan),np.zeros((4,300,2)),[1,1],{})
    def test_bootstrap_not_transition_resampling(self):
        a=bootstrap(np.ones((5,4))*.002,self.c);self.assertAlmostEqual(a['98.333_CI'][0],.002)
        with self.assertRaises(ValueError):bootstrap(np.ones((5,1200)),self.c)
    def test_weak_data_contrast_is_no_go(self):
        mat={k:np.ones((5,4))*v for k,v in [('full_n10_u5000',.03),('full_n10_u20000',.02),('compact_n10_u20000',.015),('compact_n4_u20000',.016)]}
        v=decide(mat,{k:.001 for k in mat},{k:.002 for k in mat},.015,self.c);self.assertFalse(v['GO_for_reviewing_future_acquisition_design'])
    def test_rpm_source_reconstruction_operation_order(self):
        a=np.tile([12000.,13000.,13000.,12000.],(300,1));u=rpm_commands(a,{'KF':3e-10});self.assertEqual(u.shape,(300,2));self.assertEqual(u[0,0],2*(12000.**2*3e-10))
        with self.assertRaises(ValueError):rpm_commands(a[:2],{'KF':3e-10})
    def test_frozen_splits_seeds_checkpoints(self):
        self.assertEqual(self.c['seeds'],list(range(123000,123005)));self.assertEqual(self.c['checkpoints'],[5000,20000]);self.assertEqual(self.c['partitions']['assessment'],[3,7,11,15]);self.assertEqual(self.c['partitions']['subset_sizes'],[4,10])
