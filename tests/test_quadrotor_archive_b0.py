"""Zero-solve causal-feature, split, numerical and deterministic B0 tests."""
from pathlib import Path
from copy import deepcopy
import json,tempfile,unittest
import numpy as np
import torch
from src import quadrotor_archive_data as d
from src import quadrotor_archive_learner as l
from src import quadrotor_archive_assessment as a
ROOT=Path(__file__).resolve().parents[1]

class ArchiveB0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c=json.loads((ROOT/'configs/experiment_12_b0_archive_v1.json').read_text());b=ROOT/'results/benchmark_selection/mpc80_feas_v2_study'
        cls.m=json.loads((b/'frozen.json').read_text());cls.ref=np.load(b/'reference.npy');cls.s=np.zeros((301,6));cls.s[:,2]=1;cls.u=np.tile(cls.m['hover_input'],(300,1))
    def features(self,s=None,u=None):return d.build_features(self.s if s is None else s,self.u if u is None else u,self.ref,self.m)
    def test_three_whole_trajectory_partitions_and_nested_subsets(self):
        sub=d.validate_partitions(self.c);self.assertEqual([len(v) for v in sub.values()],[2,4,8,10]);self.assertEqual(sub[4],[0,4,8,12])
        for n in [2,4,8]:self.assertEqual(sub[n],sub[10][:n])
    def test_overlap_is_rejected(self):
        c=deepcopy(self.c);c['partitions']['assessment'][0]=0
        with self.assertRaises(ValueError):d.validate_partitions(c)
    def test_preview_all_native_indices_and_padding(self):
        x=self.features();mid=(np.asarray(self.m['state_low'])+self.m['state_high'])/2;width=np.asarray(self.m['state_high'])-self.m['state_low']
        for k in [0,220,221,299]:
            expected=(2*(self.ref[np.minimum(k+np.arange(81),300)].T-mid[:,None])/width[:,None]).astype(np.float32)
            self.assertEqual(x[k,6:492].tobytes(),expected.reshape(-1).tobytes())
    def test_phase_and_reset_are_causal(self):
        x=self.features();np.testing.assert_array_equal(x[:,492],np.arange(300,dtype=np.float64).astype(np.float32)/300)
        self.assertEqual(x[0,495],1);self.assertEqual(x[1:,495].sum(),0)
    def test_current_target_is_not_input(self):
        u=self.u.copy();u[20]+=.01;x=self.features();changed=self.features(u=u)
        self.assertEqual(x[:21].tobytes(),changed[:21].tobytes());self.assertNotEqual(x[21].tobytes(),changed[21].tobytes())
        self.assertEqual(x[22:].tobytes(),changed[22:].tobytes())
    def test_future_physical_states_do_not_change_past_features(self):
        s=self.s.copy();s[100:,0]+=.2
        self.assertEqual(self.features()[:100].tobytes(),self.features(s=s)[:100].tobytes())
    def test_future_actions_do_not_change_past_features(self):
        u=self.u.copy();u[100:]+=.001
        self.assertEqual(self.features()[:101].tobytes(),self.features(u=u)[:101].tobytes())
    def test_fixed_scaling_does_not_fit_assessment_values(self):
        before=self.features();s=self.s.copy();s[0,0]=1000;self.features(s=s)
        self.assertEqual(before.tobytes(),self.features().tobytes())
    def test_no_successor_or_terminal_state_as_current_label_input(self):
        s=self.s.copy();s[-1]=1000
        self.assertEqual(self.features().tobytes(),self.features(s=s).tobytes())
    def test_nonfinite_or_incomplete_data_rejected(self):
        s=self.s.copy();s[0,0]=np.nan
        with self.assertRaises(ValueError):self.features(s=s)
        with self.assertRaises(ValueError):d.build_features(self.s[:-1],self.u,self.ref,self.m)
    def test_action_normalization_round_trip(self):
        y=d.normalized_targets(self.u,self.m);recovered=y*(np.asarray(self.m['action_high'])-self.m['action_low'])+self.m['action_low']
        np.testing.assert_allclose(recovered,self.u,rtol=0,atol=1e-16)
    def test_initialization_and_uniform_schedules_are_seed_paired(self):
        l.configure_determinism(122000);first=l.ArchivePolicy();h=l.parameters_sha(first);l.configure_determinism(122000);self.assertEqual(h,l.parameters_sha(l.ArchivePolicy()))
        l.configure_determinism(122001);self.assertNotEqual(h,l.parameters_sha(l.ArchivePolicy()))
        u=l.uniform_schedule(122000,8,16);self.assertEqual(u.tobytes(),l.uniform_schedule(122000,8,16).tobytes());self.assertTrue(np.all(np.floor(u*600)<600))
    def test_model_output_finite_bounded_and_gradients_finite(self):
        l.configure_determinism(122000);net=l.ArchivePolicy();x=torch.linspace(-10,10,4*496).reshape(4,496);p=net(x)
        self.assertTrue(torch.isfinite(p).all() and (p>=0).all() and (p<=1).all());p.square().mean().backward();self.assertTrue(all(torch.isfinite(v.grad).all() for v in net.parameters()))
    def test_synthetic_update_reproducibility(self):
        records=[]
        for _ in range(2):
            l.configure_determinism(122000);net=l.ArchivePolicy();op=torch.optim.Adam(net.parameters(),lr=.001,foreach=False)
            # Synthetic fixture, not scientific fitting or a training-data query.
            for _ in range(2):op.zero_grad();loss=(net(torch.zeros(4,496))-.2).square().mean();loss.backward();op.step()
            records.append(l.parameters_sha(net))
        self.assertEqual(records[0],records[1])
    def test_completed_output_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileExistsError):l.fit_fixed(np.zeros((2,496)),np.zeros((2,2)),1,self.c,Path(t))
    def test_checkpoint_restores_predictions_bitwise(self):
        with tempfile.TemporaryDirectory() as t:
            l.configure_determinism(122000);net=l.ArchivePolicy();x=self.features();p=l.predict(net,x);path=Path(t)/'fixture.pt';torch.save(net.state_dict(),path)
            other=l.ArchivePolicy();other.load_state_dict(torch.load(path,weights_only=True));self.assertEqual(p.tobytes(),l.predict(other,x).tobytes())
    def test_ridge_same_feature_dimension_and_finite_backward_error(self):
        x=self.features()[:20];y=np.full((20,2),.3);beta,diag=l.ridge_fit(x,y,.001);pred,raw=l.ridge_predict(beta,x)
        self.assertEqual(beta.shape,(497,2));self.assertLess(diag['backward_error'],1e-12);self.assertLess(np.max(np.abs(pred-y)),1e-8)
    def test_bootstrap_uses_seed_trajectory_matrix(self):
        z=np.full((5,4),.02);s=a.bootstrap_means(z);self.assertAlmostEqual(s['95_CI'][0],.02);self.assertAlmostEqual(s['95_CI'][1],.02)
        with self.assertRaises(ValueError):a.bootstrap_means(np.ones(1200))
    def test_metrics_units_and_phase_denominators(self):
        m=a.trajectory_metrics(np.full((300,2),.01),np.zeros((300,2)),[.2,.2],self.c['phases'])
        self.assertAlmostEqual(m['normalized_MAE'],.01);self.assertAlmostEqual(m['physical_MAE_N'],.002)
        for phase in self.c['phases']:self.assertAlmostEqual(m['phases'][phase]['normalized_MAE'],.01)
    def test_gates_do_not_pass_absent_gap_or_weak_model(self):
        def curve(value):
            matrix=np.full((5,4),value);return {'assessment_seed_trajectory_MAE':matrix.tolist(),'assessment_summary':a.bootstrap_means(matrix),'assessment_phase_macro_MAE':{'early':value}}
        curves={'4':curve(.02),'10':curve(.01)};baseline={'10':{'assessment_macro_MAE':.011}}
        self.assertTrue(a.decision(curves,baseline,self.c)['GO_for_designing_future_acquisition'])
        curves={'4':curve(.002),'10':curve(.001)};baseline={'10':{'assessment_macro_MAE':.002}}
        self.assertFalse(a.decision(curves,baseline,self.c)['GO_for_designing_future_acquisition'])
        curves={'4':curve(.02),'10':curve(.01)};baseline={'10':{'assessment_macro_MAE':.001}}
        self.assertFalse(a.decision(curves,baseline,self.c)['GO_for_designing_future_acquisition'])

if __name__=='__main__':unittest.main()
