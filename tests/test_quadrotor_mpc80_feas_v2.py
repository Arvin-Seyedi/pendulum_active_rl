"""V2 protocol inheritance/resource boundary tests; no native evaluations."""
from pathlib import Path
import unittest
from src.quadrotor_mpc80_feasibility_v2 import CONFIG,validate_inheritance
from src.quadrotor_pilot import read,sha

ROOT=Path(__file__).resolve().parents[1]


class V2ProtocolTests(unittest.TestCase):
    def test_all_scientific_parameters_and_thresholds_inherited(self):
        validate_inheritance(ROOT,read(ROOT/CONFIG))

    def test_new_reservation_fits_without_refunding_v1(self):
        c=read(ROOT/CONFIG);b=c['budget']
        self.assertEqual(b['prior_remaining_solver_calls']-b['solver_calls'],3070)
        self.assertEqual(b['prior_remaining_physics_steps']-b['physics_steps'],366000)
        self.assertFalse(c['resource_carry']['release_prior_failed_reservation'])

    def test_prior_cpu_and_wall_are_carried_not_reset(self):
        c=read(ROOT/CONFIG);r=c['resource_carry'];prior=read(ROOT/r['prior_v1_ledger'])
        self.assertEqual(sha(ROOT/r['prior_v1_ledger']),r['prior_v1_ledger_sha256'])
        self.assertTrue(r['carry_all_v1_CPU_and_active_wall_charges'])
        self.assertFalse(r['reset_1800_CPU_or_3600_wall_cap'])
        self.assertGreater(prior['charged']['cpu_seconds'],58)
        self.assertEqual(prior['charged']['solver_calls'],300)
        self.assertEqual(prior['charged']['physics_steps'],6000)

    def test_v1_remains_halted_with_only_one_failed_job(self):
        p=ROOT/'results/benchmark_selection/mpc80_feas_v1_study'
        l=read(p/'resource_ledger.json');self.assertTrue(l['halted'])
        self.assertEqual(len(l['reservations']),1);self.assertEqual(l['reservations'][0]['status'],'failed')
        m=read(p/'runs/0_nonlinear80_1000_original/completion.json')
        self.assertEqual(m['actual_native_solve_attempts'],1);self.assertEqual(m['actual_physics_steps'],1)

    def test_execution_and_learning_remain_unauthorized(self):
        c=read(ROOT/CONFIG)
        self.assertFalse(c['execution_authorized']);self.assertFalse(c['run_action_probes'])
        self.assertFalse(c['run_training']);self.assertEqual(c['original_model_agreement']['status'],'FAILED')
