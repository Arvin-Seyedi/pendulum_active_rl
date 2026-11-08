"""Simulator-free checks of the separately frozen qualification protocol."""
from fractions import Fraction
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np

from src.quadrotor_mismatch_contract_v1 import (draw_states,state_manifest,action_menu,
    bound_components,source_prediction,linear_prediction,executed_thrust)
from src.quadrotor_mmaq_verification_v1 import exact_majorant,require_nlp
from src.quadrotor_mmaq_v1 import safe_write,immutable,tagged
from src.quadrotor_verification import rk4

ROOT=Path(__file__).resolve().parents[1]


class MMAQContracts(unittest.TestCase):
    def setUp(self):
        self.f=json.loads((ROOT/'results/benchmark_selection/technical_pilot/frozen.json').read_text())

    def test_sampling_matches_independent_raw_bits(self):
        low=[-1,-.1,.5,-.1,-.2,-.1];high=[1,.1,1.5,.1,.2,.1]
        x,draws=draw_states(120001,16,low,high)
        raw=np.random.PCG64(120001).random_raw(96)
        independent=np.array(low)+(np.array(high)-low)*((raw>>11)*2.**-53).reshape(16,6)
        self.assertEqual(x.tobytes(),independent.tobytes())
        m=state_manifest(x,draws,120001)
        self.assertEqual(np.asarray([float.fromhex(v) for v in m['states'][0]['binary64_hex']]).tobytes(),x[0].tobytes())

    def test_duplicate_exclusion_uses_only_coordinates(self):
        low=np.zeros(6);high=np.ones(6);first,_=draw_states(11,1,low,high)
        result,draws=draw_states(11,1,low,high,first)
        self.assertFalse(draws[0]['accepted']);self.assertTrue(draws[1]['accepted'])
        self.assertNotEqual(first.tobytes(),result.tobytes())

    def test_original_menu_is_cartesian_and_not_evaluated(self):
        menu=np.asarray(action_menu(self.f));width=np.asarray(self.f['action_high'])-self.f['action_low']
        self.assertEqual(menu.shape,(9,2));np.testing.assert_array_equal(menu[4],np.zeros(2))
        np.testing.assert_allclose(menu[0],-.05*width,rtol=0,atol=0)
        np.testing.assert_allclose(menu[-1],.05*width,rtol=0,atol=0)

    def test_bound_outward_against_exact_rationals(self):
        for x,u,cache in [(np.array([0.,0.,1.,0.,0.,0.]),np.array(self.f['hover_input']),0.),
                           (np.array([.2,.1,1.,-.1,.2,-.3]),np.array([.07,.28]),.2003),
                           (np.array([-.2,-.1,.8,.1,-.2,.3]),np.array([.28,.07]),-.2003)]:
            for linear in [False,True]:
                b=bound_components(x,u,cache,self.f,linear)
                q=exact_majorant(x,b['model_thrust'],b['executed_thrust'],cache,self.f,linear)
                self.assertTrue(all(Fraction(float(v))>=e for v,e in zip(b['bound'],q)))

    def test_matched_state_equations_fit_analytic_bounds(self):
        x=np.array([.2,.1,1.,-.1,.2,-.3]);u=np.array([.07,.28]);cache=.2003
        for linear in [False,True]:
            b=bound_components(x,u,cache,self.f,linear)
            pred=linear_prediction(x,b['model_thrust'],self.f) if linear else rk4(x,b['model_thrust'],self.f)
            actual=source_prediction(x,b['executed_thrust'],cache,self.f)
            self.assertTrue(np.all(np.abs(pred-actual)<=b['bound']+1e-10))

    def test_reset_context_is_explicit(self):
        x=np.array([0.,0.,1.,0.,.1,.2]);u=np.array([.1,.2])
        reset=bound_components(x,u,x[4],self.f)
        past=bound_components(x,u,x[4]-.01,self.f)
        self.assertGreater(past['cache_angle_bound'],reset['cache_angle_bound'])

    def test_motor_mapping_is_known_without_successor(self):
        clipped,actual,rpm=executed_thrust(np.array([-1.,1.]),self.f)
        self.assertEqual(rpm.shape,(4,));self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_array_equal(clipped,np.array([self.f['action_low'][0],self.f['action_high'][1]]))

    def test_nonfinite_bound_input_rejected(self):
        with self.assertRaises(ValueError):bound_components(np.full(6,np.nan),np.ones(2),0,self.f)

    def test_failure_logging_preserves_partial_nonfinite_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'completion.json'
            payload={'status':'failed_or_incomplete','complete':False,'applied':2,
                     'action':np.array([np.nan,np.inf]),'residual':-np.inf}
            safe_write(path,payload);saved=json.loads(path.read_text())
            self.assertFalse(saved['complete']);self.assertEqual(saved['applied'],2)
            self.assertEqual(saved['action'][0],{'nonfinite_float':'nan'})
            self.assertEqual(saved['action'][1],{'nonfinite_float':'inf'})
            self.assertEqual(saved['residual'],{'nonfinite_float':'-inf'})
            self.assertFalse(path.with_name(path.name+'.tmp').exists())

    def test_frozen_records_cannot_be_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'frozen.json';immutable(path,{'v':1})
            with self.assertRaises(RuntimeError):immutable(path,{'v':2})

    def test_failed_solver_status_never_accepted_as_safe_optimization(self):
        m={'objective':0.,'program_reconstruction_error':0.,'primal_inf':0.,'stationarity_inf':0.,'normalized_complementarity_inf':0.}
        require_nlp(m,'Solve_Succeeded')
        with self.assertRaises(AssertionError):require_nlp(m,'Solved_To_Acceptable_Level')
        m['stationarity_inf']=np.nan
        with self.assertRaises(AssertionError):require_nlp(m,'Solve_Succeeded')

    def test_scope_and_original_gate_remain_fixed(self):
        c=json.loads((ROOT/'configs/experiment_12_mmaq_v1.json').read_text())
        self.assertEqual(c['original_symbolic_gate'],{'threshold':.001,'historical_status':'FAILED'})
        self.assertFalse(c['budget']['additional_full_study_allocation_authorized'])
        self.assertEqual(c['probe_indices'],[0,4,8,12])
        self.assertFalse(c['run_screening']);self.assertFalse(c['run_action_probes']);self.assertFalse(c['train_agents'])


if __name__=='__main__':unittest.main()
