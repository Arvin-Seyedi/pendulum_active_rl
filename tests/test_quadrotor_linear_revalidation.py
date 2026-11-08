"""Simulator-free checks of the independent QP and fail-closed supervisor."""
from pathlib import Path
import json
import unittest
from unittest.mock import patch,MagicMock
from contextlib import nullcontext
from types import SimpleNamespace,ModuleType
import tempfile
import numpy as np
from src.quadrotor_linear_verification import linear_transition,independent_program,parameter_terms,kkt_metrics,require_valid_solution


class LinearRevalidationTests(unittest.TestCase):
    def setUp(self):
        p=Path(__file__).resolve().parents[1]/'results/benchmark_selection/technical_pilot/frozen.json'
        self.f=json.loads(p.read_text());self.program=independent_program(self.f)

    def test_linear_transition_against_coordinate_polynomials(self):
        x=np.array(self.f['cases']['T2'],float);u=np.array([.01,-.01]);h=.02;g=self.f['gravity']
        a=self.f['arm_length']/np.sqrt(2)/self.f['Iyy']*(u[1]-u[0]);vertical=u.sum()/self.f['mass']
        expected=np.array([x[0]+h*x[1]+g*(x[4]*h*h/2+x[5]*h**3/6+a*h**4/24),
            x[1]+g*(x[4]*h+x[5]*h*h/2+a*h**3/6),x[2]+h*x[3]+vertical*h*h/2,
            x[3]+vertical*h,x[4]+h*x[5]+a*h*h/2,x[5]+a*h])
        A,B=linear_transition(self.f)
        actual=np.einsum('ij,j->i',A,x)+np.einsum('ij,j->i',B,u)
        np.testing.assert_allclose(actual,expected,atol=1e-15,rtol=0)

    def test_full_program_dimensions_and_strict_convexity(self):
        self.assertEqual(self.program['A'].shape,(898,326));self.assertEqual(len(self.program['Hdiag']),326)
        self.assertEqual(len(self.program['equality_rows']),246)
        self.assertEqual(self.program['Hdiag'].min(),.1);self.assertEqual(self.program['Hdiag'].max(),5)

    def test_input_bounds_preserve_hover_offset_and_native_tightening(self):
        lo,hi=self.f['action_low'],self.f['action_high'];hover=self.f['hover_input']
        np.testing.assert_allclose(self.program['upper'][18:22],np.r_[np.array(hover)-lo-1e-8,np.array(hi)-hover-1e-8],rtol=0,atol=0)

    def test_failure_and_residual_checks_cannot_accept_fallback(self):
        zero={k:0. for k in ['primal_inf','stationarity_inf','normalized_complementarity_inf','dual_sign_inf']}
        require_valid_solution(True,True,zero)
        for success,finite in [(False,True),(True,False)]:
            with self.assertRaises(AssertionError):require_valid_solution(success,finite,zero)
        for key in zero:
            bad=dict(zero);bad[key]=1e-5
            with self.assertRaises(AssertionError):require_valid_solution(True,True,bad)

    def test_initial_state_equality_is_checked(self):
        refs=np.zeros((6,41));p=np.r_[self.f['cases']['T0'],refs.reshape(-1,order='F')]
        metric=kkt_metrics(self.program,self.f,p,np.zeros(326),np.zeros(898))
        self.assertGreaterEqual(metric['primal_inf'],1)
        with self.assertRaises(AssertionError):require_valid_solution(True,True,metric)

    def test_failed_native_fallback_never_reaches_environment(self):
        from src import quadrotor_pilot as pilot
        fake_module=ModuleType('safe_control_gym.controllers.mpc.mpc_utils')
        fake_module.rk_discrete=lambda *args:None
        for nonfinite in [False,True]:
            with tempfile.TemporaryDirectory() as directory:
                Path(directory,'frozen.json').write_text(json.dumps(self.f))
                env=SimpleNamespace(symbolic=SimpleNamespace(fc_func=None),CTRL_TIMESTEP=.02,
                    reset=MagicMock(return_value=(np.array(self.f['cases']['T0']),{'current_step':0})),
                    step=MagicMock(),close=MagicMock())
                opti=SimpleNamespace(stats=lambda:{'success':nonfinite,'return_status':'unknown','iter_count':-1})
                ctrl=SimpleNamespace(u_prev=None,x_prev=None,opti_dict={'opti':opti},close=MagicMock(),
                    select_action=MagicMock(return_value=np.array([np.nan,np.nan]) if nonfinite else np.array(self.f['hover_input'])))
                physics={'states':[np.array(self.f['cases']['T0'])],'contacts':[0],'integration_seconds':[],'capture_seconds':[]}
                with patch.dict('sys.modules',{fake_module.__name__:fake_module}),patch.object(pilot,'activate'),\
                     patch.object(pilot,'make_environment',return_value=env),patch.object(pilot,'make_controller',return_value=ctrl),\
                     patch.object(pilot,'raw_state',return_value=np.array(self.f['cases']['T0'])),\
                     patch.object(pilot,'capture_physics',return_value=nullcontext(physics)):
                    if nonfinite:
                        # The frozen legacy writer rejects NaN in its failure
                        # payload. Preserve this limitation; verify that it is
                        # fail-closed, not a completed or safe episode.
                        with self.assertRaises(ValueError):
                            pilot.run_episode(Path(directory),Path(directory),Path(directory),'T0','linear40','original')
                        self.assertFalse((Path(directory)/'runs/T0_linear40_1000_original/completion.json').exists())
                    else:
                        r=pilot.run_episode(Path(directory),Path(directory),Path(directory),'T0','linear40','original')
                        self.assertFalse(r['complete']);self.assertEqual(r['transitions'],0)
                env.step.assert_not_called()


if __name__=='__main__':unittest.main()
