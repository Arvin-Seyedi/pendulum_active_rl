"""Deterministic exact KKT and failure-capture tests. No real optimization."""
from fractions import Fraction as F
from pathlib import Path
import json
import tempfile
import unittest
import numpy as np
from src.quadrotor_kkt_integrity_v2 import canonical_kkt,require_kkt,matching_program_values
from src.quadrotor_qualification_telemetry_v2 import CandidateGuard,QualificationFailure,TimingLedger,native_opti_snapshot


def scalar_exact(g,lower,upper,lam,grad_f):
    """Scalar textbook KKT oracle, exact rationals; None means absent side."""
    g,lam,grad_f=map(F,[g,lam,grad_f]);l=None if lower is None else F(lower);u=None if upper is None else F(upper)
    eq=l is not None and u is not None and l==u
    feasible=max([F(0)]+([] if l is None else [l-g])+([] if u is None else [g-u]))
    if eq:dual=F(0);products=[]
    else:
        dual=max([F(0)]+([max(-lam,0)] if l is None and u is not None else [])+
                 ([max(lam,0)] if u is None and l is not None else [])+
                 ([abs(lam)] if l is None and u is None else []))
        products=[]
        if l is not None:products.append(max(-lam,0)*(g-l))
        if u is not None:products.append(max(lam,0)*(u-g))
    comp=max([F(0)]+[abs(v) for v in products])
    return {'primal_inf':feasible,'stationarity_inf':abs(grad_f+lam),'dual_sign_inf':dual,
            'complementarity_inf':comp,'normalized_complementarity_inf':comp/(1+abs(lam))}


class KKTIntegrity(unittest.TestCase):
    CASES=[
        ('inactive_upper',-1,None,0,0,0,True),('active_upper',0,None,0,2,-2,True),
        ('inactive_lower',1,0,None,0,0,True),('active_lower',0,0,None,-2,2,True),
        ('two_sided_interior',1,0,2,0,0,True),('two_sided_lower',0,0,2,-2,2,True),
        ('two_sided_upper',2,0,2,2,-2,True),('equality_positive_dual',1,1,1,7,-7,True),
        ('equality_negative_dual',1,1,1,-7,7,True),('unbounded_zero_dual',1,None,None,0,0,True),
        ('wrong_upper_sign',0,None,0,-1,1,False),('wrong_lower_sign',0,0,None,1,-1,False),
        ('unbounded_nonzero_dual',1,None,None,2,-2,False),('upper_primal_infeasible',1,None,0,0,0,False),
        ('lower_primal_infeasible',-1,0,None,0,0,False),('equality_infeasible',2,1,1,0,0,False),
        ('genuine_upper_complementarity',-1,None,0,1,-1,False),
        ('genuine_lower_complementarity',1,0,None,-1,1,False),
        ('interior_bad_dual',1,0,2,1,-1,False),('bad_stationarity',0,None,0,1,0,False)]

    def test_exact_rational_reference_cases(self):
        for name,g,l,u,lam,grad,expected in self.CASES:
            with self.subTest(case=name):
                reference=scalar_exact(g,l,u,lam,grad)
                result=canonical_kkt([g],[-np.inf if l is None else l],[np.inf if u is None else u],[lam],[grad+lam])
                self.assertTrue(result['valid_arithmetic'])
                for key,value in reference.items():self.assertAlmostEqual(result['metrics'][key],float(value),places=14)
                if expected:require_kkt(result,'Solve_Succeeded')
                else:
                    with self.assertRaises(AssertionError):require_kkt(result,'Solve_Succeeded')

    def test_tiny_one_sided_wrong_sign_is_finite_and_below_frozen_tolerance(self):
        r=canonical_kkt([-.1],[-np.inf],[0],[-1e-30],[-1e-30])
        self.assertTrue(np.isfinite(list(r['metrics'].values())).all());require_kkt(r,'Solve_Succeeded')
        self.assertEqual(r['metrics']['dual_sign_inf'],1e-30)

    def test_legacy_one_sided_infinite_product_is_independently_reproduced(self):
        from src.quadrotor_mmaq_verification_v1 import nlp_metrics,require_nlp
        def own(z,p,lam):return [np.array([0]),np.array([-.1]),np.array([-np.inf]),np.array([0]),np.array([-1e-30])]
        def native(z,p):return own(z,p,None)[:4]
        legacy=nlp_metrics(own,native,[0],[],np.array([-1e-30]))
        self.assertTrue(np.isinf(legacy['normalized_complementarity_inf']))
        with self.assertRaises(AssertionError):require_nlp(legacy,'Solve_Succeeded')

    def test_legacy_NaN_equality_can_false_pass(self):
        from src.quadrotor_mmaq_verification_v1 import nlp_metrics,require_nlp
        def own(z,p,lam):return [np.array([0]),np.array([np.nan]),np.array([0]),np.array([0]),np.array([0])]
        def native(z,p):return own(z,p,None)[:4]
        legacy=nlp_metrics(own,native,[0],[],np.array([0]))
        require_nlp(legacy,'Solve_Succeeded')
        current=canonical_kkt([np.nan],[0],[0],[0],[0])
        self.assertFalse(current['valid_arithmetic'])

    def test_invalid_arithmetic_and_bounds_fail_closed(self):
        for g,l,u,lam,grad in [([np.nan],[0],[0],[0],[0]),([0],[0],[1],[np.inf],[0]),
                               ([0],[0],[1],[0],[np.nan]),([0],[np.nan],[1],[0],[0]),
                               ([0],[np.inf],[np.inf],[0],[0]),([0],[-np.inf],[-np.inf],[0],[0]),
                               ([0],[2],[1],[0],[0]),([0,1],[0],[1],[0],[0])]:
            r=canonical_kkt(g,l,u,lam,grad);self.assertFalse(r['valid_arithmetic'])
            with self.assertRaises(AssertionError):require_kkt(r,'Solve_Succeeded')

    def test_genuine_finite_operand_overflow_fails_closed(self):
        r=canonical_kkt([-1e308],[-np.inf],[0.],[1e308],[0.])
        self.assertFalse(r['valid_arithmetic']);self.assertTrue(r['metrics_are_rejection_sentinels'])

    def test_constraint_reconstruction_checks_NaN_and_infinity_signs(self):
        with self.assertRaises(AssertionError):matching_program_values([[np.nan]],[[np.nan]])
        with self.assertRaises(AssertionError):matching_program_values([[-np.inf]],[[np.inf]])
        self.assertEqual(matching_program_values([[-np.inf,1,np.inf]],[[-np.inf,1,np.inf]]),0)

    def test_legacy_reconstruction_does_not_check_infinity_sign(self):
        from src.quadrotor_mmaq_verification_v1 import nlp_metrics,require_nlp
        def own(z,p,lam):return [np.array([0]),np.array([-1]),np.array([-np.inf]),np.array([0]),np.array([0])]
        def native(z,p):return [np.array([0]),np.array([-1]),np.array([np.inf]),np.array([0])]
        old=nlp_metrics(own,native,[0],[],np.array([0]));require_nlp(old,'Solve_Succeeded')
        self.assertEqual(old['program_reconstruction_error'],0.)
        with self.assertRaises(AssertionError):matching_program_values(own(None,None,None)[:4],native(None,None))

    def test_solver_status_remains_separate(self):
        r=canonical_kkt([0],[0],[0],[0],[0])
        with self.assertRaises(AssertionError):require_kkt(r,'Solved_To_Acceptable_Level')

    def test_native_time_parser_does_not_call_kernel_time_selection_time(self):
        from src.quadrotor_qualification_integrity_audit import native_solver_wall_times
        row='       total  |  13.56ms ( 13.56ms)  17.23ms ( 17.23ms)         1'
        np.testing.assert_allclose(native_solver_wall_times(row),[.01723],atol=1e-15)

    def test_QRQP_iteration_failure_categories_are_not_conflated(self):
        from src.quadrotor_qualification_integrity_audit import qrqp_groups,row_label
        text='    0     0         0      0.1  326       0.2   24    0.0044   348         0\n   12     1  -1.9e+02     0.049   367     0.027   246    0.0044   348     0.079  Enforced ubz to reduce |du|, i=344'
        groups=qrqp_groups(text);self.assertEqual(groups[-1][-1]['iteration'],12)
        self.assertNotIn('Max iter',groups[-1][-1]['note'])
        self.assertEqual(row_label(344),{'constraint_row':18,'stage':0,'type':'input_lower','component':0})


class CandidateIntegrity(unittest.TestCase):
    def candidate(self):
        return {'primal':np.array([0.]),'parameters':np.array([1.]),'dual':np.array([0.]),
                'objective':0.,'constraint_values':np.array([0.]),'lower':np.array([0.]),
                'upper':np.array([0.]),'lagrangian_gradient':np.array([0.]),
                'solver_status':'Solve_Succeeded','provenance':'native_solution'}

    def snapshot(self):return {'states':np.zeros((2,6)),'actions':np.zeros((1,2))}

    def test_valid_action_returns_only_after_preservation(self):
        with tempfile.TemporaryDirectory() as tmp:
            g=CandidateGuard(Path(tmp)/'a',{'control_step':1})
            action=g.select(lambda:np.array([.1,.2]),self.candidate,self.snapshot)
            np.testing.assert_array_equal(action,[.1,.2]);self.assertTrue((g.folder/'candidate.npz').exists())
            self.assertEqual(json.loads((g.folder/'status.json').read_text())['status'],'verified_action_not_yet_applied')

    def test_rejected_candidate_is_saved_before_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            g=CandidateGuard(Path(tmp)/'a',{'control_step':1});candidate=self.candidate();candidate['constraint_values']=np.array([1.])
            with self.assertRaises(QualificationFailure):g.select(lambda:[.1,.2],lambda:candidate,self.snapshot)
            data=np.load(g.folder/'candidate.npz');np.testing.assert_array_equal(data['constraint_values'],[1.])
            state=json.loads((g.folder/'status.json').read_text());self.assertEqual(state['residuals']['primal_inf'],1.)
            self.assertFalse(state['fallback_action_applied']);self.assertTrue((g.folder/'prefix_after.npz').exists())

    def test_nonfinite_payload_bits_and_diagnostics_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            g=CandidateGuard(Path(tmp)/'a',{});candidate=self.candidate()
            payload=np.array([0x7ff8000000000042],dtype=np.uint64).view(np.float64)
            candidate['primal']=payload;candidate['constraint_values']=payload
            candidate['native_residuals']=np.array([np.inf,-np.inf,np.nan])
            with self.assertRaises(QualificationFailure):g.select(lambda:[.1,.2],lambda:candidate,self.snapshot)
            saved=np.load(g.folder/'candidate.npz');np.testing.assert_array_equal(saved['primal'].view(np.uint64),payload.view(np.uint64))
            diagnostic=json.loads((g.folder/'constraint_diagnostics.json').read_text());self.assertFalse(diagnostic['valid_arithmetic'])
            self.assertEqual(diagnostic['rows']['values'][0],{'nonfinite_float':'nan'})

    def test_native_exception_preserves_current_debug_candidate_and_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            g=CandidateGuard(Path(tmp)/'a',{})
            def fail():raise RuntimeError('native solve failed')
            candidate=self.candidate();candidate['provenance']='native_debug';candidate['solver_status']='unknown'
            with self.assertRaises(QualificationFailure):g.select(fail,lambda:candidate,self.snapshot)
            self.assertTrue((g.folder/'candidate.npz').exists())
            state=json.loads((g.folder/'status.json').read_text())
            self.assertEqual(state['native_exception']['message'],'native solve failed')
            self.assertTrue(any(r['category']=='native_action_selection' for r in state['timings']))

    def test_unavailable_candidate_is_not_replaced_by_previous_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            g=CandidateGuard(Path(tmp)/'a',{})
            with self.assertRaises(QualificationFailure):g.select(lambda:[.1,.2],lambda:{},self.snapshot)
            state=json.loads((g.folder/'status.json').read_text());self.assertFalse(state['candidate_available'])
            self.assertFalse((g.folder/'candidate.npz').exists());self.assertTrue(state['missing_fields'])

    def test_native_fallback_with_failed_status_cannot_return_an_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            g=CandidateGuard(Path(tmp)/'a',{});candidate=self.candidate();candidate['solver_status']='unknown'
            with self.assertRaises(QualificationFailure):g.select(lambda:[0.,0.],lambda:candidate,self.snapshot)
            state=json.loads((g.folder/'status.json').read_text());self.assertEqual(state['status'],'failed_or_incomplete')

    def test_numpy_false_success_flag_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            g=CandidateGuard(Path(tmp)/'a',{});candidate=self.candidate()
            candidate['native_stats']={'success':np.bool_(False)}
            with self.assertRaises(QualificationFailure):g.select(lambda:[.1,.2],lambda:candidate,self.snapshot)

    def test_attempts_cannot_overwrite_historical_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            CandidateGuard(Path(tmp)/'a',{})
            with self.assertRaises(FileExistsError):CandidateGuard(Path(tmp)/'a',{})

    def test_native_snapshot_does_not_substitute_initial_or_previous_vectors(self):
        from types import SimpleNamespace
        class Fake:
            x='x';p='p';lam_g='lambda';f='f';g='g';lbg='lower';ubg='upper'
            def stats(self):return {'return_status':'unknown','success':False}
        def unavailable(expression):raise RuntimeError('No current iterate')
        fake=Fake();fake.debug=SimpleNamespace(value=unavailable)
        result=native_opti_snapshot(fake,gradient_expression='gradient')
        self.assertEqual(result['provenance'],'native_debug_unverified')
        self.assertEqual(set(result['missing_fields']),set(CandidateGuard.REQUIRED))
        self.assertNotIn('primal',result)

    def test_solve_hook_saves_returned_candidate_before_later_rejection(self):
        from types import SimpleNamespace
        values={'x':[0.],'p':[1.],'lambda':[0.],'f':[0.],'g':[1.],'lower':[0.],'upper':[0.],'gradient':[0.]}
        fake=SimpleNamespace(x='x',p='p',lam_g='lambda',f='f',g='g',lbg='lower',ubg='upper',
                             stats=lambda:{'return_status':'Solve_Succeeded','success':True})
        solution=SimpleNamespace(value=lambda expr:values[expr])
        with tempfile.TemporaryDirectory() as tmp:
            guard=CandidateGuard(Path(tmp)/'a',{})
            guard.native_solve_hook(fake,lambda:solution,gradient_expression='gradient')
            saved=np.load(guard.folder/'candidate.npz')
            np.testing.assert_array_equal(saved['constraint_values'],[1.])
            state=json.loads((guard.folder/'candidate_metadata.json').read_text())
            self.assertEqual(state['candidate_metadata']['solver_status'],'Solve_Succeeded')

    def test_native_snapshot_derives_correct_signed_Lagrangian_gradient(self):
        import casadi as c
        from types import SimpleNamespace
        x=c.MX.sym('x');parameter=c.MX.sym('p');dual=c.MX.sym('lambda')
        symbolic=SimpleNamespace(x=x,p=parameter,lam_g=dual,f=(x-parameter)**2/2,g=x,
             lbg=c.DM(-np.inf),ubg=c.DM(1),stats=lambda:{'return_status':'Solve_Succeeded','success':True})
        solution=SimpleNamespace(value=lambda expr:np.asarray(c.Function('eval',[x,parameter,dual],[expr])(1,2,1)))
        candidate=native_opti_snapshot(symbolic,solution)
        self.assertFalse(candidate['missing_fields'])
        self.assertAlmostEqual(float(candidate['lagrangian_gradient'].item()),0.)
        result=canonical_kkt(candidate['constraint_values'],candidate['lower'],candidate['upper'],
                             candidate['dual'],candidate['lagrangian_gradient'])
        require_kkt(result,'Solve_Succeeded')

    def test_nested_timing_excludes_verification_and_persistence(self):
        ticks=iter([0.,1.,3.,5.]);cpu=iter([0.,.5,1.,2.])
        timing=TimingLedger(clock=lambda:next(ticks),cpu=lambda:next(cpu))
        with timing.span('native_action_selection'):
            with timing.span('candidate_persistence'):pass
        native=timing.records[-1]
        self.assertEqual(native['wall_inclusive'],5.);self.assertEqual(native['wall_exclusive'],3.)
        self.assertEqual(native['cpu_exclusive'],1.5)

    def test_nested_solver_time_is_part_of_native_MPC_latency(self):
        ticks=iter([0.,1.,3.,4.,5.,6.]);cpu=iter([0.,.5,1.5,2.,2.5,3.])
        timer=TimingLedger(clock=lambda:next(ticks),cpu=lambda:next(cpu))
        with timer.span('native_action_selection'):
            with timer.span('native_solver'):pass
            with timer.span('candidate_persistence'):pass
        summary=timer.summary()
        self.assertEqual(summary['native_action_selection_wall'],5.)
        self.assertEqual(summary['exclusive_categories']['candidate_persistence']['exclusive_wall'],1.)


if __name__=='__main__':unittest.main()
