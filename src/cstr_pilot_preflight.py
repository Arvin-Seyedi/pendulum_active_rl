"""Zero-real-operation qualification of the source-preserving CSTR runner."""
from pathlib import Path
import sys,json,subprocess
from unittest.mock import patch
from contextlib import contextmanager
import numpy as np
import casadi as ca
import casadi.tools as ct
from src import cstr_expert_pilot as p
from src import cstr_verification as v

class InertBackend:
    def __call__(self,*args,**kwargs):raise AssertionError('Inert backend cannot solve or integrate')

@contextmanager
def zero_operations():
    counts={'solver_factory_intercepts':0,'integrator_factory_intercepts':0,'real_operations':0}
    def nlp(*a,**k):counts['solver_factory_intercepts']+=1;return InertBackend()
    def integrator(*a,**k):counts['integrator_factory_intercepts']+=1;return InertBackend()
    def forbidden(*a,**k):counts['real_operations']+=1;raise AssertionError('Optimization/integration forbidden in preflight')
    with patch.object(ca,'nlpsol',nlp),patch.object(ct,'nlpsol',nlp),patch.object(ca,'integrator',integrator),patch.object(ct,'integrator',integrator),patch.object(ca.Opti,'solve',forbidden),patch.object(ca.Opti,'solve_limited',forbidden):yield counts

def static_graph_check(root,config):
    with zero_operations() as counts:
        model_fn,mpc_fn,sim_fn,_=p.native_templates(config);model=model_fn();ctrl=mpc_fn(model,silence_solver=True)
        ctrl.x0=np.array(config['cases'][0]['values']).reshape(4,1);ctrl.u0=np.array(config['previous_input']).reshape(2,1);ctrl.set_initial_guess()
        f=ctrl.nlp_obj;g=ctrl.nlp_cons;grad=ca.gradient(f,ctrl.opt_x.cat);lg=ca.SX.sym('lg',324);jlam=ca.jtimes(g,ctrl.opt_x.cat,lg,True)
        native=ca.Function('static_native_readout',[ctrl.opt_x,ctrl.opt_p,lg],[f,g,grad,jlam]);own=v.independent_program()
        max_error=0.
        for i in range(4):
            z=np.array(ctrl.opt_x_num.cat).reshape(-1);z+=np.sin(np.arange(376)+i)*.0001
            params=np.r_[config['cases'][i]['values'],config['previous_input']];multipliers=np.cos(np.arange(324)+i)*.01
            for a,b in zip(native(z,params,multipliers),own(z,params,multipliers)):
                error=float(np.max(abs(np.array(a)-np.array(b))));max_error=max(max_error,error)
                if error>config['graph_tolerance']:raise AssertionError('Independent NLP graph mismatch')
        expected=v.graph_bounds()
        for a,b in zip([ctrl._lb_opt_x,ctrl._ub_opt_x,ctrl.nlp_cons_lb,ctrl.nlp_cons_ub],expected):
            if p.array(a).reshape(-1).tobytes()!=b.tobytes():raise AssertionError('Independent bound indexing mismatch')
        if ctrl.settings.nlpsol_opts!={'ipopt.print_level':0,'ipopt.sb':'yes','print_time':0}:raise AssertionError('Frozen solver options differ')
        saved=p.memory(ctrl);ctrl.flags['initial_run']=True;ctrl.lam_x_num=ca.DM.ones(376);ctrl.reset_history()
        if not ctrl.flags['initial_run'] or not np.array_equal(p.array(ctrl.opt_x_num.cat).reshape(-1),saved['primal']):raise AssertionError('Native reset semantics changed')
        sim=sim_fn(model)
        if sim.settings.t_step!=.005 or sim.settings.abstol!=1e-10 or sim.settings.reltol!=1e-10:raise AssertionError('Native simulator settings differ')
        if counts['real_operations']:raise AssertionError('Real operation attempted')
    return {'passed':True,'native_independent_objective_constraint_gradient_Jlambda_max_difference':max_error,
        'source_Radau_equations_and_scaled_objective_checked':True,'all376_variable_bounds_and324_constraint_bounds_checked':True,
        'real_operations':0,'factory_intercepts':counts}

def preflight(root,out):
    root,out=Path(root),Path(out);config=p.read(root/p.CONFIG)
    if (out/'protocol_seal.json').exists():raise FileExistsError('Sealed preflight cannot be overwritten')
    identity=p.verify_identities(root,config);graph=static_graph_check(root,config);plan=p.jobs(config)
    if len(plan)!=40 or sum(x['solves'] for x in plan)!=816 or sum(x['intervals'] for x in plan)!=2400 or sum(x['monitors'] for x in plan)!=24000:raise AssertionError('Job allocation mismatch')
    for row in config['cases']:
        if np.asarray(row['values'],dtype='<f8').tobytes().hex()!=row['little_endian_bytes_hex']:raise AssertionError('Initial-state bytes changed')
    (out/'unit_tests.log').open('x').close()
    import unittest
    sys.path.insert(0,str(root/'tests'));suite=unittest.defaultTestLoader.loadTestsFromName('test_cstr_pilot')
    with zero_operations() as counts:
        with (out/'unit_tests.log').open('w') as stream:result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        if not result.wasSuccessful() or counts['real_operations']:raise AssertionError('Zero-solve unit tests failed')
    p.immutable(out/'protocol.json',config);p.immutable(out/'jobs.json',plan);p.immutable(out/'identity_checks.json',identity);p.immutable(out/'graph_checks.json',graph)
    p.immutable(out/'preflight.json',{'passed':True,'tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'new_solves':0,'new_integrations':0,
        'reset_independence':'fresh object factory tested; within-rollout native warm starts retained','monitoring':'10+20 independent replay grids, direct native endpoints only',
        'history_diagnostics':'C0..C3 at k0/k1; native prequery snapshot vs constant x/previous-u zero-dual cold initialization; no action application',
        'six_input_sufficiency_not_assumed':True})
    p.immutable(out/'protocol_seal.json',{'version':config['version'],'source_sha256':{n:p.sha(root/n) for n in p.SOURCES},
        'artifact_sha256':{n:p.sha(out/n) for n in ['authorization.txt','protocol.json','jobs.json','identity_checks.json','graph_checks.json','unit_tests.log','preflight.json','preservation_start.json']}})
    return p.read(out/'preflight.json')
