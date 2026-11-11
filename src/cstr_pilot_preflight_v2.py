"""Mandatory zero-solve qualification for the sole bound-relaxation change."""
from pathlib import Path
from unittest.mock import patch
import json,sys,unittest,hashlib,dataclasses
import numpy as np
import casadi as ca
from src import cstr_expert_pilot_v2 as p
from src.cstr_pilot_preflight import zero_operations
from src import cstr_verification as v


def graph_check(root,c):
    m,stock,_,_=p.ORIGINAL_TEMPLATES(c);_,revised,_,_=p.native_templates(c)
    with zero_operations() as counts:
        a,b=stock(m(),silence_solver=True),revised(m(),silence_solver=True)
        sa,sb=dataclasses.asdict(a.settings),dataclasses.asdict(b.settings)
        expected={**sa['nlpsol_opts'],'ipopt.bound_relax_factor':0.}
        if sb.pop('nlpsol_opts')!=expected:raise AssertionError('Additional solver setting change')
        sa.pop('nlpsol_opts')
        if sa!=sb:raise AssertionError('Additional MPC setting change')
        functions=[]
        for ctrl in [a,b]:
            ctrl.x0=np.asarray(c['cases'][0]['values']).reshape(4,1);ctrl.u0=np.asarray(c['previous_input']).reshape(2,1);ctrl.set_initial_guess()
            lg=ca.SX.sym('lambda_g',324)
            functions.append(ca.Function('graph_readout',[ctrl.opt_x,ctrl.opt_p,lg],[ctrl.nlp_obj,ctrl.nlp_cons,
                ca.gradient(ctrl.nlp_obj,ctrl.opt_x.cat),ca.jtimes(ctrl.nlp_cons,ctrl.opt_x.cat,lg,True)]))
        independent=v.independent_program();max_error=0.
        for i in range(4):
            z=np.array(a.opt_x_num.cat).reshape(-1)+np.sin(np.arange(376)+i)*.0001
            params=np.r_[c['cases'][i]['values'],c['previous_input']];lg=np.cos(np.arange(324)+i)*.01
            x,y,own=[f(z,params,lg) for f in [*functions,independent]]
            for xx,yy,zz in zip(x,y,own):
                if np.array(xx).tobytes()!=np.array(yy).tobytes():raise AssertionError('Revised mathematical graph changed')
                err=float(np.max(abs(np.array(xx)-np.array(zz))));max_error=max(max_error,err)
                if err>c['graph_tolerance']:raise AssertionError('Independent graph mismatch')
        for ctrl in [a,b]:
            for native,own in zip([ctrl._lb_opt_x,ctrl._ub_opt_x,ctrl.nlp_cons_lb,ctrl.nlp_cons_ub],v.graph_bounds()):
                if p.base.array(native).reshape(-1).tobytes()!=own.tobytes():raise AssertionError('Bounds changed')
        if counts['real_operations']:raise AssertionError('Real preflight operation')
    return {'passed':True,'native_revised_mathematical_graph_bitwise_equal':True,
        'independent_graph_max_error':max_error,'all_bounds_unchanged':True,'all_other_MPC_settings_unchanged':True,
        'only_option_difference':{'ipopt.bound_relax_factor':0.},'new_solves':0,'new_integrations':0}


def backend_check(root,out,c):
    import do_mpc
    from src.cstr_ipopt_options import inspect_options
    _,stock,_,_=p.ORIGINAL_TEMPLATES(c);m,revised,_,_=p.native_templates(c)
    def forbidden(*args,**kwargs):raise AssertionError('No optimizer evaluation during preflight')
    with patch.object(do_mpc.controller.MPC,'solve',forbidden),patch.object(do_mpc.controller.MPC,'make_step',forbidden):
        a,b=stock(m(),silence_solver=True),revised(m(),silence_solver=True)
        before=p.base.memory(b);serialized=b.S.serialize()
        original=inspect_options(a.S,out/'libcstr_option_inspector.dylib')
        observed=inspect_options(b.S,out/'libcstr_option_inspector.dylib')
        if observed!=p.expected_options():raise AssertionError('Actual revised backend options differ')
        if original!={**observed,'bound_relax_factor':1e-8}:raise AssertionError('Other backend options differ')
        if b.S.serialize()!=serialized:raise AssertionError('Inspector changed serialized backend')
        after=p.base.memory(b)
        for k in before:
            if before[k] is None:
                if after[k] is not None:raise AssertionError('Inspector changed memory')
            elif np.asarray(before[k]).tobytes()!=np.asarray(after[k]).tobytes():raise AssertionError('Inspector changed memory')
    return {'passed':True,'actual_OptionsList_getters':True,'original':original,'revised':observed,
        'controller_memory_and_serialized_backend_unchanged':True,'new_solves':0,'new_integrations':0,
        'limitation':'Inspector uses exact-version CasADi internal layout; validated only for the pinned binary/compiler/architecture.'}


def preflight(root,out):
    root,out=Path(root),Path(out);c=p.read(root/p.CONFIG)
    if (out/'protocol_seal.json').exists():raise FileExistsError('Preflight already sealed')
    old=p.read(root/'configs/cstr_expert_pilot_v1.json');compare=dict(c)
    assert compare.pop('controller_extra_ipopt_options')=={'bound_relax_factor':0.}
    compare['version']=old['version'];assert compare==old
    identity=p.verify_identities(root,c);graph=graph_check(root,c);backend=backend_check(root,out,c)
    plan=p.base.jobs(c);assert len(plan)==40
    assert tuple(sum(j[k] for j in plan) for k in ['solves','intervals','monitors'])==(816,2400,24000)
    sys.path.insert(0,str(root/'tests'))
    suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromName(n) for n in ['test_cstr_pilot','test_cstr_input_diagnosis','test_cstr_pilot_v2']])
    with zero_operations() as calls:
        with (out/'unit_tests.log').open('x') as stream:tests=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        if not tests.wasSuccessful() or calls['real_operations']:raise AssertionError('Zero-solve regression failure')
    for name,value in [('protocol.json',c),('identity_checks.json',identity),('graph_checks.json',graph),('effective_option_checks.json',backend),('jobs.json',plan)]:p.base.immutable(out/name,value)
    report={'passed':True,'tests':tests.testsRun,'failures':len(tests.failures),'errors':len(tests.errors),'new_solves':0,'new_integrations':0,
        'effective_option_observed_not_only_configured':True,'frozen_raw_and_numerical_acceptance_unchanged':True}
    p.base.immutable(out/'preflight.json',report)
    artifacts=['authorization.txt','protocol.json','identity_checks.json','graph_checks.json','effective_option_checks.json','jobs.json','preflight.json','unit_tests.log','preservation_start.json','inspector_build.json','inspection_headers_identity.json','inspector_probe.json','libcstr_option_inspector.dylib']
    p.base.immutable(out/'protocol_seal.json',{'version':c['version'],'source_sha256':{n:p.sha(root/n) for n in p.SOURCES},
        'artifact_sha256':{n:p.sha(out/n) for n in artifacts}})
    return report
