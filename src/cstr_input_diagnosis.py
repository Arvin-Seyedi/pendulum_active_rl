"""Archived CSTR input-bound diagnosis. No solver or integrator is executed."""
from pathlib import Path
from fractions import Fraction
from contextlib import ExitStack
from unittest.mock import patch
import json
import math
import re
import numpy as np
import casadi as ca
import casadi.tools as ct
from src import cstr_verification as v
from src.cstr_expert_pilot import check_seal, verify_identities, native_templates
from src.cstr_pilot_preflight import zero_operations
from src.cstr_pilot_review import blocked
from src.quadrotor_pilot import read, sha


def relaxation(lower, factor=1e-8, violation_tolerance=1e-4):
    """IPOPT 3.14.11 OrigIpoptNLP::relax_bounds scalar formula, not a solver."""
    return min(violation_tolerance, abs(factor)*max(abs(lower), 1.))


def diagnose(root, references, *, preservation=False):
    root, refs = Path(root).resolve(), Path(references).resolve()
    pilot = root/'results/cstr_expert_pilot/technical_v1'
    seal = read(pilot/'completion_integrity.json')
    for name, h in seal['artifact_sha256'].items():
        assert sha(pilot/name) == h, name
    for name, h in seal['source_sha256'].items():
        assert sha(root/name) == h, name
    c = check_seal(root, pilot)
    identity = verify_identities(root, c)
    source_ids = read(refs/'reference_source_identities.json')
    for name, row in source_ids.items():
        assert sha(refs/name) == row['sha256'], name
    orig = (refs/'IpOrigIpoptNLP.cpp').read_text()
    conv = (refs/'IpOptErrorConvCheck.cpp').read_text()
    data = (refs/'IpIpoptData.cpp').read_text()
    casadi_source = (refs/'ipopt_interface.cpp').read_text()
    nlp_source = (refs/'nlpsol.cpp').read_text()
    assert re.search(r'"bound_relax_factor",\s*"[^"]+",\s*0\., false,\s*1e-8,', orig)
    assert re.search(r'"honor_original_bounds",\s*"[^"]+",\s*false,', orig)
    assert re.search(r'"constr_viol_tol",\s*"[^"]+",\s*0\.0, true,\s*1e-4,', conv)
    assert re.search(r'"compl_inf_tol",\s*"[^"]+",\s*0\., true,\s*1e-4,', conv)
    assert re.search(r'"dual_inf_tol",\s*"[^"]+",\s*0\., true,\s*1\.,', conv)
    assert '#ifdef IPOPT_SINGLE\n      1e-5,\n#else\n      1e-8,' in data
    assert 'bound_consistency_ = false;' in nlp_source
    assert 'ElementWiseMax(*ones)' in orig and 'ElementWiseMin(*ones)' in orig
    assert 'bound_relax_factor' not in casadi_source and 'honor_original_bounds' not in casadi_source
    folder = pilot/'runs/C2_original_scientific';attempt = folder/'attempts/0000'
    a = dict(np.load(attempt/'candidate.npz')); submitted = dict(np.load(attempt/'submitted.npz'))
    for name, value in submitted.items():
        assert a[name].tobytes() == value.tobytes()
    native_stats = read(attempt/'native_stats.json')
    own = v.independent_program()
    metrics = v.candidate_metrics(a, own);v.require_candidate(metrics, native_stats, c)
    with zero_operations() as calls:
        model_fn, ctrl_fn, _, _ = native_templates(c)
        ctrl = ctrl_fn(model_fn(), silence_solver=True)
        assert ctrl.settings.nlpsol_opts == {'ipopt.print_level':0,'ipopt.sb':'yes','print_time':0}
        z = ctrl.opt_x(a['z'])
        extracted = np.array(z['_u',0,0]*ctrl._u_scaling).reshape(-1)
        labels = ctrl.opt_x.labels()
        assert 'F' in labels[336] and 'Q_dot' in labels[337]
        for name, value in [('lbx',ctrl._lb_opt_x),('ubx',ctrl._ub_opt_x)]:
            assert a[name].tobytes() == np.array(value.cat).reshape(-1).tobytes()
        native = ca.Function('archived_native_diagnosis', [ctrl.opt_x,ctrl.opt_p], [ctrl.nlp_obj,ctrl.nlp_cons])
        f, g = native(a['z'], a['p'])
        assert abs(float(f)-float(a['f'][0])) < 1e-10
        native_g = np.array(g).reshape(-1)
        assert np.max(abs(native_g-a['g'])) < 1e-10
        assert calls['real_operations'] == 0
    assert extracted.tobytes() == np.asarray(metrics['command']).tobytes()
    scaled = float(a['z'][336]);bound = float(a['lbx'][336]);command = float(extracted[0])
    assert bound == .05 and scaled < bound and command < 5.
    rel = relaxation(bound);relaxed = bound-rel
    margin_exact = Fraction.from_float(scaled)-Fraction.from_float(bound)
    multiplication_roundoff = Fraction.from_float(command)-100*Fraction.from_float(scaled)
    raw_violation = np.maximum.reduce([a['lbx']-a['z'],a['z']-a['ubx'],np.zeros(376)])
    idx = np.flatnonzero(raw_violation)
    eq = native_g.reshape(-1)
    traj = dict(np.load(folder/'trajectory.npz'))
    assert traj['states'].shape == (1,4) and traj['commands'].shape == (0,2)
    assert not (folder/'integrations').exists()
    assert not (attempt/'postquery_memory.npz').exists() and not (attempt/'native_timing.json').exists()
    ledger = read(pilot/'resource_ledger.json')
    assert ledger['halted'] and ledger['actual']['solves'] == 401
    cases={}
    for case in ['C0','C1']:
        folder0 = pilot/'runs'/f'{case}_original_scientific'
        d = dict(np.load(folder0/'trajectory.npz'))
        dense = dict(np.load(pilot/'runs'/f'{case}_original_independent_replay/monitoring_partial.npz'))
        v.require_state(d['states'], c);v.require_state(dense['refinement'], c)
        assert np.all(np.minimum(d['commands']-v.ULO, v.UHI-d['commands']) >= 0)
        cases[case]={'boundary_min_TR_margin_C':float(np.min(140-d['states'][:,2])),
            'dense_min_TR_margin_C':float(np.min(140-dense['refinement'][:,:,2])),
            'independent_max_abs_TR_disagreement_C':float(np.max(abs(dense['states'][:,2]-d['states'][:,2]))),
            'dense_min_normalized_reserve':float(np.min(v.state_margins(dense['refinement'])/v.WIDTH)-c['normalized_reserve'])}
    result={'passed_static_diagnosis':True,'new_solves':0,'new_integrations':0,'identity':identity,
        'accepted_pilot_seal_sha256':sha(pilot/'completion_integrity.json'),'reference_sources':source_ids,
        'source_default_options':{'tol':1e-8,'constr_viol_tol':1e-4,'dual_inf_tol':1.,'compl_inf_tol':1e-4,
            'acceptable_tol':1e-6,'bound_relax_factor':1e-8,'honor_original_bounds':False,'casadi_bound_consistency':False},
        'effective_option_dump_available':False,'current_cwd_ipopt_opt_present':(root/'ipopt.opt').exists(),
        'original_constraint_label':labels[336],'nlp_variable_index_zero_based':336,
        'physical_F_lower_bound':5.,'F_scaling':100.,'scaled_F_lower_bound':bound,'scaled_F_candidate':scaled,
        'original_scaled_F_margin':scaled-bound,'exact_dyadic_scaled_margin':str(margin_exact),
        'F_command':command,'F_command_hex':command.hex(),'Q_command':float(extracted[1]),
        'physical_F_margin':command-5.,'rescaling_roundoff_exact':str(multiplication_roundoff),
        'rescaling_roundoff_absolute':abs(float(multiplication_roundoff)),
        'physical_violation_in_ULPs_at_5':(5-command)/math.ulp(5.),
        'default_relaxation_scaled':rel,'default_relaxed_lower_bound_scaled':relaxed,
        'default_relaxed_lower_bound_physical':100*relaxed,'margin_to_default_relaxed_bound_scaled':scaled-relaxed,
        'fraction_of_default_relaxation_used':(bound-scaled)/rel,
        'first_F_multiplier_lam_x':float(a['lam_x'][336]),
        'bound_violations_original_NLP': [{'index':int(i),'label':labels[i],'scaled_violation':float(raw_violation[i])} for i in idx],
        'original_variable_bound_max_scaled_violation':float(raw_violation.max()),
        'original_equality_max_scaled_violation':float(np.max(abs(eq))),
        'source_nlp_objective':float(f),'source_vs_independent_constraints_max_error':metrics['constraint_reconstruction_error'],
        'candidate_metrics':metrics,'solver_status':native_stats['return_status'],'solver_iterations':native_stats['iter_count'],
        'native_iteration_final':{key:values[-1] for key,values in native_stats['iterations'].items()},
        'not_applied':True,'missing_postquery_memory_and_native_timing':True,'cases':cases,
        'mathematical_model_unchanged':True,'source_defaults_are_not_a_saved_effective_option_dump':True}
    if preservation:
        from src.quadrotor_archive_finalization import historical_preservation
        result['preservation']=historical_preservation(root,root,read(root/'results/benchmark_selection/b0_archive_v1/preservation_start.json'))
        extra={}
        for name, filename in [('b0_archive_v1','completion_integrity.json'),('b0_finalization','finalization_integrity.json'),
            ('b1_archive_v1','completion_integrity.json'),('b1_finalization','completion_integrity.json')]:
            base=root/'results/benchmark_selection'/name;old=read(base/filename)
            for n,h in old['artifact_sha256'].items():assert sha(base/n)==h,n
            extra[name]=len(old['artifact_sha256'])
        result['additional_preservation']=extra
    return result


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--references',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--full-preservation',action='store_true');args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise FileExistsError(output)
    with ExitStack() as stack:
        for module in [ca,ct]:
            stack.enter_context(patch.object(module,'nlpsol',blocked));stack.enter_context(patch.object(module,'integrator',blocked))
        stack.enter_context(patch.object(ca.Opti,'solve',blocked));stack.enter_context(patch.object(ca.Opti,'solve_limited',blocked))
        stack.enter_context(patch('scipy.integrate.solve_ivp',blocked))
        result=diagnose(Path.cwd(),args.references,preservation=args.full_preservation)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
