"""Source-preserving, separately versioned CSTR bound_relax_factor=0 pilot."""
from pathlib import Path
from contextlib import contextmanager,ExitStack
from unittest.mock import patch
import time,gc,json
import numpy as np
import casadi as ca
from src import cstr_expert_pilot as base
from src import cstr_verification as v
from src.cstr_ipopt_options import inspect_options
from src.quadrotor_pilot import read,sha

CONFIG='configs/cstr_expert_pilot_v2.json'
OUT='results/cstr_expert_pilot/no_bound_relax_v2'
SOURCES=[CONFIG,'src/cstr_expert_pilot_v2.py','src/cstr_pilot_preflight_v2.py',
 'src/cstr_ipopt_options.py','src/cstr_ipopt_option_inspector.cpp','experiments/cstr_expert_pilot_v2.py',
 'tests/test_cstr_pilot_v2.py','docs/cstr_expert_pilot_v2_protocol.md']
ORIGINAL_TEMPLATES=base.native_templates
ORIGINAL_SCIENTIFIC=base.scientific
ORIGINAL_REPLAY=base.replay
ORIGINAL_OBSERVER=base.SolverObserver
ORIGINAL_SEAL=base.check_seal
ORIGINAL_VERIFY=base.verify_identities


def expected_options():
    return {'bound_relax_factor':0.,'tol':1e-8,'constr_viol_tol':1e-4,
     'dual_inf_tol':1.,'compl_inf_tol':1e-4,'acceptable_tol':1e-6,'honor_original_bounds':0.}


def native_templates(config):
    model,mpc,sim,feedback=ORIGINAL_TEMPLATES(config)
    import do_mpc
    original_setup=do_mpc.controller.MPC.setup
    def revised(model,silence_solver=False):
        if not silence_solver:raise AssertionError('Frozen display configuration')
        def setup(controller):
            native={'ipopt.print_level':0,'ipopt.sb':'yes','print_time':0}
            if controller.settings.nlpsol_opts!=native:raise AssertionError('Unexpected native solver options')
            controller.settings.nlpsol_opts={**native,'ipopt.bound_relax_factor':0.}
            return original_setup(controller)
        with patch.object(do_mpc.controller.MPC,'setup',setup):
            return mpc(model,silence_solver=True)
    return model,revised,sim,feedback


def verify_history(root):
    root=Path(root)
    for folder in ['results/cstr_expert_pilot/technical_v1','results/cstr_numerical_diagnosis/zero_solve_v1']:
        b=root/folder;s=read(b/'completion_integrity.json')
        for n,h in s['artifact_sha256'].items():assert sha(b/n)==h,n
        for n,h in s['source_sha256'].items():assert sha(root/n)==h,n
    snap=read(root/OUT/'preservation_start.json')
    for group in ['tracked_sha256','untracked_sha256']:
        for n,h in snap[group].items():assert sha(root/n)==h,n
    return {'passed':True,'prior_tracked_files':len(snap['tracked_sha256']),
        'prior_untracked_files':len(snap['untracked_sha256']),'accepted_CSTR_pilot_and_diagnosis_unchanged':True}


def verify_identities(root,config):
    result=ORIGINAL_VERIFY(root,config)
    verify_history(root)
    if (Path.cwd()/'ipopt.opt').exists():raise AssertionError('Unsealed IPOPT options file')
    return result


def check_seal(root,out):
    config=ORIGINAL_SEAL(root,out)
    if config['version']!='CSTR_EXPERT_NO_BOUND_RELAX_v2':raise AssertionError('Wrong protocol')
    inspector=read(Path(out)/'inspector_build.json')
    if sha(Path(out)/'libcstr_option_inspector.dylib')!=inspector['library_sha256']:raise AssertionError('Inspector identity changed')
    return config


class ObservedBackend:
    def __init__(self,backend,owner):self.raw=backend;self.owner=owner
    def stats(self):return self.raw.stats()
    def __call__(self,**kwargs):
        f=self.owner.folder/'attempts'/f'{self.owner.index-1:04d}'
        observed=inspect_options(self.raw,Path.cwd()/OUT/'libcstr_option_inspector.dylib')
        base.safe_write(f/'effective_ipopt_options.json',{'method':'actual IPOPT OptionsList getters on checked-out CasADi backend memory',
            'observed':observed,'passed':observed==expected_options()})
        if observed!=expected_options():raise AssertionError('Effective IPOPT options mismatch')
        return self.raw(**kwargs)


class SolverObserver(ORIGINAL_OBSERVER):
    def __init__(self,backend,*args):
        super().__init__(backend,*args);self.backend=ObservedBackend(backend,self)
    def __call__(self,**kwargs):
        try:return super().__call__(**kwargs)
        finally:
            f=self.folder/'attempts'/f'{self.index-1:04d}'
            if (f/'candidate.npz').exists():
                a=np.load(f/'candidate.npz');z=a['z'];physical=z[336:].reshape(20,2)*v.SU
                np.savez_compressed(f/'bound_residuals.npz',scaled_lower=z-a['lbx'],scaled_upper=a['ubx']-z,
                    physical_predicted_commands=physical,input_lower=physical-v.ULO,input_upper=v.UHI-physical)


def query(ctrl,x,folder,k,observer):
    """Persist all available command/timing/memory before any raw-command check."""
    start=time.perf_counter();returned=False;command=None
    f=Path(folder)/'attempts'/f'{k:04d}'
    try:
        command=base.array(ctrl.make_step(np.asarray(x).reshape(4,1))).reshape(-1);returned=True
    finally:
        f.mkdir(parents=True,exist_ok=True)
        base.safe_write(f/'native_timing.json',{'native_action_selection_including_verification_wall':time.perf_counter()-start,
            'native_query_returned':returned,'command_available':command is not None})
        base.save_memory(f/('postquery_memory.npz' if returned else 'post_attempt_memory.npz'),base.memory(ctrl))
        if command is not None:
            np.savez_compressed(f/'returned_command.npz',command=command,lower_margin=command-v.ULO,upper_margin=v.UHI-command)
    return command


def record_margins(folder,states,commands):
    x=np.asarray(states).reshape(-1,4);u=np.asarray(commands).reshape(-1,2)
    np.savez_compressed(Path(folder)/'raw_margins.npz',state_lower=x-v.LOW,state_upper=v.HIGH-x,
        input_lower=u-v.ULO,input_upper=v.UHI-u)


def scientific(root,out,config,job,account,limits):
    folder=Path(out)/'runs'/job['id'];folder.mkdir(parents=True,exist_ok=False)
    initial=next(x['values'] for x in config['cases'] if x['id']==job['case'])
    states=[np.array(initial)];commands=[];times=[];observer=None
    base.persist_trajectory(folder,states,commands,times,[]);record_margins(folder,states,commands)
    ctrl,sim,feedback=base.initialize_objects(config,initial)
    if base.array(ctrl.x0.cat).reshape(-1).tobytes()!=states[0].tobytes() or base.array(sim.x0.cat).reshape(-1).tobytes()!=states[0].tobytes():raise AssertionError('Exact reset failure')
    if ctrl.flags['initial_run']:raise AssertionError('Fresh controller has optimizer history')
    sim.simulator=base.IntegratorObserver(sim.simulator,sim,folder,account,limits)
    observer=SolverObserver(ctrl.S,ctrl,folder,config,account,limits);ctrl.S=observer
    for k in range(config['transitions']):
        limits();v.require_state(states[-1],config);before=base.memory(ctrl);native_t=float(np.asarray(ctrl.t0).reshape(-1)[0])
        if abs(native_t-k*config['dt'])>1e-14:raise AssertionError('Native time/index mismatch')
        try:command=query(ctrl,states[-1],folder,k,observer)
        finally:base.persist_trajectory(folder,states,commands,times,observer.metrics)
        if observer.latest['p'][:4].tobytes()!=states[-1].tobytes() or observer.latest['p'][4:].tobytes()!=before['previous_u'].tobytes():raise AssertionError('Decision context mismatch')
        if command.tobytes()!=np.asarray(observer.metrics[-1]['command']).tobytes():raise AssertionError('Returned command/extraction mismatch')
        v.require_input(command)
        measurement=sim.make_step(command.reshape(2,1));next_x=base.array(feedback.make_step(measurement)).reshape(-1)
        if next_x.tobytes()!=base.array(sim.x0.cat).reshape(-1).tobytes():raise AssertionError('State-feedback mismatch')
        commands.append(command);states.append(next_x);times.append(native_t)
        base.persist_trajectory(folder,states,commands,times,observer.metrics);record_margins(folder,states,commands)
        v.require_state(next_x,config);limits()
    result={'complete':True,'mode':'scientific','case':job['case'],'phase':job['phase'],'intervals':100,'candidates_checked':len(observer.metrics),
        'costs':v.costs(states,commands,config['previous_input']),'minimum_raw_state_margins':v.state_margins(states).min(0).tolist(),
        'minimum_normalized_margin':float(np.min(v.state_margins(states)/v.WIDTH)),
        'minimum_raw_input_margins':np.minimum(np.asarray(commands)-v.ULO,v.UHI-np.asarray(commands)).min(0).tolist(),
        'only_controller_option_change':{'ipopt.bound_relax_factor':0.},'state_sample_provenance':'native control boundaries; independent dense reconstruction in replay'}
    if job['phase']=='repeat':
        original=Path(out)/'runs'/f"{job['case']}_original_scientific"
        base.compare_npz(original/'trajectory.npz',folder/'trajectory.npz')
        for k in range(100):
            for name in ['submitted.npz','candidate.npz','prequery_memory.npz','postquery_memory.npz','returned_command.npz','bound_residuals.npz']:
                base.compare_npz(original/'attempts'/f'{k:04d}'/name,folder/'attempts'/f'{k:04d}'/name)
            a,b=[read(f/'attempts'/f'{k:04d}'/'native_stats.json') for f in [original,folder]]
            if any(a.get(key)!=b.get(key) for key in ['success','return_status','iter_count']):raise AssertionError('Deterministic solver status mismatch')
        result['deterministic_scientific_repeat_bitwise']=True
    base.persist_trajectory(folder,states,commands,times,observer.metrics,True)
    base.immutable(folder/'completion.json',result);del ctrl,sim,feedback,observer;gc.collect();return result


def replay(root,out,config,job,account,limits):
    folder=Path(out)/'runs'/job['id']
    try:return ORIGINAL_REPLAY(root,out,config,job,account,limits)
    finally:
        if (folder/'trajectory.npz').exists():
            a=np.load(folder/'trajectory.npz');record_margins(folder,a['states'],a['commands'])
        if (folder/'monitoring_partial.npz').exists():
            a=np.load(folder/'monitoring_partial.npz');x=a['refinement']
            np.savez_compressed(folder/'dense_raw_margins.npz',lower=x-v.LOW,upper=v.HIGH-x)


def diagnostic(root,out,config,job,account,limits):
    folder=Path(out)/'runs'/job['id'];folder.mkdir(parents=True,exist_ok=False)
    source=Path(out)/'runs'/f"{job['case']}_original_scientific"/'attempts'/f"{job['step']:04d}"
    saved=np.load(source/'prequery_memory.npz');baseline=np.load(source/'candidate.npz');params=baseline['p'];x=params[:4];previous=params[4:]
    model_fn,mpc_fn,_,_=native_templates(config);ctrl=mpc_fn(model_fn(),silence_solver=True)
    ctrl.x0=x.reshape(4,1);ctrl.u0=previous.reshape(2,1);ctrl.t0=job['step']*config['dt'];ctrl.set_initial_guess()
    if job['initialization']=='native_prequery_primal_duals':
        ctrl.x0=saved['native_x0'].reshape(4,1);ctrl.t0=saved['native_time'];ctrl.u0=saved['previous_u'].reshape(2,1)
        ctrl.opt_x_num.master=ca.DM(saved['primal']);ctrl.flags['initial_run']=bool(saved['initial_run'])
        for name in ['lam_x_num','lam_g_num']:
            if name in saved:setattr(ctrl,name,ca.DM(saved[name]))
    else:
        ctrl.flags['initial_run']=False;ctrl.lam_x_num=ca.DM.zeros(376);ctrl.lam_g_num=ca.DM.zeros(324)
    observer=SolverObserver(ctrl.S,ctrl,folder,config,account,limits);ctrl.S=observer
    command=query(ctrl,x,folder,0,observer);v.require_input(command)
    if observer.latest['p'].tobytes()!=params.tobytes():raise AssertionError('Matched OCP parameters differ')
    if job['initialization']=='native_prequery_primal_duals':
        for name in ['z','p','lam_x','lam_g','f','g']:
            if observer.latest[name].tobytes()!=baseline[name].tobytes():raise AssertionError('Native-history initialization failed bitwise reproduction')
    delta=command-baseline['z'][336:338]*v.SU
    result={'complete':True,'case':job['case'],'step':job['step'],'initialization':job['initialization'],
        'command':command.tolist(),'difference_from_original_command':delta.tolist(),
        'normalized_action_difference_inf':float(np.max(abs(delta)/v.UWIDTH)),'metrics':observer.metrics,
        'six_input_interpretation':'finite-context sensitivity; equality is not proof of global input sufficiency'}
    base.immutable(folder/'completion.json',result);del ctrl,observer;gc.collect();return result


@contextmanager
def bindings():
    """Version-scoped reuse of immutable v1 dispatch/accounting/integration code."""
    with ExitStack() as stack:
        for name,value in [('native_templates',native_templates),('SolverObserver',SolverObserver),
            ('scientific',scientific),('diagnostic',diagnostic),('replay',replay),('check_seal',check_seal),('verify_identities',verify_identities)]:
            stack.enter_context(patch.object(base,name,value))
        yield


def run(root,out):
    with bindings():return base.run(root,out)
