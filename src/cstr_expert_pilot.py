"""Bounded native CSTR expert pilot. No learning or generic acquisition entrypoint."""
from pathlib import Path
import hashlib,json,sys,time,resource,threading,os,gc,subprocess
from unittest.mock import patch
import numpy as np
import casadi as ca
import casadi.tools as ct
from src.quadrotor_pilot import sha,read
from src.quadrotor_mmaq_v1 import safe_write
from src import cstr_verification as v

CONFIG='configs/cstr_expert_pilot_v1.json'
SOURCES=[CONFIG,'src/cstr_verification.py','src/cstr_expert_pilot.py','src/cstr_pilot_preflight.py',
    'experiments/cstr_expert_pilot.py','tests/test_cstr_pilot.py','docs/cstr_expert_pilot_protocol.md']

def immutable(path,value):
    path=Path(path)
    if path.exists():raise FileExistsError(path)
    safe_write(path,value)

def array(x):return np.asarray(x.cat if hasattr(x,'cat') else x,dtype=float).copy()

def native_templates(config):
    up=Path(config['upstream']);sys.path.insert(0,str(up));sys.path.insert(0,str(up/'examples/CSTR_approximate_mpc'))
    import do_mpc
    from template_model import template_model
    from template_mpc import template_mpc
    from template_simulator import template_simulator
    assert Path(do_mpc.__file__).resolve()==up/'do_mpc/__init__.py'
    return template_model,template_mpc,template_simulator,do_mpc.estimator.StateFeedback

def verify_identities(root,config):
    import importlib.metadata as md
    import platform
    root=Path(root);audit=root/config['audit_root'];up=Path(config['upstream'])
    assert sha(audit/'completion_integrity.json')==config['audit_completion_sha256']
    seal=read(audit/'completion_integrity.json')
    for n,h in seal['artifact_sha256'].items():assert sha(audit/n)==h,n
    source=read(audit/'checks_v2/source_identity.json');runtime=read(audit/'checks_v2/environment_identity.json')
    for n,h in source['files'].items():assert sha(up/n)==h,n
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=up,text=True).strip()==config['upstream_commit']
    assert not subprocess.check_output(['git','status','--porcelain'],cwd=up,text=True).strip()
    assert sys.version==runtime['python'] and sha(sys.executable)==runtime['executable_sha256']
    assert platform.platform()==read(audit/'audit_runtime.json')['platform']
    actual={d.metadata['Name']:d.version for d in md.distributions()};assert actual==runtime['packages']
    records=read(audit/'all_distribution_RECORD_identities.json')
    for n,h in records.items():
        d=md.distribution(n);f=next(f for f in d.files if str(f).endswith('.dist-info/RECORD'));assert sha(d.locate_file(f))==h,n
    native=Path(md.distribution('casadi').locate_file('casadi'))
    for n,h in read(audit/'additional_checks.json')['native_binary_sha256'].items():assert sha(native/n)==h,n
    assert sha(audit/'prospective_pilot_manifest.json')==config['pilot_manifest_sha256']
    assert sha(audit/'prospective_numerical_contract.json')==config['numerical_contract_sha256']
    for row,old in zip(config['cases'],read(audit/'prospective_pilot_manifest.json')['cases']):
        assert row==old and np.asarray(row['values'],dtype='<f8').tobytes().hex()==row['little_endian_bytes_hex']
    return {'passed':True,'source_files':len(source['files']),'package_identities':len(actual),'source_commit':config['upstream_commit']}

def check_seal(root,out):
    root,out=Path(root),Path(out);s=read(out/'protocol_seal.json')
    for n,h in s['source_sha256'].items():assert sha(root/n)==h,n
    for n,h in s['artifact_sha256'].items():assert sha(out/n)==h,n
    assert read(out/'preflight.json')['passed']
    return read(out/'protocol.json')

class Limits:
    def __init__(self,out):
        self.out=Path(out);x=read(self.out/'resource_ledger.json');self.base=x['charged_cumulative'];self.cap=x['caps'];self.wall=time.monotonic()
    def __call__(self):
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss>4294967296:raise MemoryError('4 GiB RSS cap')
        if self.base['cpu_seconds']+time.process_time()>self.cap['cpu_seconds']-2:raise TimeoutError('Aggregate pilot CPU ceiling')
        if self.base['wall_seconds']+time.monotonic()-self.wall>self.cap['wall_seconds']-2:raise TimeoutError('Aggregate pilot wall ceiling')

def jobs(config):
    result=[]
    for case in config['cases']:
        for phase in config['phases']:
            for mode in ['scientific','native_replay','independent_replay']:
                result.append({'id':case['id']+'_'+phase+'_'+mode,'case':case['id'],'phase':phase,'mode':mode,
                    'solves':100 if mode=='scientific' else 0,'intervals':100,'monitors':3000 if mode=='independent_replay' else 0})
    for case in config['cases']:
        for k in config['context_steps']:
            for init in config['context_initializations']:
                result.append({'id':f"{case['id']}_k{k}_{init}",'case':case['id'],'step':k,'initialization':init,'mode':'context','solves':1,'intervals':0,'monitors':0})
    return result

class Accounting:
    def __init__(self,out):self.out=Path(out)
    def reserve(self,job):
        x=read(self.out/'resource_ledger.json')
        if x['halted']:raise RuntimeError('Pilot halted; no retries')
        if any(j['id']==job['id'] for j in x['jobs']):raise RuntimeError('Job already reserved; no restart')
        for k in ['solves','intervals','monitors']:
            if x['reserved'][k]+job[k]>x['limits'][k]:raise RuntimeError('Reservation would exceed '+k)
            x['reserved'][k]+=job[k]
        x['jobs'].append({**job,'status':'reserved'});safe_write(self.out/'resource_ledger.json',x)
    def charge(self,name,n=1):
        x=read(self.out/'resource_ledger.json');x['actual'][name]+=n
        if x['actual'][name]>x['reserved'][name] or x['actual'][name]>x['limits'][name]:raise RuntimeError('Actual resource overrun')
        safe_write(self.out/'resource_ledger.json',x)
    def finish(self,job,status,failure=None):
        x=read(self.out/'resource_ledger.json');entry=next(j for j in x['jobs'] if j['id']==job['id']);entry.update(status=status,failure=failure)
        if status!='complete':x['halted']=True;x['status']='STOPPED_FIRST_ACCEPTANCE_FAILURE'
        safe_write(self.out/'resource_ledger.json',x)

def memory(ctrl):
    result={'primal':array(ctrl.opt_x_num.cat).reshape(-1),'native_x0':array(ctrl.x0.cat).reshape(-1),
        'previous_u':array(ctrl.u0.cat).reshape(-1),'native_time':array(ctrl.t0).reshape(-1),'initial_run':bool(ctrl.flags['initial_run'])}
    for name in ['lam_x_num','lam_g_num']:
        value=getattr(ctrl,name,None)
        result[name]=None if value is None else array(value).reshape(-1)
    return result

def save_memory(path,value):
    data={k:np.asarray(v) for k,v in value.items() if v is not None};data['missing_fields']=np.array([k for k,v in value.items() if v is None]);np.savez_compressed(path,**data)

class SolverObserver:
    """Copies exact submitted kwargs and available result before native mutation."""
    def __init__(self,backend,ctrl,folder,config,account,limits):
        self.backend=backend;self.ctrl=ctrl;self.folder=Path(folder);self.config=config;self.account=account;self.limits=limits
        self.program=v.independent_program();self.index=0;self.metrics=[];self.latest=None
    def stats(self):return self.backend.stats()
    def __call__(self,**kwargs):
        self.limits();k=self.index;self.index+=1;attempt=self.folder/'attempts'/f'{k:04d}';attempt.mkdir(parents=True,exist_ok=False)
        inputs={n:array(a).reshape(-1) for n,a in kwargs.items()};np.savez_compressed(attempt/'submitted.npz',**inputs)
        save_memory(attempt/'prequery_memory.npz',memory(self.ctrl))
        self.account.charge('solves');start=time.perf_counter();cpu=time.process_time()
        try:r=self.backend(**kwargs)
        except BaseException as exc:
            safe_write(attempt/'exception.json',{'kind':type(exc).__name__,'message':str(exc),'candidate_available':False,
                'elapsed_solver_wall':time.perf_counter()-start,'elapsed_solver_CPU':time.process_time()-cpu});raise
        latency={'solver_wall':time.perf_counter()-start,'solver_CPU':time.process_time()-cpu}
        stats=self.backend.stats();candidate={**inputs,**{'z':array(r['x']).reshape(-1),'p':inputs['p'],
            'lam_x':array(r['lam_x']).reshape(-1),'lam_g':array(r['lam_g']).reshape(-1),'f':array(r['f']).reshape(-1),'g':array(r['g']).reshape(-1)}}
        np.savez_compressed(attempt/'candidate.npz',**candidate);safe_write(attempt/'native_stats.json',stats);safe_write(attempt/'solver_timing.json',latency)
        checkstart=time.perf_counter();metrics=v.candidate_metrics(candidate,self.program);metrics['verification_wall']=time.perf_counter()-checkstart
        safe_write(attempt/'independent_metrics.json',metrics);self.latest=candidate;self.metrics.append({**metrics,**latency,'iterations':stats.get('iter_count'),'return_status':stats.get('return_status')})
        v.require_candidate(metrics,stats,self.config);self.limits();return r

class IntegratorObserver:
    def __init__(self,backend,sim,folder,account,limits):self.backend=backend;self.sim=sim;self.folder=Path(folder);self.account=account;self.limits=limits;self.index=0
    def __call__(self,**kwargs):
        self.limits();i=self.index;self.index+=1;f=self.folder/'integrations';f.mkdir(exist_ok=True)
        packet={n:array(a).reshape(-1) for n,a in kwargs.items()};np.savez_compressed(f/f'{i:04d}_submitted.npz',**packet)
        current=array(self.sim.sim_p_num['_u']).reshape(-1);v.require_input(current)
        if not np.array_equal(packet['p'][:2],current):raise AssertionError('Current-step integrator command mismatch')
        self.account.charge('intervals');start=time.perf_counter()
        try:r=self.backend(**kwargs)
        except BaseException as exc:safe_write(f/f'{i:04d}_exception.json',{'kind':type(exc).__name__,'message':str(exc)});raise
        np.savez_compressed(f/f'{i:04d}_result.npz',**{n:array(a) for n,a in r.items()});safe_write(f/f'{i:04d}_timing.json',{'native_integration_wall':time.perf_counter()-start})
        self.limits();return r

def initialize_objects(config,initial,need_controller=True):
    model_fn,mpc_fn,sim_fn,feedback=native_templates(config);model=model_fn();sim=sim_fn(model);sim.x0=np.array(initial).reshape(4,1);sim.set_initial_guess()
    ctrl=None
    if need_controller:
        ctrl=mpc_fn(model,silence_solver=config['silence_solver']);ctrl.x0=np.array(initial).reshape(4,1);ctrl.u0=np.array(config['previous_input']).reshape(2,1);ctrl.set_initial_guess()
    return ctrl,sim,feedback(model)

def persist_trajectory(folder,states,commands,times,metrics,complete=False):
    np.savez_compressed(Path(folder)/'trajectory.npz',states=np.asarray(states).reshape(-1,4),commands=np.asarray(commands).reshape(-1,2),native_times=np.asarray(times))
    safe_write(Path(folder)/'progress.json',{'complete':complete,'completed_intervals':len(commands),'candidate_checks':len(metrics),'metrics':metrics})

def compare_npz(left,right,keys=None):
    a=np.load(left);b=np.load(right)
    if keys is None:
        if set(a.files)!=set(b.files):raise AssertionError('Replay/repeat field count mismatch')
        keys=a.files
    for key in keys:
        x,y=a[key],b[key]
        if x.shape!=y.shape or x.dtype!=y.dtype or x.tobytes()!=y.tobytes():raise AssertionError('Bitwise replay/repeat failed: '+key)

def scientific(root,out,config,job,account,limits):
    folder=Path(out)/'runs'/job['id'];folder.mkdir(parents=True,exist_ok=False)
    initial=next(x['values'] for x in config['cases'] if x['id']==job['case']);states=[np.array(initial)];commands=[];times=[];observer=None
    persist_trajectory(folder,states,commands,times,[])
    ctrl,sim,feedback=initialize_objects(config,initial)
    if array(ctrl.x0.cat).reshape(-1).tobytes()!=states[0].tobytes() or array(sim.x0.cat).reshape(-1).tobytes()!=states[0].tobytes():raise AssertionError('Exact reset failure')
    if ctrl.flags['initial_run']:raise AssertionError('Fresh controller has prior optimization state')
    sim.simulator=IntegratorObserver(sim.simulator,sim,folder,account,limits)
    observer=SolverObserver(ctrl.S,ctrl,folder,config,account,limits);ctrl.S=observer
    for k in range(config['transitions']):
        limits();v.require_state(states[-1],config);before=memory(ctrl);native_t=float(np.asarray(ctrl.t0).reshape(-1)[0])
        if abs(native_t-k*config['dt'])>1e-14:raise AssertionError('Native time/index mismatch')
        start=time.perf_counter();command=array(ctrl.make_step(states[-1].reshape(4,1))).reshape(-1);native_latency=time.perf_counter()-start
        if observer.latest['p'][:4].tobytes()!=states[-1].tobytes() or observer.latest['p'][4:].tobytes()!=before['previous_u'].tobytes():raise AssertionError('Decision parameter context mismatch')
        if command.tobytes()!=np.asarray(observer.metrics[-1]['command']).tobytes():raise AssertionError('Returned physical action differs from verified first action')
        v.require_input(command);save_memory(folder/'attempts'/f'{k:04d}'/'postquery_memory.npz',memory(ctrl))
        safe_write(folder/'attempts'/f'{k:04d}'/'native_timing.json',{'native_action_selection_including_verification_wall':native_latency})
        measurement=sim.make_step(command.reshape(2,1));next_x=array(feedback.make_step(measurement)).reshape(-1)
        if next_x.tobytes()!=array(sim.x0.cat).reshape(-1).tobytes():raise AssertionError('StateFeedback/reset observation mismatch')
        commands.append(command);states.append(next_x);times.append(native_t);persist_trajectory(folder,states,commands,times,observer.metrics)
        v.require_state(next_x,config);limits()
    result={'complete':True,'mode':'scientific','case':job['case'],'phase':job['phase'],'intervals':len(commands),'candidates_checked':len(observer.metrics),
        'costs':v.costs(states,commands,config['previous_input']),'minimum_raw_state_margins':v.state_margins(states).min(0).tolist(),
        'minimum_normalized_margin':float(np.min(v.state_margins(states)/v.WIDTH)),
        'minimum_raw_input_margins':np.minimum(np.asarray(commands)-v.ULO,v.UHI-np.asarray(commands)).min(0).tolist(),
        'state_sample_provenance':'101 native control-boundary states; dense independent numerical samples verified in replay','source_controller_unchanged':True}
    if job['phase']=='repeat':
        previous=Path(out)/'runs'/f"{job['case']}_original_scientific"
        compare_npz(previous/'trajectory.npz',folder/'trajectory.npz')
        for k in range(100):
            for name in ['submitted.npz','candidate.npz','prequery_memory.npz','postquery_memory.npz']:compare_npz(previous/'attempts'/f'{k:04d}'/name,folder/'attempts'/f'{k:04d}'/name)
            a,b=read(previous/'attempts'/f'{k:04d}'/'native_stats.json'),read(folder/'attempts'/f'{k:04d}'/'native_stats.json')
            for key in ['success','return_status','iter_count']:
                if a.get(key)!=b.get(key):raise AssertionError('Deterministic solver status/iteration mismatch')
        result['deterministic_scientific_repeat_bitwise']=True
    persist_trajectory(folder,states,commands,times,observer.metrics,True);immutable(folder/'completion.json',result)
    del ctrl,sim,feedback,observer;gc.collect();return result

def replay(root,out,config,job,account,limits):
    from scipy.integrate import solve_ivp
    folder=Path(out)/'runs'/job['id'];folder.mkdir(parents=True,exist_ok=False);original=Path(out)/'runs'/f"{job['case']}_{job['phase']}_scientific"
    a=np.load(original/'trajectory.npz');states=[a['states'][0].copy()];commands=a['commands'];dense=[];refined=[];max_error=0.
    if job['mode']=='native_replay':
        _,sim,feedback=initialize_objects(config,states[0],False);sim.simulator=IntegratorObserver(sim.simulator,sim,folder,account,limits)
        for k,u in enumerate(commands):
            limits();v.require_input(u);states.append(array(feedback.make_step(sim.make_step(u.reshape(2,1)))).reshape(-1));persist_trajectory(folder,states,commands[:k+1],a['native_times'][:k+1],[]);v.require_state(states[-1],config)
        if np.asarray(states).tobytes()!=a['states'].tobytes():raise AssertionError('Native action replay differs bitwise')
        result={'complete':True,'mode':job['mode'],'case':job['case'],'phase':job['phase'],'intervals':100,'native_replay_bitwise':True}
    else:
        for k,u in enumerate(commands):
            limits();v.require_input(u);account.charge('intervals');account.charge('monitors',30)
            sol=solve_ivp(lambda t,x:v.rhs(t,x,u),(0.,config['dt']),states[-1],method='DOP853',rtol=config['independent_rtol'],atol=config['independent_atol'],dense_output=True)
            safe_write(folder/f'integration_{k:04d}.json',{'success':sol.success,'message':sol.message,'nfev':sol.nfev,'accepted_steps':len(sol.t)-1})
            if not sol.success:raise AssertionError('Independent integrator failed')
            primary=sol.sol(np.linspace(0,config['dt'],11)).T;refine=sol.sol(np.linspace(0,config['dt'],21)).T
            dense.append(primary);refined.append(refine);states.append(primary[-1]);np.savez_compressed(folder/'monitoring_partial.npz',primary=np.array(dense),refinement=np.array(refined),states=np.array(states))
            persist_trajectory(folder,states,commands[:k+1],a['native_times'][:k+1],[]);v.require_state(primary,config);v.require_state(refine,config)
            error=float(np.max(abs(states[-1]-a['states'][k+1])/v.WIDTH));max_error=max(max_error,error)
            if error>config['normalized_replay_tolerance']:raise AssertionError('Independent/native integration disagreement')
            if np.max(abs(primary-refine[::2])/v.WIDTH)>config['normalized_replay_tolerance']:raise AssertionError('Monitoring-grid discrepancy')
        result={'complete':True,'mode':job['mode'],'case':job['case'],'phase':job['phase'],'intervals':100,'normalized_max_native_disagreement':max_error,
            'minimum_raw_dense_margins':v.state_margins(np.array(refined)).min(axis=(0,1)).tolist(),
            'minimum_dense_normalized_reserve':float(np.min(v.state_margins(np.array(refined))/v.WIDTH)-config['normalized_reserve']),
            'primary_monitor_subintervals':1000,'refinement_monitor_subintervals':2000,'unique_dense_subintervals':2000,
            'monitor_provenance':'independent DOP853 numerical reconstruction, not native CVODES internal states',
            'costs':v.costs(states,commands,config['previous_input'])}
    immutable(folder/'completion.json',result);return result

def diagnostic(root,out,config,job,account,limits):
    folder=Path(out)/'runs'/job['id'];folder.mkdir(parents=True,exist_ok=False)
    source=Path(out)/'runs'/f"{job['case']}_original_scientific"/'attempts'/f"{job['step']:04d}"
    saved=np.load(source/'prequery_memory.npz');baseline=np.load(source/'candidate.npz');params=baseline['p'];x=params[:4];previous=params[4:]
    model_fn,mpc_fn,_,_=native_templates(config);ctrl=mpc_fn(model_fn(),silence_solver=config['silence_solver']);ctrl.x0=x.reshape(4,1);ctrl.u0=previous.reshape(2,1);ctrl.t0=job['step']*config['dt'];ctrl.set_initial_guess()
    if job['initialization']=='native_prequery_primal_duals':
        ctrl.x0=saved['native_x0'].reshape(4,1);ctrl.t0=saved['native_time'];ctrl.u0=saved['previous_u'].reshape(2,1)
        ctrl.opt_x_num.master=ca.DM(saved['primal']);ctrl.flags['initial_run']=bool(saved['initial_run'])
        for name in ['lam_x_num','lam_g_num']:
            if name in saved:setattr(ctrl,name,ca.DM(saved[name]))
    else:
        ctrl.flags['initial_run']=False;ctrl.lam_x_num=ca.DM.zeros(376);ctrl.lam_g_num=ca.DM.zeros(324)
    observer=SolverObserver(ctrl.S,ctrl,folder,config,account,limits);ctrl.S=observer;command=array(ctrl.make_step(x.reshape(4,1))).reshape(-1);v.require_input(command)
    if observer.latest['p'].tobytes()!=params.tobytes():raise AssertionError('Matched diagnostic OCP parameters differ')
    delta=command-baseline['z'][336:338]*v.SU
    if job['initialization']=='native_prequery_primal_duals':
        for name in ['z','p','lam_x','lam_g','f','g']:
            if observer.latest[name].tobytes()!=baseline[name].tobytes():raise AssertionError('Same-context native initialization did not reproduce candidate')
    result={'complete':True,'case':job['case'],'step':job['step'],'initialization':job['initialization'],'command':command.tolist(),
        'difference_from_original_command':delta.tolist(),'normalized_action_difference_inf':float(np.max(abs(delta)/v.UWIDTH)),
        'metrics':observer.metrics,'six_input_interpretation':'finite-context initialization sensitivity only; action equality does not establish global input sufficiency'}
    immutable(folder/'completion.json',result);del ctrl,observer;gc.collect();return result

def run(root,out):
    root,out=Path(root),Path(out);config=check_seal(root,out);verify_identities(root,config);account=Accounting(out);limits=Limits(out);finished=threading.Event()
    def watch():
        while not finished.wait(.1):
            try:limits()
            except BaseException as exc:
                safe_write(out/'watchdog_failure.json',{'kind':type(exc).__name__,'message':str(exc)});x=read(out/'resource_ledger.json');x['halted']=True;x['status']='WATCHDOG_STOP';safe_write(out/'resource_ledger.json',x);os._exit(124)
    thread=threading.Thread(target=watch,daemon=True);thread.start();records=[]
    try:
        for job in jobs(config):
            limits();account.reserve(job);print('dispatch',job['id'],flush=True)
            try:
                result=scientific(root,out,config,job,account,limits) if job['mode']=='scientific' else diagnostic(root,out,config,job,account,limits) if job['mode']=='context' else replay(root,out,config,job,account,limits)
            except BaseException as exc:
                failure={'kind':type(exc).__name__,'message':str(exc),'job':job['id'],'complete':False}
                safe_write(out/'runs'/job['id']/'failure.json',failure);safe_write(out/'first_failure.json',failure);account.finish(job,'failed',failure);return {'complete':False,'failure':failure,'completed_jobs':len(records)}
            account.finish(job,'complete');records.append(result);safe_write(out/'completed_records.json',records)
        safe_write(out/'study_completion.json',{'complete':True,'jobs':len(records),'records':records});return {'complete':True,'jobs':len(records)}
    finally:finished.set();thread.join(timeout=1.)
