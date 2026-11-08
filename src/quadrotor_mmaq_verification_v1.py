"""Independent E12_MMAQ checks: exact-rational bounds and scalar NLP equations."""
from fractions import Fraction as F
from pathlib import Path
import numpy as np


def exact_majorant(x, model_u, actual_u, cache, f, linear=False, frequency=1000):
    """Independent rational evaluation; no Decimal/producer-bound imports."""
    def r(v): return F(float(v))
    t=F(1,50); h=F(1,frequency); mass=r(f['mass']); inertia=r(f['Iyy'])
    a=r(f['arm_length']/np.sqrt(2))*(r(model_u[1])-r(model_u[0]))/inertia
    ap=r(.028)*(r(actual_u[1])-r(actual_u[0]))/inertia
    b=(r(model_u[0])+r(model_u[1]))/mass
    bp=(r(actual_u[0])+r(actual_u[1]))/mass
    W=abs(r(x[5]))+abs(a)*t; Wp=abs(r(x[5]))+abs(ap)*t
    cache_gap=max(abs(r(x[4])-r(cache)),h*Wp)
    da=abs(ap-a)+abs(ap)*cache_gap**2/2
    pitch=da*t*(t+h)/2+abs(a)*t*h/2
    rate=da*t
    velocity=t*(abs(bp-b)+bp*(cache_gap+pitch))+b*W*t*h/2
    xposition=t*velocity+b*t*h/2
    zposition=t*velocity+(b+r(f['gravity']))*t*h/2
    result=[xposition,velocity,zposition,velocity,pitch,rate]
    if linear:
        angle=abs(r(x[4]))+abs(r(x[5]))*t+abs(a)*t*t/2
        gl=(r(f['hover_input'][0])+r(f['hover_input'][1]))/mass
        horizontal=b*angle**3/6+abs(b-gl)*angle
        vertical=b*angle**2/2+abs(gl-r(f['gravity']))
        extra=[horizontal*t*t/2,horizontal*t,vertical*t*t/2,vertical*t,F(0),F(0)]
    else:
        dv=b*abs(a)*t**3/12+b*(W**4+6*W*W*abs(a)+3*a*a)*t**5/2880
        dp=b*W*t**3/6+b*t*t/3*(abs(r(x[5]))*t/2+abs(a)*t*t/4)
        extra=[dp,dv,dp,dv,F(0),F(0)]
    return [v+w for v,w in zip(result,extra)]


def verify_archived_bounds(root,out):
    from src import quadrotor_pilot as p
    from src.quadrotor_mismatch_contract_v1 import bound_components
    root,out=Path(root),Path(out);f=p.read(out/'accepted_frozen.json')
    checked=0;failures=[];min_outward=None
    for sub in ['technical_pilot','linear_qrqp_revalidation']:
        for folder in sorted((root/'results/benchmark_selection'/sub/'runs').iterdir()):
            m=p.read(folder/'completion.json')
            if not m['complete'] or not (m['phase']=='original' or (m['phase']=='refine' and m['frequency']==2000)):continue
            a=np.load(folder/'trajectory.npz');N=m['frequency']//50
            for k in range(m['transitions']):
                x=a['states'][k];cache=a['physics_states'][max(0,k*N-1),4]
                u=np.clip(a['actions'][k],f['action_low'],f['action_high'])
                rpm=a['rpms'][k]
                actual=f['KF']*np.array([rpm[0]**2+rpm[3]**2,rpm[1]**2+rpm[2]**2])
                exact=exact_majorant(x,u,actual,cache,f,m['controller']=='linear40',m['frequency'])
                bound=bound_components(x,a['actions'][k],cache,f,m['controller']=='linear40',m['frequency'])['bound']
                for i,(b,e) in enumerate(zip(bound,exact)):
                    slack=F(float(b))-e
                    if slack<0:failures.append({'record':folder.name,'step':k,'component':i,'underestimate':float(-slack)})
                    if min_outward is None or slack<min_outward:min_outward=slack
                checked+=6
    return {'passed':not failures,'rational_component_checks':checked,'failures':failures,
            'minimum_outward_rounding_slack':float(min_outward),'independent_arithmetic':'exact Fraction; no Decimal rounding reused'}


def independent_nlp_program(ctrl,f):
    """Rebuild objective, shooting equations, boxes and derivatives from scalars.

    This does not import native symbolic dynamics, loss or RK discretization.
    Its KKT check concerns a local NLP stationary point, not a global optimum.
    """
    import casadi as c
    N=ctrl.T;nx=6*(N+1);nv=nx+2*N;np_=6+6*(N+1)
    z=c.MX.sym('z',nv);p=c.MX.sym('p',np_);lam=c.MX.sym('lam',22*N+18)
    X=c.reshape(z[:nx],6,N+1);U=c.reshape(z[nx:],2,N);ref=c.reshape(p[6:],6,N+1)
    low=np.asarray(f['state_low']);high=np.asarray(f['state_high']);ulo=np.asarray(f['action_low']);uhi=np.asarray(f['action_high'])
    Q=np.asarray(f['Q']);R=np.asarray(f['R']);hover=np.asarray(f['hover_input']);h=.02
    def fc(x,u):
        total=(u[0]+u[1])/f['mass']
        return c.vertcat(x[1],total*c.sin(x[4]),x[3],total*c.cos(x[4])-f['gravity'],x[5],
                         f['arm_length']*(u[1]-u[0])/(np.sqrt(2)*f['Iyy']))
    def step(x,u):
        k1=fc(x,u);k2=fc(x+h*k1/2,u);k3=fc(x+h*k2/2,u);k4=fc(x+h*k3,u)
        return x+h*(k1+2*k2+2*k3+k4)/6
    gs=[];los=[];his=[];objective=0
    for j in range(N):
        err=X[:,j]-ref[:,j];du=U[:,j]-hover
        objective+=.5*c.mtimes([err.T,Q,err])+.5*c.mtimes([du.T,R,du])
        gs += [X[:,j+1]-step(X[:,j],U[:,j]),c.vertcat(-X[:,j]+low,X[:,j]-high),
               c.vertcat(-U[:,j]+ulo,U[:,j]-uhi)]
        los += [c.DM.zeros(6),c.DM.ones(12)*(-np.inf),c.DM.ones(4)*(-np.inf)]
        his += [c.DM.zeros(6),c.DM.ones(12)*(-1e-6),c.DM.ones(4)*(-1e-6)]
    err=X[:,-1]-ref[:,-1]
    objective+=.5*c.mtimes([err.T,Q,err])+.5*float(hover@R@hover)
    gs += [c.vertcat(-X[:,-1]+low,X[:,-1]-high),X[:,0]]
    los += [c.DM.ones(12)*(-np.inf),p[:6]];his += [c.DM.ones(12)*(-1e-6),p[:6]]
    g=c.vertcat(*gs);lo=c.vertcat(*los);hi=c.vertcat(*his)
    assert g.numel()==22*N+18
    gradient=c.gradient(objective+c.dot(g,lam),z)
    fn=c.Function('independent_nlp',[z,p,lam],[objective,g,lo,hi,gradient]).expand()
    opti=ctrl.opti_dict['opti'];assert opti.nx==nv and opti.np==np_ and opti.ng==g.numel()
    native=c.Function('native_program',[opti.x,opti.p],[opti.f,opti.g,opti.lbg,opti.ubg])
    return fn,native


def nlp_metrics(fn,native,z,p,lam):
    obj,g,lo,hi,gradient=fn(z,p,lam);native_values=native(z,p)
    values=[np.asarray(v).reshape(-1) for v in [obj,g,lo,hi,gradient]]
    objective,g,lo,hi,gradient=values
    errors=[]
    for own,other in zip(values[:4],native_values):
        other=np.asarray(other).reshape(-1);finite=np.isfinite(own)|np.isfinite(other)
        if not np.array_equal(np.isinf(own),np.isinf(other)):raise AssertionError('NLP bound structure differs')
        errors.append(float(np.max(np.abs(own[finite]-other[finite]),initial=0)))
    primal=float(max(0,np.max(lo-g),np.max(g-hi)))
    equal=lo==hi;active=np.where(lam>=0,hi,lo);ineq=~equal
    products=lam[ineq]*(g[ineq]-active[ineq])
    comp=float(np.max(np.abs(products)/(1+np.abs(lam[ineq])),initial=0))
    return {'objective':float(objective[0]),'program_reconstruction_error':max(errors),
            'primal_inf':primal,'stationarity_inf':float(np.max(np.abs(gradient))),
            'normalized_complementarity_inf':comp}


def require_nlp(metrics,status):
    if status!='Solve_Succeeded':raise AssertionError('Strict NLP success missing')
    if not np.isfinite(list(metrics.values())).all():raise AssertionError('Nonfinite NLP diagnostic')
    if metrics['program_reconstruction_error']>1e-10:raise AssertionError('NLP reconstruction failed')
    if metrics['primal_inf']>1e-4:raise AssertionError('Native NLP primal tolerance failed')
    if metrics['stationarity_inf']>1.01e-8 or metrics['normalized_complementarity_inf']>1.01e-8:
        raise AssertionError('Independent NLP KKT tolerance failed')


def verify_qualification(root,upstream,out):
    """Read-only episode audit, independent KKT replay and historical preservation."""
    import re
    from src import quadrotor_pilot as pilot
    from src.quadrotor_mmaq_v1 import check_seal,safe_write,preservation
    from src.quadrotor_verification import audit_record
    from src.quadrotor_linear_verification import independent_program,kkt_metrics,require_valid_solution
    root,out=Path(root),Path(out);check_seal(root,out)
    f=pilot.read(out/'frozen.json');reference=np.load(out/'reference.npy')
    program=independent_program(f);nlp={};ctrls=[];rows=[];repeat=[]
    pilot.activate(upstream,f['config'])
    factory=lambda:pilot.make_environment(upstream,f['config'],np.asarray(f['cases']['H0']),1000)
    try:
        for key in ['nonlinear20','nonlinear40','nonlinear80']:
            ctrl=pilot.make_controller(upstream,f['config'],factory,key);ctrls.append(ctrl)
            nlp[key]=independent_nlp_program(ctrl,f)
        for case in ['H0','H1','H2','H3']:
            for controller in ['linear40','nonlinear20','nonlinear40','nonlinear80']:
                for phase in ['original','repeat']:
                    folder=out/'runs'/f'{case}_{controller}_1000_{phase}'
                    m=pilot.read(folder/'completion.json')
                    if not m['complete']:
                        rows.append({'case':case,'controller':controller,'phase':phase,'complete':False,
                                     'failure':m['failure'],'safe_complete_no_contact':False,'return':None})
                        continue
                    r=audit_record(folder,f,reference);a=np.load(folder/'trajectory.npz')
                    for marker in ['replay_native.json','replay_instrumented.json']:
                        replay=pilot.read(folder/marker)
                        for k,v in replay.items():
                            if k.endswith('bitwise'):assert v,(folder,marker,k)
                    stream=np.fromfile(folder/'partial_physics.f64',dtype='<f8').reshape(-1,7)
                    assert stream.shape==(6001,7)
                    assert stream[:,:6].tobytes()==a['physics_states'].tobytes()
                    assert np.array_equal(stream[:,6],a['physics_contacts'])
                    causal=pilot.read(folder/'causal_contract.json');assert len(causal)==300
                    coverage=[];closures=[];reserves=[];exact_count=0
                    for k,b in enumerate(causal):
                        assert np.array_equal(b['initial'],a['states'][k])
                        cache=a['physics_states'][max(0,k*20-1),4];assert b['cache']==cache
                        assert np.array_equal(b['requested'],a['actions'][k])
                        exact=exact_majorant(a['states'][k],b['model_thrust'],b['executed_thrust'],cache,f,controller=='linear40')
                        assert all(F(float(v))>=e for v,e in zip(b['bound'],exact));exact_count+=6
                        error=np.abs(np.asarray(b['prediction'])-a['states'][k+1])
                        coverage.append(float(np.max(error-np.asarray(b['bound']))))
                        closures.append(float(np.max(np.abs(np.asarray(b['source_prediction'])-a['states'][k+1]))))
                        margin=np.minimum(np.asarray(b['prediction'])-f['state_low'],np.asarray(f['state_high'])-b['prediction'])
                        reserve=margin-b['bound'];assert np.max(np.abs(reserve-b['reserve']))<1e-12
                        reserves.append(reserve)
                    assert max(coverage)<=1e-10 and max(closures)<=1e-10
                    assert np.max(np.abs(a['physics_states'][:,4]))<np.pi/2
                    rpm=a['rpms'];executed=f['KF']*np.column_stack([rpm[:,0]**2+rpm[:,3]**2,rpm[:,1]**2+rpm[:,2]**2])
                    input_margins=np.minimum(executed-f['action_low'],np.asarray(f['action_high'])-executed)
                    r.update(input_minimum_margins=input_margins.min(0).tolist(),
                        input_violations=int(np.count_nonzero(input_margins<0)),
                        minimum_reserve_by_component=np.min(reserves,0).tolist(),
                        source_closure_maximum=max(closures),maximum_bound_excess=max(coverage),
                        independent_rational_checks=exact_count)
                    metrics=[]
                    if controller=='linear40':
                        solutions=np.load(folder/'kkt_solutions.npz')
                        for p,z,lam in zip(solutions['parameters'],solutions['primal'],solutions['dual']):
                            mm=kkt_metrics(program,f,p,z,lam);require_valid_solution(True,True,mm);metrics.append(mm)
                        log=out/f'{case}_{controller}_{phase}_episode.log'
                        its=[int(match.group(1)) for line in log.read_text().splitlines()
                             if (match:=re.match(r'^\s*(\d+)\s+.*\bConverged\s*$',line))]
                        assert len(its)==300;r['maximum_solver_iterations']=max(its)
                    else:
                        solutions=np.load(folder/'nlp_solutions.npz')
                        for z,p,lam in zip(solutions['primal'],solutions['parameters'],solutions['dual']):
                            mm=nlp_metrics(*nlp[controller],z,p,lam);require_nlp(mm,'Solve_Succeeded');metrics.append(mm)
                    assert len(metrics)==300
                    r['independent_solution_checks']=300
                    r['solution_metric_maxima']={k:max(v[k] for v in metrics) for k in metrics[0]}
                    r['safe_complete_no_contact']=r['safe_complete_no_contact'] and r['input_violations']==0
                    rows.append(r)
                left=out/'runs'/f'{case}_{controller}_1000_original';right=out/'runs'/f'{case}_{controller}_1000_repeat'
                x=np.load(left/'trajectory.npz');y=np.load(right/'trajectory.npz')
                fields=['states','actions','physics_states','physics_contacts','rpms','clipped_actions','rewards',
                        'predicted_states','predicted_inputs','solver_success','solver_iterations','current_steps']
                equal={k:x[k].tobytes()==y[k].tobytes() for k in fields}
                dualfile='kkt_solutions.npz' if controller=='linear40' else 'nlp_solutions.npz'
                if (left/dualfile).exists() and (right/dualfile).exists():
                    sx=np.load(left/dualfile);sy=np.load(right/dualfile)
                    equal.update({dualfile+':'+k:sx[k].tobytes()==sy[k].tobytes() for k in sx.files})
                repeat.append({'case':case,'controller':controller,'fields_bitwise':equal,'passed':all(equal.values())})
        summaries={}
        for controller in ['linear40','nonlinear20','nonlinear40','nonlinear80']:
            selected=[r for r in rows if r['controller']==controller and r['phase']=='original']
            latencies=[];overruns=[]
            for r in [v for v in rows if v['controller']==controller]:
                a=np.load(out/'runs'/f"{r['case']}_{controller}_1000_{r['phase']}"/'trajectory.npz')
                latencies.extend(a['solver_wall'].tolist())
                overruns.extend({'case':r['case'],'phase':r['phase'],'step':k,'seconds':float(v)}
                                for k,v in enumerate(a['solver_wall']) if v>.02)
            complete=all(r['complete'] for r in selected)
            s={'complete_cases':sum(r['complete'] for r in selected),
               'safe_cases':sum(r['safe_complete_no_contact'] for r in selected),
               'mean_return':float(np.mean([r['return'] for r in selected])) if complete else None,
               'worst_return':min(r['return'] for r in selected) if complete else None,
               'latency_seconds':dict(zip(['median','p95','p99','maximum'],map(float,[*np.percentile(latencies,[50,95,99]),max(latencies)]))),
               'deadline_overruns':overruns,'deadline_overrun_count':len(overruns),'measured_action_calls':len(latencies)}
            if complete:
                s.update(mean_position_rmse=float(np.mean([r['position_tracking_rmse'] for r in selected])),
                         mean_thrust_squared_integral=float(np.mean([r['absolute_thrust_squared_integral'] for r in selected])),
                         minimum_normalized_margin=min(r['minimum_normalized_margin'] for r in selected),
                         minimum_raw_margins=np.min([r['minimum_margins_by_state'] for r in selected],0).tolist(),
                         minimum_reserve_by_component=np.min([r['minimum_reserve_by_component'] for r in selected],0).tolist())
            summaries[controller]=s
        primary=[r for r in rows if r['controller']=='nonlinear80']
        passed=all(r['complete'] for r in rows) and all(v['passed'] for v in repeat)
        passed=passed and all(r['safe_complete_no_contact'] and min(r['minimum_reserve_by_component'])>0 for r in primary)
        result={'version':'E12_MMAQ_v1','passed':bool(passed),'records':rows,'summaries':summaries,'repeats':repeat,
                'historical_original_symbolic_gate':'FAILED','original_gate_passing_new_records':sum(bool(r.get('one_step_tolerance_passed')) for r in rows),
                'native_linear_software_outcome':'FAILED before action; adapter is separate',
                'screening_executed':False,'action_probes_executed':False,'training_executed':False,
                'hard_real_time_ready':False,'formal_safety_certificate':False}
        safe_write(out/'verification.json',result)
        safe_write(out/'preservation_final.json',preservation(root,upstream))
        return {'passed':result['passed'],'records':len(rows),'summaries':summaries,'historical_gate':'FAILED'}
    finally:
        for ctrl in ctrls:ctrl.close()
