"""Independent horizon-40 linear-QP and saved-record mathematics for E12.

No upstream MPC equations, Jacobians or discretization helpers are imported.
The linear transition uses the finite nilpotent matrix-exponential polynomial.
"""
from pathlib import Path
import json
import re
import numpy as np
from scipy import sparse


def linear_transition(f):
    A=np.zeros((6,6));A[0,1]=A[2,3]=A[4,5]=1;A[1,4]=f['gravity']
    B=np.zeros((6,2));B[3,:]=1/f['mass'];B[5,:]=np.array([-1,1])*f['arm_length']/np.sqrt(2)/f['Iyy']
    A2=np.einsum('ij,jk->ik',A,A);A3=np.einsum('ij,jk->ik',A2,A)
    h=.02;Ad=np.eye(6)+h*A+h*h*A2/2+h**3*A3/6
    Bd=np.einsum('ij,jk->ik',h*np.eye(6)+h*h*A/2+h**3*A2/6+h**4*A3/24,B)
    return Ad,Bd


def independent_program(f):
    N=40;nx=6;nu=2;ns=nx*(N+1);nv=ns+nu*N
    Ad,Bd=linear_transition(f);Q=np.array(f['Q']);R=np.array(f['R'])
    Hdiag=np.r_[np.tile(np.diag(Q),N+1),np.tile(np.diag(R),N)]
    A=np.zeros((898,nv));lower=np.full(898,-np.inf);upper=np.zeros(898)
    Cx=np.vstack([-np.eye(nx),np.eye(nx)]);Cu=np.vstack([-np.eye(nu),np.eye(nu)])
    xmax=np.array(f['state_high']);xmin=np.array(f['state_low'])
    umin=np.array(f['action_low']);umax=np.array(f['action_high']);ueq=np.array(f['hover_input'])
    tol=1e-8;row=0;equality=[]
    for k in range(N):
        A[row:row+nx,k*nx:(k+1)*nx]=-Ad
        A[row:row+nx,(k+1)*nx:(k+2)*nx]=np.eye(nx)
        A[row:row+nx,ns+k*nu:ns+(k+1)*nu]=-Bd
        lower[row:row+nx]=upper[row:row+nx]=0;equality.extend(range(row,row+nx));row+=nx
        A[row:row+2*nx,k*nx:(k+1)*nx]=Cx
        upper[row:row+2*nx]=np.r_[-xmin-tol,xmax-tol];row+=2*nx
        A[row:row+2*nu,ns+k*nu:ns+(k+1)*nu]=Cu
        upper[row:row+2*nu]=np.r_[ueq-umin-tol,umax-ueq-tol];row+=2*nu
    A[row:row+2*nx,N*nx:(N+1)*nx]=Cx
    upper[row:row+2*nx]=np.r_[-xmin-tol,xmax-tol];row+=2*nx
    A[row:row+nx,:nx]=np.eye(nx);lower[row:row+nx]=upper[row:row+nx]=0
    equality.extend(range(row,row+nx));row+=nx
    assert row==898 and len(equality)==246
    return {'Hdiag':Hdiag,'A':sparse.csr_matrix(A),'lower':lower,'upper':upper,'equality_rows':np.array(equality),'Ad':Ad,'Bd':Bd}


def parameter_terms(program,f,p):
    p=np.asarray(p).reshape(-1);assert len(p)==252
    refs=p[6:].reshape(6,41,order='F')
    q=np.r_[(-np.diag(f['Q'])[:,None]*refs).reshape(-1,order='F'),np.zeros(80)]
    lo=program['lower'].copy();hi=program['upper'].copy();lo[-6:]=hi[-6:]=p[:6]
    constant=.5*float(np.sum(np.diag(f['Q'])[:,None]*refs**2))
    return q,lo,hi,constant


def kkt_metrics(program,f,p,z,lam):
    q,lo,hi,c=parameter_terms(program,f,p);z=np.asarray(z).reshape(-1);lam=np.asarray(lam).reshape(-1)
    g=program['A'].dot(z);gradient=program['Hdiag']*z+q
    stationarity=gradient+program['A'].T.dot(lam)
    primal=max(0,float(np.max(lo-g)),float(np.max(g-hi)))
    inequality=lo!=hi;up=inequality&(lam>=0);down=inequality&(lam<0)
    upper_finite=up&np.isfinite(hi);lower_finite=down&np.isfinite(lo)
    products=np.r_[lam[upper_finite]*(g[upper_finite]-hi[upper_finite]),
                   lam[lower_finite]*(g[lower_finite]-lo[lower_finite])]
    normalized=np.r_[np.abs(products[:np.count_nonzero(upper_finite)])/(1+np.abs(lam[upper_finite])),
                     np.abs(products[np.count_nonzero(upper_finite):])/(1+np.abs(lam[lower_finite]))]
    bad_sign=np.r_[np.abs(lam[up&~np.isfinite(hi)]),np.abs(lam[down&~np.isfinite(lo)])]
    value=.5*float(np.sum(program['Hdiag']*z*z))+float(np.sum(q*z))+c
    return {'primal_inf':primal,'stationarity_inf':float(np.max(np.abs(stationarity))),
            'complementarity_inf':float(np.max(np.abs(products),initial=0)),
            'normalized_complementarity_inf':float(np.max(normalized,initial=0)),
            'dual_sign_inf':float(np.max(bad_sign,initial=0)),'independent_objective':value}


def require_valid_solution(success,finite,metrics):
    if not success or not finite:raise AssertionError('Solver failed or returned nonfinite values; no action may be applied')
    for k in ['primal_inf','stationarity_inf','normalized_complementarity_inf','dual_sign_inf']:
        if metrics[k]>1.01e-8:raise AssertionError('QP optimality verification failed: '+k)


def verify_revalidation(root,upstream,out):
    from src.quadrotor_verification import audit_record,digest
    root,out=Path(root),Path(out);f=json.loads((out/'frozen.json').read_text());ref=np.load(out/'reference.npy')
    program=independent_program(f);rows=[];repeat=[]
    for folder in sorted((out/'runs').iterdir()):
        r,a=audit_record(folder,f,ref);m=json.loads((folder/'completion.json').read_text())
        assert m['controller_variant']=='linear40_qrqp_v1'
        assert digest(folder/'kkt_solutions.npz')==m['kkt_solutions_sha256']
        marker=json.loads((folder/'revalidation_complete.json').read_text())
        assert digest(folder/'completion.json')==marker['completion_sha256']
        gd=out/'graphs'/f"{m['case']}_{m['phase']}";graph=json.loads((gd/'graph_audit.json').read_text())
        assert graph['graph_sha256_before']==graph['graph_sha256_after']==digest(gd/'native_program.casadi')
        coefficients=np.load(gd/'native_graph_coefficients.npz')
        assert np.max(np.abs(coefficients['H']-np.diag(program['Hdiag'])))<1e-10
        assert np.max(np.abs(coefficients['A']-program['A'].toarray()))<1e-10
        qp=np.load(folder/'kkt_solutions.npz');metrics=[]
        assert r['frequency']==1000
        for i,(p,z,lam) in enumerate(zip(qp['parameters'][:m['transitions']],qp['primal'][:m['transitions']],qp['dual'][:m['transitions']])):
            metric=kkt_metrics(program,f,p,z,lam);require_valid_solution(True,bool(np.isfinite(np.r_[p,z,lam]).all()),metric)
            assert np.max(np.abs(p[:6]-a['states'][i]))<1e-14
            wanted=ref[np.minimum(i+np.arange(41),300)].T
            assert np.array_equal(p[6:].reshape(6,41,order='F'),wanted)
            xs=z[:246].reshape(6,41,order='F');us=z[246:].reshape(2,40,order='F')
            assert np.max(np.abs(xs-a['predicted_states'][i]))<1e-12
            assert np.max(np.abs(us+np.array(f['hover_input'])[:,None]-a['predicted_inputs'][i]))<1e-12
            assert np.max(np.abs(us[:,0]+f['hover_input']-a['actions'][i]))<1e-12
            metric['objective_reconstruction_error']=abs(metric['independent_objective']-qp['objective'][i]);metrics.append(metric)
        r['kkt_maxima']={key:max(v[key] for v in metrics) for key in metrics[0]} if metrics else None
        r['controller_variant']=m['controller_variant'];r['validated_solutions']=len(metrics)
        log=out/f"{m['case']}_{m['phase']}_episode.log"
        iterations=[int(match.group(1)) for line in log.read_text().splitlines()
                    if (match:=re.match(r'^\s*(\d+)\s+.*\bConverged\s*$',line))]
        assert len(iterations)==m['transitions']
        r['QRQP_native_iterations']=iterations
        r['QRQP_native_maximum_iterations']=max(iterations,default=None)
        r['QRQP_native_mean_iterations']=float(np.mean(iterations)) if iterations else None
        r['iteration_note']='Opti reports -1; counts independently recovered from retained QRQP convergence rows'
        rows.append(r)
    for case in ['T0','T1','T2','T3']:
        a=out/'runs'/f'{case}_linear40_1000_original';b=out/'runs'/f'{case}_linear40_1000_repeat'
        x=np.load(a/'trajectory.npz');y=np.load(b/'trajectory.npz')
        fields=['states','actions','physics_states','physics_contacts','rpms','clipped_actions','rewards',
                'predicted_states','predicted_inputs','solver_success','solver_iterations','current_steps']
        equal={k:x[k].tobytes()==y[k].tobytes() for k in fields};assert all(equal.values())
        x=np.load(a/'kkt_solutions.npz');y=np.load(b/'kkt_solutions.npz')
        qp_equal={k:x[k].tobytes()==y[k].tobytes() for k in x.files};assert all(qp_equal.values())
        repeat.append({'case':case,'trajectory_fields_bitwise':equal,'QP_solutions_bitwise':qp_equal})
        originals=next(x for x in rows if x['case']==case and x['phase']=='original')
        repetitions=next(x for x in rows if x['case']==case and x['phase']=='repeat')
        assert originals['QRQP_native_iterations']==repetitions['QRQP_native_iterations']
    base=json.loads((root/'results/benchmark_selection/technical_pilot/verification.json').read_text())['records']
    comparisons=[]
    for ctrl in ['nonlinear20','nonlinear40','nonlinear80']:
        differences=[]
        for case in ['T0','T1','T2','T3']:
            native=next(x for x in base if x['case']==case and x['controller']==ctrl and x['phase']=='original')
            adapted=next(x for x in rows if x['case']==case and x['phase']=='original')
            differences.append(adapted['return']-native['return'] if adapted['complete'] else None)
        comparisons.append({'comparator':ctrl,'case_order':['T0','T1','T2','T3'],'paired_return_differences':differences,
                            'mean_paired_difference':float(np.mean(differences)) if all(v is not None for v in differences) else None})
    summary={'records':rows,'repeatability':repeat,'nonlinear_comparisons':comparisons,
             'original_native_linear_outcome':'FAILED before applying actions; preserved as separate baseline',
             'original_model_agreement_gate':'FAILED; unchanged threshold 0.001',
             'nonlinear_runs_reused':True,'nonlinear_runs_executed':0,'screening_executed':False,'action_probes_executed':False}
    (out/'verification.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    return summary
