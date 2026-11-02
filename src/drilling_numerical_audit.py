"""Independent E11A saved-history/equation audit. No plant rollout or controller design."""
from pathlib import Path
import csv
import gc
import json
import time
import zipfile
import numpy as np
from scipy.io import loadmat
from scipy import sparse
from src.drilling_reproduction import read,write,sha,freeze

STATE_SCALE=np.r_[np.ones(10),np.tile([.001,10.],5)]

def dense(x):return x.toarray() if sparse.issparse(x) else np.asarray(x)
def scalar(a,k):return float(np.real(np.asarray(a[k]).item()))
def vector(a,k):return np.asarray(a[k]).reshape(-1)
def discrepancy(a,b,scale=1.):
    a,b=np.asarray(a),np.asarray(b);d=np.abs(a-b)
    den=max(float(np.linalg.norm(a)),float(np.linalg.norm(b)),np.finfo(float).tiny)
    return {'maximum_absolute':float(d.max(initial=0)),
            'norm_relative':float(np.linalg.norm(a-b)/den),
            'maximum_scaled':float(np.max(d/np.asarray(scale),initial=0))}

def kv(w):
    x=w+10
    lower=-1.38075395166892e-6*x**3+4.77361019075787e-5*x**2-.000482078306369382*x+.00192103100788584-.001
    upper=-2.02170534249845e-5*x**3+.000550674216859688*x**2-.00429753361761171*x+.0115549758098856-.001
    return lower,upper

def physical(E,Vs,c):
    v=(E[18]+Vs[8])/c;w=(E[19]+Vs[9])/c
    lower,upper=kv(w-10)
    return v,w,np.array([w-5.23,15.7-w,v-(.001+lower),.001+upper-v])

def safety_summary(E,Vs,c,completed):
    v,w,m=physical(E,Vs,c)
    valid=np.isfinite(E).all(axis=0)&(np.imag(E)==0).all(axis=0)
    strict=valid & (np.real(m)>0).all(axis=0)
    violations=valid & (np.real(m)<=0).any(axis=0)
    crossing=np.flatnonzero(violations);invalid=np.flatnonzero(~valid)
    names=['omega_lower_rad_s','omega_upper_rad_s','axial_lower_m_s','axial_upper_m_s']
    return dict(complete=bool(completed),empirically_strictly_safe=bool(completed and strict.all()),
        raw_state_count=E.shape[1],invalid_physical_state_count=int((~valid).sum()),
        first_invalid_step=None if not len(invalid) else int(invalid[0]),
        violation_state_count=int(violations.sum()),
        first_crossing_step=None if not len(crossing) else int(crossing[0]),
        first_crossing_seconds=None if not len(crossing) else float(crossing[0]*1e-4),
        minimum_margins={n:float(np.real(m[i,valid]).min()) if valid.any() else None for i,n in enumerate(names)},
        violations_by_face={n:int((valid & (np.real(m[i])<=0)).sum()) for i,n in enumerate(names)},
        maximum_absolute_imaginary_state=float(np.abs(np.imag(E)).max(initial=0))),v,w,m

def reassemble(a):
    """Rebuild all four FEM elements and global-to-reduced operators independently."""
    pos=np.asarray(a['P_initial'])[0,:,:3];tail=vector(a,'P_tail')[:3]
    m,l,E,G,A,Ix,Iy,Iz,J=[scalar(a,k) for k in ['m','l','E','G','A','Ix','Iy','Iz','J']]
    M=np.zeros((20,20));K=M.copy();D=M.copy();grav=np.zeros(20);rot=[]
    angles=vector(a,'angle')
    for i in range(4):
        d=pos[i+1]-pos[i];q=(pos[i+2] if i<3 else tail)-pos[i]
        tangent=d/np.linalg.norm(d);normal=np.cross(d,q)/np.linalg.norm(np.cross(d,q))
        lateral=np.cross(normal,tangent)
        r=np.array([tangent[:2],lateral[:2]]);R=np.kron(np.eye(4),r);rot.append(R)
        k=np.zeros((8,8));mm=np.zeros((8,8))
        for ids,block in [([0,4],E*A/l*np.array([[1.,-1],[-1,1]])),
                          ([1,5],12*E*Iz/l**3*np.array([[1.,-1],[-1,1]])),
                          ([2,6],G*J/l*np.array([[1.,-1],[-1,1]])),
                          ([3,7],E*Iy/l*np.array([[4.,2],[2,4]]))]:k[np.ix_(ids,ids)]=block
        for ids,block in [([0,4],[[140,70],[70,140]]),([1,5],[[156,54],[54,156]]),
                          ([2,6],Ix/A*np.array([[140,-70],[-70,140]])),
                          ([3,7],l*l*np.array([[4,-3],[-3,4]]))]:mm[np.ix_(ids,ids)]=block
        kg=R.T@k@R;mg=(R.T*(m/420))@mm@R
        dg=scalar(a,'Raleigh_a')*mg+scalar(a,'Raleigh_b')*kg
        sl=slice(i*4,i*4+8);M[sl,sl]+=mg;K[sl,sl]+=kg;D[sl,sl]+=dg
        f=np.zeros(8);f[0]=(np.cos(angles[i])-np.sin(angles[i])*scalar(a,'fri_coef'))*m/2
        f[4]=(np.cos(angles[i+1])-np.sin(angles[i+1])*scalar(a,'fri_coef'))*m/2
        grav[sl]+=R.T@f*scalar(a,'g')
    transform=np.zeros((20,10))
    for i in range(5):
        transform[4*i,2*i]=transform[4*i+2,2*i+1]=1
        transform[4*i+1,2*i]=transform[4*i+3,2*i+1]=np.tan(angles[i])
    KM=np.linalg.solve(M,K)[::2]@transform;DM=np.linalg.solve(M,D)[::2]@transform
    return M,K,D,KM,DM,grav,np.stack(rot,axis=2)

def external(a,F,T,W,B,gravity=None):
    F,T,W,B=np.broadcast_arrays(F,T,W,B);n=F.size
    out=np.tile(vector(a,'Gravity')[:,None] if gravity is None else gravity[:,None],(1,n))
    out[0]+=F.reshape(-1);out[2]+=T.reshape(-1)
    angle=vector(a,'angle')[-1];r=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    bottom=np.zeros((8,n));bottom[4]=-W.reshape(-1);bottom[6]=-B.reshape(-1)
    out[-8:]+=np.kron(np.eye(4),r)@bottom
    return out

def loads(a,v,w):
    with np.errstate(all='ignore'):
        d=np.where((v<0)|(w<0),0,2*np.pi*v/w);d=np.clip(d,0,scalar(a,'threshold'))
        fv=1/(1+np.exp(scalar(a,'a1')*v+scalar(a,'b1')))
        fw=np.tanh(scalar(a,'a2')*w+scalar(a,'b2'))
        wc=np.where(d<=0,0,scalar(a,'epsilon')*scalar(a,'zeta')*scalar(a,'Rb')*d)
        tc=np.where(d<=0,0,scalar(a,'epsilon')*scalar(a,'Rb')**2*d/2)
        wf=np.where(d<=0,0,np.where(d<scalar(a,'d0'),scalar(a,'Kc')*d,
                scalar(a,'Rb')*scalar(a,'sigma')*scalar(a,'lw')*fv))
        tf=scalar(a,'gam')*scalar(a,'miu')*scalar(a,'Rb')*wf*fw/2
    return dict(d=d,fv=fv,fw=fw,WOBc=wc,TOBc=tc,WOBf=wf,TOBf=tf,WOB=wc+wf,TOB=tc+tf)

def lookup_history(a):
    n=int(scalar(a,'n'));x=vector(a,'cnt1');y=vector(a,'cnt2')
    valid=(np.isfinite(x)&np.isfinite(y)&(np.imag(x)==0)&(np.imag(y)==0)&
           (np.real(x)>=1)&(np.real(x)<=39)&(np.real(y)>=1)&(np.real(y)<=39))
    valid[0]=False;last=np.maximum.accumulate(np.where(valid,np.arange(n),-1))
    effective=np.full((2,n),-1,dtype=np.int16);selected=last>=0
    effective[0,selected]=np.real(x[last[selected]]).astype(np.int16)
    effective[1,selected]=np.real(y[last[selected]]).astype(np.int16)
    fallback=~valid;fallback[0]=False
    predicted=np.zeros((2,n),dtype=np.complex128);table=a['Mat_data2'];z=np.asarray(a['Z_b'])
    for i,j in set(map(tuple,effective[:,selected].T)):
        which=np.flatnonzero((effective[0]==i)&(effective[1]==j))
        predicted[:,which]=table[int(i)-1,int(j)-1][0,2]@z[:,which]
    actual=np.vstack([vector(a,'u1'),vector(a,'u2')])
    f=np.flatnonzero(fallback)
    memory_ok=bool(not len(f) or np.array_equal(z[-2:,f],z[-2:,f-1]))
    return dict(attempts=n-1,successful=int(valid.sum()),failures=int(fallback.sum()),
        first_lookup_failed=bool(not valid[1]),
        first_failure_step=None if not len(f) else int(f[0]),
        complex_index_count=int(((np.imag(x)!=0)|(np.imag(y)!=0)).sum()),
        nonfinite_index_count=int((~np.isfinite(x)|~np.isfinite(y)).sum()),
        stale_transformed_memory_exact=memory_ok,
        correction_discrepancy=discrepancy(predicted[:,1:],actual[:,1:],np.maximum(1,np.abs(actual[:,1:])))),effective,fallback

def case_audit(a,status,config):
    E=np.asarray(a['E1']);n=int(scalar(a,'n'));completed=bool(status['native_entrypoint_completed'] and
        n==200001 and scalar(a,'k')==n and E.shape==(20,n))
    Vs=vector(a,'Vs');c=np.cos(vector(a,'angle')[-1]);s,v,w,m=safety_summary(E,Vs,c,completed)
    valid=np.isfinite(E).all() and not np.any(np.imag(E)!=0)
    native_M=dense(a['M1_global']);native_KM=dense(a['K_M']);native_DM=dense(a['D_M'])
    M,K,D,KM,DM,gravity,R=reassemble(a)
    matrices={name:discrepancy(x,dense(a[name]).reshape(x.shape)) for name,x in
              [('M1_global',M),('K1_global',K),('D1_global',D),('K_M',KM),('D_M',DM),('Gravity',gravity),('R1',R)]}
    FM=vector(a,'F_M_d');IC=vector(a,'IC')
    r0=FM-native_DM@Vs-native_KM@IC;r1=native_KM@Vs;saved=vector(a,'err')
    Fd=scalar(a,'force_d');Td=scalar(a,'torque_d');Wd=scalar(a,'WOB_d');Bd=scalar(a,'TOB_d')
    rebuilt_FM=np.linalg.solve(M,external(a,np.array([Fd]),np.array([Td]),np.array([Wd]),np.array([Bd]),gravity))[::2,0]
    residuals={'native_matrix_r0':r0.tolist(),'native_matrix_KM_Vs':r1.tolist(),
        'saved_err':saved.tolist(),'r0_minus_saved_err':discrepancy(r0,saved),
        'independent_matrix_r0':(rebuilt_FM-DM@Vs-KM@IC).tolist(),
        'independent_matrix_KM_Vs':(KM@Vs).tolist(),
        'nominal_FM_reassembly':discrepancy(rebuilt_FM,FM),
        'KM_condition_number':float(np.linalg.cond(native_KM)),
        'interpretation':'Implemented error dynamics are unchanged; no residual compensation is applied.'}
    lookup,effective,fallback=lookup_history(a)
    rng=np.random.Generator(np.random.PCG64(config['audit_seed']));indices=set(map(int,rng.choice(n-1,64,replace=False)))|{0,1,2,n-2}
    events=list(np.flatnonzero(fallback)[:1])+list(np.flatnonzero(fallback)[-1:])
    if s['first_crossing_step'] is not None:events.append(s['first_crossing_step'])
    for event in events:
        indices.update(j for j in (event-1,event,event+1) if 0<=j<n-1)
    ids=np.array(sorted(indices));F=vector(a,'force');T=vector(a,'torque')
    W=vector(a,'WOB').copy();B=vector(a,'TOB').copy();W[0]=Wd;B[0]=Bd
    ext=external(a,F[ids],T[ids],W[ids],B[ids]);delta=np.linalg.solve(native_M,ext)[::2]-FM[:,None]
    delta[:,ids==0]=0 # Initial nominal loads are authoritative, independent of E0.
    prediction=E[:,ids]+1e-4*(np.asarray(a['Al1'])@E[:,ids]+np.vstack([np.zeros_like(delta),delta]))
    transition=discrepancy(prediction,E[:,ids+1],STATE_SCALE[:,None])
    comparison_loads=loads(a,v[ids[ids>0]],w[ids[ids>0]])
    load_errors={k:discrepancy(value,vector(a,k)[ids[ids>0]],np.maximum(1,np.abs(vector(a,k)[ids[ids>0]]))) for k,value in comparison_loads.items()}
    q=E.copy().astype(np.complex128);q[::2]*=1000
    lo,up=kv(w[ids]-10);ev=q[18,ids];ew=q[19,ids]
    with np.errstate(all='ignore'):
        vs=ev/np.emath.sqrt(up*c*1000-ev)/np.emath.sqrt(ev-lo*c*1000)
        ws=ew/np.emath.sqrt(5.7*c-ew)/np.emath.sqrt(ew+4.77*c)
        cx=np.floor(np.real((vs+.25)/.5))+1j*np.floor(np.imag((vs+.25)/.5))+20
        cy=np.floor(np.real((ws+.25)/.5))+1j*np.floor(np.imag((ws+.25)/.5))+20
    evaluated=ids>0
    lookup_indices_match=bool(np.array_equal(cx[evaluated],vector(a,'cnt1')[ids[evaluated]],equal_nan=True) and
                              np.array_equal(cy[evaluated],vector(a,'cnt2')[ids[evaluated]],equal_nan=True))
    # Compare raw transformed quantities on successful lookups only; failure memory is separately audited.
    good=(~fallback[ids])&evaluated
    transformed=discrepancy(np.vstack([vs[good],ws[good]]),np.asarray(a['Z_b'])[-2:,ids[good]],
                            np.maximum(1,np.abs(np.asarray(a['Z_b'])[-2:,ids[good]])))
    init_discrepancy=discrepancy(E[:,0]+np.asarray(a['U_d'])[:,0],np.asarray(a['U1_global'])[:,0])
    invalid={k:{'nonfinite_values':int((~np.isfinite(dense(a[k]))).sum()),
                'nonzero_imaginary_values':int((np.imag(dense(a[k]))!=0).sum()),
                'complex_storage':bool(np.iscomplexobj(a[k]))} for k in ['E1','Z_a','Z_b','cnt1','cnt2','force','torque','WOB','TOB','d']}
    checks={'complete_native_horizon':completed,'finite_real_physical_history':bool(valid),
            'independent_matrices':all(z['norm_relative']<1e-10 for z in matrices.values()),
            'one_step_scaled_error':transition['maximum_scaled']<1e-8,
            'loads_scaled_error':all(z['maximum_scaled']<1e-8 for z in load_errors.values()),
            'lookup_indices_match_on_probes':lookup_indices_match,
            'transformed_scaled_error':transformed['maximum_scaled']<1e-8,
            'correction_scaled_error':lookup['correction_discrepancy']['maximum_scaled']<1e-8,
            'fallback_memory_semantics':lookup['stale_transformed_memory_exact'] and not lookup['first_lookup_failed']}
    result={'safety':s,'checks':checks,'numerical_checks_passed':all(checks.values()),'lookup':lookup,
        'matrix_reassembly':matrices,'physical_reference_residuals':residuals,
        'one_step_probe_indices':ids.tolist(),'one_step_discrepancy':transition,
        'bit_load_discrepancies':load_errors,'transformation_discrepancy':transformed,
        'initial_native_storage_vs_error_reconstruction':init_discrepancy,'arithmetic':invalid,
        'maximum_absolute_error_components':np.max(np.abs(E),axis=1).tolist(),
        'inputs':{'absolute_peak_force_N':float(np.max(np.abs(F[:-1]))),
                  'absolute_peak_torque_Nm':float(np.max(np.abs(T[:-1]))),
                  'applied_commands':n-1,'final_computed_command_unused':[complex(F[-1]).real,complex(T[-1]).real],
                  'initial_commands':[float(np.real(F[0])),float(np.real(T[0]))],
                  'initial_applied_loads':[Wd,Bd],
                  'native_load_arrays_initial_entries_are_placeholders':[float(np.real(vector(a,'WOB')[0])),float(np.real(vector(a,'TOB')[0]))]}}
    derived={'bit_axial_velocity':v,'bit_rotational_velocity':w,'four_raw_margins':m,
             'effective_gain_indices':effective,'lookup_failure_mask':fallback,
             'applied_forces':F[:-1],'applied_torques':T[:-1],
             'applied_WOB':W[:-1],'applied_TOB':B[:-1]}
    return result,derived

def compare_native(a,b):
    comparisons=0;different=[]
    for name in sorted(set(a)&set(b)):
        if name.startswith('__') or name=='Mat_data2':continue
        x,y=dense(a[name]),dense(b[name]);comparisons+=1
        if x.dtype!=y.dtype or x.shape!=y.shape or x.tobytes()!=y.tobytes():different.append(name)
    # Nested gain-table data are separately frozen by their original file checksum.
    return {'passed':not different,'numeric_fields_compared':comparisons,'different_fields':different,
            'scope':'Native MATLAB repeated trajectories and numeric workspaces, same platform; MAT header timestamps excluded.'}

def audit(root):
    root=Path(root).resolve();config,out,_=freeze(root);started=time.perf_counter();results=[]
    for case in read(out/'case_manifest.json')['cases']:
        rows=[];a=None
        for phase in config['repetitions']:
            folder=out/case['case_id']/phase;execution=read(folder/'execution.json')
            if not (folder/'native.mat').exists() or not execution.get('native_status',{}).get('native_entrypoint_completed',False):
                rows.append({'case_id':case['case_id'],'phase':phase,'numerical_checks_passed':False,
                             'failed_native_execution':execution});
                write(out/case['case_id']/'repeatability.json',{'passed':False,'reason':'native execution failed or capture absent'})
                continue
            value=loadmat(folder/'native.mat');r,derived=case_audit(value,read(folder/'native_status.json'),config)
            r.update(case_id=case['case_id'],phase=phase)
            write(folder/'audit.json',r);np.savez_compressed(folder/'derived_histories.npz',**derived);rows.append(r)
            if phase=='original':a=value
            else:
                repeat=compare_native(a,value);write(out/case['case_id']/'repeatability.json',repeat)
                del value;gc.collect()
        results.append({'case_id':case['case_id'],'runs':rows,'repeatability':read(out/case['case_id']/'repeatability.json')})
        del a;gc.collect();print('Audited',case['case_id'],[r.get('numerical_checks_passed') for r in rows],flush=True)
    summary={'cases':results,'case_count':6,'native_attempts':12,'integration_refinement_performed':False,
             'bounded_actuator_feasibility':'BLOCKED; not tested','audit_seconds':time.perf_counter()-started}
    write(out/'summary.json',summary)
    print('Independent numerical audit saved',flush=True)
