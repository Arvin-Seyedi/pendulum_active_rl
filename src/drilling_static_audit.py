"""Static E11B audit of saved data. No integration or controller execution.

Exact products, interval reassembly and archived matched-state checks are kept
separate. Results never overwrite the accepted E11A archive.
"""
from fractions import Fraction as Q
from pathlib import Path
import gc
import hashlib
import json
import platform
import resource
import signal
import subprocess
import sys
import time
import zipfile
import numpy as np
import scipy
from scipy.io import loadmat, whosmat
from src.drilling_reproduction import read, write, sha, PREFIX
from src.drilling_numerical_audit import reassemble, external, loads, dense, scalar, vector
from src.drilling_static_interval import reassemble_intervals, nominal_load_intervals

STATIC = ['A','Al1','BotR','D1_global','D_M','Di','Do','E','F_M_d','G','Gravity','IC',
          'Ix','Iy','Iz','J','K1_global','K_M','Kc','M1_global','N1','P_initial','P_tail',
          'R1','RR','Raleigh_a','Raleigh_b','Rb','TOB_d','TopR','UU1','Vs','WOB_d',
          'a1','a2','angle','b1','b2','d0','deg','dt','epsilon','err','force_d',
          'fri_coef','g','gam','l','lw','m','miu','rho','sigma','threshold','torque_d','zeta']
PRIMITIVES = ['P_initial','P_tail','angle','m','l','E','G','A','Ix','Iy','Iz','J',
              'Raleigh_a','Raleigh_b','fri_coef','g','IC','Vs','force_d','torque_d',
              'WOB_d','TOB_d','Rb','epsilon','zeta','sigma','lw','Kc','gam','miu',
              'a1','a2','b1','b2','d0','threshold']
IQL = ['configs/experiment_06.json','docs/experiment_06_pilot_report.md',
       'experiments/06_offline_rl.py','requirements_experiment_06.txt','src/iql.py',
       'src/offline_data.py','src/offline_evaluation.py','src/offline_experiment.py',
       'tests/test_offline_rl.py']


def dyadic(x):
    return Q.from_float(float(x))


def qmatrix(a):
    return [[dyadic(v) for v in row] for row in np.atleast_2d(a)]


def qdot(a, x):
    return [sum(v*w for v,w in zip(row,x)) for row in a]


def ratio(x):
    return [str(x.numerator), str(x.denominator)]


def rational_residuals(a):
    k,d = qmatrix(a['K_M']),qmatrix(a['D_M'])
    v,ic,f = [list(map(dyadic,vector(a,key))) for key in ('Vs','IC','F_M_d')]
    kv,dv,ki = qdot(k,v),qdot(d,v),qdot(k,ic)
    r0 = [f[i]-dv[i]-ki[i] for i in range(10)]
    u = Q(1,2**53); gamma = 10*u/(1-10*u)
    sums = [sum(abs(x*y) for x,y in zip(row,v)) for row in k]
    return {'r0_exact':[ratio(x) for x in r0], 'KM_Vs_exact':[ratio(x) for x in kv],
            'r0_display':list(map(float,r0)), 'KM_Vs_display':list(map(float,kv)),
            'r0_all_exact_zero':all(x==0 for x in r0),
            'KM_Vs_all_exact_zero':all(x==0 for x in kv),
            'dot_gamma10_error_bounds':list(map(float,[gamma*x for x in sums])),
            'dot_absolute_term_sums':list(map(float,sums)),
            'cancellation_factors':[None if x==0 else float(s/abs(x)) for s,x in zip(sums,kv)],
            'rowwise_relative_K_perturbation_to_cancel':[None if s==0 else float(abs(x)/s) for s,x in zip(sums,kv)]}


def matrix_diagnostics(a):
    out={}
    for key in ('M1_global','K1_global','K_M','D_M'):
        b=dense(a[key]);s=np.linalg.svd(b,compute_uv=False)
        tol=max(b.shape)*np.finfo(float).eps*s[0]
        out[key]={'singular_values':s.tolist(),'rank_threshold':float(tol),
                  'numerical_rank':int((s>tol).sum()),'condition_2':float(s[0]/s[-1]),
                  'symmetry_relative':float(np.linalg.norm(b-b.T)/max(np.linalg.norm(b),np.finfo(float).tiny))}
    k,d,f,v,ic = dense(a['K_M']),dense(a['D_M']),vector(a,'F_M_d'),vector(a,'Vs'),vector(a,'IC')
    b=f-d@v;r=b-k@ic
    out['IC_balance']={'normwise_backward_error':float(np.linalg.norm(r)/(np.linalg.norm(k)*np.linalg.norm(ic)+np.linalg.norm(b))),
                       'stationarity_KT_r_norm':float(np.linalg.norm(k.T@r)),
                       'componentwise_backward_error':float(np.max(abs(r)/(abs(k)@abs(ic)+abs(b)))),
                       'normal_equation_note':'IC_check.m:85 uses normal equations; no IC recalculated or installed.'}
    out['scales']='SI coordinates normalized by 1 m, 1 rad, 1 s; ranks are threshold-dependent, not exact rank proofs.'
    return out


def coordinate_audit(a):
    angles=vector(a,'angle');t=np.zeros((20,10));s=np.eye(20)[::2];c=np.zeros((10,20))
    for i in range(5):
        for j in range(2):
            col=2*i+j;row=4*i+2*j
            t[row,col]=1;t[row+1,col]=np.tan(angles[i])
            c[col,row]=-np.tan(angles[i]);c[col,row+1]=1
    rot=np.asarray(a['R1']);block=[rot[:2,:2,i] for i in range(4)]
    # Literal printed lateral-x sign, compared with executed cross product.
    pos=np.asarray(a['P_initial'])[0,:,:3];tail=vector(a,'P_tail')[:3];literal=[]
    for i in range(4):
        d=pos[i+1]-pos[i];ch=(pos[i+2] if i<3 else tail)-pos[i]
        tangent=d/np.linalg.norm(d);normal=np.cross(d,ch);normal/=np.linalg.norm(normal)
        wrong=np.cross(normal,tangent);wrong[0]=normal[1]*tangent[2]+normal[2]*tangent[1]
        frame=np.array([tangent,wrong,normal]);literal.append(float(np.linalg.norm(frame@frame.T-np.eye(3),ord=np.inf)))
    m,k,d = [dense(a[x]) for x in ('M1_global','K1_global','D1_global')]
    true_mass=t.T@m@t
    physical_k=np.linalg.solve(true_mass,t.T@k@t)
    physical_d=np.linalg.solve(true_mass,t.T@d@t)
    f=external(a,a['force_d'],a['torque_d'],a['WOB_d'],a['TOB_d'])[:,0]
    nominal_acc=np.linalg.solve(m,f-k@t@vector(a,'IC')-d@t@vector(a,'Vs'))
    return {'ST_identity_maximum_absolute':float(abs(s@t-np.eye(10)).max()),
            'CT_constraint_maximum_absolute':float(abs(c@t).max()),
            'rotation_orthogonality_maximum_absolute':max(float(abs(r@r.T-np.eye(2)).max()) for r in block),
            'rotation_determinants':[float(np.linalg.det(r)) for r in block],
            'printed_plus_sign_frame_orthogonality_errors':literal,
            'constraint_acceleration_at_nominal_reference':(c@nominal_acc).tolist(),
            'constraint_K_operator_norm':float(np.linalg.norm(c@np.linalg.solve(m,k@t))),
            'constraint_D_operator_norm':float(np.linalg.norm(c@np.linalg.solve(m,d@t))),
            'projected_vs_ideal_constrained_K_relative':float(np.linalg.norm(physical_k-dense(a['K_M']))/np.linalg.norm(physical_k)),
            'projected_vs_ideal_constrained_D_relative':float(np.linalg.norm(physical_d-dense(a['D_M']))/np.linalg.norm(physical_d)),
            'physical_reduction_status':'UNRESOLVED: Chapter 5 asserts contact cancellation, but native specified forces do not establish the needed reaction closure; comparison with ideal constraints is diagnostic only.',
            'T':t.tolist(),'S':s.tolist(),'C':c.tolist(),
            'displacement_origin':'q=U1_global displacement; X1=q+UU1. Initial native position arrays remain nominal while E1 may be nonzero.',
            'units':'q odd:m, even:rad; velocity odd:m/s, even:rad/s; K_M maps q to acceleration; r0 angular:rad/s^2, K_M Vs angular:rad/s^3. Force N; torque N*m. Rayleigh a:1/s, b:s.',
            'handedness_convention':'Executed planar 2x2 blocks have determinant +1; full cross-product construction checked. Printed sign discrepancy retained, not corrected.'}


def interval_summary(result,a,exact):
    def arr(x):return np.array([[float(v.midpoint()) for v in row] for row in x])
    comparisons={}
    for name,key in [('M','M1_global'),('K','K1_global'),('D','D1_global'),('KM','K_M'),('DM','D_M'),('FM','F_M_d'),('gravity','Gravity')]:
        b=arr(result[name]);ref=dense(a[key])
        comparisons[name]={'relative':float(np.linalg.norm(b-ref)/max(np.linalg.norm(ref),np.finfo(float).tiny)),
                           'maximum_absolute':float(abs(b-ref).max())}
    r1=result['KM_Vs'];r0=result['r0'];v=vector(a,'Vs');native=dense(a['K_M'])
    # This bound includes native assembly/solve/rounding differences, not just dot rounding.
    perturb=[]
    for i,row in enumerate(result['KM']):
        bound=Q(0)
        for j,x in enumerate(row):
            lo,hi=Q(*x.lo.as_integer_ratio()),Q(*x.hi.as_integer_ratio())
            bound+=max(abs(dyadic(native[i,j])-lo),abs(dyadic(native[i,j])-hi))*abs(dyadic(v[j]))
        perturb.append(float(bound))
    return {'matrices':comparisons,
            'matrix_intervals':{key:[[v.record() for v in row] for row in result[key]]
                                for key in ('M','K','D','KM','DM','FM','T','gravity')},
            'r0':[row[0].record() for row in r0],
            'KM_Vs':[row[0].record() for row in r1],
            'native_K_assembly_perturbation_times_Vs_bound':perturb,
            'archived_K_dot_rounding_bound':exact['dot_gamma10_error_bounds'],
            'input_contract':'Frozen primitive binary64 data, independently reassembled with interval transcendentals and solves; not the exact unknown engineering model.'}


def archived_points(a,samples,exact):
    k,d=dense(a['K_M']),dense(a['D_M']);v,ic,f=vector(a,'Vs'),vector(a,'IC'),vector(a,'F_M_d')
    kq,dq=qmatrix(k),qmatrix(d);vq,icq,fq=[list(map(dyadic,x)) for x in (v,ic,f)]
    r0=[Q(int(n),int(den)) for n,den in exact['r0_exact']]
    r1=[Q(int(n),int(den)) for n,den in exact['KM_Vs_exact']]
    rows=[];u=Q(1,2**53);gamma=Q(100)*u/(1-Q(100)*u)
    for j,step in enumerate(samples['indices']):
        e=samples['E'][:,j];t=float(samples['t'].reshape(-1)[j]);applied=samples['F'][:,j]
        eq=list(map(dyadic,e[:10]));ev=list(map(dyadic,e[10:]));tq=dyadic(t)
        q=[eq[i]+icq[i]+vq[i]*tq for i in range(10)];vel=[ev[i]+vq[i] for i in range(10)]
        # Common force term cancels exactly; independently evaluate both linear RHS forms.
        phys=[-x-y for x,y in zip(qdot(kq,q),qdot(dq,vel))]
        impl=[-x-y-z for x,y,z in zip(qdot(kq,eq),qdot(dq,ev),fq)]
        diff=[phys[i]-impl[i] for i in range(10)]
        expected=[r0[i]-tq*r1[i] for i in range(10)]
        if diff!=expected:raise AssertionError('Exact algebraic error transformation failed')
        force=np.linalg.solve(dense(a['M1_global']),applied)[::2]
        qf=e[:10]+ic+v*t;vf=e[10:]+v
        physical=-k@qf-d@vf+force
        implemented=-k@e[:10]-d@e[10:]+(force-f)
        wanted=np.array(list(map(float,expected)))
        discrepancy=physical-implemented-wanted
        # Conservative gamma100 bound includes reference formation, all linear
        # products and subtraction. Exact cancellation of the common force is
        # evaluated independently above. No overflow/underflow in these records.
        magnitude=abs(k)@(abs(e[:10])+abs(ic)+abs(v)*abs(t))+abs(d)@(abs(e[10:])+abs(v))+2*abs(force)+abs(f)
        bound=np.array([float(gamma)*x for x in magnitude])+np.finfo(float).tiny
        scale=np.maximum.reduce([np.ones(10),abs(physical),abs(implemented),abs(wanted)])
        load=loads(a,(e[18]+v[8])/np.cos(vector(a,'angle')[-1]),(e[19]+v[9])/np.cos(vector(a,'angle')[-1]))
        rows.append({'step':int(step),'time_seconds':t,'exact_dyadic_identity':True,
            'equation_difference':(physical-implemented).tolist(),'expected_forcing':wanted.tolist(),
            'rounding_bound':bound.tolist(),'maximum_absolute_identity_error':float(abs(discrepancy).max()),
            'maximum_scaled_identity_error':float((abs(discrepancy)/scale).max()),
            'maximum_error_over_rounding_bound':float((abs(discrepancy)/bound).max()),
            'initial_native_minus_memoryless_bit_loads':None if step else
                [float(scalar(a,'WOB_d')-load['WOB']),float(scalar(a,'TOB_d')-load['TOB'])],
            'native_position_minus_error_reconstruction':(samples['U'][:10,j]-qf).tolist(),
            'X_minus_U_minus_origin_max':float(abs(samples['X'][:,j]-samples['U'][:,j]-vector(a,'UU1')).max())})
    return rows


def capture_samples(folder,indices):
    out={'indices':np.array(indices)}
    for key,target in [('E1','E'),('F1_global','F'),('U1_global','U'),('X1','X'),('t','t')]:
        x=loadmat(folder/'native.mat',variable_names=[key])[key]
        if np.any(np.imag(x)):raise ValueError('Complex archived physical array')
        out[target]=np.real(x[...,indices]).copy();del x;gc.collect()
    return out


def preservation(root,archive,out,baseline=None):
    from src.drilling_archive_verification import check_hashes
    accepted=read(archive/'authority_preservation_before.json')
    for path,h in accepted.items():
        if sha(path)!=h:raise AssertionError('Authority changed: '+path)
    for marker in ['completion.json','final_verification.json']:
        check_hashes(archive,read(archive/marker)['artifact_sha256'])
    if baseline is None:
        names=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().split('\0')
        baseline={name:sha(root/name) for name in names if name}
        # Preserve local unfinished work if present; it is not a dependency of
        # this static experiment in a clean checkout.
        baseline.update({name:sha(root/name) for name in IQL if (root/name).exists()})
        write(out/'preservation_before.json',baseline)
    else:
        for name,h in baseline.items():
            if sha(root/name)!=h:raise AssertionError('Previous research changed: '+name)
    return baseline


def enforce_limits(cfg):
    resource.setrlimit(resource.RLIMIT_CPU,(cfg['cpu_seconds_cap'],cfg['cpu_seconds_cap']))
    def rss_guard(signum,frame):
        # macOS reports ru_maxrss in bytes. A high-water check also detects a
        # short allocation peak between samples. No external process access.
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        if sys.platform!='darwin':rss*=1024
        if rss>cfg['rss_bytes_cap']:
            raise MemoryError('Approved 2 GiB RSS cap exceeded')
    signal.signal(signal.SIGALRM,rss_guard)
    signal.setitimer(signal.ITIMER_REAL,.1,.1)


def run(root,out,resume=False):
    started=time.perf_counter();cpu=time.process_time();root=Path(root);out=Path(out)
    cfg=read(root/'configs/experiment_11b.json');archive=root/cfg['accepted_evidence']
    enforce_limits(cfg)
    if resume:
        if not out.exists() or (out/'completion.json').exists():
            raise ValueError('Only an incomplete static audit may be resumed')
        if read(out/'protocol.json')!=cfg:raise AssertionError('Changed protocol')
        original=read(out/'source_identities.json')
        if sha(out/'frozen_binary64.npz')!=original['frozen_arrays_sha256']:
            raise AssertionError('Frozen operands changed')
        for name,h in original['implementation_sha256'].items():
            if sha(out/'implementation_snapshot_initial'/name)!=h:
                raise AssertionError('Initial implementation snapshot changed')
        amendments={name:{'initial_sha256':h,'current_sha256':sha(root/name)}
                    for name,h in original['implementation_sha256'].items() if sha(root/name)!=h}
        write(out/'implementation_amendments.json',{'reason':'Technical reader timestamp shape fix and safe partial-audit resumption only; no mathematical or numerical-setting change.',
              'files':amendments,'reused_precisions':[128,256]})
        for precision in (128,256):
            if max(x['relative'] for x in read(out/f'interval_{precision}.json')['matrices'].values())>cfg['matrix_relative_tolerance']:
                raise AssertionError('Existing interval result failed tolerance')
        baseline=read(out/'preservation_before.json');preservation(root,archive,out,baseline)
        a=dict(np.load(out/'frozen_binary64.npz'));exact=read(out/'exact_dyadic.json')
        finish(root,out,cfg,a,exact,baseline,started,cpu,resumed=True)
        return
    if out.exists():raise FileExistsError('Refusing to overwrite audit results: '+str(out))
    out.mkdir(parents=True);write(out/'protocol.json',cfg)
    baseline=preservation(root,archive,out)
    authority_cfg=read(root/'configs/experiment_11a.json');zip_path=(root/authority_cfg['archive']).resolve()
    if sha(zip_path)!=authority_cfg['archive_sha256']:raise AssertionError('ZIP identity')
    members=read(archive/'case_manifest.json')['members_sha256']
    with zipfile.ZipFile(zip_path) as z:
        for name,h in members.items():
            if hashlib.sha256(z.read(PREFIX+name)).hexdigest()!=h:raise AssertionError(name)
    a=loadmat(archive/'P0/original/native.mat',variable_names=STATIC)
    a={k:dense(a[k]) for k in STATIC}
    np.savez_compressed(out/'frozen_binary64.npz',**a)
    write(out/'frozen_binary64.json',{k:{'shape':list(v.shape),'hex':[float(x).hex() for x in v.flat]} for k,v in a.items()})
    write(out/'source_identities.json',{'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'archive':str(zip_path),'archive_sha256':sha(zip_path),'accepted_manifest_sha256':sha(archive/'case_manifest.json'),
        'accepted_completion_sha256':sha(archive/'completion.json'),'members_sha256':members,
        'frozen_arrays_sha256':sha(out/'frozen_binary64.npz'),'primitive_fields':PRIMITIVES,
        'implementation_sha256':{name:sha(root/name) for name in ['configs/experiment_11b.json','src/drilling_static_audit.py','src/drilling_static_interval.py','experiments/11b_static_audit.py','tests/test_drilling_static_audit.py']}})
    exact=rational_residuals(a);write(out/'exact_dyadic.json',exact)
    coords=coordinate_audit(a);write(out/'coordinates_and_mechanics.json',coords)
    if coords['ST_identity_maximum_absolute']!=0 or coords['rotation_orthogonality_maximum_absolute']>1e-10 or min(coords['rotation_determinants'])<=0:
        raise AssertionError('Unexplained executed-coordinate mismatch')
    diagnostics=matrix_diagnostics(a);write(out/'conditioning.json',diagnostics)
    low=reassemble(a);mapping={'M1_global':0,'K1_global':1,'D1_global':2,'K_M':3,'D_M':4,'Gravity':5,'R1':6}
    # Gravity is a 1-D reconstruction and a native column; reshape explicitly
    # to prevent NumPy broadcasting from inventing a 20-by-20 discrepancy.
    recon={k:float(np.linalg.norm(low[i].reshape(a[k].shape)-a[k])/max(np.linalg.norm(a[k]),np.finfo(float).tiny)) for k,i in mapping.items()}
    write(out/'binary64_reassembly.json',recon)
    if max(recon.values())>cfg['matrix_relative_tolerance']:raise AssertionError('Binary64 reconstruction tolerance')
    high={};load_reports={}
    for precision in cfg['precisions_bits']:
        stage=time.perf_counter();result=reassemble_intervals(a,precision)
        info=interval_summary(result,a,exact);info['wall_seconds']=time.perf_counter()-stage
        write(out/f'interval_{precision}.json',info);high[str(precision)]=info
        if max(x['relative'] for x in info['matrices'].values())>cfg['matrix_relative_tolerance']:
            raise AssertionError('Interval reconstruction tolerance')
        load_reports[str(precision)]={variant:{k:v.record() for k,v in nominal_load_intervals(a,precision,native_pi=use_native).items()}
            for variant,use_native in [('mathematical_pi',False),('native_binary64_pi',True)]}
        del result;gc.collect()
        print('completed interval precision',precision,flush=True)
    write(out/'nominal_load_intervals.json',load_reports)
    finish(root,out,cfg,a,exact,baseline,started,cpu)


def finish(root,out,cfg,a,exact,baseline,started,cpu,resumed=False):
    archive=root/cfg['accepted_evidence']
    points={};identical=0
    for case in cfg['cases']:
        pair=[]
        for phase in ['original','repeat']:
            folder=archive/case/phase
            static=loadmat(folder/'native.mat',variable_names=STATIC)
            if any(not np.array_equal(dense(static[k]),a[k]) for k in STATIC):
                raise AssertionError('Native static operands differ by case')
            del static
            sample=capture_samples(folder,cfg['indices'])
            sample['F'][:,0]=external(a,a['force_d'],a['torque_d'],a['WOB_d'],a['TOB_d'])[:,0]
            path=out/f'{case}_{phase}_points.npz'
            if path.exists():
                with np.load(path) as saved:
                    if any(saved[k].tobytes()!=v.tobytes() for k,v in sample.items()):
                        raise AssertionError('Previously captured points changed')
            else:np.savez_compressed(path,**sample)
            rows=archived_points(a,sample,exact);points[case+'_'+phase]=rows;pair.append(sample)
            if max(x['maximum_scaled_identity_error'] for x in rows)>cfg['rhs_scaled_tolerance']:
                raise AssertionError('Pointwise reconstruction tolerance')
            if max(x['maximum_error_over_rounding_bound'] for x in rows)>1:
                raise AssertionError('Unexplained RHS rounding discrepancy')
        if any(pair[0][key].tobytes()!=pair[1][key].tobytes() for key in pair[0]):
            raise AssertionError('Archived sample repeatability failed')
        identical+=1;del pair;gc.collect()
    write(out/'pointwise_equations.json',points)
    nominal=nominal_balance(a,cfg['nominal_times_seconds'])
    write(out/'nominal_balance.json',nominal)
    preservation(root,archive,out,baseline)
    statuses={'source_provenance':'PASS','matrix_and_coordinate_reconstruction':'PASS with documented printed-sign and initial-array distinctions',
        'algebraic_error_transformation':'PASS: physical minus implemented RHS equals (0,r0-t KM Vs)',
        'exact_consistency_identities':'FAIL: at least one exact dyadic component is nonzero in each identity',
        'numerical_conditioning':'See conditioning and interval results; numerical rank is not exact physical rank',
        'physical_constrained_reduction_justification':'UNRESOLVED: source asserts radial contact cancellation; reaction closure not established'}
    report={'statuses':statuses,'archived_pairs_identical':identical,'archived_points_checked':sum(map(len,points.values())),
        'preserved_project_files':len(baseline),'authority_files_preserved':len(read(archive/'authority_preservation_before.json')),
        'new_trajectories':0,'controller_calls':0,'wall_seconds':time.perf_counter()-started,
        'cpu_seconds':time.process_time()-cpu,'python':sys.version,'platform':platform.platform(),
        'numpy':np.__version__,'scipy':scipy.__version__,'precision_passes':cfg['precisions_bits'],
        'resumed_after_reader_fix':resumed,'high_precision_reassemblies_total':2,
        'initial_attempt_wall_seconds_upper_bound_on_CPU':read(out/'execution_limits_initial.json')['wall_seconds'] if resumed else 0,
        'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
    write(out/'summary.json',report)
    write(out/'completion.json',{'audit_completed':True,'physical_consistency_passed':False,
        'artifact_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*')
                           if p.is_file() and p.name!='execution_limits.json'}})
    print(json.dumps(report,indent=2),flush=True)
    signal.setitimer(signal.ITIMER_REAL,0)


def nominal_balance(a,times):
    c=np.cos(vector(a,'angle')[-1]);v=vector(a,'Vs');ic=vector(a,'IC')
    rock=loads(a,v[8]/c,v[9]/c)
    actual=external(a,a['force_d'],a['torque_d'],rock['WOB'],rock['TOB'])[:,0]
    fm=np.linalg.solve(dense(a['M1_global']),actual)[::2]
    stored=vector(a,'F_M_d');r0=stored-dense(a['D_M'])@v-dense(a['K_M'])@ic
    r1=dense(a['K_M'])@v
    rows=[]
    for t in times:
        direct=fm-dense(a['D_M'])@v-dense(a['K_M'])@(ic+v*t)
        rows.append({'seconds':t,'direct_physical_reference_acceleration':direct.tolist(),
            'stored_nominal_reference_acceleration':(r0-t*r1).tolist(),
            'additional_memoryless_nominal_load_acceleration':(fm-stored).tolist(),
            'identity_discrepancy':(direct-(r0-t*r1)-(fm-stored)).tolist()})
    return {'times':rows,'native_nominal_bit_loads':[scalar(a,'WOB_d'),scalar(a,'TOB_d')],
            'memoryless_bit_loads':[float(rock['WOB']),float(rock['TOB'])],
            'bit_loads_constant_on_reference':True,
            'force_N':scalar(a,'force_d'),'torque_Nm':scalar(a,'torque_d'),
            'nominal_inputs_are_not_actuator_limits':True}
