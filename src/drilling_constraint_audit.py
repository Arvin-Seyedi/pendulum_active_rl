"""E11C provenance, four archived snapshots and static mathematical pilot.

No simulator, controller or alternative plant is called. Generated diagnostics
are separated from the immutable E11A/E11B scientific records.
"""
from pathlib import Path
import gc
import json
import resource
import signal
import subprocess
import time
import zipfile
import hashlib
import numpy as np
from scipy.io import loadmat
from src.drilling_reproduction import read,write,sha,PREFIX
from src.drilling_archive_verification import check_hashes,check_preservation
from src.drilling_static_audit import (IQL,capture_samples,enforce_limits,coordinate_audit,rational_residuals)
from src.drilling_numerical_audit import external,reassemble
from src.drilling_static_interval import Interval,reassemble_intervals,multiply,transpose
from src.drilling_constraint_math import (matrix,zero,add,subtract,columns,horizontal,
    approximate,maximum,record,constrained_systems,force_blocks,exact_rank,ldl_positive)

SOURCES=['configs/experiment_11c.json','requirements_experiment_11c.txt',
 'src/drilling_constraint_math.py','src/drilling_constraint_audit.py',
 'src/drilling_constraint_verification.py','experiments/11c_constraint_audit.py',
 'tests/test_drilling_constraints.py']


def preserve(root,out,baseline=None):
    root,out=Path(root),Path(out);old=root/'results/drilling_static_audit/approved_static'
    native=root/'results/drilling_reproduction/native_pilot'
    for archive,marker in [(old,'final_integrity.json'),
        (root/'results/drilling_static_audit/finalization','completion.json'),
        (native,'completion.json'),(native,'final_verification.json')]:
        check_hashes(archive,read(archive/marker)['artifact_sha256'])
    for path,h in read(native/'authority_preservation_before.json').items():
        if sha(path)!=h:raise AssertionError('Authority changed: '+path)
    for path,h in read(old/'chapter_source_identities.json').items():
        if sha(path)!=h:raise AssertionError('Chapter source changed: '+path)
    check_preservation(root,read(native/'preservation_before.json')['sha256'])
    if baseline is None:
        names=subprocess.check_output(['git','ls-files','-z'],cwd=root).decode().strip('\0').split('\0')
        baseline={n:sha(root/n) for n in names}
        baseline.update({n:sha(root/n) for n in IQL})
        expected=read(root/'results/drilling_static_audit/finalization/start_manifest.json')['iql_sha256']
        if any(baseline[n]!=h for n,h in expected.items()):raise AssertionError('IQL changed before pilot')
        write(out/'preservation_before.json',baseline)
    else:
        for name,h in baseline.items():
            if sha(root/name)!=h:raise AssertionError('Earlier research changed: '+name)
    return baseline


def freeze(root,out,cfg):
    old=root/cfg['accepted_11b'];native=root/cfg['native_11a']
    a=dict(np.load(old/'frozen_binary64.npz'));np.savez_compressed(out/'frozen_binary64.npz',**a)
    coords=coordinate_audit(a);write(out/'coordinates_and_conventions.json',coords)
    t=np.array(read(old/'coordinates_and_mechanics.json')['T']);np.save(out/'frozen_T.npy',t)
    snapshots=[];manifest=[]
    for spec in cfg['snapshots']:
        case,index=spec['case'],spec['index'];phases=[]
        for phase in ('original','repeat'):
            sample=capture_samples(native/case/phase,[index])
            if index==0:sample['F'][:,0]=external(a,a['force_d'],a['torque_d'],a['WOB_d'],a['TOB_d'])[:,0]
            channels=loadmat(native/case/phase/'native.mat',variable_names=['force','torque','WOB','TOB'])
            values=np.array([float(channels[k].reshape(-1)[index]) for k in ('force','torque','WOB','TOB')])
            sample['raw_native_channels']=values[:,None].copy()
            if index==0:values=np.array([float(a[k].item()) for k in ('force_d','torque_d','WOB_d','TOB_d')])
            sample['surface_commands']=values[:2,None];sample['bit_loads']=values[2:,None]
            del channels
            phases.append(sample)
        if any(phases[0][k].tobytes()!=phases[1][k].tobytes() for k in phases[0]):
            raise AssertionError('Archived snapshot repeat mismatch')
        sample=phases[0];name=f'{case}_{index:06d}.npz';np.savez_compressed(out/name,**sample)
        snapshots.append(sample)
        manifest.append({**spec,'file':name,'sha256':sha(out/name),'time_seconds':float(sample['t'].item()),
            'q_v_source':'Saved U1_global binary64, not recomputed affine-reference coordinates',
            'force_semantics':'Initial explicit nominal vector; native F1_global zero is a placeholder' if index==0 else
                ('Final computed force vector; unused by integration' if index==200000 else 'Computed at saved state for the following interval'),
            'original_repeat_bitwise':True})
    source=read(old/'source_identities.json')
    with zipfile.ZipFile(source['archive']) as z:
        for name,h in source['members_sha256'].items():
            if hashlib.sha256(z.read(PREFIX+name)).hexdigest()!=h:raise AssertionError(name)
    write(out/'source_identities.json',{'base_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
        'authority_zip':source['archive'],'authority_sha256':sha(source['archive']),
        'accepted_11b_integrity_sha256':sha(old/'final_integrity.json'),
        'accepted_native_completion_sha256':sha(native/'completion.json'),
        'frozen_binary64_sha256':sha(out/'frozen_binary64.npz'),
        'implementation_sha256':{n:sha(root/n) for n in SOURCES}})
    write(out/'snapshot_manifest.json',manifest)
    # Independent source reconstruction is compared with original arrays, not installed.
    arrays=reassemble(a)
    errors={key:float(np.linalg.norm(arrays[i].reshape(a[key].shape)-a[key])/np.linalg.norm(a[key]))
            for key,i in [('M1_global',0),('K1_global',1),('D1_global',2),('K_M',3),('D_M',4),('Gravity',5),('R1',6)]}
    write(out/'binary64_reconstruction.json',errors)
    if max(errors.values())>cfg['matrix_relative_tolerance']:raise AssertionError('Reconstruction tolerance')
    if coords['ST_identity_maximum_absolute']!=0 or coords['rotation_orthogonality_maximum_absolute']>1e-10 or min(coords['rotation_determinants'])<=0:
        raise AssertionError('Coordinate convention failed')
    return a,t,snapshots


def transforms(t):
    n,nq=len(t),len(t[0]);s=zero(t,nq,n);c=zero(t,nq,n)
    for i in range(nq):
        s[i][2*i]=t[0][0]*0+1
        c[i][2*i]=-t[2*i+1][i];c[i][2*i+1]=t[0][0]*0+1
    return s,c


def mappings(a,precision=None):
    gu=np.zeros((20,2));gb=np.zeros((20,2))
    gu[:8,0]=a['TopR'][:,0];gu[:8,1]=a['TopR'][:,2]
    gb[-8:,0]=-a['BotR'][:,4];gb[-8:,1]=-a['BotR'][:,6]
    gu=matrix(gu,precision);gb=matrix(gb,precision)
    if precision is not None:
        angle=Interval(float(a['angle'].reshape(-1)[-1]),precision)
        co,si=angle.function('cos'),angle.function('sin')
        gb[16][0]=-co;gb[17][0]=-si;gb[18][1]=-co;gb[19][1]=-si
    return gu,gb


def evaluate(a,t_binary,samples,precision=None):
    if precision is None:
        m,k,d=[matrix(a[key]) for key in ('M1_global','K1_global','D1_global')]
        t=matrix(t_binary);gravity=matrix(a['Gravity'])
    else:
        assembled=reassemble_intervals(a,precision)
        m,k,d,t,gravity=[assembled[key] for key in ('M','K','D','T','gravity')]
    s,c=transforms(t);gu,gb=mappings(a,precision)
    snap=[(matrix(x['U'][:10],precision),matrix(x['U'][10:],precision),matrix(x['F'],precision)) for x in samples]
    ic,vs=matrix(a['IC'],precision),matrix(a['Vs'],precision)
    nominal=matrix(samples[0]['F'],precision)
    w=force_blocks(m,k,d,t,gu,gb,gravity,snap,ic,vs,nominal)
    result=constrained_systems(m,k,d,t,s,c,w)
    result.update(Gu=gu,Gb=gb,gravity=gravity,IC=ic,Vs=vs)
    result['native_K_Vs']=multiply(columns(result['native_retained_acceleration'],0,10),vs)
    result['ideal_K_Vs']=multiply(result['K_H'],vs)
    result['ideal_acceleration_slope']=multiply(columns(result['virtual_work_acceleration'],0,10),vs)
    params=matrix(np.array([[float(a['force_d'].item())],[float(a['torque_d'].item())],
                            [float(a['WOB_d'].item())],[float(a['TOB_d'].item())]]),precision)
    source_nominal=add(multiply(horizontal(gu,gb),params),gravity)
    result['nominal_mapping_rounding_difference']=subtract(source_nominal,nominal)
    mapped=[]
    for sample in samples:
        channels=matrix(np.vstack([sample['surface_commands'],sample['bit_loads']]),precision)
        mapped.append(subtract(add(multiply(horizontal(gu,gb),channels),gravity),matrix(sample['F'],precision)))
    result['snapshot_force_mapping_rounding_difference']=horizontal(*mapped)
    rigid=np.zeros((20,2));rigid[::4,0]=1;rigid[1::4,1]=1
    result['rigid_translation_stiffness']=multiply(k,matrix(rigid,precision))
    result['rigid_translation_constraint']=multiply(c,matrix(rigid,precision))
    if precision is not None:
        result['positivity']={name:ldl_positive(result[name]) for name in ('M','M_H','K_H')}
        result['source_reconstruction']={name:float(np.linalg.norm(approximate(result[key])-a[name])/np.linalg.norm(a[name]))
            for key,name in [('M','M1_global'),('K','K1_global'),('D','D1_global')]}
    else:
        result['exact_ranks']={name:exact_rank(result[name]) for name in ('M','K','M_H','K_H')}
    return result


def serialize(result):
    return {key:(value if isinstance(value,dict) else record(value)) for key,value in result.items()}


def diagnostics(result):
    out={};a=approximate(result['native_retained_acceleration'])[:,:10]
    h=approximate(result['virtual_work_acceleration'])[:,:10]
    for name,value in [('M',result['M']),('K',result['K']),('M_H',result['M_H']),('K_H',result['K_H']),
                       ('native_K',result['native_retained_acceleration']),('reaction_metric',result['reaction_metric'])]:
        x=approximate(value)
        if name=='native_K':x=x[:,:10]
        singular=np.linalg.svd(x,compute_uv=False);tol=max(x.shape)*np.finfo(float).eps*singular[0]
        out[name]={'singular_values':singular.tolist(),'threshold':float(tol),'numerical_rank':int((singular>tol).sum()),
                   'condition_2':float(singular[0]/singular[-1]),'symmetry_relative':float(np.linalg.norm(x-x.T)/np.linalg.norm(x))}
    scale=np.diag([1000.,1.]*5);inv=np.linalg.inv(scale)
    out['scale_sensitivity']={name:float(np.linalg.cond(x)) for name,x in
        [('native_K_axial1000',scale@a@inv),('ideal_K_acceleration_axial1000',scale@h@inv),
         ('M_H_axial1000',inv@approximate(result['M_H'])@inv),('K_H_axial1000',inv@approximate(result['K_H'])@inv)]}
    _,_,vt=np.linalg.svd(a);mode=vt[-1]
    out['native_near_null_mode']={'vector':mode.tolist(),'native_residual_norm':float(np.linalg.norm(a@mode)),
        'ideal_stiffness_energy':float(mode@approximate(result['K_H'])@mode),
        'meaning':'A numerical diagnostic direction, not a new initial condition or identified physical cyclic mode.'}
    out['stiffness_operator_frobenius_relative']=float(np.linalg.norm(h-a)/np.linalg.norm(h))
    out['scope']='Fixed SI coordinates; Frobenius comparison is not physical modeling error. No matrix was substituted in the native plant.'
    return out


def run(root,out):
    root,out=Path(root),Path(out);cfg=read(root/'configs/experiment_11c.json')
    if out.exists():raise FileExistsError('Refusing to overwrite pilot evidence')
    if any(out.resolve().is_relative_to((root/cfg[k]).resolve()) for k in ('accepted_11b','native_11a')):
        raise ValueError('Output is inside accepted evidence')
    out.mkdir(parents=True);write(out/'protocol.json',cfg);enforce_limits(cfg)
    start=time.perf_counter();cpu=time.process_time();baseline=preserve(root,out)
    a,t,samples=freeze(root,out,cfg);results={}
    repeats={}
    for precision in (None,128,256):
        key='exact_dyadic' if precision is None else f'interval_{precision}'
        stage=time.perf_counter();value=evaluate(a,t,samples,precision)
        for name,cert in value['identity_certificates'].items():
            if precision is None:
                if not cert['exact_zero']:raise AssertionError('Exact identity: '+name)
            elif not cert['contains_zero'] or cert['maximum_scaled_enclosure_display']>cfg['algebra_scaled_tolerance']:
                raise AssertionError('Interval identity/tolerance: '+name)
        if precision is not None and max(value['source_reconstruction'].values())>cfg['matrix_relative_tolerance']:
            raise AssertionError('Interval reconstruction tolerance')
        force_scale=max(1.,max(float(abs(sample['F']).max()) for sample in samples))
        if maximum(value['snapshot_force_mapping_rounding_difference'])/force_scale>cfg['algebra_scaled_tolerance']:
            raise AssertionError('Unexplained source load mapping mismatch')
        encoded=serialize(value)
        hashes=[hashlib.sha256(json.dumps(encoded,sort_keys=True).encode()).hexdigest()]
        for _ in range(1,cfg['arithmetic_repetitions']):
            repeated=serialize(evaluate(a,t,samples,precision))
            if repeated!=encoded:raise AssertionError('Arithmetic repeat mismatch: '+key)
            hashes.append(hashlib.sha256(json.dumps(repeated,sort_keys=True).encode()).hexdigest())
            del repeated
        repeats[key]={'identical':True,'canonical_sha256':hashes}
        write(out/(key+'.json'),encoded);results[key]=summarize(value,cfg)
        if precision is None:write(out/'rank_and_conditioning.json',diagnostics(value))
        results[key]['seconds']=time.perf_counter()-stage
        print(key,'completed',round(results[key]['seconds'],3),'seconds',flush=True)
        del value;gc.collect()
    write(out/'archived_reduced_reference.json',rational_residuals(a))
    write(out/'repeatability.json',repeats)
    preserve(root,out,baseline)
    summary={'arithmetic':results,'snapshot_count':4,'operator_columns':25,'reference_times':[0,20],
       'new_trajectories':0,'controller_calls':0,'physical_model_implemented':False,
       'cpu_seconds':time.process_time()-cpu,'wall_seconds':time.perf_counter()-start,
       'peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'one_worker':True}
    write(out/'summary.json',summary)
    write(out/'completion.json',{'static_pilot_complete':True,'artifact_sha256':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}})
    signal.setitimer(signal.ITIMER_REAL,0);print(json.dumps(summary,indent=2),flush=True)


def summarize(value,cfg):
    native=approximate(value['native_constraint_acceleration']);retained=approximate(value['native_retained_acceleration'])
    ideal=approximate(value['virtual_work_acceleration']);delta=approximate(value['retained_reaction_acceleration'])
    groups={name:{'compatibility_max_abs':float(abs(native[:,lo:hi]).max()),
                   'retained_reaction_effect_max_abs':float(abs(delta[:,lo:hi]).max())}
            for name,(lo,hi) in cfg['operator_columns'].items()}
    snapshots=[]
    for spec,col in zip(cfg['snapshots'],cfg['snapshot_columns']):
        snapshots.append({**spec,'native_constraint_acceleration':native[:,col].tolist(),
            'native_retained_acceleration':retained[:,col].tolist(),'ideal_retained_acceleration':ideal[:,col].tolist(),
            'reaction_effect':delta[:,col].tolist(),'reaction_multipliers':approximate(value['lambda'])[:,col].tolist()})
    return {'operator_groups':groups,'snapshots':snapshots,
        'reference':{name:{'native_acceleration':retained[:,col].tolist(),'ideal_acceleration':ideal[:,col].tolist()}
                     for name,col in cfg['reference_columns'].items()},
        'native_K_Vs':approximate(value['native_K_Vs']).reshape(-1).tolist(),
        'ideal_K_Vs_force_rate':approximate(value['ideal_K_Vs']).reshape(-1).tolist(),
        'ideal_K_Vs_acceleration_rate':approximate(value['ideal_acceleration_slope']).reshape(-1).tolist(),
        'identity_certificates':value['identity_certificates'],
        'nominal_force_mapping_rounding_max_abs':maximum(value['nominal_mapping_rounding_difference']),
        'snapshot_force_mapping_rounding_max_abs':maximum(value['snapshot_force_mapping_rounding_difference']),
        'rigid_translation_stiffness_max_abs':maximum(value['rigid_translation_stiffness']),
        **({'positivity':value['positivity']} if 'positivity' in value else {'exact_ranks':value['exact_ranks']})}
