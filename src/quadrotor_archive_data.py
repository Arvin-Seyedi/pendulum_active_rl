"""Causal archive-only MPC80 features. No controller, solver or plant creation."""
from pathlib import Path
import hashlib,json,subprocess,sys,platform,importlib.metadata as metadata
import numpy as np
from src.quadrotor_pilot import read,sha

BASE=Path('results/benchmark_selection')

def validate_partitions(config):
    p=config['partitions'];train=p['training_order'];val=p['validation'];test=p['assessment']
    if len(set(train+val+test))!=16 or set(train+val+test)!=set(range(16)):raise ValueError('Partitions overlap or omit trajectories')
    if any(not 0<n<=len(train) for n in p['subset_sizes']):raise ValueError('Invalid nested subset')
    return {n:train[:n] for n in p['subset_sizes']}

def build_features(states,submitted,reference,model):
    """Only x_k, known reference, phase and executed u_(k-1) enter x_k's row."""
    states=np.asarray(states);submitted=np.asarray(submitted);reference=np.asarray(reference)
    if states.shape!=(301,6) or submitted.shape!=(300,2) or reference.shape!=(301,6):raise ValueError('Expected complete six-second trajectory')
    if not all(np.isfinite(a).all() for a in [states,submitted,reference]):raise ValueError('Nonfinite input')
    low=np.asarray(model['state_low']);high=np.asarray(model['state_high']);mid=(low+high)/2;width=high-low
    alow=np.asarray(model['action_low']);ahigh=np.asarray(model['action_high']);amid=(alow+ahigh)/2;awidth=ahigh-alow
    physical=2*(states[:-1]-mid)/width
    preview=reference[np.minimum(np.arange(300)[:,None]+np.arange(81)[None,:],300)].transpose(0,2,1)
    preview=2*(preview-mid[None,:,None])/width[None,:,None]
    previous=np.vstack([np.asarray(model['hover_input']),submitted[:-1]])
    previous=2*(previous-amid)/awidth
    reset=np.zeros((300,1));reset[0]=1
    x=np.concatenate([physical,preview.reshape(300,486),np.arange(300)[:,None]/300,previous,reset],axis=1)
    assert x.shape==(300,496)
    return x.astype(np.float32)

def runtime_identity(root,upstream):
    root,upstream=Path(root),Path(upstream);model=read(root/BASE/'mpc80_feas_v2_study/frozen.json')
    ident=read(root/BASE/'mpc80_feas_v2_preflight/native_dependency_identity.json')
    assert sys.version==model['python'] and platform.platform()==model['platform']
    assert sha(sys.executable)==ident['python_executable_sha256']
    packages={n:metadata.version(n) for n in model['packages']};assert packages==model['packages']
    records={}
    for n,item in ident['package_RECORD_identities'].items():
        d=metadata.distribution(n);f=next(f for f in d.files if str(f).endswith('.dist-info/RECORD'));records[n]=sha(d.locate_file(f));assert records[n]==item['RECORD_sha256']
    for n,item in ident['core_native_binary_identities'].items():
        d=metadata.distribution('pybullet' if n.startswith('pybullet') else n.split('/')[0]);assert sha(d.locate_file(n))==item['sha256']
    for n,h in model['upstream_files'].items():assert sha(upstream/n)==h,n
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=upstream,text=True).strip();assert revision=='6b5391d014f36fdfa0f9d22d92c77387e5274308'
    assert not subprocess.check_output(['git','status','--porcelain','--untracked-files=all'],cwd=upstream,text=True).strip()
    return {'passed':True,'python':sys.version,'executable_sha256':sha(sys.executable),'packages':packages,'RECORDs':records,'upstream_commit':revision,'upstream_files':len(model['upstream_files'])}

def load_archive(root,*,check_candidates=True):
    """Validate original records and repetitions; use only originals as samples."""
    root=Path(root);study=root/BASE/'mpc80_feas_v2_study';model=read(study/'frozen.json');ref=np.load(study/'reference.npy')
    seal=read(study/'read_only_audit/final_integrity.json')
    x=[];y=[];states=[];submitted=[];sources={};candidate_count=0
    for i in range(16):
        folder=study/'runs'/f'{i}_nonlinear80_1000_original';repeat=study/'runs'/f'{i}_nonlinear80_1000_repeat'
        a=dict(np.load(folder/'trajectory.npz'));b=np.load(repeat/'trajectory.npz');meta=read(folder/'completion.json')
        assert meta['complete'] and meta['transitions']==300
        assert a['states'].shape==(301,6) and a['actions'].shape==(300,2) and a['physics_states'].shape==(6001,6)
        for name in ['states','actions','clipped_actions','rpms','rewards','physics_states','physics_contacts','predicted_states','predicted_inputs','current_steps']:
            assert a[name].tobytes()==b[name].tobytes(),(i,name)
        assert np.array_equal(a['current_steps'],np.arange(300)) and np.isfinite(a['actions']).all()
        # Native quaternion/Euler reset may introduce roundoff; frozen request
        # coordinates and the archived raw reset are distinct representations.
        requested=np.load(root/BASE/'mmaq_v1/screening_states.npy')[i]
        assert requested.tobytes()==np.asarray(model['cases'][str(i)]).tobytes()
        assert np.max(np.abs(a['states'][0]-requested))<=1e-14
        assert not np.any(a['physics_contacts']) and np.min(a['physics_states']-model['state_low'])>=0 and np.min(np.asarray(model['state_high'])-a['physics_states'])>=0
        rows=np.fromfile(folder/'submitted_scientific.f64',dtype='<f8').reshape(6000,56);force=rows[:,17:29].reshape(6000,4,3)
        actual=np.c_[force[:,0,2]+force[:,3,2],force[:,1,2]+force[:,2,2]]
        assert actual.tobytes()==np.repeat(actual[::20],20,axis=0).tobytes()
        assert rows[:,3:5].tobytes()==np.repeat(a['actions'],20,axis=0).tobytes()
        assert np.min(actual-model['action_low'])>=0 and np.min(np.asarray(model['action_high'])-actual)>=0
        if check_candidates:
            for k in range(300):
                z=np.load(folder/'attempts'/f'{k:04d}'/'candidate.npz');params=z['parameters'].reshape(-1)
                assert params[:6].tobytes()==a['states'][k].tobytes()
                assert params[6:].reshape(6,81,order='F').tobytes()==ref[np.minimum(k+np.arange(81),300)].T.tobytes()
                assert z['primal'][-160:].reshape(2,80,order='F')[:,0].tobytes()==a['actions'][k].tobytes()
                candidate_count+=1
        for name in ['trajectory.npz','completion.json','submitted_scientific.f64']:
            rel=str((folder/name).relative_to(study));h=sha(folder/name);assert seal['artifact_sha256'][rel]==h; sources[str((folder/name).relative_to(root))]=h
        x.append(build_features(a['states'],actual[::20],ref,model));y.append(a['actions']);states.append(a['states']);submitted.append(actual[::20])
    X=np.stack(x);Y=np.stack(y);S=np.stack(states);U=np.stack(submitted)
    assert X.shape==(16,300,496) and Y.shape==(16,300,2)
    return {'features':X,'targets':Y,'states':S,'submitted':U,'reference':ref,'model':model,'provenance':{'original_trajectories':16,'unique_labels':4800,'repeated_labels_not_used':4800,'candidate_parameter_preview_action_checks':candidate_count,'source_sha256':sources,'feature_sha256':hashlib.sha256(X.tobytes()).hexdigest(),'target_sha256':hashlib.sha256(Y.tobytes()).hexdigest()}}

def normalized_targets(targets,model):
    return (np.asarray(targets)-model['action_low'])/(np.asarray(model['action_high'])-model['action_low'])

def data_only_view(data):
    """Avoid giving learner code raw solver/candidate/reward/successor archives."""
    return data['features'],normalized_targets(data['targets'],data['model'])
