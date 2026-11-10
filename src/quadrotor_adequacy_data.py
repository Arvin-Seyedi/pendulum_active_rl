"""B1 archival provenance and causal fixed-task representations; no simulation."""
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET
import numpy as np
from src.quadrotor_archive_data import build_features
from src.quadrotor_counterfactual_v1 import commands
from src.quadrotor_pilot import read,sha

BASE=Path('results/benchmark_selection')
CONFIG='configs/experiment_12_b1_archive_v1.json'

def compact(features):
    x=np.asarray(features)
    if x.shape[-1]!=496:raise ValueError('Expected full fixed-task representation')
    return x[...,list(range(6))+[6+81*j for j in range(6)]+[492,493,494,495]].copy()

def rpm_commands(rpm,model):
    """Source reconstruction, never direct motor-submission telemetry."""
    rpm=np.asarray(rpm)
    if rpm.shape!=(300,4) or not np.isfinite(rpm).all():raise ValueError('Incomplete/nonfinite RPM archive')
    force=np.array(rpm**2)*model['KF']
    return np.c_[force[:,0]+force[:,3],force[:,1]+force[:,2]]

def same(a,b):
    a,b=np.asarray(a),np.asarray(b)
    return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()

def audit(root,upstream):
    root,upstream=Path(root),Path(upstream);base=root/BASE;b0=base/'b0_archive_v1';old=base/'mmaq_v1'
    model=read(b0/'model_constants.json');p=np.load(b0/'processed.npz');ref=p['reference']
    freeze=read(old/'phase_a_freeze.json')
    for name,h in freeze['source_sha256'].items():assert sha(root/name)==h,name
    for name,h in freeze['artifact_sha256'].items():assert sha(old/name)==h,name
    for name,h in read(old/'heldout_seal.json')['sha256'].items():assert sha(old/name)==h,name
    assert same(ref,np.load(old/'reference.npy'))
    km=float(ET.parse(upstream/'safe_control_gym/envs/gym_pybullet_drones/assets/cf2x.urdf').getroot().find('properties').attrib['km'])
    b0_rows=[];features=[];targets=[];sources={};hrows=[]
    for i in range(16):
        folder=base/'mpc80_feas_v2_study/runs'/f'{i}_nonlinear80_1000_original';a=np.load(folder/'trajectory.npz')
        u=rpm_commands(a['rpms'],model);x=build_features(a['states'],u,ref,model)
        assert same(x,p['features'][i]) and same(u,p['submitted'][i])
        mapped=[commands(v,model,km) for v in a['actions']]
        assert same(np.array([c['rpm'] for c in mapped]),a['rpms'])
        assert same(np.array([c['submitted_grouped_thrust'] for c in mapped]),u)
        b0_rows.append({'id':i,'exact_feature_rows':300,'exact_source_command_rows':300})
    held=read(old/'heldout_manifest.json');initial=np.load(old/'heldout_states.npy');screen=np.load(old/'screening_states.npy')
    assert held['seed']==120101 and held['bit_generator']=='PCG64'
    for i in range(4):
        entry=held['states'][i];assert entry['id']==i and same(np.array(entry['coordinates']),initial[i])
        assert initial[i].astype('<f8').tobytes().hex()==entry['little_endian_bytes_hex']
        assert not any(same(initial[i],v) for v in screen)
        f=old/'runs'/f'H{i}_nonlinear80_1000_original';a=np.load(f/'trajectory.npz');b=np.load(old/'runs'/f'H{i}_nonlinear80_1000_repeat/trajectory.npz')
        meta=read(f/'completion.json');z=np.load(f/'nlp_solutions.npz');params=z['parameters']
        assert meta['complete'] and meta['transitions']==300 and sha(f/'trajectory.npz')==meta['trajectory_sha256']
        assert a['states'].shape==(301,6) and a['physics_states'].shape==(6001,6)
        assert np.max(abs(a['states'][0]-initial[i]))<=1e-14
        for key in ['states','actions','clipped_actions','rpms','current_steps']:assert same(a[key],b[key]),key
        assert np.array_equal(a['current_steps'],np.arange(300))
        assert same(params[:,:6],a['states'][:-1])
        assert same(z['primal'][:,-160:].reshape(300,80,2)[:,0,:],a['actions'])
        expected=ref[np.minimum(np.arange(300)[:,None]+np.arange(81)[None,:],300)]
        assert same(params[:,6:].reshape(300,81,6),expected)
        mapped=[commands(v,model,km) for v in a['actions']]
        assert same(np.array([c['clipped'] for c in mapped]),a['clipped_actions'])
        assert same(np.array([c['rpm'] for c in mapped]),a['rpms'])
        u=rpm_commands(a['rpms'],model)
        assert same(u,np.array([c['submitted_grouped_thrust'] for c in mapped]))
        x=build_features(a['states'],u,ref,model);assert np.isfinite(x).all()
        # Features at k only use the past recorded command, never the target u_k.
        assert same(x[0,493:495],p['features'][0,0,493:495])
        for name in ['trajectory.npz','completion.json','nlp_solutions.npz']:
            sources[str((f/name).relative_to(root))]=sha(f/name)
        hrows.append({'id':'H'+str(i),'rows':300,'target_solver_identity':True,'reference_parameter_identity':True,
            'source_RPM_identity':True,'native_reset_max_abs_difference':float(abs(a['states'][0]-initial[i]).max()),
            'direct_motor_submission_stream_available':False,'history_provenance':'source reconstruction from recorded RPM, independently matches recorded request conversion',
            'time_provenance':'recorded integer control index; seconds = index/50; no wall-time synchronization assumption'})
        features.append(x);targets.append(a['actions'])
    full=p['features'];small=compact(full)
    for k in range(300):
        assert np.array_equal(np.rint(full[:,k,492]*300).astype(int),np.full(16,k))
        assert same(full[:,k,6:492],np.repeat(full[0:1,k,6:492],16,axis=0))
    report={'passed':True,'B0_bitwise_feature_reconstruction':b0_rows,'external':hrows,'external_source_sha256':sources,
        'external_global_prior_exposure':'expert/model diagnostics reviewed; not globally unseen; no prior learner-fitting role',
        'external_labels':1200,'external_initial_clusters':4,'compact_dimension':16,
        'compact_full_preview_equivalence':'only this fixed reference and nonperiodic phase; not Markov sufficiency',
        'unobserved_quantities':['H actual motor-submission arguments','physical actuator measurements'],
        'eligibility_scope':'requested-action supervision with explicitly reconstructed causal command history; no new execution or safety inference'}
    return report,{'full':full,'compact':small,'targets':p['targets'],'external_full':np.stack(features),'external_compact':compact(np.stack(features)),
        'external_targets':np.stack(targets),'reference':ref},model
