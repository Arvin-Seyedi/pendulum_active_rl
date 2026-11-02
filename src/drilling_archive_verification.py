"""Read-only E11A archive verification. Never launches MATLAB or writes the archive."""
from pathlib import Path
import gc
import hashlib
import io
import subprocess
import time
import zipfile
import numpy as np
from scipy.io import loadmat
from src.drilling_reproduction import PREFIX, cases, read, sha
from src.drilling_numerical_audit import case_audit, compare_native, dense, external, discrepancy, STATE_SCALE

CONTEXT_FILES={'CURRENT_STATUS.md','RESEARCH_CONTEXT.md','docs/RESEARCH_DECISION_LOG.md'}
HISTORICAL_COMMIT='a164fa6469132fef9f7222f1ccfa8fb3b8fadf99'


def check_hashes(base,hashes):
    base=Path(base).resolve()
    for name,expected in hashes.items():
        p=(base/name).resolve()
        if not p.is_relative_to(base):raise ValueError('Hash path escapes archive: '+name)
        assert sha(p)==expected,str(p)


def check_preservation(root,hashes):
    root=Path(root)
    for name,expected in hashes.items():
        if name in CONTEXT_FILES:
            original=subprocess.check_output(['git','show',HISTORICAL_COMMIT+':'+name],cwd=root)
            assert hashlib.sha256(original).hexdigest()==expected,name
        else:assert sha(root/name)==expected,name


def nested_equal(a,b):
    a,b=np.asarray(a),np.asarray(b)
    if a.shape!=b.shape or a.dtype!=b.dtype:return False
    if a.dtype.hasobject:return all(nested_equal(x,y) for x,y in zip(a.flat,b.flat))
    return a.tobytes()==b.tobytes()


def outside_archive(path,archive):
    if Path(path).resolve().is_relative_to(Path(archive).resolve()):
        raise ValueError('Verification output must be outside the accepted archive')


def verify_archive(project_root,code_root=None,check_prior=True):
    started=time.perf_counter();root=Path(project_root).resolve()
    code=Path(code_root or root).resolve();out=root/'results/drilling_reproduction/native_pilot'
    config=read(code/'configs/experiment_11a.json')
    assert config==read(out/'protocol.json')
    assert read(out/'case_manifest.json')['cases']==cases()
    for name in ['completion.json','final_verification.json']:
        marker=read(out/name);check_hashes(out,marker['artifact_sha256'])
        # Preserve the exact reviewed execution-time report even after authorized clarification.
        assert sha(out/'scientific_report.md')==marker['report_sha256']
    for name in ['source_hashes.json','additional_audit_sources.json']:
        hashes=read(out/name);check_hashes(code,hashes)
        check_hashes(out/'source_snapshot',hashes)
    if check_prior:check_preservation(root,read(out/'preservation_before.json')['sha256'])
    authority=read(out/'authority_preservation_before.json')
    for name,h in authority.items():assert sha(name)==h,name
    archive=(root/config['archive']).resolve();assert sha(archive)==config['archive_sha256']
    with zipfile.ZipFile(archive) as z:
        members=read(out/'case_manifest.json')['members_sha256']
        for name,h in members.items():assert hashlib.sha256(z.read(PREFIX+name)).hexdigest()==h,name
        gain_source=loadmat(io.BytesIO(z.read(PREFIX+'Mat_data2.mat')))['Mat_data2']
    summary=read(out/'summary.json');assert len(summary['cases'])==6
    records=[];comparisons=0;transitions=0;states=0;worst_abs=worst_scaled=0.
    for c in summary['cases']:
        pair=[];rows=[]
        for phase in ['original','repeat']:
            folder=out/c['case_id']/phase;execution=read(folder/'execution.json')
            assert execution['process_returncode']==0 and not execution['timeout']
            assert execution['native_status']['native_entrypoint_completed']
            check_hashes(folder,execution['artifacts_sha256'])
            request=read(folder/'request.json');assert sha(folder/'request.json')==execution['request_sha256']
            for name,h in request['copies_sha256'].items():
                assert h==members[request['source_namespace']+name],name
            a=loadmat(folder/'native.mat');assert nested_equal(a['Mat_data2'],gain_source)
            fresh,derived=case_audit(a,read(folder/'native_status.json'),config)
            saved=read(folder/'audit.json');assert fresh['numerical_checks_passed']
            assert fresh['safety']==saved['safety']
            assert fresh['lookup']['failures']==saved['lookup']['failures']
            assert fresh['lookup']['first_failure_step']==saved['lookup']['first_failure_step']
            with np.load(folder/'derived_histories.npz') as saved_arrays:
                assert set(saved_arrays.files)==set(derived)
                assert all(nested_equal(saved_arrays[k],v) for k,v in derived.items())
            bits=read(folder/'initial_state_identity.json')['binary64_hex']
            actual=np.ascontiguousarray(np.real(a['E1'][:,0]),dtype=np.float64).view(np.uint64)
            assert bits==[f'{int(x):016x}' for x in actual]
            nominal=external(a,a['force_d'],a['torque_d'],a['WOB_d'],a['TOB_d'])
            applied=np.column_stack([nominal[:,0],a['F1_global'][:,1:-1]])
            delta=np.linalg.solve(dense(a['M1_global']),applied)[::2]-a['F_M_d'];delta[:,0]=0
            prediction=a['E1'][:,:-1]+1e-4*(a['Al1']@a['E1'][:,:-1]+np.vstack([np.zeros_like(delta),delta]))
            difference=discrepancy(prediction,a['E1'][:,1:],STATE_SCALE[:,None])
            assert difference['maximum_scaled']<1e-8
            with np.load(folder/'applied_vector_histories.npz') as v:
                assert nested_equal(v['applied_external_force_vectors'],applied)
                assert nested_equal(v['reconstructed_error_load_vectors'],delta)
            worst_abs=max(worst_abs,difference['maximum_absolute']);worst_scaled=max(worst_scaled,difference['maximum_scaled'])
            native=read(out/c['case_id']/'independent_native_safety.json')
            assert native['violating_states']==fresh['safety']['violation_state_count']
            assert native['violations_by_face']==list(fresh['safety']['violations_by_face'].values())
            assert (native['first_crossing_step'] or None)==fresh['safety']['first_crossing_step']
            assert np.max(np.abs(np.array(native['minimum_margins'])-list(fresh['safety']['minimum_margins'].values())))<1e-12
            if c['case_id']=='P5':assert np.count_nonzero(a['E1'])==0
            pair.append(a);rows.append({'safety':fresh['safety'],'lookup_failures':fresh['lookup']['failures'],
                'first_lookup_failure_step':fresh['lookup']['first_failure_step'],
                'matrix_reassembly':fresh['matrix_reassembly'],
                'physical_reference_residuals':fresh['physical_reference_residuals'],
                'one_step_probe_count':len(fresh['one_step_probe_indices'])})
            transitions+=200000;states+=200001
            del derived,prediction,delta,applied;gc.collect()
        repeat=compare_native(*pair);assert repeat['passed'] and repeat['numeric_fields_compared']==85
        comparisons+=repeat['numeric_fields_compared'];records.append({'case_id':c['case_id'],'runs':rows,'repeat':repeat})
        del pair,a;gc.collect()
    native=read(out/'native_reference_residuals.json')
    r0=np.array(native['r0']);r1=np.array(native['KM_Vs']);omitted=r0-20*r1
    assert np.max(np.abs(omitted-native['physical_missing_term_at_20_seconds']))<1e-15
    assert np.max(np.abs(r0))>0 and np.max(np.abs(r1))>0
    p3=records[3]['runs'][0]
    assert p3['lookup_failures']==411 and p3['first_lookup_failure_step']==1318
    assert p3['safety']['violation_state_count']==401 and p3['safety']['first_crossing_step']==1323
    assert [r['runs'][0]['safety']['empirically_strictly_safe'] for r in records]==[True,True,True,False,True,True]
    return {'passed':True,'native_cases':6,'native_executions_verified':12,'numeric_field_comparisons':comparisons,
        'transitions_reconstructed':transitions,'raw_states_checked':states,'cases':records,
        'maximum_transition_absolute_error':worst_abs,'maximum_transition_scaled_error':worst_scaled,
        'native_reference_residuals':native,'current_report_sha256':sha(code/'docs/experiment_11a_report.md'),
        'prior_preservation_checked':check_prior,'authority_files_preserved':len(authority),
        'authorized_context_exceptions':sorted(CONTEXT_FILES),'new_MATLAB_processes':0,
        'new_plant_rollouts':0,'writes_to_accepted_archive':0,'seconds':time.perf_counter()-started,
        'scope':'Numerical reproduction passes; P3 unsafe; physical-reference identities nonzero; bounded-input feasibility blocked.'}
