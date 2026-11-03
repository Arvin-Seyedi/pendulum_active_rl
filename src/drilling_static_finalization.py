"""Read-only E11B finalization: reproduce static arithmetic, never a trajectory.

Accepted numerical records and manifests are immutable inputs. New verification
JSON is written only to a separately specified, unused path. Code and data roots
are explicit so a clean export can verify the accepted external scientific data.
"""
from pathlib import Path
from fractions import Fraction
import gc
import hashlib
import json
import signal
import time
import zipfile
import numpy as np
from scipy.io import loadmat
from src.drilling_reproduction import read,sha,PREFIX
from src.drilling_archive_verification import check_hashes
from src.drilling_numerical_audit import dense,external
from src.drilling_static_audit import (STATIC,rational_residuals,interval_summary,
    capture_samples,archived_points,enforce_limits)
from src.drilling_static_interval import reassemble_intervals,nominal_load_intervals
from src.drilling_static_verification import verify,verification_destination


def check_sealed_evidence(archive):
    archive=Path(archive)
    check_hashes(archive,read(archive/'final_integrity.json')['artifact_sha256'])


def static_reproduction(archive,native_archive):
    archive,native_archive=Path(archive),Path(native_archive)
    cfg=read(archive/'protocol.json');a=dict(np.load(archive/'frozen_binary64.npz'))
    original=loadmat(native_archive/'P0/original/native.mat',variable_names=STATIC)
    for name in STATIC:
        if not np.array_equal(a[name],dense(original[name])):
            raise AssertionError('Native operand identity: '+name)
    del original
    exact=rational_residuals(a)
    if exact!=read(archive/'exact_dyadic.json'):raise AssertionError('Exact residual replay')
    intervals={};load_count=0
    for precision in cfg['precisions_bits']:
        assembled=reassemble_intervals(a,precision)
        reproduced=interval_summary(assembled,a,exact)
        accepted=read(archive/f'interval_{precision}.json');accepted.pop('wall_seconds')
        if reproduced!=accepted:raise AssertionError('Directed-rounding replay: '+str(precision))
        for key in ('r0','KM_Vs'):
            for component in reproduced[key]:
                lo,hi=[Fraction(int(component[k][0]),int(component[k][1])) for k in ('lo','hi')]
                if lo<=0<=hi:raise AssertionError('Reference identity not separated from zero')
        saved_loads=read(archive/'nominal_load_intervals.json')[str(precision)]
        for variant,use_native in [('mathematical_pi',False),('native_binary64_pi',True)]:
            fresh={k:v.record() for k,v in nominal_load_intervals(a,precision,native_pi=use_native).items()}
            if fresh!=saved_loads[variant]:raise AssertionError('Nominal-load replay')
            load_count+=1
        intervals[str(precision)]={'exact_endpoint_replay':True,
            'r0_maximum_component':max(abs(x['midpoint_display']) for x in reproduced['r0']),
            'KM_Vs_maximum_component':max(abs(x['midpoint_display']) for x in reproduced['KM_Vs'])}
        del assembled;gc.collect()
    accepted_points=read(archive/'pointwise_equations.json');count=0
    for case in cfg['cases']:
        for phase in ['original','repeat']:
            captured=capture_samples(native_archive/case/phase,cfg['indices'])
            captured['F'][:,0]=external(a,a['force_d'],a['torque_d'],a['WOB_d'],a['TOB_d'])[:,0]
            with np.load(archive/f'{case}_{phase}_points.npz') as saved:
                if set(saved.files)!=set(captured) or any(saved[k].tobytes()!=v.tobytes() for k,v in captured.items()):
                    raise AssertionError('Native archived-point lineage: '+case+' '+phase)
            rows=archived_points(a,captured,exact)
            if rows!=accepted_points[case+'_'+phase]:raise AssertionError('Pointwise replay')
            count+=len(rows);del captured;gc.collect()
    return {'passed':True,'interval_reassemblies':intervals,'nominal_load_variants_replayed':load_count,
            'native_points_reextracted_and_replayed':count,'exact_rational_vectors_replayed':2,
            'new_trajectories':0,'controller_calls':0}


def finalize(project_root,code_root,output):
    started=time.perf_counter();root,code=Path(project_root).resolve(),Path(code_root).resolve()
    cfg=read(code/'configs/experiment_11b.json');archive=root/cfg['output']
    output=verification_destination(output,archive)
    native=root/cfg['accepted_evidence']
    if output.is_relative_to(native.resolve()):raise ValueError('Cannot write into E11A evidence')
    if cfg!=read(archive/'protocol.json'):raise AssertionError('Changed frozen protocol')
    enforce_limits(cfg)
    check_sealed_evidence(archive)
    marker_hash=sha(archive/'final_integrity.json')
    source=read(archive/'source_identities.json')
    if sha(native/'case_manifest.json')!=source['accepted_manifest_sha256']:raise AssertionError('E11A cases changed')
    if sha(native/'completion.json')!=source['accepted_completion_sha256']:raise AssertionError('E11A completion changed')
    for marker in ('completion.json','final_verification.json'):
        check_hashes(native,read(native/marker)['artifact_sha256'])
    authority=read(code/'configs/experiment_11a.json');zip_path=(root/authority['archive']).resolve()
    if sha(zip_path)!=authority['archive_sha256']:raise AssertionError('Authority ZIP changed')
    with zipfile.ZipFile(zip_path) as z:
        for name,expected in source['members_sha256'].items():
            if hashlib.sha256(z.read(PREFIX+name)).hexdigest()!=expected:raise AssertionError(name)
    for name,expected in read(archive/'chapter_source_identities.json').items():
        if sha(name)!=expected:raise AssertionError('Chapter provenance changed: '+name)
    replay=static_reproduction(archive,native)
    # The independent checker uses fractions rather than the MPFR producer.
    numerical=verify(root,archive,report_path=output,code_root=code)
    check_sealed_evidence(archive)
    if sha(archive/'final_integrity.json')!=marker_hash:raise AssertionError('Accepted marker changed')
    result={'passed':True,'independent_verification':numerical,'static_reproduction':replay,
        'accepted_integrity_sha256':marker_hash,'writes_to_accepted_evidence':0,
        'new_MATLAB_processes':0,'new_trajectories':0,'controller_calls':0,
        'source_roots':{'code':str(code),'scientific_data':str(root)},
        'current_report_sha256':sha(code/'docs/experiment_11b_report.md'),
        'seconds':time.perf_counter()-started,
        'interpretation':'Archived numerical inconsistency confirmed; physical constrained reduction unresolved; no physical plant validation.'}
    # output is a newly created finalization record; accepted data stay untouched.
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    signal.setitimer(signal.ITIMER_REAL,0)
    return result
