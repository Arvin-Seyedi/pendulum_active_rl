"""Immutable E11C replay and independent checks; no new plant investigation.

Legacy writers run only in disposable copies. The accepted pilot and all earlier
scientific records are read-only. Mathematical expressions and inputs are reused
unchanged; historical context bytes are checked against their original Git object.
"""
from pathlib import Path
import gc
import hashlib
import json
import shutil
import signal
import subprocess
import tempfile
import time
import numpy as np
from src.drilling_reproduction import read,sha
from src.drilling_archive_verification import check_hashes,check_preservation
from src.drilling_static_audit import enforce_limits
from src.drilling_static_verification import verification_destination
from src.drilling_constraint_audit import freeze,evaluate,serialize
from src.drilling_constraint_verification import verify,decode,mm,trans,sub,allzero
from src.drilling_constraint_report import analyze

CONTEXT_FILES={'CURRENT_STATUS.md','RESEARCH_CONTEXT.md','docs/RESEARCH_DECISION_LOG.md'}
PILOT_COMMIT='e1a2e00415f45d31810415bed66bfddc9480c454'


def historical_bytes(root,name,expected):
    """Validate live preservation; allow only the reviewed context edits."""
    path=Path(root)/name
    if sha(path)==expected:return None
    if name not in CONTEXT_FILES:raise AssertionError('Preserved file changed: '+name)
    previous=subprocess.check_output(['git','show',PILOT_COMMIT+':'+name],cwd=root)
    if hashlib.sha256(previous).hexdigest()!=expected:
        raise AssertionError('Historical context identity mismatch: '+name)
    return previous


def pilot_integrity(archive):
    archive=Path(archive)
    for name in ('completion.json','final_integrity.json','review_integrity.json'):
        check_hashes(archive,read(archive/name)['artifact_sha256'])


def check_prior(root,archive):
    """All checks are reads; no historical manifest is edited."""
    root,archive=Path(root),Path(archive)
    pilot_integrity(archive)
    baseline=read(archive/'preservation_before.json')
    for name,h in baseline.items():historical_bytes(root,name,h)
    native=root/'results/drilling_reproduction/native_pilot'
    for prior,marker in [(native,'completion.json'),(native,'final_verification.json'),
        (root/'results/drilling_static_audit/approved_static','final_integrity.json'),
        (root/'results/drilling_static_audit/finalization','completion.json')]:
        check_hashes(prior,read(prior/marker)['artifact_sha256'])
    historical=read(native/'preservation_before.json')['sha256'];check_preservation(root,historical)
    authority=read(native/'authority_preservation_before.json')
    for path,h in authority.items():
        if sha(path)!=h:raise AssertionError('Authority changed: '+path)
    for path,h in read(root/'results/drilling_static_audit/approved_static/chapter_source_identities.json').items():
        if sha(path)!=h:raise AssertionError('Chapter changed: '+path)
    return {'historical_files_checked':len(historical),'authority_files_checked':len(authority),
            'pilot_preservation_files_checked':len(baseline),'accepted_archives_unchanged':True}


def verification_view(root,code,archive,view):
    """Private hash-only root for the unchanged historical checker.

    This root is never added to sys.path. IQL bytes are preservation targets,
    not imports or numerical inputs. Changed context files use proven old bytes.
    """
    root,code,archive,view=map(Path,(root,code,archive,view))
    baseline=read(archive/'preservation_before.json')
    expected=read(archive/'source_identities.json')['implementation_sha256']
    for name,h in baseline.items():
        path=(view/name).resolve()
        if not path.is_relative_to(view.resolve()):raise ValueError('Preservation path escapes view')
        path.parent.mkdir(parents=True,exist_ok=True)
        previous=historical_bytes(root,name,h)
        if previous is None:path.symlink_to((root/name).resolve())
        else:path.write_bytes(previous)
    for name,h in expected.items():
        if sha(code/name)!=h:raise AssertionError('Accepted mathematical source changed: '+name)
        path=view/name;path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            if sha(path)!=h:raise AssertionError('Source/preservation collision: '+name)
        else:path.symlink_to((code/name).resolve())


def geometric_balance(data):
    """Verify coordinate enforcement separately from full mechanical balance."""
    m,t,c,mh,w,qn,qh=[data[k] for k in ('M','T','C','M_H','W',
        'native_retained_acceleration','virtual_work_acceleration')]
    embedded=mm(t,qn)
    if not allzero(mm(c,embedded)):raise AssertionError('Embedded geometry failed')
    needed=sub(mm(m,embedded),w);work=mm(trans(t),needed)
    if work!=mm(mh,sub(qn,qh)):raise AssertionError('Force/virtual-work identity failed')
    nonzero=[j for j in range(len(work[0])) if any(row[j]!=0 for row in work)]
    return {'geometric_enforcement_exact':True,'required_force_virtual_work_identity_exact':True,
            'nonzero_required_virtual_work_columns':nonzero,
            'scope':'The embedded native coordinates satisfy geometry. The supplied complete force balance, and compatibility with workless ideal reactions, are different tests.'}


def finalize(project_root,code_root,output):
    root,code=Path(project_root).resolve(),Path(code_root).resolve();start=time.perf_counter()
    cfg=read(code/'configs/experiment_11c.json');archive=root/cfg['output']
    output=verification_destination(output,archive)
    for name in ('accepted_11b','native_11a'):
        if output.is_relative_to((root/cfg[name]).resolve()):raise ValueError('Output in prior accepted evidence')
    if cfg!=read(archive/'protocol.json'):raise AssertionError('Changed frozen protocol')
    enforce_limits(cfg);before=sha(archive/'review_integrity.json');preservation=check_prior(root,archive)
    expected_source=read(archive/'source_identities.json')
    if expected_source['base_commit']!=PILOT_COMMIT:raise AssertionError('Unexpected historical context anchor')
    helper='src/drilling_constraint_report.py'
    if sha(code/helper)!=read(archive/'additional_analysis_sources.json')[helper]:raise AssertionError('Classifier changed')
    with tempfile.TemporaryDirectory(prefix='e11c_finalization_',dir='/private/tmp') as folder:
        folder=Path(folder);inputs=folder/'reextracted';inputs.mkdir()
        # Re-extraction reads only the same four saved original/repeat snapshots.
        a,t,samples=freeze(root,inputs,cfg)
        frozen=dict(np.load(archive/'frozen_binary64.npz'))
        if set(a)!=set(frozen) or any(a[k].tobytes()!=frozen[k].tobytes() for k in a):
            raise AssertionError('Frozen operand mismatch')
        if not np.array_equal(t,np.load(archive/'frozen_T.npy')):raise AssertionError('Frozen T changed')
        for spec,sample in zip(read(archive/'snapshot_manifest.json'),samples):
            with np.load(archive/spec['file']) as saved:
                if set(saved.files)!=set(sample) or any(saved[k].tobytes()!=v.tobytes() for k,v in sample.items()):
                    raise AssertionError('Snapshot lineage mismatch')
        replay={}
        for precision in (None,128,256):
            key='exact_dyadic' if precision is None else f'interval_{precision}'
            fresh=serialize(evaluate(a,t,samples,precision))
            if fresh!=read(archive/(key+'.json')):raise AssertionError('Static replay mismatch: '+key)
            digest=hashlib.sha256(json.dumps(fresh,sort_keys=True).encode()).hexdigest()
            repeat=read(archive/'repeatability.json')[key]
            if not repeat['identical'] or any(x!=digest for x in repeat['canonical_sha256']):
                raise AssertionError('Original repeat identity mismatch')
            replay[key]={'canonical_sha256':digest,'accepted_repeats_reproduced':2}
            del fresh;gc.collect()
        # Legacy checker/report writers operate solely in a private evidence copy.
        copy=folder/'evidence';shutil.copytree(archive,copy)
        view=folder/'hash_root';view.mkdir();verification_view(root,code,archive,view)
        independent=verify(view,copy)
        analyze(copy)
        for name in ('hypotheses.json','scientific_analysis.json','additional_exact_rank_verification.json'):
            if read(copy/name)!=read(archive/name):raise AssertionError('Scientific classification replay mismatch: '+name)
    raw=read(archive/'exact_dyadic.json')
    data={k:decode(raw[k]) for k in ('M','T','C','M_H','W','native_retained_acceleration','virtual_work_acceleration')}
    geometry=geometric_balance(data)
    if not all(col in geometry['nonzero_required_virtual_work_columns'] for col in cfg['snapshot_columns']):
        raise AssertionError('Missing snapshot mechanical-balance witness')
    analysis=read(archive/'scientific_analysis.json')
    if len(analysis['operator_block_tests'])!=5 or len(analysis['snapshot_results'])!=4:raise AssertionError('Wrong cohort')
    if any(x['status']!='REJECTED_FOR_DECLARED_MODEL' for x in analysis['operator_block_tests'].values()):
        raise AssertionError('Operator classification changed')
    pilot_integrity(archive)
    if sha(archive/'review_integrity.json')!=before:raise AssertionError('Accepted integrity marker changed')
    result={'passed':True,'static_arithmetic_replay':replay,'snapshot_pairs_reextracted':4,
        'operator_groups_verified':5,'independent_verification':independent,'geometric_balance':geometry,
        'preservation':preservation,'accepted_review_integrity_sha256':before,
        'writes_to_accepted_archives':0,'new_MATLAB_processes':0,'new_trajectories':0,'controller_calls':0,
        'code_root':str(code),'scientific_data_root':str(root),'seconds':time.perf_counter()-start,
        'current_report_sha256':sha(code/'docs/experiment_11c_report.md'),
        'physical_applicability':'UNRESOLVED; further physical development paused pending external clarification'}
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    signal.setitimer(signal.ITIMER_REAL,0)
    return result
