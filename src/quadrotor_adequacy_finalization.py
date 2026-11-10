"""Read-only B1 milestone audit, preserving the original strict verifier.

Only authorized context-document evolution is handled here. Scientific source
and artifact hashes must remain identical. Historical documentation is supplied
to the unchanged verifier in a disposable source view, never written to the repo.
"""
from pathlib import Path
import hashlib,json,subprocess,tempfile
from unittest.mock import patch
from src.quadrotor_pilot import read,sha
from src.quadrotor_adequacy_data import BASE

CONTEXT={'CURRENT_STATUS.md','RESEARCH_CONTEXT.md','docs/RESEARCH_DECISION_LOG.md'}

def context_versions(data,code,snapshot):
    versions={};changed={}
    for name,h in snapshot['tracked_sha256'].items():
        current=Path(code)/name
        if sha(current)==h:continue
        if name not in CONTEXT:raise AssertionError('Historical scientific source changed: '+name)
        original=subprocess.check_output(['git','show',snapshot['HEAD']+':'+name],cwd=data)
        assert hashlib.sha256(original).hexdigest()==h,name
        versions[name]=original;changed[name]={'historical_sha256':h,'current_sha256':sha(current)}
    return versions,changed

def verify(data,code,upstream,output,full_preservation=False):
    data,code,upstream,output=map(lambda p:Path(p).resolve(),(data,code,upstream,output))
    if output.exists():raise FileExistsError(output)
    if output.is_relative_to((data/BASE).resolve()) and not output.is_relative_to((data/BASE/'b1_finalization').resolve()):
        raise ValueError('Output inside accepted evidence')
    study=data/BASE/'b1_archive_v1';integrity=read(study/'completion_integrity.json')
    for name,h in integrity['artifact_sha256'].items():assert sha(study/name)==h,name
    for name,h in integrity['source_sha256'].items():assert sha(code/name)==h,name
    snapshot=read(study/'preservation_start.json');versions,changed=context_versions(data,code,snapshot)
    from src import quadrotor_adequacy_verification as original
    from src import quadrotor_adequacy_learner as learners
    from src import quadrotor_archive_learner as old_learners
    def forbidden(*a,**kw):raise AssertionError('No scientific fitting during finalization')
    with tempfile.TemporaryDirectory(prefix='b1_frozen_document_view_') as temp:
        view=Path(temp)/'code';view.mkdir()
        names=set(snapshot['tracked_sha256'])|set(integrity['source_sha256'])|{'docs/experiment_12_b1_adequacy_design.md'}
        for name in names:
            dest=view/name;dest.parent.mkdir(parents=True,exist_ok=True)
            if name in versions:dest.write_bytes(versions[name])
            else:dest.symlink_to(code/name)
        with patch.object(learners,'fit',forbidden),patch.object(learners,'ridge_fit',forbidden),patch.object(old_learners,'fit_fixed',forbidden),patch.object(old_learners,'ridge_fit',forbidden):
            result=original.verify(data,view,upstream,study,Path(temp)/'read_only.json')
    preservation={}
    if full_preservation:
        from src.quadrotor_archive_finalization import historical_preservation
        preservation=historical_preservation(data,code,read(data/BASE/'b0_archive_v1/preservation_start.json'))
    result.update({'accepted_B1_artifacts':len(integrity['artifact_sha256']),
        'accepted_integrity_sha256':sha(study/'completion_integrity.json'),
        'authorized_context_updates':changed,'context_handling':'historical document bytes proven by Git and supplied only in disposable verifier view; live scientific code checked exactly',
        'full_preservation':preservation,'scientific_fitters_forbidden':True})
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2)+'\n');return result
