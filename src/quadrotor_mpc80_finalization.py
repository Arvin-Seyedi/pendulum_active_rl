"""Read-only MPC80 milestone verification with explicit scientific-data roots.

No historical seal or ledger is rewritten. Only the three research-context
files may differ, proved against their exact historical Git bytes.
"""
from pathlib import Path
from unittest.mock import patch
import hashlib
import json
import subprocess
import time
from src import quadrotor_finalization as archive
from src import quadrotor_mpc80_feasibility_v1 as ledger
from src import quadrotor_mpc80_report_v2 as report
from src.quadrotor_mmaq_v1 import tagged

BASE=Path('results/benchmark_selection')


def verify_snapshot(data,code,upstream,snapshot):
    changed=[]
    for group in ['tracked_sha256','all_prior_result_and_data_sha256']:
        for name,expected in snapshot[group].items():
            path=archive.preserved_path(data,code,name)
            if archive.sha(path)==expected:continue
            if name not in archive.CONTEXT:raise AssertionError('Preserved file changed: '+name)
            old=subprocess.check_output(['git','show',snapshot['project_head']+':'+name],cwd=data)
            if hashlib.sha256(old).hexdigest()!=expected:raise AssertionError('Unproved historical context bytes')
            changed.append(name)
    old=archive.preservation(data,code,upstream)
    return {'passed':True,'tracked_files':len(snapshot['tracked_sha256']),
        'prior_result_and_data_files':len(snapshot['all_prior_result_and_data_sha256']),
        'authority_files':old['authority_files_checked'],'upstream_files':old['upstream_files_checked'],
        'IQL_files':len(archive.IQL),'accepted_artifacts':old['accepted_artifacts_checked']},sorted(changed)


def verify(data,code,upstream,output):
    data,code,upstream,output=map(lambda p:Path(p).resolve(),(data,code,upstream,output))
    if output.exists():raise FileExistsError(output)
    accepted=data/BASE
    for name in ['mpc80_feas_v1_preflight','mpc80_feas_v1_study','mpc80_feas_v2_preflight','mpc80_feas_v2_study']:
        if output.is_relative_to(accepted/name):raise ValueError('Output inside accepted evidence')
    started=time.process_time();wall=time.perf_counter()
    study=accepted/'mpc80_feas_v2_study';pref=accepted/'mpc80_feas_v2_preflight'
    integrity=archive.read(study/'read_only_audit/final_integrity.json')
    for name,h in integrity['artifact_sha256'].items():assert archive.sha(study/name)==h,name
    for name,h in integrity['additional_reporting_source_sha256'].items():assert archive.sha(code/name)==h,name
    snapshot=archive.read(study/'preservation_start.json');changed=[]
    def preservation(root,up,snap):
        assert snap==snapshot
        checked,updates=verify_snapshot(data,code,upstream,snapshot);changed.extend(updates)
        return checked
    # Only preservation path routing changes for clean exports/context edits;
    # every numerical audit and accepted experimental source remains unchanged.
    with patch.object(ledger,'verify_preservation',preservation):
        scientific=report.summarize(code,upstream,pref,study)
    original=archive.read(study/'read_only_audit/scientific_results.json')
    # The accepted report records ledger usage BEFORE its two read-only audits.
    # Prove the later charges rather than rewriting either historical snapshot.
    initial=original['shared_resource_accounting_at_audit_start']
    audited=archive.read(study/'read_only_audit/audit_resources.json')
    current=scientific['shared_resource_accounting_at_audit_start']
    assert current['solver_calls']==initial['solver_calls']==9600
    assert current['physics_steps']==initial['physics_steps']==576000
    for key,a,b in [('cpu_seconds','CPU_seconds_charged','post_audit_CPU_seconds_charged'),
                    ('wall_seconds','active_wall_seconds_charged','post_audit_active_wall_seconds_charged')]:
        assert abs(current[key]-initial[key]-audited[a]-integrity[b])<1e-8
    scientific['shared_resource_accounting_at_audit_start']=initial
    mismatches=[k for k in original if tagged(scientific[k])!=original[k]]
    if mismatches:
        output.parent.mkdir(parents=True,exist_ok=True)
        output.with_suffix('.mismatch.json').write_text(json.dumps({'mismatches':mismatches},indent=2)+'\n')
        raise AssertionError('Reconstructed scientific results differ: '+repr(mismatches))
    assert scientific['new_expert_feasibility_gate_passed']
    assert scientific['complete_initial_state_pairs']==16 and scientific['qualified_scientific_executions']==32
    assert scientific['independently_accepted_candidates']==9600
    assert scientific['actual_physics_steps']==576000
    assert all(r['prefix_state_violation_samples']==r['prefix_contacts']==0 for r in scientific['records'])
    encoded=json.dumps(tagged(scientific),sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    result={'passed':True,'scientific_results':scientific,
        'canonical_scientific_sha256':hashlib.sha256(encoded).hexdigest(),
        'accepted_v2_files_hashed':len(integrity['artifact_sha256']),
        'authorized_context_updates':sorted(set(changed)),
        'historical_context_commit':snapshot['project_head'],
        'resource_ledger_preserved_sha256':archive.sha(study/'resource_ledger.json'),
        'new_solves':0,'new_physics_steps':0,
        'read_only_CPU_seconds':time.process_time()-started,'read_only_wall_seconds':time.perf_counter()-wall}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(tagged(result),indent=2,allow_nan=False)+'\n')
    return result
