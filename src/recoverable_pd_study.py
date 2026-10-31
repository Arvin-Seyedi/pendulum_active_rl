"""Additive full-grid E09 audit and gated certificate-first orchestration."""

import io
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback
import unittest

import gmpy2 as g
import gymnasium as gym
import numpy as np

from src.counterfactual import assert_bitwise_equal
from src.horizon_experiment import (
    _assert_hashes, immutable_json, load_npz, read_json, save_npz, sha256, write_json,
)
from src.horizon_study import exclusive
from src.recoverability import replay_and_measure, validate_execution
from src.recoverability_experiment import protected_manifest
from src.recoverable_pd import PILOT_IDS, case_identity, compare_archive, evaluate_gain, load_benchmark
from src.recoverable_pd_experiment import verify_completed as verify_pilot
from src.viability_experiment import _attempt

CONFIG = 'configs/experiment_09_full_study.json'
ADDITIONS = [CONFIG, 'src/recoverable_pd_study.py', 'src/recoverable_pd_study_analysis.py',
             'experiments/09_full_study.py', 'tests/test_recoverable_pd_study.py']


def settings(root):
    control = read_json(root / CONFIG)
    scientific = read_json(root / control['scientific_config'])
    pilot = root / control['accepted_pilot']
    assert control['authorized_phase'] == 'full_3230_state_study_after_reviewed_pilot'
    assert scientific == read_json(pilot / 'protocol.json')
    assert control['reuse_completed_pilot'] and control['skip_search_and_optimization_if_all_certified']
    assert control['optimization_max_states'] == 64 and control['optimization_selection_seed'] == 909003
    assert control['optimization_random_start_seed'] == 909004
    assert control['optimization_settings_source'] == 'configs/experiment_07.json'
    assert control['optimization_max_scientific_solves'] == 192
    assert control['certificate_worker_hours'] == 2 and control['workers'] == 1
    assert not any(control[k] for k in ('scientific_method_changes', 'training_data_use', 'global_viability_grid'))
    return scientific, control, pilot, root / control['output']


def select_controls(benchmark, config):
    rows = [case_identity(benchmark, sid) for sid in range(6561)]
    nominal = [r['grid_id'] for r in rows if r['group'] == 'nominal_safe']
    challenging = [r['grid_id'] for r in rows if r['group'] == 'challenging_safe' and r['grid_id'] != 44]
    c = config['certificate_control_selection']
    assert c == {'nominal':16, 'challenging':16, 'include_challenging_id':44,
                 'equilibrium_id':3280, 'nominal_seed':909001, 'challenging_seed':909002}
    a = np.random.default_rng(909001).choice(nominal, 16, replace=False).tolist()
    b = np.random.default_rng(909002).choice(challenging, 15, replace=False).tolist()
    result = sorted(a+b+[44,3280])
    assert len(result) == len(set(result)) == 33
    return result


def followup_ids(primary_ids, screening):
    return [sid for sid in primary_ids if not screening[sid]['certified_nonviable']]


def _freeze(folder):
    marker = folder / 'completion.json'
    hashes = {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob('*'))
              if p.is_file() and p != marker}
    write_json(marker, {'passed': True, 'artifacts_sha256': hashes})


def _baseline_chunk(root, output, pilot, benchmark, ids, scientific):
    folder = output / 'baseline' / f'chunk_{ids[0]:04d}_{ids[-1]:04d}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'completion.json').exists():
        _assert_hashes(folder, read_json(folder/'completion.json')['artifacts_sha256'])
        return read_json(folder/'records.json')
    originals, replays, legacy, records = [], [], [], []
    costs = {'new_steps':0, 'new_resets':0, 'new_wall_seconds':0., 'reused_pilot_ids':[]}
    for sid in ids:
        identity = case_identity(benchmark, sid)
        initial = np.array(identity['initial'], dtype=np.float64)
        seed = scientific['seeds']['environment']+sid
        if sid in PILOT_IDS:
            origin = pilot / f'state_{sid:04d}/gains/kp_8_kd_2'
            original, replay = load_npz(origin/'rollout.npz'), load_npz(origin/'replay.npz')
            diagnostics, metrics = read_json(origin/'diagnostics.json'), read_json(origin/'metrics.json')
            costs['reused_pilot_ids'].append(sid)
        else:
            original, a, diagnostics = evaluate_gain(initial, seed, 8., 2.)
            replay, b, metrics = replay_and_measure(original, initial, seed, scientific)
            costs['new_steps'] += int(a['step_attempts'])+int(b['step_attempts'])
            costs['new_resets'] += int(a['reset_attempts'])+int(b['reset_attempts'])
            costs['new_wall_seconds'] += float(a['wall_seconds'])+float(b['wall_seconds'])
        validate_execution(original, initial); assert_bitwise_equal(original, replay)
        assert diagnostics['action_regeneration_bitwise']
        comparison = compare_archive(identity, original, diagnostics)
        records.append({'state_id':sid, 'identity':identity, 'comparison':comparison,
                        'metrics':metrics, 'reused_pilot':sid in PILOT_IDS})
        originals.append(original); replays.append(replay)
        legacy.append(diagnostics['legacy_pre_action_margins'])
    arrays = {name:np.concatenate([a[name] for a in originals]) for name in originals[0]}
    repeats = {name:np.concatenate([a[name] for a in replays]) for name in replays[0]}
    save_npz(folder/'original.npz', arrays); save_npz(folder/'replay.npz', repeats)
    save_npz(folder/'identity_and_legacy.npz', {'grid_ids':np.array(ids), 'legacy_margins':np.array(legacy)})
    write_json(folder/'records.json', records); write_json(folder/'costs.json', costs)
    _freeze(folder)
    return records


def _certificate(output, pilot, benchmark, sid, scientific, budget):
    folder = output/'certificates'/f'state_{sid:04d}'
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'completion.json').exists():
        _assert_hashes(folder, read_json(folder/'completion.json')['artifacts_sha256'])
        record = read_json(folder/'summary.json')
        if not record['reused_pilot']:
            budget['spent_seconds'] += sum(a['generation_seconds']+a['generation_repeat_seconds']+a['checker_seconds']
                for t in record['trials'] for a in t['attempts'] if 'checker' in a)
        return record
    identity = case_identity(benchmark, sid)
    immutable_json(folder/'identity.json', identity)
    trials = []
    for action in scientific['interval_actions_in_order']:
        name = 'u_minus2' if action == -2 else 'u_plus2'
        direction = folder/name; attempts = []
        for precision in scientific['precision_bits_in_order']:
            target = direction/f'precision_{precision}'
            reused = sid in PILOT_IDS
            if reused and not target.exists():
                source = pilot/f'state_{sid:04d}/extrema'/name/f'precision_{precision}'
                shutil.copytree(source, target)
            if target.exists() and (target/'completion.json').exists():
                _assert_hashes(target, read_json(target/'completion.json')['artifacts_sha256'])
                record = read_json(target/'record.json'); certificate = read_json(target/'certificate.json')
            else:
                if budget['spent_seconds'] >= budget['limit_seconds']:
                    attempts.append({'resource_limit':True, 'precision_bits':precision})
                    break
                certificate, record = _attempt(target, sid, identity['initial_hex'], action, precision)
            assert record['checker']['passed'] and record['generation_repeat_exact']
            if not reused:
                budget['spent_seconds'] += record['generation_seconds']+record['generation_repeat_seconds']+record['checker_seconds']
            attempts.append(record)
            if certificate['status'] != 'boundary_unresolved': break
        final = next((a for a in reversed(attempts) if 'checker' in a), None)
        trials.append({'action':action, 'attempts':attempts, 'final':final,
                       'resource_limited':any(a.get('resource_limit',False) for a in attempts)})
    certified = [t for t in trials if t['final'] and t['final']['checker']['certified']]
    result = {'state_id':sid, 'identity':identity, 'trials':trials,
              'certified_nonviable':bool(certified), 'real_model_only':True,
              'reused_pilot':sid in PILOT_IDS, 'resource_limited':any(t['resource_limited'] for t in trials)}
    write_json(folder/'summary.json',result); _freeze(folder)
    return result


def verify_completed(root):
    root = Path(root).resolve()
    _, _, _, output = settings(root)
    verify_pilot(root)
    done = read_json(output/'completion.json')
    assert done['passed'] and done['primary_denominator'] == 3230
    _assert_hashes(output, done['artifacts_sha256'])
    _assert_hashes(root, read_json(output/'source_hashes.json'))
    _assert_hashes(root, read_json(output/'preservation_before.json')['sha256'])
    assert sha256(root/'docs/experiment_09_report.md') == done['report_sha256']
    print('Completed full E09 study verified; no trajectories, certificates or optimizations regenerated.',flush=True)


def run_study(root, resume=False, snapshot=None):
    root = Path(root).resolve()
    scientific, control, pilot, output = settings(root)
    if output.exists() and not resume: raise FileExistsError('Study exists; inspect before --resume or use --verify.')
    output.mkdir(parents=True,exist_ok=True)
    with exclusive(output):
        if (output/'completion.json').exists(): verify_completed(root); return
        try:
            _run(root, output, pilot, scientific, control, snapshot)
        except BaseException as exc:
            folder=output/'failed_checks'; folder.mkdir(exist_ok=True)
            write_json(folder/f'failure_{len(list(folder.glob("*.json")))+1:03d}.json',
                       {'type':type(exc).__name__,'message':str(exc),'traceback':traceback.format_exc()})
            raise


def _run(root, output, pilot, scientific, control, snapshot):
    verify_pilot(root)
    immutable_json(output/'scientific_protocol.json',scientific)
    immutable_json(output/'authorization.json',control)
    if not (output/'preservation_before.json').exists():
        write_json(output/'preservation_before.json',read_json(snapshot) if snapshot else protected_manifest(root,output))
    before=read_json(output/'preservation_before.json'); _assert_hashes(root,before['sha256'])
    sources=sorted(set(list(read_json(pilot/'source_hashes.json'))+ADDITIONS))
    hashes={name:sha256(root/name) for name in sources}
    immutable_json(output/'source_hashes.json',hashes)
    for name in sources:
        p=output/'source_snapshot'/name; p.parent.mkdir(parents=True,exist_ok=True)
        if not p.exists(): p.write_bytes((root/name).read_bytes())
        assert sha256(p)==hashes[name]
    immutable_json(output/'metadata.json',{'python':sys.version,'executable':sys.executable,
                   'platform':platform.platform(),'numpy':np.__version__,'gymnasium':gym.__version__,
                   'gmpy2':g.version(),'MPFR':g.mpfr_version(),
                   'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()})
    if not (output/'technical_tests.json').exists():
        suite=unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_recoverable_pd_study.py')
        stream=io.StringIO(); result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
        (output/'technical_tests.txt').write_text(stream.getvalue())
        write_json(output/'technical_tests.json',{'passed':result.wasSuccessful(),'tests':result.testsRun,
                   'failures':len(result.failures),'errors':len(result.errors),'simulator_steps':0})
    assert read_json(output/'technical_tests.json')['passed']
    benchmark=load_benchmark(root/scientific['benchmark'])
    primary=np.flatnonzero((benchmark['arrays']['margin_map']<0).ravel()).tolist()
    assert len(primary)==3230
    controls=select_controls(benchmark,control)
    immutable_json(output/'selection.json',{'primary_ids':primary,'control_ids':controls,'baseline_ids':list(range(6561))})
    baseline=[]; chunk=control['baseline_chunk_size']
    for start in range(0,6561,chunk):
        ids=list(range(start,min(start+chunk,6561)))
        baseline.extend(_baseline_chunk(root,output,pilot,benchmark,ids,scientific))
        if start%(chunk*8)==0 or ids[-1]==6560: print(f'Baseline audit complete: {ids[-1]+1}/6561',flush=True)
    write_json(output/'baseline_summary.json',baseline)
    budget={'spent_seconds':0.,'limit_seconds':control['certificate_worker_hours']*3600}
    screening={}
    for index,sid in enumerate(sorted(primary+controls)):
        record=_certificate(output,pilot,benchmark,sid,scientific,budget); screening[sid]=record
        if index%128==0 or index==len(primary+controls)-1:
            print(f'Certificate screening: {index+1}/{len(primary+controls)}, certified={sum(r["certified_nonviable"] for r in screening.values())}',flush=True)
    unresolved=followup_ids(primary,screening)
    immutable_json(output/'screening_gate.json',{'primary_denominator':3230,
                   'certified_count':3230-len(unresolved),'uncertified_ids':unresolved,
                   'search_required':bool(unresolved),'certificate_budget':budget})
    if unresolved:
        # Outcome-gated continuation is implemented only if needed; no extra
        # controller evaluation or optimization is authorized on certified states.
        write_json(output/'followup_required.json',{'uncertified_ids':unresolved,
                   'next':'Frozen eight-controller bank, then capped fixed-seed E07 fallback.'})
        print('Screen complete; authorized conditional search required for uncertified cases.',flush=True)
        return
    from src.recoverable_pd_study_analysis import finalize
    finalize(root,output,pilot,baseline,screening,primary,controls,before,hashes)
