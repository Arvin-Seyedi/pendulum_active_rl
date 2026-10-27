"""Resumable full-study execution; all scientific routines are reused unchanged."""

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import asdict, replace
import fcntl
import io
from pathlib import Path
import platform
import subprocess
import time
import traceback
import unittest

import gymnasium as gym
import matplotlib
import numpy as np
import torch

from src.active_query import (CollectionFailure, budget_prefix, collection_summary,
    nearest_d0_distance, rank_candidates, sample_states, seed_streams)
from src.acquisition_analysis import coverage_diagnostics
from src.ensemble import (EnsembleConfig, configure_determinism, load_member,
    member_seeds, predict_ensemble, save_member)
from src.expert import pd_controller
from src.horizon_experiment import (array_digest, assert_same_arrays, canonical,
    condition_name, immutable_json, load_npz, log, paired_contrasts, read_json,
    save_npz, sha256, write_json, _assert_hashes, _protected_manifest)
from src.policy_evaluation import audit_policy, evaluate_rollouts, rollout_metrics
from src.ragged_ensemble import combine_datasets, fit_ragged_member
from src.rollout_horizon import (collect_metered, episode_slices, expert_safety_audit,
    operational_summary, selection_diagnostics, shared_prefix_check,
    summarize_safety_audit, validate_horizon_collection)

SEEDS = tuple(range(52, 62))
IGNORED = {'artifact_hashes.json', 'execution.log', 'worker_stdout.log', '.lock'}


@contextmanager
def exclusive(folder):
    """Kernel releases this lock after a crash; a second runner cannot overlap."""
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / '.lock').open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def frozen_npz(path, arrays):
    if path.exists():
        assert_same_arrays(load_npz(path), arrays)
    else:
        save_npz(path, arrays)


def manifest(folder):
    return {str(p.relative_to(folder)): sha256(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.name not in IGNORED and '.tmp.' not in p.name}


def completed(folder):
    """Require both completion marker and a valid immutable artifact manifest."""
    path = folder / 'artifact_hashes.json'
    if not path.exists():
        return False
    _assert_hashes(folder, read_json(path))
    assert read_json(folder / 'checks.json')['all_passed']
    return True


def environment_identity():
    return {'python': platform.python_version(), 'numpy': np.__version__,
            'torch': str(torch.__version__), 'gymnasium': gym.__version__,
            'matplotlib': matplotlib.__version__, 'platform': platform.platform()}


def verify_evaluation(arrays, initial_conditions):
    n = len(initial_conditions)
    assert arrays['raw_states'].shape == (n, 201, 2)
    assert arrays['actions'].shape == (n, 200, 1)
    np.testing.assert_array_equal(arrays['initial_conditions'], initial_conditions)
    raw = arrays['raw_states']
    margins = np.minimum(1 - np.abs(raw[..., 0]), 1 - np.abs(raw[..., 1]) / 3)
    np.testing.assert_array_equal(arrays['margins'], margins)
    np.testing.assert_array_equal(arrays['episode_returns'], arrays['rewards'].sum(axis=1))
    np.testing.assert_array_equal(arrays['episode_min_margins'], margins.min(axis=1))
    assert not arrays['terminated'].any()
    assert arrays['truncated'][:, -1].all() and not arrays['truncated'][:, :-1].any()
    assert np.max(np.abs(arrays['actions'])) <= 2
    assert all(np.isfinite(v).all() for v in arrays.values())
    return rollout_metrics(arrays)


def cached_evaluation(folder, models, held, reset_seed):
    metrics = {}
    for region in ('wide', 'nominal'):
        path = folder / f'{region}_rollouts.npz'
        if path.exists():
            arrays = load_npz(path)
        else:
            arrays, _ = evaluate_rollouts(models, held[region], reset_seed, horizon=200)
            save_npz(path, arrays)
        metrics[region] = verify_evaluation(arrays, held[region])
    if models is not None:
        # Cheap independent loaded-checkpoint inference; no environment replay.
        arrays, values = audit_policy(models, held['audit'])
        frozen_npz(folder / 'imitation_audit.npz', arrays)
        metrics['audit'] = values
    immutable_json(folder / 'metrics.json', metrics)
    return metrics


def verify_audit(audit, costs, pool, ids):
    np.testing.assert_array_equal(audit['audited_candidate_ids'], ids)
    np.testing.assert_array_equal(audit['initial_conditions'], pool[ids])
    assert int(costs['step_attempts']) == len(audit['states'])
    assert int(costs['reset_attempts']) == len(ids)
    for key, raw in [('margins', 'raw_states'), ('next_margins', 'raw_next_states')]:
        states = audit[raw]
        np.testing.assert_array_equal(audit[key], np.minimum(1-np.abs(states[:, 0]), 1-np.abs(states[:, 1])/3))
    np.testing.assert_array_equal(audit['safety_stop'], audit['next_margins'] < 0)
    assert (audit['margins'] >= 0).all() and not audit['terminated'].any()
    np.testing.assert_allclose(audit['actions'][:, 0], pd_controller(*audit['states'].T), rtol=0, atol=1e-7)
    for i, (a, b) in enumerate(zip(audit['episode_offsets'][:-1], audit['episode_offsets'][1:])):
        assert 1 <= b-a <= 200
        assert not audit['safety_stop'][a:b-1].any()
        np.testing.assert_array_equal(audit['raw_states'][a], pool[ids[i]])
        np.testing.assert_array_equal(audit['raw_states'][a+1:b], audit['raw_next_states'][a:b-1])
        np.testing.assert_array_equal(audit['step_ids'][a:b], np.arange(b-a))
        event = audit['first_violation_step'][i]
        if audit['safety_stop'][b-1]: assert event == b-a
        else: assert event == -1 and b-a == 200
    assert all(np.isfinite(v).all() for v in audit.values())


def verify_member(model, payload, data, config, seeds):
    record = payload['record']
    assert canonical(payload['config']) == canonical(asdict(config))
    assert record['seeds'] == seeds
    ids, oob = record['bootstrap_ids'], record['oob_ids']
    lengths = np.diff(data[2])
    assert not set(ids) & set(oob)
    assert sorted(set(ids) | set(oob)) == list(range(len(lengths)))
    assert record['dataset_episode_lengths'] == lengths.tolist()
    assert record['training_transitions_with_multiplicity'] == int(lengths[ids].sum())
    assert record['oob_transitions'] == int(lengths[oob].sum())
    assert record['best_oob_normalized_mse'] == min(record['history']['oob_mse'])
    assert all(np.isfinite(v).all() for v in record['history'].values())
    assert all(torch.isfinite(v).all() for v in model.state_dict().values())


def cached_training(folder, name, data, config):
    """Same trainer and seeds as the pilot; publish each checkpoint atomically."""
    target = folder / 'training' / name
    target.mkdir(parents=True, exist_ok=True)
    immutable_json(target / 'identity.json', {'data_sha256': array_digest(*data), 'config': asdict(config)})
    models, histories = [], []
    for i, seeds in enumerate(member_seeds(config)):
        path = target / f'member_{i}.pt'
        cost_path = target / f'member_{i}_cost.json'
        if not path.exists():
            log(folder, f'Training {name}, member {i + 1}/5')
            start = time.perf_counter()
            trained, record = fit_ragged_member(*data, config, seeds)
            elapsed = time.perf_counter() - start
            temporary = target / f'member_{i}.tmp.pt'
            save_member(temporary, trained, config, record)
            restored, payload = load_member(temporary)
            verify_member(restored, payload, data, config, seeds)
            assert record == payload['record']
            for key in trained.state_dict():
                assert torch.equal(trained.state_dict()[key], restored.state_dict()[key])
            np.testing.assert_array_equal(predict_ensemble([trained], data[0][:128])['raw_predictions'],
                                          predict_ensemble([restored], data[0][:128])['raw_predictions'])
            lengths, ids = np.diff(data[2]), np.asarray(record['bootstrap_ids'])
            # Cost first, checkpoint last: a published checkpoint always has cost metadata.
            write_json(cost_path, {'elapsed_seconds': elapsed,
                'optimizer_updates': record['epochs_run'] * int(np.ceil(record['training_transitions_with_multiplicity'] / config.batch_size)),
                'bootstrap_D0_transitions': int(lengths[ids[ids < 100]].sum()),
                'bootstrap_new_transitions': int(lengths[ids[ids >= 100]].sum())})
            temporary.replace(path)
        assert cost_path.exists()
        model, payload = load_member(path)
        verify_member(model, payload, data, config, seeds)
        models.append(model)
        histories.append(payload['record'])
    immutable_json(target / 'histories.json', histories)
    return models


def initialize(root, output):
    """Freeze the approved design and preserve the entire accepted pilot."""
    pilot = root / 'results/rollout_horizon/pilot_seed41'
    assert read_json(pilot / 'checks.json')['all_passed']
    assert read_json(pilot / 'independent_verification.json')['all_passed']
    _assert_hashes(pilot, read_json(pilot / 'artifact_hashes.json'))
    old = read_json(pilot / 'metadata.json')
    _assert_hashes(root, old['protected_hashes'])
    _assert_hashes(root, old['source_hashes'])
    config = read_json(root / 'configs/experiment_05.json')
    assert config == read_json(pilot / 'protocol.json')
    launch = read_json(root / 'configs/experiment_05_full_study.json')
    assert launch['master_seeds'] == list(SEEDS) and not launch['methodology_changes']
    assert sha256(pilot / 'protocol.json') == launch['approved_pilot_protocol_sha256']
    training = asdict(EnsembleConfig()); training.pop('seed')
    assert config['training'] == canonical(training)
    assert sha256(root / config['D0']['path']) == config['D0']['sha256']
    assert output == root / 'results/rollout_horizon/full_study_52_61'
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / 'protocol.json', config)
    immutable_json(output / 'authorization.json', launch)
    sources = sorted([*root.glob('src/*.py'), *root.glob('experiments/*.py'),
                      *root.glob('tests/*.py'), *root.glob('configs/*.json')])
    metadata = {'phase': 'full_study', 'master_seeds': list(SEEDS), 'pilot_excluded': True,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
        'protocol_sha256': sha256(output / 'protocol.json'),
        'authorization_sha256': sha256(output / 'authorization.json'),
        'source_hashes': {str(p.relative_to(root)): sha256(p) for p in sources},
        'protected_hashes': _protected_manifest(root, output),
        'versions': environment_identity(), 'evaluation_seeds': config['evaluation']['full_study_seeds']}
    immutable_json(output / 'metadata.json', metadata)
    for path in sources:
        dest = output / 'source_snapshot' / path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists(): dest.write_bytes(path.read_bytes())
        assert sha256(dest) == sha256(path)
    if not (output / 'technical_tests.txt').exists():
        stream = io.StringIO()
        tests = unittest.TextTestRunner(stream=stream, verbosity=2).run(
            unittest.defaultTestLoader.discover(str(root / 'tests')))
        (output / 'technical_tests.txt').write_text(stream.getvalue())
        assert tests.wasSuccessful(), stream.getvalue()
        write_json(output / 'technical_checks.json', {'all_passed': True, 'tests': tests.testsRun})
    assert read_json(output / 'technical_checks.json')['all_passed']
    configure_determinism(52)
    e, seeds = config['evaluation'], config['evaluation']['full_study_seeds']
    held = {'wide': sample_states(seeds['wide'], e['wide_episodes']),
            'nominal': sample_states(seeds['nominal'], e['nominal_episodes'], e['nominal_low'], e['nominal_high']),
            'audit': sample_states(seeds['audit'], e['audit_states'])}
    frozen_npz(output / 'held_out_states.npz', held)
    cached_evaluation(output / 'PD_reference', None, held, seeds['resets'])
    return config


def run_seed(root, output, seed):
    root, output = Path(root), Path(output)
    assert seed in SEEDS, 'Pilot and other seeds are forbidden in the full-study runner.'
    folder = output / f'seed_{seed}'
    with exclusive(folder):
        if completed(folder): return seed
        with (folder / 'worker_stdout.log').open('a', buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
            try:
                return _run_seed(root, output, folder, seed)
            except BaseException as exc:
                # Keep every failed attempt; never silently replace or omit a seed.
                write_json(folder / f'failure_{time.time_ns()}.json',
                           {'seed': seed, 'error': str(exc), 'traceback': traceback.format_exc()})
                raise


def _run_seed(root, output, folder, seed):
    start = time.monotonic()
    config, study = read_json(output / 'protocol.json'), read_json(output / 'metadata.json')
    assert environment_identity() == study['versions']
    _assert_hashes(root, study['source_hashes'])
    assert sha256(root / config['D0']['path']) == config['D0']['sha256']
    configure_determinism(seed)
    seeds = seed_streams(seed)
    sc, tc = (replace(EnsembleConfig(), seed=seeds[k]) for k in ('selector', 'retraining'))
    immutable_json(folder / 'metadata.json', {'master_seed': seed, 'seed_streams': seeds,
        'study_metadata_sha256': sha256(output / 'metadata.json'),
        'selector_config': asdict(sc), 'training_config': asdict(tc),
        'selector_member_seeds': member_seeds(sc), 'training_member_seeds': member_seeds(tc),
        'torch_threads': torch.get_num_threads(), 'deterministic': torch.are_deterministic_algorithms_enabled()})
    d0 = load_npz(root / config['D0']['path'])
    base = combine_datasets(d0)
    selector = cached_training(folder, 'selector', base, sc)
    pool = sample_states(seeds['pool'], config['candidate_count'])
    rankings, scores = rank_candidates(pool, selector, d0['states'], seeds)
    repeated_rankings, repeated_scores = rank_candidates(pool, selector, d0['states'], seeds)
    assert_same_arrays(rankings, repeated_rankings)
    assert_same_arrays(scores, repeated_scores)
    frozen_npz(folder / 'candidates.npz', dict(initial_conditions=pool, **scores,
               **{f'ranking_{s}': r for s, r in rankings.items()}))
    streams, prefixes, diagnostics = {}, {}, {}
    for strategy in config['strategies']:
        for horizon in config['collection_horizons']:
            arm = f'{strategy}_H{horizon}'
            target = folder / 'acquisition' / arm
            target.mkdir(parents=True, exist_ok=True)
            maximum = target / 'budget_10000.npz'
            if (target / 'collection_failure.json').exists():
                raise RuntimeError(f'Unresolved scientific collection failure: {target}')
            if not maximum.exists():
                try:
                    data, costs = collect_metered(pool, rankings[strategy], 10000, seeds['environment'], horizon)
                except CollectionFailure as exc:
                    save_npz(target / 'incomplete_stream.npz', exc.partial)
                    write_json(target / 'collection_failure.json', {'error': str(exc), 'attempts': exc.attempts})
                    raise
                save_npz(target / 'operation_costs.npz', costs)
                save_npz(maximum, data)
            data, costs = load_npz(maximum), load_npz(target / 'operation_costs.npz')
            validate_horizon_collection(data, 10000, horizon)
            assert int(costs['step_attempts']) == 10000
            assert int(costs['reset_attempts']) == len(data['initial_conditions'])
            streams[arm] = data
            coverage_path = target / 'coverage.npz'
            if not coverage_path.exists():
                save_npz(coverage_path, {f'{key}_distances': nearest_d0_distance(data[field], d0['states'])
                    for key, field in [('pre_action', 'states'), ('next_state', 'next_states'), ('initial_state', 'initial_conditions')]})
            distances = load_npz(coverage_path)
            for budget in config['budgets']:
                name = condition_name(strategy, horizon, budget)
                prefix = budget_prefix(data, budget)
                validate_horizon_collection(prefix, budget, horizon)
                frozen_npz(target / f'budget_{budget}.npz', prefix)
                prefixes[name] = prefix
                d = distances['pre_action_distances'][:budget]
                coverage = coverage_diagnostics(prefix, d, d0['states'])
                selected, selection = selection_diagnostics(prefix, rankings, scores, strategy)
                frozen_npz(target / f'selected_initial_conditions_{budget}.npz', selected)
                immutable_json(target / f'trajectory_coverage_{budget}.json', coverage)
                diagnostics[name] = {'collection': collection_summary(prefix, d), 'coverage': coverage['summary'],
                    'selection': selection, 'operations': operational_summary(prefix, costs, horizon)}
            log(folder, f'{arm}: 10000 transitions, {len(data["initial_conditions"])} resets')
    shared = {f'{s}_{b}': shared_prefix_check(prefixes[condition_name(s, 20, b)], prefixes[condition_name(s, 200, b)])
              for s in config['strategies'] for b in config['budgets']}
    immutable_json(folder / 'acquisition_diagnostics.json', diagnostics)
    models = {'D0': cached_training(folder, 'D0', base, tc)}
    for name, additional in prefixes.items():
        data = combine_datasets(d0, additional)
        assert len(data[0]) == 20000 + len(additional['states'])
        models[name] = cached_training(folder, name, data, tc)
    held = load_npz(output / 'held_out_states.npz')
    known = set(map(tuple, np.concatenate([d0['states'], pool, *[v['states'] for v in streams.values()]])))
    for states in held.values(): assert not known.intersection(map(tuple, states))
    metrics = {'PD': read_json(output / 'PD_reference/metrics.json')}
    for name, ensemble in models.items():
        log(folder, f'Evaluating {name}: full 200 steps on shared held-out states')
        metrics[name] = cached_evaluation(folder / 'evaluation' / name, ensemble, held, config['evaluation']['full_study_seeds']['resets'])
    immutable_json(folder / 'metrics.json', metrics)
    immutable_json(folder / 'paired_differences.json', {str(b): paired_contrasts(metrics, b) for b in config['budgets']})
    # Freeze EVERY condition before any post-selection expert safety audit.
    paths = [folder / 'candidates.npz', folder / 'metrics.json',
             *[p for sub in ('acquisition', 'training', 'evaluation') for p in (folder / sub).rglob('*') if p.is_file() and '.tmp.' not in p.name]]
    frozen = {str(p.relative_to(folder)): sha256(p) for p in sorted(paths)}
    audit_folder = folder / 'post_selection_safety_audit'
    immutable_json(audit_folder / 'frozen_inputs_before_audit.json', frozen)
    ids = sorted(set(cid for s in config['strategies'] for h in config['collection_horizons']
                     for cid in episode_slices(prefixes[condition_name(s, h, 5000)])))
    audit_path = audit_folder / 'audit_trajectories.npz'
    if not audit_path.exists():
        log(folder, f'Frozen acquisition/training/evaluation. Separate expert audit: {len(ids)} initial conditions')
        audit, costs = expert_safety_audit(pool, ids, seeds['environment'])
        save_npz(audit_folder / 'operation_costs.npz', costs)
        save_npz(audit_path, audit)
    audit, costs = load_npz(audit_path), load_npz(audit_folder / 'operation_costs.npz')
    verify_audit(audit, costs, pool, ids)
    safety = {condition_name(s, h, 5000): summarize_safety_audit(prefixes[condition_name(s, h, 5000)], audit, h)
              for s in config['strategies'] for h in config['collection_horizons']}
    immutable_json(audit_folder / 'condition_summaries.json', safety)
    _assert_hashes(folder, frozen)
    immutable_json(audit_folder / 'isolation_checks.json', {'all_frozen_files_unchanged': True,
        'frozen_file_count': len(frozen), 'all_audit_physical_prefixes_match_collection': True,
        'no_continuation_data_in_training': True, 'late_denominator_excludes_budget_cutoffs': True})
    immutable_json(folder / 'cost_accounting.json', {'acquisition_transitions': 60000,
        'post_selection_safety_audit_transitions': int(costs['step_attempts']),
        'post_selection_safety_audit_resets': int(costs['reset_attempts']),
        'policy_evaluation_environment_steps': 19 * 1200 * 200,
        'shared_PD_reference_steps_counted_once_for_study': 1200 * 200,
        'unique_trained_members': 100, 'full_study_technical_simulation_replays': 0,
        'pilot_replay_verification_preserved': True,
        'interrupted_work_note': 'Successful published work counts; interrupted uncommitted work may incur unmeasured overhead. Logs preserve interruptions.'})
    _assert_hashes(root, study['source_hashes'])
    _assert_hashes(root, study['protected_hashes'])
    write_json(folder / 'checks.json', {'all_passed': True, 'seed': seed,
        'exact_budget_datasets': 18, 'saved_member_checkpoints': 100, 'collection_failures': 0,
        'shared_candidate_physical_prefixes': shared, 'fixed_configuration_paired_seeds': True,
        'whole_episode_bootstrap_and_oob': True, 'atomic_checkpoint_reload_integrity': True,
        'candidate_scores_and_rankings_bitwise_repeat': True, 'audit_raw_margins_and_event_times': True,
        'loaded_checkpoint_imitation_predictions': True, 'evaluation_episodes': 19 * 1200,
        'complete_200_step_evaluation_raw_safety_recomputed': True,
        'evaluation_no_exact_training_overlap': True, 'separate_audit_prefixes_and_isolation': True,
        'protected_sources_D0_prior_results_pilot_unchanged': True,
        'elapsed_seconds': time.monotonic() - start})
    write_json(folder / 'artifact_hashes.json', manifest(folder))
    log(folder, f'SEED {seed} COMPLETE: all checks passed')
    return seed
