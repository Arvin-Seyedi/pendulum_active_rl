"""Reproducible Experiment 05 pilot orchestration using verified source functions."""

from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import time
import unittest

import gymnasium as gym
import matplotlib
import numpy as np
import torch

from src.active_query import (
    BUDGETS, CollectionFailure, budget_prefix, collect_expert, collection_summary,
    nearest_d0_distance, rank_candidates, sample_states, seed_streams,
)
from src.acquisition_analysis import coverage_diagnostics, flatten_numeric
from src.dataset import load_dataset
from src.ensemble import (
    EnsembleConfig, configure_determinism, load_member, member_seeds,
    predict_ensemble, save_member,
)
from src.policy_evaluation import audit_policy, evaluate_rollouts
from src.ragged_ensemble import combine_datasets, fit_ragged_member
from src.rollout_horizon import (
    collect_metered, episode_slices, expert_safety_audit, operational_summary,
    selection_diagnostics, shared_prefix_check, summarize_safety_audit,
    validate_horizon_collection,
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def canonical(value):
    return json.loads(json.dumps(value, allow_nan=False))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.json')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def immutable_json(path, value):
    if path.exists():
        if read_json(path) != canonical(value):
            raise RuntimeError(f'Frozen metadata mismatch: {path}')
    else:
        write_json(path, value)


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def save_npz(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def assert_same_arrays(a, b):
    assert set(a) == set(b)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def condition_name(strategy, horizon, budget):
    return f'{strategy}_H{horizon}_B{budget}'


def log(folder, message):
    line = f'[{time.strftime("%H:%M:%S")}] {message}'
    print(line, flush=True)
    with (folder / 'execution.log').open('a') as stream:
        stream.write(line + '\n')


def train_ensemble(folder, name, data, config):
    """Cache orchestration only; the approved ragged trainer is unchanged."""
    target = folder / 'training' / name
    target.mkdir(parents=True, exist_ok=True)
    immutable_json(target / 'identity.json', {'data_sha256': array_digest(*data), 'config': asdict(config)})
    models, records, costs = [], [], {}
    cost_path = target / 'training_costs.json'
    if cost_path.exists():
        costs = read_json(cost_path)
    for i, seeds in enumerate(member_seeds(config)):
        path = target / f'member_{i}.pt'
        if path.exists():
            model, payload = load_member(path)
            record = payload['record']
        else:
            log(folder, f'Training {name}, member {i + 1}/5')
            start = time.perf_counter()
            model, record = fit_ragged_member(*data, config, seeds)
            save_member(path, model, config, record)
            lengths = np.diff(data[2])
            ids = np.asarray(record['bootstrap_ids'])
            costs[str(i)] = {
                'elapsed_seconds': time.perf_counter() - start,
                'optimizer_updates': record['epochs_run'] * int(np.ceil(record['training_transitions_with_multiplicity'] / config.batch_size)),
                'bootstrap_D0_transitions': int(lengths[ids[ids < 100]].sum()),
                'bootstrap_new_transitions': int(lengths[ids[ids >= 100]].sum()),
            }
            write_json(cost_path, costs)
        restored, payload = load_member(path)
        assert canonical(payload['config']) == canonical(asdict(config))
        assert record == payload['record'] and record['seeds'] == seeds
        for key in model.state_dict():
            assert torch.equal(model.state_dict()[key], restored.state_dict()[key])
        ids, oob = record['bootstrap_ids'], record['oob_ids']
        assert not set(ids) & set(oob)
        assert sorted(set(ids) | set(oob)) == list(range(len(data[2]) - 1))
        assert record['training_transitions_with_multiplicity'] == int(np.diff(data[2])[ids].sum())
        assert record['oob_transitions'] == int(np.diff(data[2])[oob].sum())
        assert record['dataset_episode_lengths'] == np.diff(data[2]).tolist()
        assert all(np.isfinite(v).all() for v in record['history'].values())
        assert all(torch.isfinite(v).all() for v in model.state_dict().values())
        assert record['best_oob_normalized_mse'] == min(record['history']['oob_mse'])
        np.testing.assert_array_equal(predict_ensemble([model], data[0][:128])['raw_predictions'],
                                      predict_ensemble([restored], data[0][:128])['raw_predictions'])
        models.append(restored)
        records.append(record)
    write_json(target / 'histories.json', records)
    return models, records


def paired_contrasts(metrics, budget):
    """Single-seed differences; seed-level inference is reserved for the full study."""
    names = {f'{s}{h}': condition_name(s, h, budget) for s in ('uncertainty', 'random', 'distance') for h in (20, 200)}
    pairs = [
        (names['uncertainty20'], names['uncertainty200']),
        (names['uncertainty20'], names['random20']),
        (names['random20'], names['random200']),
        (names['distance20'], names['distance200']),
        (names['uncertainty20'], names['distance20']),
        *[(name, 'D0') for name in names.values()],
    ]
    flat = {name: flatten_numeric(value) for name, value in metrics.items()}
    result = {}
    for left, right in pairs:
        result[f'{left}_minus_{right}'] = {key: value - flat[right][key] for key, value in flat[left].items() if key in flat[right]}
    u = result[f"{names['uncertainty20']}_minus_{names['uncertainty200']}"]
    r = result[f"{names['random20']}_minus_{names['random200']}"]
    result['uncertainty_horizon_effect_minus_random_horizon_effect'] = {key: value - r[key] for key, value in u.items()}
    return result


def _protected_manifest(root, output):
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=root).decode().split('\0')
    paths = {root / name for name in tracked if name}
    paths.add(root / 'data/pendulum_D0.npz')
    paths.update(path for path in (root / 'results').rglob('*') if path.is_file() and output not in path.parents)
    return {str(path.relative_to(root)): sha256(path) for path in sorted(paths)}


def _assert_hashes(root, mapping):
    for name, digest in mapping.items():
        assert sha256(root / name) == digest, f'Protected artifact changed: {name}'


def run_pilot(root, output, resume=False):
    root, output = Path(root).resolve(), Path(output).resolve()
    results_root = root / 'results/rollout_horizon'
    if results_root not in output.parents:
        raise ValueError('Experiment 05 outputs must be below results/rollout_horizon/.')
    if output.exists() and not resume:
        raise FileExistsError(f'Output exists: {output}; only identical incomplete runs may resume.')
    config = read_json(root / 'configs/experiment_05.json')
    assert config['pilot_seed'] == 41 and config['full_study_seeds'] == list(range(52, 62))
    assert not config['full_study_authorized_to_launch']
    assert config['collection_horizons'] == [20, 200] and config['budgets'] == list(BUDGETS)
    default_training = asdict(EnsembleConfig())
    default_training.pop('seed')
    assert config['training'] == canonical(default_training)
    accepted = read_json(root / 'configs/experiment_04.json')
    _assert_hashes(root, accepted['provenance']['source_sha256'])
    assert sha256(root / config['D0']['path']) == config['D0']['sha256']
    output.mkdir(parents=True, exist_ok=True)
    immutable_json(output / 'protocol.json', config)
    configure_determinism(41)
    sources = [*root.glob('src/*.py'), *root.glob('experiments/*.py'), *root.glob('tests/*.py'), root / 'configs/experiment_05.json']
    metadata = {
        'phase': 'technical_pilot_only', 'master_seed': 41,
        'git_head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
        'source_hashes': {str(path.relative_to(root)): sha256(path) for path in sorted(sources)},
        'protected_hashes': _protected_manifest(root, output),
        'protocol_sha256': sha256(output / 'protocol.json'),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__, 'torch': str(torch.__version__),
                     'gymnasium': gym.__version__, 'matplotlib': matplotlib.__version__, 'platform': platform.platform()},
        'seed_streams': seed_streams(41), 'evaluation_seeds': config['evaluation']['pilot_seeds'],
        'cpu_threads': torch.get_num_threads(), 'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
        'full_study_seeds_executed': [],
    }
    immutable_json(output / 'metadata.json', metadata)
    if (output / 'checks.json').exists() and read_json(output / 'checks.json').get('all_passed'):
        _assert_hashes(output, read_json(output / 'artifact_hashes.json'))
        log(output, 'Verified completed pilot; no experiments restarted.')
        return read_json(output / 'checks.json')
    for path in sources:
        snapshot = output / 'source_snapshot' / path.relative_to(root)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            snapshot.write_bytes(path.read_bytes())
        assert sha256(snapshot) == sha256(path)
    started = time.perf_counter()
    try:
        return _pilot_work(root, output, config, metadata, started)
    except BaseException as exc:
        import traceback
        write_json(output / 'FAILURE.json', {'error': str(exc), 'traceback': traceback.format_exc(), 'complete': False})
        raise


def _pilot_work(root, output, config, metadata, started):
    checks = {}
    log(output, 'Running technical invariant tests; only seed 41 is authorized.')
    transcript = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(root / 'tests'))
    tests = unittest.TextTestRunner(stream=transcript, verbosity=2).run(suite)
    (output / 'technical_tests.txt').write_text(transcript.getvalue())
    assert tests.wasSuccessful(), transcript.getvalue()
    checks['technical_tests_passed'] = tests.testsRun
    with load_dataset(root / config['D0']['path']) as archive:
        d0 = {key: archive[key] for key in archive.files}
    d0_data = combine_datasets(d0)
    seeds = metadata['seed_streams']
    selector_config = replace(EnsembleConfig(), seed=seeds['selector'])
    training_config = replace(EnsembleConfig(), seed=seeds['retraining'])
    immutable_json(output / 'member_seeds.json', {'selector_config': asdict(selector_config),
                   'retraining_config': asdict(training_config), 'selector': member_seeds(selector_config),
                   'retraining_shared_across_all_conditions': member_seeds(training_config)})
    selector, _ = train_ensemble(output, 'selector', d0_data, selector_config)
    ranking_start = time.perf_counter()
    pool = sample_states(seeds['pool'], config['candidate_count'])
    rankings, scores = rank_candidates(pool, selector, d0['states'], seeds)
    ranking_seconds = time.perf_counter() - ranking_start
    repeated_ranks, repeated_scores = rank_candidates(pool, selector, d0['states'], seeds)
    assert_same_arrays(rankings, repeated_ranks)
    assert_same_arrays(scores, repeated_scores)
    save_npz(output / 'candidates.npz', dict(initial_conditions=pool, **scores, **{f'ranking_{k}': v for k, v in rankings.items()}))
    checks['frozen_candidate_scores_and_rankings_bitwise_repeat'] = True

    strategies, horizons, budgets = config['strategies'], config['collection_horizons'], config['budgets']
    streams, prefixes, diagnostics = {}, {}, {}
    replay_collection_steps = 0
    acquisition_seconds = 0.
    for strategy in strategies:
        for horizon in horizons:
            arm = f'{strategy}_H{horizon}'
            target = output / 'acquisition' / arm
            target.mkdir(parents=True, exist_ok=True)
            log(output, f'Collecting {arm}: exactly 10000 transitions.')
            if (target / 'budget_10000.npz').exists():
                data = load_npz(target / 'budget_10000.npz')
                costs = load_npz(target / 'operation_costs.npz')
            else:
                try:
                    data, costs = collect_metered(pool, rankings[strategy], 10000, seeds['environment'], horizon)
                except CollectionFailure as exc:
                    save_npz(target / 'incomplete_stream.npz', exc.partial)
                    if hasattr(exc, 'operation_costs'):
                        save_npz(target / 'failed_operation_costs.npz', exc.operation_costs)
                    write_json(target / 'collection_failure.json', {'error': str(exc), 'attempted_steps': exc.attempts})
                    raise
                save_npz(target / 'operation_costs.npz', costs)
                save_npz(target / 'budget_10000.npz', data)
            validate_horizon_collection(data, 10000, horizon)
            streams[arm] = data
            acquisition_seconds += float(costs['collection_wall_seconds'])
            distances = {
                'pre_action_distances': nearest_d0_distance(data['states'], d0['states']),
                'next_state_distances': nearest_d0_distance(data['next_states'], d0['states']),
                'initial_state_distances': nearest_d0_distance(data['initial_conditions'], d0['states']),
            }
            save_npz(target / 'coverage.npz', distances)
            for budget in budgets:
                name = condition_name(strategy, horizon, budget)
                prefix = budget_prefix(data, budget)
                validate_horizon_collection(prefix, budget, horizon)
                independent = collect_expert(pool, rankings[strategy], budget, seeds['environment'], horizon)
                replay_collection_steps += len(independent['states'])
                assert_same_arrays(prefix, independent)
                prefixes[name] = prefix
                save_npz(target / f'budget_{budget}.npz', prefix)
                d = distances['pre_action_distances'][:budget]
                trajectory = coverage_diagnostics(prefix, d, d0['states'])
                selected, selected_metrics = selection_diagnostics(prefix, rankings, scores, strategy)
                save_npz(target / f'selected_initial_conditions_{budget}.npz', selected)
                write_json(target / f'trajectory_coverage_{budget}.json', trajectory)
                diagnostics[name] = {'collection': collection_summary(prefix, d), 'coverage': trajectory['summary'],
                                     'selection': selected_metrics, 'operations': operational_summary(prefix, costs, horizon)}
            checks[f'{arm}_exact_budgets_independent_replays_and_safety_stopping'] = True
    checks['all_shared_candidate_physical_prefixes'] = {}
    for strategy in strategies:
        for budget in budgets:
            result = shared_prefix_check(prefixes[condition_name(strategy, 20, budget)], prefixes[condition_name(strategy, 200, budget)])
            checks['all_shared_candidate_physical_prefixes'][f'{strategy}_{budget}'] = result
    write_json(output / 'acquisition_diagnostics.json', diagnostics)

    models = {}
    models['D0'], _ = train_ensemble(output, 'D0', d0_data, training_config)
    for name, additional in prefixes.items():
        data = combine_datasets(d0, additional)
        assert len(data[0]) == 20000 + len(additional['states'])
        models[name], _ = train_ensemble(output, name, data, training_config)
    checks['all_100_checkpoints_reload_exactly_and_use_whole_episode_oob'] = True
    checks['paired_training_member_seeds_and_unchanged_hyperparameters'] = True

    # Independent full-configuration replays, one member per primary-budget arm.
    replays = []
    for strategy in strategies:
        for horizon in horizons:
            name = condition_name(strategy, horizon, 5000)
            log(output, f'Replaying full-configuration training: {name}, member 1.')
            repeated, record = fit_ragged_member(*combine_datasets(d0, prefixes[name]), training_config, member_seeds(training_config)[0])
            saved, payload = load_member(output / 'training' / name / 'member_0.pt')
            for key in saved.state_dict():
                assert torch.equal(saved.state_dict()[key], repeated.state_dict()[key])
            assert record == payload['record']
            replays.append({'condition': name, 'member': 0, 'bitwise_parameters_and_history': True, 'epochs_run': record['epochs_run']})
    write_json(output / 'training_replay_checks.json', replays)
    checks['all_six_primary_arm_training_replays_bitwise_equal'] = True

    legacy = root / 'results/active_acquisition/pilot_seed41'
    assert read_json(legacy / 'checks.json')['all_passed']
    assert_same_arrays(load_npz(output / 'candidates.npz'), load_npz(legacy / 'candidates.npz'))
    legacy_names = [('selector', 'selector'), ('D0', 'D0')]
    for strategy in strategies:
        for budget in budgets:
            name = condition_name(strategy, 200, budget)
            assert_same_arrays(prefixes[name], load_npz(legacy / 'acquisition' / strategy / f'budget_{budget}.npz'))
            legacy_names.append((name, f'{strategy}_{budget}'))
    for name, old_name in legacy_names:
        for member in range(5):
            current, payload = load_member(output / 'training' / name / f'member_{member}.pt')
            previous, old_payload = load_member(legacy / 'training' / old_name / f'member_{member}.pt')
            for key in current.state_dict():
                assert torch.equal(current.state_dict()[key], previous.state_dict()[key])
            assert payload['record'] == old_payload['record']
    checks['H200_candidates_all_budget_data_and_55_checkpoints_match_accepted_pilot'] = True

    e = config['evaluation']
    eval_seeds = e['pilot_seeds']
    held = {
        'wide': sample_states(eval_seeds['wide'], e['wide_episodes']),
        'nominal': sample_states(eval_seeds['nominal'], e['nominal_episodes'], e['nominal_low'], e['nominal_high']),
        'audit': sample_states(eval_seeds['audit'], e['audit_states']),
    }
    known = set(map(tuple, np.concatenate([d0['states'], pool, *[data['states'] for data in streams.values()]])))
    for states in held.values():
        assert not known.intersection(map(tuple, states))
    save_npz(output / 'held_out_states.npz', held)
    checks['pilot_only_held_out_seeds_independent_no_exact_training_overlap'] = True
    metrics = {}
    eval_replay_steps = 0
    for name, ensemble in {'PD': None, **models}.items():
        log(output, f'Evaluating {name}: 1000 wide + 200 nominal, always 200 steps.')
        target = output / 'evaluation' / name
        metrics[name] = {}
        for region in ('wide', 'nominal'):
            arrays, values = evaluate_rollouts(ensemble, held[region], eval_seeds['resets'], horizon=200)
            save_npz(target / f'{region}_rollouts.npz', arrays)
            metrics[name][region] = values
            assert arrays['raw_states'].shape == (len(held[region]), 201, 2)
            raw_margins = np.minimum(1 - np.abs(arrays['raw_states'][:, :, 0]), 1 - np.abs(arrays['raw_states'][:, :, 1]) / 3)
            np.testing.assert_array_equal(raw_margins, arrays['margins'])
            if name in (condition_name('uncertainty', 20, 5000), condition_name('uncertainty', 200, 5000)):
                repeated, repeated_values = evaluate_rollouts(ensemble, held[region], eval_seeds['resets'], horizon=200)
                assert_same_arrays(arrays, repeated)
                assert values == repeated_values
                eval_replay_steps += len(held[region]) * 200
        if ensemble is not None:
            arrays, values = audit_policy(ensemble, held['audit'])
            save_npz(target / 'imitation_audit.npz', arrays)
            metrics[name]['audit'] = values
        write_json(target / 'metrics.json', metrics[name])
    write_json(output / 'metrics.json', metrics)
    checks['all_200_step_evaluations_and_raw_safety_recomputed'] = True
    checks['both_primary_controller_evaluations_bitwise_repeat'] = True

    # This boundary is after every acquisition, training, checkpoint selection,
    # and policy evaluation. Audit labels are never arguments to those routines.
    frozen_paths = [output / 'candidates.npz', output / 'metrics.json',
                    *[p for sub in ('acquisition', 'training', 'evaluation') for p in (output / sub).rglob('*') if p.is_file()]]
    frozen = {str(p.relative_to(output)): sha256(p) for p in sorted(frozen_paths)}
    audit_folder = output / 'post_selection_safety_audit'
    write_json(audit_folder / 'frozen_inputs_before_audit.json', frozen)
    selected_union = sorted(set(cid for s in strategies for h in horizons for cid in episode_slices(prefixes[condition_name(s, h, 5000)])))
    log(output, f'Acquisition/training frozen. Separate safety audit of {len(selected_union)} unique selected initial conditions.')
    audit, audit_costs = expert_safety_audit(pool, selected_union, seeds['environment'])
    save_npz(audit_folder / 'audit_trajectories.npz', audit)
    save_npz(audit_folder / 'operation_costs.npz', audit_costs)
    safety_summaries = {}
    for strategy in strategies:
        for horizon in horizons:
            name = condition_name(strategy, horizon, 5000)
            safety_summaries[name] = summarize_safety_audit(prefixes[name], audit, horizon)
    write_json(audit_folder / 'condition_summaries.json', safety_summaries)
    repeated_audit, replay_audit_costs = expert_safety_audit(pool, selected_union, seeds['environment'])
    assert_same_arrays(audit, repeated_audit)
    _assert_hashes(output, frozen)
    write_json(audit_folder / 'isolation_checks.json', {'all_frozen_files_unchanged': True,
               'frozen_file_count': len(frozen), 'all_audit_physical_prefixes_match_collection': True,
               'audit_bitwise_replay': True, 'no_continuation_data_in_training': True,
               'late_violation_denominator_excludes_partial_budget_cutoffs': True})
    checks['post_selection_audit_isolated_and_bitwise_reproduced'] = True
    checks['all_audit_prefixes_match_acquisition'] = True

    contrasts = {str(b): paired_contrasts(metrics, b) for b in budgets}
    write_json(output / 'descriptive_paired_differences.json', {'master_seed': 41, 'pilot_only_no_confidence_intervals': True, 'budgets': contrasts})
    costs = {
        'acquisition_transitions': 60000,
        'technical_collection_replay_transitions': replay_collection_steps,
        'post_selection_safety_audit_transitions': int(audit_costs['step_attempts']),
        'post_selection_safety_audit_resets': int(audit_costs['reset_attempts']),
        'technical_safety_audit_replay_transitions': int(replay_audit_costs['step_attempts']),
        'policy_evaluation_environment_steps_including_PD_reference': 20 * 1200 * 200,
        'technical_evaluation_replay_environment_steps': eval_replay_steps,
        'unique_trained_members': 100, 'technical_training_replay_members': 6,
        'ranking_elapsed_seconds': ranking_seconds, 'acquisition_elapsed_seconds': acquisition_seconds,
        'unit_test_simulations_not_counted_as_acquisition': True,
        'timing_is_descriptive_not_a_physical_reset_cost_estimate': True,
    }
    write_json(output / 'cost_accounting.json', costs)
    from src.horizon_reporting import save_pilot_report
    save_pilot_report(output, config, metrics, diagnostics, safety_summaries, contrasts, costs)
    _assert_hashes(root, metadata['protected_hashes'])
    _assert_hashes(root, metadata['source_hashes'])
    checks.update(all_passed=True, master_seed=41, full_study_seeds_executed=[],
                  protected_sources_D0_and_prior_results_unchanged=True,
                  protected_file_count=len(metadata['protected_hashes']),
                  exact_budget_datasets=18, saved_member_checkpoints=100,
                  collection_failures=0, performance_tuning=False,
                  elapsed_seconds=time.perf_counter() - started)
    write_json(output / 'checks.json', checks)
    manifest = {str(p.relative_to(output)): sha256(p) for p in sorted(output.rglob('*'))
                if p.is_file() and p.name not in ('artifact_hashes.json', 'execution.log')}
    write_json(output / 'artifact_hashes.json', manifest)
    log(output, f'PILOT COMPLETE: all checks passed in {checks["elapsed_seconds"]:.1f}s. Full study not launched.')
    return checks
