"""Experiment 04: seed-41 technical pilot of fixed-ranking expert acquisition.

Run: MPLCONFIGDIR=/private/tmp/pendulum_active_rl_mpl python3 \
     experiments/04_active_acquisition.py --pilot

The full study is intentionally not launchable from this entry point yet.
Its frozen design uses paired seeds 42--51; pilot performance must not tune it.
Existing verified source, data, and results are never overwritten.
"""

import argparse
from dataclasses import asdict, replace
import hashlib
import io
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import unittest

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.active_query import (
    BUDGETS, STRATEGIES, CollectionFailure, budget_prefix, collect_expert,
    collection_summary, nearest_d0_distance, rank_candidates, sample_states,
    seed_streams, validate_collection,
)
from src.dataset import load_dataset
from src.ensemble import (
    EnsembleConfig, configure_determinism, fit_member, load_member, member_seeds,
    predict_ensemble, save_member,
)
from src.policy_evaluation import audit_policy, evaluate_rollouts, rollout_metrics
from src.ragged_ensemble import combine_datasets, fit_ragged_member

D0_HASH = "7ca6e83d77d2d358e2b77975da25bc4a49411b2681dba9155a206b42bd09a832"
SEED = 41
EVAL_SEEDS = {"wide": 404001, "nominal": 404002, "audit": 404003, "resets": 404004}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def array_digest(*arrays):
    digest = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        digest.update(str((a.shape, a.dtype.str)).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def log(folder, message):
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    with (folder / "execution.log").open("a") as stream:
        stream.write(line + "\n")


def protocol():
    return {
        "experiment": "04_active_expert_acquisition", "phase": "technical_pilot",
        "pilot_seed": SEED, "full_study_seeds": list(range(42, 52)),
        "pilot_excluded_from_full_study": True, "performance_tuning_allowed": False,
        "environment": "Pendulum-v1", "horizon": 200,
        "candidate_count": 10000, "candidate_low": [-.8, -2.], "candidate_high": [.8, 2.],
        "strategies": list(STRATEGIES), "budgets": list(BUDGETS),
        "ranking": "Frozen D0 selector; descending bounded std or exact nearest-D0 distance; independent tie/random streams; no future safety filtering",
        "distance": "Euclidean [theta/1,omega/3], nearest among all 20000 D0 pre-action states",
        "collection": "First negative raw next-state margin stops episode; crossing transition counts; next candidate in frozen rank; complete stream prefixes",
        "collection_failure": "Fatal, preserve partial log; no easier replacement or silent exclusion",
        "budget_cutoff": "Not an MDP terminal; separate flag; whole prefix episode remains eligible for bootstrap",
        "training": json.loads(json.dumps(asdict(EnsembleConfig()))),
        "training_seeds": "Independent selector/retraining streams; identical retraining member streams across strategies and budgets",
        "bootstrap": "N draws of whole episodes with replacement; ragged flatten; transition-mean MSE; whole-episode OOB",
        "deployment": "Mean of five individually clipped member torques; population std; preserve raw outputs",
        "safety": "Shared safety_margin on raw env.unwrapped.state, including all 201 visited states; no plotting clips or angle wrapping for safety",
        "policy_state": "Shared get_state(observation), canonical angle; same convention as D0/03",
        "evaluation": {"wide_episodes": 1000, "nominal_episodes": 200, "audit_states": 10000,
                       "seeds": EVAL_SEEDS, "continue_after_safety_violation": True,
                       "nominal_low": [-.25, -.5], "nominal_high": [.25, .5]},
        "primary_comparison": {"budget": 5000, "contrast": "uncertainty minus random",
                               "separate_outcomes": ["wide.mean_return", "wide.safety_violation_rate"]},
        "additional_outcomes": ["distance strategy", "audit imitation MAE/RMSE by saturation",
                                "worst return", "worst 10% mean return", "minimum raw safety margin",
                                "nominal retention", "new-transition D0 distance distribution"],
        "full_study_summary": "Paired seed differences, mean/sample-SD/min/max and 10000 seed-bootstrap 95% intervals; no pilot inferential claims",
        "coverage": "Save exact nearest-D0 distance for every new s_t and s_(t+1), plus per-budget distributions and initial-state distances separately",
        "leakage_control": "Selection sees D0 and pool only; training sees selected prefixes only; evaluation ICs and labels never passed to either",
    }


def train_set(folder, name, data, config, checks):
    target = folder / "training" / name
    target.mkdir(parents=True, exist_ok=True)
    x, y, offsets = data
    identity = {"data_sha256": array_digest(x, y, offsets), "config": asdict(config)}
    identity = json.loads(json.dumps(identity))
    manifest = target / "identity.json"
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise RuntimeError(f"Training cache identity mismatch: {name}")
    write_json(manifest, identity)
    models, records = [], []
    for i, seeds in enumerate(member_seeds(config)):
        path = target / f"member_{i}.pt"
        if path.exists():
            model, payload = load_member(path)
            record = payload["record"]
            assert record["seeds"] == seeds
        else:
            log(folder, f"Training {name}, member {i + 1}/5")
            model, record = fit_ragged_member(x, y, offsets, config, seeds,
                progress=lambda text: log(folder, f"  {name}/{i}: {text}"))
            save_member(path, model, config, record)
        train_ids, oob_ids = record["bootstrap_ids"], record["oob_ids"]
        assert not set(train_ids) & set(oob_ids)
        assert sorted(set(train_ids) | set(oob_ids)) == list(range(len(offsets) - 1))
        assert record["training_transitions_with_multiplicity"] == int(np.diff(offsets)[train_ids].sum())
        assert record["oob_transitions"] == int(np.diff(offsets)[oob_ids].sum())
        restored, _ = load_member(path)
        for key in model.state_dict():
            assert torch.equal(model.state_dict()[key], restored.state_dict()[key])
        models.append(model)
        records.append(record)
    write_json(target / "histories.json", records)
    checks[f"{name}_whole_episode_oob_and_checkpoint_reload"] = True
    return models, records


def assert_same_arrays(first, second):
    assert set(first) == set(second)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key], err_msg=key)


def save_plots(folder, metrics, coverage, audit_arrays):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for strategy in STRATEGIES:
        returns = [metrics[f"{strategy}_{b}"]["wide"]["mean_return"] for b in BUDGETS]
        rates = [metrics[f"{strategy}_{b}"]["wide"]["safety_violation_rate"] for b in BUDGETS]
        axes[0].plot(BUDGETS, returns, "o-", label=strategy)
        axes[1].plot(BUDGETS, rates, "o-", label=strategy)
        values = np.sort(coverage[strategy][:5000])
        axes[2].plot(values, np.arange(1, len(values) + 1) / len(values), label=strategy)
    for ax, metric in zip(axes[:2], ("mean_return", "safety_violation_rate")):
        ax.axhline(metrics["D0"]["wide"][metric], color="black", linestyle="--", label="D0 only")
        ax.axhline(metrics["PD"]["wide"][metric], color="gray", linestyle=":", label="PD expert")
        ax.set_xlabel("Additional expert transitions")
    axes[0].set_ylabel("Mean episode return")
    axes[1].set_ylabel("Fraction of episodes violating safety")
    axes[2].set(xlabel="Normalized nearest-D0 distance, new s_t (B=5000)", ylabel="Empirical CDF")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("Seed-41 technical pilot — descriptive outcomes, no method tuning")
    fig.savefig(folder / "pilot_outcomes_and_coverage.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for row, strategy in enumerate(STRATEGIES):
        for col, budget in enumerate(BUDGETS):
            histories = json.loads((folder / "training" / f"{strategy}_{budget}" / "histories.json").read_text())
            for record in histories:
                h = record["history"]
                axes[row, col].semilogy(h["epoch"], h["oob_mse"], alpha=.7)
            axes[row, col].set(title=f"{strategy}, B={budget}", xlabel="Epoch", ylabel="OOB normalized MSE")
    fig.savefig(folder / "training_histories.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    for row, strategy in enumerate(STRATEGIES):
        values = audit_arrays[f"{strategy}_5000"]
        for col, (key, title) in enumerate((("uncertainty", "Bounded disagreement"),
                                           ("raw_uncertainty", "Raw disagreement"),
                                           ("absolute_error", "Bounded action error"))):
            dots = axes[row, col].scatter(*values["states"].T, c=values[key], s=2, cmap="viridis", rasterized=True)
            axes[row, col].set(title=f"{strategy}: {title}", xlabel="theta [rad]", ylabel="omega [rad/s]")
            fig.colorbar(dots, ax=axes[row, col])
    fig.savefig(folder / "audit_maps_5000.png", dpi=160)
    plt.close(fig)


def run(folder, resume=False):
    start_time = time.monotonic()
    if folder.exists() and not resume:
        raise FileExistsError(f"Output already exists: {folder}. Use --resume for an identical incomplete pilot.")
    folder.mkdir(parents=True, exist_ok=True)
    current_protocol = protocol()
    protocol_path = folder / "protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != current_protocol:
        raise RuntimeError("Frozen pilot protocol mismatch.")
    write_json(protocol_path, current_protocol)
    configure_determinism(SEED)
    d0_path = ROOT / "data" / "pendulum_D0.npz"
    assert sha256(d0_path) == D0_HASH, "D0 is not the verified dataset."
    with load_dataset(d0_path) as archive:
        d0 = {key: archive[key] for key in archive.files}
    assert (int(d0["num_trajectories"]), int(d0["steps_per_trajectory"])) == (100, 200)
    d0_data = combine_datasets(d0)
    # Snapshot every preexisting tracked file except the formerly empty target
    # active_query module. All other existing source/docs must remain unchanged.
    tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
    protected = {name: sha256(ROOT / name) for name in tracked if name and name != "src/active_query.py"}
    protected["data/pendulum_D0.npz"] = sha256(d0_path)
    source_files = ["src/active_query.py", "src/ragged_ensemble.py", "src/policy_evaluation.py",
                    "experiments/04_active_acquisition.py", "tests/test_active_acquisition.py"]
    metadata = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "protected_hashes": protected, "source_hashes": {name: sha256(ROOT / name) for name in source_files},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "torch": str(torch.__version__), "gymnasium": gym.__version__,
                     "matplotlib": matplotlib.__version__, "platform": platform.platform()},
        "seed_streams": seed_streams(SEED), "evaluation_seeds": EVAL_SEEDS,
        "cpu_threads": torch.get_num_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    metadata_path = folder / "metadata.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise RuntimeError("Reproducibility metadata changed; use a fresh output directory.")
    write_json(metadata_path, metadata)
    checks = {}
    log(folder, "Running technical invariant tests before the pilot.")
    test_output = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    outcome = unittest.TextTestRunner(stream=test_output, verbosity=2).run(suite)
    (folder / "technical_tests.txt").write_text(test_output.getvalue())
    if not outcome.wasSuccessful():
        raise RuntimeError(test_output.getvalue())
    checks["technical_tests"] = {"passed": outcome.testsRun, "failures": 0, "errors": 0}
    seeds = metadata["seed_streams"]
    selector_config = replace(EnsembleConfig(), seed=seeds["selector"])
    retraining_config = replace(EnsembleConfig(), seed=seeds["retraining"])
    selector, _ = train_set(folder, "selector", d0_data, selector_config, checks)
    pool = sample_states(seeds["pool"], 10000)
    rankings, scores = rank_candidates(pool, selector, d0["states"], seeds)
    np.savez_compressed(folder / "candidates.npz", initial_conditions=pool,
                        **scores, **{f"ranking_{k}": v for k, v in rankings.items()})
    repeated_ranks, repeated_scores = rank_candidates(pool, selector, d0["states"], seeds)
    assert_same_arrays(rankings, repeated_ranks)
    assert_same_arrays(scores, repeated_scores)
    checks["frozen_rankings_bitwise_repeat"] = True

    streams, coverage, acquisition_metrics = {}, {}, {}
    for strategy in STRATEGIES:
        log(folder, f"Collecting {strategy}: exactly 10000 transitions, seed 41.")
        target = folder / "acquisition" / strategy
        target.mkdir(parents=True, exist_ok=True)
        try:
            stream = collect_expert(pool, rankings[strategy], max(BUDGETS), seeds["environment"])
        except CollectionFailure as exc:
            np.savez_compressed(target / "incomplete_stream.npz", **exc.partial)
            write_json(target / "FAILURE.json", {"error": str(exc), "attempted_steps": exc.attempts, "complete": False})
            raise
        validate_collection(stream, max(BUDGETS))
        streams[strategy] = stream
        distances = nearest_d0_distance(stream["states"], d0["states"])
        next_distances = nearest_d0_distance(stream["next_states"], d0["states"])
        initial_distances = nearest_d0_distance(stream["initial_conditions"], d0["states"])
        coverage[strategy] = distances
        np.savez_compressed(target / "coverage.npz", pre_action_distances=distances,
                            next_state_distances=next_distances, initial_state_distances=initial_distances)
        acquisition_metrics[strategy] = {}
        for budget in BUDGETS:
            prefix = budget_prefix(stream, budget)
            validate_collection(prefix, budget)
            # Independent collection to each budget verifies both replay and the
            # exact prefix cutoff: lower-budget metadata cannot see later safety.
            independent = collect_expert(pool, rankings[strategy], budget, seeds["environment"])
            assert_same_arrays(prefix, independent)
            np.savez_compressed(target / f"budget_{budget}.npz", **prefix)
            acquisition_metrics[strategy][str(budget)] = collection_summary(prefix, distances[:budget])
        checks[f"{strategy}_all_exact_budgets_and_bitwise_independent_prefix_replay"] = True
        checks[f"{strategy}_raw_safety_first_crossing_and_variable_lengths"] = True
        log(folder, f"  {strategy}: {len(stream['initial_conditions'])} rollouts; {int(stream['safety_stop'].sum())} safety stops.")
    write_json(folder / "acquisition_metrics.json", acquisition_metrics)

    all_models = {}
    all_models["D0"], _ = train_set(folder, "D0", d0_data, retraining_config, checks)
    for strategy in STRATEGIES:
        for budget in BUDGETS:
            name = f"{strategy}_{budget}"
            data = combine_datasets(d0, budget_prefix(streams[strategy], budget))
            all_models[name], _ = train_set(folder, name, data, retraining_config, checks)
            assert len(data[0]) == 20000 + budget
    checks["identical_fixed_training_configuration_and_paired_member_seeds"] = True

    # Full-config deterministic training replay: baseline first member and one
    # ragged member of each strategy at the predefined primary budget. Also
    # verify baseline equivalence to the unchanged fixed-length implementation.
    for name in ("D0", *(f"{s}_5000" for s in STRATEGIES)):
        log(folder, f"Replaying full-config training for {name}, member 1.")
        data = (d0_data if name == "D0" else
                combine_datasets(d0, budget_prefix(streams[name.rsplit('_', 1)[0]], 5000)))
        member_seed = member_seeds(retraining_config)[0]
        if name == "D0":
            repeated, record = fit_member(data[0].reshape(100, 200, 2), data[1].reshape(100, 200, 1),
                                          retraining_config, member_seed)
        else:
            repeated, record = fit_ragged_member(*data, retraining_config, member_seed)
        saved, payload = load_member(folder / "training" / name / "member_0.pt")
        for key in saved.state_dict():
            assert torch.equal(saved.state_dict()[key], repeated.state_dict()[key])
        for key, value in record.items():
            assert value == payload["record"][key], (name, key)
        checks[f"{name}_full_config_first_member_bitwise_training_replay"] = True

    eval_states = {
        "wide": sample_states(EVAL_SEEDS["wide"], 1000),
        "nominal": sample_states(EVAL_SEEDS["nominal"], 200, (-.25, -.5), (.25, .5)),
        "audit": sample_states(EVAL_SEEDS["audit"], 10000),
    }
    np.savez_compressed(folder / "held_out_states.npz", **eval_states)
    # Exact accidental duplicates are not expected for independent continuous
    # streams; check them explicitly without making any outcome-based exclusion.
    training_states = np.concatenate([d0["states"], *(v["states"] for v in streams.values()), pool])
    known = set(map(tuple, training_states))
    for values in eval_states.values():
        assert not known.intersection(map(tuple, values))
    checks["evaluation_streams_independent_and_no_exact_state_overlap"] = True
    metrics, audit_arrays = {}, {}
    for name, models in {"PD": None, **all_models}.items():
        log(folder, f"Evaluating {name}: 1000 wide + 200 nominal complete 200-step episodes.")
        target = folder / "evaluation" / name
        target.mkdir(parents=True, exist_ok=True)
        metrics[name] = {}
        for region in ("wide", "nominal"):
            arrays, result = evaluate_rollouts(models, eval_states[region], EVAL_SEEDS["resets"])
            np.savez_compressed(target / f"{region}_rollouts.npz", **arrays)
            metrics[name][region] = result
            assert arrays["raw_states"].shape == (len(eval_states[region]), 201, 2)
            assert np.allclose(arrays["episode_returns"], arrays["rewards"].sum(axis=1), rtol=0, atol=0)
            if name in ("PD", "uncertainty_5000"):
                repeated, repeated_metrics = evaluate_rollouts(models, eval_states[region], EVAL_SEEDS["resets"])
                assert_same_arrays(arrays, repeated)
                assert result == repeated_metrics
                checks[f"{name}_{region}_full_evaluation_bitwise_replay"] = True
        if models is not None:
            arrays, result = audit_policy(models, eval_states["audit"])
            np.savez_compressed(target / "imitation_audit.npz", **arrays)
            audit_arrays[name] = arrays
            metrics[name]["audit"] = result
        write_json(target / "metrics.json", metrics[name])
    checks["all_evaluations_complete_horizons_and_raw_safety"] = True
    checks["all_saved_arrays_finite"] = True
    write_json(folder / "metrics.json", metrics)
    primary = {
        "seed": 41, "budget": 5000, "status": "technical pilot; descriptive only; excluded from full study",
        "uncertainty_minus_random_mean_return": metrics["uncertainty_5000"]["wide"]["mean_return"] - metrics["random_5000"]["wide"]["mean_return"],
        "uncertainty_minus_random_safety_violation_rate": metrics["uncertainty_5000"]["wide"]["safety_violation_rate"] - metrics["random_5000"]["wide"]["safety_violation_rate"],
        "favorable_directions": {"mean_return": "positive", "safety_violation_rate": "negative"},
    }
    write_json(folder / "primary_pilot_comparison.json", primary)
    save_plots(folder, metrics, coverage, audit_arrays)
    for name, expected in protected.items():
        assert sha256(ROOT / name) == expected, f"Protected file changed: {name}"
    checks["verified_code_and_D0_hashes_unchanged"] = True
    checks["full_study_not_launched"] = True
    checks["all_passed"] = True
    checks["elapsed_seconds"] = time.monotonic() - start_time
    write_json(folder / "checks.json", checks)
    summary = ["# Experiment 04 seed-41 technical pilot", "",
               "All technical checks passed. No full-study seeds were run. No performance tuning.", "",
               "Primary budget: 5,000 additional transitions; 1,000 shared wide-region evaluation episodes.", "",
               "| Controller | Mean return | Violation rate | Worst return | Minimum raw margin |",
               "|---|---:|---:|---:|---:|"]
    for name in ("PD", "D0", *(f"{s}_5000" for s in STRATEGIES)):
        m = metrics[name]["wide"]
        summary.append(f"| {name} | {m['mean_return']:.6f} | {m['safety_violation_rate']:.3%} | {m['worst_return']:.6f} | {m['minimum_raw_safety_margin']:.6f} |")
    summary.extend(["", "Raw simulator angles are unwrapped for safety; policy inputs use observation angles.",
                    "Collection stops at the first raw safety crossing; evaluation continues for 200 steps.",
                    "The pilot is an implementation check, not evidence of seed-robust superiority.", ""])
    (folder / "pilot_report.md").write_text("\n".join(summary))
    log(folder, f"PILOT COMPLETE: all checks passed in {checks['elapsed_seconds']:.1f} seconds.")
    print("\n".join(summary), flush=True)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", required=True, action="store_true", help="Run only the approved seed-41 technical pilot.")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "active_acquisition" / "pilot_seed41")
    parser.add_argument("--resume", action="store_true", help="Resume checkpoints with identical protocol, source hashes, and training inputs.")
    args = parser.parse_args()
    run(args.output.resolve(), args.resume)


if __name__ == "__main__":
    main()
