"""Accepted Experiment 04, all ten paired seeds, with diagnostic-only additions.

Run: MPLCONFIGDIR=/private/tmp/pendulum_active_rl_mpl python3 \
     experiments/04_full_study.py --workers 3

The accepted pilot and its implementation remain byte-for-byte unchanged.
Every planned seed is mandatory; failures abort aggregation, never exclude seeds.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout
from dataclasses import asdict, replace
import importlib.util
import io
import json
import multiprocessing
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
import unittest

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("accepted_pilot", ROOT / "experiments/04_active_acquisition.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)

from src.active_query import (
    BUDGETS, STRATEGIES, CollectionFailure, budget_prefix, collect_expert,
    collection_summary, nearest_d0_distance, rank_candidates, sample_states,
    seed_streams, validate_collection,
)
from src.acquisition_analysis import (
    BIN_WIDTHS, BOOTSTRAP_REPEATS, BOOTSTRAP_SEED, EQUILIBRIUM_RADII,
    SUPPORT_THRESHOLDS, coverage_diagnostics, flatten_numeric, seed_summary,
)
from src.dataset import load_dataset
from src.ensemble import EnsembleConfig, configure_determinism, member_seeds
from src.policy_evaluation import audit_policy, evaluate_rollouts
from src.ragged_ensemble import combine_datasets

SEEDS = tuple(range(42, 52))
PILOT = ROOT / "results/active_acquisition/pilot_seed41"
DEFAULT_OUTPUT = ROOT / "results/active_acquisition/full_study_42_51"


def read_json(path):
    return json.loads(Path(path).read_text())


def load_npz(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def save_npz(path, **arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    temporary.replace(path)


def immutable_json(path, record):
    record = json.loads(json.dumps(record, allow_nan=False))
    if path.exists():
        if read_json(path) != record:
            raise RuntimeError(f"Frozen metadata mismatch: {path}")
    else:
        pilot.write_json(path, record)


def study_protocol():
    accepted = read_json(PILOT / "protocol.json")
    assert accepted == pilot.protocol()
    result = dict(accepted, phase="full_study", master_seeds=list(SEEDS))
    result["training_default_seed_is_overridden"] = "Actual master-derived selector/retraining seeds and member streams are saved per seed."
    result["inference"] = {
        "unit": "master seed, paired across methods", "n_seeds": 10,
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_repeats": BOOTSTRAP_REPEATS,
        "interval": "2.5/97.5 percentiles of bootstrapped paired mean differences",
        "standard_deviation": "sample SD, ddof=1, of per-seed differences",
        "conditioning": "Fixed shared evaluation ICs; intervals describe training/acquisition seed variation, not test-population uncertainty",
        "multiplicity": "Separate marginal intervals for two primary outcomes; secondary analyses descriptive, no familywise adjustment",
    }
    result["diagnostic_only_additions"] = {
        "support_distance_thresholds": list(SUPPORT_THRESHOLDS),
        "normalized_equilibrium_radii": list(EQUILIBRIUM_RADII),
        "normalized_bin_widths": list(BIN_WIDTHS), "bin_origin": [0, 0],
        "trajectory_phases": "first 20 vs remaining observed transitions",
        "tail": "Largest ceil(5%*N) distances, tail mean and share of total distance",
        "settling": "First observed suffix entirely within radius .05; no inference about censored continuation",
        "weighting": "Report both transition-weighted and equally weighted trajectory diagnostics",
        "bin_interpretation": "Resolution-dependent occupancy, not continuous covered volume",
    }
    result["replay_policy"] = "Pilot established deterministic equivalence. Full study does not repeat acquisition; physical executed expert collection steps equal maximum budget per strategy."
    result["PD_reference"] = "One shared deterministic evaluation on identical fixed held-out ICs, reused across seeds."
    return result


def diagnose_pilot(folder, d0):
    target = folder / "pilot_diagnostic"
    target.mkdir(exist_ok=True)
    for strategy in STRATEGIES:
        data = load_npz(PILOT / "acquisition" / strategy / "budget_5000.npz")
        distances = load_npz(PILOT / "acquisition" / strategy / "coverage.npz")["pre_action_distances"][:5000]
        # Independent direct hypot calculation: no einsum/chunked distance helper.
        independent = np.array([
            np.hypot(d0["states"][:, 0] - state[0], (d0["states"][:, 1] - state[1]) / 3).min()
            for state in data["states"]
        ])
        np.testing.assert_allclose(independent, distances, rtol=1e-12, atol=1e-14)
        result = coverage_diagnostics(data, distances, d0["states"])
        result["independent_distance_check"] = {
            "count": len(distances), "passed": True,
            "maximum_absolute_difference": float(np.max(np.abs(independent - distances))),
            "formula": "min hypot(theta-theta_D0, (omega-omega_D0)/3)",
        }
        pilot.write_json(target / f"{strategy}_5000.json", result)
    u = read_json(target / "uncertainty_5000.json")["summary"]["tail"]
    print(f"Pilot distance diagnostic: mean={u['distribution']['mean']:.8g}, p95={u['distribution']['p95']:.8g}, top-5% mass share={u['largest_5_percent_share_of_distance_sum']:.3%}; independent distances agree.", flush=True)


def run_seed(seed, output):
    folder = Path(output) / f"seed_{seed}"
    folder.mkdir(exist_ok=True)
    with (folder / "worker_stdout.log").open("a", buffering=1) as stream, redirect_stdout(stream):
        try:
            return _run_seed(seed, Path(output), folder)
        except BaseException as exc:
            pilot.write_json(folder / "FAILURE.json", {"seed": seed, "error": str(exc), "traceback": traceback.format_exc(), "complete": False})
            raise


def _run_seed(seed, output, folder):
    assert seed in SEEDS
    started = time.monotonic()
    configure_determinism(seed)
    seeds = seed_streams(seed)
    selector_config = replace(EnsembleConfig(), seed=seeds["selector"])
    training_config = replace(EnsembleConfig(), seed=seeds["retraining"])
    immutable_json(folder / "metadata.json", {
        "master_seed": seed, "seed_streams": seeds,
        "selector_config": asdict(selector_config), "retraining_config": asdict(training_config),
        "selector_member_seeds": member_seeds(selector_config),
        "retraining_member_seeds_shared_across_conditions": member_seeds(training_config),
        "study_manifest_sha256": pilot.sha256(output / "metadata.json"),
        "protocol_sha256": pilot.sha256(output / "protocol.json"),
        "cpu_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    })
    if (folder / "checks.json").exists() and read_json(folder / "checks.json").get("all_passed"):
        for name, digest in read_json(folder / "artifact_hashes.json").items():
            assert pilot.sha256(folder / name) == digest, name
        return seed
    d0 = load_npz(ROOT / "data/pendulum_D0.npz")
    assert pilot.sha256(ROOT / "data/pendulum_D0.npz") == pilot.D0_HASH
    d0_data = combine_datasets(d0)
    checks = {}
    selector, _ = pilot.train_set(folder, "selector", d0_data, selector_config, checks)
    pool = sample_states(seeds["pool"], 10000)
    rankings, scores = rank_candidates(pool, selector, d0["states"], seeds)
    candidate_arrays = dict(initial_conditions=pool, **scores, **{f"ranking_{k}": v for k, v in rankings.items()})
    if (folder / "candidates.npz").exists():
        pilot.assert_same_arrays(candidate_arrays, load_npz(folder / "candidates.npz"))
    else:
        save_npz(folder / "candidates.npz", **candidate_arrays)
    acquisition, streams, coverage = {}, {}, {}
    for strategy in STRATEGIES:
        target = folder / "acquisition" / strategy
        target.mkdir(parents=True, exist_ok=True)
        maximum_path = target / "budget_10000.npz"
        if maximum_path.exists():
            data = load_npz(maximum_path)
        else:
            try:
                data = collect_expert(pool, rankings[strategy], max(BUDGETS), seeds["environment"])
            except CollectionFailure as exc:
                save_npz(target / "incomplete_stream.npz", **exc.partial)
                pilot.write_json(target / "collection_failure.json", {"error": str(exc), "attempted_steps": exc.attempts})
                raise
            save_npz(maximum_path, **data)
        validate_collection(data, 10000)
        used = data["candidate_ids"][data["episode_offsets"][:-1]]
        np.testing.assert_array_equal(used, rankings[strategy][:len(used)])
        streams[strategy] = data
        coverage_path = target / "coverage.npz"
        if coverage_path.exists():
            distances = load_npz(coverage_path)
        else:
            distances = {
                "pre_action_distances": nearest_d0_distance(data["states"], d0["states"]),
                "next_state_distances": nearest_d0_distance(data["next_states"], d0["states"]),
                "initial_state_distances": nearest_d0_distance(data["initial_conditions"], d0["states"]),
            }
            save_npz(coverage_path, **distances)
        acquisition[strategy], coverage[strategy] = {}, {}
        for budget in BUDGETS:
            prefix = budget_prefix(data, budget)
            validate_collection(prefix, budget)
            save_npz(target / f"budget_{budget}.npz", **prefix)
            d = distances["pre_action_distances"][:budget]
            acquisition[strategy][str(budget)] = collection_summary(prefix, d)
            diagnostics = coverage_diagnostics(prefix, d, d0["states"])
            coverage[strategy][str(budget)] = diagnostics["summary"]
            pilot.write_json(target / f"trajectory_coverage_{budget}.json", diagnostics)
        checks[f"{strategy}_exact_nested_budgets_safety_stops_frozen_order"] = True
        pilot.log(folder, f"seed {seed} {strategy}: 10000 transitions, {len(used)} resets, {int(data['safety_stop'].sum())} safety crossings")
    pilot.write_json(folder / "acquisition_metrics.json", acquisition)
    pilot.write_json(folder / "coverage_metrics.json", coverage)
    all_models = {}
    all_models["D0"], _ = pilot.train_set(folder, "D0", d0_data, training_config, checks)
    for strategy in STRATEGIES:
        for budget in BUDGETS:
            name = f"{strategy}_{budget}"
            data = combine_datasets(d0, budget_prefix(streams[strategy], budget))
            assert len(data[0]) == 20000 + budget
            all_models[name], _ = pilot.train_set(folder, name, data, training_config, checks)
    checks["paired_member_seeds_and_fixed_configuration"] = True
    evaluation = load_npz(output / "held_out_states.npz")
    known = set(map(tuple, np.concatenate([d0["states"], pool, *(v["states"] for v in streams.values())])))
    for values in evaluation.values():
        assert not known.intersection(map(tuple, values))
    checks["evaluation_state_streams_independent_no_exact_overlap"] = True
    metrics = {"PD": read_json(output / "PD_reference/metrics.json")}
    for name, models in all_models.items():
        pilot.log(folder, f"seed {seed}: evaluating {name}")
        target = folder / "evaluation" / name
        target.mkdir(parents=True, exist_ok=True)
        metrics[name] = {}
        for region in ("wide", "nominal"):
            arrays, result = evaluate_rollouts(models, evaluation[region], pilot.EVAL_SEEDS["resets"])
            save_npz(target / f"{region}_rollouts.npz", **arrays)
            assert arrays["raw_states"].shape == (len(evaluation[region]), 201, 2)
            metrics[name][region] = result
        arrays, result = audit_policy(models, evaluation["audit"])
        save_npz(target / "imitation_audit.npz", **arrays)
        metrics[name]["audit"] = result
        pilot.write_json(target / "metrics.json", metrics[name])
    pilot.write_json(folder / "metrics.json", metrics)
    checks.update(all_passed=True, seed=seed, complete_evaluation_horizons=True,
                  actual_expert_collection_steps=30000, additional_training_transitions_per_strategy=10000,
                  collection_failures=0, elapsed_seconds=time.monotonic() - started)
    pilot.write_json(folder / "checks.json", checks)
    hashes = {str(p.relative_to(folder)): pilot.sha256(p) for p in sorted(folder.rglob("*"))
              if p.is_file() and p.name not in ("artifact_hashes.json", "worker_stdout.log", "execution.log")}
    pilot.write_json(folder / "artifact_hashes.json", hashes)
    pilot.log(folder, f"seed {seed} COMPLETE in {checks['elapsed_seconds']:.1f}s")
    return seed


def summarize(output):
    metrics, acquisition, coverage = {}, {}, {}
    for seed in SEEDS:
        folder = output / f"seed_{seed}"
        if not read_json(folder / "checks.json")["all_passed"]:
            raise RuntimeError(f"Missing/failed seed {seed}; refusing an incomplete study summary.")
        metrics[seed] = read_json(folder / "metrics.json")
        acquisition[seed] = read_json(folder / "acquisition_metrics.json")
        coverage[seed] = read_json(folder / "coverage_metrics.json")
    draws = np.random.default_rng(BOOTSTRAP_SEED).integers(0, len(SEEDS), size=(BOOTSTRAP_REPEATS, len(SEEDS)))
    save_npz(output / "bootstrap_indices.npz", indices=draws, master_seeds=np.array(SEEDS))
    per_condition = {}
    for condition in metrics[SEEDS[0]]:
        flat = [flatten_numeric(metrics[seed][condition]) for seed in SEEDS]
        per_condition[condition] = {key: seed_summary([r[key] for r in flat], draws) for key in flat[0]}
    differences = {}
    for budget in BUDGETS:
        for left, right in ((f"uncertainty_{budget}", f"random_{budget}"),
                            (f"distance_{budget}", f"random_{budget}"),
                            *((f"{s}_{budget}", "D0") for s in STRATEGIES)):
            left_flat = [flatten_numeric(metrics[s][left]) for s in SEEDS]
            right_flat = [flatten_numeric(metrics[s][right]) for s in SEEDS]
            differences[f"{left}_minus_{right}"] = {
                key: seed_summary([a[key] - b[key] for a, b in zip(left_flat, right_flat)], draws)
                for key in left_flat[0] if key in right_flat[0]
            }
    collection_summary_all, coverage_summary_all = {}, {}
    for strategy in STRATEGIES:
        collection_summary_all[strategy], coverage_summary_all[strategy] = {}, {}
        for budget in BUDGETS:
            for source, dest in ((acquisition, collection_summary_all), (coverage, coverage_summary_all)):
                flat = [flatten_numeric(source[s][strategy][str(budget)]) for s in SEEDS]
                # Stop-reason strata can be absent; do not replace missing
                # conditional means with zero. Keep universal fields here and
                # all stratified per-seed values in the original artifacts.
                common = set.intersection(*(set(row) for row in flat))
                dest[strategy][str(budget)] = {
                    key: seed_summary([r[key] for r in flat], draws) for key in sorted(common)
                }
    primary = {key: differences["uncertainty_5000_minus_random_5000"][key]
               for key in ("wide.mean_return", "wide.safety_violation_rate")}
    summary = {"seeds": list(SEEDS), "pilot_excluded": True,
               "primary": primary, "condition_statistics": per_condition,
               "paired_differences": differences, "collection_statistics": collection_summary_all,
               "coverage_statistics": coverage_summary_all}
    pilot.write_json(output / "summary.json", summary)
    pilot.write_json(output / "primary_comparison.json", primary)
    save_summary_plots(output, summary)
    save_report(output, summary)
    return summary


def save_summary_plots(output, summary):
    conditions = summary["condition_statistics"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for strategy in STRATEGIES:
        for ax, key, factor in zip(axes, ("wide.mean_return", "wide.safety_violation_rate", "audit.all.bounded_action_error.mae"), (1, 100, 1)):
            means = np.array([conditions[f"{strategy}_{b}"][key]["mean"] for b in BUDGETS]) * factor
            cis = np.array([conditions[f"{strategy}_{b}"][key]["bootstrap_95_ci_mean"] for b in BUDGETS]) * factor
            ax.plot(BUDGETS, means, "o-", label=strategy)
            ax.fill_between(BUDGETS, cis[:, 0], cis[:, 1], alpha=.15)
    for ax, key, factor in zip(axes, ("wide.mean_return", "wide.safety_violation_rate", "audit.all.bounded_action_error.mae"), (1, 100, 1)):
        ax.axhline(conditions["D0"][key]["mean"] * factor, color="black", linestyle="--", label="D0 only")
        ax.set_xlabel("Added expert transitions")
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Mean episode return")
    axes[1].set_ylabel("Episodes violating safety [%]")
    axes[2].set_ylabel("Held-out action MAE [N m]")
    fig.suptitle("Seeds 42–51; shaded marginal 95% seed-bootstrap intervals")
    fig.savefig(output / "learning_curves.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, key, factor, ylabel in zip(axes, ("wide.mean_return", "wide.safety_violation_rate"),
                                      (1, 100), ("Return difference", "Violation-rate difference [percentage points]")):
        record = summary["primary"][key]
        ax.scatter(SEEDS, np.array(record["per_seed"]) * factor)
        ax.axhline(0, color="black", linewidth=1)
        ax.axhline(record["mean"] * factor, color="tab:orange", label="Paired mean")
        ci = np.array(record["bootstrap_95_ci_mean"]) * factor
        ax.axhspan(*ci, alpha=.15, color="tab:orange", label="95% seed-bootstrap CI")
        ax.set(xlabel="Master seed", ylabel=ylabel, xticks=SEEDS)
        ax.legend(fontsize=8)
    fig.suptitle("Primary comparison at B=5,000: uncertainty minus random")
    fig.savefig(output / "primary_paired_differences.png", dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    for strategy in STRATEGIES:
        all_distances, trajectory_fractions, by_steps = [], [], []
        for seed in SEEDS:
            target = output / f"seed_{seed}" / "acquisition" / strategy
            all_distances.append(load_npz(target / "coverage.npz")["pre_action_distances"][:5000])
            diag = read_json(target / "trajectory_coverage_5000.json")
            trajectory_fractions.extend(t["fraction_near_equilibrium"]["0.05"] for t in diag["trajectories"])
            by_steps.append([r["mean_distance"] for r in diag["by_step"]])
        values = np.sort(np.concatenate(all_distances))
        axes[0].plot(values, np.arange(1, len(values) + 1) / len(values), label=strategy)
        fractions = np.sort(trajectory_fractions)
        axes[1].plot(fractions, np.arange(1, len(fractions) + 1) / len(fractions), label=strategy)
        # Conditional on a trajectory being observed at that step; never impute
        # terminated rollouts as zero-distance successors.
        step_arrays = np.array(by_steps, dtype=float)
        axes[2].plot(np.arange(200), np.nanmean(step_arrays, axis=0), label=strategy)
    axes[0].set(xlabel="New-transition nearest-D0 distance", ylabel="Pooled transition ECDF")
    axes[1].set(xlabel="Per-trajectory fraction near equilibrium (radius .05)", ylabel="Pooled trajectory ECDF")
    axes[2].set(xlabel="Observed rollout step", ylabel="Mean distance, conditional on observation")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.suptitle("Coverage at B=5,000; descriptive pooling, not independent-sample inference")
    fig.savefig(output / "trajectory_coverage.png", dpi=180)
    plt.close(fig)


def save_report(output, summary):
    lines = ["# Experiment 04 — full study, seeds 42–51", "",
             "All ten planned paired seeds are included. The seed-41 pilot is excluded. Acquisition, model architecture, hyperparameters, safety stopping, budgets, and evaluation initial conditions match the accepted pilot. No iterative querying or offline RL was added.", "",
             "## Primary comparison: 5,000 transitions, uncertainty minus random", "",
             "Intervals are 95% percentile seed-bootstrap intervals for the paired mean (10,000 resamples, seed 404010). SD is the sample SD of the ten paired seed differences. Positive return differences favor uncertainty; negative violation-rate differences favor uncertainty.", "",
             "| Outcome | Paired mean | Sample SD | Minimum | Maximum | 95% CI for mean |", "|---|---:|---:|---:|---:|---|"]
    for key, label, factor in (("wide.mean_return", "Mean episode return", 1), ("wide.safety_violation_rate", "Violation rate [percentage points]", 100)):
        r = summary["primary"][key]
        ci = [v * factor for v in r["bootstrap_95_ci_mean"]]
        lines.append(f"| {label} | {r['mean']*factor:.6f} | {r['sample_std']*factor:.6f} | {r['minimum']*factor:.6f} | {r['maximum']*factor:.6f} | [{ci[0]:.6f}, {ci[1]:.6f}] |")
    conditions = summary["condition_statistics"]
    lines += ["", "## Controller outcomes", "", "Table entries are across-seed means. Worst return and minimum margin are first calculated within each seed's 1,000 episodes; the table averages those worst-case values. Full ranges, sample SDs, confidence intervals, nominal-region metrics, raw/clipped disagreement, and saturation-stratified errors are in `summary.json`.", "",
              "| Controller | Mean return | Violation rate | Mean worst return | Mean minimum raw margin | Audit MAE | Audit RMSE |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in ("PD", "D0", *(f"{s}_{b}" for b in BUDGETS for s in STRATEGIES)):
        r = conditions[name]
        def mean(k):
            return r[k]["mean"]
        mae = f"{mean('audit.all.bounded_action_error.mae'):.6f}" if name != "PD" else "—"
        rmse = f"{mean('audit.all.bounded_action_error.rmse'):.6f}" if name != "PD" else "—"
        lines.append(f"| {name} | {mean('wide.mean_return'):.6f} | {mean('wide.safety_violation_rate'):.3%} | {mean('wide.worst_return'):.6f} | {mean('wide.minimum_raw_safety_margin'):.6f} | {mae} | {rmse} |")
    lines += ["", "## Collection costs and coverage", "",
              "Resets count every selected initial condition. Every safety-crossing transition is retained and counted, followed immediately by stopping that rollout. Each strategy executes 10,000 collection transitions per seed once; smaller budgets are prefixes, not extra rollouts. There are no collection replays in the full study. Diagnostic and evaluation calls do not enter training.", "",
              "| Strategy | Budget | Mean resets | Mean collection safety crossings | Mean distance | Mean p95 distance | Near equilibrium radius .05 | Near D0 distance .01 | Added bins, width .05 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for b in BUDGETS:
        for s in STRATEGIES:
            c = summary["collection_statistics"][s][str(b)]
            d = summary["coverage_statistics"][s][str(b)]
            lines.append(f"| {s} | {b} | {c['rollouts']['mean']:.2f} | {c['safety_crossings']['mean']:.2f} | {d['tail.distribution.mean']['mean']:.6f} | {d['tail.distribution.p95']['mean']:.6f} | {d['transition_fraction_near_equilibrium.0.05']['mean']:.3%} | {d['transition_fraction_near_D0.0.01']['mean']:.3%} | {d['bin_coverage.0.05.additional_occupied_bins']['mean']:.2f} |")
    diag = read_json(output / "pilot_diagnostic/uncertainty_5000.json")
    tail = diag["summary"]["tail"]
    lines += ["", "## Pilot mean-versus-percentile diagnostic", "",
              f"The independent direct-distance calculation agreed with all 5,000 stored uncertainty-strategy distances (maximum absolute difference {diag['independent_distance_check']['maximum_absolute_difference']:.3g}). Mean {tail['distribution']['mean']:.8f} exceeds p95 {tail['distribution']['p95']:.8f} because of a right tail: the largest {tail['largest_5_percent_count']} distances average {tail['largest_5_percent_mean']:.8f} and contribute {tail['largest_5_percent_share_of_distance_sum']:.3%} of the distance sum. The remaining 95% average {tail['remaining_mean']:.8f}. The weighted components reconstruct the reported mean. This is not a percentile or distance-calculation error.", "",
              "Per-trajectory records retain initial distance, lengths, near-D0 and near-equilibrium fractions, first-20 versus later transitions, and observed settled suffixes. Summaries distinguish equal trajectory weighting from equal transition weighting and separate stop reasons. Occupied-bin changes are provided at two fixed resolutions; these are descriptive measures rather than continuous-volume estimates.", "",
              "## Interpretation and limits", "",
              "The primary outcomes must be interpreted separately. Improvements in held-out expert-action error do not establish better returns or safety. Return trajectories continue after violations, whereas expert acquisition stops at the first violation, so post-violation behavior is largely extrapolation beyond the acquired region. Raw unwrapped simulator angles enter every safety diagnostic; policy inputs retain D0's observation-derived canonical angle.", "",
              "Intervals reflect master-seed variability conditional on this fixed shared evaluation set, D0, architecture, and environment. They do not include test-population uncertainty. Ten seeds give limited precision; secondary contrasts and diagnostic thresholds are descriptive, and the two primary intervals are marginal rather than jointly multiplicity-adjusted. Early safety stopping gives unequal rollout lengths and reset costs despite equal transition budgets; both are reported. Bootstrap samples are whole trajectories, but transition-mean training gives longer trajectories more weight.", "",
              "All datasets, candidate scores/rankings, raw/clipped predictions, histories, checkpoints, full evaluation arrays, trajectory diagnostics, and source/seed/environment hashes are saved. No favorable seeds were selected. No commit or push was performed.", ""]
    (output / "full_study_report.md").write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3, choices=range(1, 5))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summarize-only", action="store_true", help="Recreate summaries of a completed unchanged study; no training.")
    args = parser.parse_args()
    output = args.output.resolve()
    if output == PILOT.resolve() or PILOT.resolve() in output.parents:
        raise ValueError("The accepted pilot directory cannot be used for full-study outputs.")
    if args.summarize_only:
        summarize(output)
        return
    if output.exists() and not args.resume:
        raise FileExistsError(f"{output} already exists; --resume requires identical frozen metadata.")
    output.mkdir(parents=True, exist_ok=True)
    protocol = study_protocol()
    immutable_json(output / "protocol.json", protocol)
    accepted_meta = read_json(PILOT / "metadata.json")
    accepted_hashes = {**accepted_meta["protected_hashes"], **accepted_meta["source_hashes"]}
    for name, digest in accepted_hashes.items():
        assert pilot.sha256(ROOT / name) == digest, f"Accepted implementation changed: {name}"
    pilot_hashes = {str(p.relative_to(ROOT)): pilot.sha256(p) for p in sorted(PILOT.rglob("*")) if p.is_file()}
    source_paths = sorted([*ROOT.glob("src/*.py"), *ROOT.glob("experiments/*.py"), *ROOT.glob("tests/*.py")])
    metadata = {
        "seeds": list(SEEDS), "workers": args.workers,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip(),
        "accepted_implementation_hashes": accepted_hashes, "protected_pilot_hashes": pilot_hashes,
        "source_hashes": {str(p.relative_to(ROOT)): pilot.sha256(p) for p in source_paths},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "torch": str(torch.__version__),
                     "gymnasium": gym.__version__, "matplotlib": matplotlib.__version__, "platform": platform.platform()},
        "multiprocessing": "spawn; each worker uses deterministic single-thread CPU PyTorch",
    }
    immutable_json(output / "metadata.json", metadata)
    for p in source_paths:
        snapshot = output / "source_snapshot" / p.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            snapshot.write_bytes(p.read_bytes())
        assert pilot.sha256(snapshot) == pilot.sha256(p)
    test_output = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
    tests = unittest.TextTestRunner(stream=test_output, verbosity=2).run(suite)
    (output / "technical_tests.txt").write_text(test_output.getvalue())
    if not tests.wasSuccessful():
        raise RuntimeError(test_output.getvalue())
    configure_determinism(BOOTSTRAP_SEED)
    d0 = load_npz(ROOT / "data/pendulum_D0.npz")
    diagnose_pilot(output, d0)
    evaluation = {
        "wide": sample_states(pilot.EVAL_SEEDS["wide"], 1000),
        "nominal": sample_states(pilot.EVAL_SEEDS["nominal"], 200, (-.25, -.5), (.25, .5)),
        "audit": sample_states(pilot.EVAL_SEEDS["audit"], 10000),
    }
    pilot.assert_same_arrays(evaluation, load_npz(PILOT / "held_out_states.npz"))
    save_npz(output / "held_out_states.npz", **evaluation)
    pd_metrics = {}
    for region in ("wide", "nominal"):
        arrays, pd_metrics[region] = evaluate_rollouts(None, evaluation[region], pilot.EVAL_SEEDS["resets"])
        save_npz(output / "PD_reference" / f"{region}_rollouts.npz", **arrays)
        pilot.assert_same_arrays(arrays, load_npz(PILOT / "evaluation/PD" / f"{region}_rollouts.npz"))
    pilot.write_json(output / "PD_reference/metrics.json", pd_metrics)
    completed = []
    print(f"Launching all fixed seeds {list(SEEDS)}, {args.workers} workers; accepted methods unchanged.", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as executor:
        futures = {executor.submit(run_seed, seed, str(output)): seed for seed in SEEDS}
        for future in as_completed(futures):
            seed = future.result()
            completed.append(seed)
            pilot.write_json(output / "progress.json", {"completed_seeds": sorted(completed), "planned_seeds": list(SEEDS)})
            print(f"Completed seed {seed}: {len(completed)}/10", flush=True)
    for name, digest in {**accepted_hashes, **pilot_hashes, **metadata["source_hashes"]}.items():
        assert pilot.sha256(ROOT / name) == digest, f"Protected source/data/pilot changed: {name}"
    summary = summarize(output)
    pilot.write_json(output / "completion_checks.json", {
        "all_passed": True, "seeds": sorted(completed), "technical_tests_passed": tests.testsRun,
        "accepted_code_D0_and_pilot_unchanged": True, "same_fixed_held_out_states": True,
        "PD_reference_matches_pilot_exactly": True, "all_budget_prefixes_valid": True,
        "collection_failures": 0, "total_executed_expert_collection_steps": 300000,
        "trained_member_checkpoints": 550, "no_seed_exclusion": True,
    })
    print(json.dumps(summary["primary"], indent=2), flush=True)
    print(f"FULL STUDY COMPLETE: {output}", flush=True)


if __name__ == "__main__":
    main()
