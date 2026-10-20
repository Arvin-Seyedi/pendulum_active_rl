"""Experiment 03 robustness: fixed training protocol, seeds 42--46, zero ties.

Run: python experiments/03_robustness_analysis.py
Reuse the existing ensemble and grid evaluation without changing their defaults.
Conditional populations are retrospective diagnostics, not deployable selection
rules: their error-based masks require the expert labels. Recompute top-10%
targets and budgets within each subset. Do not collect data or run policies.
"""

from dataclasses import asdict, replace
import csv
import importlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

base = importlib.import_module("experiments.03_uncertainty_map")
from src.ensemble import (
    EnsembleConfig, configure_determinism, fit_member, load_member,
    member_seeds, predict_ensemble, save_member,
)

MASTER_SEEDS = (42, 43, 44, 45, 46)
ZERO_TOLERANCE = 1e-8  # N m; diagnostic sensitivity, never a training parameter.
OUTPUT_DIR = base.RESULTS_DIR / "robustness"
REGIONS = ("all", "unsaturated", "saturated", "positive_saturated", "negative_saturated")
SUBSETS = ("full", "error_positive", "uncertainty_positive", "both_positive",
           "exclude_joint_zeros", "both_above_1e_minus_8")
METHODS = ("uncertainty", "support_distance", "raw_uncertainty")


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def subset_masks(error, uncertainty):
    return {
        "full": np.ones(len(error), dtype=bool),
        "error_positive": error > 0,
        "uncertainty_positive": uncertainty > 0,
        "both_positive": (error > 0) & (uncertainty > 0),
        "exclude_joint_zeros": (error > 0) | (uncertainty > 0),
        "both_above_1e_minus_8": (error > ZERO_TOLERANCE) & (uncertainty > ZERO_TOLERANCE),
    }


def evaluate_subsets(grid):
    """Retain counts, correlations, exact selections, and random replicates."""
    error, uncertainty = grid["absolute_error"], grid["uncertainty"]
    masks = subset_masks(error, uncertainty)
    metrics, arrays = {}, {}
    for region_index, region_name in enumerate(REGIONS):
        region = grid[f"region_{region_name}"]
        e, u = error[region], uncertainty[region]
        metrics[region_name] = {
            "region_count": int(region.sum()),
            "joint_zero_count": int(np.count_nonzero((e == 0) & (u == 0))),
            "zero_uncertainty_positive_error_count": int(np.count_nonzero((u == 0) & (e > 0))),
            "zero_error_positive_uncertainty_count": int(np.count_nonzero((e == 0) & (u > 0))),
            "subsets": {},
        }
        # Identical masks get exactly identical tie breaks and random samples.
        cache = {}
        for subset_name, subset in masks.items():
            mask = region & subset
            ids = np.flatnonzero(mask)
            prefix = f"{region_name}_{subset_name}"
            arrays[prefix + "_population_ids"] = ids
            cache_key = mask.tobytes()
            if cache_key not in cache:
                if len(ids) < 2:
                    cache[cache_key] = ({"count": len(ids), "status": "insufficient states",
                                         "spearman": None, "selection": None}, {}, np.array([]))
                else:
                    selected_error = error[mask]
                    scores = {name: grid[name][mask] for name in METHODS}
                    selection, selections, random_captures = base.selection_comparison(
                        selected_error, scores, base.EVALUATION_SEED + region_index
                    )
                    result = {
                        "count": len(ids), "status": "evaluated",
                        "bounded_action_error": base.error_metrics(grid["bounded_mean"][mask], grid["expert_actions"][mask]),
                        "uncertainty": base.describe(scores["uncertainty"]),
                        "raw_uncertainty": base.describe(scores["raw_uncertainty"]),
                        "spearman": {
                            "uncertainty_error": base.spearman(scores["uncertainty"], selected_error),
                            "uncertainty_distance": base.spearman(scores["uncertainty"], scores["support_distance"]),
                            "raw_uncertainty_error": base.spearman(scores["raw_uncertainty"], selected_error),
                            "raw_uncertainty_distance": base.spearman(scores["raw_uncertainty"], scores["support_distance"]),
                            "distance_error": base.spearman(scores["support_distance"], selected_error),
                        },
                        "selection": selection,
                    }
                    cache[cache_key] = result, selections, random_captures
            result, selections, random_captures = cache[cache_key]
            metrics[region_name]["subsets"][subset_name] = result
            for method, selected in selections.items():
                arrays[prefix + "_" + method + "_ids"] = ids[selected]
            arrays[prefix + "_random_captures"] = random_captures
    return metrics, arrays


def across_seeds(values):
    """Sample standard deviation (ddof=1) across the five master seeds."""
    finite = [float(value) for value in values if value is not None]
    if not finite:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None, "values": values}
    return {
        "n": len(finite), "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=1)) if len(finite) > 1 else None,
        "min": float(np.min(finite)), "max": float(np.max(finite)), "values": values,
    }


def aggregate_runs(runs):
    aggregate = {}
    for region in REGIONS:
        aggregate[region] = {}
        for subset in SUBSETS:
            results = [run["subsets"][region]["subsets"][subset] for run in runs]
            summary = {"counts": across_seeds([r["count"] for r in results]), "capture": {}, "spearman": {}}
            for method in METHODS:
                summary["capture"][method] = across_seeds([
                    r["selection"]["methods"][method]["capture_fraction"] if r["selection"] else None
                    for r in results
                ])
            for key in ("single_draw_capture_fraction", "mean_capture_fraction", "expected_capture_fraction"):
                summary["capture"]["random_" + key] = across_seeds([
                    r["selection"]["random"][key] if r["selection"] else None for r in results
                ])
            for key in ("uncertainty_error", "uncertainty_distance", "raw_uncertainty_error", "raw_uncertainty_distance", "distance_error"):
                summary["spearman"][key] = across_seeds([
                    r["spearman"][key] if r["spearman"] else None for r in results
                ])
            # Paired differences preserve which baselines each seed outperformed.
            u = summary["capture"]["uncertainty"]["values"]
            summary["paired_capture_difference"] = {}
            for baseline in ("support_distance", "raw_uncertainty", "random_mean_capture_fraction"):
                b = summary["capture"][baseline]["values"]
                differences = [a - c if a is not None and c is not None else None for a, c in zip(u, b)]
                summary["paired_capture_difference"][baseline] = {
                    **across_seeds(differences),
                    "seeds_strictly_better": sum(value is not None and value > 0 for value in differences),
                }
            aggregate[region][subset] = summary
    return aggregate


def write_csv(aggregate, runs):
    with (OUTPUT_DIR / "aggregate_capture.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["region", "subset", "method", "seeds", "mean", "sample_std", "min", "max"])
        for region, subsets in aggregate.items():
            for subset, result in subsets.items():
                for method, summary in result["capture"].items():
                    writer.writerow([region, subset, method, summary["n"], summary["mean"], summary["std"], summary["min"], summary["max"]])
    with (OUTPUT_DIR / "per_seed_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["seed", "region", "subset", "count", "budget", "rho_U_error", "rho_raw_U_error", "U_capture", "distance_capture", "raw_U_capture", "random_mean_capture", "random_single_capture"])
        for run in runs:
            for region in REGIONS:
                for subset in SUBSETS:
                    r = run["subsets"][region]["subsets"][subset]
                    if not r["selection"]:
                        continue
                    s = r["selection"]
                    writer.writerow([run["seed"], region, subset, r["count"], s["budget"],
                                     r["spearman"]["uncertainty_error"], r["spearman"]["raw_uncertainty_error"],
                                     *[s["methods"][method]["capture_fraction"] for method in METHODS],
                                     s["random"]["mean_capture_fraction"], s["random"]["single_draw_capture_fraction"]])


def save_robustness_plots(runs):
    plt = base.plt
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row, subset in enumerate(("full", "both_positive")):
        for col, region in enumerate(("all", "unsaturated", "saturated")):
            ax = axes[row, col]
            for method, label in zip(METHODS, ("Bounded U", "D0 distance", "Raw U")):
                values = [r["subsets"][region]["subsets"][subset]["selection"]["methods"][method]["capture_fraction"] for r in runs]
                ax.plot(MASTER_SEEDS, np.array(values) * 100, "o-", label=label)
            random = [r["subsets"][region]["subsets"][subset]["selection"]["random"]["mean_capture_fraction"] for r in runs]
            ax.plot(MASTER_SEEDS, np.array(random) * 100, "k--", label="Random mean")
            ax.set(title=f"{region}: {subset}", xlabel="Master seed", ylabel="Top-10% error capture [%]", ylim=(0, 100))
            ax.set_xticks(MASTER_SEEDS)
            ax.grid(alpha=0.2)
            ax.legend(fontsize=8)
    fig.suptitle("Five independently retrained ensembles; equal budgets within each population")
    fig.savefig(OUTPUT_DIR / "capture_across_seeds.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for ax, region in zip(axes, ("all", "unsaturated", "saturated")):
        for subset, style in (("full", "-"), ("both_positive", "--")):
            for key, color in (("uncertainty_error", "tab:blue"), ("raw_uncertainty_error", "tab:orange")):
                values = [r["subsets"][region]["subsets"][subset]["spearman"][key] for r in runs]
                ax.plot(MASTER_SEEDS, values, marker="o", linestyle=style, color=color,
                        label=f"{'Bounded' if key == 'uncertainty_error' else 'Raw'} U, {subset}")
        ax.set(title=region.capitalize(), xlabel="Master seed", ylabel="Spearman correlation with action error", ylim=(-1, 1.05))
        ax.set_xticks(MASTER_SEEDS)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    fig.savefig(OUTPUT_DIR / "zero_tie_correlations.png", dpi=180)
    plt.close(fig)


def make_report(runs, aggregate):
    lines = [
        "Experiment 03 robustness: master seeds 42, 43, 44, 45, 46",
        "",
        "All 25 networks were freshly trained with the unchanged 2-64-64-1 ReLU architecture,",
        "trajectory bootstrap, Adam 1e-3, batch 256, 300-epoch cap, and 30-epoch OOB patience.",
        "Bounded action = mean of five individually clipped torques; disagreement std uses ddof=0.",
        "Across-master-seed standard deviations below use ddof=1 (five ensembles).",
        "",
        "Each subset has its own ceil(0.1*N) budget and top-error target on the same fixed grid.",
        "Error-dependent masks are retrospective diagnostics; they are not available for unlabeled query selection.",
        "Random is the mean of 1,000 uniform draws per seed/subset; single-draw statistics are also saved.",
        "Random-subset intervals and variation across master seeds measure different sources of variability.",
        "Exact >0 masks are primary; >1e-8 N m is a fixed numerical sensitivity diagnostic.",
        "",
        "Seed 42: zero-tie analysis",
        "",
        "| Region | Subset | N | rho U-error | rho raw-U-error | U capture | Distance capture | Random mean | Raw U capture |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for region in ("all", "unsaturated", "saturated"):
        for subset in SUBSETS:
            r = runs[0]["subsets"][region]["subsets"][subset]
            if r["selection"]:
                s = r["selection"]
                captures = [s["methods"][m]["capture_fraction"] for m in ("uncertainty", "support_distance")]
                captures += [s["random"]["mean_capture_fraction"], s["methods"]["raw_uncertainty"]["capture_fraction"]]
                lines.append(f"| {region} | {subset} | {r['count']} | {r['spearman']['uncertainty_error']:.6f} | "
                             f"{r['spearman']['raw_uncertainty_error']:.6f} | " + " | ".join(f"{100*x:.2f}%" for x in captures) + " |")
    lines.extend(["", "Across-seed capture summaries (percent; SD in percentage points)", "",
                  "| Region | Subset | Method | Mean | SD | Min | Max |", "|---|---|---|---:|---:|---:|---:|"])
    for region in ("all", "unsaturated", "saturated"):
        for subset in ("full", "both_positive", "both_above_1e_minus_8"):
            for method in (*METHODS, "random_mean_capture_fraction", "random_single_draw_capture_fraction"):
                s = aggregate[region][subset]["capture"][method]
                lines.append(f"| {region} | {subset} | {method} | " + " | ".join(f"{100*s[key]:.2f}" for key in ("mean", "std", "min", "max")) + " |")
    lines.extend(["", "Per-seed counts and bounded uncertainty-error correlations", "",
                  "| Seed | Region | Joint zeros | Positive error N | Positive U N | Both positive N | Full rho | Both-positive rho |",
                  "|---:|---|---:|---:|---:|---:|---:|---:|"])
    for run in runs:
        for region in ("all", "unsaturated", "saturated"):
            r = run["subsets"][region]
            s = r["subsets"]
            lines.append(f"| {run['seed']} | {region} | {r['joint_zero_count']} | {s['error_positive']['count']} | "
                         f"{s['uncertainty_positive']['count']} | {s['both_positive']['count']} | "
                         f"{s['full']['spearman']['uncertainty_error']:.6f} | {s['both_positive']['spearman']['uncertainty_error']:.6f} |")
    return "\n".join(lines) + "\n"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    protected_names = (
        "data/pendulum_D0.npz", "experiments/01_expert_map.py", "experiments/02_generate_D0.py",
        "src/expert.py", "src/safety.py", "src/dataset.py", "src/ensemble.py", "experiments/03_uncertainty_map.py",
    )
    protected = {name: base.file_hash(PROJECT_ROOT / name) for name in protected_names}
    original_outputs = {str(p.relative_to(PROJECT_ROOT)): base.file_hash(p)
                        for p in base.RESULTS_DIR.rglob("*") if p.is_file() and OUTPUT_DIR not in p.parents}
    data, states, actions = base.read_trajectories(base.DATASET_PATH)
    runs, seed42_verification = [], {}
    base_config = EnsembleConfig()
    for master_seed in MASTER_SEEDS:
        config = replace(base_config, seed=master_seed)
        configure_determinism(master_seed)
        folder = OUTPUT_DIR / f"seed_{master_seed}"
        folder.mkdir(parents=True, exist_ok=True)
        models, records = [], []
        for index, seeds in enumerate(member_seeds(config)):
            print(f"Seed {master_seed}, member {index + 1}/5...", flush=True)
            model, record = fit_member(states, actions, config, seeds,
                                       progress=lambda message: print("  " + message, flush=True))
            oob = np.asarray(record["oob_ids"])
            predictions = predict_ensemble([model], states[oob].reshape(-1, 2))
            record["oob_bounded_action_error"] = base.error_metrics(predictions["bounded_mean"], actions[oob].ravel())
            record["oob_raw_action_error"] = base.error_metrics(predictions["raw_mean"], actions[oob].ravel())
            original_checkpoint = base.RESULTS_DIR / "checkpoints" / f"member_{index + 1}.pt"
            if master_seed == 42 and original_checkpoint.exists():
                original, payload = load_member(original_checkpoint)
                assert record == payload["record"], "Seed 42 training record changed"
                assert all(torch.equal(value, original.state_dict()[key]) for key, value in model.state_dict().items()), "Seed 42 weights changed"
                seed42_verification[f"member_{index + 1}"] = "weights and full record exactly match original"
            models.append(model)
            records.append(record)
            save_member(folder / "checkpoints" / f"member_{index + 1}.pt", model, config, record)
            print(f"  Best/run epochs: {record['best_epoch']}/{record['epochs_run']}", flush=True)
        print(f"Seed {master_seed}: evaluating grid and conditional subsets...", flush=True)
        grid, full_metrics, selections = base.evaluate_grid(models, data["states"])
        subset_metrics, subset_arrays = evaluate_subsets(grid)
        # Full-subset comparisons must reproduce the original evaluation exactly.
        for region in REGIONS:
            assert subset_metrics[region]["subsets"]["full"]["selection"] == full_metrics[region]["selection"]
        original_grid = base.RESULTS_DIR / "grid_evaluation.npz"
        if master_seed == 42 and original_grid.exists():
            with np.load(original_grid, allow_pickle=False) as old:
                for key, value in grid.items():
                    np.testing.assert_array_equal(value, old[key], err_msg=key)
            seed42_verification["grid"] = "every grid array exactly matches original"
        np.savez_compressed(folder / "grid_evaluation.npz", **grid)
        np.savez_compressed(folder / "selection_masks.npz", **selections)
        np.savez_compressed(folder / "subset_selections.npz", **subset_arrays)
        history_arrays = {f"member_{i + 1}_{key}": np.asarray(value)
                          for i, record in enumerate(records) for key, value in record["history"].items()}
        np.savez_compressed(folder / "training_histories.npz", **history_arrays)
        write_json(folder / "training_metadata.json", {"config": asdict(config), "members": records})
        run = {"seed": master_seed, "full_grid": full_metrics, "subsets": subset_metrics}
        write_json(folder / "metrics.json", run)
        base.save_plots(grid, full_metrics, records, data["states"], folder)
        runs.append(run)
        for subset in ("full", "both_positive"):
            r = subset_metrics["all"]["subsets"][subset]
            s = r["selection"]
            print(f"  {subset}: N={r['count']}, rho={r['spearman']['uncertainty_error']:.6f}; "
                  + ", ".join(f"{m} capture={s['methods'][m]['capture_fraction']:.4%}" for m in METHODS), flush=True)

    aggregate = aggregate_runs(runs)
    write_json(OUTPUT_DIR / "aggregate_metrics.json", aggregate)
    write_csv(aggregate, runs)
    save_robustness_plots(runs)
    report = make_report(runs, aggregate)
    (OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")
    metadata = {
        "master_seeds": MASTER_SEEDS, "base_config": asdict(base_config),
        "training": "only master seed varies; all 25 models freshly trained",
        "protected_sha256": protected, "original_output_sha256": original_outputs,
        "runner_sha256": base.file_hash(Path(__file__)),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "python": platform.python_version(), "numpy": np.__version__, "torch": torch.__version__,
        "matplotlib": base.matplotlib.__version__, "platform": platform.platform(),
        "device": "cpu", "dtype": "float32", "threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "grid": "unchanged 161 x 161; theta [-1,1], omega [-3,3]",
        "bounded_action": "mean of five individually clipped outputs", "uncertainty_ddof": 0,
        "across_master_seeds_std_ddof": 1,
        "subsets": {"full": "all states in region", "error_positive": "absolute bounded action error > 0",
                    "uncertainty_positive": "bounded disagreement > 0", "both_positive": "error > 0 AND disagreement > 0",
                    "exclude_joint_zeros": "error > 0 OR disagreement > 0",
                    "both_above_1e_minus_8": "error > 1e-8 AND disagreement > 1e-8, in N m"},
        "subset_membership": "depends on trained ensemble; underlying evaluation grid remains fixed",
        "selection": "ceil(0.1*N) per region/subset; recompute top-error target within same population",
        "evaluation_seed": base.EVALUATION_SEED, "evaluation_seed_rule": "4242 + region index; fixed across master seeds/subsets",
        "regions": REGIONS, "random_repeats": base.RANDOM_REPEATS,
        "random_aggregation": "report both one seeded draw and mean of 1000 draws per training seed; random expectation also saved",
        "limitations": "conditional error masks are oracle diagnostics; five seeds are descriptive, not calibration or acquisition/safety evidence",
    }
    for name, checksum in {**protected, **original_outputs}.items():
        assert base.file_hash(PROJECT_ROOT / name) == checksum, f"Protected file changed: {name}"
    write_json(OUTPUT_DIR / "metadata.json", metadata)
    write_json(OUTPUT_DIR / "verification.json", {
        "protected_files_unchanged": True, "original_outputs_unchanged": True,
        "seed_42_matches_original": seed42_verification,
        "full_subset_matches_original_evaluator_all_seeds": True,
    })
    print(f"Robustness complete. Results: {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
