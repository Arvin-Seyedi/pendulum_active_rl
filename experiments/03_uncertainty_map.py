"""Train D0 bootstrap BC policies and evaluate disagreement on a fixed grid.

Run: python experiments/03_uncertainty_map.py
No simulator rollouts or active data acquisition occur here. Grid expert labels
are evaluation-only. The bounded ensemble action is the mean of individually
clipped member actions. Correlations and selection captures are descriptive
grid statistics, not independent-sample significance tests or safety guarantees.
"""

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import load_dataset
from src.ensemble import (
    EnsembleConfig, configure_determinism, fit_member, member_seeds,
    predict_ensemble, save_member,
)
from src.expert import KD, KP, MAX_TORQUE, pd_controller
from src.safety import OMEGA_SAFE, THETA_SAFE, safety_margin

DATASET_PATH = PROJECT_ROOT / "data" / "pendulum_D0.npz"
RESULTS_DIR = PROJECT_ROOT / "results" / "ensemble_uncertainty"
GRID_SIZE = 161
EVALUATION_SEED = 4242
SELECTION_FRACTION = 0.10
RANDOM_REPEATS = 1000


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_trajectories(path):
    """Validate D0 episode-major layout; never regenerate or write the data."""
    with load_dataset(path) as archive:
        data = {key: archive[key] for key in archive.files}
    n = int(data["num_trajectories"])
    steps = int(data["steps_per_trajectory"])
    if (n, steps) != (100, 200):
        raise ValueError("Experiment 03 expects the verified 100 x 200 D0.")
    if data["states"].shape != (n * steps, 2) or data["actions"].shape != (n * steps, 1):
        raise ValueError("Incorrect state/action layout.")
    for key in ("states", "actions", "next_states"):
        if not np.isfinite(data[key]).all():
            raise ValueError(f"Nonfinite {key}.")
    states = data["states"].reshape(n, steps, 2)
    actions = data["actions"].reshape(n, steps, 1)
    expected_dones = np.zeros((n, steps), dtype=bool)
    expected_dones[:, -1] = True
    if not np.array_equal(data["dones"].reshape(n, steps), expected_dones):
        raise ValueError("Unexpected episode boundaries.")
    if not np.array_equal(states[:, 1:], data["next_states"].reshape(n, steps, 2)[:, :-1]):
        raise ValueError("Broken transition continuity.")
    if not np.allclose(states[:, 0], data["initial_conditions"], atol=1e-14, rtol=0):
        raise ValueError("Initial conditions do not match trajectory starts.")
    expected_actions = pd_controller(*data["states"].T)
    if not np.allclose(data["actions"][:, 0], expected_actions, atol=1e-6, rtol=0):
        raise ValueError("D0 actions differ from the shared PD expert.")
    return data, states, actions


def support_distance(states, support, chunk_size=128):
    """Exact nearest-neighbor distance in safety-normalized coordinates."""
    scale = np.array([THETA_SAFE, OMEGA_SAFE], dtype=np.float64)
    support = np.asarray(support, dtype=np.float64) / scale
    states = np.asarray(states, dtype=np.float64) / scale
    distances = np.empty(len(states))
    for start in range(0, len(states), chunk_size):
        delta = states[start:start + chunk_size, None, :] - support[None, :, :]
        squared = np.einsum("ijk,ijk->ij", delta, delta)
        distances[start:start + chunk_size] = np.sqrt(squared.min(axis=1))
    return distances


def average_ranks(values):
    """One-based average ranks, including ties, without a SciPy dependency."""
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ordered = values[order]
    starts = np.r_[0, np.flatnonzero(ordered[1:] != ordered[:-1]) + 1]
    ends = np.r_[starts[1:], len(values)]
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.repeat((starts + ends + 1) / 2, ends - starts)
    return ranks


def spearman(x, y):
    """Return null for undefined (constant or insufficient) correlations."""
    if len(x) < 2:
        return None
    x = average_ranks(x)
    y = average_ranks(y)
    x -= x.mean()
    y -= y.mean()
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(np.dot(x, y) / denominator, -1, 1)) if denominator else None


def describe(values):
    return {
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "p95": float(np.quantile(values, 0.95)), "max": float(np.max(values)),
    }


def error_metrics(prediction, target):
    error = np.asarray(prediction) - np.asarray(target)
    return {
        "mae": float(np.abs(error).mean()), "rmse": float(np.sqrt(np.mean(error ** 2))),
        "p95_absolute_error": float(np.quantile(np.abs(error), 0.95)),
        "max_absolute_error": float(np.abs(error).max()),
    }


def top_indices(values, k, seed):
    """Descending score with independent seeded, score-agnostic tie breaks."""
    priority = np.random.default_rng(seed).permutation(len(values))
    return np.lexsort((priority, -np.asarray(values)))[:k]


def selection_comparison(error, scores, seed):
    """Capture = selected top-error count / number of top-error states.

Use ceil(0.1*N) for both the selection budget and top-error target. All methods
share the target but use independent tie priorities. Random replicates are
actual uniform subsets without replacement. Intervals describe random-selection
variability only, not training-seed uncertainty.
"""
    n = len(error)
    k = int(np.ceil(SELECTION_FRACTION * n))
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(len(scores) + 2)]
    target_ids = top_indices(error, k, seeds[0])
    target = np.zeros(n, dtype=bool)
    target[target_ids] = True
    masks = {"top_error": target}
    cutoff = float(np.min(error[target]))
    result = {
        "states": n, "budget": k, "actual_fraction": k / n,
        "error_cutoff": cutoff,
        "error_cutoff_ties": int(np.count_nonzero(error == cutoff)),
        "error_tie_seed": seeds[0], "methods": {},
    }
    for (name, values), tie_seed in zip(scores.items(), seeds[1:-1]):
        ids = top_indices(values, k, tie_seed)
        mask = np.zeros(n, dtype=bool)
        mask[ids] = True
        masks[name] = mask
        score_cutoff = float(np.min(values[ids]))
        captured = int(target[ids].sum())
        result["methods"][name] = {
            "captured": captured, "capture_fraction": captured / k,
            "lift_over_random_expectation": (captured / k) / (k / n),
            "score_cutoff": score_cutoff,
            "score_cutoff_ties": int(np.count_nonzero(values == score_cutoff)),
            "tie_seed": tie_seed,
        }
    rng = np.random.default_rng(seeds[-1])
    captures = np.empty(RANDOM_REPEATS)
    for repeat in range(RANDOM_REPEATS):
        ids = rng.choice(n, size=k, replace=False)
        captures[repeat] = target[ids].sum() / k
        if repeat == 0:
            masks["random"] = np.zeros(n, dtype=bool)
            masks["random"][ids] = True
    result["random"] = {
        "seed": seeds[-1], "repeats": RANDOM_REPEATS,
        "expected_capture_fraction": k / n,
        "single_draw_capture_fraction": float(captures[0]),
        "mean_capture_fraction": float(captures.mean()),
        "std_capture_fraction": float(captures.std(ddof=0)),
        "central_95_percent_interval": np.quantile(captures, [0.025, 0.975]).tolist(),
    }
    return result, masks, captures


def evaluate_grid(models, support):
    theta = np.linspace(-THETA_SAFE, THETA_SAFE, GRID_SIZE)
    omega = np.linspace(-OMEGA_SAFE, OMEGA_SAFE, GRID_SIZE)
    theta_grid, omega_grid = np.meshgrid(theta, omega, indexing="xy")
    states = np.column_stack([theta_grid.ravel(), omega_grid.ravel()])
    arrays = predict_ensemble(models, states)
    expert = pd_controller(*states.T)
    error = np.abs(arrays["bounded_mean"] - expert)
    distance = support_distance(states, support)
    # Treat values within 1e-12 of the exact cap as saturated, avoiding grid
    # roundoff changing the classification of points on a saturation boundary.
    saturated = np.isclose(np.abs(expert), MAX_TORQUE, rtol=0, atol=1e-12)
    regions = {
        "all": np.ones(len(states), dtype=bool),
        "unsaturated": ~saturated,
        "saturated": saturated,
        "positive_saturated": saturated & (expert > 0),
        "negative_saturated": saturated & (expert < 0),
    }
    arrays.update(
        theta_values=theta, omega_values=omega, states=states,
        expert_actions=expert, absolute_error=error, support_distance=distance,
        safety_margins=np.array([safety_margin(*state) for state in states]),
    )
    metrics = {}
    selection_arrays = {}
    for index, (name, mask) in enumerate(regions.items()):
        arrays[f"region_{name}"] = mask
        u = arrays["uncertainty"][mask]
        raw_u = arrays["raw_uncertainty"][mask]
        e = error[mask]
        d = distance[mask]
        selection, masks, random_captures = selection_comparison(
            e, {"uncertainty": u, "support_distance": d, "raw_uncertainty": raw_u},
            EVALUATION_SEED + index,
        )
        for key, selected in masks.items():
            full_mask = np.zeros(len(states), dtype=bool)
            full_mask[np.flatnonzero(mask)[selected]] = True
            selection_arrays[f"{name}_{key}"] = full_mask
        selection_arrays[f"{name}_random_captures"] = random_captures
        bins = []
        for ids in np.array_split(np.argsort(d, kind="stable"), 10):
            bins.append({
                "count": len(ids), "distance_min": float(d[ids].min()),
                "distance_max": float(d[ids].max()), "distance_mean": float(d[ids].mean()),
                "uncertainty": describe(u[ids]), "raw_uncertainty": describe(raw_u[ids]),
                "absolute_error": describe(e[ids]),
            })
        metrics[name] = {
            "count": int(mask.sum()),
            "bounded_action_error": error_metrics(arrays["bounded_mean"][mask], expert[mask]),
            "uncertainty": describe(u), "raw_uncertainty": describe(raw_u),
            "support_distance": describe(d),
            "zero_bounded_uncertainty_fraction": float(np.mean(u == 0)),
            "member_clipped_fraction": float(np.mean(np.abs(arrays["raw_predictions"][:, mask]) >= MAX_TORQUE)),
            "spearman": {
                "uncertainty_error": spearman(u, e),
                "uncertainty_distance": spearman(u, d),
                "distance_error": spearman(d, e),
                "raw_uncertainty_error": spearman(raw_u, e),
                "raw_uncertainty_distance": spearman(raw_u, d),
            },
            "distance_deciles": bins,
            "selection": selection,
        }
    return arrays, metrics, selection_arrays


def save_plots(arrays, metrics, records, support, folder):
    theta, omega = arrays["theta_values"], arrays["omega_values"]
    shape = (len(omega), len(theta))
    t, w = np.meshgrid(theta, omega)
    fields = [
        ("uncertainty", "Bounded torque disagreement [N m]"),
        ("raw_uncertainty", "Raw torque disagreement [N m]"),
        ("absolute_error", "Bounded ensemble absolute error [N m]"),
        ("support_distance", "Normalized distance to D0"),
        ("bounded_mean", "Mean of clipped member torques [N m]"),
        ("expert_actions", "PD expert torque [N m]"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, fields):
        torque = key in ("bounded_mean", "expert_actions")
        plot = ax.pcolormesh(theta, omega, arrays[key].reshape(shape), shading="auto",
                             cmap="coolwarm" if torque else "viridis",
                             vmin=-MAX_TORQUE if torque else 0,
                             vmax=MAX_TORQUE if torque else None)
        # These lines show the analytical saturation boundaries, not new labels.
        ax.contour(t, w, KP * t + KD * w, levels=[-MAX_TORQUE, MAX_TORQUE],
                   colors="white", linewidths=0.7, linestyles="--")
        ax.scatter(*support.T, s=0.2, color="black", alpha=0.08, rasterized=True)
        ax.set(title=title, xlabel="Angle [rad]", ylabel="Angular velocity [rad/s]")
        fig.colorbar(plot, ax=ax)
    fig.suptitle("Fixed 161 x 161 grid; dashed lines: expert saturation; black: D0")
    fig.savefig(folder / "uncertainty_error_maps.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for index, record in enumerate(records):
        h = record["history"]
        for ax, key in zip(axes, ("train_mse", "oob_mse")):
            ax.semilogy(h["epoch"], h[key], label=f"Member {index + 1}")
            best = record["best_epoch"]
            ax.plot(best, h[key][best - 1], "o", color=f"C{index}", markersize=4)
    for ax, title in zip(axes, ("Bootstrap training loss", "Member-specific OOB validation loss")):
        ax.set(title=title, xlabel="Epoch", ylabel="Normalized raw-action MSE")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(folder / "training_histories.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8), constrained_layout=True)
    for row, region in enumerate(("unsaturated", "saturated")):
        mask = arrays[f"region_{region}"]
        for ax, xkey, ykey in zip(axes[row],
                                  ("support_distance", "uncertainty", "raw_uncertainty"),
                                  ("uncertainty", "absolute_error", "absolute_error")):
            ax.scatter(arrays[xkey][mask], arrays[ykey][mask], s=2, alpha=0.12, rasterized=True)
            ax.set(title=region.capitalize(), xlabel=xkey.replace("_", " "),
                   ylabel=ykey.replace("_", " "))
            ax.grid(alpha=0.2)
    fig.savefig(folder / "uncertainty_relationships.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for name in ("all", "unsaturated", "saturated"):
        bins = metrics[name]["distance_deciles"]
        distance = [b["distance_mean"] for b in bins]
        axes[0].plot(distance, [b["uncertainty"]["mean"] for b in bins], "o-", label=name)
        axes[1].plot(distance, [b["absolute_error"]["mean"] for b in bins], "o-", label=name)
    for ax, ylabel in zip(axes, ("Mean bounded disagreement [N m]", "Mean absolute error [N m]")):
        ax.set(xlabel="Mean normalized D0 distance in decile", ylabel=ylabel)
        ax.legend()
        ax.grid(alpha=0.2)
    fig.savefig(folder / "distance_deciles.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    names = ("uncertainty", "support_distance", "raw_uncertainty")
    for ax, region in zip(axes, ("all", "unsaturated", "saturated")):
        selection = metrics[region]["selection"]
        ax.bar(range(3), [selection["methods"][name]["capture_fraction"] for name in names])
        random = selection["random"]
        ax.axhspan(*random["central_95_percent_interval"], color="tab:orange", alpha=0.2)
        ax.axhline(random["expected_capture_fraction"], color="tab:orange", linestyle="--", label="Random expectation / 95% range")
        ax.set_xticks(range(3), ("Bounded U", "D0 distance", "Raw U"))
        ax.set(title=region.capitalize(), ylabel="Fraction of top-10%-error states captured", ylim=(0, 1))
        ax.legend(fontsize=7)
    fig.suptitle("Equal top-10% selection budgets within each region")
    fig.savefig(folder / "selection_comparison.png", dpi=180)
    plt.close(fig)


def make_summary(metrics, records):
    lines = [
        "Experiment 03: five trajectory-bootstrap BC networks; seed 42",
        "Grid: 161 x 161 over [-1,1] rad x [-3,3] rad/s (25921 states)",
        "Bounded action = mean of individually clipped outputs; std ddof=0",
        "Errors below are torque errors in N m; OOB results are model-selection diagnostics.",
    ]
    for index, record in enumerate(records):
        error = record["oob_bounded_action_error"]
        lines.append(f"Member {index + 1}: unique/OOB trajectories={record['unique_training_trajectories']}/{len(record['oob_ids'])}; "
                     f"best/run epochs={record['best_epoch']}/{record['epochs_run']}; "
                     f"OOB MAE/RMSE/max={error['mae']:.6f}/{error['rmse']:.6f}/{error['max_absolute_error']:.6f}")
    for name in ("all", "unsaturated", "saturated", "positive_saturated", "negative_saturated"):
        m = metrics["grid"][name]
        e, r, selection = m["bounded_action_error"], m["spearman"], m["selection"]
        lines.extend([
            f"\n{name}: n={m['count']}; MAE/RMSE/max={e['mae']:.6f}/{e['rmse']:.6f}/{e['max_absolute_error']:.6f}",
            f"  Spearman U-error={r['uncertainty_error']}; U-distance={r['uncertainty_distance']}; distance-error={r['distance_error']}",
            f"  Spearman raw-U-error={r['raw_uncertainty_error']}; raw-U-distance={r['raw_uncertainty_distance']}",
            f"  Zero bounded disagreement: {100 * m['zero_bounded_uncertainty_fraction']:.2f}%",
            f"  Top-error capture (budget={selection['budget']}): " + "; ".join(
                f"{key}={100 * value['capture_fraction']:.2f}%" for key, value in selection["methods"].items()),
            f"  Random expected/mean={100 * selection['random']['expected_capture_fraction']:.2f}%/"
            f"{100 * selection['random']['mean_capture_fraction']:.2f}%; "
            f"95% random range={selection['random']['central_95_percent_interval']}",
        ])
    lines.append("\nD0 ensemble fit (not held-out full-ensemble performance):")
    for name, result in metrics["d0"].items():
        e = result["bounded_action_error"]
        lines.append(f"  {name}: MAE/RMSE/max={e['mae']:.6f}/{e['rmse']:.6f}/{e['max_absolute_error']:.6f}")
    return "\n".join(lines) + "\n"


def main():
    config = EnsembleConfig()
    configure_determinism(config.seed)
    dataset_hash = file_hash(DATASET_PATH)
    data, states, actions = read_trajectories(DATASET_PATH)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    models, records = [], []
    for index, seeds in enumerate(member_seeds(config)):
        print(f"Training member {index + 1}/{config.members}...", flush=True)
        model, record = fit_member(
            states, actions, config, seeds,
            progress=lambda message: print(f"  {message}", flush=True),
        )
        oob = np.asarray(record["oob_ids"])
        oob_predictions = predict_ensemble([model], states[oob].reshape(-1, 2))
        record["oob_bounded_action_error"] = error_metrics(oob_predictions["bounded_mean"], actions[oob].ravel())
        record["oob_raw_action_error"] = error_metrics(oob_predictions["raw_mean"], actions[oob].ravel())
        models.append(model)
        records.append(record)
        save_member(RESULTS_DIR / "checkpoints" / f"member_{index + 1}.pt", model, config, record)
        print(f"  Restored epoch {record['best_epoch']} of {record['epochs_run']}; OOB RMSE={record['oob_bounded_action_error']['rmse']:.6f} N m", flush=True)

    print("Evaluating fixed grid, support distances, and selection baselines...", flush=True)
    arrays, grid_metrics, selection_arrays = evaluate_grid(models, data["states"])
    d0_predictions = predict_ensemble(models, data["states"])
    timestep = np.tile(np.arange(states.shape[1]), len(states))
    d0_metrics = {}
    for name, mask in {"all": timestep >= 0, "first_40_steps": timestep < 40, "remaining_160_steps": timestep >= 40}.items():
        d0_metrics[name] = {
            "count": int(mask.sum()),
            "bounded_action_error": error_metrics(d0_predictions["bounded_mean"][mask], data["actions"][mask, 0]),
            "uncertainty": describe(d0_predictions["uncertainty"][mask]),
        }
    metrics = {"grid": grid_metrics, "d0": d0_metrics}
    np.savez_compressed(RESULTS_DIR / "grid_evaluation.npz", **arrays)
    np.savez_compressed(RESULTS_DIR / "selection_masks.npz", **selection_arrays)
    np.savez_compressed(RESULTS_DIR / "d0_predictions.npz", **d0_predictions)
    history_arrays = {}
    for index, record in enumerate(records):
        for key, value in record["history"].items():
            history_arrays[f"member_{index + 1}_{key}"] = np.asarray(value)
        history_arrays[f"member_{index + 1}_bootstrap_ids"] = np.asarray(record["bootstrap_ids"])
        history_arrays[f"member_{index + 1}_oob_ids"] = np.asarray(record["oob_ids"])
    np.savez_compressed(RESULTS_DIR / "training_histories.npz", **history_arrays)
    metadata = {
        "config": asdict(config), "members": records,
        "dataset_path": str(DATASET_PATH.relative_to(PROJECT_ROOT)), "dataset_sha256": dataset_hash,
        "source_sha256": {name: file_hash(PROJECT_ROOT / name) for name in (
            "src/ensemble.py", "experiments/03_uncertainty_map.py", "src/expert.py",
            "src/safety.py", "src/dataset.py", "experiments/01_expert_map.py", "experiments/02_generate_D0.py")},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip(),
        "python": platform.python_version(), "platform": platform.platform(),
        "numpy": np.__version__, "torch": torch.__version__, "matplotlib": matplotlib.__version__,
        "device": "cpu", "dtype": "float32", "torch_threads": torch.get_num_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "input_scale": [THETA_SAFE, OMEGA_SAFE], "target_scale": MAX_TORQUE,
        "grid": {"size": GRID_SIZE, "theta_bounds": [-THETA_SAFE, THETA_SAFE], "omega_bounds": [-OMEGA_SAFE, OMEGA_SAFE], "flatten_order": "omega rows, theta columns; C order"},
        "uncertainty": "population std (ddof=0) of individually clipped torque outputs",
        "bounded_action": "mean of individually clipped network torques",
        "distance": "exact nearest stored D0 pre-action state, Euclidean in [theta/THETA_SAFE, omega/OMEGA_SAFE]",
        "saturation_mask": "abs(expert torque) within 1e-12 of MAX_TORQUE",
        "selection": {"fraction": SELECTION_FRACTION, "rounding": "ceil(fraction * region_size)", "evaluation_seed": EVALUATION_SEED,
                      "random_repeats": RANDOM_REPEATS, "tie_breaking": "independent seeded permutations per target/method/region",
                      "target": "top absolute errors of mean of individually clipped members relative to clipped PD expert",
                      "regions": "selections and top-error targets recomputed within each region"},
        "validation": "member-specific omitted trajectories; selected best raw normalized OOB MSE; no independent full-ensemble test set",
        "training_weighting": "uniform transitions including all 200 steps of each bootstrap draw; duplicates retained",
    }
    for name, value in (("metrics.json", metrics), ("metadata.json", metadata)):
        (RESULTS_DIR / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    save_plots(arrays, grid_metrics, records, data["states"], RESULTS_DIR)
    if file_hash(DATASET_PATH) != dataset_hash:
        raise RuntimeError("D0 changed during the experiment.")
    summary = make_summary(metrics, records)
    (RESULTS_DIR / "summary.txt").write_text(summary, encoding="utf-8")
    print(summary, flush=True)
    print(f"Artifacts: {RESULTS_DIR}", flush=True)


if __name__ == "__main__":
    main()
