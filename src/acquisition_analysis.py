"""Seed-paired statistics and descriptive acquisition coverage diagnostics.

These functions do not participate in selection, collection, or fitting.
Thresholds and bin widths are frozen before the full study, not tuned to scores.
"""

import numpy as np

from src.active_query import distribution
from src.safety import OMEGA_SAFE, THETA_SAFE

SUPPORT_THRESHOLDS = (.001, .01, .05)
EQUILIBRIUM_RADII = (.01, .05, .1)
BIN_WIDTHS = (.05, .1)
BOOTSTRAP_SEED = 404010
BOOTSTRAP_REPEATS = 10000
SCALE = np.array([THETA_SAFE, OMEGA_SAFE])


def seed_summary(values, bootstrap_indices):
    """Sample SD and percentile CI for the mean; resampling unit = master seed."""
    values = np.asarray(values, dtype=np.float64)
    indices = np.asarray(bootstrap_indices)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite seed-level values.")
    if indices.ndim != 2 or indices.shape[1] != len(values):
        raise ValueError("Bootstrap draws must contain N seed indices each.")
    bootstrap_means = values[indices].mean(axis=1)
    ci = np.quantile(bootstrap_means, [.025, .975])
    return {"n_seeds": len(values), "per_seed": values.tolist(),
            "mean": float(values.mean()), "sample_std": float(values.std(ddof=1)),
            "minimum": float(values.min()), "maximum": float(values.max()),
            "bootstrap_95_ci_mean": ci.tolist(),
            "negative_count": int(np.sum(values < 0)),
            "zero_count": int(np.sum(values == 0)), "positive_count": int(np.sum(values > 0))}


def tail_diagnostic(distances):
    """Decompose a mean into the bottom 95% and largest 5% order statistics."""
    distances = np.asarray(distances, dtype=np.float64)
    ordered = np.sort(distances)
    k = max(1, int(np.ceil(.05 * len(ordered))))
    lower, upper = ordered[:-k], ordered[-k:]
    total = float(ordered.sum())
    mean_from_parts = (lower.sum() + upper.sum()) / len(ordered)
    return {
        "distribution": distribution(distances), "largest_5_percent_count": k,
        "largest_5_percent_mean": float(upper.mean()),
        "remaining_mean": float(lower.mean()) if len(lower) else None,
        "largest_5_percent_share_of_distance_sum": float(upper.sum() / total) if total else None,
        "mean_reconstructed_from_parts": float(mean_from_parts),
        "exact_zero_count": int(np.sum(distances == 0)),
        "quantile_method": "numpy linear interpolation", "finite": bool(np.isfinite(distances).all()),
    }


def occupied_bins(states, width):
    coordinates = np.floor(np.asarray(states) / SCALE / width).astype(np.int64)
    return set(map(tuple, coordinates))


def coverage_diagnostics(data, distances, d0_states):
    """Retain every trajectory's diagnostics; summarize both weighting schemes.

    Equilibrium radii use ||[theta/1,omega/3]||_2 on raw collected states.
    'Observed settled step' only describes the observed suffix, not a guarantee
    about the unobserved continuation of an early-terminated trajectory.
    """
    distances = np.asarray(distances, dtype=np.float64)
    if distances.shape != (len(data["states"]),):
        raise ValueError("One support distance per newly collected transition required.")
    if not np.isfinite(distances).all() or np.any(distances < 0):
        raise ValueError("Invalid support distances.")
    radii = np.linalg.norm(data["raw_states"] / SCALE, axis=1)
    offsets = data["episode_offsets"]
    trajectories = []
    for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        d, r = distances[a:b], radii[a:b]
        outside = np.flatnonzero(r > .05)
        settled = int(outside[-1] + 1) if len(outside) else 0
        record = {
            "episode": i, "candidate_id": int(data["candidate_ids"][a]),
            "length": int(b - a), "initial_condition": data["initial_conditions"][i].tolist(),
            "stop_reason": str(data["stop_reasons"][i]), "distance": distribution(d),
            "initial_distance": float(d[0]),
            "distance_mean_first_20": float(d[:20].mean()),
            "distance_mean_after_20": float(d[20:].mean()) if len(d) > 20 else None,
            "fraction_near_D0": {str(t): float(np.mean(d <= t)) for t in SUPPORT_THRESHOLDS},
            "fraction_near_equilibrium": {str(t): float(np.mean(r <= t)) for t in EQUILIBRIUM_RADII},
            "observed_settled_step_radius_0.05": settled if settled < len(r) else None,
            "observed_settled_suffix_length": int(len(r) - settled) if settled < len(r) else 0,
        }
        trajectories.append(record)
    time_records = []
    for step in range(200):
        mask = data["step_ids"] == step
        time_records.append({"step": step, "count": int(mask.sum()),
                             "mean_distance": float(distances[mask].mean()) if mask.any() else None,
                             "median_distance": float(np.median(distances[mask])) if mask.any() else None,
                             "near_equilibrium_fraction": float(np.mean(radii[mask] <= .05)) if mask.any() else None})
    bins = {}
    for width in BIN_WIDTHS:
        old, new = occupied_bins(d0_states, width), occupied_bins(data["states"], width)
        coords = np.floor(data["states"] / SCALE / width).astype(np.int64)
        bins[str(width)] = {
            "D0_occupied_bins": len(old), "new_data_occupied_bins": len(new),
            "additional_occupied_bins": len(new - old), "union_occupied_bins": len(old | new),
            "fraction_transitions_outside_D0_bins": float(np.mean([tuple(s) not in old for s in coords])),
        }
    phases = {}
    for name, mask in (("first_20", data["step_ids"] < 20), ("after_20", data["step_ids"] >= 20)):
        phases[name] = {"count": int(mask.sum()),
                        "distance": distribution(distances[mask]) if mask.any() else None,
                        "fraction_near_equilibrium_0.05": float(np.mean(radii[mask] <= .05)) if mask.any() else None}
    strata = {}
    for reason in np.unique(data["stop_reasons"]):
        records = [t for t in trajectories if t["stop_reason"] == reason]
        lengths = np.array([t["length"] for t in records])
        strata[str(reason)] = {
            "trajectories": len(records), "transitions": int(lengths.sum()),
            "trajectory_mean_distance": float(np.mean([t["distance"]["mean"] for t in records])),
            "transition_mean_distance": float(np.average([t["distance"]["mean"] for t in records], weights=lengths)),
            "trajectory_mean_fraction_near_equilibrium_0.05": float(np.mean([t["fraction_near_equilibrium"]["0.05"] for t in records])),
        }
    summary = {
        "transitions": len(distances), "trajectories": len(trajectories),
        "tail": tail_diagnostic(distances),
        "trajectory_mean_distances": distribution([t["distance"]["mean"] for t in trajectories]),
        "trajectory_initial_distances": distribution([t["initial_distance"] for t in trajectories]),
        "trajectory_near_equilibrium_fractions_0.05": distribution([t["fraction_near_equilibrium"]["0.05"] for t in trajectories]),
        "transition_fraction_near_D0": {str(t): float(np.mean(distances <= t)) for t in SUPPORT_THRESHOLDS},
        "transition_fraction_near_equilibrium": {str(t): float(np.mean(radii <= t)) for t in EQUILIBRIUM_RADII},
        "distance_by_phase": phases, "stop_reason_strata": strata, "bin_coverage": bins,
        "thresholds_are_descriptive_only": True,
    }
    return {"summary": summary, "trajectories": trajectories, "by_step": time_records}


def flatten_numeric(record, prefix=""):
    result = {}
    for key, value in record.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten_numeric(value, name))
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            result[name] = value
    return result
