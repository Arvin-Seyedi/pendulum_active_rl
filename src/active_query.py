"""Fixed-ranking expert acquisition with explicit costs and raw-state safety.

Selection uses only D0 and candidate states. Expert labels and future safety
are never arguments to ranking. Budget prefixes contain no later transitions.
"""

import gymnasium as gym
import numpy as np

from src.dataset import get_state
from src.ensemble import predict_ensemble
from src.expert import pd_controller
from src.safety import OMEGA_SAFE, THETA_SAFE, safety_margin

STRATEGIES = ("uncertainty", "distance", "random")
BUDGETS = (2000, 5000, 10000)
WIDE_LOW = (-0.8, -2.0)
WIDE_HIGH = (0.8, 2.0)


def seed_streams(master_seed):
    names = ("pool", "selector", "retraining", "ties", "random", "environment")
    children = np.random.SeedSequence(master_seed).spawn(len(names))
    return {name: int(child.generate_state(1)[0]) for name, child in zip(names, children)}


def sample_states(seed, count, low=WIDE_LOW, high=WIDE_HIGH):
    return np.random.default_rng(seed).uniform(low, high, size=(count, 2))


def nearest_d0_distance(states, support, chunk_size=128):
    """Exact Euclidean distance to all D0 states, scaled by safety limits."""
    scale = np.array([THETA_SAFE, OMEGA_SAFE], dtype=np.float64)
    points = np.asarray(states, dtype=np.float64) / scale
    support = np.asarray(support, dtype=np.float64) / scale
    if points.ndim != 2 or points.shape[1] != 2 or not len(support):
        raise ValueError("Expected (N,2) states and nonempty support.")
    if not np.isfinite(points).all() or not np.isfinite(support).all():
        raise ValueError("Nonfinite distance input.")
    result = np.empty(len(points))
    for start in range(0, len(points), chunk_size):
        delta = points[start:start + chunk_size, None, :] - support[None, :, :]
        result[start:start + chunk_size] = np.sqrt(
            np.einsum("ijk,ijk->ij", delta, delta).min(axis=1)
        )
    return result


def rank_candidates(pool, models, d0_states, seeds):
    predictions = predict_ensemble(models, pool)
    distances = nearest_d0_distance(pool, d0_states)
    # A common independent random key breaks exact score ties. It is unrelated
    # to error labels, safety outcomes, or the random strategy's permutation.
    tie_keys = np.random.default_rng(seeds["ties"]).permutation(len(pool))
    ranks = {
        "uncertainty": np.lexsort((tie_keys, -predictions["uncertainty"])),
        "distance": np.lexsort((tie_keys, -distances)),
        "random": np.random.default_rng(seeds["random"]).permutation(len(pool)),
    }
    diagnostics = dict(predictions, support_distance=distances, tie_keys=tie_keys)
    return ranks, diagnostics


def reset_at(env, state, seed):
    env.reset(seed=int(seed))
    env.unwrapped.state = np.array(state, dtype=np.float64, copy=True)
    theta, omega = state
    return np.array([np.cos(theta), np.sin(theta), omega], dtype=np.float64)


TRANSITION_KEYS = (
    "states", "raw_states", "actions", "rewards", "next_states", "raw_next_states",
    "margins", "next_margins", "terminated", "truncated", "dones", "safety_stop",
    "candidate_ids", "step_ids",
)


def _pack(rows, ends, initial_conditions, reasons):
    data = {key: np.asarray(rows[key]) for key in TRANSITION_KEYS}
    for key in ("states", "raw_states", "next_states", "raw_next_states"):
        data[key] = data[key].astype(np.float64).reshape(-1, 2)
    data["actions"] = data["actions"].astype(np.float32).reshape(-1, 1)
    for key in ("terminated", "truncated", "dones", "safety_stop"):
        data[key] = data[key].astype(bool)
    for key in ("candidate_ids", "step_ids"):
        data[key] = data[key].astype(np.int64)
    data["episode_offsets"] = np.asarray([0, *ends], dtype=np.int64)
    data["initial_conditions"] = np.asarray(initial_conditions, dtype=np.float64).reshape(-1, 2)
    data["stop_reasons"] = np.asarray(reasons, dtype="U24")
    data["budget_cutoff"] = np.zeros(len(data["states"]), dtype=bool)
    for end, reason in zip(ends, reasons):
        data["budget_cutoff"][end - 1] = reason == "budget"
    data["episode_end"] = np.zeros(len(data["states"]), dtype=bool)
    if ends:
        data["episode_end"][np.asarray(ends) - 1] = True
    offsets = data["episode_offsets"]
    data["episode_returns"] = np.asarray([
        data["rewards"][a:b].sum() for a, b in zip(offsets[:-1], offsets[1:])
    ])
    data["episode_min_margins"] = np.asarray([
        min(data["margins"][a:b].min(), data["next_margins"][a:b].min())
        for a, b in zip(offsets[:-1], offsets[1:])
    ])
    return data


class CollectionFailure(RuntimeError):
    """Fatal collection failure; partial valid data are retained for diagnosis."""

    def __init__(self, message, partial, attempts):
        super().__init__(message)
        self.partial = partial
        self.attempts = attempts


def collect_expert(pool, ranking, budget, env_seed, horizon=200, env_factory=None):
    """Consume a frozen ranking; count crossing transitions and all valid steps.

    Stop a rollout immediately after its first negative raw successor margin.
    Environment flags retain their original meanings; safety/budget stops have
    separate flags. Exceptions abort the run without choosing a replacement.
    """
    if budget < 1 or len(np.unique(ranking)) != len(ranking):
        raise ValueError("Positive budget and nonrepeating candidate ranking required.")
    rows = {key: [] for key in TRANSITION_KEYS}
    ends, starts, reasons = [], [], []
    env = (env_factory() if env_factory else
           gym.make("Pendulum-v1", max_episode_steps=horizon))
    attempts = 0
    try:
        for candidate_id in ranking:
            ic = np.asarray(pool[candidate_id])
            if not np.isfinite(ic).all() or safety_margin(*ic) < 0:
                raise ValueError("Candidate is not initially safe and finite.")
            reset_seed = int(np.random.SeedSequence([env_seed, int(candidate_id)]).generate_state(1)[0])
            obs = reset_at(env, ic, reset_seed)
            for step in range(horizon):
                state = np.asarray(get_state(obs), dtype=np.float64)
                raw = np.array(env.unwrapped.state, dtype=np.float64, copy=True)
                action = np.array([pd_controller(*state)], dtype=np.float32)
                attempts += 1
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = np.asarray(get_state(next_obs), dtype=np.float64)
                raw_next = np.array(env.unwrapped.state, dtype=np.float64, copy=True)
                if not np.isfinite(np.r_[state, raw, action, reward, next_state, raw_next]).all():
                    raise RuntimeError("Nonfinite transition.")
                margin, next_margin = safety_margin(*raw), safety_margin(*raw_next)
                crossed = next_margin < 0
                values = (state, raw, action, float(reward), next_state, raw_next,
                          margin, next_margin, terminated, truncated, terminated or truncated,
                          crossed, int(candidate_id), step)
                for key, value in zip(TRANSITION_KEYS, values):
                    rows[key].append(value)
                reason = ("safety" if crossed else "terminated" if terminated else
                          "time_limit" if truncated or step + 1 == horizon else
                          "budget" if len(rows["states"]) == budget else None)
                if reason is not None:
                    ends.append(len(rows["states"]))
                    starts.append(ic)
                    reasons.append(reason)
                    break
                obs = next_obs
            if len(rows["states"]) == budget:
                return _pack(rows, ends, starts, reasons)
        raise RuntimeError("Candidate pool exhausted before exact budget.")
    except Exception as exc:
        # Retain complete valid steps in an interrupted episode, if any. An
        # invalid simulator transition is never fabricated into the dataset.
        if len(rows["states"]) > (ends[-1] if ends else 0):
            ends.append(len(rows["states"]))
            starts.append(ic)
            reasons.append("technical_failure")
        raise CollectionFailure(str(exc), _pack(rows, ends, starts, reasons), attempts) from exc
    finally:
        env.close()


def budget_prefix(data, budget):
    """Slice flat transitions and reconstruct ragged boundaries without lookahead."""
    if not 0 < budget <= len(data["states"]):
        raise ValueError("Budget outside collected stream.")
    offsets = data["episode_offsets"]
    count = int(np.searchsorted(offsets[1:], budget, side="left")) + 1
    ends = np.minimum(offsets[1:count + 1], budget)
    reasons = data["stop_reasons"][:count].copy()
    if budget < offsets[count]:
        reasons[-1] = "budget"
    rows = {key: data[key][:budget].copy() for key in TRANSITION_KEYS}
    return _pack(rows, ends.tolist(), data["initial_conditions"][:count], reasons)


def validate_collection(data, budget, horizon=200):
    """Check budget, continuity, safety stopping, torque labels, and raw margins."""
    assert len(data["states"]) == budget
    offsets = data["episode_offsets"]
    assert offsets[0] == 0 and offsets[-1] == budget
    assert np.all((np.diff(offsets) > 0) & (np.diff(offsets) <= horizon))
    assert len(data["initial_conditions"]) == len(offsets) - 1
    assert np.allclose(data["actions"][:, 0], pd_controller(*data["states"].T), rtol=0, atol=1e-7)
    assert np.array_equal(data["dones"], data["terminated"] | data["truncated"])
    assert np.array_equal(data["margins"], [safety_margin(*s) for s in data["raw_states"]])
    assert np.array_equal(data["next_margins"], [safety_margin(*s) for s in data["raw_next_states"]])
    assert np.all(data["margins"] >= 0)
    assert np.array_equal(data["safety_stop"], data["next_margins"] < 0)
    visited = []
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        ids = data["candidate_ids"][start:end]
        assert np.all(ids == ids[0])
        visited.append(ids[0])
        assert np.array_equal(data["step_ids"][start:end], np.arange(end - start))
        assert np.array_equal(data["states"][start + 1:end], data["next_states"][start:end - 1])
        assert np.array_equal(data["raw_states"][start + 1:end], data["raw_next_states"][start:end - 1])
        assert np.array_equal(data["raw_states"][start], data["initial_conditions"][i])
        assert not data["safety_stop"][start:end - 1].any()
        assert not data["dones"][start:end - 1].any()
        reason = data["stop_reasons"][i]
        assert (reason == "safety") == data["safety_stop"][end - 1]
        assert (reason == "budget") == data["budget_cutoff"][end - 1]
        if reason == "budget":
            assert i == len(offsets) - 2
        assert data["episode_end"][end - 1]
    assert len(visited) == len(set(visited))
    return True


def distribution(values):
    values = np.asarray(values)
    return dict(count=int(values.size), mean=float(values.mean()),
                std=float(values.std()), minimum=float(values.min()),
                p05=float(np.quantile(values, .05)), median=float(np.median(values)),
                p95=float(np.quantile(values, .95)), maximum=float(values.max()))


def collection_summary(data, distances):
    return {
        "transitions": len(data["states"]),
        "rollouts": len(data["initial_conditions"]),
        "lengths": distribution(np.diff(data["episode_offsets"])),
        "stop_counts": {str(x): int(np.sum(data["stop_reasons"] == x)) for x in np.unique(data["stop_reasons"])},
        "safety_crossings": int(data["safety_stop"].sum()),
        "minimum_raw_margin": float(min(data["margins"].min(), data["next_margins"].min())),
        "new_transition_nearest_D0_distance": distribution(distances),
    }
