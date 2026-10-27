"""Rollout-horizon diagnostics; no changes to the verified acquisition rule.

The post-selection audit accepts only selected IDs, a pool, and reset seeds.
It cannot update rankings, training data, or fitted models.
"""

import time

import gymnasium as gym
import numpy as np

from src.active_query import (
    CollectionFailure, STRATEGIES, collect_expert, distribution, reset_at,
    validate_collection,
)
from src.dataset import get_state
from src.expert import pd_controller
from src.safety import safety_margin

HORIZONS = (20, 200)
RESET_COSTS = (0, 20, 200)
PHYSICAL_KEYS = (
    "states", "raw_states", "actions", "rewards", "next_states", "raw_next_states",
    "margins", "next_margins", "terminated", "safety_stop", "candidate_ids", "step_ids",
)


class MeteredEnvironment(gym.Wrapper):
    """Observe reset/step costs without changing dynamics or termination flags."""

    def __init__(self, horizon):
        super().__init__(gym.make("Pendulum-v1", max_episode_steps=horizon))
        self.reset_seconds, self.step_seconds = [], []
        self.reset_attempts = self.step_attempts = 0

    def reset(self, **kwargs):
        self.reset_attempts += 1
        start = time.perf_counter()
        result = self.env.reset(**kwargs)
        self.reset_seconds.append(time.perf_counter() - start)
        return result

    def step(self, action):
        self.step_attempts += 1
        start = time.perf_counter()
        result = self.env.step(action)
        self.step_seconds.append(time.perf_counter() - start)
        return result

    def costs(self):
        return {
            "reset_seconds": np.asarray(self.reset_seconds),
            "step_seconds": np.asarray(self.step_seconds),
            "reset_attempts": np.asarray(self.reset_attempts),
            "step_attempts": np.asarray(self.step_attempts),
        }


def collect_metered(pool, ranking, budget, env_seed, horizon):
    env = MeteredEnvironment(horizon)
    start = time.perf_counter()
    try:
        data = collect_expert(pool, ranking, budget, env_seed, horizon, lambda: env)
    except CollectionFailure as exc:
        exc.operation_costs = env.costs()
        raise
    costs = env.costs()
    costs["collection_wall_seconds"] = np.asarray(time.perf_counter() - start)
    validate_horizon_collection(data, budget, horizon)
    assert int(costs["step_attempts"]) == budget
    assert int(costs["reset_attempts"]) == len(data["initial_conditions"])
    return data, costs


def validate_horizon_collection(data, budget, horizon):
    validate_collection(data, budget, horizon)
    assert horizon in HORIZONS
    assert not data["terminated"].any()
    for a, b, reason in zip(data["episode_offsets"][:-1], data["episode_offsets"][1:], data["stop_reasons"]):
        assert bool(data["truncated"][b - 1]) == (b - a == horizon)
        if reason == "time_limit":
            assert b - a == horizon and not data["safety_stop"][b - 1]
        if reason == "budget":
            assert b - a < horizon and not data["dones"][b - 1]
    for value in data.values():
        if np.issubdtype(value.dtype, np.number):
            assert np.isfinite(value).all()
    return True


def episode_slices(data):
    return {int(data["candidate_ids"][a]): (int(a), int(b))
            for a, b in zip(data["episode_offsets"][:-1], data["episode_offsets"][1:])}


def shared_prefix_check(short, long):
    """Compare physical prefixes, excluding legitimate horizon/budget flag changes."""
    short_slices, long_slices = episode_slices(short), episode_slices(long)
    shared = sorted(set(short_slices) & set(long_slices))
    compared = 0
    for candidate in shared:
        a, b = short_slices[candidate]
        c, d = long_slices[candidate]
        length = min(b - a, d - c)
        for key in PHYSICAL_KEYS:
            np.testing.assert_array_equal(short[key][a:a + length], long[key][c:c + length],
                                          err_msg=f"candidate {candidate}, {key}")
        compared += length
    assert len(shared) == min(len(short_slices), len(long_slices))
    return {"passed": True, "shared_candidates": len(shared), "physical_transitions_compared": compared}


def selection_diagnostics(data, rankings, scores, strategy):
    """One row per queried initial condition, not per collected transition."""
    ids = data["candidate_ids"][data["episode_offsets"][:-1]]
    arrays = {"candidate_ids": ids.copy(), "initial_conditions": data["initial_conditions"].copy(),
              "episode_lengths": np.diff(data["episode_offsets"])}
    for name in STRATEGIES:
        ranks = np.empty(len(rankings[name]), dtype=np.int64)
        ranks[rankings[name]] = np.arange(1, len(ranks) + 1)
        arrays[f"{name}_rank_1based"] = ranks[ids]
    for key in ("uncertainty", "raw_uncertainty", "support_distance"):
        arrays[key] = scores[key][ids]
    np.testing.assert_array_equal(ids, rankings[strategy][:len(ids)])
    positive = scores["uncertainty"] > 0
    zeros = np.flatnonzero(arrays["uncertainty"] == 0)
    score_key = {"uncertainty": "uncertainty", "distance": "support_distance", "random": None}[strategy]
    summary = {
        "queried_initial_conditions": len(ids), "pool_count": len(positive),
        "selection_score_name": score_key,
        "selection_score": distribution(arrays[score_key]) if score_key else None,
        "random_selection_note": "Uniform permutation; no numerical acquisition score" if score_key is None else None,
        "bounded_disagreement": distribution(arrays["uncertainty"]),
        "raw_disagreement": distribution(arrays["raw_uncertainty"]),
        "support_distance": distribution(arrays["support_distance"]),
        "rank_positions_1based": {name: distribution(arrays[f"{name}_rank_1based"]) for name in STRATEGIES},
        "fraction_zero_bounded_disagreement": float(np.mean(arrays["uncertainty"] == 0)),
        "first_zero_disagreement_query_1based": int(zeros[0] + 1) if len(zeros) else None,
        "positive_disagreement_pool_count": int(positive.sum()),
        "positive_disagreement_queried_count": int(positive[ids].sum()),
        "fraction_positive_pool_queried": float(positive[ids].sum() / positive.sum()) if positive.any() else None,
        "pool_fraction_queried": float(len(ids) / len(positive)),
        "last_queried_candidate_bounded_disagreement": float(arrays["uncertainty"][-1]),
    }
    return arrays, summary


def operational_summary(data, costs, horizon):
    budget, resets = len(data["states"]), len(data["initial_conditions"])
    lengths = np.diff(data["episode_offsets"])
    crossings = int(data["safety_stop"].sum())
    first_crossings = [int(length) for length, reason in zip(lengths, data["stop_reasons"]) if reason == "safety"]
    return {
        "valid_transitions": budget, "resets": resets, "successful_reset_attempts": resets,
        "successful_step_attempts": budget, "technical_failures": 0,
        "collection_horizon": horizon, "minimum_resets_without_early_stops": int(np.ceil(budget / horizon)),
        "collection_safety_crossings": crossings,
        "crossings_per_1000_transitions": 1000 * crossings / budget,
        "fraction_rollouts_with_observed_crossing": crossings / resets,
        "first_crossing_steps": first_crossings,
        "first_crossing_step_distribution": distribution(first_crossings) if first_crossings else None,
        "safely_horizon_censored_rollouts": int(np.sum(data["stop_reasons"] == "time_limit")),
        "budget_censored_rollouts": int(np.sum(data["stop_reasons"] == "budget")),
        "simulator_step_seconds": float(costs["step_seconds"][:budget].sum()),
        "simulator_reset_seconds": float(costs["reset_seconds"][:resets].sum()),
        "transition_equivalent_cost": {str(cost): budget + cost * resets for cost in RESET_COSTS},
        "cost_note": "Prefix simulator timings; excludes ranking, training, evaluation, technical replay, and post-selection audit. Reset equivalents are illustrative.",
    }


def expert_safety_audit(pool, candidate_ids, env_seed, horizon=200):
    """Isolated PD audit through first crossing or 200 steps, never training data.

    Reuses expert, reset, observation conversion, and safety functions. Unlike
    budgeted acquisition, every requested candidate must finish its own audit.
    """
    ids = np.asarray(sorted(set(map(int, candidate_ids))), dtype=np.int64)
    env = MeteredEnvironment(horizon)
    rows = {key: [] for key in PHYSICAL_KEYS}
    offsets, violations = [0], []
    start = time.perf_counter()
    try:
        for candidate_id in ids:
            ic = pool[candidate_id]
            assert np.isfinite(ic).all() and safety_margin(*ic) >= 0
            reset_seed = int(np.random.SeedSequence([env_seed, int(candidate_id)]).generate_state(1)[0])
            obs = reset_at(env, ic, reset_seed)
            violation = -1
            for step in range(horizon):
                state = np.asarray(get_state(obs), dtype=np.float64)
                raw = np.array(env.unwrapped.state, dtype=np.float64, copy=True)
                action = np.array([pd_controller(*state)], dtype=np.float32)
                obs, reward, term, trunc, _ = env.step(action)
                next_state = np.asarray(get_state(obs), dtype=np.float64)
                raw_next = np.array(env.unwrapped.state, dtype=np.float64, copy=True)
                margin, next_margin = safety_margin(*raw), safety_margin(*raw_next)
                values = (state, raw, action, float(reward), next_state, raw_next, margin,
                          next_margin, term, next_margin < 0, int(candidate_id), step)
                assert np.isfinite(np.r_[state, raw, action, reward, next_state, raw_next]).all()
                assert not term and (not trunc or step + 1 == horizon)
                for key, value in zip(PHYSICAL_KEYS, values):
                    rows[key].append(value)
                if next_margin < 0:
                    violation = step + 1
                    break
            offsets.append(len(rows["states"]))
            violations.append(violation)
    finally:
        env.close()
    arrays = {key: np.asarray(value) for key, value in rows.items()}
    arrays.update(audited_candidate_ids=ids, initial_conditions=pool[ids],
                  episode_offsets=np.asarray(offsets, dtype=np.int64),
                  first_violation_step=np.asarray(violations, dtype=np.int64))
    costs = env.costs()
    costs["audit_wall_seconds"] = np.asarray(time.perf_counter() - start)
    assert int(costs["step_attempts"]) == len(arrays["states"])
    return arrays, costs


def summarize_safety_audit(data, audit, collection_horizon):
    lookup = {int(cid): (int(a), int(b), int(v)) for cid, a, b, v in zip(
        audit["audited_candidate_ids"], audit["episode_offsets"][:-1],
        audit["episode_offsets"][1:], audit["first_violation_step"])}
    selected = list(episode_slices(data))
    events, capped_events = [], []
    compared = 0
    for i, (cid, (a, b)) in enumerate(episode_slices(data).items()):
        c, d, event = lookup[cid]
        assert d - c >= b - a
        for key in PHYSICAL_KEYS:
            np.testing.assert_array_equal(data[key][a:b], audit[key][c:c + b - a], err_msg=f"audit prefix {cid}/{key}")
        events.append(event)
        compared += b - a
        if collection_horizon == 20 and b - a == 20 and data["stop_reasons"][i] == "time_limit":
            assert event == -1 or event > 20
            capped_events.append(event)
    events, capped_events = np.asarray(events), np.asarray(capped_events)
    late = int(np.sum(capped_events > 20))
    return {
        "selected_initial_conditions": len(selected),
        "expert_violations_by_step_20": int(np.sum((events > 0) & (events <= 20))),
        "expert_violations_by_step_200": int(np.sum(events > 0)),
        "expert_violation_fraction_by_step_200": float(np.mean(events > 0)),
        "safely_capped_at_20_count": len(capped_events),
        "subsequent_violation_steps_21_to_200_count": late,
        "subsequent_violation_fraction_among_safely_capped_20": late / len(capped_events) if len(capped_events) else None,
        "subsequent_violation_step_distribution": distribution(capped_events[capped_events > 20]) if late else None,
        "collection_prefixes_reproduced": True, "physical_transitions_compared": compared,
        "audit_only_never_training_data": True,
    }
