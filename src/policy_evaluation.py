"""Independent simulator rollouts and held-out expert imitation diagnostics."""

import gymnasium as gym
import numpy as np

from src.active_query import distribution, reset_at
from src.dataset import get_state
from src.ensemble import predict_ensemble
from src.expert import KD, KP, MAX_TORQUE, pd_controller
from src.safety import safety_margin


def evaluate_rollouts(models, initial_conditions, reset_seed=404004, horizon=200):
    """Evaluate the ensemble (or PD if models=None) for the complete horizon.

    Batch neural inference, but use actual Gymnasium environments for dynamics.
    No safety termination: all 201 raw visited states enter safety diagnostics.
    Policy inputs are canonical observation states, consistent with D0. Raw
    simulator angles are not wrapped or clipped for safety measurements.
    """
    n = len(initial_conditions)
    states = np.empty((n, horizon + 1, 2), dtype=np.float64)
    raw_states = np.empty_like(states)
    margins = np.empty((n, horizon + 1), dtype=np.float64)
    actions = np.empty((n, horizon, 1), dtype=np.float32)
    rewards = np.empty((n, horizon), dtype=np.float64)
    terminated = np.zeros((n, horizon), dtype=bool)
    truncated = np.zeros_like(terminated)
    envs = []
    try:
        for i, ic in enumerate(initial_conditions):
            env = gym.make("Pendulum-v1", max_episode_steps=horizon)
            envs.append(env)
            obs = reset_at(env, ic, reset_seed + i)
            states[i, 0] = get_state(obs)
            raw_states[i, 0] = env.unwrapped.state
            margins[i, 0] = safety_margin(*raw_states[i, 0])
        for step in range(horizon):
            current = states[:, step]
            torques = (pd_controller(*current.T) if models is None else
                       predict_ensemble(models, current)["bounded_mean"])
            actions[:, step, 0] = torques
            for i, env in enumerate(envs):
                obs, reward, term, trunc, _ = env.step(actions[i, step])
                states[i, step + 1] = get_state(obs)
                raw_states[i, step + 1] = env.unwrapped.state
                margins[i, step + 1] = safety_margin(*raw_states[i, step + 1])
                rewards[i, step] = reward
                terminated[i, step], truncated[i, step] = term, trunc
                if (term or trunc) and step + 1 < horizon:
                    raise RuntimeError("Unexpected early environment termination during evaluation.")
        if not np.all(truncated[:, -1]) or terminated.any():
            raise RuntimeError("Unexpected Pendulum-v1 termination semantics.")
    finally:
        for env in envs:
            env.close()
    arrays = dict(initial_conditions=np.asarray(initial_conditions), states=states,
                  raw_states=raw_states, margins=margins, actions=actions, rewards=rewards,
                  terminated=terminated, truncated=truncated,
                  episode_returns=rewards.sum(axis=1), episode_min_margins=margins.min(axis=1))
    for key in ("states", "raw_states", "margins", "actions", "rewards"):
        if not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"Nonfinite evaluation {key}.")
    return arrays, rollout_metrics(arrays)


def rollout_metrics(arrays):
    returns, minima = arrays["episode_returns"], arrays["episode_min_margins"]
    return {
        "episodes": len(returns), "horizon": arrays["rewards"].shape[1],
        "mean_return": float(returns.mean()), "worst_return": float(returns.min()),
        "worst_10_percent_mean_return": float(np.sort(returns)[:int(np.ceil(.1 * len(returns)))].mean()),
        "episode_return_std": float(returns.std(ddof=1)),
        "safety_violation_rate": float(np.mean(minima < 0)),
        "violating_episodes": int(np.sum(minima < 0)),
        "minimum_raw_safety_margin": float(minima.min()),
        "unsafe_visited_state_fraction": float(np.mean(arrays["margins"] < 0)),
    }


def action_error(predicted, expert):
    error = np.abs(predicted - expert)
    return {"mae": float(error.mean()), "rmse": float(np.sqrt(np.mean(error**2))),
            "p95": float(np.quantile(error, .95)), "maximum": float(error.max())}


def audit_policy(models, states):
    """Labels are called only after acquisition/training, on independent states."""
    predictions = predict_ensemble(models, states)
    expert = pd_controller(*states.T)
    saturated = np.abs(-KP * states[:, 0] - KD * states[:, 1]) >= MAX_TORQUE
    arrays = dict(predictions, states=states, expert_actions=expert, saturated=saturated,
                  absolute_error=np.abs(predictions["bounded_mean"] - expert))
    metrics = {}
    for name, mask in (("all", np.ones(len(states), dtype=bool)),
                       ("unsaturated", ~saturated), ("saturated", saturated)):
        metrics[name] = {
            "count": int(mask.sum()),
            "bounded_action_error": action_error(predictions["bounded_mean"][mask], expert[mask]),
            "raw_action_error": action_error(predictions["raw_mean"][mask], expert[mask]),
            "bounded_disagreement": distribution(predictions["uncertainty"][mask]),
            "raw_disagreement": distribution(predictions["raw_uncertainty"][mask]),
            "zero_bounded_disagreement_fraction": float(np.mean(predictions["uncertainty"][mask] == 0)),
        }
    return arrays, metrics
