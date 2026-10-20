"""Collect the nominal Pendulum-v1 dataset with the shared PD expert.

Run from any directory with ``python /path/to/experiments/02_generate_D0.py``.
Transitions are flattened in episode-major order. States are [theta, omega],
actions are the float32 torques passed to Gymnasium, and rewards are unmodified.
``margins`` and ``episode_min_margins`` describe pre-action states, matching the
sampling convention in 01_expert_map.py. ``next_margins`` also records safety
at successor states, including the final state of each trajectory.
``dones`` is terminated OR truncated; the separate flags preserve the distinction
between an MDP terminal state and a time limit for subsequent offline learning.
"""

from pathlib import Path
import sys

import gymnasium as gym
import matplotlib
import numpy as np

# Save figures without opening windows or waiting for user interaction.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_state
from src.expert import KD, KP, MAX_TORQUE, pd_controller
from src.safety import OMEGA_SAFE, THETA_SAFE, safety_margin


SEED = 42
NUM_TRAJECTORIES = 100
STEPS_PER_TRAJECTORY = 200
THETA_BOUNDS = (-0.25, 0.25)
OMEGA_BOUNDS = (-0.5, 0.5)
DATASET_PATH = PROJECT_ROOT / "data" / "pendulum_D0.npz"
RESULTS_DIR = PROJECT_ROOT / "results" / "D0"


def generate_dataset():
    """Generate exactly 100 independent, 200-step expert trajectories."""
    rng = np.random.default_rng(SEED)
    # Each row consumes a theta draw followed by an omega draw. Keep the IC
    # stream independent of Gymnasium's reset RNG and record its algorithm.
    initial_conditions = rng.uniform(
        low=[THETA_BOUNDS[0], OMEGA_BOUNDS[0]],
        high=[THETA_BOUNDS[1], OMEGA_BOUNDS[1]],
        size=(NUM_TRAJECTORIES, 2),
    )
    num_transitions = NUM_TRAJECTORIES * STEPS_PER_TRAJECTORY
    dataset = {
        "states": np.empty((num_transitions, 2), dtype=np.float64),
        "actions": np.empty((num_transitions, 1), dtype=np.float32),
        "rewards": np.empty(num_transitions, dtype=np.float64),
        "next_states": np.empty((num_transitions, 2), dtype=np.float64),
        "margins": np.empty(num_transitions, dtype=np.float64),
        "next_margins": np.empty(num_transitions, dtype=np.float64),
        "dones": np.empty(num_transitions, dtype=bool),
        "terminated": np.empty(num_transitions, dtype=bool),
        "truncated": np.empty(num_transitions, dtype=bool),
        "initial_conditions": initial_conditions,
    }

    env = gym.make("Pendulum-v1", max_episode_steps=STEPS_PER_TRAJECTORY)
    try:
        dt = env.unwrapped.dt
        for episode, (theta0, omega0) in enumerate(initial_conditions):
            env.reset(seed=SEED if episode == 0 else None)
            env.unwrapped.state = np.array([theta0, omega0], dtype=np.float64)
            # As in 01_expert_map.py, reconstruct the observation after setting
            # the initial state; the observation returned by reset is stale.
            obs = np.array([np.cos(theta0), np.sin(theta0), omega0])

            for step in range(STEPS_PER_TRAJECTORY):
                index = episode * STEPS_PER_TRAJECTORY + step
                state = get_state(obs)
                action = np.array([pd_controller(*state)], dtype=np.float32)
                next_obs, reward, terminated, truncated, _ = env.step(action)
                next_state = get_state(next_obs)
                done = terminated or truncated

                dataset["states"][index] = state
                dataset["actions"][index] = action
                dataset["rewards"][index] = reward
                dataset["next_states"][index] = next_state
                dataset["margins"][index] = safety_margin(*state)
                dataset["next_margins"][index] = safety_margin(*next_state)
                dataset["dones"][index] = done
                dataset["terminated"][index] = terminated
                dataset["truncated"][index] = truncated

                if done and step != STEPS_PER_TRAJECTORY - 1:
                    raise RuntimeError(
                        f"Episode {episode} ended early at step {step + 1}; "
                        "refusing to save an incomplete D0 dataset."
                    )
                if step == STEPS_PER_TRAJECTORY - 1 and not done:
                    raise RuntimeError("Expected the 200-step time limit.")
                obs = next_obs
    finally:
        env.close()

    episode_shape = (NUM_TRAJECTORIES, STEPS_PER_TRAJECTORY)
    dataset.update(
        episode_returns=dataset["rewards"].reshape(episode_shape).sum(axis=1),
        episode_min_margins=dataset["margins"].reshape(episode_shape).min(axis=1),
        seed=np.array(SEED),
        rng_algorithm=np.array(type(rng.bit_generator).__name__),
        num_trajectories=np.array(NUM_TRAJECTORIES),
        steps_per_trajectory=np.array(STEPS_PER_TRAJECTORY),
        theta_bounds=np.array(THETA_BOUNDS),
        omega_bounds=np.array(OMEGA_BOUNDS),
        dt=np.array(dt),
        environment=np.array("Pendulum-v1"),
        numpy_version=np.array(np.__version__),
        gymnasium_version=np.array(gym.__version__),
        expert_kp=np.array(KP),
        expert_kd=np.array(KD),
        max_torque=np.array(MAX_TORQUE),
        theta_safe=np.array(THETA_SAFE),
        omega_safe=np.array(OMEGA_SAFE),
        margin_convention=np.array("margins=m(s_t); next_margins=m(s_{t+1})"),
    )
    return dataset


def save_diagnostics(dataset, output_dir):
    """Plot support, per-episode outcomes, and all closed-loop transients."""
    output_dir.mkdir(parents=True, exist_ok=True)
    states = dataset["states"]
    initial_conditions = dataset["initial_conditions"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].scatter(states[:, 0], states[:, 1], s=3, alpha=0.15,
                    label="D0 pre-action states")
    axes[0].scatter(*initial_conditions.T, s=14, color="tab:orange",
                    label="Initial conditions")
    axes[0].add_patch(Rectangle(
        (-THETA_SAFE, -OMEGA_SAFE), 2 * THETA_SAFE, 2 * OMEGA_SAFE,
        fill=False, linestyle="--", color="tab:red", label="Safety boundary",
    ))
    # Include the full safe set and any observed outliers; never clip data.
    axes[0].set_xlim(-1.1 * max(THETA_SAFE, np.abs(states[:, 0]).max()),
                     1.1 * max(THETA_SAFE, np.abs(states[:, 0]).max()))
    axes[0].set_ylim(-1.1 * max(OMEGA_SAFE, np.abs(states[:, 1]).max()),
                     1.1 * max(OMEGA_SAFE, np.abs(states[:, 1]).max()))
    axes[0].set_title("D0 coverage within the safety set")
    axes[0].legend(loc="upper right", fontsize=8)
    points = axes[1].scatter(
        *initial_conditions.T, c=dataset["episode_min_margins"],
        s=30, cmap="viridis",
    )
    axes[1].set_title("Initial conditions and rollout safety")
    axes[1].set_xlim(THETA_BOUNDS)
    axes[1].set_ylim(OMEGA_BOUNDS)
    fig.colorbar(points, ax=axes[1], label="Episode minimum pre-action margin")
    for ax in axes:
        ax.set_xlabel("Angle [rad]")
        ax.set_ylabel("Angular velocity [rad/s]")
        ax.grid(alpha=0.2)
    fig.savefig(output_dir / "state_coverage.png", dpi=180)
    plt.close(fig)

    episodes = np.arange(1, NUM_TRAJECTORIES + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    axes[0].plot(episodes, dataset["episode_returns"], ".", color="tab:blue")
    axes[0].axhline(dataset["episode_returns"].mean(), color="black",
                    linestyle="--", label="Mean return")
    axes[0].set_ylabel("Undiscounted episode return")
    axes[0].set_title("Expert performance")
    axes[0].legend()
    axes[1].plot(episodes, dataset["episode_min_margins"], ".",
                 color="tab:green", label="Minimum pre-action margin")
    axes[1].axhline(0, color="tab:red", linestyle="--", label="Safety boundary")
    axes[1].set_ylabel("Normalized safety margin")
    axes[1].set_title("Expert safety")
    axes[1].legend()
    for ax in axes:
        ax.set_xlabel("Episode")
        ax.grid(alpha=0.2)
    fig.savefig(output_dir / "episode_diagnostics.png", dpi=180)
    plt.close(fig)

    trajectory_shape = (NUM_TRAJECTORIES, STEPS_PER_TRAJECTORY, 2)
    paths = np.concatenate([
        states.reshape(trajectory_shape),
        dataset["next_states"].reshape(trajectory_shape)[:, -1:, :],
    ], axis=1)
    margin_paths = np.concatenate([
        dataset["margins"].reshape(NUM_TRAJECTORIES, STEPS_PER_TRAJECTORY),
        dataset["next_margins"].reshape(NUM_TRAJECTORIES, STEPS_PER_TRAJECTORY)[:, -1:],
    ], axis=1)
    time = np.arange(STEPS_PER_TRAJECTORY + 1) * float(dataset["dt"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    axes[0, 0].plot(time, paths[:, :, 0].T, alpha=0.25, linewidth=0.8)
    axes[0, 0].set_ylabel("Angle [rad]")
    axes[0, 1].plot(time, paths[:, :, 1].T, alpha=0.25, linewidth=0.8)
    axes[0, 1].set_ylabel("Angular velocity [rad/s]")
    axes[1, 0].step(
        time[:-1], dataset["actions"].reshape(NUM_TRAJECTORIES, -1).T,
        where="post", alpha=0.25, linewidth=0.8,
    )
    for bound in (-MAX_TORQUE, MAX_TORQUE):
        axes[1, 0].axhline(bound, color="tab:red", linestyle="--", linewidth=1)
    axes[1, 0].set_ylabel("Torque [N m]")
    axes[1, 1].plot(time, margin_paths.T, alpha=0.25, linewidth=0.8)
    axes[1, 1].axhline(0, color="tab:red", linestyle="--", label="Safety boundary")
    axes[1, 1].set_ylabel("Normalized safety margin")
    axes[1, 1].legend()
    for ax in axes.flat:
        ax.set_xlabel("Time [s]")
        ax.grid(alpha=0.2)
    fig.suptitle("All 100 PD expert trajectories (states include the final sample)")
    fig.savefig(output_dir / "trajectories.png", dpi=180)
    plt.close(fig)


def main():
    dataset = generate_dataset()
    DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(DATASET_PATH, **dataset)
    save_diagnostics(dataset, RESULTS_DIR)

    returns = dataset["episode_returns"]
    final_margins = dataset["next_margins"].reshape(
        NUM_TRAJECTORIES, STEPS_PER_TRAJECTORY
    )[:, -1]
    summary = "\n".join([
        f"D0: Pendulum-v1, seed={SEED}, RNG={dataset['rng_algorithm']}",
        f"Trajectories: {NUM_TRAJECTORIES} ({STEPS_PER_TRAJECTORY} steps each)",
        f"Transitions: {len(dataset['states'])}",
        f"State shape: {dataset['states'].shape}; action shape: {dataset['actions'].shape}",
        f"Episode return mean/best/worst: "
        f"{returns.mean():.6f} / {returns.max():.6f} / {returns.min():.6f}",
        f"Minimum dataset safety margin (pre-action): {dataset['margins'].min():.6f}",
        f"Safety violations (pre-action samples): {np.count_nonzero(dataset['margins'] < 0)}",
        f"Final successor states: min margin={final_margins.min():.6f}; "
        f"violations={np.count_nonzero(final_margins < 0)}",
        f"Dataset: {DATASET_PATH}",
        f"Diagnostics: {RESULTS_DIR}",
    ])
    (RESULTS_DIR / "summary.txt").write_text(summary + "\n", encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
