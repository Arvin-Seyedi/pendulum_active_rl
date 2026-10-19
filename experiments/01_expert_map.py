# experiments/01_expert_map.py

from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# MAKE PROJECT ROOT IMPORTABLE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.expert import pd_controller
from src.safety import safety_margin


# ============================================================
# PARAMETERS
# ============================================================

DT = 0.05
MAX_STEPS = 200

THETA_MIN = -0.8
THETA_MAX = 0.8

OMEGA_MIN = -2.0
OMEGA_MAX = 2.0

GRID_SIZE = 81


# ============================================================
# RUN ONE INITIAL CONDITION
# ============================================================

def run_episode(env, theta0, omega0):

    obs, info = env.reset()

    env.unwrapped.state = np.array([
        theta0,
        omega0
    ])

    obs = np.array([
        np.cos(theta0),
        np.sin(theta0),
        omega0
    ])

    minimum_margin = np.inf
    first_violation_time = np.nan

    for step in range(MAX_STEPS):

        theta = np.arctan2(
            obs[1],
            obs[0]
        )

        omega = obs[2]

        margin = safety_margin(
            theta,
            omega
        )

        minimum_margin = min(
            minimum_margin,
            margin
        )

        if (
            margin < 0
            and np.isnan(first_violation_time)
        ):
            first_violation_time = step * DT

        action = pd_controller(
            theta,
            omega
        )

        obs, reward, terminated, truncated, info = env.step(
            np.array(
                [action],
                dtype=np.float32
            )
        )

        if terminated or truncated:
            break

    return (
        minimum_margin,
        first_violation_time
    )


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    env = gym.make(
        "Pendulum-v1"
    )

    theta_values = np.linspace(
        THETA_MIN,
        THETA_MAX,
        GRID_SIZE
    )

    omega_values = np.linspace(
        OMEGA_MIN,
        OMEGA_MAX,
        GRID_SIZE
    )

    margin_map = np.zeros(
        (
            len(omega_values),
            len(theta_values)
        )
    )

    violation_time_map = np.full(
        (
            len(omega_values),
            len(theta_values)
        ),
        np.nan
    )

    print(
        "Running refined PD expert "
        "initial-condition benchmark..."
    )

    for i, omega0 in enumerate(omega_values):

        for j, theta0 in enumerate(theta_values):

            (
                min_margin,
                first_violation_time
            ) = run_episode(
                env,
                theta0,
                omega0
            )

            margin_map[i, j] = min_margin

            violation_time_map[
                i,
                j
            ] = first_violation_time

    env.close()


    # ========================================================
    # STATISTICS
    # ========================================================

    safe_mask = margin_map >= 0
    unsafe_mask = margin_map < 0

    safe_count = np.sum(
        safe_mask
    )

    unsafe_count = np.sum(
        unsafe_mask
    )

    total_count = margin_map.size

    safe_percentage = (
        100
        * safe_count
        / total_count
    )


    valid_violation_times = (
        violation_time_map[
            ~np.isnan(
                violation_time_map
            )
        ]
    )


    print()
    print(
        "=============================================="
    )

    print(
        "PD EXPERT SAFETY BENCHMARK"
    )

    print(
        "=============================================="
    )

    print(
        f"Grid size: "
        f"{GRID_SIZE} x {GRID_SIZE}"
    )

    print(
        f"Total initial conditions: "
        f"{total_count}"
    )

    print(
        f"Safe trajectories: "
        f"{safe_count}"
    )

    print(
        f"Unsafe trajectories: "
        f"{unsafe_count}"
    )

    print(
        f"Safe percentage: "
        f"{safe_percentage:.2f}%"
    )

    print(
        f"Worst minimum safety margin: "
        f"{np.min(margin_map):.4f}"
    )


    if len(valid_violation_times) > 0:

        print(
            f"Earliest violation: "
            f"{np.min(valid_violation_times):.4f} s"
        )

        print(
            f"Average first violation: "
            f"{np.mean(valid_violation_times):.4f} s"
        )

        print(
            f"Latest first violation: "
            f"{np.max(valid_violation_times):.4f} s"
        )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results_folder = (
        PROJECT_ROOT
        / "results"
        / "expert_safety_map"
    )

    results_folder.mkdir(
        parents=True,
        exist_ok=True
    )


    np.savez_compressed(
        results_folder
        / "expert_safety_map.npz",

        theta_values=theta_values,
        omega_values=omega_values,

        margin_map=margin_map,

        violation_time_map=violation_time_map
    )


    # ========================================================
    # PLOT 1 — SAFETY MAP
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    image = plt.imshow(
        margin_map,

        origin="lower",

        extent=[
            theta_values[0],
            theta_values[-1],

            omega_values[0],
            omega_values[-1]
        ],

        aspect="auto"
    )

    plt.colorbar(
        image,
        label="Minimum Safety Margin"
    )

    plt.contour(
        theta_values,
        omega_values,
        margin_map,

        levels=[0],

        linewidths=2
    )

    plt.xlabel(
        "Initial Angle θ₀ [rad]"
    )

    plt.ylabel(
        "Initial Angular Velocity ω₀ [rad/s]"
    )

    plt.title(
        "PD Expert Safety Map"
    )

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        results_folder
        / "expert_safety_map.png",

        dpi=300
    )

    plt.show()


    # ========================================================
    # PLOT 2 — TIME TO FIRST VIOLATION
    # ========================================================

    plt.figure(
        figsize=(9, 6)
    )

    violation_plot = (
        np.ma.masked_invalid(
            violation_time_map
        )
    )

    image = plt.imshow(
        violation_plot,

        origin="lower",

        extent=[
            theta_values[0],
            theta_values[-1],

            omega_values[0],
            omega_values[-1]
        ],

        aspect="auto"
    )

    plt.colorbar(
        image,
        label="Time to First Violation [s]"
    )

    plt.contour(
        theta_values,
        omega_values,
        margin_map,

        levels=[0],

        linewidths=2
    )

    plt.xlabel(
        "Initial Angle θ₀ [rad]"
    )

    plt.ylabel(
        "Initial Angular Velocity ω₀ [rad/s]"
    )

    plt.title(
        "Time to First Safety Violation"
    )

    plt.grid()

    plt.tight_layout()

    plt.savefig(
        results_folder
        / "time_to_first_violation.png",

        dpi=300
    )

    plt.show()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()