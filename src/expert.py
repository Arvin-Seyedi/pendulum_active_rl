# src/expert.py

import numpy as np


KP = 8.0
KD = 2.0
MAX_TORQUE = 2.0


def pd_controller(theta, omega):
    """
    PD expert controller for Pendulum-v1.

    Desired equilibrium:
        theta = 0
        omega = 0

    Returns:
        clipped torque command
    """

    u = -KP * theta - KD * omega

    return np.clip(
        u,
        -MAX_TORQUE,
        MAX_TORQUE
    )