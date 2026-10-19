# src/dataset.py

from pathlib import Path
import numpy as np


def get_state(obs):
    """
    Convert Pendulum-v1 observation

        [cos(theta), sin(theta), omega]

    into

        [theta, omega]
    """

    theta = np.arctan2(
        obs[1],
        obs[0]
    )

    omega = obs[2]

    return theta, omega


def load_dataset(path):
    """
    Load saved offline dataset from .npz file.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return np.load(
        path,
        allow_pickle=False
    )