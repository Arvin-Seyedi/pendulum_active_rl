"""Trajectory-bootstrap behavior cloning and policy-disagreement utilities.

Training targets are recorded, bounded actions. Linear network outputs are
unbounded; clip each member before computing the bounded ensemble mean/std.
Out-of-bag losses are member-specific model-selection diagnostics, not an
independent test of the full ensemble.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn

from src.expert import MAX_TORQUE
from src.safety import OMEGA_SAFE, THETA_SAFE


@dataclass(frozen=True)
class EnsembleConfig:
    members: int = 5
    hidden_sizes: tuple = (64, 64)
    batch_size: int = 256
    learning_rate: float = 1e-3
    max_epochs: int = 300
    patience: int = 30
    seed: int = 42


class PolicyNetwork(nn.Module):
    """Predict normalized torque from physically scaled [theta, omega]."""

    def __init__(self, hidden_sizes=(64, 64)):
        super().__init__()
        self.hidden_sizes = tuple(hidden_sizes)
        self.register_buffer("input_scale", torch.tensor([THETA_SAFE, OMEGA_SAFE]))
        self.register_buffer("torque_scale", torch.tensor(float(MAX_TORQUE)))
        layers = []
        width = 2
        for hidden in hidden_sizes:
            layers.extend([nn.Linear(width, hidden), nn.ReLU()])
            width = hidden
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, states):
        return self.network(states / self.input_scale)


def configure_determinism(seed):
    """Use deterministic, single-thread CPU execution in this environment."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def member_seeds(config):
    """Independent, recorded initialization/bootstrap/shuffle RNG streams."""
    seeds = []
    for child in np.random.SeedSequence(config.seed).spawn(config.members):
        streams = child.spawn(3)
        seeds.append(dict(zip(
            ("initialization", "bootstrap", "shuffle"),
            [int(stream.generate_state(1)[0]) for stream in streams],
        )))
    return seeds


def bootstrap_trajectories(num_trajectories, seed):
    """Draw N complete trajectories with replacement, retaining duplicates."""
    if num_trajectories < 2:
        raise ValueError("Out-of-bag validation requires multiple trajectories.")
    ids = np.random.default_rng(seed).integers(
        0, num_trajectories, size=num_trajectories
    )
    oob_ids = np.setdiff1d(np.arange(num_trajectories), np.unique(ids))
    if not len(oob_ids):
        raise ValueError("Bootstrap has no out-of-bag trajectories.")
    return ids, oob_ids


def fit_member(states, actions, config, seeds, progress=None):
    """Fit one network; inputs have shapes (episodes, steps, 2/1).

Use a single fixed trajectory bootstrap throughout training. Uniformly shuffle
all its transitions each epoch. Select the checkpoint using raw normalized
action MSE on omitted whole trajectories; no grid expert labels enter training.
"""
    if (states.ndim != 3 or states.shape[-1] != 2
            or actions.shape != (*states.shape[:2], 1)):
        raise ValueError("Expected trajectory-shaped states/actions.")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Training data contain nonfinite values.")
    bootstrap_ids, oob_ids = bootstrap_trajectories(len(states), seeds["bootstrap"])
    torch.manual_seed(seeds["initialization"])
    model = PolicyNetwork(config.hidden_sizes).cpu()
    train_x = torch.as_tensor(states[bootstrap_ids].reshape(-1, 2), dtype=torch.float32)
    train_y = torch.as_tensor(actions[bootstrap_ids].reshape(-1, 1), dtype=torch.float32)
    val_x = torch.as_tensor(states[oob_ids].reshape(-1, 2), dtype=torch.float32)
    val_y = torch.as_tensor(actions[oob_ids].reshape(-1, 1), dtype=torch.float32)
    train_y = train_y / model.torque_scale
    val_y = val_y / model.torque_scale
    generator = torch.Generator(device="cpu").manual_seed(seeds["shuffle"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    history = {"epoch": [], "train_mse": [], "oob_mse": []}
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        for indices in order.split(config.batch_size):
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(train_x[indices]), train_y[indices])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss.")
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss = float(loss_fn(model(train_x), train_y))
            val_loss = float(loss_fn(model(val_x), val_y))
        if not np.isfinite([train_loss, val_loss]).all():
            raise RuntimeError("Nonfinite epoch loss.")
        history["epoch"].append(epoch)
        history["train_mse"].append(train_loss)
        history["oob_mse"].append(val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
        if progress and (epoch == 1 or epoch % 50 == 0):
            progress(f"epoch {epoch}: train MSE={train_loss:.6g}, OOB MSE={val_loss:.6g}")
        if stale_epochs >= config.patience:
            break

    if best_state is None:
        raise RuntimeError("No valid training checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    record = {
        "seeds": seeds,
        "bootstrap_ids": bootstrap_ids.tolist(),
        "oob_ids": oob_ids.tolist(),
        "unique_training_trajectories": len(np.unique(bootstrap_ids)),
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_oob_normalized_mse": best_loss,
        "history": history,
    }
    return model, record


def predict_ensemble(models, states, batch_size=4096):
    """Return float64 arrays: member outputs (M,N), means/stds (N).

Primary action = mean(clip(raw member torque)), never clip(mean(raw)).
Disagreement uses population standard deviation (ddof=0), in N m.
"""
    states = np.asarray(states)
    if states.ndim != 2 or states.shape[1] != 2 or not len(states) or not models:
        raise ValueError("Expected nonempty (N,2) states and at least one model.")
    if not np.isfinite(states).all():
        raise ValueError("Prediction states contain nonfinite values.")
    raw = []
    for model in models:
        model.eval()
        chunks = []
        with torch.no_grad():
            for start in range(0, len(states), batch_size):
                x = torch.as_tensor(states[start:start + batch_size], dtype=torch.float32)
                chunks.append((model(x) * model.torque_scale).numpy().ravel())
        raw.append(np.concatenate(chunks))
    raw = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(raw).all():
        raise RuntimeError("Nonfinite policy predictions.")
    bounded = np.clip(raw, -MAX_TORQUE, MAX_TORQUE)
    return {
        "raw_predictions": raw,
        "bounded_predictions": bounded,
        "raw_mean": raw.mean(axis=0),
        "bounded_mean": bounded.mean(axis=0),
        "raw_uncertainty": raw.std(axis=0, ddof=0),
        "uncertainty": bounded.std(axis=0, ddof=0),
    }


def save_member(path, model, config, record):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "config": asdict(config),
        "record": record,
    }, path)


def load_member(path):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    model = PolicyNetwork(payload["config"]["hidden_sizes"])
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, payload
