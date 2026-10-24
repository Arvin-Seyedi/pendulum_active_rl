"""Variable-length trajectory bootstrap using the verified ensemble model.

This extends fixed-length BC to flat transitions plus episode offsets, without
padding, dropping short rollouts, or changing the verified fixed-length trainer.
"""

import numpy as np
import torch
from torch import nn

from src.ensemble import PolicyNetwork, bootstrap_trajectories


def validate_ragged(states, actions, offsets):
    states, actions, offsets = np.asarray(states), np.asarray(actions), np.asarray(offsets)
    if states.shape != (len(states), 2) or actions.shape != (len(states), 1):
        raise ValueError("Expected flat (N,2) states and (N,1) actions.")
    if (offsets.ndim != 1 or not np.issubdtype(offsets.dtype, np.integer)
            or len(offsets) < 3 or offsets[0] != 0 or offsets[-1] != len(states)
            or np.any(np.diff(offsets) <= 0)):
        raise ValueError("Expected at least two nonempty episodes and complete offsets.")
    if not np.isfinite(states).all() or not np.isfinite(actions).all():
        raise ValueError("Nonfinite training data.")


def episode_transition_indices(offsets, episode_ids):
    return np.concatenate([np.arange(offsets[i], offsets[i + 1]) for i in episode_ids])


def combine_datasets(d0, additional=None):
    n, length = int(d0["num_trajectories"]), int(d0["steps_per_trajectory"])
    states, actions = d0["states"], d0["actions"]
    offsets = np.arange(n + 1, dtype=np.int64) * length
    if additional is not None:
        offsets = np.r_[offsets, additional["episode_offsets"][1:] + len(states)]
        states = np.concatenate([states, additional["states"]])
        actions = np.concatenate([actions, additional["actions"]])
    validate_ragged(states, actions, offsets)
    return states, actions, offsets


def fit_ragged_member(states, actions, offsets, config, seeds, progress=None):
    """Same optimizer/loss/early stopping as fit_member; whole ragged episodes.

    Bootstrap N episode IDs with replacement. Every real transition of each
    drawn episode is retained (including multiplicities). OOB is episode-disjoint.
    Both optimization and checkpoint selection use transition-mean raw normalized
    action MSE; long episodes therefore receive proportionally more weight.
    """
    validate_ragged(states, actions, offsets)
    bootstrap_ids, oob_ids = bootstrap_trajectories(len(offsets) - 1, seeds["bootstrap"])
    train_ids = episode_transition_indices(offsets, bootstrap_ids)
    val_ids = episode_transition_indices(offsets, oob_ids)
    torch.manual_seed(seeds["initialization"])
    model = PolicyNetwork(config.hidden_sizes).cpu()
    train_x = torch.as_tensor(states[train_ids], dtype=torch.float32)
    train_y = torch.as_tensor(actions[train_ids], dtype=torch.float32) / model.torque_scale
    val_x = torch.as_tensor(states[val_ids], dtype=torch.float32)
    val_y = torch.as_tensor(actions[val_ids], dtype=torch.float32) / model.torque_scale
    generator = torch.Generator(device="cpu").manual_seed(seeds["shuffle"])
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.MSELoss()
    history = {"epoch": [], "train_mse": [], "oob_mse": []}
    best_loss, best_state, best_epoch, stale = float("inf"), None, 0, 0
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
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
        if progress and (epoch == 1 or epoch % 50 == 0):
            progress(f"epoch {epoch}: train MSE={train_loss:.6g}, OOB MSE={val_loss:.6g}")
        if stale >= config.patience:
            break
    if best_state is None:
        raise RuntimeError("No valid training checkpoint.")
    model.load_state_dict(best_state)
    model.eval()
    record = {
        "seeds": seeds, "bootstrap_ids": bootstrap_ids.tolist(), "oob_ids": oob_ids.tolist(),
        "unique_training_trajectories": len(np.unique(bootstrap_ids)),
        "best_epoch": best_epoch, "epochs_run": epoch,
        "best_oob_normalized_mse": best_loss, "history": history,
        "training_transitions_with_multiplicity": len(train_ids),
        "oob_transitions": len(val_ids), "dataset_episode_lengths": np.diff(offsets).tolist(),
    }
    return model, record
