"""Exact Experiment 01 provenance and observation-faithful E09 gain evaluation.

The archived classification is never replaced by a new raw-state measurement.
No interval mathematics, optimizer, or prior experiment is modified here.
"""

from fractions import Fraction
from pathlib import Path
import time

import numpy as np

from src.active_query import reset_at
from src.dataset import get_state
from src.expert import MAX_TORQUE, pd_controller
from src.horizon_experiment import sha256
from src.recoverability import validate_execution
from src.rollout_horizon import MeteredEnvironment
from src.safety import safety_margin


PILOT_IDS = [3280, 0, 6560, 767, 44]
GAINS = tuple((kp, kd) for kp in (6.0, 8.0, 12.0) for kd in (1.0, 2.0, 4.0))
BENCHMARK_SHA256 = '2234e7c2c1d5e09595ed7f924286aa824748e747e1a4facc8be435c5cd6c622a'
HORIZON = 200
DT = 0.05


def load_benchmark(path):
    """Read the exact reviewed archive; never regenerate its coordinate axes."""
    path = Path(path)
    digest = sha256(path)
    if digest != BENCHMARK_SHA256:
        raise ValueError('Experiment 01 archive does not match the approved SHA-256.')
    names = {'theta_values', 'omega_values', 'margin_map', 'violation_time_map'}
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != names:
            raise ValueError('Unexpected Experiment 01 archive schema.')
        arrays = {name: archive[name].copy() for name in sorted(names)}
    for name, array in arrays.items():
        expected = (81,) if name.endswith('_values') else (81, 81)
        if array.dtype != np.float64 or array.shape != expected:
            raise ValueError('Unexpected shape or dtype: ' + name)
        array.setflags(write=False)
    assert np.isfinite(arrays['theta_values']).all()
    assert np.isfinite(arrays['omega_values']).all()
    assert np.isfinite(arrays['margin_map']).all()
    assert np.all(np.diff(arrays['theta_values']) > 0)
    assert np.all(np.diff(arrays['omega_values']) > 0)
    assert np.array_equal(arrays['theta_values'][[0, -1]], [-.8, .8])
    assert np.array_equal(arrays['omega_values'][[0, -1]], [-2., 2.])
    unsafe = arrays['margin_map'] < 0
    assert int(unsafe.sum()) == 3230 and int((~unsafe).sum()) == 3331
    times = arrays['violation_time_map']
    assert np.isfinite(times[unsafe]).all() and np.isnan(times[~unsafe]).all()
    return {'arrays': arrays, 'metadata': {
        'source': str(path), 'sha256': digest, 'grid_shape': [81, 81],
        'grid_indexing': 'grid_id = 81 * omega_index + theta_index (C order)',
        'state_count': 6561, 'archived_unsafe_count': 3230, 'archived_safe_count': 3331,
        'dtypes': {name: str(array.dtype) for name, array in arrays.items()},
        'stored_arrays_read_only': True,
        'archived_safety_convention': 'Wrapped observation-derived pre-action states at t=0..199; final successor absent.',
        'original_trajectories_available': False,
        'theta_bitwise_reflection_symmetric': bool(arrays['theta_values'].tobytes() == (-arrays['theta_values'][::-1]).tobytes()),
        'omega_bitwise_reflection_symmetric': bool(arrays['omega_values'].tobytes() == (-arrays['omega_values'][::-1]).tobytes()),
    }}


def case_identity(benchmark, grid_id):
    """Identity and historical label of one actual archived binary grid point."""
    if isinstance(grid_id, (bool, np.bool_)) or not isinstance(grid_id, (int, np.integer)) or not 0 <= grid_id < 6561:
        raise ValueError('Expected an integer Experiment 01 grid ID in [0,6560].')
    grid_id = int(grid_id)
    i, j = divmod(grid_id, 81)
    source = benchmark['arrays']
    initial = np.array([source['theta_values'][j], source['omega_values'][i]], dtype=np.float64)
    unsafe = bool(source['margin_map'][i, j] < 0)
    eq = bool(np.all(initial == 0))
    nominal = bool(abs(initial[0]) <= .25 and abs(initial[1]) <= .5)
    time_value = source['violation_time_map'][i, j]
    return {
        'grid_id': grid_id, 'state_id': grid_id, 'omega_index': i, 'theta_index': j,
        'initial': initial.tolist(), 'initial_hex': [float(v).hex() for v in initial],
        'initial_ratios': [[str(v) for v in Fraction.from_float(float(x)).as_integer_ratio()] for x in initial],
        'initial_raw_bytes_hex': initial.tobytes().hex(),
        'archived_unsafe': unsafe,
        'archived_minimum_margin': float(source['margin_map'][i, j]),
        'archived_first_crossing_seconds': None if np.isnan(time_value) else float(time_value),
        'group': 'equilibrium' if eq else 'PD_unsafe' if unsafe else 'nominal_safe' if nominal else 'challenging_safe',
        'benchmark_sha256': benchmark['metadata']['sha256'],
    }


def frozen_gain_action(theta, omega, kp, kd):
    """Preserve E01 NumPy scalar arithmetic before the final float32 torque cast."""
    pair = (float(kp), float(kd))
    if pair not in GAINS:
        raise ValueError('Only the nine preregistered gain combinations are permitted.')
    if pair == (8.0, 2.0):
        return pd_controller(theta, omega)
    return np.clip(-pair[0] * theta - pair[1] * omega, -MAX_TORQUE, MAX_TORQUE)


def _crossing(margins):
    indices = np.flatnonzero(np.asarray(margins) < 0)
    step = int(indices[0]) if len(indices) else None
    return step, None if step is None else step * DT


def evaluate_gain(initial, seed, kp, kd, *, env_factory=MeteredEnvironment):
    """200 steps with native observation scalars; state storage never feeds PD.

The returned arrays match the existing recorded-action replay schema. The optional
factory is a test seam; production uses the unchanged metered Gymnasium wrapper.
"""
    if (float(kp), float(kd)) not in GAINS:
        raise ValueError('Undeclared gain pair.')
    initial = np.asarray(initial, dtype=np.float64)
    if initial.shape != (2,) or not np.isfinite(initial).all():
        raise ValueError('Expected two finite initial coordinates.')
    raw = np.empty((1, HORIZON + 1, 2), dtype=np.float64)
    states = np.empty_like(raw)
    margins = np.empty((1, HORIZON + 1), dtype=np.float64)
    actions = np.empty((1, HORIZON, 1), dtype=np.float32)
    rewards = np.empty((1, HORIZON), dtype=np.float64)
    terminated = np.zeros((1, HORIZON), dtype=bool)
    truncated = np.zeros_like(terminated)
    legacy_margins = np.empty(HORIZON, dtype=np.float64)
    observation_dtypes, input_dtypes, command_dtypes = [], [], []
    env = env_factory(HORIZON)
    started = time.perf_counter()
    try:
        obs = reset_at(env, initial, seed)
        if env.unwrapped.state.dtype != np.float64 or env.unwrapped.state.tobytes() != initial.tobytes():
            raise AssertionError('Initial physical state changed during reset.')
        raw[0, 0], states[0, 0] = env.unwrapped.state, get_state(obs)
        margins[0, 0] = safety_margin(*raw[0, 0])
        observation_dtypes.append(str(obs.dtype))
        for t in range(HORIZON):
            # This is exactly E01's observation path. Do not pass states[0,t]
            # to the controller: that storage array promotes later float32 data.
            theta, omega = get_state(obs)
            input_dtypes.append([str(np.asarray(theta).dtype), str(np.asarray(omega).dtype)])
            legacy_margins[t] = safety_margin(theta, omega)
            command = frozen_gain_action(theta, omega, kp, kd)
            command_dtypes.append(str(np.asarray(command).dtype))
            actions[0, t] = np.array([command], dtype=np.float32)
            obs, reward, term, trunc, _ = env.step(actions[0, t])
            raw[0, t + 1], states[0, t + 1] = env.unwrapped.state, get_state(obs)
            margins[0, t + 1] = safety_margin(*raw[0, t + 1])
            rewards[0, t] = reward
            terminated[0, t], truncated[0, t] = term, trunc
            observation_dtypes.append(str(obs.dtype))
            if term or (trunc and t != HORIZON - 1):
                raise RuntimeError('Unexpected simulator termination; require all 200 steps.')
        costs = {key: value.tolist() for key, value in env.costs().items()}
    finally:
        env.close()
    costs['wall_seconds'] = time.perf_counter() - started
    result = dict(initial_conditions=initial[None].copy(), states=states, raw_states=raw,
                  margins=margins, actions=actions, rewards=rewards, terminated=terminated,
                  truncated=truncated, episode_returns=rewards.sum(axis=1),
                  episode_min_margins=margins.min(axis=1))
    validate_execution(result, initial)
    assert costs['step_attempts'] == HORIZON and costs['reset_attempts'] == 1
    # Independently reconstruct the command from saved observations' scalar types.
    regenerated = np.empty_like(actions)
    for t in range(HORIZON):
        scalar = np.float64 if t == 0 else np.float32
        theta, omega = (scalar(v) for v in states[0, t])
        regenerated[0, t, 0] = frozen_gain_action(theta, omega, kp, kd)
    assert regenerated.tobytes() == actions.tobytes(), 'Observation dtype path failed command reconstruction.'
    legacy_step, legacy_seconds = _crossing(legacy_margins)
    raw_step, raw_seconds = _crossing(margins[0])
    diagnostics = {
        'gains': [float(kp), float(kd)], 'baseline_original_expert': (kp, kd) == (8.0, 2.0),
        'legacy_minimum_margin': float(legacy_margins.min()),
        'legacy_violated': legacy_step is not None,
        'legacy_first_crossing_step': legacy_step,
        'legacy_first_crossing_seconds': legacy_seconds,
        'legacy_pre_action_margins': legacy_margins.tolist(),
        'raw_violated': raw_step is not None, 'raw_first_crossing_step': raw_step,
        'raw_first_crossing_seconds': raw_seconds,
        'raw_minimum_margin': float(margins.min()),
        'observation_dtypes': observation_dtypes, 'controller_input_dtypes': input_dtypes,
        'command_dtypes': command_dtypes, 'executed_action_dtype': str(actions.dtype),
        'action_regeneration_bitwise': True, 'all_201_raw_states_checked': True,
        'initial_reset_bitwise': True,
        'policy_input_convention': 'E01 float64 reconstructed initial observation then native Gym float32 observations; no float64 state-storage feedback.',
    }
    return result, costs, diagnostics


def compare_archive(identity, arrays, diagnostics):
    """Record all disagreements; never repair or overwrite the archived label."""
    if not diagnostics['baseline_original_expert']:
        raise ValueError('Archive comparison is only meaningful for the original (8,2) expert.')
    initial = np.asarray(identity['initial'], dtype=np.float64)
    assert arrays['initial_conditions'][0].tobytes() == initial.tobytes()
    assert initial.tobytes().hex() == identity['initial_raw_bytes_hex']
    archived = identity['archived_unsafe']
    legacy = diagnostics['legacy_violated']
    raw = diagnostics['raw_violated']
    old_time = identity['archived_first_crossing_seconds']
    new_time = diagnostics['legacy_first_crossing_seconds']
    time_equal = old_time == new_time
    return {
        'grid_id': identity['grid_id'], 'archived_unsafe': archived,
        'remeasured_legacy_unsafe': legacy, 'remeasured_raw_unsafe': raw,
        'archived_vs_legacy_label_disagreement': archived != legacy,
        'archived_vs_raw_label_disagreement': archived != raw,
        'legacy_vs_raw_label_disagreement': legacy != raw,
        'archived_first_crossing_seconds': old_time,
        'remeasured_legacy_first_crossing_seconds': new_time,
        'remeasured_raw_first_crossing_seconds': diagnostics['raw_first_crossing_seconds'],
        'archived_vs_legacy_crossing_disagreement': not time_equal,
        'legacy_crossing_time_difference': None if old_time is None or new_time is None else new_time-old_time,
        'archived_minimum_margin': identity['archived_minimum_margin'],
        'remeasured_legacy_minimum_margin': diagnostics['legacy_minimum_margin'],
        'remeasured_raw_minimum_margin': diagnostics['raw_minimum_margin'],
        'legacy_margin_difference': diagnostics['legacy_minimum_margin']-identity['archived_minimum_margin'],
        'archive_has_no_recorded_actions_or_trajectories': True,
        'historical_trajectory_bitwise_reproduction_claimed': False,
    }
