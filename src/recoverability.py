"""Finite-horizon trajectory feasibility diagnostics, separate from all RL code."""

from pathlib import Path
import time

import casadi as ca
import numpy as np

from src.active_query import reset_at
from src.counterfactual import assert_bitwise_equal, episode_metrics
from src.dataset import get_state
from src.expert import KP, KD, MAX_TORQUE
from src.horizon_experiment import save_npz, write_json
from src.rollout_horizon import MeteredEnvironment
from src.safety import THETA_SAFE, OMEGA_SAFE, safety_margin


def symbolic_dynamics(clipped=False):
    x, u = ca.SX.sym('x', 2), ca.SX.sym('u')
    torque = ca.fmin(2., ca.fmax(-2., u)) if clipped else u
    velocity = x[1] + .05 * (15 * ca.sin(x[0]) + 3 * torque)
    if clipped:
        velocity = ca.fmin(8., ca.fmax(-8., velocity))
    successor = ca.vertcat(x[0] + .05 * velocity, velocity)
    return ca.Function('clipped_step' if clipped else 'smooth_step', [x, u], [successor])


def pack(states, actions):
    """x1(theta,omega), x2,...,xH, then u0,...,u(H-1). x0 is a parameter."""
    return np.r_[np.asarray(states)[1:].reshape(-1), np.asarray(actions).reshape(-1)]


def unpack(vector, initial, horizon):
    vector = np.asarray(vector, dtype=np.float64).reshape(-1)
    assert vector.shape == (3 * horizon,)
    return np.vstack([initial, vector[:2 * horizon].reshape(horizon, 2)]), vector[2 * horizon:]


def shooting_problem(horizon=200):
    initial = ca.SX.sym('initial', 2)
    decision = ca.SX.sym('decision', 3 * horizon)
    successors = ca.reshape(decision[:2 * horizon], 2, horizon)
    actions = decision[2 * horizon:]
    states = ca.horzcat(initial, successors)
    step = symbolic_dynamics()
    residuals = ca.vertcat(*[states[:, t + 1] - step(states[:, t], actions[t]) for t in range(horizon)])
    objective = sum(states[0, t] ** 2 + .1 * states[1, t] ** 2 + .001 * actions[t] ** 2 for t in range(horizon))
    lower = np.r_[np.tile([-THETA_SAFE, -OMEGA_SAFE], horizon), np.full(horizon, -MAX_TORQUE)]
    upper = -lower
    nlp = {'x': decision, 'p': initial, 'f': objective, 'g': residuals}
    check = ca.Function('check_transcription', [decision, initial], [objective, residuals])
    return nlp, lower, upper, check


def candidate_diagnostics(vector, initial, horizon=200):
    states, actions = unpack(vector, initial, horizon)
    _, lower, upper, check = shooting_problem(horizon)
    objective, residuals = check(vector, initial)
    return {
        'objective': float(objective),
        'maximum_dynamics_residual': float(np.max(np.abs(np.array(residuals)))),
        'maximum_bound_violation': float(max(0., np.max(lower - vector), np.max(vector - upper))),
        'predicted_minimum_raw_margin': float(min(safety_margin(*x) for x in states)),
        'predicted_finite': bool(np.isfinite(states).all() and np.isfinite(actions).all()),
        'torque_abs_max': float(np.abs(actions).max()),
    }


class IterateRecorder(ca.Callback):
    """Persist last finite primal iterate, including before watchdog termination."""
    def __init__(self, folder, nx, ng):
        ca.Callback.__init__(self)
        self.folder, self.nx, self.ng = Path(folder), nx, ng
        self.iteration = 0
        self.construct('iteration_recorder', {})

    def get_n_in(self):
        return ca.nlpsol_n_out()

    def get_n_out(self):
        return 1

    def get_name_in(self, i):
        return ca.nlpsol_out(i)

    def get_name_out(self, i):
        return 'ret'

    def get_sparsity_in(self, i):
        name = self.get_name_in(i)
        size = {'x': self.nx, 'f': 1, 'g': self.ng, 'lam_x': self.nx, 'lam_g': self.ng, 'lam_p': 2}[name]
        return ca.Sparsity.dense(size, 1)

    def get_sparsity_out(self, i):
        return ca.Sparsity.scalar()

    def eval(self, args):
        values = {ca.nlpsol_out(i): np.array(arg).reshape(-1) for i, arg in enumerate(args)}
        if np.isfinite(values['x']).all():
            save_npz(self.folder / 'last_iterate.npz', {'decision': values['x'], 'iteration': np.array(self.iteration)})
        with (self.folder / 'iteration_trace.csv').open('a') as stream:
            if self.iteration == 0:
                stream.write('callback_iteration,objective,max_dynamic_residual\n')
            stream.write(f"{self.iteration},{values['f'][0]:.17g},{np.max(np.abs(values['g'])):.17g}\n")
        self.iteration += 1
        return [0]


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serializable(v) for v in value]
    if isinstance(value, (np.ndarray, ca.DM)):
        return serializable(np.array(value).tolist())
    if isinstance(value, np.generic):
        return serializable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def solve_attempt(folder, config, initial, guess):
    """Called only in a separately limited worker process. No simulator calls."""
    folder = Path(folder)
    assert ca.__version__ == config['solver']['casadi_version']
    horizon = config['horizon']
    nlp, lower, upper, _ = shooting_problem(horizon)
    recorder = IterateRecorder(folder, 3 * horizon, 2 * horizon)
    ipopt_options = {k: v for k, v in config['solver'].items() if k != 'casadi_version'}
    ipopt_options.update(print_level=5, print_user_options='yes', print_options_documentation='yes',
                         print_timing_statistics='yes', option_file_name='')
    options = {'ipopt': ipopt_options, 'iteration_callback': recorder, 'error_on_fail': False,
               'record_time': True, 'print_time': True}
    write_json(folder / 'effective_options.json', {'ipopt': ipopt_options,
               'casadi': {k: v for k, v in options.items() if k not in ('ipopt', 'iteration_callback')},
               'unmodified_ipopt_defaults': 'Complete default documentation preserved in solver.log'})
    start = time.perf_counter()
    solver = ca.nlpsol('recoverability_solver', 'ipopt', nlp, options)
    construction_seconds = time.perf_counter() - start
    started = time.perf_counter()
    result = solver(x0=guess, p=initial, lbx=lower, ubx=upper, lbg=0, ubg=0)
    elapsed = time.perf_counter() - started
    decision = np.array(result['x']).reshape(-1)
    save_npz(folder / 'candidate.npz', {'decision': decision})
    stats = serializable(solver.stats())
    write_json(folder / 'solver_result.json', {'casadi_version': ca.__version__, 'stats': stats,
               'solver_seconds': elapsed, 'construction_seconds': construction_seconds,
               'callback_calls': recorder.iteration,
               'diagnostics': candidate_diagnostics(decision, initial, horizon)})


def evaluate_controls(initial, seed, actions, reference_states=None):
    """Record actual float32 torques and raw states; never stop at a violation."""
    actions = np.asarray(actions, dtype=np.float64).reshape(-1)
    horizon = len(actions)
    assert np.isfinite(actions).all()
    if reference_states is not None:
        reference_states = np.asarray(reference_states, dtype=np.float64)
        assert reference_states.shape == (horizon + 1, 2) and np.isfinite(reference_states).all()
    raw = np.empty((1, horizon + 1, 2), dtype=np.float64)
    states = np.empty_like(raw)
    margins = np.empty((1, horizon + 1), dtype=np.float64)
    executed = np.empty((1, horizon, 1), dtype=np.float32)
    rewards = np.empty((1, horizon), dtype=np.float64)
    terminated, truncated = np.zeros((1, horizon), bool), np.zeros((1, horizon), bool)
    env = MeteredEnvironment(horizon)
    started = time.perf_counter()
    try:
        obs = reset_at(env, initial, seed)
        raw[0, 0], states[0, 0] = env.unwrapped.state, get_state(obs)
        np.testing.assert_array_equal(raw[0, 0], initial)
        margins[0, 0] = safety_margin(*raw[0, 0])
        for t in range(horizon):
            command = actions[t]
            if reference_states is not None:
                error = raw[0, t] - reference_states[t]
                command -= KP * error[0] + KD * error[1]
            executed[0, t, 0] = np.clip(command, -MAX_TORQUE, MAX_TORQUE)
            obs, reward, term, trunc, _ = env.step(executed[0, t])
            raw[0, t + 1], states[0, t + 1] = env.unwrapped.state, get_state(obs)
            margins[0, t + 1] = safety_margin(*raw[0, t + 1])
            rewards[0, t] = reward
            terminated[0, t], truncated[0, t] = term, trunc
            if term or (trunc and t != horizon - 1):
                raise RuntimeError('Unexpected simulator termination.')
        costs = {k: v.tolist() for k, v in env.costs().items()}
    finally:
        env.close()
    costs['wall_seconds'] = time.perf_counter() - started
    arrays = dict(initial_conditions=np.asarray(initial, dtype=np.float64)[None].copy(),
                  states=states, raw_states=raw, margins=margins, actions=executed, rewards=rewards,
                  terminated=terminated, truncated=truncated, episode_returns=rewards.sum(axis=1),
                  episode_min_margins=margins.min(axis=1))
    validate_execution(arrays, initial)
    assert costs['step_attempts'] == horizon and costs['reset_attempts'] == 1
    return arrays, costs


def validate_execution(arrays, initial):
    horizon = arrays['rewards'].shape[1]
    assert arrays['raw_states'].shape == (1, horizon + 1, 2)
    assert arrays['actions'].shape == (1, horizon, 1) and arrays['actions'].dtype == np.float32
    np.testing.assert_array_equal(arrays['raw_states'][0, 0], initial)
    assert np.max(np.abs(arrays['actions'])) <= MAX_TORQUE
    assert not arrays['terminated'].any() and not arrays['truncated'][:, :-1].any()
    assert arrays['truncated'][:, -1].all()
    for v in arrays.values():
        assert np.isfinite(v).all()
    expected = np.array([[safety_margin(*state) for state in row] for row in arrays['raw_states']])
    np.testing.assert_array_equal(arrays['margins'], expected)
    np.testing.assert_array_equal(arrays['episode_returns'], arrays['rewards'].sum(axis=1))
    np.testing.assert_array_equal(arrays['episode_min_margins'], expected.min(axis=1))


def replay_and_measure(arrays, initial, seed, config, planned_states=None):
    replay, costs = evaluate_controls(initial, seed, arrays['actions'][0, :, 0])
    assert_bitwise_equal(arrays, replay)
    values = episode_metrics(arrays, config['discount'], config['dt'])
    metrics = {key: value[0].item() for key, value in values.items()}
    if not metrics['violated']:
        metrics['first_crossing_step'] = metrics['first_crossing_seconds'] = None
    metrics.update(replay_bitwise=True, witness=bool(metrics['minimum_margins'] >= 0),
                   saturation_fraction=float(np.mean(np.abs(arrays['actions']) == MAX_TORQUE)),
                   terminal_raw_state=arrays['raw_states'][0, -1].tolist())
    for buffer in config['margin_buffers']:
        metrics[f'margin_at_least_{buffer}'] = bool(metrics['minimum_margins'] >= buffer)
    if planned_states is not None:
        discrepancy = np.abs(arrays['raw_states'][0] - planned_states)
        metrics['maximum_plan_replay_state_discrepancy'] = discrepancy.max(axis=0).tolist()
        metrics['terminal_plan_replay_state_discrepancy'] = discrepancy[-1].tolist()
    return replay, costs, metrics
