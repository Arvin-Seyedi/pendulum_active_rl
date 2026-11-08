"""Frozen E12 technical pilot. Calls native upstream controllers without repairs.

Physics instrumentation only reads PyBullet after its original stepSimulation.
It never updates the environment's cached state or modifies motor commands.
"""
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import signal
import subprocess
import sys
import time

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def activate(upstream, config):
    upstream = Path(upstream).resolve()
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=upstream, text=True).strip()
    if commit != config['upstream_commit']:
        raise AssertionError('Wrong upstream commit')
    if subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=upstream):
        raise AssertionError('Upstream tracked sources changed')
    sys.path.insert(0, str(upstream))
    import torch
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])


def make_environment(upstream, config, initial=None, physics_frequency=1000):
    import yaml
    from safe_control_gym.utils.registration import make, registry
    task = yaml.safe_load((Path(upstream) / config['task_config']).read_text())['task_config']
    defaults = registry.specs['quadrotor'].get_config()
    defaults.update(deepcopy(task))
    defaults.update(randomized_init=False, randomized_inertial_prop=False,
                    init_state=initial, pyb_freq=physics_frequency,
                    normalized_rl_action_space=False, disturbances=None,
                    adversary_disturbance=None, seed=config['seed'])
    return make('quadrotor', **defaults)


def raw_state(env):
    import pybullet as p
    pos, quat = p.getBasePositionAndOrientation(env.DRONE_ID, physicsClientId=env.PYB_CLIENT)
    vel, ang = p.getBaseVelocity(env.DRONE_ID, physicsClientId=env.PYB_CLIENT)
    euler = p.getEulerFromQuaternion(quat)
    return np.array([pos[0], vel[0], pos[2], vel[2], euler[1], ang[1]], dtype=np.float64)


@contextmanager
def capture_physics(env):
    """Queries only; preserve native function arguments, return and caches."""
    import pybullet as p
    original = p.stepSimulation
    data = {'states': [raw_state(env)], 'contacts': [len(p.getContactPoints(bodyA=env.DRONE_ID, physicsClientId=env.PYB_CLIENT))],
            'integration_seconds': [], 'capture_seconds': []}

    def instrument(*args, **kwargs):
        start = time.perf_counter()
        result = original(*args, **kwargs)
        end = time.perf_counter()
        client = kwargs.get('physicsClientId', args[0] if args else 0)
        if client == env.PYB_CLIENT:
            data['integration_seconds'].append(end - start)
            data['states'].append(raw_state(env))
            data['contacts'].append(len(p.getContactPoints(bodyA=env.DRONE_ID, physicsClientId=client)))
            data['capture_seconds'].append(time.perf_counter() - end)
        return result
    p.stepSimulation = instrument
    try:
        yield data
    finally:
        p.stepSimulation = original


def freeze(root, upstream, out, config):
    import yaml
    out = Path(out)
    if (out / 'frozen.json').exists():
        return read(out / 'frozen.json')
    activate(upstream, config)
    env = make_environment(upstream, config)
    reference = env.X_GOAL.copy()
    cases = {k: reference[0].tolist() if isinstance(v, str) else v for k, v in config['cases'].items()}
    frozen = {'config': config, 'cases': cases,
              'initial_binary64_hex': {k: [float(v).hex() for v in x] for k, x in cases.items()},
              'state_labels': env.STATE_LABELS, 'state_units': env.STATE_UNITS,
              'action_labels': env.ACTION_LABELS, 'action_units': env.ACTION_UNITS,
              'state_low': env.state_space.low.tolist(), 'state_high': env.state_space.high.tolist(),
              'action_low': env.physical_action_bounds[0].tolist(), 'action_high': env.physical_action_bounds[1].tolist(),
              'Q': env.Q.tolist(), 'R': env.R.tolist(), 'hover_input': env.U_GOAL.tolist(),
              'mass': env.MASS, 'Iyy': env.J[1, 1], 'arm_length': env.L, 'gravity': env.GRAVITY_ACC,
              'KF': env.KF, 'PWM2RPM_SCALE': env.PWM2RPM_SCALE, 'PWM2RPM_CONST': env.PWM2RPM_CONST,
              'MIN_PWM': env.MIN_PWM, 'MAX_PWM': env.MAX_PWM,
              'native_task_yaml': yaml.safe_load((Path(upstream) / config['task_config']).read_text()),
              'python': sys.version, 'executable': sys.executable, 'platform': platform.platform(),
              'packages': dict(sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()))}
    frozen['upstream_files'] = {str(p.relative_to(upstream)): sha(p) for p in Path(upstream).rglob('*')
                                if p.is_file() and '.git' not in p.parts and '__pycache__' not in p.parts}
    np.save(out / 'reference.npy', reference)
    frozen['reference_sha256'] = sha(out / 'reference.npy')
    write(out / 'frozen.json', frozen)
    (out / 'environment.lock.txt').write_text('\n'.join(k + '==' + v for k, v in frozen['packages'].items()) + '\n')
    archive = subprocess.check_output(['git', 'archive', '--format=tar.gz', config['upstream_commit']], cwd=upstream)
    (out / 'upstream.tar.gz').write_bytes(archive)
    env.close()
    return frozen


def make_controller(upstream, config, env_factory, key):
    import yaml
    from safe_control_gym.utils.registration import make, registry
    algo, horizon = config['controllers'][key]
    path = config['linear_config'] if algo == 'linear_mpc' else config['nonlinear_config']
    args = registry.specs[algo].get_config()
    args.update(yaml.safe_load((Path(upstream) / path).read_text())['algo_config'])
    args.update(horizon=horizon, training=False, seed=config['seed'], use_gpu=False)
    ctrl = make(algo, env_factory, **args)
    ctrl.reset()
    return ctrl


def timeout_handler(signum, frame):
    raise TimeoutError('Approved one-second action-selection cap exceeded')


def run_episode(root, upstream, out, case, controller, phase, frequency=1000):
    """A completed run is immutable. Failures retain their partial histories."""
    out = Path(out); frozen = read(out / 'frozen.json'); config = frozen['config']
    folder = out / 'runs' / f'{case}_{controller}_{frequency}_{phase}'
    if (folder / 'completion.json').exists():
        return read(folder / 'completion.json')
    if folder.exists():
        raise RuntimeError('Partial run exists; inspect before resuming: ' + str(folder))
    folder.mkdir(parents=True)
    initial = np.array(frozen['cases'][case], dtype=float)
    activate(upstream, config)
    factory = lambda: make_environment(upstream, config, initial, frequency)
    env = factory(); ctrl = None
    wall = time.perf_counter(); cpu = time.process_time()
    arrays = {k: [] for k in ['states', 'actions', 'clipped_actions', 'rpms', 'rewards', 'dones',
               'solver_wall', 'solver_cpu', 'solver_iterations', 'solver_success', 'current_steps',
               'warm_u_error', 'warm_x_error', 'predicted_states', 'predicted_inputs', 'native_model_predictions']}
    statuses = []; failure = None; physics = None; reset_delta = None
    try:
        ctrl = make_controller(upstream, config, factory, controller)
        from safe_control_gym.controllers.mpc.mpc_utils import rk_discrete
        native_rk = rk_discrete(env.symbolic.fc_func, 6, 2, env.CTRL_TIMESTEP)
        obs, info = env.reset()
        reset_delta = float(np.max(np.abs(raw_state(env) - initial)))
        if reset_delta > 1e-14:
            raise AssertionError('Reset mismatch')
        arrays['states'].append(raw_state(env))
        with capture_physics(env) as physics:
            for step in range(config['control_transitions']):
                previous_u = None if ctrl.u_prev is None else ctrl.u_prev.copy()
                previous_x = None if ctrl.x_prev is None else ctrl.x_prev.copy()
                arrays['current_steps'].append(info['current_step'])
                start = time.perf_counter(); start_cpu = time.process_time()
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.setitimer(signal.ITIMER_REAL, config['solver_timeout_seconds'])
                try:
                    action = np.asarray(ctrl.select_action(obs, info)).copy()
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                elapsed = time.perf_counter() - start
                stats = ctrl.opti_dict['opti'].stats()
                statuses.append(str(stats.get('return_status')))
                arrays['solver_wall'].append(elapsed)
                arrays['solver_cpu'].append(time.process_time() - start_cpu)
                arrays['solver_iterations'].append(stats.get('iter_count', -1))
                arrays['solver_success'].append(bool(stats.get('success', False)))
                if not stats.get('success', False) or not np.isfinite(action).all():
                    failure = {'kind': 'solver_failure', 'step': step, 'status': statuses[-1], 'native_returned_action': action.tolist()}
                    break
                opti = ctrl.opti_dict['opti']
                if previous_u is not None:
                    expected_u = previous_u.copy(); expected_x = previous_x.copy()
                    if controller != 'linear40':
                        expected_u[:-1] = expected_u[1:]
                        expected_x[:, :-1] = expected_x[:, 1:]
                    initial_u = np.array(opti.debug.value(ctrl.opti_dict['u_var'], opti.initial()))
                    initial_x = np.array(opti.debug.value(ctrl.opti_dict['x_var'], opti.initial()))
                    arrays['warm_u_error'].append(float(np.max(np.abs(initial_u - expected_u))))
                    arrays['warm_x_error'].append(float(np.max(np.abs(initial_x - expected_x))))
                else:
                    arrays['warm_u_error'].append(0.); arrays['warm_x_error'].append(0.)
                arrays['predicted_states'].append(np.array(ctrl.results_dict['horizon_states'][-1]))
                arrays['predicted_inputs'].append(np.array(ctrl.results_dict['horizon_inputs'][-1]))
                arrays['actions'].append(action.copy())
                clipped = np.clip(action, env.physical_action_bounds[0], env.physical_action_bounds[1])
                arrays['native_model_predictions'].append(np.array(native_rk(x0=raw_state(env), p=clipped)['xf']).reshape(-1))
                obs, reward, done, info = env.step(action)
                arrays['clipped_actions'].append(env.current_clipped_action.copy())
                arrays['rpms'].append(env.last_clipped_action.copy())
                arrays['states'].append(raw_state(env))
                arrays['rewards'].append(reward); arrays['dones'].append(done)
                if done and step != config['control_transitions'] - 1:
                    failure = {'kind': 'early_native_termination', 'step': step + 1}
                    break
    except Exception as exc:
        failure = {'kind': type(exc).__name__, 'message': str(exc), 'applied_transitions': len(arrays['rewards'])}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        native_stats = None
        if ctrl is not None:
            try:
                st = ctrl.opti_dict['opti'].stats()
                native_stats = {k: st.get(k) for k in ['return_status', 'success', 'iter_count', 't_wall_total']}
            except Exception:
                pass
            ctrl.close()
        env.close()
    data = {k: np.asarray(v) for k, v in arrays.items()}
    if physics:
        data.update({f'physics_{k}': np.asarray(v) for k, v in physics.items()})
    np.savez_compressed(folder / 'trajectory.npz', **data)
    complete = failure is None and len(arrays['rewards']) == config['control_transitions']
    summary = {'case': case, 'controller': controller, 'phase': phase, 'frequency': frequency,
               'complete': complete, 'failure': failure, 'reset_maximum_absolute_error': reset_delta,
               'wall_seconds': time.perf_counter() - wall, 'cpu_seconds': time.process_time() - cpu,
               'peak_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               'transitions': len(arrays['rewards']), 'solver_calls': len(arrays['current_steps']),
               'solver_statuses': statuses, 'last_native_solver_stats': native_stats,
               'trajectory_sha256': sha(folder / 'trajectory.npz')}
    write(folder / 'completion.json', summary)
    return summary


def replay(upstream, out, folder, instrument):
    """New independent world, original recorded actions, no controller calls."""
    out, folder = Path(out), Path(folder)
    frozen = read(out / 'frozen.json'); config = frozen['config']; metadata = read(folder / 'completion.json')
    activate(upstream, config)
    saved = np.load(folder / 'trajectory.npz')
    env = make_environment(upstream, config, np.array(frozen['cases'][metadata['case']]), metadata['frequency'])
    states = []; rewards = []; physics = None
    try:
        env.reset(); states.append(raw_state(env))
        def execute():
            for action in saved['actions'][:metadata['transitions']]:
                _, reward, _, _ = env.step(action)
                states.append(raw_state(env)); rewards.append(reward)
        if instrument:
            with capture_physics(env) as physics:
                execute()
        else:
            execute()
        a = np.array(states); b = saved['states']
        result = {'control_states_bitwise': a.tobytes() == b.tobytes(),
                  'rewards_bitwise': np.asarray(rewards).tobytes() == saved['rewards'].tobytes(),
                  'maximum_control_state_error': float(np.max(np.abs(a - b))),
                  'instrumented': instrument, 'steps': len(rewards)}
        if instrument:
            result['physics_states_bitwise'] = np.asarray(physics['states']).tobytes() == saved['physics_states'].tobytes()
            result['contacts_bitwise'] = np.asarray(physics['contacts']).tobytes() == saved['physics_contacts'].tobytes()
        write(folder / ('replay_instrumented.json' if instrument else 'replay_native.json'), result)
        return result
    finally:
        env.close()
