"""Independent model checks and frozen case selection for Experiment 07."""

import numpy as np
import casadi as ca

from src.active_query import reset_at
from src.counterfactual import fixed_state_menu, assert_bitwise_equal
from src.horizon_experiment import load_npz, read_json
from src.recoverability import pack, shooting_problem, symbolic_dynamics
from src.rollout_horizon import MeteredEnvironment


def select_cases(root, config):
    source = root / config['source_experiment']
    menu = load_npz(source / 'state_action_manifest.npz')
    generated = fixed_state_menu(read_json(root / 'configs/experiment_06a.json'))
    assert_bitwise_equal(menu, generated)
    baseline = load_npz(source / 'PD_reference_replay.npz')
    unsafe = baseline['episode_min_margins'] < 0
    unsafe_ids = np.flatnonzero(unsafe)
    assert len(unsafe_ids) == config['selection']['all_PD_unsafe'] == 40
    selected = {}
    for name, key in [('nominal', 'nominal_selection'), ('challenging', 'challenging_selection')]:
        eligible = np.flatnonzero((menu['strata'] == name) & ~unsafe)
        selected[name + '_safe'] = np.sort(np.random.default_rng(config['seeds'][key]).choice(eligible, 8, replace=False))
    selected['PD_unsafe'], selected['equilibrium'] = unsafe_ids, np.array([96])
    full_ids = np.sort(np.concatenate(list(selected.values())))
    assert len(full_ids) == len(np.unique(full_ids)) == 57
    pilot = [96, int(selected['nominal_safe'][0]), int(selected['challenging_safe'][0]), int(unsafe_ids[0])]
    records = []
    for state_id in full_ids:
        group = next(name for name, ids in selected.items() if state_id in ids)
        records.append({'state_id': int(state_id), 'initial': menu['initial_conditions'][state_id].tolist(),
                        'group': group, 'PD_unsafe': bool(unsafe[state_id]), 'pilot': bool(state_id in pilot),
                        'reset_seed': config['seeds']['environment'] + int(state_id)})
    return {'all_57_cases': records, 'pilot_ids_in_order': pilot,
            'pilot_cases': [next(r for r in records if r['state_id'] == i) for i in pilot]}


def check_dynamics(config):
    """One-step parity only: no optimization or full-study rollouts."""
    smooth, clipped = symbolic_dynamics(), symbolic_dynamics(clipped=True)
    rng = np.random.default_rng(config['seeds']['dynamics_checks'])
    random = rng.uniform([-1, -3, -2], [1, 3, 2], size=(config['dynamics_verification']['random_points'], 3))
    boundaries = np.array([[t, w, u] for t in [-1., 0., 1.] for w in [-3., 0., 3.] for u in [-2., 0., 2.]])
    outside = np.array([[np.pi / 2, 8., 2.], [-np.pi / 2, -8., -2.], [.8, 7.9, 2.], [-.8, -7.9, -2.]])
    triples = np.vstack([random, boundaries, outside])
    smooth_error, clipped_error, clip_active, actual = [], [], [], []
    env = MeteredEnvironment(200)
    try:
        for i, (theta, omega, torque) in enumerate(triples):
            reset_at(env, [theta, omega], config['seeds']['dynamics_checks'] + i)
            executed = np.array([torque], dtype=np.float32)
            env.step(executed)
            observed = env.unwrapped.state.copy()
            a = np.array(smooth([theta, omega], float(executed[0]))).reshape(2)
            b = np.array(clipped([theta, omega], float(executed[0]))).reshape(2)
            actual.append(observed)
            smooth_error.append(np.abs(observed - a))
            clipped_error.append(np.abs(observed - b))
            clip_active.append(abs(a[1]) > 8.)
        costs = {k: v.tolist() for k, v in env.costs().items()}
    finally:
        env.close()
    smooth_error, clipped_error, clip_active = np.array(smooth_error), np.array(clipped_error), np.array(clip_active)
    n_feasible_domain = len(random) + len(boundaries)
    tolerance = config['dynamics_verification']['one_step_atol']
    assert smooth_error[:n_feasible_domain].max() <= tolerance
    assert not clip_active[:n_feasible_domain].any()
    assert clip_active[n_feasible_domain:].all()
    assert smooth_error[n_feasible_domain:].max() > tolerance
    assert clipped_error.max() <= tolerance
    # Smooth-model derivatives, including trajectory indexing and Hessians.
    nlp, _, _, check = shooting_problem(4)
    z, initial = nlp['x'], nlp['p']
    jac = ca.Function('constraint_jacobian', [z, initial], [ca.jacobian(nlp['g'], z)])
    grad = ca.Function('objective_gradient', [z, initial], [ca.gradient(nlp['f'], z)])
    hess = ca.Function('objective_hessian', [z, initial], [ca.hessian(nlp['f'], z)[0]])
    h, errors = config['dynamics_verification']['finite_difference_step'], []
    for _ in range(8):
        x0 = rng.uniform([-.8, -2], [.8, 2])
        states = np.vstack([x0, rng.uniform([-.8, -2], [.8, 2], size=(4, 2))])
        vector = pack(states, rng.uniform(-2, 2, size=4))
        directions = np.eye(12)
        fd_jac, fd_grad, fd_hess = [], [], []
        for d in directions:
            fp, gp = check(vector + h * d, x0)
            fm, gm = check(vector - h * d, x0)
            fd_jac.append(np.array((gp - gm) / (2 * h)).reshape(-1))
            fd_grad.append(float((fp - fm) / (2 * h)))
            fd_hess.append(np.array((grad(vector + h*d, x0) - grad(vector - h*d, x0)) / (2*h)).reshape(-1))
        errors.append([np.max(np.abs(np.array(jac(vector, x0)) - np.array(fd_jac).T)),
                       np.max(np.abs(np.array(grad(vector, x0)).reshape(-1) - fd_grad)),
                       np.max(np.abs(np.array(hess(vector, x0)) - np.array(fd_hess).T))])
    assert np.max(errors) <= config['dynamics_verification']['derivative_atol']
    metrics = {'passed': True, 'random_feasible_domain_points': len(random), 'boundary_points': len(boundaries),
               'speed_clipping_cases': len(outside), 'max_smooth_error_feasible_domain': float(smooth_error[:n_feasible_domain].max()),
               'max_clipped_model_error_all_cases': float(clipped_error.max()),
               'max_unclipped_error_outside_domain': float(smooth_error[n_feasible_domain:].max()),
               'max_derivative_errors_jacobian_gradient_hessian': np.max(errors, axis=0).tolist(),
               'note': 'Float64 symbolic arithmetic versus Gymnasium float32 torque arithmetic; no safety threshold relaxation'}
    arrays = dict(test_triples=triples, actual_successors=np.array(actual), smooth_errors=smooth_error,
                  clipped_errors=clipped_error, speed_clip_active=clip_active, derivative_errors=np.array(errors))
    return metrics, arrays, costs
