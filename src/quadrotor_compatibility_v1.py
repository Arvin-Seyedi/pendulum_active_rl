"""E12 diagnostic-only solver dispatch adapter, version 1.

Not connected to the accepted pilot or any controller. This helper only selects
the published QRQP backend for an already constructed linear conic problem.
No objective, variable, parameter, constraint or initial guess is changed.
"""
ADAPTER_VERSION = 'linear_conic_qrqp_v1'


def configure_linear_conic_qrqp_v1(opti, requested_solver, native_backend, native_options):
    """Replace the exact pinned Darwin dispatch, or fail closed.

Future use must be scoped to LinearMPC.setup_optimizer after its unmodified
problem construction. Do not apply this helper to nonlinear MPC.
"""
    if requested_solver != 'qrqp' or native_backend != 'ipopt':
        raise ValueError('Adapter only covers the published QRQP / Darwin IPOPT mismatch')
    if native_options != {'expand': True, 'ipopt.max_iter': 100}:
        raise ValueError('Pinned upstream solver options changed; review required')
    # ipopt.max_iter is backend-specific and invalid for QRQP. Keep expand.
    # Do not translate it into a new QRQP iteration limit.
    opti.solver(requested_solver, {'expand': True})
    return {'adapter_version': ADAPTER_VERSION, 'backend': requested_solver,
            'options': {'expand': True}, 'removed_backend_specific_option': 'ipopt.max_iter'}
