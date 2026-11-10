"""Archive parity gate around a separately guarded ORIGINAL MPC query.

No controller/solver/environment is instantiated. Native candidate capture and
v2 KKT checking must precede this additional parity check; failures propagate.
"""
from pathlib import Path
import numpy as np
from src.quadrotor_counterfactual_v1 import same


def require_archived_query(folder,step,observation,reference,candidate,warmstart):
    """Compare declared numerical context, not unavailable solver internals."""
    folder=Path(folder)
    with np.load(folder/'attempts'/f'{step:04d}'/'candidate.npz') as old:
        for key in ['primal','parameters','dual']:
            current=np.asarray(candidate[key])
            if not np.isfinite(current).all() or not same(current,old[key]):
                raise ValueError('Native candidate differs bitwise from archive: '+key)
        params=old['parameters'].reshape(-1)
        if not same(np.asarray(observation),params[:6]):raise ValueError('Observation differs from archived query')
        if not same(np.asarray(reference),params[6:].reshape((6,81),order='F')):raise ValueError('Reference preview differs from archive')
    with np.load(folder/'attempts'/f'{step:04d}'/'warmstart.npz') as old:
        for key in ['initial_guess_not_solver_candidate','previous_x','previous_u']:
            if not same(np.asarray(warmstart[key]),old[key]):raise ValueError('Initial guess/history differs from archive: '+key)
    return True


def parity_query(native_verified_query,evidence,controller,folder,*,zero_offset):
    """For nominal require all queries; for alternative require first query only.

    evidence() returns the captured current candidate and warmstart, never asks
    the optimizer to solve or substitutes a previous plan as a candidate.
    """
    def query(observation,info):
        action=native_verified_query(observation,info)
        step=info['current_step']
        if zero_offset or step==0:
            point=evidence()
            require_archived_query(folder,step,observation,controller.get_references(step),point['candidate'],point['warmstart'])
        return action  # preserve original view; outer selector alone detaches it
    return query


def prepare_selector(controller,native_verified_query,evidence,folder,archive,model,km,menu_index,*,opti_state,contract,sink):
    """Compose parity AFTER native verification and BEFORE detached substitution.

    Runtime factories, candidate guard, application observer, raw journal and
    resource reservation are unchanged dependencies of the future runner.
    """
    from src.quadrotor_counterfactual_v1 import CounterfactualPath
    query=parity_query(native_verified_query,evidence,controller,folder,zero_offset=menu_index==4)
    return CounterfactualPath(controller,query,model,km,menu_index,archive,
        opti_state=opti_state,contract=contract,sink=sink)
