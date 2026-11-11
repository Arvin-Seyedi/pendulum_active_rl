"""Read-only actual IPOPT OptionsList getter for the pinned CasADi runtime."""
from pathlib import Path
import ctypes
import numpy as np

NAMES=['bound_relax_factor','tol','constr_viol_tol','dual_inf_tol','compl_inf_tol','acceptable_tol','honor_original_bounds']

def inspect_options(backend,library):
    lib=ctypes.CDLL(str(Path(library).resolve()))
    fn=lib.cstr_ipopt_options
    fn.argtypes=[ctypes.c_size_t,ctypes.POINTER(ctypes.c_double),ctypes.c_char_p,ctypes.c_size_t]
    fn.restype=ctypes.c_int
    values=(ctypes.c_double*7)();error=ctypes.create_string_buffer(2048)
    code=fn(int(backend.this),values,error,len(error))
    if code:raise RuntimeError('IPOPT option inspection: '+error.value.decode())
    result=dict(zip(NAMES,list(values)))
    if not np.isfinite(list(result.values())).all():raise FloatingPointError('Invalid observed backend options')
    return result
