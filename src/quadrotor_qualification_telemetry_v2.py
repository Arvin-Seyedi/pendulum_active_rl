"""Fail-closed candidate preservation for a FUTURE versioned qualification runner.

No controller, simulator or solver is imported or configured here. The guard
wraps an unmodified native selection callback, snapshots BEFORE verification,
and returns an action only after checks pass. It is not wired into frozen v1.
"""
from contextlib import contextmanager
from pathlib import Path
import hashlib
import os
import time
import numpy as np
from src.quadrotor_mmaq_v1 import safe_write
from src.quadrotor_kkt_integrity_v2 import canonical_kkt,require_kkt


class QualificationFailure(RuntimeError):
    pass


class TimingLedger:
    """Nested spans expose native inclusive and exclusive wall/CPU times.

    A nested solve-hook persistence span is subtracted from native-exclusive
    selection time. Verification, replay and physics instrumentation get their
    own spans; no retrospective subtraction from historical data is invented.
    """
    def __init__(self,clock=time.perf_counter,cpu=time.process_time):
        self.clock=clock;self.cpu=cpu;self.records=[];self.stack=[]

    @contextmanager
    def span(self,category):
        frame={'category':category,'wall_start':self.clock(),'cpu_start':self.cpu(),
               'child_wall':0.,'child_cpu':0.}
        self.stack.append(frame)
        try:yield
        finally:
            wall=self.clock()-frame['wall_start'];cpu=self.cpu()-frame['cpu_start']
            assert self.stack.pop() is frame
            self.records.append({'category':category,'wall_inclusive':wall,'cpu_inclusive':cpu,
                'wall_exclusive':max(0.,wall-frame['child_wall']),
                'cpu_exclusive':max(0.,cpu-frame['child_cpu'])})
            if self.stack:
                self.stack[-1]['child_wall']+=wall;self.stack[-1]['child_cpu']+=cpu

    def summary(self):
        """Sum native compute categories; the nested solver remains MPC work."""
        measured=any(r['category']=='native_action_selection' for r in self.records)
        native=[r for r in self.records if r['category'] in ['native_action_selection','native_solver']]
        wall=sum(r['wall_exclusive'] for r in native);cpu=sum(r['cpu_exclusive'] for r in native)
        by_category={}
        for row in self.records:
            bucket=by_category.setdefault(row['category'],{'exclusive_wall':0.,'exclusive_cpu':0.})
            bucket['exclusive_wall']+=row['wall_exclusive'];bucket['exclusive_cpu']+=row['cpu_exclusive']
        return {'native_action_selection_measured':measured,
                'native_action_selection_wall':wall if measured else None,
                'native_action_selection_cpu':cpu if measured else None,
                'native_deadline_20ms_overrun':wall>.02 if measured else None,
                'exclusive_categories':by_category}


def atomic_arrays(path,arrays):
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    converted={k:np.asarray(v) for k,v in arrays.items()}
    if any(v.dtype.hasobject for v in converted.values()):raise TypeError('Object arrays are not permitted')
    with tmp.open('wb') as stream:
        np.savez_compressed(stream,**converted);stream.flush();os.fsync(stream.fileno())
    tmp.replace(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def native_opti_snapshot(opti,solution=None,gradient_expression=None):
    """Snapshot a current native solution/debug point before validation/cleanup.

    This NEVER requests opti.initial() and never substitutes controller.u_prev.
    Unavailable debug values are marked missing; debug data are not claimed to
    be the optimizer's final iterate. No optimizer is configured/called here.
    """
    result={'provenance':'native_solution' if solution is not None else 'native_debug_unverified',
            'capture_errors':{},'solver_status':None}
    try:
        stats=opti.stats();result['native_stats']=stats
        result['solver_status']=stats.get('return_status')
    except Exception as exc:result['capture_errors']['native_stats']=str(exc)
    getter=solution.value if solution is not None else opti.debug.value
    expressions={'primal':opti.x,'parameters':opti.p,'dual':opti.lam_g,'objective':opti.f,
                 'constraint_values':opti.g,'lower':opti.lbg,'upper':opti.ubg}
    try:
        if gradient_expression is None:
            import casadi as c
            gradient_expression=c.gradient(opti.f+c.dot(opti.g,opti.lam_g),opti.x)
        expressions['lagrangian_gradient']=gradient_expression
    except Exception as exc:result['capture_errors']['lagrangian_gradient']=str(exc)
    for name,expression in expressions.items():
        try:result[name]=np.asarray(getter(expression)).copy()
        except Exception as exc:result['capture_errors'][name]=str(exc)
    result['missing_fields']=[k for k in CandidateGuard.REQUIRED if k not in result]
    return result


class CandidateGuard:
    """Testable future guard, with mandatory raw snapshot and partial-prefix APIs.

    extract() must return a dictionary of the CURRENT candidate, with explicit
    provenance ('native_solution' or 'native_debug'), solver_status and arrays.
    It must never synthesize a missing candidate from a previous solution.
    Call this before native Opti/controller cleanup. snapshot() must expose the
    last committed trajectory prefix, even when selection itself raises.
    """
    REQUIRED=('primal','parameters','dual','objective','constraint_values','lower','upper','lagrangian_gradient')

    def __init__(self,folder,context,*,timing=None):
        self.folder=Path(folder);self.folder.mkdir(parents=True,exist_ok=False)
        self.timing=timing or TimingLedger();self.context=context
        self.state={'version':'candidate_guard_v2','status':'incomplete','context':context,
                    'fallback_action_applied':False,'candidate_available':False,
                    'missing_fields':list(self.REQUIRED),'residuals_available':False}
        self.last_diagnostic=None
        safe_write(self.folder/'status.json',self.state)

    def preserve_candidate(self,candidate):
        """Also usable inside a future native solve hook, before any validation."""
        numeric={k:v for k,v in candidate.items() if isinstance(v,(np.ndarray,np.generic,int,float,list,tuple))}
        with self.timing.span('candidate_persistence'):
            digest=atomic_arrays(self.folder/'candidate.npz',numeric)
            metadata={k:v for k,v in candidate.items() if k not in numeric}
            missing=[k for k in self.REQUIRED if k not in candidate or candidate[k] is None]
            self.state.update(candidate_available=True,candidate_sha256=digest,
                              candidate_metadata=metadata,available_fields=sorted(candidate),missing_fields=missing)
            safe_write(self.folder/'candidate_metadata.json',self.state)
            safe_write(self.folder/'status.json',self.state)
        diagnostic_fields=['constraint_values','lower','upper','dual','lagrangian_gradient']
        if all(k in candidate and candidate[k] is not None for k in diagnostic_fields):
            with self.timing.span('verification'):
                diagnostic=canonical_kkt(*(candidate[k] for k in diagnostic_fields))
                self.last_diagnostic=diagnostic
            with self.timing.span('diagnostic_persistence'):
                safe_write(self.folder/'constraint_diagnostics.json',diagnostic)
                self.state['residuals']=diagnostic['metrics'];self.state['residuals_available']=True
                safe_write(self.folder/'status.json',self.state)
        return missing

    def native_solve_hook(self,opti,solve,gradient_expression=None):
        """Future scoped solve hook: persist even a subsequently rejected return.

        Install only in a separately authorized future runner. When called inside
        native selection, TimingLedger subtracts snapshot/persistence overhead
        from native-exclusive action latency. An exception is re-raised unchanged.
        """
        solution=None
        try:
            with self.timing.span('native_solver'):
                solution=solve()
        finally:
            with self.timing.span('candidate_extraction'):
                candidate=native_opti_snapshot(opti,solution,gradient_expression)
            self.preserve_candidate(candidate)
        return solution

    def _prefix(self,snapshot,label):
        with self.timing.span('partial_trajectory_persistence'):
            arrays=snapshot()
            self.state[label+'_sha256']=atomic_arrays(self.folder/(label+'.npz'),arrays)

    def select(self,native_select,extract,snapshot,*,primal_tolerance=1e-4,required_status='Solve_Succeeded'):
        failure=None;action=None;native_error=None
        try:
            self._prefix(snapshot,'prefix_before')
            try:
                with self.timing.span('native_action_selection'):
                    action=native_select()
            except Exception as exc:
                native_error=exc
                self.state['native_exception']={'kind':type(exc).__name__,'message':str(exc)}
            # Extract even when native selection failed. Any unavailable field is
            # recorded as missing, not replaced by a fabricated/previous candidate.
            with self.timing.span('candidate_extraction'):
                candidate=extract()
            if not isinstance(candidate,dict) or not candidate:
                self.state['missing_fields']=list(self.REQUIRED)
                raise QualificationFailure('Current candidate unavailable')
            missing=self.preserve_candidate(candidate)
            if missing:raise QualificationFailure('Missing current candidate fields: '+str(missing))
            diagnostic=self.last_diagnostic
            if native_error is not None:raise QualificationFailure('Native selection failed; candidate retained') from native_error
            if not all(np.isfinite(np.asarray(candidate[k])).all() for k in ['primal','parameters','dual','objective']):
                raise QualificationFailure('Nonfinite candidate')
            if np.asarray(candidate['primal']).size!=np.asarray(candidate['lagrangian_gradient']).size:
                raise QualificationFailure('Candidate/gradient dimensions differ')
            if not np.isfinite(np.asarray(action)).all():raise QualificationFailure('Nonfinite returned action')
            if 'success' in candidate.get('native_stats',{}):
                success=candidate['native_stats']['success']
                if not isinstance(success,(bool,np.bool_)) or not bool(success):
                    raise QualificationFailure('Native solver success flag is false or invalid')
            require_kkt(diagnostic,candidate.get('solver_status'),primal_tolerance=primal_tolerance,
                        required_status=required_status)
            self.state['status']='verified_action_not_yet_applied'
        except Exception as exc:
            failure=exc;self.state['status']='failed_or_incomplete'
            self.state['failure']={'kind':type(exc).__name__,'message':str(exc)}
        finally:
            try:self._prefix(snapshot,'prefix_after')
            except Exception as exc:
                self.state['prefix_error']={'kind':type(exc).__name__,'message':str(exc)}
                self.state['status']='failed_or_incomplete';failure=failure or exc
            self.state['timings']=self.timing.records
            self.state['timing_summary']=self.timing.summary()
            self.state['returned_action']=action
            safe_write(self.folder/'status.json',self.state)
        if failure is not None:raise QualificationFailure('No action authorized; inspect retained attempt') from failure
        return np.asarray(action).copy()
