"""Passive application-boundary telemetry for the pinned PYB plant.

Direct observations: native preprocessing output, _physics(rpm) argument and
forwarded force/torque API arguments. KF-derived grouped thrust is labelled as
derived. last_clipped_action is retained only as explicitly stale debug context.
No controller, optimizer, state update, reset, random draw or integrator exists
in this module. Install inside the existing raw-physics capture context.
"""
from contextlib import ExitStack
from unittest.mock import patch
import copy
import numpy as np

VERSION='applied_action_boundary_observer_v2'


class ObservabilityError(RuntimeError):pass


def copied(value):
    """Detached diagnostics must never alias a plant/controller argument."""
    return copy.deepcopy(value)


class AppliedActionObserver:
    def __init__(self,env,backend,*,after_step=None,sink=None):
        self.env=env;self.backend=backend;self.after_step=after_step;self.sink=sink
        self.preprocess_records=[];self.applications=[];self.steps=[]
        self.pending=None;self.active=False;self.foreign_submissions=[]
        self.stack=None;self.control_index=-1;self.physics_index=0

    def emit(self,packet):
        if self.sink is not None:self.sink(copied(packet))

    def target(self,args,kwargs,kind):
        body=args[0] if args else kwargs.get('objectUniqueId')
        client=kwargs.get('physicsClientId',0)
        return body==self.env.DRONE_ID and client==self.env.PYB_CLIENT

    def __enter__(self):
        if self.stack is not None:raise RuntimeError('Observer is not reentrant')
        if getattr(self.env.PHYSICS,'value',self.env.PHYSICS)!='pyb':
            raise ObservabilityError('Only the frozen plain PYB mode is covered')
        if self.env.disturbances or self.env.adversary_disturbance is not None:
            raise ObservabilityError('Disturbed plant is outside the frozen contract')
        stack=ExitStack();self.stack=stack
        env,b=self.env,self.backend
        preprocess=env._preprocess_control;physics=env._physics
        force=b.applyExternalForce;torque=b.applyExternalTorque;step=b.stepSimulation

        def preprocess_observed(action,*args,**kwargs):
            self.control_index+=1
            r={'control_index':self.control_index,'requested_argument':copied(action),
               'provenance':'native preprocessing call; no independent conversion substituted'}
            self.preprocess_records.append(r)
            try:
                result=preprocess(action,*args,**kwargs)
                r.update(native_returned_rpm=copied(result),
                         native_physical_action=copied(env.current_physical_action),
                         native_clipped_action=copied(env.current_clipped_action),complete=True)
                return result  # Same object, not the diagnostic copy.
            except BaseException as exc:
                r.update(complete=False,error=repr(exc));raise

        def submission(original,kind,args,kwargs):
            event={'kind':kind,'args':copied(args),'kwargs':copied(kwargs),'returned':False}
            target=self.target(args,kwargs,kind)
            if target and self.active:self.pending['submissions'].append(event)
            elif target:self.foreign_submissions.append(event)
            try:
                result=original(*args,**kwargs);event['returned']=True;return result
            except BaseException as exc:event['error']=repr(exc);raise

        def physics_observed(rpm,*args,**kwargs):
            if self.pending is not None:raise ObservabilityError('Application was not associated with one physics step')
            packet={'version':VERSION,'application_index':len(self.applications),
                    'control_index':self.control_index,'physics_index':self.physics_index,
                    'direct_rpm_argument':copied(rpm),'submissions':[],
                    'stale_cached_rpm_debug_only':copied(env.last_clipped_action),
                    'force_application_returned':False,'physics_returned':False,
                    'provenance':'direct native _physics argument and submitted API arguments'}
            packet['preprocess_record']=copied(self.preprocess_records[-1]) if self.preprocess_records else None
            self.applications.append(packet);self.pending=packet;self.active=True
            try:
                result=physics(rpm,*args,**kwargs)
                packet['force_application_returned']=True;self.emit(packet);return result
            except BaseException as exc:packet['application_error']=repr(exc);self.emit(packet);raise
            finally:self.active=False

        def step_observed(*args,**kwargs):
            client=kwargs.get('physicsClientId',args[0] if args else 0)
            if client!=env.PYB_CLIENT:return step(*args,**kwargs)
            packet=self.pending
            if packet is None:raise ObservabilityError('No directly observed motor application for this step')
            try:
                self.validate_packet(packet)
                result=step(*args,**kwargs)  # Calls existing raw capture before the callback.
                packet['physics_returned']=True;self.physics_index+=1;self.pending=None
                self.steps.append(packet);self.emit(packet)
                if self.after_step is not None:self.after_step(copied(packet))
                return result
            except BaseException as exc:packet['step_or_callback_error']=repr(exc);self.emit(packet);raise

        try:
            stack.enter_context(patch.object(env,'_preprocess_control',preprocess_observed))
            stack.enter_context(patch.object(env,'_physics',physics_observed))
            stack.enter_context(patch.object(b,'applyExternalForce',lambda *a,**k:submission(force,'force',a,k)))
            stack.enter_context(patch.object(b,'applyExternalTorque',lambda *a,**k:submission(torque,'torque',a,k)))
            stack.enter_context(patch.object(b,'stepSimulation',step_observed))
        except BaseException:stack.close();self.stack=None;raise
        return self

    def validate_packet(self,packet):
        rpm=np.asarray(packet['direct_rpm_argument'])
        if rpm.shape!=(4,) or not np.isfinite(rpm).all() or np.any(rpm<0):
            raise ObservabilityError('Invalid direct motor argument')
        control=packet['preprocess_record']
        if not control or not control.get('complete') or control['control_index']!=packet['control_index']:
            raise ObservabilityError('Missing current preprocessing record')
        if np.asarray(control['native_returned_rpm']).tobytes()!=rpm.tobytes():
            raise ObservabilityError('Applied motor command differs from observed preprocessing output')
        if not packet['force_application_returned']:raise ObservabilityError('Force application did not complete')
        forces=[e for e in packet['submissions'] if e['kind']=='force']
        torques=[e for e in packet['submissions'] if e['kind']=='torque']
        if len(forces)!=4 or len(torques)!=1:raise ObservabilityError('Incomplete force/torque application record')
        z=[]
        for link,event in enumerate(forces):
            if not event['returned']:raise ObservabilityError('Force API call failed')
            args,kw=event['args'],event['kwargs']
            actual_link=args[1] if len(args)>1 else kw.get('linkIndex')
            v=np.asarray(kw.get('forceObj',args[2] if len(args)>2 else None))
            if actual_link!=link or v.shape!=(3,) or not np.isfinite(v).all():raise ObservabilityError('Invalid motor-link record')
            if kw.get('flags')!=self.backend.LINK_FRAME or not np.array_equal(v[:2],[0,0]):
                raise ObservabilityError('Unexpected application frame or transverse force')
            if not np.array_equal(kw.get('posObj'),[0,0,0]):raise ObservabilityError('Unexpected application point')
            z.append(v[2])
        torque=torques[0];v=np.asarray(torque['kwargs']['torqueObj'])
        if not torque['returned'] or v.shape!=(3,) or not np.isfinite(v).all():raise ObservabilityError('Invalid torque record')
        link=torque['args'][1] if len(torque['args'])>1 else torque['kwargs'].get('linkIndex')
        if link!=4 or torque['kwargs'].get('flags')!=self.backend.LINK_FRAME:
            raise ObservabilityError('Unexpected torque application link/frame')
        if self.foreign_submissions:raise ObservabilityError('Additional force submission outside the observed application')
        packet['direct_submitted_force_z_by_motor']=np.asarray(z)
        packet['grouped_submitted_force_argument_sums']=np.array([z[0]+z[3],z[1]+z[2]])
        packet['derived_grouped_thrust_from_direct_rpm']=self.env.KF*np.array([rpm[0]**2+rpm[3]**2,rpm[1]**2+rpm[2]**2])
        packet['observation_complete']=True

    def __exit__(self,*exc):
        try:
            if self.pending is not None:
                self.pending['context_exit_with_unconsumed_application']=True;self.emit(self.pending)
        finally:
            self.stack.close();self.stack=None

