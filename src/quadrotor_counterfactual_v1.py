"""Separately versioned step-zero intervention helpers; no experiment launcher.

The preflight uses scripted native control flow, never a live plant or solver.
A future authorized runner must supply native factories and an independently
verified query callback. Helpers never repair or overwrite MPC optimizer memory.
"""
from copy import deepcopy
from fractions import Fraction
from pathlib import Path
import hashlib
import itertools
import random
import numpy as np
from src.quadrotor_mmaq_v1 import safe_write
from src.quadrotor_mismatch_contract_v1 import bound_components
from src.quadrotor_verification import rk4

VERSION='E12_MPC80_COUNTERFACTUAL_v1'
PROBES=(0,4,8,12)


def same(a,b):
    if isinstance(a,np.ndarray) or isinstance(b,np.ndarray):
        a,b=np.asarray(a),np.asarray(b)
        return a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
    if isinstance(a,dict) and isinstance(b,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def menu_offsets(model):
    width=np.asarray(model['action_high'])-model['action_low']
    return [width*np.asarray(pair) for pair in itertools.product([-.05,0.,.05],repeat=2)]


def commands(request,model,km):
    """Static binary64 recurrence; NOT directly observed live telemetry."""
    request=np.asarray(request,dtype=np.float64)
    if request.shape!=(2,) or not np.isfinite(request).all():raise ValueError('Invalid request')
    clipped=np.clip(request,model['action_low'],model['action_high'])
    thrust=np.clip(clipped,np.zeros_like(clipped),None)
    pwm=(np.sqrt(thrust/2/model['KF'])-model['PWM2RPM_CONST'])/model['PWM2RPM_SCALE']
    pwm=np.clip(np.concatenate([pwm,pwm[::-1]]),model['MIN_PWM'],model['MAX_PWM'])
    rpm=model['PWM2RPM_SCALE']*pwm+model['PWM2RPM_CONST']
    forces=np.zeros((4,3));forces[:,2]=np.array(rpm**2)*model['KF']
    torques=np.array(rpm**2)*km
    torque=np.array([0.,0.,-torques[0]+torques[1]-torques[2]+torques[3]])
    grouped=np.array([forces[0,2]+forces[3,2],forces[1,2]+forces[2,2]])
    derived=model['KF']*np.array([rpm[0]**2+rpm[3]**2,rpm[1]**2+rpm[2]**2])
    lower=grouped-model['action_low'];upper=np.asarray(model['action_high'])-grouped
    admissible=bool(np.all(lower>=0) and np.all(upper>=0) and
                    np.all(derived>=model['action_low']) and np.all(derived<=model['action_high']))
    # Native fixed planar mapping: identical link forces/torque mean identical
    # submitted inputs, even if motor RPM labels differ below force resolution.
    signature=(forces.astype('<f8').tobytes()+torque.astype('<f8').tobytes()).hex()
    return {'provenance':'static source recurrence, not observed execution',
        'requested':request.copy(),'clipped':clipped,'pwm':pwm,'rpm':rpm,'forces':forces,
        'torque':torque,'submitted_grouped_thrust':grouped,'derived_thrust':derived,
        'lower_margins':lower,'upper_margins':upper,'admissible':admissible,
        'submitted_signature':signature}


def separation(left,right,model):
    widths=np.asarray(model['action_high'])-model['action_low']
    ratios=[abs(Fraction(float(a))-Fraction(float(b)))/Fraction(float(w))
            for a,b,w in zip(left,right,widths)]
    value=max(ratios)
    return {'exact_ratio':str(value),'ratio_float':float(value),'at_least_five_percent':value>=Fraction(1,20)}


def labels(model,initial_actions,km):
    rows=[]
    for state in PROBES:
        nominal=np.asarray(initial_actions[state]);baseline=commands(nominal,model,km);seen={}
        for index,offset in enumerate(menu_offsets(model)):
            request=nominal.copy() if index==4 else nominal.copy()+offset
            c=commands(request,model,km);sig=c['submitted_signature']
            representative=seen.setdefault(sig,index)
            rows.append({'state_id':state,'menu_index':index,'offset':offset,'commands':c,
                'duplicate_of':representative if representative!=index else None,
                'input_status':'admissible' if c['admissible'] else 'input_inadmissible',
                'execution_status':'not_run','nominal':index==4,
                'separation_from_nominal':separation(c['submitted_grouped_thrust'],baseline['submitted_grouped_thrust'],model)})
    return rows


def select_alternative(rows):
    """Freeze ID 0, earliest lexicographic nonnominal admissible separated item."""
    for row in rows:
        if row['state_id']==0 and not row['nominal'] and row['duplicate_of'] is None and row['commands']['admissible'] and row['separation_from_nominal']['at_least_five_percent']:
            return deepcopy(row)
    raise ValueError('No eligible prespecified ID-0 alternative; no replacements')


def job_plan(scope):
    if scope not in ['A','B']:raise ValueError(scope)
    cases=[(i,4) for i in (PROBES if scope=='A' else [0])]+[(0,3)]
    return [{'id':f'{s:02d}_menu{m}_{phase}_{kind}','state_id':s,'menu_index':m,'phase':phase,'kind':kind,
             'solver_calls':300 if kind=='scientific' else 0,'physics_steps':6000}
            for s,m in cases for phase in ['original','repeat'] for kind in ['scientific','native_replay','motor_replay']]


def history(controller,opti_state=lambda:None):
    """Copies only; never requests another solution or changes an initial guess."""
    return {**{name:deepcopy(getattr(controller,name,None)) for name in ['x_prev','u_prev','prev_action']},
        'prev_action_attribute_present':hasattr(controller,'prev_action'),'optimizer_state':deepcopy(opti_state())}


def passive_reset_state(env,backend):
    """Read-only base/joint/cache snapshot; NEVER refresh link kinematics/FK."""
    body,client=env.DRONE_ID,env.PYB_CLIENT
    pos,quat=backend.getBasePositionAndOrientation(body,physicsClientId=client)
    vel,ang=backend.getBaseVelocity(body,physicsClientId=client)
    count=backend.getNumJoints(body,physicsClientId=client)
    joints=backend.getJointStates(body,list(range(count)),physicsClientId=client) if count else []
    return {'base_position':np.asarray(pos).copy(),'base_quaternion':np.asarray(quat).copy(),
        'base_linear_velocity':np.asarray(vel).copy(),'base_angular_velocity':np.asarray(ang).copy(),
        'joint_states':deepcopy(joints),'contacts':deepcopy(backend.getContactPoints(bodyA=body,physicsClientId=client)),
        'native_cached_arrays':{k:deepcopy(getattr(env,k,None)) for k in ['pos','quat','rpy','vel','ang_v','last_clipped_action']},
        'control_counter':env.ctrl_step_counter,'physics_counter':getattr(env,'pyb_step_counter',None),
        'environment_RNG':deepcopy(env.np_random.bit_generator.state),
        'numpy_RNG':deepcopy(np.random.get_state()),'python_RNG':random.getstate(),
        'unobservable_internal_engine_state':'not a complete PyBullet checkpoint; no FK query or restore is used'}


def initialize_native_context(environment_factory,controller_factory,raw_state,initial,archived_initial,read_simulator):
    """Same native ordering as accepted p.run_episode; callers require approval.

    Controller factory already calls native ctrl.reset(). Do not call it again
    after the plant reset. Tests supply only fake factories and scripted states.
    """
    env=environment_factory();ctrl=controller_factory(environment_factory)
    obs,info=env.reset()
    x=np.asarray(raw_state(env)).copy()
    if np.max(np.abs(x-initial))>1e-14 or not same(x,np.asarray(archived_initial)):
        raise ValueError('Native reset differs from accepted raw initial state')
    if info['current_step']!=0 or env.ctrl_step_counter!=0 or ctrl.x_prev is not None or ctrl.u_prev is not None:
        raise ValueError('Noninitial reference/controller/reset context')
    if np.any(env.last_clipped_action!=0):raise ValueError('Native reset motor cache differs')
    return env,ctrl,obs,info,deepcopy(read_simulator(env))


class CounterfactualPath:
    """Wrap an independently verified ORIGINAL query; substitute only step zero.

    The candidate's contract callback must use the substituted request and current
    raw/cache context. It must not reuse the nominal optimizer's prediction bound.
    No environment is created or stepped by this selector.
    """
    def __init__(self,controller,verified_query,model,km,menu_index,archive,*,opti_state=lambda:None,
                 contract=None,sink=lambda record:None):
        if menu_index not in range(9):raise ValueError('Menu must be original index 0..8')
        self.ctrl=controller;self.query=verified_query;self.model=model;self.km=km;self.index=menu_index
        self.archive=archive;self.opti_state=opti_state;self.contract=contract;self.sink=sink;self.step=0
    def select(self,obs,info):
        k=self.step;record={'step':k,'menu_index':self.index,'status':'started','fallback_applied':False}
        try:
            if info['current_step']!=k or not 0<=k<300:raise ValueError('Reference/control indexing mismatch')
            if k==0 and (self.ctrl.x_prev is not None or self.ctrl.u_prev is not None):raise ValueError('Initial MPC memory not empty')
            record['observation']=np.asarray(obs).copy();record['reference_preview']=self.ctrl.get_references(k).copy()
            record['pre_query_history']=history(self.ctrl,self.opti_state)
            native=self.query(obs,info)
            nominal=np.asarray(native).copy()
            if nominal.shape!=(2,) or not np.isfinite(nominal).all():raise ValueError('Nonfinite/malformed native action')
            post=history(self.ctrl,self.opti_state);record['post_query_history']=post;record['nominal_request']=nominal
            if k==0:
                if not same(nominal,self.archive['actions'][0]):raise ValueError('Nominal first query differs from archive')
                if not same(self.ctrl.x_prev,self.archive['predicted_states'][0]) or not same(self.ctrl.u_prev,self.archive['predicted_inputs'][0]):
                    raise ValueError('First post-query planned trajectory differs')
            request=nominal.copy()
            if k==0 and self.index!=4:request=request+menu_offsets(self.model)[self.index]
            c=commands(request,self.model,self.km);record['application_request']=request.copy();record['source_predicted_commands']=c
            if not c['admissible']:record['status']='input_inadmissible';raise ValueError('Negative submitted/derived input margin')
            if self.contract is None:raise RuntimeError('Candidate-specific numerical contract callback required')
            record['causal_contract']=self.contract(request.copy(),k)
            reserve=np.asarray(record['causal_contract']['reserve'])
            if reserve.shape!=(6,) or not np.isfinite(reserve).all() or np.any(reserve<=0):raise ValueError('Invalid/nonpositive candidate reserve')
            if not same(post,history(self.ctrl,self.opti_state)):raise ValueError('Intervention changed native post-query memory')
            record['post_intervention_history_unchanged']=True;record['status']='ready_for_native_step'
            self.sink(deepcopy(record));self.step+=1
            return request  # detached even for zero offset; never mutate native view
        except BaseException as exc:
            record.update(error=repr(exc),status=record['status'] if record['status']=='input_inadmissible' else 'failed_or_incomplete')
            self.sink(deepcopy(record));raise


def candidate_contract(x,request,cache,model,*,predict=rk4):
    """Future runtime callback. Preflight uses archived predictions or spy only."""
    b=bound_components(x,request,cache,model);prediction=np.asarray(predict(x,b['model_thrust'],model))
    reserve=np.minimum(prediction-model['state_low'],np.asarray(model['state_high'])-prediction)-b['bound']
    return {'request':np.asarray(request).copy(),'prediction':prediction,'reserve':reserve,**b}


def classify_branch(*,input_admissible,duplicate_of=None,complete=False,observed_violation=False,
                    observed_contact=False,numerical_verified=False,replays_verified=False):
    if duplicate_of is not None:return 'duplicate'
    if not input_admissible:return 'input_inadmissible'
    if observed_violation or observed_contact:return 'empirically_unsafe'
    if not complete or not numerical_verified or not replays_verified:return 'incomplete_or_unverified'
    return 'empirically_safe'


def failure_sink(folder,record,arrays):
    """Additional branch evidence; preserve nonfinite values, never fallback."""
    from src.quadrotor_qualification_telemetry_v2 import atomic_arrays
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'failure.json').exists():raise FileExistsError('Failed branch already preserved')
    atomic_arrays(folder/'partial_context.npz',arrays)
    safe_write(folder/'failure.json',{**record,'fallback_applied':False})
