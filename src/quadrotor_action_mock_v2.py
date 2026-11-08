"""Pinned native action methods on a SCRIPTED backend, never a simulation.

No model integration is performed: stepSimulation merely advances an index into
archived or inert scripted state rows. Its calls are MOCK calls, not PyBullet
physics steps. Native method source and arguments remain unchanged.
"""
from contextlib import contextmanager
from pathlib import Path
from types import MethodType,SimpleNamespace
from unittest.mock import patch
import copy
import hashlib
import json
import sys
import numpy as np
from src.quadrotor_mmaq_v1 import tagged


class ScriptedBackend:
    LINK_FRAME=1;WORLD_FRAME=2
    def __init__(self,states):
        self.states=np.asarray(states).copy();self.index=0;self.trace=[];self.failure=None
    def applyExternalForce(self,*args,**kwargs):
        self.trace.append(('force',copy.deepcopy(args),copy.deepcopy(kwargs)))
        if self.failure=='force':raise RuntimeError('Injected mock force submission failure')
        return None
    def applyExternalTorque(self,*args,**kwargs):
        self.trace.append(('torque',copy.deepcopy(args),copy.deepcopy(kwargs)));return None
    def stepSimulation(self,*args,**kwargs):
        self.trace.append(('step',copy.deepcopy(args),copy.deepcopy(kwargs)))
        if self.failure=='step':raise RuntimeError('Injected mock step failure')
        self.index+=1
        if self.index>=len(self.states):raise RuntimeError('Scripted archive exhausted')
        return None


def trace_digest(trace):
    return hashlib.sha256(json.dumps(tagged(trace),sort_keys=True,separators=(',',':')).encode()).hexdigest()


@contextmanager
def native_harness(upstream,model,states,stride=20):
    upstream=Path(upstream);sys.path.insert(0,str(upstream))
    import safe_control_gym.envs.gym_pybullet_drones.base_aviary as base
    from safe_control_gym.envs.gym_pybullet_drones.quadrotor import Quadrotor
    from safe_control_gym.envs.gym_pybullet_drones.base_aviary import BaseAviary,Physics
    import xml.etree.ElementTree as ET
    asset=upstream/'safe_control_gym/envs/gym_pybullet_drones/assets/cf2x.urdf'
    km=float(ET.parse(asset).getroot().find('properties').attrib['km'])
    backend=ScriptedBackend(states)
    env=SimpleNamespace(PHYSICS=Physics.PYB,PYB_STEPS_PER_CTRL=stride,
        DRONE_ID=11,PYB_CLIENT=17,KM=km,last_clipped_action=np.zeros(4),disturbances={},
        adversary_disturbance=None,update_count=0,
        physical_action_bounds=(np.asarray(model['action_low']),np.asarray(model['action_high'])))
    for name in ['KF','PWM2RPM_SCALE','PWM2RPM_CONST','MIN_PWM','MAX_PWM']:setattr(env,name,model[name])
    env.denormalize_action=lambda action:action
    def update():env.update_count+=1
    env._update_and_store_kinematic_information=update
    env._preprocess_control=MethodType(Quadrotor._preprocess_control,env)
    env._physics=MethodType(BaseAviary._physics,env)
    env._advance_simulation=MethodType(BaseAviary._advance_simulation,env)
    with patch.object(base,'p',backend):yield env,backend
