"""Scripted archived solutions/states; no dynamics or optimization is evaluated."""
from pathlib import Path
from types import SimpleNamespace,MethodType
from copy import deepcopy
import ast
import numpy as np


def native_methods(upstream):
    """Compile the unedited native Python methods against a scripted Opti API.

    This avoids invoking the real MPC class (blocked by preflight). Only its exact
    Python control flow is exercised. solve() below is a fixture array lookup.
    """
    from termcolor import colored
    from safe_control_gym.envs.benchmark_env import Task
    path=Path(upstream)/'safe_control_gym/controllers/mpc/mpc.py'
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MPC')
    nodes=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ['select_action','get_references']]
    scope={'np':np,'deepcopy':deepcopy,'colored':colored,'Task':Task}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope,Task


class ScriptedOpti:
    def __init__(self,archive):
        self.a=archive;self.i=0;self.values={};self.guesses={'x':np.zeros((6,81)),'u':np.zeros((2,80))};self.calls=[]
    def set_value(self,key,value):self.values[key]=np.asarray(value).copy()
    def set_initial(self,key,value):self.guesses[key]=np.asarray(value).copy()
    def solve(self):
        i=self.i;self.i+=1
        self.calls.append({'values':deepcopy(self.values),'guesses':deepcopy(self.guesses)})
        arrays={'x':self.a['predicted_states'][i].copy(),'u':self.a['predicted_inputs'][i].copy()}
        return SimpleNamespace(value=lambda key:arrays[key])
    def stats(self):return {'t_wall_total':0.,'return_status':'Solve_Succeeded','success':True,'mock':True}
    def snapshot(self):return {'values':deepcopy(self.values),'guesses':deepcopy(self.guesses),'scripted_queries':self.i}


def controller(upstream,archive,reference):
    methods,Task=native_methods(upstream);opti=ScriptedOpti(archive)
    ctrl=SimpleNamespace(T=80,traj=reference.T,env=SimpleNamespace(TASK=Task.TRAJ_TRACKING),
        warmstart=True,x_prev=None,u_prev=None,solver='ipopt',use_lqr_gain_and_terminal_cost=False,
        results_dict={name:[] for name in ['horizon_states','horizon_inputs','goal_states','t_wall']},
        opti_dict={'opti':opti,'x_var':'x','u_var':'u','x_init':'initial','x_ref':'reference'})
    ctrl.extract_step=lambda info:info['current_step'] if info is not None else 0
    ctrl.get_references=MethodType(methods['get_references'],ctrl)
    ctrl.select_action=MethodType(methods['select_action'],ctrl)
    return ctrl,opti
