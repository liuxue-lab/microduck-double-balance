"""Use original command extraction, entity call order and delay-buffer methods."""
import ast
import dataclasses
from enum import Enum
from pathlib import Path
from types import ModuleType,SimpleNamespace as NS
from collections.abc import Sequence
import sys
import torch

from bridge_fixture import BRIDGE,bridge_data

FIX=Path(__file__).parent/'fixtures'


def module(name,file,select=None,**values):
    m=ModuleType(name);m.__file__=str(FIX/file);sys.modules[name]=m
    m.__dict__.update(torch=torch,dataclasses=dataclasses,dataclass=dataclasses.dataclass,
                      Enum=Enum,Sequence=Sequence,**values)
    tree=ast.parse((FIX/file).read_bytes())
    if select:
        tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in select]
        for n in tree.body:
            methods=select[n.name]
            if methods is not None:
                n.bases=[];n.keywords=[];n.decorator_list=[]
                n.body=[f for f in n.body if isinstance(f,ast.FunctionDef) and f.name in methods]
    else:tree.body=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    tree.body.insert(0,ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0))
    exec(compile(ast.fix_missing_locations(tree),m.__file__,'exec'),m.__dict__)
    return m


ACT=module('original_command_actuator','actuator-v1.3.0.py',
           dict(TransmissionType=None,ActuatorCmd=None,Actuator=('get_command','apply_delay')))
DATA=module('original_command_entity_data','entity-data.py',dict(EntityData=('joint_pos','joint_vel')))
ENTITY=module('original_command_entity','entity-v1.3.0.py',dict(Entity=('_apply_actuator_controls',)))
CIRCULAR=module('original_command_circular','circular-buffer-v1.3.0.py')
DELAY=module('original_command_delay','delay-buffer-v1.3.0.py',CircularBuffer=CIRCULAR.CircularBuffer)
MODULES={'mjlab.actuator.actuator':ACT,'mjlab.entity.data':DATA,'mjlab.entity.entity':ENTITY,
         'mjlab.sim.sim_data':BRIDGE,'mjlab.utils.buffers.delay_buffer':DELAY,
         'mjlab.utils.buffers.circular_buffer':CIRCULAR}


def prepare_commands(guard,delay=False):
    d=DATA.EntityData();d.data=guard.raw.sim.data
    d.indexing=NS(joint_q_adr=torch.tensor(guard.qids),joint_v_adr=torch.tensor(guard.vids))
    d.joint_pos_target=torch.zeros(1,len(guard.qids))
    d.joint_vel_target=torch.zeros_like(d.joint_pos_target)
    d.joint_effort_target=torch.zeros_like(d.joint_pos_target)
    guard.robot.data=d
    for group in guard.groups:
        type(group).get_command=ACT.Actuator.get_command
        type(group).apply_delay=ACT.Actuator.apply_delay
        group.transmission_type=ACT.TransmissionType.JOINT;group.command_field='position'
        group._delay_buffer=DELAY.DelayBuffer(min_lag=0,max_lag=3,batch_size=1,
            hold_prob=.25,update_period=3,per_env_phase=True,
            generator=torch.Generator().manual_seed(8172)) if delay else None
    return d
