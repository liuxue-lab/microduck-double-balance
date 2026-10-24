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
           dict(TransmissionType=None,ActuatorCmd=None,Actuator=('get_command','apply_delay',
                'initialize','_init_delay_buffer','has_delay','transmission_type','target_ids',
                'ctrl_ids','global_ctrl_ids')))
DATA=module('original_command_entity_data','entity-data.py',dict(EntityData=('joint_pos','joint_vel')))
def hinge_width(kind):
    assert kind==3
    return 1
ENTITY=module('original_command_entity','entity-v1.3.0.py',
              dict(EntityIndexing=None,Entity=('_apply_actuator_controls','_compute_indexing')),
              mujoco=NS(mjtJoint=NS(mjJNT_FREE=0)),dof_width=hinge_width,qpos_width=hinge_width)
CIRCULAR=module('original_command_circular','circular-buffer-v1.3.0.py')
DELAY=module('original_command_delay','delay-buffer-v1.3.0.py',CircularBuffer=CIRCULAR.CircularBuffer)
MODULES={'mjlab.actuator.actuator':ACT,'mjlab.entity.data':DATA,'mjlab.entity.entity':ENTITY,
         'mjlab.sim.sim_data':BRIDGE,'mjlab.utils.buffers.delay_buffer':DELAY,
         'mjlab.utils.buffers.circular_buffer':CIRCULAR}


def original_indexing(qids,vids,ctrl_ids=None,with_free=False):
    """Original method; substitute only compiled MuJoCo metadata on CPU."""
    assert len(qids)==len(vids)
    joints=[NS(id=i+int(with_free),name='joint_'+str(i)) for i in range(len(qids))]
    spec_joints=[NS(id=0,name='root_free')]+joints if with_free else joints
    table={j.name:NS(type=[3],qposadr=[q],dofadr=[v]) for j,q,v in zip(joints,qids,vids)}
    if with_free:table['root_free']=NS(type=[0],qposadr=[0],dofadr=[0])
    e=ENTITY.Entity();e._non_free_joints=joints;e.is_actuated=True;e.is_fixed_base=False
    e.spec=NS(bodies=[NS(id=0),NS(id=1)],joints=spec_joints,
              actuators=[NS(id=i) for i in (ctrl_ids if ctrl_ids is not None else range(len(joints)))],
              **{k:[] for k in ('geoms','sites','tendons','cameras','lights','materials','pairs')})
    return e._compute_indexing(NS(joint=lambda name:table[name]),'cpu')


def original_actuator_indices(indexing,target_ids,global_ids):
    a=ACT.Actuator();a.entity=NS(indexing=indexing)
    a._target_ids_list=list(target_ids);a._mjs_actuators=[NS(id=int(x)) for x in global_ids]
    a.cfg=NS(transmission_type=ACT.TransmissionType.JOINT,delay_max_lag=0)
    a.initialize(None,None,NS(nworld=1),'cpu')
    return a.target_ids,a.ctrl_ids,a.global_ctrl_ids


def prepare_commands(guard,delay=False):
    d=DATA.EntityData();d.data=guard.raw.sim.data
    d.indexing=original_indexing(guard.qids,guard.vids,guard.ctrl_indices)
    d.joint_pos_target=torch.zeros(1,len(guard.qids))
    d.joint_vel_target=torch.zeros_like(d.joint_pos_target)
    d.joint_effort_target=torch.zeros_like(d.joint_pos_target)
    guard.robot.data=d
    for group in guard.groups:
        group.target_ids,_,_=original_actuator_indices(d.indexing,group.target_ids.tolist(),
                                                      group.global_ctrl_ids.tolist())
        type(group).get_command=ACT.Actuator.get_command
        type(group).apply_delay=ACT.Actuator.apply_delay
        group.transmission_type=ACT.TransmissionType.JOINT;group.command_field='position'
        group._delay_buffer=DELAY.DelayBuffer(min_lag=0,max_lag=3,batch_size=1,
            hold_prob=.25,update_period=3,per_env_phase=True,
            generator=torch.Generator().manual_seed(8172)) if delay else None
    return d
