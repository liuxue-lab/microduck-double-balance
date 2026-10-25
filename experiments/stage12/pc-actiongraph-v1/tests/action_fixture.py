"""Original reviewed method bodies, using a CPU substitute for MuJoCo data."""
import ast
from pathlib import Path
import os
import sys
from types import ModuleType, SimpleNamespace as NS
import torch

ROOT=Path(__file__).resolve().parents[1]
REF=Path(os.environ['MICRODUCK_REFERENCE_TESTS'])
sys.path.insert(0,str(REF))
from command_fixture import module as legacy_module
from bridge_fixture import BRIDGE,bridge_data


def extract(name,path,classes,**values):
    m=ModuleType(name);m.__file__=str(path);sys.modules[name]=m
    m.__dict__.update(torch=torch,**values)
    tree=ast.parse(path.read_bytes())
    tree.body=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in classes]
    for n in tree.body:
        n.bases=[];n.keywords=[];n.decorator_list=[]
        n.body=[f for f in n.body if isinstance(f,ast.FunctionDef) and f.name in classes[n.name]]
    tree.body.insert(0,ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0))
    exec(compile(ast.fix_missing_locations(tree),str(path),'exec'),m.__dict__)
    return m


AM=extract('action_original_manager',ROOT/'tests/fixtures/action-manager.py',
           {'ActionManager':('process_action','apply_action','total_action_dim','action_term_dim')})
AC=extract('action_original_terms',ROOT/'tests/fixtures/actions.py',
           {'BaseAction':('process_actions',),'JointPositionAction':('apply_actions',)})
AC.JointPositionAction.process_actions=AC.BaseAction.process_actions
EN=extract('action_original_entity',REF/'fixtures/entity-v1.3.0.py',
           {'Entity':('data','set_joint_position_target','write_external_wrench_to_sim')})
DA=extract('action_original_data',REF/'fixtures/entity-data.py',
           {'EntityData':('write_external_wrench','_resolve_env_ids','joint_pos','joint_vel')})
MD=extract('action_original_hold',REF/'fixtures/mdp-reviewed.py',
           {'BallHoldAction':('process_actions','apply_actions')},_bb_state=lambda env:env.bb)
from test_head_graph import MATH
MD._quat_rotate=MATH.quat_apply


def attach(raw,robot_data=None):
    """Attach real method bodies to a minimal, possibly pre-existing plant."""
    data_type=DA.EntityData if robot_data is None else type(robot_data)
    for name in ('write_external_wrench','_resolve_env_ids'):
        setattr(data_type,name,getattr(DA.EntityData,name))
    class_data=NS(EntityData=data_type)
    simdata=raw.sim.data
    def body(ids,data=None):
        d=data if data is not None else data_type.__new__(data_type)
        d.data=simdata
        if not hasattr(d,'indexing'):
            d.indexing=NS(joint_q_adr=torch.arange(14,dtype=torch.int32),
                          joint_v_adr=torch.arange(14,dtype=torch.int32))
        d.indexing.body_ids=torch.tensor(ids,dtype=torch.int32)
        for k in ('root_link_pos_w','root_link_lin_vel_w','root_link_ang_vel_w'):
            if not hasattr(d,k):setattr(d,k,torch.zeros(1,3))
        if not hasattr(d,'root_link_quat_w'):d.root_link_quat_w=torch.tensor([[1.,0.,0.,0.]])
        entity=EN.Entity();entity._data=d
        return entity
    robot=body([2,3],robot_data)
    ball=body([1])
    if 'robot' in raw.scene:
        robot.__dict__.update({k:v for k,v in vars(raw.scene['robot']).items() if k not in ('data','_data')})
    for k in ('joint_pos_target','joint_vel_target','joint_effort_target'):
        if not hasattr(robot.data,k):setattr(robot.data,k,torch.zeros(1,14))
    robot.data.encoder_bias=torch.arange(14,dtype=torch.float32).reshape(1,14)*.0003
    # Preserve unrelated non-default destination contents when creating terms.
    raw.scene['robot']=robot
    raw.scene['ball']=ball
    raw.bb=NS(hold=torch.zeros(1),anchor=torch.zeros(1,3))
    joint=AC.JointPositionAction();joint._entity=robot
    joint._target_ids=torch.arange(13,-1,-1,dtype=torch.int64)
    joint._raw_actions=torch.zeros(1,14);joint._processed_actions=torch.zeros(1,14)
    joint._scale=1.;joint._offset=0.;joint.cfg=NS(clip=None);joint.action_dim=14
    hold=MD.BallHoldAction();hold._entity=ball;hold._duck=robot;hold._env=raw
    hold._force=None;hold._up=torch.tensor([[0.,0.,1.]]);hold.action_dim=0
    hold.cfg=NS(kp=40.,kd=4.,kd_rot=.08,duck_kp_rot=3.,duck_kd_rot=.08,duck_kp=40.,duck_kd=4.)
    manager=AM.ActionManager();manager.device='cpu';manager._terms=dict(joint_pos=joint,ball_hold=hold)
    manager._action=torch.zeros(1,14);manager._prev_action=torch.zeros(1,14)
    manager._prev_prev_action=torch.zeros(1,14)
    raw.action_manager=manager
    modules={'mjlab.managers.action_manager':AM,'mjlab.envs.mdp.actions.actions':AC,
             'mjlab.entity.entity':EN,'mjlab.entity.data':class_data,'mjlab_microduck.tasks.mdp':MD}
    return modules


def plant():
    raw=NS(sim=NS(data=bridge_data(NS(xfrc_applied=torch.zeros(1,5,6),
                                    qpos=torch.zeros(1,14),qvel=torch.zeros(1,14)))),scene={})
    modules=attach(raw)
    return raw,modules


def cpu_capture(runner):
    # Simulates graph replay with the SAME method bodies, not CUDA scheduling.
    before={k:x.clone() for k,x in runner.live.items()}
    for _ in range(3):runner.private.compute(runner.live)
    for k,x in runner.live.items():assert torch.equal(x.view(torch.int32),before[k].view(torch.int32))
    runner.graph=NS(replay=runner.compute_commit)
