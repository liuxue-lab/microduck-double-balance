"""Real mjlab managers, step/reset methods and snapshots on a CPU scene fixture.

The scene, simulation and macro physics are explicit doubles. No MjModel,
ManagerBasedRlEnv.__init__, forward, integration, CUDA or PPO is invoked.
"""
from pathlib import Path
from types import SimpleNamespace,MethodType
import traceback


def fixture(torch,wp,active_recorder=False,terminated=False,truncated=False):
    from mjlab.envs import manager_based_rl_env as em
    from mjlab.managers.action_manager import ActionManager,ActionTerm,ActionTermCfg
    from mjlab.managers.command_manager import CommandManager
    from mjlab.managers.observation_manager import ObservationManager,ObservationGroupCfg,ObservationTermCfg
    from mjlab.managers.event_manager import EventManager
    from mjlab.managers.reward_manager import RewardManager,RewardTermCfg
    from mjlab.managers.termination_manager import TerminationManager,TerminationTermCfg
    from mjlab.managers.metrics_manager import MetricsManager
    from mjlab.managers.curriculum_manager import NullCurriculumManager
    from mjlab.managers.recorder_manager import RecorderManager,RecorderTerm,RecorderTermCfg,NullRecorderManager
    from mjlab.utils.buffers import DelayBuffer
    from integration.owner_roster import roots_for
    from formal_owner_keys import FormalOwnerSnapshot
    from source_contracts import build_step_candidate
    class Action(ActionTerm):
        def __init__(self,cfg,env):super().__init__(cfg,env);self.raw=torch.zeros(1,14)
        @property
        def action_dim(self):return 14
        @property
        def raw_action(self):return self.raw
        def process_actions(self,actions):self.raw.copy_(actions)
        def apply_actions(self):pass
    class ActionCfg(ActionTermCfg):
        def build(self,env):return Action(self,env)
    class Recorder(RecorderTerm):
        def __init__(self,cfg,env):super().__init__(cfg,env);self.samples=torch.zeros(1);self.closed=0
        def record_post_step(self):self.samples+=1
        def close(self):self.closed+=1
    class Scene(dict):
        num_envs=1
        def reset(self,ids):pass
        def write_data_to_sim(self):pass
        def update(self,dt):assert dt==.002
    scene=Scene();scene.sensors={}
    act=SimpleNamespace(**{n:torch.zeros(1) for n in ('_prev_motor_torque','_base_kp','vin_tensor','vin_drop_gain',
        'kp_scale','kd_scale','default_kp_scale','default_kd_scale','friction_scale','default_friction_scale')},
        _dt=.002,_friction_fields_checked=True,_delay_buffer=DelayBuffer(min_lag=0,max_lag=0),
        diagnostics_enabled=False,last_diagnostics={},_bam_model=SimpleNamespace(actuator=SimpleNamespace(kp=1.,vin=6.)))
    entity=SimpleNamespace(actuators=[act],data=SimpleNamespace(q=torch.zeros(1,14)))
    scene['robot']=entity;scene.entities={'robot':entity}
    sim=SimpleNamespace(device='cpu',reset=lambda ids:None,forward=lambda:None,sense=lambda:None,
        _reset_mask=SimpleNamespace(_tensor=torch.zeros(1,dtype=torch.bool),_wp_array=wp.zeros(1,dtype=wp.bool,device='cpu')))
    env=em.ManagerBasedRlEnv.__new__(em.ManagerBasedRlEnv)
    env.scene=scene;env.sim=sim
    env.cfg=SimpleNamespace(auto_reset=False,decimation=10,episode_length_s=10.,sim=SimpleNamespace(mujoco=SimpleNamespace(timestep=.002)))
    env.episode_length_buf=torch.zeros(1,dtype=torch.long);env._manual_reset_pending=torch.zeros(1,dtype=torch.bool)
    env._sim_step_counter=90;env.common_step_counter=0;env._g11_tick=0;env._offline_renderer=None;env.extras={}
    env.scenario_terminated=terminated;env.scenario_truncated=truncated
    env._double_balance_tray_site_ids={(id(entity),'stage11_head_material_frame'):7}
    env.action_manager=ActionManager({'joint_pos':ActionCfg(entity_name='robot')},env)
    env.command_manager=CommandManager({},env)
    env.event_manager=EventManager({},env)
    def observation(e,width):return torch.ones(1,width)*e.episode_length_buf[:,None]
    env.observation_manager=ObservationManager({name:ObservationGroupCfg(terms={'value':ObservationTermCfg(func=observation,params={'width':width})})
        for name,width in (('actor',61),('critic',85))},env)
    env.reward_manager=RewardManager({'constant':RewardTermCfg(func=lambda e:torch.ones(1),weight=1.)},env)
    env.termination_manager=TerminationManager({
        'terminated':TerminationTermCfg(func=lambda e:torch.tensor([e.scenario_terminated],dtype=torch.bool)),
        'timeout':TerminationTermCfg(func=lambda e:torch.tensor([e.scenario_truncated],dtype=torch.bool),time_out=True)},env)
    env.metrics_manager=MetricsManager({},env);env.curriculum_manager=NullCurriculumManager()
    env.recorder_manager=RecorderManager({'fixture':RecorderTermCfg(func=Recorder)},env) if active_recorder else NullRecorderManager()
    bank=SimpleNamespace(states={str(i):torch.zeros(8,dtype=torch.uint8) for i in range(9)},
        seeds={str(i):i for i in range(9)},draw_counts={str(i):0 for i in range(9)})
    calls={'macro_fixture':0,'snapshot_checks':0}
    def macro(e):
        e._sim_step_counter+=1;e.action_manager.apply_action();e.scene.write_data_to_sim()
        roots,_=roots_for(e,bank,torch,wp)
        FormalOwnerSnapshot(roots,torch,wp).assert_restored()
        calls['snapshot_checks']+=1;e._g11_tick+=1
        e.scene.update(.002);e.metrics_manager.compute_substep();calls['macro_fixture']+=1
    env.step=MethodType(build_step_candidate(vars(em),macro,derived_state_policy='accepted_endpoint'),env)
    return env,bank,calls


def verify(out,torch,wp):
    from integration.owner_roster import roots_for,recorder_terms
    from integration.transitions import TerminalBridge
    from integration.durable import write_json
    from formal_owner_keys import FormalOwnerSnapshot,owner_state
    from tensordict import TensorDict
    rows=[]
    def run_case(active,t,x):
        env,bank,calls=fixture(torch,wp,active,t,x)
        bridge=TerminalBridge(env,torch,TensorDict,6)
        # The previous code must reproduce the exact reported exception.
        if not active:
            from formal_owners import roots_for as legacy
            try:legacy(env,bank,torch,wp)
            except AttributeError as exc:assert "'_terms'" in str(exc)
            else:raise AssertionError('v4 NullRecorder defect was not reproduced')
        from integration.owner_inventory import record
        name='owner-inventory-%s-%s-%s.json'%(active,t,x)
        meta=record(Path(out)/name,env,bank,torch,wp)
        assert meta['owners']>=13 and not meta['metadata_truncated']
        roots,roster=roots_for(env,bank,torch,wp)
        if not active:
            assert any(label=='recorderManager' and 'active_terms' in names for label,_,names in roots)
        else:assert any(label=='Recorder:0' for label,_,_ in roots)
        snap=FormalOwnerSnapshot(roots,torch,wp);snap.assert_restored()
        original=env.episode_length_buf;original.fill_(11);env.episode_length_buf=torch.ones_like(original)*12
        env.recorder_manager.active_terms.append('mutation')
        bank.states['0'].fill_(3)
        env._double_balance_tray_site_ids[(999,'changed')]=-1
        snap.restore();snap.assert_restored()
        assert env.episode_length_buf is original and original.tolist()==[0]
        raw,scalars,topology=owner_state(roots,torch,wp);assert raw and topology
        for _ in range(2):
            obs,reward,done,extras=bridge.step(torch.zeros(1,14))
            assert done.tolist()==[int(t or x)]
            assert extras['stage12_transition']['bootstrap_mask'].tolist()==[x and not t]
            if t or x:
                assert obs['actor'][0,0].item()==0
                assert extras['stage12_transition']['terminal_observation']['actor'][0,0].item()==1
            roots,_=roots_for(env,bank,torch,wp);FormalOwnerSnapshot(roots,torch,wp).assert_restored()
        assert calls==dict(macro_fixture=20,snapshot_checks=20) and env._sim_step_counter==110
        bridge.reset();roots,_=roots_for(env,bank,torch,wp);FormalOwnerSnapshot(roots,torch,wp).assert_restored()
        terms=list(recorder_terms(env.recorder_manager));env.close()
        if active:assert terms[0].closed==1 and terms[0].samples.tolist()==[2]
        return dict(owner_roots=len(roots),snapshot_bytes=snap.bytes,public_steps_fixture=2,
            real_manager_classes=9,real_pinned_step_reset_methods=True,native_integrations=0)
    for active in (False,True):
        for t,x in ((False,False),(True,False),(False,True),(True,True)):
            name='recorder-%s/terminated-%s/truncated-%s'%(active,t,x)
            try:rows.append(dict(name=name,passed=True,result=run_case(active,t,x)))
            except Exception as exc:rows.append(dict(name=name,passed=False,error=repr(exc),traceback=traceback.format_exc()))
    report=dict(passed=all(r['passed'] for r in rows),cases=rows,physical_environment=False,
        real_Warp_CPU_arrays=True,scene_and_physics_are_fixtures=True,environment_constructions=0,
        native_integrations=0,CPU_forward=0,GPU_forward=0,PPO=0,cloud_training=0)
    write_json(Path(out)/'owner-chain-regression.json',report)
    assert report['passed'], 'Owner chain regression failed; all independent cases retained'
    return report
