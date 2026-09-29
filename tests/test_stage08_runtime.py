"""Real CPU Adam, registered cfg/command/action/curriculum tests; no PPO rollout."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab.managers.curriculum_manager import CurriculumManager
from mjlab.tasks.registry import load_env_cfg
import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.double_balance_training import checkpoint_metadata
from mjlab_microduck.double_balance_stage08_plan import apply_training_profile, NUM_ENVS, INITIAL_LR, TRAIN_SEEDS
from mjlab_microduck.double_balance_stage08_state import (
    TASK_ID, assert_zero_assistance, metadata, restore, runtime_profile, save,
    validate_stage08_checkpoint, load_checked, build_training_config,
    check_training_step,
)


class Entity:
    def __init__(self, raw, body):
        self.raw = raw
        self.indexing = NS(body_ids=torch.tensor([body]))
        quat = torch.zeros(NUM_ENVS,4); quat[:,0]=1
        self.data = NS(root_link_pos_w=torch.ones(NUM_ENVS,3)*.1, root_link_quat_w=quat,
                       root_link_lin_vel_w=torch.ones(NUM_ENVS,3)*.2,
                       root_link_ang_vel_w=torch.ones(NUM_ENVS,3)*.3,
                       root_link_lin_vel_b=torch.zeros(NUM_ENVS,3),root_link_ang_vel_b=torch.zeros(NUM_ENVS,3),
                       heading_w=torch.zeros(NUM_ENVS))

    def write_external_wrench_to_sim(self,force,torque,body_ids):
        index = int(self.indexing.body_ids[0])
        self.raw.sim.data.xfrc_applied[:,index,:3] = force[:,0]
        self.raw.sim.data.xfrc_applied[:,index,3:] = torque[:,0]


def runtime(profile='E'):
    cfg=load_env_cfg(TASK_ID); cfg.scene.num_envs=NUM_ENVS
    cfg=apply_training_profile(cfg,profile)
    raw=NS(cfg=cfg,device='cpu',num_envs=NUM_ENVS,step_dt=.02,
           common_step_counter=24000,_sim_step_counter=240000,
           episode_length_buf=torch.full((NUM_ENVS,),100),
           sim=NS(data=NS(xfrc_applied=torch.zeros(NUM_ENVS,2,6))))
    raw.scene={'ball':Entity(raw,0),'robot':Entity(raw,1)}
    term=cfg.actions['ball_hold'].build(raw)
    raw._basketball_state.anchor=torch.zeros(NUM_ENVS,3)
    raw.action_manager=NS(get_term=lambda name:term)
    twist=deepcopy(cfg.commands['twist']).build(raw)
    ids=torch.arange(NUM_ENVS)
    twist._resample_command(ids); twist._update_command()
    raw.command_manager=NS(get_term=lambda name:twist,get_command=lambda name:twist.command)
    rewards,events=deepcopy(cfg.rewards),deepcopy(cfg.events)
    raw.reward_manager=NS(get_term_cfg=lambda name:rewards[name])
    raw.event_manager=NS(get_term_cfg=lambda name:events[name])
    raw.curriculum_manager=CurriculumManager(cfg.curriculum,raw)
    actor,critic=torch.nn.Linear(2,1),torch.nn.Linear(2,1)
    actor.reset=lambda:None; critic.reset=lambda:None
    optimizer=torch.optim.Adam([*actor.parameters(),*critic.parameters()],lr=INITIAL_LR)
    (actor(torch.ones(1,2)).sum()+critic(torch.ones(1,2)).sum()).backward(); optimizer.step()
    for state in optimizer.state.values():
        state['step'].fill_(20000)
    alg=NS(actor=actor,critic=critic,optimizer=optimizer,learning_rate=INITIAL_LR,
           num_learning_epochs=5,num_mini_batches=4)
    alg.save=lambda:{'actor_state_dict':actor.state_dict(),'critic_state_dict':critic.state_dict(),
                     'optimizer_state_dict':optimizer.state_dict()}
    runner=NS(alg=alg,current_learning_iteration=999)
    raw._basketball_state.level[:]=torch.arange(NUM_ENVS)%6
    raw._basketball_state.hold[:]=torch.tensor((1,.5,.25,.1,.03,0))[raw._basketball_state.level]
    checkpoint=deepcopy(alg.save())
    checkpoint.update(iter=999,infos={'env_state':{'common_step_counter':24000},
        'stage07':checkpoint_metadata(runner,raw,1000,'source-test')})
    def reset():
        raw.curriculum_manager.compute(ids)
        raw.episode_length_buf.zero_()
        twist._resample_command(ids); twist._update_command()
    env=NS(unwrapped=raw,num_envs=NUM_ENVS,reset=reset,
           get_observations=lambda:{'actor':torch.zeros(NUM_ENVS,61),'critic':torch.zeros(NUM_ENVS,85)})
    return runner,env,checkpoint


@pytest.mark.parametrize('profile',tuple('ABCDE'))
def test_restore_real_adam_and_hold_after_age_zero_curriculum(profile):
    runner,env,checkpoint=runtime(profile)
    original=deepcopy(runner.alg.optimizer.state_dict())
    assert restore(runner,env,checkpoint,profile=profile,seed=TRAIN_SEEDS[0],campaign_id='campaign',resume=False)==0
    raw=env.unwrapped
    assert runner.current_learning_iteration==1000
    assert raw.common_step_counter==24000 and raw._sim_step_counter==240000
    assert not raw.episode_length_buf.any()
    for k,v in original['state'].items():
        assert torch.equal(v['exp_avg'],runner.alg.optimizer.state_dict()['state'][k]['exp_avg'])
        assert torch.equal(v['exp_avg_sq'],runner.alg.optimizer.state_dict()['state'][k]['exp_avg_sq'])
    assert runner.alg.learning_rate==runner.alg.optimizer.param_groups[0]['lr']==INITIAL_LR
    if profile=='E':
        assert (raw._basketball_state.level==5).all()
        assert_zero_assistance(raw)
    else:
        assert torch.equal(raw._basketball_state.level,checkpoint['infos']['stage07']['level'])


@pytest.mark.parametrize('profile',tuple('ABCDE'))
def test_real_curriculum_manager_preserves_boundary_and_profile_overrides(profile):
    runner,env,checkpoint=runtime(profile)
    restore(runner,env,checkpoint,profile=profile,seed=TRAIN_SEEDS[0],campaign_id='campaign',resume=False)
    raw=env.unwrapped
    before=runtime_profile(raw,profile)
    assert before['action_rate_weight']==-.4
    assert before['com_range']==[-.005,.005]
    raw.common_step_counter=24001
    raw.curriculum_manager.compute(torch.arange(NUM_ENVS))
    after=runtime_profile(raw,profile)
    assert after['action_rate_weight']==(-.6 if profile in 'AB' else -.4)
    assert after['com_range']==([-.01,.01] if profile in 'ABC' else [-.005,.005])
    # Original cfg is a separate object; changing it cannot satisfy the live-manager check.
    if profile in 'CDE':
        raw.reward_manager.get_term_cfg('action_rate_l2').weight=-.6
        with pytest.raises(ValueError,match='overwritten'):
            runtime_profile(raw,profile)


@pytest.mark.parametrize('profile',tuple('BCDE'))
def test_real_command_sampler_is_zero_after_repeated_resampling(profile):
    _,env,_=runtime(profile)
    twist=env.unwrapped.command_manager.get_term('twist')
    for _ in range(5):
        twist._resample_command(torch.arange(NUM_ENVS)); twist._update_command()
        assert not twist.command.any()
        assert twist.is_standing_env.all()


def test_zero_assistance_clears_both_stale_wrenches():
    runner,env,checkpoint=runtime('E')
    raw=env.unwrapped
    term=raw.action_manager.get_term('ball_hold')
    term.process_actions(torch.empty(NUM_ENVS,0)); term.apply_actions()
    assert raw.sim.data.xfrc_applied[:,0].any() and raw.sim.data.xfrc_applied[:,1].any()
    restore(runner,env,checkpoint,profile='E',seed=TRAIN_SEEDS[0],campaign_id='campaign',resume=False)
    assert_zero_assistance(raw)
    check_training_step(raw,'E')
    raw.sim.data.xfrc_applied[0,1,4]=1
    with pytest.raises(ValueError,match='applied wrench'):
        assert_zero_assistance(raw)
    with pytest.raises(ValueError,match='Nonzero command or assistance'):
        check_training_step(raw,'E')


def test_checkpoint_roundtrip_and_cross_experiment_rejection(tmp_path):
    runner,env,checkpoint=runtime('E')
    restore(runner,env,checkpoint,profile='E',seed=TRAIN_SEEDS[0],campaign_id='campaign',resume=False)
    def registered_save(path,infos):
        value=deepcopy(runner.alg.save())
        value.update(iter=runner.current_learning_iteration,infos={**infos,'env_state':{'common_step_counter':24000}})
        torch.save(value,path)
    path=save(runner,env,tmp_path,0,registered_save,profile='E',seed=TRAIN_SEEDS[0],campaign_id='campaign',head='test',purpose='formal')
    saved,state,_=load_checked(path,resume=True,profile='E',seed=TRAIN_SEEDS[0],campaign_id='campaign')
    assert saved['iter']==999 and state['expected_adam_steps']==20000
    for key,value in (('profile','D'),('seed',TRAIN_SEEDS[1]),('campaign_id','new-budget')):
        with pytest.raises(ValueError):
            validate_stage08_checkpoint(saved,**{key:value})
    damaged=deepcopy(saved); damaged['infos']['stage08']['learning_rate']=.001
    with pytest.raises(ValueError,match='LR mismatch'):
        validate_stage08_checkpoint(damaged)
    damaged=deepcopy(saved); damaged['infos']['stage08']['common_step_counter']+=24
    with pytest.raises(ValueError,match='progress'):
        validate_stage08_checkpoint(damaged)


def test_capacity_checkpoints_cannot_initialize_formal_training(tmp_path):
    runner,env,checkpoint=runtime('E')
    restore(runner,env,checkpoint,profile='E',seed=TRAIN_SEEDS[0],campaign_id='campaign',resume=False)
    def registered_save(path,infos):
        value=deepcopy(runner.alg.save()); value.update(iter=999,infos={**infos,'env_state':{'common_step_counter':24000}})
        torch.save(value,path)
    path=save(runner,env,tmp_path,0,registered_save,profile='E',seed=TRAIN_SEEDS[0],campaign_id='campaign',head='test',purpose='capacity')
    with pytest.raises(ValueError,match='Capacity optimization'):
        load_checked(path,resume=True)


def test_source_lr_is_used_when_constructing_registered_agent():
    _,_,checkpoint=runtime('A')
    cfg,agent=build_training_config('D',TRAIN_SEEDS[0],checkpoint)
    assert agent['algorithm']['learning_rate']==INITIAL_LR
    assert agent['algorithm']['schedule']=='adaptive'
    assert agent['algorithm']['num_learning_epochs']*agent['algorithm']['num_mini_batches']==20


def test_nonzero_stage08_resume_keeps_actual_adaptive_lr_and_three_counters():
    runner,env,checkpoint=runtime('E')
    restore(runner,env,checkpoint,profile='E',seed=TRAIN_SEEDS[0],campaign_id='campaign',resume=False)
    raw=env.unwrapped
    raw.common_step_counter=36000; raw._sim_step_counter=360000
    runner.current_learning_iteration=1499
    runner.alg.learning_rate=3.375e-5
    for group in runner.alg.optimizer.param_groups: group['lr']=runner.alg.learning_rate
    for value in runner.alg.optimizer.state.values(): value['step'].fill_(30000)
    state=metadata(runner,raw,500,profile='E',seed=TRAIN_SEEDS[0],campaign_id='campaign',head='test',purpose='formal')
    saved=deepcopy(runner.alg.save())
    saved.update(iter=1499,infos={'stage08':state,'env_state':{'common_step_counter':36000}})
    assert restore(runner,env,saved,profile='E',seed=TRAIN_SEEDS[0],campaign_id='campaign',resume=True)==500
    assert state['source_completed_updates']==1000 and state['lineage_completed_updates']==1500
    assert runner.current_learning_iteration==1500 and raw.common_step_counter==36000
    assert runner.alg.learning_rate==runner.alg.optimizer.param_groups[0]['lr']==3.375e-5
