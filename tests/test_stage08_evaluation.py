"""Protocol separation, exact initial states and cached metric timing."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch
from mjlab.managers.metrics_manager import MetricsManager
from mjlab.tasks.registry import load_env_cfg
import mjlab_microduck.tasks  # noqa: F401
from mjlab_microduck.tasks import mdp
from mjlab_microduck.double_balance_training import terminal_episodes
from mjlab_microduck.double_balance_stage08_evaluation import (
    MetricObserver, evaluation_config, seal_initial_states, summarize, validate_selection,
)


def test_nominal_task_and_random_protocol_are_separate():
    original=load_env_cfg('Mjlab-DoubleBalance-MicroDuck',play=True)
    nominal,dev=evaluation_config('nominal'),evaluation_config('dev')
    assert nominal.events['reset_double_balance'].params==original.events['reset_double_balance'].params
    assert dev.events['reset_double_balance'].params['xy_noise']==.01
    assert nominal.events['reset_double_balance'].params['xy_noise']==0
    for cfg in (nominal,dev):
        assert cfg.metrics==original.metrics and cfg.rewards==original.rewards
        assert cfg.terminations==original.terminations
        assert cfg.episode_length_s==10 and cfg.actions['ball_hold'].levels==(0.,)
        assert set(cfg.events)==set(original.events) and not cfg.curriculum


def test_actual_metrics_compute_called_once_and_final_success_is_not_ever_success(monkeypatch):
    n=2
    speed=torch.zeros(n)
    root=torch.tensor([[0,0,.36]]*n)
    ball=torch.tensor([[0,0,.12]]*n)
    top=torch.tensor([[0,0,.02]]*n)
    lower_velocity=torch.zeros(n,3)
    raw=NS(num_envs=n,device='cpu',step_dt=.02,
           episode_length_buf=torch.zeros(n,dtype=torch.long),
           scene={'robot':NS(data=NS(root_link_pos_w=root)),
                  'ball':NS(data=NS(root_link_lin_vel_w=lower_velocity))},
           _basketball_state=NS(hold=torch.zeros(n)),
           action_manager=NS(action=torch.zeros(n,14),prev_action=torch.zeros(n,14)),
           reset_terminated=torch.tensor([False,True]),termination_manager=NS(active_terms=[]))
    monkeypatch.setattr(mdp,'_double_balance_top_ball_kinematics',lambda env,**kwargs:(top,torch.zeros(n,3),torch.zeros(n,3)))
    monkeypatch.setattr(mdp,'_bb_ball_pos',lambda env:ball)
    monkeypatch.setattr(mdp,'wrestle_tilt',lambda env,name:torch.zeros(n))
    cfg=load_env_cfg('Mjlab-DoubleBalance-MicroDuck',play=True)
    raw.metrics_manager=MetricsManager({k:cfg.metrics[k] for k in
        ('double_balance_stable_fraction','double_balance_success','top_ball_center_error_m')},raw)
    observer=MetricObserver(raw,torch.ones(n,dtype=torch.bool))
    for step in range(1,501):
        raw.episode_length_buf.fill_(step)
        if step==451:
            lower_velocity[0,0]=.16
        observer.compute()
    assert raw.metrics_manager._step_count.tolist()==[500,500]
    assert raw._double_balance_stable_time[0]==0
    assert raw._double_balance_stable_time[1].item()==pytest.approx(10,abs=1e-4)
    episodes,_=terminal_episodes(raw,torch.ones(n,dtype=torch.bool),torch.ones(n,dtype=torch.bool))
    trace=observer.finish(episodes,['state-a','state-b'],.02)
    assert [e['success'] for e in episodes]==[False,False]
    assert episodes[0]['longest_stable_seconds']==9
    assert episodes[0]['terminal_stable_seconds']==0
    assert episodes[0]['lower_speed_violation_fraction']==pytest.approx(.1)
    assert trace['values'].shape[0]==500
    result=summarize(episodes,'nominal')
    assert result['successes']==0 and result['success_wilson95'] is None


def initial_env():
    class Scene(dict):
        env_origins=torch.zeros(2,3)
    scene=Scene()
    for name in ('robot','ball','top_ball'):
        pose=torch.zeros(2,7); pose[:,3]=1; pose[1,0]=.001
        scene[name]=NS(data=NS(root_link_pose_w=pose,root_link_vel_w=torch.zeros(2,6),
                              joint_pos=torch.zeros(2,2),joint_vel=torch.zeros(2,2)))
    return NS(num_envs=2,scene=scene)


def test_persisted_initial_states_reject_new_states_in_same_set(tmp_path):
    raw=initial_env()
    first=seal_initial_states(raw,'dev',tmp_path)
    second=seal_initial_states(raw,'dev',tmp_path)
    assert first['sha256']==second['sha256']
    raw.scene['robot'].data.joint_pos[0,0]=.01
    with pytest.raises(ValueError,match='initial states'):
        seal_initial_states(raw,'dev',tmp_path)


def test_dev_test_overlap_and_unfrozen_test_access_are_rejected(tmp_path):
    seal_initial_states(initial_env(),'dev',tmp_path)
    with pytest.raises(ValueError,match='overlap'):
        seal_initial_states(initial_env(),'test-8201',tmp_path)
    with pytest.raises(ValueError,match='frozen selection'):
        validate_selection(None,'checkpoint')
