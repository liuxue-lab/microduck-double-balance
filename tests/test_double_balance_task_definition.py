"""Stage-04 observation, reward, termination, success, and eval-contract tests."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_double_balance_env_cfg import (
    ACTOR_OBSERVATION_DIM,
    CRITIC_OBSERVATION_DIM,
    DOUBLE_BALANCE_REWARD_WEIGHT,
    SUCCESS_STABLE_DURATION_S,
    TOP_BALL_ACTOR_NORMALIZED_NOISE,
    TOP_BALL_ACTOR_POSITION_SCALE,
    TOP_BALL_ACTOR_VELOCITY_SCALE,
    TOP_BALL_CRITIC_ANGULAR_VELOCITY_SCALE,
    TOP_BALL_RADIUS,
    TOP_BALL_SPEED_WEIGHT,
    TRAY_TILT_WEIGHT,
    make_microduck_double_balance_env_cfg,
)


def test_actor_stays_61d_and_critic_privilege_is_separate():
    cfg = make_microduck_double_balance_env_cfg(play=False, blind=True, history=1)
    actor_terms = cfg.observations["actor"].terms
    critic_terms = cfg.observations["critic"].terms

    assert list(actor_terms)[-2:] == ["head_command", "body_command"]
    actor_top = actor_terms["body_command"]
    assert actor_top.func is mdp.double_balance_top_ball_actor_state
    assert actor_top.params["position_scale"] == TOP_BALL_ACTOR_POSITION_SCALE
    assert actor_top.params["velocity_scale"] == TOP_BALL_ACTOR_VELOCITY_SCALE
    assert actor_top.noise.n_min == -TOP_BALL_ACTOR_NORMALIZED_NOISE
    assert actor_top.noise.n_max == TOP_BALL_ACTOR_NORMALIZED_NOISE
    assert "top_ball_state" not in actor_terms

    assert critic_terms["body_command"].func is mdp.basketball_state
    privileged = critic_terms["top_ball_state"]
    assert privileged.func is mdp.double_balance_top_ball_critic_state
    assert (
        privileged.params["angular_velocity_scale"]
        == TOP_BALL_CRITIC_ANGULAR_VELOCITY_SCALE
    )

    with pytest.raises(ValueError, match="lower basketball must remain actor-blind"):
        make_microduck_double_balance_env_cfg(blind=False)
    with pytest.raises(ValueError, match="requires history=1"):
        make_microduck_double_balance_env_cfg(history=2)


def test_reward_signs_and_functions_follow_repository_convention():
    cfg = make_microduck_double_balance_env_cfg(play=False)
    task = cfg.rewards["double_balance"]
    speed = cfg.rewards["top_ball_speed"]
    tilt = cfg.rewards["tray_tilt"]
    assert task.func is mdp.double_balance_task_reward
    assert task.weight == DOUBLE_BALANCE_REWARD_WEIGHT > 0.0
    assert speed.func is mdp.double_balance_top_ball_speed_l2
    assert speed.weight == TOP_BALL_SPEED_WEIGHT < 0.0
    assert tilt.func is mdp.double_balance_tray_tilt_l2
    assert tilt.weight == TRAY_TILT_WEIGHT < 0.0


def test_top_score_is_multiplicative_and_not_farmable_from_bad_states():
    ideal_position = torch.tensor([[0.0, 0.0, TOP_BALL_RADIUS]])
    zero_velocity = torch.zeros(1, 3)
    ideal = mdp.double_balance_top_score_from_values(
        ideal_position, zero_velocity, ball_radius=TOP_BALL_RADIUS
    )
    torch.testing.assert_close(ideal, torch.ones(1))

    displaced = mdp.double_balance_top_score_from_values(
        torch.tensor([[0.040, 0.0, TOP_BALL_RADIUS]]),
        zero_velocity,
        ball_radius=TOP_BALL_RADIUS,
    )
    wrong_height = mdp.double_balance_top_score_from_values(
        torch.tensor([[0.0, 0.0, -TOP_BALL_RADIUS]]),
        zero_velocity,
        ball_radius=TOP_BALL_RADIUS,
    )
    moving = mdp.double_balance_top_score_from_values(
        ideal_position,
        torch.tensor([[0.30, 0.0, 0.0]]),
        ball_radius=TOP_BALL_RADIUS,
    )
    assert 0.0 < displaced.item() < 0.03
    assert wrong_height.item() < 1e-10
    assert moving.item() < 0.01


def test_top_ball_lost_uses_recoverable_geometry_envelope():
    values = torch.tensor(
        [
            [0.069, 0.000, TOP_BALL_RADIUS],
            [0.071, 0.000, TOP_BALL_RADIUS],
            [0.000, 0.061, TOP_BALL_RADIUS],
            [0.000, 0.000, -0.021],
        ]
    )
    lost = mdp.double_balance_top_ball_lost_from_values(values)
    torch.testing.assert_close(lost, torch.tensor([False, True, True, True]))


def test_success_requires_both_layers_and_low_motion():
    n = 7
    position = torch.tensor([[0.0, 0.0, TOP_BALL_RADIUS]] * n)
    velocity = torch.zeros(n, 3)
    lower_offset = torch.zeros(n)
    lower_height = torch.full((n,), 0.245)
    tilt = torch.zeros(n)
    lower_speed = torch.zeros(n)

    position[1, 0] = 0.013
    velocity[2, 0] = 0.081
    lower_offset[3] = 0.061
    lower_height[4] = 0.199
    tilt[5] = torch.deg2rad(torch.tensor(20.1))
    lower_height[6] = 0.291
    stable = mdp.double_balance_stable_from_values(
        position,
        velocity,
        lower_offset,
        lower_height,
        tilt,
        lower_speed,
        ball_radius=TOP_BALL_RADIUS,
    )
    torch.testing.assert_close(
        stable, torch.tensor([True, False, False, False, False, False, False])
    )


def test_success_timer_is_continuous_not_a_one_step_jackpot(monkeypatch):
    env = SimpleNamespace(
        num_envs=2,
        device="cpu",
        step_dt=1.0,
        episode_length_buf=torch.tensor([2, 2]),
    )
    states = iter(
        (
            torch.tensor([True, True]),
            torch.tensor([True, False]),
            torch.tensor([True, True]),
        )
    )
    monkeypatch.setattr(mdp, "double_balance_stable", lambda *args, **kwargs: next(states))
    assert not mdp.double_balance_success(env, stable_duration_s=3.0).any()
    assert not mdp.double_balance_success(env, stable_duration_s=3.0).any()
    result = mdp.double_balance_success(env, stable_duration_s=3.0)
    torch.testing.assert_close(result, torch.tensor([True, False]))


def test_official_play_config_is_deterministic_and_unassisted():
    cfg = make_microduck_double_balance_env_cfg(play=True)
    assert cfg.actions["ball_hold"].levels == (0.0,)
    twist = cfg.commands["twist"]
    assert twist.ranges.lin_vel_x == (0.0, 0.0)
    assert twist.ranges.lin_vel_y == (0.0, 0.0)
    assert twist.ranges.ang_vel_z == (0.0, 0.0)
    assert twist.rel_standing_envs == 1.0
    assert cfg.curriculum == {}
    assert set(cfg.events) == {
        "reset_double_balance",
        "reset_action_history",
        "expand_bam_friction_fields",
    }
    assert "push_robot" not in cfg.events
    assert cfg.observations["actor"].terms["body_command"].noise is None
    assert cfg.events["reset_double_balance"].params["top_ball_xy_noise"] == 0.0

    metrics = cfg.metrics
    assert metrics["double_balance_success"].reduce == "last"
    assert metrics["double_balance_success"].params["stable_duration_s"] == SUCCESS_STABLE_DURATION_S
    assert metrics["double_balance_stable_fraction"].params["require_unassisted"] is True


def test_real_manager_resolves_frozen_dimensions_and_finite_terms():
    cfg = make_microduck_double_balance_env_cfg(play=True)
    cfg.scene.num_envs = 1
    env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
    obs, _ = env.reset(seed=20260920)
    assert obs["actor"].shape == (1, ACTOR_OBSERVATION_DIM)
    assert obs["critic"].shape == (1, CRITIC_OBSERVATION_DIM)
    assert torch.isfinite(obs["actor"]).all()
    assert torch.isfinite(obs["critic"]).all()

    obs, reward, terminated, truncated, _ = env.step(torch.zeros(1, 14))
    assert torch.isfinite(obs["actor"]).all()
    assert torch.isfinite(obs["critic"]).all()
    assert torch.isfinite(reward).all()
    assert torch.isfinite(terminated).all()
    assert torch.isfinite(truncated).all()
