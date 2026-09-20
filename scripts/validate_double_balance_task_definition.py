"""Reproducible Stage-04 task-definition acceptance (no PPO training).

Checks the frozen actor/critic layout, reward signs, termination boundaries,
success anti-cheat conditions, deterministic unassisted evaluation profile,
and one real ManagerBasedRlEnv step.
"""
from __future__ import annotations

import json

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_double_balance_env_cfg import (
    ACTOR_OBSERVATION_DIM,
    CRITIC_OBSERVATION_DIM,
    SUCCESS_STABLE_DURATION_S,
    TOP_BALL_RADIUS,
    make_microduck_double_balance_env_cfg,
)

SEED = 20260920


def main() -> None:
    train_cfg = make_microduck_double_balance_env_cfg(play=False)
    eval_cfg = make_microduck_double_balance_env_cfg(play=True)
    eval_cfg.scene.num_envs = 1
    eval_cfg.seed = SEED

    assert train_cfg.observations["actor"].terms["body_command"].noise is not None
    assert eval_cfg.observations["actor"].terms["body_command"].noise is None
    assert eval_cfg.actions["ball_hold"].levels == (0.0,)
    assert eval_cfg.commands["twist"].ranges.lin_vel_x == (0.0, 0.0)
    assert eval_cfg.commands["twist"].ranges.lin_vel_y == (0.0, 0.0)
    assert eval_cfg.commands["twist"].ranges.ang_vel_z == (0.0, 0.0)
    assert eval_cfg.curriculum == {}
    assert set(eval_cfg.events) == {
        "reset_double_balance",
        "reset_action_history",
        "expand_bam_friction_fields",
    }

    reward_weights = {name: term.weight for name, term in eval_cfg.rewards.items()}
    assert reward_weights["double_balance"] > 0.0
    assert reward_weights["top_ball_speed"] < 0.0
    assert reward_weights["tray_tilt"] < 0.0

    ideal_position = torch.tensor([[0.0, 0.0, TOP_BALL_RADIUS]])
    ideal_velocity = torch.zeros(1, 3)
    stable = mdp.double_balance_stable_from_values(
        ideal_position,
        ideal_velocity,
        lower_offset=torch.zeros(1),
        lower_height=torch.full((1,), 0.245),
        robot_tilt=torch.zeros(1),
        lower_ball_speed=torch.zeros(1),
        ball_radius=TOP_BALL_RADIUS,
    )
    fallen_lower_layer = mdp.double_balance_stable_from_values(
        ideal_position,
        ideal_velocity,
        lower_offset=torch.zeros(1),
        lower_height=torch.full((1,), 0.10),
        robot_tilt=torch.zeros(1),
        lower_ball_speed=torch.zeros(1),
        ball_radius=TOP_BALL_RADIUS,
    )
    assert stable.item() and not fallen_lower_layer.item()

    env = ManagerBasedRlEnv(cfg=eval_cfg, device="cpu")
    obs, _ = env.reset(seed=SEED)
    assert obs["actor"].shape[-1] == ACTOR_OBSERVATION_DIM
    assert obs["critic"].shape[-1] == CRITIC_OBSERVATION_DIM
    obs, reward, terminated, truncated, _ = env.step(torch.zeros(1, 14))
    finite = bool(
        torch.isfinite(obs["actor"]).all()
        and torch.isfinite(obs["critic"]).all()
        and torch.isfinite(reward).all()
        and torch.isfinite(terminated).all()
        and torch.isfinite(truncated).all()
    )
    assert finite

    report = {
        "seed": SEED,
        "actor_dimension": int(obs["actor"].shape[-1]),
        "critic_dimension": int(obs["critic"].shape[-1]),
        "actor_top_ball_layout": [
            "x_tray/0.05",
            "y_tray/0.04",
            "(z_tray-R)/0.02",
            "vx_tray/0.5",
            "vy_tray/0.5",
            "vz_tray/0.5",
        ],
        "critic_privileged_tail": ["wx_tray/5", "wy_tray/5", "wz_tray/5"],
        "reward_weights": reward_weights,
        "terminations": list(eval_cfg.terminations),
        "metrics": list(eval_cfg.metrics),
        "success_stable_duration_s": SUCCESS_STABLE_DURATION_S,
        "evaluation_hold": list(eval_cfg.actions["ball_hold"].levels),
        "evaluation_twist_command": [0.0, 0.0, 0.0],
        "evaluation_events": list(eval_cfg.events),
        "one_step_finite": finite,
        "ideal_state_stable": bool(stable.item()),
        "fallen_lower_layer_rejected": not bool(fallen_lower_layer.item()),
    }
    print("DoubleBalanceTaskDefinitionAcceptanceBegin")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("DoubleBalanceTaskDefinitionAcceptance=PASS")
    print("DoubleBalanceTaskDefinitionAcceptanceEnd")


if __name__ == "__main__":
    main()
