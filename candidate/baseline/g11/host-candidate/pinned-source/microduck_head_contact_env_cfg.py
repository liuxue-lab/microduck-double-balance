"""Independent Stage 11 STATIC prototype. Not registered and not runnable.

Only configuration/Scene/semantic audit is released. Real crown boundaries,
contact acceptance and material convergence remain blockers. No runner config.
Old tray configuration, task registration and checkpoint handling are unchanged.
"""
from copy import deepcopy
from functools import partial

from mjlab.managers import EventTermCfg
from mjlab_microduck.robot.head_contact import make_head_contact_top_ball_spec
from mjlab_microduck.robot.head_reference import make_head_reference_robot_spec
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_double_balance_env_cfg import make_microduck_double_balance_env_cfg


def make_microduck_head_contact_env_cfg(*, static_audit_only=False):
    if static_audit_only is not True:
        mdp.stage11_head_release_block()
    cfg = deepcopy(make_microduck_double_balance_env_cfg(play=True, physics_dt=.002))
    cfg.scene.num_envs = 1
    robot, ball = cfg.scene.entities["robot"], cfg.scene.entities["top_ball"]
    robot.spec_fn = partial(make_head_reference_robot_spec, robot.spec_fn)
    ball.spec_fn = partial(make_head_contact_top_ball_spec, ball.spec_fn)
    spawn = cfg.events.pop("reset_double_balance")
    spawn.func = mdp.reset_stage11_head_reference
    for key in ("tray_site_name", "top_ball_radius", "top_ball_xy_noise", "top_ball_clearance"):
        spawn.params.pop(key)
    cfg.events = {
        "stage11_release_block": EventTermCfg(func=mdp.stage11_head_release_block, mode="startup"),
        "reset_stage11_head_reference": spawn, **cfg.events,
    }
    actor = cfg.observations["actor"].terms["body_command"]
    actor.func, actor.params = mdp.stage11_head_actor_state, {}
    critic = cfg.observations["critic"].terms["top_ball_state"]
    critic.func, critic.params = mdp.stage11_head_critic_state, {}
    cfg.rewards["double_balance"].func = mdp.stage11_head_task_reward
    cfg.rewards["double_balance"].params = {}
    cfg.rewards["top_ball_speed"].func = mdp.stage11_head_speed_l2
    cfg.rewards["top_ball_speed"].params = {}
    normal = cfg.rewards.pop("tray_tilt")
    normal.func, normal.params = mdp.stage11_head_normal_tilt_l2, {}
    cfg.rewards["head_surface_normal_tilt"] = normal
    # Do not turn the old tray rectangle or mesh AABB into a claimed head rim.
    cfg.terminations["top_ball_lost"].func = mdp.stage11_head_release_block
    cfg.terminations["top_ball_lost"].params = {}
    cfg.metrics.pop("double_balance_stable_fraction")
    cfg.metrics.pop("double_balance_success")
    for name, func in (
        ("top_ball_center_error_m", mdp.stage11_head_center_error),
        ("top_ball_relative_speed_m_s", mdp.stage11_head_relative_speed),
    ):
        cfg.metrics[name].func, cfg.metrics[name].params = func, {}
    return cfg
