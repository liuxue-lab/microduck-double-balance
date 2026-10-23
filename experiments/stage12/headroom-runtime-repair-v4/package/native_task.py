"""Stage12 training task: native mjlab stepping and the basketball PPO recipe.

This is a new training configuration, not a release of the historical strict
RK4/contact audit. Native MuJoCo convex-mesh collision semantics apply. Head
training metrics are explicitly named geometric surrogates, not old G11 proofs.
"""
from copy import deepcopy
from functools import partial
import math
import numpy as np
import torch
from mjlab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg, CurriculumTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_double_balance_env_cfg import make_microduck_double_balance_env_cfg
from mjlab_microduck.robot.head_reference import make_head_reference_robot_spec, REFERENCE_SITE
from mjlab_microduck.robot import head_contact as head

HOLD_LEVELS = (0.,)
FOUNDATION_UPDATES = 0
HEAD_RAMP_UPDATES = 1000

def head_scale(env):
    update = env.common_step_counter / 24
    return 0. if update < FOUNDATION_UPDATES else min(1., .25 + .75 * (update-FOUNDATION_UPDATES) / HEAD_RAMP_UPDATES)

def startup_geometry(env, env_ids=None):
    """One-time model construction geometry, without stepping or evaluation."""
    model, data = env.sim.mj_model, env.sim.mj_data
    geom = next(i for i in range(model.ngeom) if model.geom(i).name.split('/')[-1] == head.HEAD_GEOM_NAME)
    site = next(i for i in range(model.nsite) if model.site(i).name.split('/')[-1] == REFERENCE_SITE)
    vertices, faces = head.mesh_triangles(model, data, geom)
    local = (vertices-data.site_xpos[site]) @ data.site_xmat[site].reshape(3, 3)
    center, _, _, _ = head.sphere_center_above((0., 0.), local[faces], radius=.020, clearance=.0005)
    env._s12_spawn = torch.tensor(center, dtype=torch.float32, device=env.device)
    env._s12_stable_ticks = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._s12_success_ever = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._s12_nan_total = 0
    bb = mdp._bb_state(env)
    bb.level.zero_(); bb.hold.zero_()  # Retained Stage08 policy starts unassisted.

def reset_head(env, env_ids, **params):
    if len(env_ids) == 0:
        return
    # Use the upstream noisy lower-stack reset; upper placement is rigidly
    # transformed from the fresh head frame, with no tangent-plane xy noise.
    mdp.reset_basketball(env, env_ids, **params)
    env.sim.forward()
    robot, upper = env.scene['robot'], env.scene['top_ball']
    ids, _ = robot.find_sites('^'+REFERENCE_SITE+'$')
    if len(ids) != 1:
        raise RuntimeError('Missing unique head reference')
    pose = robot.data.site_pose_w[env_ids, ids[0]]
    local = env._s12_spawn.expand(len(env_ids), 3)
    pos = pose[:, :3] + mdp._quat_rotate(pose[:, 3:7], local)
    q = torch.zeros(len(env_ids), 4, device=env.device); q[:, 0] = 1.
    upper.write_root_link_pose_to_sim(torch.cat((pos, q), -1), env_ids=env_ids)
    upper.write_root_link_velocity_to_sim(torch.zeros(len(env_ids), 6, device=env.device), env_ids=env_ids)
    env._s12_stable_ticks[env_ids] = 0
    env._s12_success_ever[env_ids] = False

def top_lost(env):
    p, _, _ = mdp._stage11_head_kinematics(env)
    # A broad training termination envelope, NOT an asserted mesh crown rim.
    lost = (torch.linalg.vector_norm(p[:, :2], dim=-1) > .09) | (p[:, 2] < -.02) | (p[:, 2] > .15)
    return lost & (env.common_step_counter >= FOUNDATION_UPDATES*24)

def head_reward(env):
    return head_scale(env)*mdp.stage11_head_task_reward(env)

def head_actor_observation(env):
    value = mdp.stage11_head_actor_state(env)
    return torch.zeros_like(value) if head_scale(env) == 0. else value

def head_critic_observation(env):
    value = mdp.stage11_head_critic_state(env)
    return torch.zeros_like(value) if head_scale(env) == 0. else value

def head_speed_cost(env):
    return head_scale(env)*torch.nan_to_num(mdp.stage11_head_speed_l2(env), nan=0., posinf=0., neginf=0.)

def lower_stable(env):
    r, b = env.scene['robot'], env.scene['ball']
    offset = torch.linalg.vector_norm(r.data.root_link_pos_w[:, :2]-b.data.root_link_pos_w[:, :2], dim=-1)
    height = r.data.root_link_pos_w[:, 2]-b.data.root_link_pos_w[:, 2]
    speed = torch.linalg.vector_norm(b.data.root_link_lin_vel_w[:, :2], dim=-1)
    return (offset <= .06) & (height >= .20) & (height <= .29) & (mdp.wrestle_tilt(env, 'robot') <= math.radians(20)) & (speed <= .15)

def lower_unassisted(env):
    return lower_stable(env) & (mdp._bb_state(env).hold <= 1e-6)

def geometric_joint_stable(env):
    p, v, _ = mdp._stage11_head_kinematics(env)
    return lower_unassisted(env) & (torch.linalg.vector_norm(p[:, :2], dim=-1) <= .012) & ((p[:, 2]-.02).abs() <= .006) & (torch.linalg.vector_norm(v, dim=-1) <= .08)

def geometric_success(env):
    # Exactly 250 consecutive 50-Hz ticks; no float-accumulated boundary.
    if not hasattr(env, '_s12_stable_ticks'):
        env._s12_stable_ticks = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        env._s12_success_ever = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    env._s12_stable_ticks = torch.where(geometric_joint_stable(env), env._s12_stable_ticks+1, torch.zeros_like(env._s12_stable_ticks))
    env._s12_success_ever |= env._s12_stable_ticks >= 250
    return env._s12_success_ever

def unassisted_fraction(env):
    return (mdp._bb_state(env).hold <= 1e-6).float()

def make_cfg(num_envs=2048, seed=20261020):
    cfg = deepcopy(make_microduck_double_balance_env_cfg(play=False, physics_dt=.002))
    cfg.scene.num_envs = num_envs
    cfg.seed, cfg.auto_reset = seed, True
    cfg.sim.mujoco.integrator = 'implicitfast'
    # These are the actual flat basketball recipe's finite iteration settings.
    cfg.sim.mujoco.iterations, cfg.sim.mujoco.ls_iterations = 10, 20
    cfg.scene.entities['robot'].spec_fn = partial(make_head_reference_robot_spec, cfg.scene.entities['robot'].spec_fn)
    cfg.scene.entities['top_ball'].spec_fn = partial(head.make_head_contact_top_ball_spec, cfg.scene.entities['top_ball'].spec_fn)
    spawn = cfg.events.pop('reset_double_balance')
    for key in ('tray_site_name', 'top_ball_radius', 'top_ball_xy_noise', 'top_ball_clearance'):
        spawn.params.pop(key, None)
    spawn.func = reset_head
    spawn.params.update(xy_noise=.005, yaw_range=(-math.pi, math.pi), tilt_noise_deg=1., joint_noise=.025, ball_vel_noise=0.)
    cfg.events = {'native_head_geometry': EventTermCfg(func=startup_geometry, mode='startup'), 'reset_native_head': spawn, **cfg.events}
    cfg.events.pop('push_robot', None)  # This run targets stationary balance.
    cfg.actions['ball_hold'].levels = HOLD_LEVELS
    twist = cfg.commands['twist']
    twist.ranges.lin_vel_x = twist.ranges.lin_vel_y = twist.ranges.ang_vel_z = (0., 0.)
    twist.rel_standing_envs, twist.rel_heading_envs = 1., 0.
    twist.rel_turn_in_place_envs = 0.
    actor = cfg.observations['actor'].terms['body_command']
    actor.func, actor.params = head_actor_observation, {}
    critic = cfg.observations['critic'].terms['top_ball_state']
    critic.func, critic.params = head_critic_observation, {}
    cfg.rewards['double_balance'] = RewardTermCfg(func=head_reward, weight=4.)
    cfg.rewards['top_ball_speed'] = RewardTermCfg(func=head_speed_cost, weight=-.10)
    cfg.rewards.pop('tray_tilt')
    cfg.rewards['head_surface_normal_tilt'] = RewardTermCfg(func=mdp.stage11_head_normal_tilt_l2, weight=-.10)
    cfg.rewards['action_rate_l2'].weight = -.4  # Preserve Stage08 E smoothing.
    for event in ('randomize_com','randomize_head_com'):
        cfg.events[event].params['ranges'] = (-.005,.005)  # Stage08 E fixed CoM ranges.
    # Stationary task: no inherited velocity-command or late large smoothing
    # ramps. The retained successful lower-balance policy starts without assistance.
    cfg.curriculum = {}
    cfg.terminations['top_ball_lost'] = TerminationTermCfg(func=top_lost, time_out=False)
    for name in ('double_balance_stable_fraction', 'double_balance_success'):
        cfg.metrics.pop(name, None)
    cfg.metrics['top_ball_center_error_m'].func = mdp.stage11_head_center_error
    cfg.metrics['top_ball_center_error_m'].params = {}
    cfg.metrics['top_ball_relative_speed_m_s'].func = mdp.stage11_head_relative_speed
    cfg.metrics['top_ball_relative_speed_m_s'].params = {}
    for name, func, reduce in (
        ('native_lower_unassisted_fraction', lower_unassisted, 'mean'),
        ('native_unassisted_env_fraction', unassisted_fraction, 'mean'),
        ('native_joint_geometric_stable_fraction', geometric_joint_stable, 'mean'),
        ('native_joint_geometric_success_5s', geometric_success, 'last'),
    ):
        cfg.metrics[name] = MetricsTermCfg(func=func, reduce=reduce)
    cfg.recorders = {}
    return cfg
