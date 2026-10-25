"""Separate G11 nominal head task. No registry or installed source is edited.

Released only to the bounded integration worker after the startup geometry
check. Training/random spawn and direct checkpoint resume remain blocked.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import numpy as np
import torch
from mjlab.managers import EventTermCfg, TerminationTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg
from mjlab_microduck.tasks import mdp
from mjlab_microduck.tasks.microduck_head_contact_env_cfg import make_microduck_head_contact_env_cfg
from mjlab_microduck.robot import head_contact as head
from formal_geometry import HeadSurface
from runtime_accounting import require, save


def unique(model, kind, suffix):
    n = getattr(model, 'n'+kind)
    ids = [i for i in range(n) if getattr(model, kind)(i).name.split('/')[-1] == suffix]
    require(len(ids) == 1, 'Missing/ambiguous '+kind+': '+suffix)
    return ids[0]


def startup(env, env_ids=None):
    model, data = env.sim.mj_model, env.sim.mj_data
    require(env.num_envs == 1 and env.physics_dt == .002 and env.step_dt == .020
            and env.cfg.decimation == 10, 'Formal cadence/world scope differs')
    require((model.nq, model.nv, model.nu, model.ngeom) == (35, 32, 14, 85),
            'Geometry roster no longer matches accepted physics')
    require(not any('double_balance_tray' in model.body(i).name for i in range(model.nbody)),
            'Tray body still present')
    hg, bg = unique(model, 'geom', head.HEAD_GEOM_NAME), unique(model, 'geom', head.TOP_GEOM)
    sid = unique(model, 'site', 'stage11_head_material_frame')
    require((hg, bg) == (49, 84), 'Accepted contact pair index changed')
    allowed = [i for i in range(model.ngeom) if head.can_contact(model, bg, i)]
    require(set(allowed) == {0, hg}, 'Top ball collision mask differs')
    require(abs(float(model.geom_size[bg, 0])-.020) < 1e-12, 'Ball radius changed')
    require(np.allclose(model.geom_solref[bg], [.003, 1.], rtol=0, atol=1e-12), 'Head material changed')
    require(np.allclose(model.geom_friction[bg], [.6, .005, .0001], rtol=0, atol=1e-12), 'Friction changed')
    vertices, faces = head.mesh_triangles(model, data, hg)
    rotation = data.site_xmat[sid].reshape(3, 3)
    local = (vertices-data.site_xpos[sid]) @ rotation
    surface = HeadSurface(local[faces], head.closest_on_triangles,
                          head.vertical_surface_height, head.sphere_center_above)
    witness = surface.measure(surface.spawn)
    require(witness['valid'] and not witness['lost'] and
            abs(witness['surface_gap_m']-.0005) < 1e-9, 'Nominal mesh placement invalid')
    env._g11_surface = surface
    env._g11_tick = 0
    env._g11_surface_cache = None
    env._g11_stable_time = torch.zeros(1, device=env.device)
    env._g11_forced_failure = torch.zeros(1, dtype=torch.bool, device=env.device)
    env._g11_events = torch.zeros((4, 1), dtype=torch.int64, device=env.device)
    env._g11_interval_resets = torch.zeros(1, dtype=torch.int64, device=env.device)
    env._g11_interval_fires = torch.zeros(1, dtype=torch.int64, device=env.device)
    env._g11_startup_report = dict(
        geometry='actual triangles transformed to audited material frame',
        triangle_count=len(faces), triangle_sha256=surface.sha256,
        recoverable_envelope='upper envelope of actual mesh projection, disk dilation by R=0.020m; lower depth R=0.020m',
        rectangular_or_AABB_boundary=False, nominal_only=True,
        spawn=surface.spawn.tolist(), witness=witness,
        contact_pair=[hg, bg], collision_partners=allowed,
        material_support_tau_s=.0025, top_contact_tau_s=.003,
        derived_state_policy='accepted_endpoint', training_released=False)


def reset_nominal(env, env_ids, **params):
    require(all(v == 0. for k in ('xy_noise','tilt_noise_deg','joint_noise','ball_vel_noise')
                for v in [params.get(k, 0.)]) and
            tuple(params.get('yaw_range', (0., 0.))) == (0., 0.), 'Random spawn remains blocked')
    require(params.get('ball_radius', .12) == .12 and params.get('root_height', .125) == .125,
            'Nominal lower placement changed')
    if len(env_ids) == 0:
        return
    require(env_ids.tolist() == [0], 'Only one native world is released')
    origins = env.scene.env_origins[env_ids]
    q = torch.zeros((len(env_ids), 4), device=env.device); q[:, 0] = 1.
    velocity = torch.zeros((len(env_ids), 6), device=env.device)
    lower, robot, upper = env.scene['ball'], env.scene['robot'], env.scene['top_ball']
    p = origins + origins.new_tensor([0., 0., .121])
    lower.write_root_link_pose_to_sim(torch.cat((p, q), -1), env_ids=env_ids)
    lower.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
    state = mdp._bb_state(env)
    if not hasattr(state, 'anchor'):
        state.anchor = torch.zeros((env.num_envs, 3), device=env.device)
    state.anchor[env_ids] = p
    state.hold[env_ids] = 0.; state.level[env_ids] = 0
    rp = origins + origins.new_tensor([0., 0., .368])
    robot.write_root_link_pose_to_sim(torch.cat((rp, q), -1), env_ids=env_ids)
    robot.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
    joints = robot.data.default_joint_pos[env_ids].clone()
    robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)
    env.sim.forward()
    sites, _ = robot.find_sites('^stage11_head_material_frame$')
    require(len(sites) == 1, 'Head reference missing')
    pose = robot.data.site_pose_w[env_ids, sites[0]]
    local = pose.new_tensor(env._g11_surface.spawn).unsqueeze(0)
    top_position = pose[:, :3] + mdp._quat_rotate(pose[:, 3:7], local)
    upper.write_root_link_pose_to_sim(torch.cat((top_position, q), -1), env_ids=env_ids)
    upper.write_root_link_velocity_to_sim(velocity, env_ids=env_ids)
    env._g11_tick = 0
    env._g11_stable_time[env_ids] = 0.
    env._g11_forced_failure[env_ids] = False
    env._g11_surface_cache = None


def measured_surface(env):
    pos, rate, spin = mdp._stage11_head_kinematics(env)
    require(bool(torch.isfinite(pos).all() and torch.isfinite(rate).all() and
                 torch.isfinite(spin).all()), 'Nonfinite raw head state')
    # Read on every call: no stale cross-reset/same-clock cache is permitted.
    rows = [env._g11_surface.measure(p) for p in pos.detach().cpu().numpy()]
    return pos, rate, rows


def top_lost(env):
    _, _, rows = measured_surface(env)
    return torch.tensor([r['lost'] for r in rows], dtype=torch.bool, device=env.device)


def stable(env):
    position, rate, rows = measured_surface(env)
    # Keep all original numerical limits; measure height against the real mesh.
    p = position.clone()
    p[:, 2] = p.new_tensor([r['surface_distance_m'] for r in rows])
    robot, ball = env.scene['robot'], env.scene['ball']
    root, lower = robot.data.root_link_pos_w, ball.data.root_link_pos_w
    value = mdp.double_balance_stable_from_values(
        p, rate, torch.linalg.vector_norm(root[:, :2]-lower[:, :2], dim=-1),
        root[:, 2]-lower[:, 2], mdp.wrestle_tilt(env, 'robot'),
        torch.linalg.vector_norm(ball.data.root_link_lin_vel_w[:, :2], dim=-1))
    value &= ~torch.tensor([r['lost'] for r in rows], dtype=torch.bool, device=env.device)
    value &= mdp._bb_state(env).hold <= 1e-6
    return value


def success(env):
    env._g11_stable_time[env.episode_length_buf <= 1] = 0.
    env._g11_stable_time = torch.where(stable(env), env._g11_stable_time+env.step_dt,
                                     torch.zeros_like(env._g11_stable_time))
    # Deliberately retain the old float32 5 s boundary; no epsilon/rounding fix.
    return env._g11_stable_time >= 5.


def step_probe(env, env_ids):
    env._g11_events[0] += 1


def local_probe(env, env_ids):
    env._g11_events[1, env_ids] += 1


class LocalProbe:
    def __init__(self, cfg, env):
        self.env = env
    def reset(self, env_ids=None):
        self.env._g11_interval_resets[env_ids] += 1
    def __call__(self, env, env_ids):
        env._g11_events[2, env_ids] += 1
        env._g11_interval_fires[env_ids] += 1


def global_probe(env, env_ids):
    require(env_ids is None, 'Global interval must fire exactly once')
    env._g11_events[3] += 1


def forced_failure(env):
    return env._g11_forced_failure.clone()


class Recorder(RecorderTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.entries = []
        self.closed = False
    def entry(self, hook, env_ids=None):
        env = self._env
        self.entries.append(dict(hook=hook, ids=None if env_ids is None else env_ids.tolist(),
            sim_counter=env._sim_step_counter, control_counter=env.common_step_counter,
            episode=env.episode_length_buf.detach().cpu().tolist(),
            qpos=env.sim.data.qpos.clone(), time=env.sim.data.time.clone(),
            action=env.action_manager.action.clone(),
            observations={k: v.clone() for k, v in env.obs_buf.items()},
            reward=getattr(env, 'reward_buf', torch.zeros(1, device=env.device)).clone()))
    def record_pre_reset(self, env_ids):
        self.entry('pre_reset', env_ids)
    def record_post_reset(self, env_ids):
        self.entry('post_reset', env_ids)
    def record_post_step(self):
        self.entry('post_step')
    def close(self):
        self.closed = True


def make_cfg():
    cfg = make_microduck_head_contact_env_cfg(static_audit_only=True)
    require('stage11_release_block' in cfg.events and
            cfg.terminations['top_ball_lost'].func is mdp.stage11_head_release_block,
            'Original prototype release guards changed')
    # A new isolated candidate replaces the guard with executable startup
    # checks; original source, prototype and task registry remain untouched.
    cfg.events['stage11_release_block'] = EventTermCfg(func=startup, mode='startup')
    cfg.events['reset_stage11_head_reference'].func = reset_nominal
    cfg.terminations['top_ball_lost'] = TerminationTermCfg(func=top_lost)
    cfg.metrics['double_balance_stable_fraction'] = MetricsTermCfg(func=stable, reduce='mean')
    cfg.metrics['double_balance_success'] = MetricsTermCfg(func=success, reduce='last')
    cfg.events['g11_step_probe'] = EventTermCfg(func=step_probe, mode='step')
    cfg.events['g11_local_probe'] = EventTermCfg(func=local_probe, mode='interval', interval_range_s=(.02, .02))
    cfg.events['g11_class_probe'] = EventTermCfg(func=LocalProbe, mode='interval', interval_range_s=(.02, .02))
    cfg.events['g11_global_probe'] = EventTermCfg(func=global_probe, mode='interval', interval_range_s=(.02, .02), is_global_time=True)
    cfg.terminations['g11_fault_injection'] = TerminationTermCfg(func=forced_failure)
    cfg.recorders['g11_record'] = RecorderTermCfg(func=Recorder)
    cfg.seed = 1111003
    cfg.auto_reset = False
    cfg.sim.nconmax = 256; cfg.sim.njmax = 1024
    require(cfg.episode_length_s == 10. and cfg.decimation == 10 and
            cfg.actions['ball_hold'].levels == (0.0,), 'Task timing or assistance changed')
    # play=True preserves four inherited actor UniformNoiseCfg terms. Keep
    # their amplitudes and group flags unchanged; v2 binds private RNG on the
    # manager's actual deep-copied terms before the first public reset.
    return cfg
