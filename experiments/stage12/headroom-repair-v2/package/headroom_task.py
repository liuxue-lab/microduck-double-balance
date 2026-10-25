"""Offline-prepared Stage12 head-roll margin overlay. No training entry point.

Requires the pinned project's original speed_task/native_task and dependencies.
The original 61D actor, 85D critic, actuator, observation/DR/delay, geometry,
nominal ranges, solver, 50/500 Hz and unfiltered policy all remain unchanged.
GPU environment construction, smoke and learning have NOT been run.
"""
import math

from headroom_costs import (costs, MARGIN_RAD, POSITION_WEIGHT, OUTWARD_WEIGHT)


def _measured_head_roll(env):
    robot = env.scene['robot']
    cache = getattr(env, '_s12_headroom_mapping', None)
    if cache is None or cache[0] is not robot:
        ids, names = robot.find_joints('^head_roll$')
        if len(ids) != 1 or list(names) != ['head_roll']:
            raise RuntimeError('Expected exactly one named head_roll joint')
        hard = robot.data.joint_pos_limits[:, ids]
        soft = robot.data.soft_joint_pos_limits[:, ids]
        lo, hi = hard[..., 0], hard[..., 1]
        expected = math.radians(25.0)
        if not bool(((abs(lo + expected) < 1e-6)
                     & (abs(hi - expected) < 1e-6)).all()):
            raise RuntimeError('Original head_roll nominal range changed')
        if not bool(((abs(soft[..., 0] - (lo + MARGIN_RAD)) < 1e-6)
                     & (abs(soft[..., 1] - (hi - MARGIN_RAD)) < 1e-6)).all()):
            raise RuntimeError('Original head_roll soft range changed')
        if not bool((hi - lo > 2 * MARGIN_RAD).all()):
            raise RuntimeError('Proximity bands overlap')
        cache = (robot, ids)
        env._s12_headroom_mapping = cache
    ids = cache[1]
    return (robot.data.joint_pos[:, ids], robot.data.joint_vel[:, ids],
            robot.data.joint_pos_limits[:, ids])


def head_roll_position_cost(env):
    return costs(*_measured_head_roll(env))[0]


def head_roll_outward_cost(env):
    return costs(*_measured_head_roll(env))[1]


def make_cfg(num_envs=4096, seed=2026102301):
    from mjlab.managers import RewardTermCfg
    from speed_task import make_cfg as make_selected_stage12_cfg
    cfg = make_selected_stage12_cfg(num_envs, seed)
    # A new training horizon; the original source file remains read-only.
    cfg.episode_length_s = 60.0
    for name in ('head_roll_margin', 'head_roll_outward'):
        if name in cfg.rewards:
            raise RuntimeError('Refusing to replace an existing reward: ' + name)
    cfg.rewards['head_roll_margin'] = RewardTermCfg(
        func=head_roll_position_cost, weight=POSITION_WEIGHT)
    cfg.rewards['head_roll_outward'] = RewardTermCfg(
        func=head_roll_outward_cost, weight=OUTWARD_WEIGHT)
    return cfg
