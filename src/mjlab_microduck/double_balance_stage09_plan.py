"""User-approved Stage 09 pilot contract; importing this never trains."""
from __future__ import annotations

import math
from copy import deepcopy

BASELINE_COMMIT = 'a76019399a4d6a2946ccb6a5713095ec7dbca717'
PRIMARY_SHA = '86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97'
SEED = 20261001
NUM_ENVS = 4096
SOURCE_UPDATES = 5000
MAX_UPDATES = 500
PROFILES = {'F0': (0., 0.), 'F1': (-.25, 0.), 'F2': (0., -.25), 'F3': (-.25, -.25)}
TERMS = ('stage09_head_yaw', 'stage09_head_margin')
JOINTS = ('neck_pitch', 'head_pitch', 'head_yaw', 'head_roll')
BUFFER_FRACTION = .05
POSTURE_DEG = (15., 5., 5.)


def progress(completed):
    if type(completed) is not int or not 0 <= completed <= MAX_UPDATES:
        raise ValueError('Stage 09 branch progress must be 0..500')
    total = SOURCE_UPDATES + completed
    return dict(experiment_completed_updates=completed, lineage_completed_updates=total,
                last_completed_iteration=total-1, next_iteration=total,
                common_step_counter=total*24, sim_step_counter=total*240,
                expected_adam_steps=total*20, experiment_transitions=completed*NUM_ENVS*24)


def profile_record(name):
    yaw, margin = PROFILES[name]
    return dict(name=name, base='E', yaw_weight=yaw, margin_weight=margin,
                buffer_fraction=BUFFER_FRACTION, head_joint_names=list(JOINTS))


def apply_profile(cfg, name):
    from mjlab.managers import RewardTermCfg
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.double_balance_stage08_plan import apply_training_profile
    result = apply_training_profile(cfg, 'E')
    for term, weight, function in zip(TERMS, PROFILES[name],
            (mdp.stage09_head_yaw_cost, mdp.stage09_head_margin_cost)):
        if term in result.rewards:
            raise ValueError('Stage 09 overlay applied twice')
        if weight:
            result.rewards[term] = RewardTermCfg(func=function, weight=weight)
    return result


def budget(gpu):
    if gpu != '5090':
        raise ValueError('Only cloud RTX 5090 is authorized for Stage 09 PPO')
    return dict(gpu='5090', maximum_cloud_hours=4., finalization_reserve_hours=.75,
                maximum_updates_per_profile=500, maximum_total_updates=2000)


def budget_decision(gpu, *, spent_seconds, additional_updates, seconds_per_update, overhead_seconds=0):
    contract = budget(gpu)
    if type(additional_updates) is not int or additional_updates < 0:
        raise ValueError('Invalid update request')
    if not all(math.isfinite(v) for v in (spent_seconds, seconds_per_update, overhead_seconds)) or min(spent_seconds,overhead_seconds)<0 or seconds_per_update<=0:
        raise ValueError('Invalid budget clock or estimate')
    remaining = (contract['maximum_cloud_hours']-contract['finalization_reserve_hours'])*3600-spent_seconds
    needed = additional_updates*seconds_per_update+overhead_seconds
    return dict(admit=needed<=remaining, remaining_seconds=max(0.,remaining), requested_seconds=needed)
