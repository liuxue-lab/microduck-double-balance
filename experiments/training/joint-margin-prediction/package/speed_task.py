"""One reward-only overlay; the original native task remains byte-identical."""
import torch
from mjlab_microduck.tasks import mdp
from native_task import make_cfg as original_make_cfg,head_scale

from reward_spec import REWARD_SPEC

def speed_aligned_head_reward(env):
    position,rate,_=mdp._stage11_head_kinematics(env)
    top=mdp.double_balance_top_score_from_values(position,rate,velocity_std=.08)
    lower=mdp.double_balance_lower_score(env)
    lower_velocity=env.scene['ball'].data.root_link_lin_vel_w[:,:2]
    lower_speed_factor=torch.exp(-torch.sum(lower_velocity.square(),dim=-1)/(.15*.15))
    return head_scale(env)*torch.nan_to_num(top*lower*lower_speed_factor,nan=0.,posinf=0.,neginf=0.)

def make_cfg(num_envs=4096,seed=20261023):
    cfg=original_make_cfg(num_envs,seed)
    cfg.rewards['double_balance'].func=speed_aligned_head_reward
    return cfg
