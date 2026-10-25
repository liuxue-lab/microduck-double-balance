"""Tiny real-Torch transition check; no simulator, model, optimizer or PPO."""
import math
from joint_statistics import JointStatistics


def check_statistics_runtime(torch,device='cpu'):
    with torch.inference_mode(False):
        q=torch.zeros((2,14),device=device)
        limits=torch.tensor([[-math.radians(25),math.radians(25)]]*14,device=device)
        stats=JointStatistics(torch,q,limits,['joint-'+str(i) for i in range(13)]+['head_roll'])
    with torch.inference_mode():
        q[1,13]=math.radians(27)
        stats.observe(q)
        stats.after_control_step(torch.tensor([False,True],device=device),torch.tensor([False,True],device=device))
        q[:]=0;stats.observe(q)
    # This is the actual runner transition that the NumPy double cannot model.
    first=stats.take_rollout(2)
    if first['head_roll_violation_fraction']!=.25 or first['ended_episodes_with_joint_violation']!=1:
        raise RuntimeError('Physical statistics lost a pre-reset violation')
    for _ in range(3):
        with torch.inference_mode():
            for _ in range(240):stats.observe(q)
        clean=stats.take_rollout()
        if clean['any_joint_violation_fraction']!=0:
            raise RuntimeError('Rollout reset retained a previous-window violation')
    return dict(status='TORCH_STATISTICS_TRANSITIONS_PASSED',torch_version=torch.__version__,
        device=str(device),inference_rollouts=4,physics_integrations=0,PPO_updates=0,
        actual_robot_acceptance=False)
