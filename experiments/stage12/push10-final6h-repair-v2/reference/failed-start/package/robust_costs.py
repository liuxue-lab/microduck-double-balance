"""Nonnegative measured-state costs for the final bounded disturbance fine-tune.

No action/state clamp, filtering, displacement target or changed physical limit.
The algebra works with either NumPy arrays or Torch tensors for offline checks.
"""
import math

MARGIN_RAD = math.radians(3.0)
LOOKAHEAD_SECONDS = .08
EXCESS_SLOPE = 4.0
POSITION_WEIGHT = -2.0
OUTWARD_WEIGHT = -.25
FAILURE_MAGNITUDE = 100.0
FAILURE_WEIGHT = -1.0
PRIORITY_JOINTS = ('left_hip_yaw', 'right_hip_yaw', 'right_hip_roll')
SERVO_JOINTS = ('left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee',
                'left_ankle', 'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
                'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee',
                'right_ankle')
OTHER_JOINTS = tuple(name for name in SERVO_JOINTS if name != 'head_roll')


def joint_importance(names):
    if len(names) != len(set(names)) or set(names) != set(OTHER_JOINTS):
        raise ValueError('Expected the thirteen named servos except head_roll')
    # Retain the full pressure on the three demonstrated failures; introduce
    # only a small new tax on joints that already recovered successfully.
    return tuple(2.0 if name in PRIORITY_JOINTS else .25 for name in names)


def proximity(q, bounds, margin=MARGIN_RAD):
    if q.ndim < 1 or bounds.shape != q.shape + (2,):
        raise ValueError('Nominal bounds must match measured joint positions')
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError('The qpos proximity margin must be finite and positive')
    return ((bounds[..., 0] + margin-q)/margin).clip(0., None) + \
           ((q-bounds[..., 1] + margin)/margin).clip(0., None)


def costs(q, qdot, bounds, importance, margin=MARGIN_RAD,
          lookahead_seconds=LOOKAHEAD_SECONDS):
    if q.shape != qdot.shape or importance.shape != (q.shape[-1],):
        raise ValueError('Measured joint velocity or importance shape differs')
    if not math.isfinite(lookahead_seconds) or lookahead_seconds <= 0:
        raise ValueError('Predictive horizon must be finite and positive')
    p = proximity(q, bounds, margin)
    predicted = proximity(q + lookahead_seconds*qdot, bounds, margin)
    position = ((p**2 + EXCESS_SLOPE*(p-1.).clip(0., None))*importance).sum(-1)
    outward = ((predicted**2-p**2).clip(0., None)*importance).sum(-1)
    return position, outward
