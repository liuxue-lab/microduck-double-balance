"""Measured-qpos shaping. No state, action, target, or physical-limit clamps.

This algebra guides PPO; only the unchanged physical-step evaluator accepts
models. NumPy checks do not claim to verify the cloud Torch/GPU runtime.
"""
import math
MARGIN_RAD = math.radians(5.0)
MODEL_SOFT_MARGIN_RAD = math.radians(2.5)
LOOKAHEAD_SECONDS = 0.08
POSITION_WEIGHT = -6.0
OUTWARD_WEIGHT = -0.5
EXCESS_SLOPE = 4.0


def proximity(q, hard_bounds, margin=MARGIN_RAD):
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError('margin must be finite and positive')
    if q.ndim < 1 or hard_bounds.shape != q.shape + (2,):
        raise ValueError('hard bounds must match measured joints')
    lower = ((hard_bounds[..., 0] + margin - q) / margin).clip(0.0, None)
    upper = ((q - hard_bounds[..., 1] + margin) / margin).clip(0.0, None)
    return lower + upper


def costs(q, qdot, hard_bounds, margin=MARGIN_RAD,
          lookahead_seconds=LOOKAHEAD_SECONDS):
    if q.shape != qdot.shape:
        raise ValueError('joint array shape mismatch')
    if not math.isfinite(lookahead_seconds) or lookahead_seconds <= 0:
        raise ValueError('lookahead must be finite and positive')
    p = proximity(q, hard_bounds, margin)
    predicted = proximity(q + lookahead_seconds*qdot, hard_bounds, margin)
    position = (p**2 + EXCESS_SLOPE*(p-1.0).clip(0.0, None)).sum(-1)
    outward = (predicted**2 - p**2).clip(0.0, None).sum(-1)
    return position, outward


def retention_factor(q, hard_bounds):
    """One at/below 20deg, tapering to zero at the unchanged +/-25deg limit."""
    p = proximity(q, hard_bounds)
    if p.shape[-1] != 1:
        raise ValueError('This reward overlay targets exactly the named head_roll')
    return ((1.0-p.clip(0.0, 1.0))**2)[..., 0]
