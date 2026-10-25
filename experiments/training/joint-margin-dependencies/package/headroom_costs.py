"""Draft qpos-side shaping; array math works on NumPy and Torch tensors.

Only this algebra has been exercised here. It is not a safety controller or
a certificate of constraint satisfaction. Actual nominal bounds remain the
acceptance criterion at every 2 ms physics step.
"""
import math

MARGIN_RAD = math.radians(2.5)
LOOKAHEAD_SECONDS = 0.1
POSITION_WEIGHT = -2.0
OUTWARD_WEIGHT = -0.25


def costs(q, qdot, hard_bounds, margin=MARGIN_RAD,
          lookahead_seconds=LOOKAHEAD_SECONDS):
    """Return two nonnegative per-environment costs, summed over joints.

    q/qdot: (..., joints), hard_bounds: (..., joints, 2), in rad and rad/s.
    Outward velocity is priced only inside the proximity band (or outside).
    The 0.1 s factor is a shaping scale, not a maximum joint-speed rating.
    Nonfinite input propagates, to be rejected by the runner's NaN guard.
    """
    if not math.isfinite(margin) or margin <= 0:
        raise ValueError('margin must be finite and positive')
    if not math.isfinite(lookahead_seconds) or lookahead_seconds <= 0:
        raise ValueError('lookahead_seconds must be finite and positive')
    if q.shape != qdot.shape or q.ndim < 1 or hard_bounds.shape[-1] != 2:
        raise ValueError('joint array shape mismatch')
    if hard_bounds.shape[:-1] != q.shape:
        raise ValueError('hard bounds must match each measured joint')
    lower, upper = hard_bounds[..., 0], hard_bounds[..., 1]
    lower_proximity = ((lower + margin - q) / margin).clip(0.0, None)
    upper_proximity = ((q - upper + margin) / margin).clip(0.0, None)
    position = (lower_proximity + upper_proximity).sum(-1)
    outward = (lower_proximity.clip(0.0, 1.0) * (-qdot).clip(0.0, None)
               + upper_proximity.clip(0.0, 1.0) * qdot.clip(0.0, None))
    outward = (outward * (lookahead_seconds / margin)).sum(-1)
    return position, outward
