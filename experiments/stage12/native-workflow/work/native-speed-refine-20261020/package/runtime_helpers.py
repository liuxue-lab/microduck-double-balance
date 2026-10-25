import json
import math
from pathlib import Path
from common import require

class JsonScalarWriter:
    """Dependency-free fallback; logging must not require another download."""
    def __init__(self,path):
        self.file=Path(path).open('a',buffering=1)
    def add_scalar(self,tag,value,global_step=None,**kwargs):
        value=float(value)
        self.file.write(json.dumps(dict(tag=tag,value=value if math.isfinite(value) else None,
            step=global_step),ensure_ascii=False,allow_nan=False)+'\n')
    def flush(self):self.file.flush()
    def close(self):self.file.close()

def require_finite_group(torch, values, message):
    """One host read on the success path; still check every input each call."""
    checks = [torch.isfinite(v).all() for v in values]
    if checks:
        require(bool(torch.stack(checks).all().item()), message)

def require_step_outputs(torch, observations, rewards, dones, nan_state):
    """Check before returning to PPO; never delay detection to batch end."""
    checks = [~nan_state.any()]
    checks.extend(torch.isfinite(v).all() for v in observations.values())
    checks.extend((torch.isfinite(rewards).all(), torch.isfinite(dones).all()))
    if not bool(torch.stack(checks).all().item()):
        # Additional reads only on the exceptional path, for a useful reason.
        require(not bool(nan_state.any().item()), 'Nonfinite simulator state detected; preserve, no automatic retry')
        raise RuntimeError('Nonfinite observations, rewards or dones')

def host_scalar_means(torch, values):
    """Retain float32 mean semantics; transfer each device group together."""
    groups = {}
    result = [None]*len(values)
    for i, value in enumerate(values):
        scalar = torch.as_tensor(value).detach().float().mean()
        groups.setdefault(str(scalar.device), []).append((i, scalar))
    for group in groups.values():
        host = torch.stack([value for _, value in group]).cpu().tolist()
        for (i, _), value in zip(group, host):
            result[i] = value
    return result
