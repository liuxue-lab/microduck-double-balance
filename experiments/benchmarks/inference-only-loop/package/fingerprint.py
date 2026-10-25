"""State fingerprint used by the already validated Stage12 observer."""
import hashlib


def fingerprint(tensors):
    h=hashlib.sha256()
    for name,tensor in sorted(tensors.items()):
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str((str(value.dtype),tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()
