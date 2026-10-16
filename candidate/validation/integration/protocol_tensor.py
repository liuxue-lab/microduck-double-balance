"""NumPy development double; never represented as real Torch/TensorDict coverage."""
import numpy as np


class Tensor(np.ndarray):
    def clone(self): return self.copy().view(Tensor)
    def fill_(self, value): self.fill(value); return self
    def long(self): return self.astype(np.int64).view(Tensor)
    def to(self, dtype):
        return self if isinstance(dtype, str) else self.astype(dtype).view(Tensor)
    def detach(self): return self
    def cpu(self): return self


class Torch:
    bool, float32, float64, int32, int64 = np.bool_, np.float32, np.float64, np.int32, np.int64
    @staticmethod
    def tensor(values, dtype=None, device=None):
        return np.array(values, dtype=dtype).view(Tensor)
    @staticmethod
    def zeros(*shape): return np.zeros(shape, dtype=np.float32).view(Tensor)
    @staticmethod
    def isfinite(v): return np.isfinite(v).view(Tensor)
    @staticmethod
    def logical_or(a, b): return np.logical_or(a, b).view(Tensor)
    @staticmethod
    def logical_and(a, b): return np.logical_and(a, b).view(Tensor)
    @staticmethod
    def logical_not(a): return np.logical_not(a).view(Tensor)
    @staticmethod
    def cat(rows, dim=0): return np.concatenate(rows, axis=dim).view(Tensor)


class TD(dict):
    # Match the overlapping tensor-looking API that caused integration-v1 to fail.
    @property
    def dtype(self): return None
    @property
    def shape(self): return self.batch_size
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return list(iter(self))
    def __init__(self, values, batch_size):
        super().__init__(values)
        self.batch_size = tuple(batch_size)
    def clone(self): return TD({k: v.clone() for k, v in self.items()}, list(self.batch_size))
    def __iter__(self):
        # TD iteration is over batches, not keys; catches the v6 set(obs) bug.
        for i in range(self.batch_size[0]):
            yield {k: v[i] for k, v in self.items()}
