"""Previously bit-exact fixed-shape quaternion adapter; no delay replacement."""
import importlib
from pathlib import Path

from deploy_contract import require, sha
from phase_cache import Bindings

CHECK_CALLS = 40
SOURCE_PINS = {
    'mjlab.utils.lab_api.math': '47757748386173cc4ded5fb5dc5619336393ee6296769fa8b985738ebe50d436',
    'mjlab.entity.data': 'fd1116692f82278bda721d44c015d27e253d2e866608c8e12824df039dfb4657',
}


def verify_sources(names):
    modules, rows = {}, {}
    for name in names:
        module = importlib.import_module(name)
        path = Path(module.__file__)
        require(sha(path) == SOURCE_PINS[name], 'Unreviewed synchronization adapter source: ' + name)
        modules[name] = module
        rows[name] = dict(path=str(path), sha256=SOURCE_PINS[name])
    return modules, rows


def exact(torch, a, b, label):
    require(a.shape == b.shape and a.dtype == b.dtype and a.device == b.device, 'Schema differs: ' + label)
    left, right = a.detach().contiguous(), b.detach().contiguous()
    if a.dtype == torch.float32:
        left, right = left.view(torch.int32), right.view(torch.int32)
    require(bool(torch.equal(left, right)), 'Exact same-input comparison failed: ' + label)


class StaticQuaternion(Bindings):
    def __init__(self, torch):
        super().__init__()
        modules, self.sources = verify_sources(('mjlab.utils.lab_api.math', 'mjlab.entity.data'))
        self.torch = torch
        entity, math = modules['mjlab.entity.data'], modules['mjlab.utils.lab_api.math']
        self.original = entity.quat_from_matrix
        require(self.original is math.quat_from_matrix, 'Original entity quaternion binding changed')
        from static_math import quat_from_matrix
        self.calculate = quat_from_matrix
        self.calls = self.checked_calls = self.scalar_comparisons = 0
        self.set(entity, 'quat_from_matrix', self.convert)

    def convert(self, matrix):
        require(matrix.dtype == self.torch.float32 and not matrix.requires_grad,
                'Fixed-shape quaternion adapter requires original FP32 inference inputs')
        value = self.calculate(matrix)
        if self.checked_calls < CHECK_CALLS:
            expected = self.original(matrix)
            exact(self.torch, expected, value, 'quaternion')
            self.checked_calls += 1
            self.scalar_comparisons += value.numel()
        self.calls += 1
        return value

    def stats(self):
        return dict(calls=self.calls, checked_calls=self.checked_calls,
                    scalar_comparisons=self.scalar_comparisons, sources=self.sources,
                    comparison='exact FP32 bit pattern', original_arithmetic_order=True,
                    cached_kinematics=False, compiler_or_precision_change=False,
                    closed=not self.saved)

