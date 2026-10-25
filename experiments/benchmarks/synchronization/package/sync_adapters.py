"""Reversible inference-only adapters for two measured synchronization sites."""
import copy
import importlib
from pathlib import Path

from deploy_contract import require, sha
from phase_cache import Bindings

CHECK_CALLS = 40
SOURCE_PINS = {
    'mjlab.utils.lab_api.math': '47757748386173cc4ded5fb5dc5619336393ee6296769fa8b985738ebe50d436',
    'mjlab.entity.data': 'fd1116692f82278bda721d44c015d27e253d2e866608c8e12824df039dfb4657',
    'mjlab.actuator.actuator': 'ed57149b527c529b37b6322f20cac3eeed89b7355847ad712c5d44b13549f1ab',
    'mjlab.utils.buffers.delay_buffer': '845e68da9e681adee1e93f90b4b1937c282a27f82a147f1fe3c19d83204049b5',
    'mjlab.utils.buffers.circular_buffer': '6a96cb79afbfb2d1234c3ee8e7531db371262673ddc509da7f3e66410ef2ae13',
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


def append_fixed(buffer, data, torch):
    """Retain backfill/reset behavior with GPU selection, without a CPU branch."""
    if data.shape[0] != buffer._batch_size:
        raise ValueError(f'Expected batch size {buffer._batch_size}, got {data.shape[0]}')
    data = data.to(buffer._device)
    if buffer._buffer is None:
        buffer._pointer = -1
        buffer._buffer = torch.empty((buffer._max_len, *data.shape), dtype=data.dtype, device=buffer._device)
    buffer._pointer = (buffer._pointer + 1) % buffer._max_len
    buffer._buffer[buffer._pointer] = data
    mask = (buffer._num_pushes == 0).reshape((1, buffer._batch_size) + (1,) * (data.ndim - 1))
    buffer._buffer.copy_(torch.where(mask, data.unsqueeze(0), buffer._buffer))
    buffer._num_pushes += 1


def get_fixed(buffer, key, torch):
    """Read the same lag, always returning owned data rather than a live view."""
    if buffer._buffer is None:
        raise RuntimeError('Buffer not initialized. Call append() first.')
    if isinstance(key, int):
        key = torch.full((buffer._batch_size,), key, dtype=torch.long, device=buffer._device)
    else:
        if key.ndim == 0:
            key = key.expand(buffer._batch_size)
        key = key.to(device=buffer._device, dtype=torch.long)
    if key.numel() != buffer._batch_size:
        raise ValueError(f'Expected {buffer._batch_size} lags, got {key.numel()}')
    pushes = buffer._num_pushes.clamp_min(1)
    valid = torch.minimum(key, pushes - 1).clamp_min(0)
    index = torch.remainder(buffer._pointer - valid, buffer._max_len)
    return buffer._buffer[index, buffer._all_indices]


def compare_delay(torch, a, b):
    for name in ('_current_lags', '_step_count', '_phase_offsets'):
        exact(torch, getattr(a, name), getattr(b, name), 'delay.' + name)
    for name in ('_buffer', '_num_pushes', '_all_indices', '_max_len_tensor'):
        exact(torch, getattr(a._buffer, name), getattr(b._buffer, name), 'ring.' + name)
    require(a._buffer._pointer == b._buffer._pointer, 'Delay history pointer differs')


class FixedDelay(Bindings):
    def __init__(self, group, torch, modules):
        super().__init__()
        self.group, self.torch, self.delay = group, torch, group._delay_buffer
        actuator = modules['mjlab.actuator.actuator']
        require(type(group).__name__ == 'FrictionDRBamActuator' and group.command_field == 'position',
                'Expected original BAM position delay consumer')
        require(getattr(group.apply_delay, '__func__', None) is actuator.Actuator.apply_delay,
                'Unreviewed delay call chain')
        require(type(self.delay) is modules['mjlab.utils.buffers.delay_buffer'].DelayBuffer and
                type(self.delay._buffer) is modules['mjlab.utils.buffers.circular_buffer'].CircularBuffer,
                'Unreviewed delay buffer type')
        require(self.delay.generator is None and self.delay.batch_size == 1, 'Expected original one-world default RNG')
        require(not any(n in vars(self.delay) for n in ('append', 'compute', '_update_lags', '_sample_lags')),
                'Delay instance already modified')
        self.original_apply = group.apply_delay
        self.shadow = copy.deepcopy(self.delay)
        self.original_update = self.delay._update_lags.__func__
        self.original_sample = self.delay._sample_lags.__func__
        self.config = self.configuration()
        self.calls = self.checked_calls = self.appends = self.computes = 0
        self.rng_comparisons = self.output_scalars = 0
        try:
            self.set(self.delay, 'append', self.append)
            self.set(self.delay, 'compute', self.compute)
            self.set(group, 'apply_delay', self.apply_delay)
        except BaseException:
            self.close()
            raise

    def configuration(self):
        d = self.delay
        return (d.min_lag, d.max_lag, d.batch_size, str(d.device), d.per_env, d.hold_prob,
                d.update_period, d.per_env_phase, id(d.generator), d._buffer.max_length)

    def append(self, data):
        append_fixed(self.delay._buffer, data, self.torch)
        self.appends += 1

    def compute(self):
        d, t = self.delay, self.torch
        if not d.is_initialized:
            raise RuntimeError('Buffer not initialized. Call append() first.')
        # Original RNG function and its call order are unchanged.
        d._update_lags()
        valid_lags = t.minimum(d._current_lags, d._buffer.current_length - 1).clamp_min(0)
        result = get_fixed(d._buffer, valid_lags, t)
        self.computes += 1
        return result

    def apply_delay(self, command):
        d, t = self.delay, self.torch
        require(self.group._delay_buffer is d and self.configuration() == self.config and
                d._update_lags.__func__ is self.original_update and d._sample_lags.__func__ is self.original_sample,
                'Delay configuration or RNG routine changed')
        checking = self.checked_calls < CHECK_CALLS
        device = command.position_target.device
        require(command.position_target.dtype == t.float32 and not command.position_target.requires_grad,
                'Expected original FP32 position target')
        if checking:
            cpu_before = t.get_rng_state()
            gpu_before = t.cuda.get_rng_state(device) if device.type == 'cuda' else None
        answer = self.original_apply(command)
        if checking:
            cpu_after = t.get_rng_state()
            gpu_after = t.cuda.get_rng_state(device) if device.type == 'cuda' else None
            devices = [device] if device.type == 'cuda' else []
            # Independent shadow consumes exactly the same draws; fork_rng restores
            # the real post-candidate RNG even if shadow validation raises.
            with t.random.fork_rng(devices=devices):
                t.set_rng_state(cpu_before)
                if gpu_before is not None:
                    t.cuda.set_rng_state(gpu_before, device)
                self.shadow.append(command.position_target)
                expected = self.shadow.compute()
                exact(t, expected, answer.position_target, 'delayed position command')
                compare_delay(t, self.shadow, d)
                exact(t, cpu_after, t.get_rng_state(), 'CPU RNG consumption')
                if gpu_after is not None:
                    exact(t, gpu_after, t.cuda.get_rng_state(device), 'CUDA RNG consumption')
            self.checked_calls += 1
            self.rng_comparisons += 1
            self.output_scalars += expected.numel()
        self.calls += 1
        return answer

    def stats(self):
        return dict(calls=self.calls, appends=self.appends, computes=self.computes,
                    checked_calls=self.checked_calls, rng_comparisons=self.rng_comparisons,
                    output_scalars=self.output_scalars, original_delay_sampling=True,
                    original_reset_and_set_lags=True, comparison='exact FP32 bits and integer state',
                    closed=not self.saved)


class FixedDelays(Bindings):
    def __init__(self, groups, torch):
        super().__init__()
        names = ('mjlab.actuator.actuator', 'mjlab.utils.buffers.delay_buffer', 'mjlab.utils.buffers.circular_buffer')
        modules, self.sources = verify_sources(names)
        self.runners = []
        try:
            for group in groups:
                require(group._delay_buffer is not None, 'Expected active original command delay')
                self.runners.append(FixedDelay(group, torch, modules))
        except BaseException:
            self.close()
            raise

    def close(self):
        for runner in reversed(self.runners):
            runner.close()
        super().close()

    def stats(self):
        return dict(groups=[r.stats() for r in self.runners], sources=self.sources,
                    closed=all(not r.saved for r in self.runners))
