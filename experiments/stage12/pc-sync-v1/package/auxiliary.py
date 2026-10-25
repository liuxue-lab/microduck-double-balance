"""Capture hidden delay and RNG state outside the timed environment step."""
import hashlib
import json


def tensor_record(tensor):
    if tensor is None:
        return None
    data = tensor.detach().contiguous().cpu()
    return dict(shape=list(data.shape), dtype=str(data.dtype),
                sha256=hashlib.sha256(data.numpy().tobytes()).hexdigest())


def snapshot(groups, torch):
    rows = []
    for group in groups:
        delay = group._delay_buffer
        if delay is None:
            rows.append(None)
            continue
        ring = delay._buffer
        rows.append(dict(lags=tensor_record(delay._current_lags), steps=tensor_record(delay._step_count),
            phases=tensor_record(delay._phase_offsets), history=tensor_record(ring._buffer),
            pushes=tensor_record(ring._num_pushes), indices=tensor_record(ring._all_indices),
            capacity=tensor_record(ring._max_len_tensor), pointer=ring._pointer,
            configuration=[delay.min_lag, delay.max_lag, delay.batch_size, delay.per_env,
                           delay.hold_prob, delay.update_period, delay.per_env_phase]))
    return dict(groups=rows, cpu_rng=tensor_record(torch.get_rng_state()),
                cuda_rng=[tensor_record(s) for s in torch.cuda.get_rng_state_all()])


def compare(first, second, steps=160):
    count = 0
    with first.open() as a, second.open() as b:
        for index in range(steps + 1):
            x, y = a.readline(65537), b.readline(65537)
            if not x.endswith('\n') or not y.endswith('\n') or max(len(x), len(y)) > 65536:
                raise RuntimeError('Missing delay/RNG comparison row')
            left, right = json.loads(x), json.loads(y)
            if left.get('step') != index or right.get('step') != index:
                raise RuntimeError('Delay/RNG comparison step order differs')
            if left != right:
                raise RuntimeError('Delay history or RNG state differs at control step ' + str(index))
            count += 1
        if a.read(1) or b.read(1):
            raise RuntimeError('Extra delay/RNG comparison rows')
    return dict(passed=True, snapshots=count, exact_delay_history_and_rng=True,
                scope='Initial state and after every control step; tensor bytes and integer counters')
