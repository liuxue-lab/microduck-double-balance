"""Offline two-world candidate. No simulator or Data publication is implemented.

Scalar dt is a cohort contract, not a claim that the native Model supports
different timesteps per world. All active rows validate before state commits.
Ground Stage11's original coordinate mapping, arithmetic and bounds are kept.
"""
from dataclasses import dataclass
import numbers
import numpy as np

Q = np.array([0, 1, 2, *range(7, 21), 21, 22, 23, 28, 29, 30])
V = np.array([0, 1, 2, *range(6, 20), 20, 21, 22, 26, 27, 28])
OTHER = np.array([3, 4, 5, 6, 24, 25, 26, 27, 31, 32, 33, 34])
WEIGHTS = np.array([1, 2, 2, 1], dtype=np.float64) / 6


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def array(value, shape, dtype, label):
    require(isinstance(value, np.ndarray) and value.shape == shape
            and value.dtype == np.dtype(dtype) and np.isfinite(value).all(),
            label + ': shape, dtype or finiteness differs')
    return value


def timestep(dt):
    require(isinstance(dt, numbers.Real) and not isinstance(dt, (bool, np.bool_))
            and np.ndim(dt) == 0 and np.isfinite(dt) and dt > 0,
            'One finite positive scalar cohort dt required')
    return float(dt)


@dataclass(frozen=True)
class Snapshot:
    value: np.ndarray
    expected: np.ndarray
    enabled: np.ndarray
    steps: np.ndarray


class BatchPositionAccumulator:
    def __init__(self, qpos, enabled):
        q = array(qpos, (1, 35), np.float32, 'qpos')
        self.value = q[:, Q].astype(np.float64)
        self.expected = q[:, Q].copy()
        self.enabled = array(enabled, (1,), np.bool_, 'enabled').copy()
        self.steps = np.zeros(1, dtype=np.int64)

    def snapshot(self):
        return Snapshot(*(v.copy() for v in
                          (self.value, self.expected, self.enabled, self.steps)))

    def restore(self, state):
        require(isinstance(state, Snapshot), 'Whole-cohort snapshot required')
        array(state.value, (1, 23), np.float64, 'snapshot value')
        array(state.expected, (1, 23), np.float32, 'snapshot expected')
        array(state.enabled, (1,), np.bool_, 'snapshot enabled')
        array(state.steps, (1,), np.int64, 'snapshot steps')
        require((state.steps >= 0).all() and
                np.array_equal(state.value[state.enabled].astype(np.float32),
                               state.expected[state.enabled]),
                'Snapshot compensation consistency differs')
        # Validate first; preserve the accumulator's array bindings on restore.
        for destination, source in zip(
                (self.value, self.expected, self.enabled, self.steps),
                (state.value, state.expected, state.enabled, state.steps)):
            destination[...] = source

    def advance(self, before, stages, native, dt, active):
        array(before, (1, 35), np.float32, 'before')
        array(native, (1, 35), np.float32, 'native')
        array(stages, (4, 1, 32), np.float32, 'stages')
        array(active, (1,), np.bool_, 'active')
        dt = timestep(dt)
        require(np.array_equal(before[:, Q], self.expected),
                'Accumulator/physical start mismatch')
        require(before[~active].tobytes() == native[~active].tobytes(),
                'Inactive rows changed outside this candidate')
        next_value = self.value.copy()
        next_expected = self.expected.copy()
        next_steps = self.steps.copy()
        result = native.copy()
        rows = []
        for w in range(1):
            if not active[w]:
                rows.append(dict(world=w, active=False))
                continue
            require(self.steps[w] < np.iinfo(np.int64).max, 'Step counter overflow')
            velocity = stages[:, w, :][:, V].astype(np.float64)
            # Per-world order intentionally matches the pinned scalar oracle.
            next_value[w] += dt * (WEIGHTS @ velocity)
            target = next_value[w].astype(np.float32)
            proposal = target.astype(float) - native[w, Q].astype(float)
            scale = np.maximum.reduce(
                [abs(before[w, Q]), abs(native[w, Q]), abs(target)])
            bound = (2 * np.spacing(scale).astype(float) +
                     8 * np.finfo(np.float32).eps * dt * np.max(abs(velocity), axis=0))
            require(all(np.isfinite(v).all() for v in (next_value[w], target, proposal, bound)),
                    'Accumulation overflow')
            if self.enabled[w]:
                require(np.all(abs(proposal) <= bound),
                        'World ' + str(w) + ' exceeded original correction bound')
                result[w, Q] = target
            next_expected[w] = result[w, Q]
            next_steps[w] += 1
            rows.append(dict(world=w, active=True,
                position_compensation_enabled=bool(self.enabled[w]),
                position_accumulator=next_value[w].copy(), native_qpos=native[w].copy(),
                position_correction_mixed_units=proposal if self.enabled[w] else np.zeros(23),
                position_correction_bound_mixed_units=bound,
                shadow_native_error_mixed_units=native[w, Q].astype(float)-next_value[w],
                accumulated_steps=int(next_steps[w])))
        require(result[:, OTHER].tobytes() == native[:, OTHER].tobytes(),
                'Quaternion bytes changed')
        # No mutation occurs until every active row has passed.
        self.value[...] = next_value
        self.expected[...] = next_expected
        self.steps[...] = next_steps
        return result, rows


def batch_stage_position(accumulator, start, previous_velocity, native,
                         dt, stage, coordinates):
    array(accumulator, (1, 23), np.float64, 'stage accumulator')
    array(start, (1, 35), np.float32, 'stage start')
    array(previous_velocity, (1, 32), np.float32, 'previous velocity')
    array(native, (1, 35), np.float32, 'native stage')
    dt = timestep(dt)
    require(type(stage) is int and stage in (1, 2, 3), 'Stage must be 1, 2 or 3')
    require(isinstance(coordinates, tuple) and len(coordinates) == 1
            and all(c in (None, 'upper', 'all') for c in coordinates),
            'Declare per-world coordinate scope explicitly')
    result, rows = native.copy(), []
    for w, scope in enumerate(coordinates):
        if scope is None:
            rows.append(dict(world=w, coordinates=None))
            continue
        require(np.array_equal(accumulator[w].astype(np.float32), start[w, Q]),
                'Stage accumulator/start mismatch')
        a = .5 if stage in (1, 2) else 1.
        velocity = previous_velocity[w, V].astype(np.float64)
        exact = accumulator[w] + dt * a * velocity
        rounded = exact.astype(np.float32)
        selected = np.arange(23) if scope == 'all' else np.arange(20, 23)
        result[w, Q[selected]] = rounded[selected]
        delta = result[w, Q].astype(float) - native[w, Q].astype(float)
        scale = np.maximum.reduce([abs(start[w, Q]), abs(native[w, Q]), abs(rounded)])
        bound = (2 * np.spacing(scale).astype(float) +
                 8 * np.finfo(np.float32).eps * dt * abs(velocity))
        require(all(np.isfinite(v).all() for v in (exact, rounded, delta, bound))
                and np.all(abs(delta) <= bound), 'Stage correction exceeds original bound')
        untouched = np.setdiff1d(np.arange(35), Q[selected])
        require(result[w, untouched].tobytes() == native[w, untouched].tobytes(),
                'Unselected stage coordinates changed')
        rows.append(dict(world=w, stage=stage, coordinates=scope,
            selected_qpos_indices=Q[selected], native_qpos=native[w].copy(),
            shadow_position=exact, correction_mixed_units=delta, bound_mixed_units=bound))
    return result, rows


def assess_cohort(coarse, fine, floors, assess_one):
    """Aggregate the pinned original assessor; never publish or splice Data rows.

The caller supplies the unchanged scalar assessor and explicit predeclared
floors. This helper introduces neither a schedule nor another integration tier.
"""
    require(len(coarse) == len(fine) == len(floors) == 1, 'One owned world decision required')
    require(all(type(f) is int and f >= 0 for f in floors), 'Explicit floors required')
    for trials in (coarse, fine):
        require(all(type(t['resolution']) is int and t['resolution'] > 0 for t in trials)
                and len(trials) == 1,
                'Only uniform-resolution cohorts are represented')
    rows = []
    for w in range(1):
        original = assess_one(coarse[w], fine[w])
        numeric_ok = original['accepted']
        floor_ok = fine[w]['resolution'] >= floors[w]
        rows.append(dict(world=w, original=original, numerical_accepted=numeric_ok,
            resolution_floor=floors[w], resolution_floor_satisfied=floor_ok,
            floor_only_rejection=numeric_ok and not floor_ok,
            accepted=numeric_ok and floor_ok))
    accepted = all(r['accepted'] for r in rows)
    return dict(worlds=rows, accepted=accepted,
                action='commit_whole_cohort' if accepted else 'rollback_whole_cohort',
                rejected_worlds=[r['world'] for r in rows if not r['accepted']],
                row_splicing_permitted=False, simulator_publication_implemented=False)
