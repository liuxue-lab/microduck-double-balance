"""Test original source against adapters on CPU; no CUDA performance claims."""
import ast
import copy
import dataclasses
import importlib.util
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'package'))
import bootstrap
bootstrap.load_base(Path(os.environ.get('MICRODUCK_REFERENCE_PACKAGE', str(bootstrap.BASE_PACKAGE))))
import static_math
import sync_adapters as adapters

FIX = Path(__file__).parent / 'fixtures'
torch.set_num_threads(1)


def original_module(name, file, only=None, **values):
    path = FIX / file
    module = ModuleType(name)
    module.__file__ = str(path)
    sys.modules[name] = module
    module.__dict__.update(torch=torch, dataclasses=dataclasses, dataclass=dataclasses.dataclass, **values)
    tree = ast.parse(path.read_bytes())
    tree.body = [n for n in tree.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
    if only:
        tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in only]
        for cls in tree.body:
            if only[cls.name] is not None:
                cls.bases = []
                cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in only[cls.name]]
    tree.body.insert(0, ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0))
    exec(compile(ast.fix_missing_locations(tree), str(path), 'exec'), module.__dict__)
    return module


CIRCULAR = original_module('sync_original_circular', 'circular-buffer-v1.3.0.py')
DELAY = original_module('sync_original_delay', 'delay-buffer-v1.3.0.py', CircularBuffer=CIRCULAR.CircularBuffer)
ACT = original_module('sync_original_actuator', 'actuator-v1.3.0.py',
                      only={'ActuatorCmd': None, 'Actuator': ('apply_delay',)})
spec = importlib.util.spec_from_file_location('sync_original_math', FIX / 'mjlab-math-v1.3.0.py')
MATH = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = MATH
spec.loader.exec_module(MATH)
MODULES = {'mjlab.actuator.actuator': ACT, 'mjlab.utils.buffers.delay_buffer': DELAY,
           'mjlab.utils.buffers.circular_buffer': CIRCULAR}


def bits_equal(a, b):
    return torch.equal(a.contiguous().view(torch.int32), b.contiguous().view(torch.int32))


class QuaternionTests(unittest.TestCase):
    def test_original_sources_match_reviewed_hashes(self):
        for name, module in {**MODULES, 'mjlab.utils.lab_api.math': MATH}.items():
            self.assertEqual(bootstrap.sha(Path(module.__file__)), adapters.SOURCE_PINS[name])

    def test_random_rotations_exact_bits_and_shape(self):
        g = torch.Generator().manual_seed(461)
        q = torch.randn(16384, 4, generator=g)
        q /= torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        matrices = MATH.matrix_from_quat(q).reshape(64, 256, 3, 3)
        for value in (matrices, matrices[:1, :1], matrices[0, 0], matrices[:, ::2], matrices[:0]):
            a, b = MATH.quat_from_matrix(value), static_math.quat_from_matrix(value)
            self.assertEqual(a.shape, b.shape)
            self.assertTrue(bits_equal(a, b))

    def test_ties_pi_rotations_and_near_singular_inputs(self):
        qs = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 0., 0., 1.],
                           [.5, .5, .5, .5], [.5, -.5, .5, -.5], [1e-7, 1., 0., 0.]])
        rotations = MATH.matrix_from_quat(qs)
        inputs = torch.cat((rotations, torch.zeros_like(rotations), rotations + 1e-7))
        self.assertTrue(bits_equal(MATH.quat_from_matrix(inputs), static_math.quat_from_matrix(inputs)))

    def test_positive_sqrt_preserves_zero_negative_nan_semantics(self):
        values = torch.tensor([-float('inf'), -1., -0., 0., 1e-38, 1., float('inf'), float('nan')])
        self.assertTrue(bits_equal(MATH._sqrt_positive_part(values), static_math.sqrt_positive(values)))

    def test_nonrotation_and_nonfinite_inputs_preserve_output_bits(self):
        x = torch.randn(512, 3, 3, generator=torch.Generator().manual_seed(61))
        x[0, 0, 0] = float('nan')
        x[1, 1, 1] = float('inf')
        x[2, 2, 2] = -float('inf')
        self.assertTrue(bits_equal(MATH.quat_from_matrix(x), static_math.quat_from_matrix(x)))

    def test_missing_or_changed_source_is_rejected(self):
        with patch.object(adapters.importlib, 'import_module', return_value=MATH), \
             patch.object(adapters, 'sha', return_value='wrong'):
            with self.assertRaisesRegex(RuntimeError, 'Unreviewed'):
                adapters.verify_sources(('mjlab.utils.lab_api.math',))


class RingTests(unittest.TestCase):
    def test_ring_wrap_backfill_partial_reset_and_clamped_lags(self):
        for length in (1, 2, 4, 9):
            a = CIRCULAR.CircularBuffer(length, 3, 'cpu')
            b = copy.deepcopy(a)
            for step in range(50):
                if step in (7, 24):
                    a.reset([1]); b.reset([1])
                data = torch.arange(42, dtype=torch.float32).reshape(3, 14) + step * .125
                a.append(data); adapters.append_fixed(b, data, torch)
                self.assertTrue(bits_equal(a._buffer, b._buffer))
                self.assertEqual(a._pointer, b._pointer)
                self.assertTrue(torch.equal(a._num_pushes, b._num_pushes))
                for key in (0, 1, 99, -5, torch.tensor([0, 2, 7])):
                    self.assertTrue(bits_equal(a[key], adapters.get_fixed(b, key, torch)))

    def test_nonfinite_history_and_owned_zero_lag_result(self):
        a = CIRCULAR.CircularBuffer(3, 1, 'cpu')
        b = copy.deepcopy(a)
        data = torch.tensor([[float('nan'), float('inf'), -0.]])
        a.append(data); adapters.append_fixed(b, data, torch)
        output = adapters.get_fixed(b, 0, torch)
        self.assertTrue(bits_equal(a[0], output))
        self.assertNotEqual(output.untyped_storage().data_ptr(), b._buffer.untyped_storage().data_ptr())
        snapshot = output.clone()
        for _ in range(4): adapters.append_fixed(b, torch.ones_like(data), torch)
        self.assertTrue(bits_equal(output, snapshot))


def group_for(delay):
    cls = type('FrictionDRBamActuator', (), {'apply_delay': ACT.Actuator.apply_delay})
    group = cls()
    group.command_field = 'position'
    group._delay_buffer = delay
    return group


def command(step):
    target = torch.arange(14, dtype=torch.float32).reshape(1, 14) * .025 + step
    return ACT.ActuatorCmd(position_target=target, velocity_target=torch.zeros_like(target),
                           effort_target=torch.zeros_like(target), pos=torch.ones_like(target), vel=torch.zeros_like(target))


class DelayTests(unittest.TestCase):
    def test_original_rng_and_all_hidden_state_match_across_configurations(self):
        for period, hold, low, high, per_env in ((0, 0., 0, 3, True), (3, .25, 1, 4, True),
                                                 (2, 1., 0, 3, True), (0, 0., 0, 0, False)):
            torch.manual_seed(812)
            first = DELAY.DelayBuffer(min_lag=low, max_lag=high, batch_size=1,
                                      update_period=period, hold_prob=hold, per_env=per_env)
            a, b = group_for(first), group_for(copy.deepcopy(first))
            runner = adapters.FixedDelay(b, torch, MODULES)
            try:
                for step in range(70):
                    if step in (8, 45):
                        before = torch.get_rng_state()
                        a._delay_buffer.reset([0]); after = torch.get_rng_state()
                        torch.set_rng_state(before); b._delay_buffer.reset([0])
                        self.assertTrue(torch.equal(after, torch.get_rng_state()))
                        # Shadow follows the same explicitly controlled reset.
                        with torch.random.fork_rng(devices=[]):
                            torch.set_rng_state(before); runner.shadow.reset([0])
                    if step == 12:
                        for d in (a._delay_buffer, b._delay_buffer, runner.shadow): d.set_lags(torch.tensor([high]))
                    cmd = command(step)
                    before = torch.get_rng_state()
                    expected = a.apply_delay(cmd); after = torch.get_rng_state()
                    torch.set_rng_state(before)
                    actual = b.apply_delay(cmd)
                    self.assertTrue(torch.equal(after, torch.get_rng_state()))
                    self.assertTrue(bits_equal(expected.position_target, actual.position_target))
                    adapters.compare_delay(torch, a._delay_buffer, b._delay_buffer)
                self.assertEqual(runner.calls, 70)
                self.assertEqual(runner.checked_calls, 40)
                self.assertEqual(runner.rng_comparisons, 40)
            finally:
                runner.close()
            self.assertFalse(runner.saved)
            self.assertIs(b.apply_delay.__func__, ACT.Actuator.apply_delay)
            self.assertNotIn('append', vars(b._delay_buffer))

    def test_failure_restores_post_candidate_rng_and_bindings(self):
        group = group_for(DELAY.DelayBuffer(max_lag=3))
        runner = adapters.FixedDelay(group, torch, MODULES)
        original = runner.shadow.compute
        runner.shadow.compute = lambda: original() + 1.
        before = torch.get_rng_state()
        # Expected draw count from the original sampling method, with no hooks.
        plain = DELAY.DelayBuffer(max_lag=3)
        torch.set_rng_state(before)
        plain.append(command(0).position_target); plain.compute()
        expected_after = torch.get_rng_state()
        torch.set_rng_state(before)
        try:
            with self.assertRaisesRegex(RuntimeError, 'Exact same-input'):
                group.apply_delay(command(0))
            self.assertTrue(torch.equal(expected_after, torch.get_rng_state()))
        finally:
            runner.close()
        self.assertNotIn('apply_delay', vars(group))

    def test_unreviewed_rng_and_configuration_changes_rejected(self):
        group = group_for(DELAY.DelayBuffer(generator=torch.Generator()))
        with self.assertRaisesRegex(RuntimeError, 'default RNG'):
            adapters.FixedDelay(group, torch, MODULES)
        group = group_for(DELAY.DelayBuffer())
        runner = adapters.FixedDelay(group, torch, MODULES)
        try:
            group._delay_buffer.hold_prob = .8
            with self.assertRaisesRegex(RuntimeError, 'configuration'):
                group.apply_delay(command(0))
        finally:
            runner.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
