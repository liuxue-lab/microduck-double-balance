"""CPU fake-integrator checks; never runs robot simulation or PPO."""
from pathlib import Path
import math
import sys
import types
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'package'))
import torch
from force_runtime import VectorPushes


class Raw:
    def __init__(self, n=2):
        ns = types.SimpleNamespace
        self.num_envs, self.device, self.step_dt, self.physics_dt = n, 'cpu', .02, .002
        self.cfg = ns(decimation=10)
        self.scene = dict(robot=ns(indexing=ns(body_ids=torch.tensor([2, 3, 4])),
                                    find_bodies=lambda name: ([0], ['trunk_base'])),
                          ball=ns(indexing=ns(body_ids=torch.tensor([0]))),
                          top_ball=ns(indexing=ns(body_ids=torch.tensor([1]))))
        mass = torch.tensor([[100., 200., .3, .4, .1]]).repeat(n, 1)
        mass[:, 2:] *= torch.arange(1, n+1)[:, None]
        self.sim = ns(data=ns(xfrc_applied=torch.zeros(n, 5, 6)),
                      model=ns(body_mass=mass), mj_model=ns(opt=ns(gravity=[0., 0., -9.81])))
        self.sim.step = self.integrate
        self.calls = 0
        self.applied = []
        self.raise_on_call = None
        self.inject_unknown = False

    def integrate(self):
        self.calls += 1
        force = self.sim.data.xfrc_applied[:, 2, :3]
        if bool((force != 0).any()):
            self.applied.append((self.calls-1, force.clone()))
        if self.inject_unknown:
            self.sim.data.xfrc_applied[0, 0, 5] = .25
        if self.calls == self.raise_on_call:
            raise RuntimeError('fixture integrator failure')
        return self.calls


def control(push, raw, dones=None):
    push.begin_control()
    for _ in range(10):
        # Native scene/action writers run before the sim.step adapter.
        raw.sim.data.xfrc_applied.zero_()
        raw.sim.step()
    push.end_control(torch.zeros(raw.num_envs, dtype=torch.bool) if dones is None else dones)


class ForceRuntimeTests(unittest.TestCase):
    def test_complete_60_seconds_exact_29_pulses_and_mass_scaling(self):
        raw = Raw(); push = VectorPushes(raw, torch, 2026102901, .1, 0.)
        try:
            for step in range(3000):
                control(push, raw, torch.ones(2, dtype=torch.bool) if step == 2999 else None)
            snap = push.snapshot()
            self.assertEqual(snap['physics_calls'], 30000)
            self.assertEqual(snap['completed_pulses'], 58)
            self.assertEqual(snap['applied_world_physics_steps'], 2900)
            self.assertEqual(len(raw.applied), 1450)
            self.assertEqual([x[0] for x in raw.applied[:50]], list(range(1000, 1050)))
            self.assertEqual(raw.applied[-1][0], 29049)
            first = raw.applied[0][1]
            self.assertTrue(torch.allclose(torch.linalg.vector_norm(first, dim=1), torch.tensor([.7848, 1.5696])))
            self.assertTrue(all(torch.equal(v[:, 2], torch.zeros(2)) for _, v in raw.applied))
            self.assertTrue(torch.equal(push.age, torch.zeros(2, dtype=torch.long)))
            receipts = push.take_rollout()['completed_episode_receipts']
            self.assertEqual(len(receipts), 2)
            self.assertTrue(all(r['completed_pulses'] == 29 and r['force_physics_steps'] == 1450 for r in receipts))
            self.assertEqual(push.take_rollout()['completed_episode_receipts'], [])
            self.assertTrue(snap['owned_force_cleared'] and snap['external_wrench_zero'])
        finally: push.close()

    def test_private_random_stream_and_independent_world_directions(self):
        torch.manual_seed(777); before = torch.get_rng_state().clone()
        raw = Raw(64); push = VectorPushes(raw, torch, 2026102901, .05, 0.)
        try:
            for _ in range(101): control(push, raw)
            self.assertTrue(torch.equal(before, torch.get_rng_state()))
            self.assertEqual(len(torch.unique(push.direction, dim=0)), 64)
            self.assertTrue(torch.allclose(torch.linalg.vector_norm(push.direction, dim=1), torch.ones(64), atol=1e-6))
            self.assertFalse(push.state_dict()['physical_episode_resume_supported'])
        finally: push.close()

    def test_no_push_bucket_and_stage_only_changes_at_reset(self):
        raw = Raw(128); push = VectorPushes(raw, torch, 12, .05, .25)
        try:
            original = push.episode_fraction.clone()
            self.assertGreater(int((original == 0).sum()), 0)
            self.assertGreater(int((original > 0).sum()), 0)
            push.set_stage_fraction(.075)
            control(push, raw)
            self.assertTrue(torch.equal(push.episode_fraction, original))
            done = torch.arange(128) % 2 == 0
            control(push, raw, done)
            self.assertTrue(torch.equal(push.episode_fraction[~done], original[~done]))
            self.assertTrue(bool(((push.episode_fraction[done] == 0) | (push.episode_fraction[done] == .075)).all()))
            self.assertTrue(bool((push.age[done] == 0).all()))
            self.assertTrue(bool((push.age[~done] == 20).all()))
            with self.assertRaises(RuntimeError): push.set_stage_fraction(.05)
            with self.assertRaises(RuntimeError): push.set_stage_fraction(.11)
        finally: push.close()

    def test_asynchronous_reset_does_not_shift_other_world_pulse_clock(self):
        raw = Raw(); push = VectorPushes(raw, torch, 91, .1, 0.)
        try:
            for _ in range(50): control(push, raw)
            control(push, raw, torch.tensor([True, False]))
            for _ in range(50): control(push, raw)
            self.assertEqual(push.age.tolist(), [500, 1010])
            self.assertEqual(push.started_pulses.tolist(), [0, 1])
            self.assertTrue(all(bool((v[0] == 0).all()) for _, v in raw.applied))
            for _ in range(51): control(push, raw)
            self.assertEqual(push.started_pulses.tolist(), [1, 1])
        finally: push.close()

    def test_dynamic_mass_change_updates_actual_force(self):
        raw = Raw(1); push = VectorPushes(raw, torch, 44, .1, 0.)
        try:
            for _ in range(100): control(push, raw)
            raw.sim.model.body_mass[:, 2:] *= 1.25
            control(push, raw)
            expected = .1*1.*9.81
            self.assertAlmostEqual(float(torch.linalg.vector_norm(raw.applied[0][1][0])), expected, places=6)
        finally: push.close()

    def test_shared_current_model_mass_is_broadcast_without_mutation(self):
        raw = Raw(4)
        raw.sim.model.body_mass = raw.sim.model.body_mass[:1].clone()
        original = raw.sim.model.body_mass.clone()
        push = VectorPushes(raw, torch, 44, .1, 0.)
        try:
            for _ in range(101): control(push, raw)
            expected = torch.full((4,), .1*.8*9.81)
            self.assertTrue(torch.allclose(torch.linalg.vector_norm(raw.applied[0][1], dim=1), expected))
            self.assertTrue(torch.equal(original, raw.sim.model.body_mass))
            self.assertEqual(push.snapshot()['mass_layout'], 'shared_current_model')
        finally: push.close()

    def test_unknown_force_is_preserved_and_integrator_not_called(self):
        raw = Raw(1); push = VectorPushes(raw, torch, 1)
        raw.sim.data.xfrc_applied[0, 0, 5] = 1.
        with self.assertRaisesRegex(RuntimeError, 'Unknown external'): push.begin_control()
        self.assertEqual(raw.calls, 0)
        self.assertEqual(float(raw.sim.data.xfrc_applied[0, 0, 5]), 1.)
        with self.assertRaisesRegex(RuntimeError, 'Unknown external'): push.close()
        self.assertEqual(float(raw.sim.data.xfrc_applied[0, 0, 5]), 1.)

    def test_nonfinite_mass_stops_before_force_or_integration(self):
        raw = Raw(1); push = VectorPushes(raw, torch, 1, .1, 0.)
        try:
            for _ in range(100): control(push, raw)
            raw.sim.model.body_mass[0, 2] = float('nan')
            push.begin_control()
            with self.assertRaisesRegex(RuntimeError, 'Invalid current robot masses'): raw.sim.step()
            self.assertEqual(raw.calls, 1000)
            self.assertTrue(bool((raw.sim.data.xfrc_applied == 0).all()))
            self.assertEqual(push.snapshot()['nonfinite_faults'], 1)
        finally: push.close()

    def test_integrator_exception_clears_only_owned_force(self):
        raw = Raw(1); push = VectorPushes(raw, torch, 1, .1, 0.)
        try:
            for _ in range(100): control(push, raw)
            raw.raise_on_call = 1001
            push.begin_control()
            with self.assertRaisesRegex(RuntimeError, 'fixture integrator'): raw.sim.step()
            self.assertTrue(bool((raw.sim.data.xfrc_applied == 0).all()))
            self.assertEqual(push.snapshot()['integration_errors'], 1)
        finally: push.close()

    def test_unknown_integrator_wrench_survives_cleanup(self):
        raw = Raw(1); push = VectorPushes(raw, torch, 1, .1, 0.)
        for _ in range(100): control(push, raw)
        raw.inject_unknown = True
        push.begin_control()
        with self.assertRaisesRegex(RuntimeError, 'Unknown external'): raw.sim.step()
        self.assertEqual(float(raw.sim.data.xfrc_applied[0, 0, 5]), .25)
        self.assertTrue(bool((raw.sim.data.xfrc_applied[0, 2, :3] == 0).all()))
        with self.assertRaises(RuntimeError): push.close()

    def test_monitor_outside_force_observes_zero_and_integration_receives_force(self):
        raw = Raw(1); push = VectorPushes(raw, torch, 6, .1, 0.)
        force_step = raw.sim.step
        calls = []
        def monitor():
            self.assertTrue(bool((raw.sim.data.xfrc_applied == 0).all()))
            result = force_step()
            self.assertTrue(bool((raw.sim.data.xfrc_applied == 0).all()))
            calls.append(result)
            return result
        raw.sim.step = monitor
        try:
            for _ in range(105): control(push, raw)
            self.assertEqual(len(calls), 1050)
            self.assertEqual(push.snapshot()['completed_pulses'], 1)
            self.assertEqual(len(raw.applied), 50)
        finally:
            raw.sim.step = force_step
            push.close()

    def test_inference_mode_statistics_can_be_read_and_reset_outside(self):
        raw = Raw(); push = VectorPushes(raw, torch, 9)
        try:
            with torch.inference_mode():
                control(push, raw, torch.ones(2, dtype=torch.bool))
            row = push.take_rollout()
            self.assertEqual(len(row['completed_episode_receipts']), 2)
            push.set_stage_fraction(.1)
            with torch.inference_mode(): control(push, raw)
            self.assertEqual(push.snapshot()['physics_calls'], 20)
        finally: push.close()

    def test_unaccounted_or_incomplete_step_is_rejected(self):
        raw = Raw(1); push = VectorPushes(raw, torch, 1)
        try:
            with self.assertRaisesRegex(RuntimeError, 'Unaccounted'): raw.sim.step()
            push.begin_control(); raw.sim.step()
            with self.assertRaisesRegex(RuntimeError, 'ten complete'): push.end_control(torch.tensor([False]))
        finally: push.close()


if __name__ == '__main__': unittest.main(verbosity=2)
