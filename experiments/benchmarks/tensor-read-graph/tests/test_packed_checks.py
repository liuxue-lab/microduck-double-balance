"""Exercise original Audit and EvaluationSession against the packed-read path.

CPU fixtures cover control flow, mapping and exact numbers; they cannot measure
CUDA transfers or prove speed on the user's RTX5060.
"""
import copy
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'package'))
from eval_runtime import EvaluationSession, FreshStateGuard
from eval_plan import make_schedule
from model_limits import Audit, LimitStop
from packed_checks import PackedChecks
from bam_graph import BatchedAuditReads
from test_bam_graph import fixture


def make_session():
    s = EvaluationSession.__new__(EvaluationSession)
    group, command = fixture()
    # A deliberately nonidentity control permutation exercises named mapping.
    group.global_ctrl_ids = torch.arange(13, -1, -1)
    group.target_ids = torch.arange(14)
    group.target_names = ['joint_' + str(i) for i in range(14)]
    data = NS(qpos=torch.zeros(1, 14), qvel=torch.zeros(1, 14),
              ctrl=torch.zeros(1, 14), actuator_force=torch.zeros(1, 14),
              xfrc_applied=torch.zeros(1, 4, 6))
    s.torch = torch
    s.raw = NS(sim=NS(data=data, model=group._mjwarp_model))
    guard = FreshStateGuard.__new__(FreshStateGuard)
    guard.raw, guard.torch = s.raw, torch
    guard.robot = NS(data=NS(joint_pos=data.qpos))
    guard.groups, guard.last_seen = [group], [None]
    guard.qids, guard.vids, guard.ctrl_indices = list(range(14)), list(range(14)), list(range(13, -1, -1))
    joints = [dict(name=name, lower_rad=-3., upper_rad=3.,
                   soft_lower_rad=-2.5, soft_upper_rad=2.5) for name in group.target_names]
    motors = [dict(name=name, local_joint_id=i, kt=group._bam_model.kt.value,
                   resistance_ohm=group._bam_model.R.value, max_pwm=1.,
                   max_current_A=1.75, force_lower_Nm=-1., force_upper_Nm=1.)
              for i, name in enumerate(group.target_names)]
    guard.audit = Audit(joints, motors, joint_violation_handling='record_and_continue_simulation')
    guard.audit.observe_joints([0.] * 14, [0.] * 14, initial=True)
    guard.capture = lambda f: f()
    s.guard = guard
    s.enabled, s.in_control, s.closed, s.owned_force_present = True, True, False, False
    s.trial, s.force_tensor = NS(ended=False), None
    s.physics_calls = s.checked_physics_steps = s.applied_physics_steps = 0
    s.applied_pulses = s.completed_pulses = 0
    s.schedule_verified = True
    s.body_id, s.force_vectors, s.pulses = 2, [], []
    for p in make_schedule(2026102500):
        vector = torch.tensor([.714985426 * x for x in p['unit_force_world']])
        values = vector.tolist()
        s.force_vectors.append(vector)
        s.pulses.append(dict(**p, force_world_N=values,
            actual_force_N=math.sqrt(sum(x * x for x in values)),
            applied_physics_steps=0, force_write_attempts=0,
            actual_start_seconds=None, actual_end_seconds=None,
            applied_impulse_N_s=0., applied_impulse_world_N_s=[0., 0., 0.], complete=False))
    events = NS(boundaries=0, integrated=0, wrench_samples=[])
    def boundary(): events.boundaries += 1
    s.boundary = boundary
    def integrate():
        events.integrated += 1
        events.wrench_samples.append(data.xfrc_applied.clone())
        data.actuator_force.copy_(data.ctrl)
        data.qvel.copy_(data.ctrl[:, group.global_ctrl_ids] * .01)
        data.qpos.add_(data.qvel * .002)
        return 'integrated'
    s.physics_step_original = integrate
    s.raw.sim.step = s.physics_step
    def command_step(i=0):
        command.pos.copy_(data.qpos)
        command.vel.copy_(data.qvel)
        command.position_target.fill_(.02 * math.sin(i * .021))
        data.ctrl[:, group.global_ctrl_ids] = group.compute(command)
    command_step()
    return s, command_step, events


class PackedReadTests(unittest.TestCase):
    def test_1600_steps_match_original_full_audit_and_complete_force_pulse(self):
        original, set_a, ea = make_session()
        candidate, set_b, eb = make_session()
        baseline_reads = BatchedAuditReads(original.guard, torch)
        packed = PackedChecks(candidate, torch)
        try:
            for step in range(1600):
                set_a(step); set_b(step)
                self.assertEqual(original.physics_step(), packed.physics_step())
                for field in ('qpos', 'qvel', 'ctrl', 'actuator_force', 'xfrc_applied'):
                    self.assertTrue(torch.equal(getattr(original.raw.sim.data, field),
                                                getattr(candidate.raw.sim.data, field)), (step, field))
            self.assertEqual(original.guard.audit.snapshot(), candidate.guard.audit.snapshot())
            self.assertEqual(original.pulses, candidate.pulses)
            for field in ('physics_calls', 'checked_physics_steps', 'applied_physics_steps',
                          'applied_pulses', 'completed_pulses', 'owned_force_present', 'schedule_verified'):
                self.assertEqual(getattr(original, field), getattr(candidate, field))
            self.assertEqual(candidate.applied_physics_steps, 50)
            self.assertEqual(candidate.completed_pulses, 1)
            self.assertEqual((packed.pre_reads, packed.post_reads, packed.zero_checks), (1600, 1600, 1600))
            self.assertEqual(candidate.guard.audit.joint_checks, 1601)
            self.assertEqual(ea.boundaries, eb.boundaries)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(ea.wrench_samples, eb.wrench_samples)))
        finally: baseline_reads.close()

    def test_multiple_groups_keep_motor_and_joint_order(self):
        a, _, _ = make_session(); b, _, _ = make_session()
        for s in (a, b):
            group = s.guard.groups[0]; groups = []
            for start in (0, 7):
                g = copy.copy(group)
                g.global_ctrl_ids = group.global_ctrl_ids[start:start + 7]
                g.target_ids = group.target_ids[start:start + 7]
                g.last_diagnostics = {key: value[:, start:start + 7] if value.ndim == 2 and value.shape[1] == 14 else value
                                      for key, value in group.last_diagnostics.items()}
                groups.append(g)
            s.guard.groups = groups; s.guard.last_seen = [None, None]
        a.guard.before_integration()
        PackedChecks(b, torch).before([0.] * 24)
        self.assertEqual(a.guard.audit.snapshot(), b.guard.audit.snapshot())

    def test_unexpected_wrench_is_preserved_before_any_write_or_integration(self):
        for packed in (False, True):
            s, _, events = make_session(); s.physics_calls = 1000
            s.raw.sim.data.xfrc_applied[0, 2, 0] = .123
            before = s.raw.sim.data.xfrc_applied.clone()
            method = PackedChecks(s, torch).physics_step if packed else s.physics_step
            with self.assertRaisesRegex(RuntimeError, 'Unexpected external'):
                method()
            self.assertTrue(torch.equal(before, s.raw.sim.data.xfrc_applied))
            self.assertEqual(events.integrated, 0); self.assertEqual(s.pulses[0]['force_write_attempts'], 0)

    def test_precheck_rejects_changed_force_before_integration_and_clears_only_owned(self):
        s, _, events = make_session(); s.physics_calls = 1000
        packed = PackedChecks(s, torch); before = packed.before
        def inject(expected):
            s.raw.sim.data.xfrc_applied[0, 0, 5] = .25
            before(expected)
        packed.before = inject
        with self.assertRaisesRegex(RuntimeError, 'Applied wrench differs'): packed.physics_step()
        self.assertEqual(events.integrated, 0); self.assertFalse(s.schedule_verified)
        self.assertEqual(s.raw.sim.data.xfrc_applied[0, 0, 5].item(), .25)
        self.assertTrue(torch.equal(s.raw.sim.data.xfrc_applied[0, 2], torch.zeros(6)))

    def test_faults_preserve_original_stop_phase_counts_and_cleanup(self):
        for fault in ('motor_nan', 'post_force', 'post_joint_nan', 'integrator_error', 'stale'):
            results = []
            for packed in (False, True):
                s, _, events = make_session(); s.physics_calls = 1000
                integrate = s.physics_step_original
                if fault == 'motor_nan': s.guard.groups[0].last_diagnostics['motor_torque'][0, 4] = float('nan')
                if fault == 'stale': s.guard.last_seen[0] = s.guard.groups[0].last_diagnostics
                def faulty(fault=fault, s=s, integrate=integrate):
                    if fault == 'integrator_error': raise RuntimeError('injected integrator failure')
                    result = integrate()
                    if fault == 'post_force': s.raw.sim.data.xfrc_applied[0, 0, 4] = .125
                    if fault == 'post_joint_nan': s.raw.sim.data.qpos[0, 5] = float('nan')
                    return result
                s.physics_step_original = faulty
                method = PackedChecks(s, torch).physics_step if packed else s.physics_step
                with self.assertRaises((RuntimeError, LimitStop)) as caught: method()
                self.assertFalse(s.owned_force_present)
                self.assertFalse(s.schedule_verified)
                self.assertTrue(torch.equal(s.raw.sim.data.xfrc_applied[0, 2], torch.zeros(6)))
                results.append((type(caught.exception), str(caught.exception), events.integrated,
                    s.physics_calls, s.checked_physics_steps, s.guard.audit.snapshot(), s.pulses))
            self.assertEqual(results[0], results[1], fault)

    def test_joint_excursion_is_recorded_and_not_erased_by_reentry(self):
        snapshots = []
        for packed in (False, True):
            s, set_command, _ = make_session()
            integrate = s.physics_step_original
            def excursion(s=s, integrate=integrate):
                integrate(); s.raw.sim.data.qpos[0, 3] = 3.2
            s.physics_step_original = excursion
            method = PackedChecks(s, torch).physics_step if packed else s.physics_step
            method()
            s.raw.sim.data.qpos.zero_(); set_command()
            s.physics_step_original = integrate
            method()
            snapshots.append(s.guard.audit.snapshot())
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertTrue(snapshots[1]['any_model_bound_violation_observed'])
        self.assertEqual(snapshots[1]['completed_physics_steps'], 2)

    def test_boundary_cancel_precedes_force_write(self):
        s, _, events = make_session(); s.physics_calls = 1000
        def cancel(): raise RuntimeError('cancelled')
        s.boundary = cancel
        with self.assertRaisesRegex(RuntimeError, 'cancelled'): PackedChecks(s, torch).physics_step()
        self.assertEqual(events.integrated, 0); self.assertFalse(s.owned_force_present)

    def test_no_dtype_promotion_and_exact_wrench_comparison(self):
        s, _, events = make_session(); packed = PackedChecks(s, torch)
        s.raw.sim.data.ctrl = s.raw.sim.data.ctrl.double()
        with self.assertRaisesRegex(RuntimeError, 'schema changed'): packed.physics_step()
        self.assertEqual(events.integrated, 0)
        s, _, _ = make_session(); packed = PackedChecks(s, torch)
        s.raw.sim.data.xfrc_applied[0, 0, 0] = 1e-30
        with self.assertRaisesRegex(RuntimeError, 'Applied wrench differs'): packed.before([0.] * 24)
        self.assertEqual(s.guard.audit.motor_checks, 0)


if __name__ == '__main__': unittest.main()
