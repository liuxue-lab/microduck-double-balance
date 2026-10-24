"""Batch reads, not audit frequency; preserve original per-substep ordering.

The original zero-wrench test still runs before any force write. Before and
after each integrator call, copy the entire applied wrench together with the
original audit inputs. Host equality is exact; no epsilon is used for forces.
The unchanged CPU Audit still runs before/after each 2-ms physics step.
"""
from demo_common import need
from eval_plan import pulse_index, PHYSICS_DT, PULSE_STEPS

KEYS = ('voltage', 'velocity', 'kd_scale', 'duty', 'torque', 'ctrl',
        'position', 'position_error')


class PackedChecks:
    def __init__(self, session, torch):
        self.session, self.guard, self.torch = session, session.guard, torch
        data = session.raw.sim.data
        wrench = data.xfrc_applied
        need(wrench.ndim == 3 and wrench.shape[0] == 1 and wrench.shape[2] == 6,
             'Expected one complete world of Cartesian body wrenches')
        need(0 <= session.body_id < wrench.shape[1], 'Force body mapping changed')
        self.wrench_shape = tuple(wrench.shape)
        self.device = wrench.device
        self.width = wrench.numel()
        self.zero = [0.] * self.width
        self.expected = []
        for pulse in session.pulses:
            values = list(self.zero)
            values[session.body_id * 6:session.body_id * 6 + 3] = pulse['force_world_N']
            need(len(values) == self.width, 'Planned force vector shape changed')
            self.expected.append(values)
        g = self.guard
        self.qids = torch.tensor(g.qids, dtype=torch.long, device=self.device)
        self.vids = torch.tensor(g.vids, dtype=torch.long, device=self.device)
        self.cids = torch.tensor(g.ctrl_indices, dtype=torch.long, device=self.device)
        self.nq, self.nv, self.nf = len(g.qids), len(g.vids), len(g.ctrl_indices)
        self.pre_reads = self.post_reads = self.zero_checks = 0

    def packed(self, parts):
        # Do not silently promote any original value into another precision.
        need(all(p.dtype == self.torch.float32 and p.device == self.device
                 and not p.requires_grad for p in parts), 'Packed audit input schema changed')
        need(tuple(self.session.raw.sim.data.xfrc_applied.shape) == self.wrench_shape,
             'Applied wrench shape changed')
        return self.torch.cat([p.reshape(-1) for p in parts]).detach().cpu().tolist()

    def before(self, expected):
        s, g, t = self.session, self.guard, self.torch
        parts = [s.raw.sim.data.xfrc_applied]
        diagnostics = []
        for index, group in enumerate(g.groups):
            d = group.last_diagnostics
            # On malformed/stale diagnostics retain the original failure order:
            # validate the force first, then the original guard reports failure.
            if d is None or d is g.last_seen[index]:
                actual = parts[0].detach().reshape(-1).cpu().tolist()
                need(actual == expected, 'Applied wrench differs from planned force')
                s.checked_physics_steps += 1
                g.before_integration()
                raise RuntimeError('Original guard did not reject stale diagnostics')
            velocity = d['joint_velocity']
            fields = (d['effective_vin'].expand_as(velocity), velocity,
                      group.kd_scale.expand_as(velocity), d['duty_applied'],
                      d['motor_torque'], s.raw.sim.data.ctrl[:, group.global_ctrl_ids],
                      g.robot.data.joint_pos[:, group.target_ids], d['position_error'])
            parts.append(t.stack(fields, dim=-1)[0])
            diagnostics.append(d)
        values = self.packed(parts)
        self.pre_reads += 1
        need(values[:self.width] == expected, 'Applied wrench differs from planned force')
        s.checked_physics_steps += 1
        rows = values[self.width:]
        need(len(rows) == self.nf * len(KEYS), 'Motor transfer mapping changed')
        points = [dict(zip(KEYS, rows[i:i + len(KEYS)]))
                  for i in range(0, len(rows), len(KEYS))]
        g.last_seen[:] = diagnostics
        g.capture(lambda: g.audit.before_integration(points))

    def after(self, expected):
        data, g = self.session.raw.sim.data, self.guard
        values = self.packed((data.xfrc_applied, data.qpos[0, self.qids],
                              data.qvel[0, self.vids], data.actuator_force[0, self.cids]))
        self.post_reads += 1
        need(values[:self.width] == expected, 'Integrator modified external wrench unexpectedly')
        values = values[self.width:]
        need(len(values) == self.nq + self.nv + self.nf, 'State transfer mapping changed')
        g.capture(lambda: g.audit.after_integration(
            values[:self.nq], values[self.nq:self.nq + self.nv], values[self.nq + self.nv:]))

    def physics_step(self):
        # Same pulse bookkeeping, physical call and cleanup as EvaluationSession.
        s = self.session
        if s.in_control:
            s.boundary()
        need(s.in_control, 'Unaccounted physics step outside the one authorized episode')
        need(s.trial is not None and not s.trial.ended, 'No active one-shot episode')
        step = s.physics_calls
        index = pulse_index(step) if s.enabled else None
        s.assert_zero_wrench()
        self.zero_checks += 1
        pulse = None if index is None else s.pulses[index]
        expected = self.zero if index is None else self.expected[index]
        try:
            if pulse is not None:
                s.force_tensor = s.force_vectors[index]
                s.raw.sim.data.xfrc_applied[0, s.body_id, :3].copy_(s.force_tensor)
                s.owned_force_present = True
                pulse['force_write_attempts'] += 1
            self.before(expected)
            result = s.physics_step_original()
            s.physics_calls += 1
            if pulse is not None:
                if pulse['applied_physics_steps'] == 0:
                    s.applied_pulses += 1
                    pulse['actual_start_seconds'] = step * PHYSICS_DT
                pulse['applied_physics_steps'] += 1
                s.applied_physics_steps += 1
                pulse['actual_end_seconds'] = s.physics_calls * PHYSICS_DT
                pulse['applied_impulse_N_s'] += pulse['actual_force_N'] * PHYSICS_DT
                pulse['applied_impulse_world_N_s'] = [
                    old + force * PHYSICS_DT for old, force in
                    zip(pulse['applied_impulse_world_N_s'], pulse['force_world_N'])]
                if pulse['applied_physics_steps'] == PULSE_STEPS:
                    pulse['complete'] = True
                    s.completed_pulses += 1
            self.after(expected)
            return result
        except BaseException:
            s.schedule_verified = False
            raise
        finally:
            s._clear_owned()

    def stats(self):
        return dict(zero_wrench_checks_before_write=self.zero_checks,
                    packed_reads_before_integration=self.pre_reads,
                    packed_reads_after_integration=self.post_reads,
                    all_body_force_and_torque_values_per_check=self.width,
                    force_comparison='exact host float equality; no tolerance',
                    unchanged_CPU_Audit=True, physics_hz=500,
                    no_check_decimation=True, no_audit_deferral=True,
                    remaining_explicit_physics_check_read_points_per_substep=3)
