"""Read-only 200-Hz boundary monitor for the pinned Stage12 BAM runtime."""
import importlib
import json
import math
from pathlib import Path

from demo_common import need, sha, write
from model_limits import Audit, LimitStop

SOURCE_PINS = {
    'bam.dynamixel.actuator': '8d3ed39d68758981204901dd18dcce11dcb107b842067e036d2b8ac7b1411cec',
    'bam.mjlab': 'af3de252939ca868712423979c2ab52e198d382d7aca613b5b33c77d74baa440',
    'bam.actuator': '6927d7b2341cbaca0e5d872fae4ddce6200056885310d9ee5acaba1f34e09c13',
    'mjlab_microduck.actuator.friction_dr_bam': '75cb3e993cd8fdc9d90081129f793deb1eb22fc1ad921f75a97511fa98d38fb7',
}


def verify_sources():
    observed = {}
    for name, expected in SOURCE_PINS.items():
        module = importlib.import_module(name)
        path = Path(module.__file__)
        actual = sha(path)
        need(actual == expected, 'Limit monitor source differs: '+name+'; preserve installed runtime')
        observed[name] = dict(path=str(path), sha256=actual)
    return observed


def joint_bounds(values):
    lo, hi = map(float, values)
    need(not math.isnan(lo) and not math.isnan(hi), 'Invalid NaN joint range')
    if lo == -math.inf and hi == math.inf:
        return None, None
    need(math.isfinite(lo) and math.isfinite(hi) and lo < hi, 'Invalid joint range')
    return lo, hi


class Guard:
    def __init__(self, raw, torch, run):
        self.raw, self.torch, self.robot = raw, torch, raw.scene['robot']
        source_checks = verify_sources()
        r, model = self.robot, raw.sim.mj_model
        self.groups = list(r.actuators)
        need(self.groups, 'No actuator groups to monitor')
        joint_ids = r.indexing.joint_ids.detach().cpu().tolist()
        names = list(r.joint_names)
        need(len(joint_ids) == len(names) == r.data.joint_pos.shape[1], 'Expected scalar hinge joints')
        need(all(int(model.jnt_type[j]) == 3 for j in joint_ids), 'This monitor expects the original hinge-joint model')
        hard = r.data.joint_pos_limits[0].detach().cpu().tolist()
        soft = r.data.soft_joint_pos_limits[0].detach().cpu().tolist()
        need(len(hard) == len(soft) == len(names), 'Joint range mapping differs')
        joints = []
        for name, gid, bounds, soft_bounds in zip(names, joint_ids, hard, soft):
            lo, hi = joint_bounds(bounds)
            soft_lo, soft_hi = joint_bounds(soft_bounds)
            need(bool(model.jnt_limited[gid]) == (lo is not None), 'Joint limit enable flag differs')
            need((lo is None and soft_lo is None) or
                 (lo is not None and soft_lo is not None and lo <= soft_lo < soft_hi <= hi),
                 'Soft joint range differs from nominal range')
            joints.append(dict(name=name, global_joint_id=gid, lower_rad=lo, upper_rad=hi,
                soft_lower_rad=soft_lo, soft_upper_rad=soft_hi,
                solver_solref=[float(x) for x in model.jnt_solref[gid]],
                solver_solimp=[float(x) for x in model.jnt_solimp[gid]],
                solver_margin_rad=float(model.jnt_margin[gid])))
        motors, self.ctrl_indices, seen = [], [], []
        for group in self.groups:
            need(type(group).__name__ == 'FrictionDRBamActuator', 'Expected the original non-backlash BAM actuator')
            need(hasattr(group, 'diagnostics_enabled') and hasattr(group, 'last_diagnostics'), 'Existing BAM diagnostic interface missing')
            local_ids = group.target_ids.detach().cpu().tolist()
            global_ctrl = group.global_ctrl_ids.detach().cpu().tolist()
            need(len(local_ids) == len(global_ctrl) == len(group.target_names), 'Actuator target mapping differs')
            bam, act = group._bam_model, group._bam_model.actuator
            kt, resistance, pwm = float(bam.kt.value), float(bam.R.value), float(act.max_pwm)
            cap = None if act.max_current is None else float(act.max_current)
            need(all(math.isfinite(x) and x > 0 for x in (kt, resistance, pwm)), 'Invalid motor parameters')
            need(cap is None or math.isfinite(cap) and cap > 0, 'Invalid current limit')
            ranges = raw.sim.model.actuator_forcerange[0, group.global_ctrl_ids].detach().cpu().tolist()
            for local, ctrl, name, force_range in zip(local_ids, global_ctrl, group.target_names, ranges):
                need(names[local] == name and not name.startswith('passive_'), 'Servo joint mapping differs')
                need(int(model.actuator_trnid[ctrl, 0]) == joint_ids[local], 'Actuator transmission target differs')
                need(int(model.actuator_trntype[ctrl]) == 0 and int(model.actuator_dyntype[ctrl]) == 0
                     and int(model.actuator_gaintype[ctrl]) == 0 and int(model.actuator_biastype[ctrl]) == 0,
                     'Expected direct fixed-gain motor transmission')
                need(list(model.actuator_gear[ctrl]) == [1., 0., 0., 0., 0., 0.], 'Expected unit motor gear')
                need(abs(float(model.actuator_gainprm[ctrl, 0])-1.) < 1e-12, 'Expected unit motor gain')
                limited = bool(model.actuator_forcelimited[ctrl])
                lo, hi = (float(force_range[0]), float(force_range[1])) if limited else (None, None)
                need(limited and math.isfinite(lo) and math.isfinite(hi) and lo < hi, 'Original motor force range missing')
                motors.append(dict(name=name, local_joint_id=local, global_ctrl_id=ctrl,
                    kt=kt, resistance_ohm=resistance, max_pwm=pwm, max_current_A=cap,
                    force_lower_Nm=lo, force_upper_Nm=hi, independent_speed_limit_rad_s=None))
                seen.append(local)
                self.ctrl_indices.append(ctrl)
        expected = [i for i, name in enumerate(names) if not name.startswith('passive_')]
        need(len(seen) == len(set(seen)) == 14 and sorted(seen) == expected, 'Expected all fourteen original servos exactly once')
        self.audit = Audit(joints, motors, joint_violation_handling='record_and_continue_simulation')
        self.previous_flags = [g.diagnostics_enabled for g in self.groups]
        self.previous_diagnostics = [g.last_diagnostics for g in self.groups]
        self.last_seen = [None] * len(self.groups)
        self.event_path = run/'model-limit-event.json'
        self.stop_event_path = run/'model-stop-event.json'
        self.first_event_saved = False
        self.metadata = dict(schema='stage12-model-boundaries-v3', source_checks=source_checks,
            joints=joints, motors=motors, physics_hz=200,
            comparison='actual joint states and actual motor outputs; targets recorded separately',
            joint_violation_handling='record_and_continue_simulation',
            joint_violations_permanently_fail_acceptance=True,
            observation_only=True, hardware_control=False,
            hard_stop_on=['modeled current beyond configured limit', 'PWM beyond configured limit',
                          'applied torque outside compiled motor force range', 'nonfinite state'],
            unknown_limits=['independent rated maximum joint speed', 'continuous/thermal motor ratings',
                            'certified recoverable whole-body tilt envelope'],
            original_fall_termination_retained=True, real_hardware_validation=False,
            current_is_model_estimate=True, additional_physics_steps=0, changes_to_control_law=False)
        write(run/'model-boundaries.json', self.metadata, exclusive=True)
        for group in self.groups: group.diagnostics_enabled = True

    def capture(self, callback):
        try:
            return callback()
        except LimitStop as stop:
            if not self.stop_event_path.exists(): write(self.stop_event_path, stop.event, exclusive=True)
            raise
        finally:
            if self.audit.first_event is not None and not self.first_event_saved:
                write(self.event_path, self.audit.first_event, exclusive=True)
                self.first_event_saved = True

    def after_reset(self):
        q = self.robot.data.joint_pos[0].detach().cpu().tolist()
        v = self.robot.data.joint_vel[0].detach().cpu().tolist()
        self.capture(lambda: self.audit.observe_joints(q, v, initial=True))

    def before_integration(self):
        points = []
        for index, group in enumerate(self.groups):
            d = group.last_diagnostics
            need(d is not None and d is not self.last_seen[index], 'Missing fresh per-substep BAM diagnostics')
            self.last_seen[index] = d
            velocity = d['joint_velocity']
            fields = (d['effective_vin'].expand_as(velocity), velocity, group.kd_scale.expand_as(velocity),
                      d['duty_applied'], d['motor_torque'], self.raw.sim.data.ctrl[:, group.global_ctrl_ids],
                      self.robot.data.joint_pos[:, group.target_ids], d['position_error'])
            rows = self.torch.stack(fields, dim=-1)[0].detach().cpu().tolist()
            keys = ('voltage', 'velocity', 'kd_scale', 'duty', 'torque', 'ctrl', 'position', 'position_error')
            points.extend(dict(zip(keys, row)) for row in rows)
        self.capture(lambda: self.audit.before_integration(points))

    def after_integration(self):
        q = self.robot.data.joint_pos[0].detach().cpu().tolist()
        v = self.robot.data.joint_vel[0].detach().cpu().tolist()
        forces = self.raw.sim.data.actuator_force[0, self.ctrl_indices].detach().cpu().tolist()
        self.capture(lambda: self.audit.after_integration(q, v, forces))

    def snapshot(self):
        return dict(**self.audit.snapshot(), boundaries=self.metadata)

    def close(self):
        for group, flag, diagnostics in zip(self.groups, self.previous_flags, self.previous_diagnostics):
            group.diagnostics_enabled = flag
            group.last_diagnostics = diagnostics
