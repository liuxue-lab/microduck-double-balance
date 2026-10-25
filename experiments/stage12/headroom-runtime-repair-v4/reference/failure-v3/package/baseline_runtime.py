"""Observe the frozen native task and assert zero external body wrench."""
import json
import math

from demo_common import need
from baseline import Trial, DT
from model_limits_runtime import Guard

class Probe:
    """Observe at the SAME pre-forward metric time used by the original evaluation."""
    def __init__(self, raw, torch, mdp):
        self.raw, self.torch, self.mdp = raw, torch, mdp
        self.original = raw.metrics_manager.compute
        self.col = raw.metrics_manager.active_terms.index('native_joint_geometric_stable_fraction')
        self.calls, self.last = 0, None
        raw.metrics_manager.compute = self.compute

    def compute(self):
        self.original()
        raw, t, mdp = self.raw, self.torch, self.mdp
        p, v, _ = mdp._stage11_head_kinematics(raw)
        r, b = raw.scene['robot'], raw.scene['ball']
        center = t.linalg.vector_norm(p[:, :2], dim=-1)
        height_error = (p[:, 2]-.02).abs()
        speed = t.linalg.vector_norm(v, dim=-1)
        offset = t.linalg.vector_norm(r.data.root_link_pos_w[:, :2]-b.data.root_link_pos_w[:, :2], dim=-1)
        height = r.data.root_link_pos_w[:, 2]-b.data.root_link_pos_w[:, 2]
        tilt = mdp.wrestle_tilt(raw, 'robot')
        lower_speed = t.linalg.vector_norm(b.data.root_link_lin_vel_w[:, :2], dim=-1)
        hold = mdp._bb_state(raw).hold
        values = t.stack((center, height_error, speed, offset, height, tilt, lower_speed, hold), dim=1)
        gates = t.stack((center <= .012, height_error <= .006, speed <= .08,
                        offset <= .06, height >= .20, height <= .29,
                        tilt <= math.radians(20.), lower_speed <= .15, hold <= 1e-6), dim=1)
        cached = raw.metrics_manager._step_values[:, self.col].bool()
        need(bool(t.isfinite(values).all()) and bool((gates.all(dim=1) == cached).all()),
             'Nonfinite measurement or original stability gate changed')
        need(bool((hold <= 1e-6).all()) and bool((raw.command_manager.get_command('twist') == 0).all()),
             'Assistance or nonzero movement command found')
        names = ('top_center_m','top_height_error_m','top_speed_m_s','lower_offset_m',
                 'lower_height_m','robot_tilt_rad','lower_ball_speed_m_s','hold')
        self.last = dict(stable=bool(cached[0]), values=dict(zip(names, values[0].detach().cpu().tolist())),
                         root_xy=r.data.root_link_pos_w[0, :2].detach().cpu().tolist())
        self.calls += 1

    def close(self):
        self.raw.metrics_manager.compute = self.original


class Session:
    def __init__(self, raw, torch, mdp, run, seed):
        need(raw.num_envs == 1 and abs(raw.step_dt-DT) < 1e-12, 'Expected one 50-Hz environment')
        need(raw.cfg.decimation == 10 and abs(raw.physics_dt-.002) < 1e-12, 'Expected ten 2-ms physics substeps')
        self.raw, self.seed = raw, seed
        self.guard = Guard(raw, torch, run)
        self.probe = Probe(raw, torch, mdp)
        self.trial, self.rows = None, []
        self.trace_path = str(run/'baseline-trace.jsonl')
        self.trace = (run/'baseline-trace.jsonl').open('x', encoding='utf-8')
        self.physics_step_original = raw.sim.step
        self.in_control = False
        self.physics_calls = self.zero_wrench_checks = 0
        raw.sim.step = self.physics_step

    def assert_zero_wrench(self):
        # Read only: include every body, both force and torque. Never clear a
        # nonzero wrench to make this control appear clean; preserve the error.
        need(bool((self.raw.sim.data.xfrc_applied == 0).all()),
             'Nonzero external body wrench found in the no-disturbance control')

    def physics_step(self):
        if not self.in_control:
            return self.physics_step_original()
        # AFTER original actions/scene writes, immediately BEFORE integration.
        self.assert_zero_wrench()
        self.zero_wrench_checks += 1
        self.guard.before_integration()
        result = self.physics_step_original()
        self.physics_calls += 1
        self.guard.after_integration()
        return result

    def before_reset(self):
        self.finish('manual_reset')

    def after_reset(self, episode):
        self.assert_zero_wrench()
        self.probe.last = None
        xy = self.raw.scene['robot'].data.root_link_pos_w[0, :2].detach().cpu().tolist()
        self.trial = Trial(episode, self.seed, xy)
        self.guard.after_reset()

    def step(self, env, actions):
        need(self.trial is not None and not self.trial.ended, 'No active baseline trial')
        before, physics_before = self.probe.calls, self.physics_calls
        self.in_control = True
        try:
            returned = env.step(actions)
        finally:
            self.in_control = False
        self.assert_zero_wrench()
        need(self.physics_calls == physics_before+10, 'Expected ten physics steps per control tick')
        need(self.probe.calls == before+1 and self.probe.last is not None, 'Expected one metric sample per control step')
        self.trial.complete(self.probe.last)
        self.trace.write(json.dumps(dict(episode=self.trial.episode, step=self.trial.steps,
                                        external_body_wrench_verified_zero=True,
                                        **self.probe.last), allow_nan=False)+'\n')
        if self.trial.steps % 25 == 0: self.trace.flush()
        return returned

    def finish(self, reason, physical=False, terms=()):
        if self.trial is not None and not self.trial.ended:
            self.trial.finish(reason, physical, terms)
            self.rows.append(self.trial.snapshot())
            self.trace.flush()

    def snapshot(self):
        limits = self.guard.snapshot()
        retained = self.trial is not None and self.trial.snapshot()['goal_60s_no_physical_termination_met']
        checked = (limits['completed_physics_steps'] == 30000 and
                   limits['motor_checks_before_integration'] == 30000 and
                   limits['joint_state_checks'] == 30001 and limits['first_limit_event'] is None)
        return dict(model_limits=limits,
                    retention_and_checked_model_bounds_met=bool(retained and checked),
                    full_hardware_limits_verified=False,
                    baseline_trial=None if self.trial is None else self.trial.snapshot(),
                    baseline_trials=list(self.rows), baseline_trace_path=self.trace_path,
                    control_type='60-second zero-external-wrench observation',
                    disturbance_enabled=False, applied_disturbance_pulses=0,
                    zero_wrench_checks_before_physics=self.zero_wrench_checks,
                    observed_physics_steps=self.physics_calls,
                    force_writes_by_baseline_adapter=0)

    def close(self):
        try:
            self.assert_zero_wrench()
        finally:
            self.raw.sim.step = self.physics_step_original
            self.probe.close()
            self.guard.close()
            self.trace.close()
