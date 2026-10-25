"""One episode of real Cartesian pushes; observation does not change the policy."""
import json
import math

from demo_common import need, write
from baseline import Trial, DT
from baseline_runtime import Probe
from native_guard import Guard
from native_plan import (make_schedule, pulse_index, PHYSICS_DT, HORIZON_PHYSICS_STEPS,
                      FORCE_FRACTION, PULSE_STEPS)
from native_plan import FORCE_SEEDS, ENVIRONMENT_SEEDS


class FreshStateGuard(Guard):
    """Read post-integration scalar joint state directly from current qpos/qvel."""
    def __init__(self, raw, torch, run):
        super().__init__(raw, torch, run)
        try:
            gids = [j['global_joint_id'] for j in self.metadata['joints']]
            self.qids = [int(raw.sim.mj_model.jnt_qposadr[j]) for j in gids]
            self.vids = [int(raw.sim.mj_model.jnt_dofadr[j]) for j in gids]
            need(len(set(self.qids)) == len(set(self.vids)) == len(gids), 'Direct joint mapping is not unique')
        except BaseException:
            super().close()
            raise

    def states(self):
        data = self.raw.sim.data
        return (data.qpos[0, self.qids].detach().cpu().tolist(),
                data.qvel[0, self.vids].detach().cpu().tolist())

    def after_reset(self):
        q, v = self.states()
        need(self.torch.allclose(self.raw.sim.data.qpos[:, self.qids], self.robot.data.joint_pos,
                                rtol=0., atol=1e-7), 'Named joints disagree with direct reset qpos')
        self.capture(lambda: self.audit.observe_joints(q, v, initial=True))

    def after_integration(self):
        q, v = self.states()
        forces = self.raw.sim.data.actuator_force[0, self.ctrl_indices].detach().cpu().tolist()
        self.capture(lambda: self.audit.after_integration(q, v, forces))


class EvaluationSession:
    def __init__(self, raw, torch, mdp, run, seed, force_seed):
        need(raw.num_envs == 1 and abs(raw.step_dt-DT) < 1e-12,
             'Expected one 50-Hz environment')
        need(raw.cfg.decimation == 4 and abs(raw.physics_dt-PHYSICS_DT) < 1e-12,
             'Expected four 5-ms physics substeps')
        need(seed in ENVIRONMENT_SEEDS and (force_seed is None or force_seed in FORCE_SEEDS),
             'This one-shot experiment has fixed environment and disturbance seeds')
        self.raw, self.torch, self.run, self.seed, self.force_seed = raw, torch, run, seed, force_seed
        self.boundary=lambda:None
        self.enabled = force_seed is not None
        self.plan = make_schedule(force_seed)
        write(run/'planned-force-schedule.json', dict(
            schema='stage12-native-mujoco-5ms-schedule-v1',
            force_seed=force_seed, environment_seed=seed, force_fraction=FORCE_FRACTION if self.enabled else 0.,
            period_seconds=2., pulse_seconds=.1, horizon_seconds=60.,
            direction_distribution='independent uniform world horizontal angle in [0, 2*pi)',
            random_generator='private Python random.Random',
            schedule_type='fixed onsets at 2,4,...58 seconds; no stability-dependent postponement',
            planned_pulses=len(self.plan), pulses=self.plan), exclusive=True)
        self.robot = raw.scene['robot']
        self.body_ids = [int(x) for x in self.robot.indexing.body_ids.detach().cpu().tolist()]
        need(self.body_ids and len(set(self.body_ids)) == len(self.body_ids), 'Robot body mapping differs')
        local_ids, names = self.robot.find_bodies('^trunk_base$')
        need(len(local_ids) == 1 and list(names) == ['trunk_base'], 'Expected one original trunk_base body')
        self.local_body_id = int(local_ids[0])
        self.body_id = self.body_ids[self.local_body_id]
        for name in ('ball', 'top_ball'):
            other = {int(x) for x in raw.scene[name].indexing.body_ids.detach().cpu().tolist()}
            need(not other.intersection(self.body_ids), 'Robot mass must exclude both free balls')
        self.trial = None
        self.physics_calls = self.checked_physics_steps = self.applied_physics_steps = 0
        self.applied_pulses = self.completed_pulses = 0
        self.in_control = self.owned_force_present = self.closed = False
        self.schedule_verified = True
        self.force_tensor = None
        self.mass = self.gravity = self.force_newtons = None
        self.force_vectors = []
        self.pulses = [dict(**p, applied_physics_steps=0, force_write_attempts=0,
                            actual_start_seconds=None, actual_end_seconds=None,
                            applied_impulse_N_s=0., applied_impulse_world_N_s=[0., 0., 0.],
                            complete=False) for p in self.plan]
        self.trace_path = str(run/'push10-trace.jsonl')
        self.trace = (run/'push10-trace.jsonl').open('x', encoding='utf-8')
        self.guard = self.probe = None
        try:
            self.guard = FreshStateGuard(raw, torch, run)
            self.probe = Probe(raw, torch, mdp)
        except BaseException:
            if self.probe is not None: self.probe.close()
            if self.guard is not None: self.guard.close()
            self.trace.close()
            raise
        self.physics_step_original = raw.sim.step
        raw.sim.step = self.physics_step

    def assert_zero_wrench(self):
        need(bool((self.raw.sim.data.xfrc_applied == 0).all()),
             'Unexpected external body force or torque; preserve rather than clearing it')

    def after_reset(self, episode):
        need(self.trial is None and self.physics_calls == 0 and episode == 1,
             'Only one episode is authorized; reset/retry is disabled')
        self.assert_zero_wrench()
        mass_tensor = self.raw.sim.model.body_mass
        need(mass_tensor.ndim == 2 and mass_tensor.shape[0] == 1,
             'Expected current per-world body masses')
        masses = mass_tensor[0, self.body_ids]
        need(bool(self.torch.isfinite(masses).all()) and bool((masses >= 0).all()),
             'Invalid current robot mass')
        self.mass = float(masses.sum().item())
        gravity = [float(x) for x in self.raw.sim.mj_model.opt.gravity]
        need(len(gravity) == 3 and all(math.isfinite(x) for x in gravity)
             and abs(gravity[0]) < 1e-12 and abs(gravity[1]) < 1e-12 and gravity[2] < 0,
             'Expected original vertical gravity')
        self.gravity = abs(gravity[2])
        self.force_newtons = (FORCE_FRACTION if self.enabled else 0.)*self.mass*self.gravity
        need(math.isfinite(self.force_newtons) and (self.force_newtons > 0 if self.enabled else self.force_newtons == 0),
             'Invalid force magnitude')
        xfrc = self.raw.sim.data.xfrc_applied
        for pulse in self.pulses:
            vector = [self.force_newtons*x for x in pulse['unit_force_world']]
            tensor = self.torch.tensor(vector, dtype=xfrc.dtype, device=xfrc.device)
            actual_vector = tensor.detach().cpu().tolist()
            pulse.update(force_world_N=actual_vector, nominal_force_N=self.force_newtons,
                         actual_force_N=math.sqrt(sum(x*x for x in actual_vector)),
                         nominal_impulse_N_s=self.force_newtons*.1)
            self.force_vectors.append(tensor)
        write(self.run/'force-target.json', dict(
            robot_body_ids=self.body_ids, robot_mass_kg=self.mass,
            robot_mass_excludes=['ball', 'top_ball'], gravity_m_s2=self.gravity,
            force_newtons=self.force_newtons, force_fraction=FORCE_FRACTION if self.enabled else 0.,
            body_name='trunk_base', local_body_id=self.local_body_id, global_body_id=self.body_id,
            frame='world', application_point='trunk_base inertial COM',
            applied_torque_N_m=[0., 0., 0.], additional_PPO_updates=0), exclusive=True)
        self.probe.last = None
        xy = self.robot.data.root_link_pos_w[0, :2].detach().cpu().tolist()
        self.trial = Trial(episode, self.seed, xy)
        self.guard.after_reset()

    def _clear_owned(self):
        if self.owned_force_present:
            # Subtract only the force installed by this adapter. Unexpected
            # other forces/torques remain visible and cause a preserved failure.
            self.raw.sim.data.xfrc_applied[0, self.body_id, :3].sub_(self.force_tensor)
            self.owned_force_present = False
            self.force_tensor = None
            self.raw.sim.data.push()

    def physics_step(self):
        if self.in_control:self.boundary()
        need(self.in_control, 'Unaccounted physics step outside the one authorized episode')
        need(self.trial is not None and not self.trial.ended, 'No active one-shot episode')
        step = self.physics_calls
        index = pulse_index(step) if self.enabled else None
        self.assert_zero_wrench()
        pulse = None if index is None else self.pulses[index]
        try:
            if pulse is not None:
                self.force_tensor = self.force_vectors[index]
                self.raw.sim.data.xfrc_applied[0, self.body_id, :3].copy_(self.force_tensor)
                self.owned_force_present = True
                pulse['force_write_attempts'] += 1
            actual = self.raw.sim.data.xfrc_applied
            expected = self.torch.zeros_like(actual)
            if pulse is not None:
                expected[0, self.body_id, :3] = self.force_tensor
            need(bool(self.torch.equal(actual, expected)), 'Applied wrench differs from planned force')
            self.checked_physics_steps += 1
            self.guard.before_integration()
            result = self.physics_step_original()
            self.physics_calls += 1
            if pulse is not None:
                if pulse['applied_physics_steps'] == 0:
                    self.applied_pulses += 1
                    pulse['actual_start_seconds'] = step*PHYSICS_DT
                pulse['applied_physics_steps'] += 1
                self.applied_physics_steps += 1
                pulse['actual_end_seconds'] = self.physics_calls*PHYSICS_DT
                pulse['applied_impulse_N_s'] += pulse['actual_force_N']*PHYSICS_DT
                pulse['applied_impulse_world_N_s'] = [
                    old+force*PHYSICS_DT for old, force in
                    zip(pulse['applied_impulse_world_N_s'], pulse['force_world_N'])]
                if pulse['applied_physics_steps'] == PULSE_STEPS:
                    pulse['complete'] = True
                    self.completed_pulses += 1
            need(bool(self.torch.equal(actual, expected)), 'Integrator modified external wrench unexpectedly')
            self.guard.after_integration()
            return result
        except BaseException:
            self.schedule_verified = False
            raise
        finally:
            self._clear_owned()

    def step(self, env, actions):
        need(self.trial is not None and not self.trial.ended and not self.closed,
             'No active one-shot control step')
        before, physics_before = self.probe.calls, self.physics_calls
        self.assert_zero_wrench()
        self.in_control = True
        try:
            returned = env.step(actions)
        finally:
            self.in_control = False
        self.assert_zero_wrench()
        need(self.physics_calls == physics_before+4, 'Expected four physics steps per control tick')
        need(self.probe.calls == before+1 and self.probe.last is not None,
             'Expected one original geometric metric sample per control tick')
        self.trial.complete(self.probe.last)
        self.trace.write(json.dumps(dict(
            episode=1, step=self.trial.steps, observed_physics_steps=self.physics_calls,
            applied_pulses=self.applied_pulses, completed_pulses=self.completed_pulses,
            applied_force_physics_steps=self.applied_physics_steps, **self.probe.last), allow_nan=False)+'\n')
        if self.trial.steps % 25 == 0: self.trace.flush()
        return returned

    def finish(self, reason, physical=False, terms=()):
        if self.trial is not None and not self.trial.ended:
            self.trial.finish(reason, physical, terms)
            self.trace.flush()

    def snapshot(self):
        trial = None if self.trial is None else self.trial.snapshot()
        if trial is not None:
            trial['applied_disturbance_pulses'] = self.applied_pulses
            trial['goal'] = ('60-second dual-ball retention under fixed 10%-weight horizontal pulses'
                             if self.enabled else '60-second dual-ball retention without external force') + '; translation allowed'
            trial['disturbance_enabled'] = self.enabled
        return dict(
            trial=trial, model_limits=self.guard.snapshot(),
            observed_physics_steps=self.physics_calls, physics_hz=200,
            physics_state_source='direct current qpos/qvel using named-joint address mapping',
            disturbance=dict(
                planned_pulses=len(self.plan), applied_pulses=self.applied_pulses,
                completed_pulses=self.completed_pulses,
                applied_physics_steps=self.applied_physics_steps,
                checked_physics_steps=self.checked_physics_steps,
                schedule_verified=self.schedule_verified,
                force_fraction=FORCE_FRACTION if self.enabled else 0., period_seconds=2., duration_seconds=.1,
                horizon_seconds=60., force_seed=self.force_seed,
                robot_mass_kg=self.mass, force_newtons=self.force_newtons,
                gravity_m_s2=self.gravity, pulses=[dict(p) for p in self.pulses],
                frame='world', force_body='trunk_base', application_point='inertial COM',
                disturbance_enabled=self.enabled,
                force_schedule_path=str(self.run/'planned-force-schedule.json'),
                unknown_wrenches_cleared=False),
            perturbation_trace_path=self.trace_path, additional_PPO_updates=0,
            full_hardware_limits_verified=False, position_drift_allowed=True,
            strict_contact_acceptance=False, population_success_rate_claimed=False)

    def close(self):
        if self.closed: return
        self.closed = True
        try:
            self._clear_owned()
            self.assert_zero_wrench()
        finally:
            self.raw.sim.step = self.physics_step_original
            self.probe.close()
            self.guard.close()
            self.trace.close()
