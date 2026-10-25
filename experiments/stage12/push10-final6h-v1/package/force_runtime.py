"""Vectorized real horizontal force pulses with episode-local integer clocks.

Install after the initial reset, then install TrainingMotorMonitor on top.
The monitor sees the ordinary zero-wrench scene. This adapter applies and
checks its owned force immediately around the original simulator integration.
"""
import math


def need(value, message):
    if not value:
        raise RuntimeError(message)


class VectorPushes:
    PHYSICS_DT = .002
    CONTROL_SUBSTEPS = 10
    PERIOD_STEPS = 1000
    PULSE_STEPS = 50
    HORIZON_STEPS = 30000
    PULSES = 29
    STAGES = (.05, .075, .10)

    def __init__(self, raw, torch, force_seed, initial_fraction=.05,
                 no_push_probability=.25):
        need(type(force_seed) is int and force_seed >= 0, 'Invalid private force seed')
        need(initial_fraction in self.STAGES, 'Unexpected initial force stage')
        need(0. <= no_push_probability <= 1., 'Invalid no-push episode probability')
        need(raw.num_envs > 0 and abs(raw.step_dt-.02) < 1e-12
             and abs(raw.physics_dt-self.PHYSICS_DT) < 1e-12
             and raw.cfg.decimation == self.CONTROL_SUBSTEPS,
             'Expected original 50-Hz control and 500-Hz physics')
        self.raw, self.t, self.n = raw, torch, raw.num_envs
        self.seed, self.stage = force_seed, float(initial_fraction)
        self.no_push_probability = float(no_push_probability)
        robot = raw.scene['robot']
        self.body_ids = [int(x) for x in robot.indexing.body_ids.detach().cpu().tolist()]
        need(self.body_ids and len(set(self.body_ids)) == len(self.body_ids), 'Invalid robot body mapping')
        local, names = robot.find_bodies('^trunk_base$')
        need(len(local) == 1 and list(names) == ['trunk_base'], 'Expected one trunk_base body')
        self.body_id = self.body_ids[int(local[0])]
        for name in ('ball', 'top_ball'):
            other = set(raw.scene[name].indexing.body_ids.detach().cpu().tolist())
            need(not other.intersection(self.body_ids), 'Robot mass includes a free ball')
        gravity = [float(x) for x in raw.sim.mj_model.opt.gravity]
        need(len(gravity) == 3 and all(math.isfinite(x) for x in gravity)
             and abs(gravity[0]) < 1e-12 and abs(gravity[1]) < 1e-12 and gravity[2] < 0,
             'Expected original vertical gravity')
        self.gravity = abs(gravity[2])
        xfrc = raw.sim.data.xfrc_applied
        need(xfrc.ndim == 3 and xfrc.shape[0] == self.n and xfrc.shape[2] == 6,
             'Unexpected external-force layout')
        self.device, self.dtype = xfrc.device, xfrc.dtype
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(force_seed)
        self.env_ids = torch.arange(self.n, device=self.device)
        self.age = torch.zeros(self.n, dtype=torch.long, device=self.device)
        self.episode_id = torch.zeros_like(self.age)
        self.direction = torch.zeros((self.n, 3), dtype=self.dtype, device=self.device)
        self.episode_fraction = self._draw_fraction()
        self.pulse_fraction = torch.full((self.n, self.PULSES), -1., dtype=self.dtype, device=self.device)
        self.pulse_steps = torch.zeros((self.n, self.PULSES), dtype=torch.long, device=self.device)
        self.started_pulses = torch.zeros_like(self.age)
        self.completed_pulses = torch.zeros_like(self.age)
        self.episode_force_steps = torch.zeros_like(self.age)
        self.sampled_buckets = self._bucket_counts(self.episode_fraction)
        self.completed_buckets = torch.zeros(4, dtype=torch.long, device=self.device)
        self.completed_episode_count = 0
        self.rollout_completed_receipts = []
        self.total_started_pulses = torch.zeros((), dtype=torch.long, device=self.device)
        self.total_completed_pulses = torch.zeros_like(self.total_started_pulses)
        self.total_applied_world_steps = torch.zeros_like(self.total_started_pulses)
        self.total_force_write_world_steps = torch.zeros_like(self.total_started_pulses)
        self.physics_calls = self.checked_physics_calls = self.control_calls = 0
        self.rollout_physics_calls = 0
        self.in_control = self.closed = False
        self.control_start = None
        self.owned_force = None
        self.integration_errors = self.cleanup_errors = 0
        self.nonfinite_faults = self.unknown_wrench_faults = 0
        self.minimum_mass = self.maximum_mass = None
        self.last_mass = None
        self.assert_zero_wrench()
        self._mass()
        self.original = raw.sim.step
        raw.sim.step = self.physics_step

    def _draw_fraction(self):
        t = self.t
        pushed = t.rand(self.n, generator=self.generator, device=self.device) >= self.no_push_probability
        return t.where(pushed, t.full((self.n,), self.stage, dtype=self.dtype, device=self.device),
                       t.zeros(self.n, dtype=self.dtype, device=self.device))

    def _bucket_counts(self, values):
        return self.t.stack([(values == f).sum() for f in (0., *self.STAGES)])

    def _mass(self):
        masses = self.raw.sim.model.body_mass
        need(masses.ndim == 2 and masses.shape[0] in (1, self.n),
             'Expected current shared or per-world body masses')
        robot_masses = masses[:, self.body_ids]
        self.mass_layout = 'shared_current_model' if masses.shape[0] == 1 and self.n > 1 else 'per_world_current_model'
        if robot_masses.shape[0] == 1 and self.n > 1:
            robot_masses = robot_masses.expand(self.n, -1)
        mass = robot_masses.sum(dim=1)
        valid = bool(self.t.stack((self.t.isfinite(robot_masses).all(), (robot_masses >= 0).all(),
                                  self.t.isfinite(mass).all(), (mass > 0).all())).all())
        if not valid and not bool(self.t.isfinite(robot_masses).all()):
            self.nonfinite_faults += 1
        need(valid, 'Invalid current robot masses')
        self.last_mass = mass
        low, high = mass.amin(), mass.amax()
        self.minimum_mass = low if self.minimum_mass is None else self.t.minimum(self.minimum_mass, low)
        self.maximum_mass = high if self.maximum_mass is None else self.t.maximum(self.maximum_mass, high)
        return mass

    def assert_zero_wrench(self):
        wrench = self.raw.sim.data.xfrc_applied
        zero = bool((wrench == 0).all())
        if not zero:
            self.unknown_wrench_faults += 1
            if not bool(self.t.isfinite(wrench).all()):
                self.nonfinite_faults += 1
        need(zero, 'Unknown external force/torque; preserved instead of cleared')

    def set_stage_fraction(self, fraction):
        need(fraction in self.STAGES and fraction >= self.stage, 'Force curriculum must not reverse or exceed 10%')
        self.stage = float(fraction)

    def begin_control(self):
        need(not self.closed and not self.in_control, 'Overlapping or closed force control step')
        need(bool(((self.age % self.CONTROL_SUBSTEPS == 0) & (self.age < self.HORIZON_STEPS)).all()),
             'Episode clock/reset or 60-second horizon mismatch')
        self.assert_zero_wrench()
        # Draw from a private generator, independent of policy/environment RNG.
        # One new direction is selected only at a scheduled pulse onset.
        angle = self.t.rand(self.n, generator=self.generator, device=self.device, dtype=self.dtype) * (2*math.pi)
        proposed = self.t.stack((angle.cos(), angle.sin(), self.t.zeros_like(angle)), dim=1)
        onset = (self.age >= self.PERIOD_STEPS) & (self.age % self.PERIOD_STEPS == 0)
        self.direction = self.t.where(onset[:, None], proposed, self.direction)
        self.control_start = self.physics_calls
        self.in_control = True

    def _clear_owned(self):
        if self.owned_force is not None:
            self.raw.sim.data.xfrc_applied[:, self.body_id, :3].sub_(self.owned_force)
            self.owned_force = None

    def physics_step(self):
        need(self.in_control and not self.closed, 'Unaccounted integration outside force control step')
        need(self.physics_calls-self.control_start < self.CONTROL_SUBSTEPS, 'Too many physics substeps')
        self.assert_zero_wrench()
        t = self.t
        active = (self.age >= self.PERIOD_STEPS) & (self.age < self.HORIZON_STEPS) \
                 & (self.age % self.PERIOD_STEPS < self.PULSE_STEPS) & (self.episode_fraction > 0)
        index = (self.age // self.PERIOD_STEPS-1).clamp(0, self.PULSES-1)
        mass = self._mass()
        force = self.direction * (self.episode_fraction * mass * self.gravity * active)[:, None]
        finite = bool(t.isfinite(force).all())
        if not finite:
            self.nonfinite_faults += 1
        need(finite, 'Nonfinite commanded disturbance')
        actual = self.raw.sim.data.xfrc_applied
        expected = t.zeros_like(actual)
        expected[:, self.body_id, :3] = force
        self.total_force_write_world_steps = self.total_force_write_world_steps + active.sum()
        try:
            self.owned_force = force
            actual[:, self.body_id, :3].copy_(force)
            need(bool(t.equal(actual, expected)), 'Applied disturbance differs from expected wrench')
            self.checked_physics_calls += 1
            result = self.original()
            self.physics_calls += 1
            self.rollout_physics_calls += 1
            need(bool(t.equal(actual, expected)), 'Integrator changed external force/torque')
            onset = active & (self.age % self.PERIOD_STEPS == 0)
            ending = active & (self.age % self.PERIOD_STEPS == self.PULSE_STEPS-1)
            old_fraction = self.pulse_fraction[self.env_ids, index]
            recorded = t.where(onset, self.episode_fraction, old_fraction)
            updated = self.pulse_fraction.clone()
            updated[self.env_ids, index] = recorded
            self.pulse_fraction = updated
            need(bool((~active | (recorded == self.episode_fraction)).all()),
                 'Episode pulse fraction changed inside a pulse')
            counts = self.pulse_steps.clone()
            counts[self.env_ids, index] = counts[self.env_ids, index] + active
            self.pulse_steps = counts
            self.started_pulses = self.started_pulses + onset
            self.completed_pulses = self.completed_pulses + ending
            self.episode_force_steps = self.episode_force_steps + active
            self.total_started_pulses = self.total_started_pulses + onset.sum()
            self.total_completed_pulses = self.total_completed_pulses + ending.sum()
            self.total_applied_world_steps = self.total_applied_world_steps + active.sum()
            self.age = self.age + 1
            return result
        except BaseException:
            self.integration_errors += 1
            raise
        finally:
            try:
                self._clear_owned()
                self.assert_zero_wrench()
            except BaseException:
                self.cleanup_errors += 1
                raise

    def end_control(self, dones):
        need(self.in_control and not self.closed, 'No force control step to finish')
        need(self.physics_calls-self.control_start == self.CONTROL_SUBSTEPS,
             'Expected ten complete physics steps per control tick')
        self.assert_zero_wrench()
        t = self.t
        done = (dones != 0).reshape(-1)
        need(tuple(done.shape) == (self.n,), 'Unexpected reset mask shape')
        need(bool(((self.age < self.HORIZON_STEPS) | done).all()),
             '60-second episode did not reset')
        ids = done.nonzero(as_tuple=False).flatten()
        if ids.numel():
            rows = t.stack((self.env_ids[ids], self.episode_id[ids], self.age[ids],
                            self.started_pulses[ids], self.completed_pulses[ids],
                            self.episode_force_steps[ids]), dim=1).detach().cpu().tolist()
            fractions = self.episode_fraction[ids].detach().cpu().tolist()
            for row, fraction in zip(rows, fractions):
                receipt = dict(env_id=row[0], episode_id=row[1], physics_steps=row[2],
                    fraction=fraction, pushed=fraction > 0, started_pulses=row[3],
                    completed_pulses=row[4], force_physics_steps=row[5],
                    pulse_fraction_encoding='same fraction for each started pulse in index order',
                    first_pulse_seconds=2., period_seconds=2., pulse_seconds=.1)
                self.rollout_completed_receipts.append(receipt)
            self.completed_episode_count += len(rows)
            self.completed_buckets = self.completed_buckets + self._bucket_counts(self.episode_fraction[ids])
            new_fraction = self._draw_fraction()
            self.sampled_buckets = self.sampled_buckets + self._bucket_counts(new_fraction[ids])
            self.episode_fraction = t.where(done, new_fraction, self.episode_fraction)
            self.episode_id = self.episode_id + done
            for name in ('age', 'started_pulses', 'completed_pulses', 'episode_force_steps'):
                value = getattr(self, name)
                setattr(self, name, t.where(done, t.zeros_like(value), value))
            self.direction = t.where(done[:, None], t.zeros_like(self.direction), self.direction)
            self.pulse_fraction = t.where(done[:, None], t.full_like(self.pulse_fraction, -1.), self.pulse_fraction)
            self.pulse_steps = t.where(done[:, None], t.zeros_like(self.pulse_steps), self.pulse_steps)
        self.control_calls += 1
        self.in_control = False
        self.control_start = None

    def snapshot(self):
        sampled = self.sampled_buckets.detach().cpu().tolist()
        completed = self.completed_buckets.detach().cpu().tolist()
        all_episodes = sum(sampled)
        return dict(force_seed=self.seed, physics_calls=self.physics_calls,
            checked_physics_calls=self.checked_physics_calls, control_calls=self.control_calls,
            force_write_world_physics_steps=int(self.total_force_write_world_steps.item()),
            applied_world_physics_steps=int(self.total_applied_world_steps.item()),
            started_pulses=int(self.total_started_pulses.item()), completed_pulses=int(self.total_completed_pulses.item()),
            completed_episodes=self.completed_episode_count, current_stage_fraction=self.stage,
            current_episode_bucket_counts=self._bucket_counts(self.episode_fraction).detach().cpu().tolist(),
            episode_bucket_fractions=[0., *self.STAGES], sampled_episode_bucket_counts=sampled,
            completed_episode_bucket_counts=completed,
            sampled_no_push_fraction=sampled[0]/all_episodes if all_episodes else None,
            configured_no_push_probability=self.no_push_probability,
            minimum_robot_mass_kg=float(self.minimum_mass.item()), maximum_robot_mass_kg=float(self.maximum_mass.item()),
            mass_layout=self.mass_layout,
            robot_mass_excludes=['ball', 'top_ball'], force_body='trunk_base',
            frame='world', application_point='inertial COM', added_torque=[0., 0., 0.],
            period_seconds=2., pulse_seconds=.1, episode_seconds=60., pulses_per_full_push_episode=29,
            stage_changes_only_at_new_episode=True, private_generator=True,
            original_velocity_unmodified=True, unknown_wrenches_cleared=False,
            integration_errors=self.integration_errors, cleanup_errors=self.cleanup_errors,
            nonfinite_faults=self.nonfinite_faults, unknown_wrench_faults=self.unknown_wrench_faults,
            owned_force_cleared=self.owned_force is None,
            external_wrench_zero=bool((self.raw.sim.data.xfrc_applied == 0).all()),
            accepted_policy_evaluation=False)

    def take_rollout(self):
        result = self.snapshot()
        result['rollout_physics_calls'] = self.rollout_physics_calls
        result['completed_episode_receipts'] = self.rollout_completed_receipts
        self.rollout_completed_receipts = []
        self.rollout_physics_calls = 0
        return result

    def state_dict(self):
        """For Torch checkpoint provenance; no physical-state resume is implied."""
        return dict(force_seed=self.seed, generator_state=self.generator.get_state(),
                    generator_device=str(self.device), stage_fraction=self.stage,
                    physical_episode_resume_supported=False, snapshot=self.snapshot())

    def close(self):
        if self.closed:
            return
        try:
            self._clear_owned()
            self.assert_zero_wrench()
        finally:
            self.raw.sim.step = self.original
            self.closed = True

