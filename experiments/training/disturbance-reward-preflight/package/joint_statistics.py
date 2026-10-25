"""Aggregate direct, post-integration qpos before the environment can reset.

No physics/control writes. A rollout may cross resets: the violating terminal
samples remain in the window, while completed episodes are counted separately.
"""
import math


class JointStatistics:
    def __init__(self, backend, reference_q, bounds, names, dt=0.002):
        self.t = t = backend
        self.names = list(names)
        if len(self.names) != 14 or len(set(self.names)) != 14 or 'head_roll' not in names:
            raise ValueError('Expected fourteen named servo joints')
        if reference_q.shape[1] != 14 or bounds.shape != (14, 2):
            raise ValueError('Joint statistics shape mismatch')
        if not math.isfinite(dt) or abs(dt-0.002) > 1e-12:
            raise ValueError('Expected original 500-Hz physics')
        self.n = reference_q.shape[0]
        self.bounds, self.dt = bounds, dt
        self.head = self.names.index('head_roll')
        device = reference_q.device
        self.episode_steps = t.zeros(self.n, dtype=t.long, device=device)
        self.episode_bad_steps = t.zeros_like(self.episode_steps)
        self.safe_run = t.zeros_like(self.episode_steps)
        self.bad_joint_samples = t.zeros(14, dtype=t.long, device=device)
        self.bad_world_samples = t.zeros((), dtype=t.long, device=device)
        self.peak_excursion = t.zeros(14, device=device)
        self.peak_head_abs = t.zeros((), device=device)
        self.peak_safe_run = t.zeros((), dtype=t.long, device=device)
        self.ended = t.zeros(4, dtype=t.long, device=device)
        self.steps = 0

    def observe(self, q):
        t = self.t
        if q.shape != (self.n, 14):
            raise ValueError('Measured qpos dimensions changed')
        lo, hi = self.bounds[:, 0], self.bounds[:, 1]
        excursion = t.maximum(lo-q, q-hi).clip(0.0, None)
        scale = t.maximum(t.abs(q), t.maximum(t.abs(lo), t.abs(hi)))
        tolerance = (32*2**-23)*t.maximum(t.ones_like(q), scale)
        bad = excursion > tolerance
        any_bad = t.any(bad, dim=1)
        # Rollout runs in inference_mode; logging runs outside it. Never mutate
        # a persistent tensor in place across that boundary.
        self.bad_joint_samples = self.bad_joint_samples + t.sum(bad, dim=0)
        self.bad_world_samples = self.bad_world_samples + t.sum(any_bad)
        self.peak_excursion = t.maximum(self.peak_excursion, t.amax(excursion, dim=0))
        self.peak_head_abs = t.maximum(self.peak_head_abs, t.amax(t.abs(q[:, self.head])))
        self.episode_steps = self.episode_steps + 1
        self.episode_bad_steps = self.episode_bad_steps + any_bad
        self.safe_run = t.where(any_bad, t.zeros_like(self.safe_run), self.safe_run+1)
        self.peak_safe_run = t.maximum(self.peak_safe_run, t.amax(self.safe_run))
        self.steps += 1

    def after_control_step(self, dones, physically_terminated):
        """Called after env.step only to finalize/reset counters, never to sample q."""
        t = self.t
        done = dones != 0
        full = done & (self.episode_steps == 30000) & ~physically_terminated
        safe_full = full & (self.episode_bad_steps == 0)
        self.ended = self.ended + t.stack((t.sum(done), t.sum(full), t.sum(safe_full),
                              t.sum(done & (self.episode_bad_steps > 0))))
        for name in ('episode_steps', 'episode_bad_steps', 'safe_run'):
            value = getattr(self, name)
            setattr(self, name, t.where(done, t.zeros_like(value), value))

    def take_rollout(self, expected_steps=240):
        if self.steps != expected_steps:
            raise RuntimeError('Incomplete physical-step statistics window')
        counts = self.bad_joint_samples.detach().cpu().tolist()
        peaks = self.peak_excursion.detach().cpu().tolist()
        ended = self.ended.detach().cpu().tolist()
        samples = self.steps*self.n
        result = dict(
            measurement='direct qpos after every physics step, before auto-reset',
            physics_steps=self.steps, environments=self.n, world_physics_samples=samples,
            any_joint_violation_fraction=int(self.bad_world_samples.item())/samples,
            head_roll_violation_fraction=int(counts[self.head])/samples,
            head_roll_peak_abs_degrees=math.degrees(float(self.peak_head_abs.item())),
            peak_nominal_excursion_degrees=max(map(math.degrees, peaks)),
            longest_observed_safe_run_seconds=int(self.peak_safe_run.item())*self.dt,
            ended_episodes=int(ended[0]), ended_full_60s_episodes=int(ended[1]),
            ended_full_60s_no_joint_violation=int(ended[2]),
            ended_episodes_with_joint_violation=int(ended[3]),
            accepted_policy_evaluation=False,
            joints=[dict(name=n, violation_samples=int(c), peak_excursion_degrees=math.degrees(p))
                    for n,c,p in zip(self.names,counts,peaks)])
        for name in ('bad_joint_samples','bad_world_samples','peak_excursion',
                     'peak_head_abs','peak_safe_run','ended'):
            setattr(self, name, self.t.zeros_like(getattr(self, name)))
        self.steps = 0
        return result
