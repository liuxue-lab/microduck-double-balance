"""Read-only 60-second stability accounting. No force scheduler or training."""
import math
import struct

from demo_common import HORIZON, need

DT = .02
SETTLE_STEPS = 100
OUTCOMES = {
    'RUNNING': '运行中',
    'COMPLETED_60S_NO_PHYSICAL_TERMINATION': '完成60秒；未触发机器人跌落或头球丢失等物理终止',
    'PHYSICAL_TERMINATION': '发生物理终止，未完成无扰动保持',
    'COMPLETED_60S_HELD_AFTER_SETTLE': '完成60秒；首次连续稳定2秒后没有再次越界',
    'COMPLETED_60S_WITH_STABILITY_BREAKS': '完成60秒，但稳定门限有反复越界',
    'COMPLETED_60S_NO_2S_STABILITY': '完成60秒，但未形成连续2秒稳定',
    'INTERRUPTED': '提前关闭或中断，不能作为完整60秒结果',
}


def violations(values):
    # Probe values come from float32 tensors. Match Torch's scalar rounding at
    # inclusive boundaries; keep the original cached boolean authoritative.
    f32 = lambda value: struct.unpack('<f', struct.pack('<f', value))[0]
    return dict(
        top_center=values['top_center_m'] > f32(.012),
        top_height=values['top_height_error_m'] > f32(.006),
        top_speed=values['top_speed_m_s'] > f32(.08),
        lower_offset=values['lower_offset_m'] > f32(.06),
        lower_height=not f32(.20) <= values['lower_height_m'] <= f32(.29),
        robot_tilt=values['robot_tilt_rad'] > f32(math.radians(20.)),
        lower_ball_speed=values['lower_ball_speed_m_s'] > f32(.15),
        assistance=values['hold'] > f32(1e-6),
    )


class Trial:
    def __init__(self, episode, seed, initial_xy):
        self.episode, self.seed = episode, seed
        self.steps = self.stable_steps = self.stable_run = self.longest_stable = 0
        self.unstable_run = self.longest_unstable = self.post_settle_unstable = 0
        self.stability_breaks_after_settle = 0
        self.first_stable_step = self.settle_step = self.last_unstable_step = None
        self.first_unstable_after_settle = None
        self.initial_xy = [float(x) for x in initial_xy]
        need(len(self.initial_xy) == 2 and all(math.isfinite(x) for x in self.initial_xy), 'Invalid reset origin')
        self.settle_xy = self.final = None
        self.max_drift_reset = self.max_drift_settle = 0.
        self.violations_all, self.violations_after_settle = {}, {}
        self.minima, self.maxima = {}, {}
        self.ended, self.reason, self.physical, self.terms = False, None, False, []
        self.phase = 'OBSERVING_INITIAL_STABILITY'

    def complete(self, sample):
        need(not self.ended and self.steps < HORIZON, 'No active baseline step')
        values, xy = sample['values'], sample['root_xy']
        need(all(math.isfinite(float(x)) for x in list(values.values()) + list(xy)), 'Nonfinite baseline measurement')
        failed = violations(values)
        stable = bool(sample['stable'])
        need(stable == (not any(failed.values())), 'Baseline gate differs from original metric')
        self.steps += 1
        self.final = sample
        for name, value in values.items():
            self.minima[name] = min(value, self.minima.get(name, value))
            self.maxima[name] = max(value, self.maxima.get(name, value))
        for name, flag in failed.items():
            self.violations_all[name] = self.violations_all.get(name, 0) + int(flag)
            self.violations_after_settle.setdefault(name, 0)
            if self.settle_step is not None:
                self.violations_after_settle[name] += int(flag)
        self.max_drift_reset = max(self.max_drift_reset, math.dist(xy, self.initial_xy))
        if self.settle_xy is not None:
            self.max_drift_settle = max(self.max_drift_settle, math.dist(xy, self.settle_xy))
        if stable:
            self.stable_steps += 1
            self.stable_run += 1
            self.unstable_run = 0
            self.longest_stable = max(self.longest_stable, self.stable_run)
            if self.first_stable_step is None: self.first_stable_step = self.steps
            if self.settle_step is None and self.stable_run >= SETTLE_STEPS:
                self.settle_step, self.settle_xy = self.steps, list(xy)
        else:
            if self.settle_step is not None:
                self.post_settle_unstable += 1
                if self.unstable_run == 0: self.stability_breaks_after_settle += 1
                if self.first_unstable_after_settle is None: self.first_unstable_after_settle = self.steps
            self.stable_run = 0
            self.unstable_run += 1
            self.longest_unstable = max(self.longest_unstable, self.unstable_run)
            self.last_unstable_step = self.steps
        self.phase = ('STABLE' if stable else 'STABILITY_GATE_VIOLATION') if self.settle_step is not None else 'OBSERVING_INITIAL_STABILITY'

    def finish(self, reason, physical=False, terms=()):
        if self.ended: return
        self.ended, self.reason, self.physical, self.terms = True, reason, bool(physical), list(terms)
        self.phase = 'ENDED'

    def snapshot(self):
        completed = self.ended and not self.physical and self.steps == HORIZON and self.reason == 'time_out' and 'time_out' in self.terms
        if not self.ended: outcome = 'RUNNING'
        elif self.physical: outcome = 'PHYSICAL_TERMINATION'
        elif not completed: outcome = 'INTERRUPTED'
        elif self.settle_step is None: outcome = 'COMPLETED_60S_NO_2S_STABILITY'
        elif self.post_settle_unstable: outcome = 'COMPLETED_60S_WITH_STABILITY_BREAKS'
        else: outcome = 'COMPLETED_60S_HELD_AFTER_SETTLE'
        geometric_summary = outcome
        if completed: outcome = 'COMPLETED_60S_NO_PHYSICAL_TERMINATION'
        seconds = lambda step: None if step is None else step * DT
        after = 0 if self.settle_step is None else self.steps - self.settle_step
        return dict(
            episode=self.episode, seed=self.seed, phase=self.phase, outcome=outcome,
            goal_60s_no_physical_termination_met=completed,
            goal='Keep robot on basketball and head ball retained for 60s; world translation is allowed',
            geometric_stability_summary=geometric_summary,
            geometric_thresholds_are_secondary=True,
            displacement_is_failure_criterion=False,
            original_drop_termination_triggered=any(t in ('fell', 'top_ball_lost') for t in self.terms),
            control_steps=self.steps, duration_seconds=self.steps * DT, completed_60_seconds=completed,
            stable_control_steps=self.stable_steps,
            stable_fraction=None if not self.steps else self.stable_steps / self.steps,
            longest_continuous_stable_seconds=self.longest_stable * DT,
            final_continuous_stable_seconds=self.stable_run * DT,
            longest_continuous_unstable_seconds=self.longest_unstable * DT,
            first_stable_sample_seconds=seconds(self.first_stable_step),
            first_2s_stability_confirmed_at_seconds=seconds(self.settle_step),
            post_settle_observed_seconds=after * DT,
            post_settle_unstable_steps=self.post_settle_unstable,
            post_settle_stable_fraction=None if not after else (after-self.post_settle_unstable) / after,
            stability_breaks_after_settle=self.stability_breaks_after_settle,
            first_unstable_after_settle_seconds=seconds(self.first_unstable_after_settle),
            last_unstable_sample_seconds=seconds(self.last_unstable_step),
            violation_steps_by_gate=dict(self.violations_all),
            violation_steps_by_gate_after_settle=dict(self.violations_after_settle),
            minimum_values=dict(self.minima), maximum_values=dict(self.maxima),
            reset_root_xy_m=list(self.initial_xy), first_settled_root_xy_m=self.settle_xy,
            max_root_drift_from_reset_m=self.max_drift_reset,
            final_root_drift_from_reset_m=None if self.final is None else math.dist(self.final['root_xy'], self.initial_xy),
            max_root_drift_from_first_settle_m=None if self.settle_xy is None else self.max_drift_settle,
            final_root_drift_from_first_settle_m=None if self.settle_xy is None else math.dist(self.final['root_xy'], self.settle_xy),
            final_sample=self.final, end_reason=self.reason, physically_terminated=self.physical,
            terminations=list(self.terms), applied_disturbance_pulses=0,
            formal_acceptance=False,
            measurement_convention='50-Hz original pre-forward geometric gate; run duration = sample count x 0.02s; drift is descriptive, not a position-return acceptance',
        )
