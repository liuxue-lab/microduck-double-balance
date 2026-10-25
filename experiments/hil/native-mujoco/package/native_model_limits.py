"""Model-boundary accounting, independent of GPU libraries and policy state."""
import math

from demo_common import need

PHYSICS_DT = .005
FLOAT32_EPS = 2.0**-23


def tolerance(*values):
    # Preserve finite filtering, positive-zero ties and the original cold error.
    scale=1.;seen=False
    for x in values:
        if math.isfinite(x):
            seen=True
            value=abs(x)
            if value>scale:scale=value
    if not seen:return 32*FLOAT32_EPS*max(1.)
    return 32*FLOAT32_EPS*scale


def excursion(value, lower, upper):
    low=0. if lower is None else lower-value
    high=0. if upper is None else value-upper
    result=low if low>0. else 0.
    return high if high>result else result


class LimitStop(BaseException):
    """Leave the viewer immediately; its ordinary error handler must not resume."""
    def __init__(self, event):
        self.event = event
        super().__init__(event['kind'] + ': ' + event['name'])


class Audit:
    def __init__(self, joints, motors, *, joint_violation_handling='stop'):
        need(joint_violation_handling in ('stop', 'record_and_continue_simulation'), 'Invalid joint observation mode')
        self.joint_violation_handling = joint_violation_handling
        self.joints, self.motors = joints, motors
        self.completed_physics_steps = self.motor_checks = self.joint_checks = 0
        self.first_event = self.stop_event = None
        self.latest_motor_samples = {}
        self.joint_stats = {j['name']: dict(min_q_rad=None, max_q_rad=None, peak_speed_rad_s=0.,
            peak_nominal_excursion_rad=0., minimum_hard_margin_rad=None,
            soft_limit_samples=0, current_soft_run=0, longest_soft_run=0,
            limit_violation_samples=0, violation_episodes=0, initial_limit_violation=False,
            current_violation_run=0, longest_violation_run=0, outside_at_last_sample=False,
            first_violation_seconds=None, last_violation_seconds=None, last_reentry_seconds=None,
            first_violation_context=None, last_q_rad=None, last_speed_rad_s=None) for j in joints}
        self.motor_stats = {m['name']: dict(peak_modeled_torque_Nm=0., peak_applied_torque_Nm=0.,
            peak_modeled_current_A=0., peak_speed_rad_s=0., peak_pwm_fraction=0.,
            pwm_saturated_samples=0, current_pwm_run=0, longest_pwm_run=0,
            force_saturated_samples=0, current_force_run=0, longest_force_run=0,
            target_outside_joint_range_samples=0) for m in motors}

    def event(self, kind, name, value, bounds, phase, seconds):
        event = dict(kind=kind, name=name, value=value, bounds=bounds,
                     phase=phase, simulated_seconds=seconds,
                     completed_physics_steps=self.completed_physics_steps)
        if self.first_event is None: self.first_event = event
        return event

    def stop(self, kind, name, value, bounds, phase, seconds):
        event = self.event(kind, name, value, bounds, phase, seconds)
        if self.stop_event is None: self.stop_event = event
        # A prior recorded joint excursion must not hide a later electrical or
        # numerical fault that actually stops integration.
        raise LimitStop(self.stop_event)

    def observe_joints(self, positions, velocities, *, initial=False):
        need(len(positions) == len(velocities) == len(self.joints), 'Joint mapping changed')
        self.joint_checks += 1
        when = self.completed_physics_steps * PHYSICS_DT
        for spec, q, speed in zip(self.joints, positions, velocities):
            name, lo, hi = spec['name'], spec['lower_rad'], spec['upper_rad']
            if not math.isfinite(q) or not math.isfinite(speed):
                self.stop('NONFINITE_JOINT_STATE', name, None, None, 'reset' if initial else 'after_integration', when)
            stats = self.joint_stats[name]
            stats['min_q_rad'] = q if stats['min_q_rad'] is None else min(q, stats['min_q_rad'])
            stats['max_q_rad'] = q if stats['max_q_rad'] is None else max(q, stats['max_q_rad'])
            stats['last_q_rad'], stats['last_speed_rad_s'] = q, speed
            stats['peak_speed_rad_s'] = max(abs(speed), stats['peak_speed_rad_s'])
            excess = excursion(q, lo, hi)
            stats['peak_nominal_excursion_rad'] = max(excess, stats['peak_nominal_excursion_rad'])
            margins = ([q-lo] if lo is not None else []) + ([hi-q] if hi is not None else [])
            if margins:
                margin = min(margins)
                previous = stats['minimum_hard_margin_rad']
                stats['minimum_hard_margin_rad'] = margin if previous is None else min(margin, previous)
            soft = excursion(q, spec['soft_lower_rad'], spec['soft_upper_rad']) > tolerance(q)
            if not initial:
                stats['soft_limit_samples'] += int(soft)
                stats['current_soft_run'] = stats['current_soft_run']+1 if soft else 0
                stats['longest_soft_run'] = max(stats['longest_soft_run'], stats['current_soft_run'])
            numerical_tolerance = tolerance(q, *(x for x in (lo, hi) if x is not None))
            outside = excess > numerical_tolerance
            previous_outside = stats['outside_at_last_sample']
            stats['outside_at_last_sample'] = outside
            if outside:
                stats['violation_episodes'] += int(not previous_outside)
                stats['last_violation_seconds'] = when
                if stats['first_violation_seconds'] is None:
                    stats['first_violation_seconds'] = when
                    stats['first_violation_context'] = dict(position_rad=q, velocity_rad_s=speed,
                        before_integration_motor_sample=self.latest_motor_samples.get(name),
                        motor_sample_time_s=max(0., when-PHYSICS_DT) if not initial else None)
                if initial:
                    stats['initial_limit_violation'] = True
                else:
                    stats['limit_violation_samples'] += 1
                    stats['current_violation_run'] += 1
                    stats['longest_violation_run'] = max(stats['longest_violation_run'], stats['current_violation_run'])
                record = self.stop if self.joint_violation_handling == 'stop' else self.event
                record('JOINT_POSITION_LIMIT', name, q, dict(lower=lo, upper=hi,
                    overshoot_rad=excess, numerical_tolerance_rad=numerical_tolerance),
                    'reset' if initial else 'after_integration', when)
            else:
                stats['current_violation_run'] = 0
                if previous_outside: stats['last_reentry_seconds'] = when

    def before_integration(self, samples):
        need(len(samples) == len(self.motors), 'Motor mapping changed')
        self.motor_checks += 1
        when = self.completed_physics_steps * PHYSICS_DT
        for spec, point in zip(self.motors, samples):
            name = spec['name']
            if not all(math.isfinite(x) for x in point.values()):
                self.stop('NONFINITE_MOTOR_STATE', name, None, None, 'before_integration', when)
            self.latest_motor_samples[name] = dict(point)
            voltage, omega, kd, duty, tau = (point[k] for k in ('voltage', 'velocity', 'kd_scale', 'duty', 'torque'))
            need(voltage > 0 and kd > 0, 'Invalid BAM voltage/gain')
            kt, resistance, pwm = spec['kt'], spec['resistance_ohm'], spec['max_pwm']
            expected = kt / resistance * (voltage*duty - kt*omega*kd)
            # Match the torque that BAM actually supplied, not an imagined
            # unsaturated policy request or a motor's no-load speed label.
            need(abs(tau-expected) <= tolerance(tau, expected), 'BAM diagnostic torque formula changed')
            need(abs(point['ctrl']-tau) <= tolerance(tau), 'BAM-to-control mapping changed')
            current = tau / kt
            stats = self.motor_stats[name]
            stats['peak_modeled_torque_Nm'] = max(abs(tau), stats['peak_modeled_torque_Nm'])
            stats['peak_modeled_current_A'] = max(abs(current), stats['peak_modeled_current_A'])
            stats['peak_speed_rad_s'] = max(abs(omega), stats['peak_speed_rad_s'])
            stats['peak_pwm_fraction'] = max(abs(duty)/pwm, stats['peak_pwm_fraction'])
            saturated = abs(duty) >= pwm-tolerance(pwm)
            stats['pwm_saturated_samples'] += int(saturated)
            stats['current_pwm_run'] = stats['current_pwm_run']+1 if saturated else 0
            stats['longest_pwm_run'] = max(stats['longest_pwm_run'], stats['current_pwm_run'])
            target = point['position'] + point['position_error']
            joint = self.joints[spec['local_joint_id']]
            stats['target_outside_joint_range_samples'] += int(excursion(target, joint['lower_rad'], joint['upper_rad']) > tolerance(target))
            if abs(duty) > pwm+tolerance(pwm):
                self.stop('PWM_LIMIT', name, duty, [-pwm, pwm], 'before_integration', when)
            cap = spec['max_current_A']
            if cap is not None and abs(current) > cap+tolerance(current, cap):
                self.stop('MODELED_CURRENT_LIMIT', name, current,
                    dict(max_abs_A=cap, note='Voltage saturation/back-EMF can defeat the model current limiter'),
                    'before_integration', when)

    def after_integration(self, positions, velocities, applied_forces):
        self.completed_physics_steps += 1
        need(len(applied_forces) == len(self.motors), 'Applied force mapping changed')
        for spec, value in zip(self.motors, applied_forces):
            if not math.isfinite(value):
                self.stop('NONFINITE_APPLIED_TORQUE', spec['name'], None, None,
                          'after_integration', self.completed_physics_steps * PHYSICS_DT)
            lo, hi = spec['force_lower_Nm'], spec['force_upper_Nm']
            stats = self.motor_stats[spec['name']]
            stats['peak_applied_torque_Nm'] = max(abs(value), stats['peak_applied_torque_Nm'])
            saturated = (lo is not None and value <= lo+tolerance(lo)) or (hi is not None and value >= hi-tolerance(hi))
            stats['force_saturated_samples'] += int(saturated)
            stats['current_force_run'] = stats['current_force_run']+1 if saturated else 0
            stats['longest_force_run'] = max(stats['longest_force_run'], stats['current_force_run'])
            if excursion(value, lo, hi) > tolerance(value):
                self.stop('APPLIED_TORQUE_LIMIT', spec['name'], value, [lo, hi],
                          'after_integration', self.completed_physics_steps * PHYSICS_DT)
        self.observe_joints(positions, velocities)

    def snapshot(self):
        joint_rows = []
        for name, stats in self.joint_stats.items():
            joint_rows.append(dict(name=name, **stats,
                longest_soft_limit_seconds=stats['longest_soft_run']*PHYSICS_DT,
                sampled_limit_violation_seconds=stats['limit_violation_samples']*PHYSICS_DT,
                longest_limit_violation_seconds=stats['longest_violation_run']*PHYSICS_DT,
                peak_nominal_excursion_deg=math.degrees(stats['peak_nominal_excursion_rad'])))
        motor_rows = []
        for name, stats in self.motor_stats.items():
            motor_rows.append(dict(name=name, **stats,
                longest_pwm_saturation_seconds=stats['longest_pwm_run']*PHYSICS_DT,
                longest_force_saturation_seconds=stats['longest_force_run']*PHYSICS_DT))
        status = ('MODEL_LIMIT_STOPPED' if self.stop_event else
                  'MODEL_BOUND_VIOLATION_OBSERVED' if self.first_event else
                  'OBSERVED_WITHIN_CHECKED_MODEL_BOUNDS' if self.completed_physics_steps else 'NOT_YET_OBSERVED')
        return dict(status=status, first_limit_event=self.first_event, stopping_limit_event=self.stop_event,
                    joint_violation_handling=self.joint_violation_handling,
                    any_model_bound_violation_observed=self.first_event is not None,
                    joint_violations_permanently_fail_acceptance=True,
                    completed_physics_steps=self.completed_physics_steps,
                    simulated_seconds=self.completed_physics_steps*PHYSICS_DT,
                    motor_checks_before_integration=self.motor_checks, joint_state_checks=self.joint_checks,
                    joints=joint_rows, motors=motor_rows,
                    standalone_speed_limit_verified=False, thermal_and_continuous_ratings_verified=False,
                    real_hardware_limits_verified=False,
                    notes=['Speed is evaluated jointly with modeled voltage/current/torque; no-load speed is not a hard speed cap.',
                           'Soft-limit proximity and saturation are recorded, not silently promoted to hardware ratings.',
                           'Actual joint positions are checked; wide servo position targets are separately recorded.',
                           '32 float32 epsilons at the value scale are numerical tolerance only.'])
