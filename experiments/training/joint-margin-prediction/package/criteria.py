"""Joint goal: keep both balls for 60s AND pass checked model boundaries."""
import math


def accept_episode(result):
    trial = result.get('baseline_trial') or {}
    limits = result.get('model_limits') or {}
    joints, motors = limits.get('joints', []), limits.get('motors', [])
    conditions = dict(
        completed=result.get('status') == 'EPISODE_COMPLETE',
        duration=trial.get('control_steps') == 3000 and trial.get('completed_60_seconds') is True,
        retention=trial.get('goal_60s_no_physical_termination_met') is True
                  and trial.get('physically_terminated') is False
                  and trial.get('terminations') == ['time_out'],
        physics=result.get('observed_physics_steps') == 30000
                and limits.get('completed_physics_steps') == 30000,
        audit_counts=limits.get('motor_checks_before_integration') == 30000
                     and limits.get('joint_state_checks') == 30001,
        no_event=limits.get('any_model_bound_violation_observed') is False
                 and limits.get('first_limit_event') is None
                 and limits.get('stopping_limit_event') is None,
        complete_joint_rows=len(joints) == 14 and len({j.get('name') for j in joints}) == 14,
        complete_motor_rows=len(motors) == 14 and len({m.get('name') for m in motors}) == 14,
        no_joint_excursion=all(j.get('limit_violation_samples') == 0
                              and j.get('initial_limit_violation') is False
                              and math.isfinite(j.get('peak_nominal_excursion_rad',float('nan')))
                              and 0 <= j['peak_nominal_excursion_rad'] <= 32*2**-23*max(
                                  1.,abs(j.get('min_q_rad',0.)),abs(j.get('max_q_rad',0.)))
                              for j in joints),
        zero_wrench=result.get('zero_wrench_checks_before_physics') == 30000
                    and result.get('disturbance_enabled') is False
                    and result.get('applied_disturbance_pulses') == 0
                    and result.get('external_body_wrench_zero_at_close') is True,
        unchanged=result.get('policy_and_normalizer_unchanged') is True
                  and result.get('checkpoint_unchanged') is True,
        no_error=result.get('error') is None and result.get('cleanup_error') is None,
    )
    return dict(passed=all(conditions.values()), conditions=conditions,
                position_drift_allowed=True, hardware_acceptance=False,
                strict_contact_acceptance=False)


def summarize(results, expected_seeds, checkpoint_sha256):
    seeds=[r['seed'] for r in results]
    if seeds != list(expected_seeds) or len(set(seeds)) != len(seeds):
        raise RuntimeError('Incomplete or duplicate evaluation seed set')
    if any(r['checkpoint']['sha256'] != checkpoint_sha256 for r in results):
        raise RuntimeError('Mixed evaluation checkpoints')
    outcomes=[accept_episode(r) for r in results]
    return dict(episodes=len(results), passes=sum(x['passed'] for x in outcomes),
                all_passed=all(x['passed'] for x in outcomes), seed_results=outcomes,
                population_success_rate_claimed=False, hardware_acceptance=False)
