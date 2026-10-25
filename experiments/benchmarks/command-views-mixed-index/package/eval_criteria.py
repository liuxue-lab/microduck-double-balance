"""Strict frozen-policy selection and heldout acceptance; no reward-based pass."""
import math
from eval_plan import case_plan, make_schedule, need


def close(a,b):
    return type(a) in (int,float) and type(b) in (int,float) and math.isfinite(a) and math.isfinite(b) \
        and math.isclose(a,b,rel_tol=1e-7,abs_tol=1e-10)


def planned_force_checks(row,case):
    force=row.get('disturbance') or {};pulses=force.get('pulses') or []
    plan=make_schedule(case['force_seed']);enabled=case['force_seed'] is not None
    checks=dict(plan_identity=force.get('force_seed')==case['force_seed']
                and force.get('disturbance_enabled') is enabled,
        exact_totals=force.get('planned_pulses')==len(plan)
                and force.get('applied_pulses')==len(plan) and force.get('completed_pulses')==len(plan)
                and force.get('applied_physics_steps')==50*len(plan)
                and force.get('checked_physics_steps')==30000 and force.get('schedule_verified') is True,
        schedule=force.get('force_fraction')==(.1 if enabled else 0.)
                and force.get('period_seconds')==2. and force.get('duration_seconds')==.1,
        complete_pulse_records=len(pulses)==len(plan),directions=True,timing=True,vectors=True,impulses=True)
    mass=force.get('robot_mass_kg');gravity=force.get('gravity_m_s2');magnitude=force.get('force_newtons')
    checks['current_robot_mass']=type(mass) in (int,float) and math.isfinite(mass) and mass>0 \
        and type(gravity) in (int,float) and math.isfinite(gravity) and gravity>0 \
        and close(magnitude,(.1 if enabled else 0.)*mass*gravity)
    for actual,wanted in zip(pulses,plan):
        for key,value in wanted.items():
            got=actual.get(key)
            if isinstance(value,list):
                checks['directions'] &= isinstance(got,list) and len(got)==len(value) \
                    and all(close(a,b) for a,b in zip(got,value))
            else: checks['directions'] &= close(got,value)
        checks['timing'] &= actual.get('complete') is True and actual.get('applied_physics_steps')==50 \
            and actual.get('force_write_attempts')==50 \
            and close(actual.get('actual_start_seconds'),wanted['start_seconds']) \
            and close(actual.get('actual_end_seconds'),wanted['end_seconds'])
        vector=actual.get('force_world_N') or []
        valid_vector=len(vector)==3 and all(type(v) in (int,float) and math.isfinite(v) for v in vector)
        checks['vectors'] &= valid_vector and vector[2]==0. \
            and close(math.hypot(*vector[:2]),magnitude) \
            and close(actual.get('actual_force_N'),magnitude) \
            and all(close(v,magnitude*u) for v,u in zip(vector,wanted['unit_force_world']))
        impulse=actual.get('applied_impulse_world_N_s') or []
        checks['impulses'] &= valid_vector and len(impulse)==3 \
            and all(close(v,f*.1) for v,f in zip(impulse,vector)) \
            and close(actual.get('applied_impulse_N_s'),magnitude*.1)
    return {k:bool(v) for k,v in checks.items()}


def accept_case(row,case,checkpoint_sha256):
    trial=row.get('trial') or {};limits=row.get('model_limits') or {}
    joints=limits.get('joints') or [];motors=limits.get('motors') or []
    force=planned_force_checks(row,case)
    def valid_joint(j):
        values=[j.get(k) for k in ('peak_nominal_excursion_rad','min_q_rad','max_q_rad')]
        return all(type(v) in (int,float) and math.isfinite(v) for v in values) \
            and j.get('limit_violation_samples')==0 and j.get('initial_limit_violation') is False \
            and 0<=values[0]<=32*2**-23*max(1.,abs(values[1]),abs(values[2]))
    conditions=dict(
        identity=row.get('case_id')==case['case_id'] and row.get('seed')==case['seed']
             and row.get('force_seed')==case['force_seed']
             and (row.get('checkpoint') or {}).get('sha256')==checkpoint_sha256,
        execution=row.get('execution_completed') is True,
        duration=row.get('completed_control_steps')==3000 and trial.get('control_steps')==3000
             and trial.get('completed_60_seconds') is True,
        retention=trial.get('goal_60s_no_physical_termination_met') is True
             and trial.get('physically_terminated') is False and trial.get('terminations')==['time_out'],
        physics=row.get('observed_physics_steps')==30000 and limits.get('completed_physics_steps')==30000,
        audit_counts=limits.get('motor_checks_before_integration')==30000 and limits.get('joint_state_checks')==30001,
        no_model_event=limits.get('any_model_bound_violation_observed') is False
             and limits.get('first_limit_event') is None and limits.get('stopping_limit_event') is None,
        complete_joint_rows=len(joints)==14 and len({j.get('name') for j in joints})==14,
        complete_motor_rows=len(motors)==14 and len({m.get('name') for m in motors})==14,
        no_joint_excursion=all(valid_joint(j) for j in joints),
        exact_force=all(force.values()),
        cleared=row.get('external_body_wrench_zero_at_close') is True,
        unchanged=row.get('policy_and_normalizer_unchanged') is True
             and row.get('checkpoint_unchanged') is True and row.get('frozen_inputs_unchanged') is True,
        no_error=not any(row.get(k) for k in ('error','cleanup_error','environment_close_error','model_fault')),
        no_training=row.get('added_PPO_updates')==0 and row.get('training_started') is False,
    )
    boundary_keys=('physics','audit_counts','no_model_event','complete_joint_rows','complete_motor_rows','no_joint_excursion')
    return dict(passed=all(conditions.values()),conditions=conditions,force_checks=force,
        checked_model_boundaries_met=all(conditions[k] for k in boundary_keys),
        retention_met=conditions['duration'] and conditions['retention'],
        position_drift_allowed=True,strict_contact_acceptance=False,hardware_acceptance=False,
        population_success_rate_claimed=False)


def joint_margin(row):
    limits=row.get('model_limits') or {};bounds=(limits.get('boundaries') or {}).get('joints') or []
    measured={j.get('name'):j for j in limits.get('joints',[])};margins=[]
    for spec in bounds:
        joint=measured.get(spec.get('name'),{})
        lo,hi=spec.get('lower_rad'),spec.get('upper_rad')
        qlo,qhi=joint.get('min_q_rad'),joint.get('max_q_rad')
        if all(type(v) in (int,float) and math.isfinite(v) for v in (lo,hi,qlo,qhi)) and hi>lo:
            margins.append(min(qlo-lo,hi-qhi)/(hi-lo))
    return min(margins) if len(margins)==14 else None


def summarize_phase(rows,phase,checkpoint_sha256):
    plan=case_plan(phase)
    need(len(rows)<=len(plan),'Too many evaluation cases')
    need([r.get('case_id') for r in rows]==[p['case_id'] for p in plan[:len(rows)]],
         'Missing, reordered or duplicated evaluation case')
    need(all((r.get('checkpoint') or {}).get('sha256')==checkpoint_sha256 for r in rows),'Mixed candidate models')
    outcomes=[accept_case(r,p,checkpoint_sha256) for r,p in zip(rows,plan)]
    complete=len(rows)==len(plan)
    margins=[joint_margin(r) for r in rows]
    margins=[v for v in margins if v is not None]
    durations=[float((r.get('trial') or {}).get('duration_seconds') or 0.) for r in rows]
    excursions=[j.get('peak_nominal_excursion_rad',0.) for r in rows
                for j in (r.get('model_limits') or {}).get('joints',[])]
    excursions=[v for v in excursions if type(v) in (int,float) and math.isfinite(v)]
    passed=complete and all(o['passed'] for o in outcomes)
    return dict(all_passed=passed,qualified=phase=='selection' and passed,complete_set=complete,
        planned_cases=len(plan),evaluated_cases=len(rows),passes=sum(o['passed'] for o in outcomes),
        quiet_passes=sum(o['passed'] for o,p in zip(outcomes,plan) if p['force_seed'] is None),
        push_passes=sum(o['passed'] for o,p in zip(outcomes,plan) if p['force_seed'] is not None),
        model_boundary_passes=sum(o['checked_model_boundaries_met'] for o in outcomes),
        minimum_joint_margin_fraction=min(margins) if margins else None,
        minimum_retention_seconds=min(durations) if durations else 0.,
        maximum_excursion_rad=max(excursions) if excursions else None,
        case_acceptance=outcomes,population_success_rate_claimed=False,
        hardware_acceptance=False,strict_contact_acceptance=False,all_possible_sequences_proven=False,
        old_64_case_suite_replaced_or_claimed_passed=False)

