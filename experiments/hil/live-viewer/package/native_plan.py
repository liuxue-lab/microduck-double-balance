"""Fixed candidate selection and independent heldout cases for the final 6-hour run."""
import math
import random

PHYSICS_DT = .005
HORIZON_PHYSICS_STEPS = 12000
PERIOD_STEPS = 400
PULSE_STEPS = 20
FORCE_FRACTION = .1
MAX_EVALUATIONS = 27
MAX_CASE_SECONDS = 900
ENVIRONMENT_SEEDS = (2026102406, 2026102407)
FORCE_SEEDS = (2026102500, 2026102700, *range(2026102800, 2026102816))
SELECTION_CASES = (
    dict(case_id='quiet-2407', seed=2026102407, force_seed=None),
    dict(case_id='regression-2500', seed=2026102407, force_seed=2026102500),
    dict(case_id='selection-2700', seed=2026102407, force_seed=2026102700),
)
HELDOUT_CASES = (
    dict(case_id='quiet-2406', seed=2026102406, force_seed=None),
    dict(case_id='quiet-2407', seed=2026102407, force_seed=None),
    *(dict(case_id='push-%02d' % i, seed=2026102407, force_seed=2026102800+i) for i in range(16)),
)


def need(ok, message):
    if not ok: raise RuntimeError(message)


def case_plan(phase):
    need(phase in ('selection','heldout'), 'Unknown evaluation phase')
    return [dict(v) for v in (SELECTION_CASES if phase=='selection' else HELDOUT_CASES)]


def make_schedule(force_seed):
    if force_seed is None: return []
    need(type(force_seed) is int and force_seed in FORCE_SEEDS, 'Unapproved evaluation force seed')
    rng=random.Random(force_seed)
    rows=[]
    for number,start in enumerate(range(PERIOD_STEPS,HORIZON_PHYSICS_STEPS,PERIOD_STEPS),1):
        angle=rng.uniform(0.,2.*math.pi)
        rows.append(dict(pulse=number,start_physics_step=start,end_physics_step_exclusive=start+PULSE_STEPS,
            start_seconds=start*PHYSICS_DT,end_seconds=(start+PULSE_STEPS)*PHYSICS_DT,
            expected_physics_steps=PULSE_STEPS,direction_angle_world_degrees=math.degrees(angle),
            unit_force_world=[math.cos(angle),math.sin(angle),0.]))
    return rows


def pulse_index(completed_physics_steps):
    need(type(completed_physics_steps) is int and 0<=completed_physics_steps<HORIZON_PHYSICS_STEPS,
         'Physics step outside one authorized episode')
    period,offset=divmod(completed_physics_steps,PERIOD_STEPS)
    return period-1 if period>=1 and offset<PULSE_STEPS else None


def rank_candidate(result):
    need(result.get('phase')=='selection','Only the selection set may choose a candidate')
    s=result.get('summary') or {}
    qualified=s.get('qualified') is True
    margin=s.get('minimum_joint_margin_fraction')
    margin=margin if isinstance(margin,(int,float)) and math.isfinite(margin) else -math.inf
    duration=s.get('minimum_retention_seconds',0.)
    excursion=s.get('maximum_excursion_rad')
    excursion=excursion if isinstance(excursion,(int,float)) and math.isfinite(excursion) else math.inf
    # Qualification dominates. No drift term is part of selection.
    return (int(qualified),int(s.get('quiet_passes',0)),int(s.get('push_passes',0)),
            int(s.get('model_boundary_passes',0)),float(duration),-float(excursion),
            float(margin),int(result.get('checkpoint',{}).get('policy_updates',0)))


def choose_candidate(results):
    from eval_criteria import summarize_phase
    need(bool(results),'No candidate evaluation is available')
    need(all(r.get('status')=='EVALUATION_COMPLETE' and r.get('phase')=='selection'
             and (r.get('summary') or {}).get('complete_set') is True for r in results),
         'Incomplete or erroneous selections cannot choose a model')
    shas=[r['checkpoint']['sha256'] for r in results]
    need(len(shas)==len(set(shas)) and len(shas)<=3,'Duplicate or excess candidate selection')
    for result in results:
        actual=summarize_phase(result.get('cases') or [],'selection',result['checkpoint']['sha256'])
        need(actual==result['summary'],'Candidate summary differs from retained case evidence')
    best=max(results,key=rank_candidate)
    return dict(checkpoint=best['checkpoint'],qualified=best['summary']['qualified'],
                selection=best,ranking=[v if math.isfinite(v) else None for v in rank_candidate(best)],
                automatic_training=False,heldout_can_run=best['summary']['qualified'],
                best_of_evaluated_candidates_only=True)

