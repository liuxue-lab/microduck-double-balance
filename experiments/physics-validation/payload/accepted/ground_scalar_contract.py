"""Bounded step doubling and contact-event refinement, independent of GPU APIs.

Trial states are unpublished. Only the final accepted trial may advance time.
Contact brackets bound changes of sampled constraint occupancy, not exact roots.
"""
import json
import math
from pathlib import Path
import numpy as np
from lifecycle_support import require
from convergence_contract import PAIRS, quaternion_angle, criteria as numerical_criteria

CASES=('head-exit','ground-lift')
MACROS=100
CONTROLS=10
INTEGRATION_CAP=83712
TRIAL_CAP=842
ENDPOINT_CAP=886
FORWARD_CAP=335735
ROLLBACK_CAP=443
CLOCK_CAP=1242
GRID_CAPS={32:6400,64:12800,128:51456,256:8704,512:5120}
RETRY_CAPS={128:200,256:32,512:8}
WORST_CASE_STEPS={32:6400,64:12800,128:51200,256:8192,512:5120}

def criteria():
    return json.loads(Path(__file__).with_name('adaptive-criteria.json').read_text())

def state_error(a,b):
    aq,bq=np.asarray(a['qpos']),np.asarray(b['qpos'])
    av,bv=np.asarray(a['qvel']),np.asarray(b['qvel'])
    require(aq.shape==bq.shape==(35,) and av.shape==bv.shape==(32,), 'Adaptive state shape differs')
    require(all(np.isfinite(x).all() for x in (aq,bq,av,bv)), 'Nonfinite adaptive state')
    v={k:0. for k in numerical_criteria()['state_limits']}
    for qa,va in ((0,0),(21,20),(28,26)):
        v['free_translation_m']=max(v['free_translation_m'],float(np.linalg.norm(aq[qa:qa+3]-bq[qa:qa+3])))
        v['free_rotation_rad']=max(v['free_rotation_rad'],quaternion_angle(aq[qa+3:qa+7],bq[qa+3:qa+7]))
        v['free_linear_velocity_m_s']=max(v['free_linear_velocity_m_s'],float(np.linalg.norm(av[va:va+3]-bv[va:va+3])))
        v['angular_velocity_rad_s']=max(v['angular_velocity_rad_s'],float(np.linalg.norm(av[va+3:va+6]-bv[va+3:va+6])))
    v['hinge_position_rad']=float(np.max(abs(aq[7:21]-bq[7:21])))
    v['angular_velocity_rad_s']=max(v['angular_velocity_rad_s'],float(np.max(abs(av[6:20]-bv[6:20]))))
    limits=criteria()['local_state_limits']
    checks=[dict(metric=k,value=x,limit=limits[k],passed=x<=limits[k]) for k,x in v.items()]
    return dict(passed=all(c['passed'] for c in checks),checks=checks)

def signature(row):
    """Boolean occupancy, not manifold point count or positive normal force."""
    pairs=set()
    require(len(row['contact_geom'])==len(row['contact_type']), 'Contact signature shape differs')
    for g,t in zip(row['contact_geom'],row['contact_type']):
        if int(t)&1:pairs.add(tuple(sorted(map(int,g))))
    return {name:pair in pairs for name,pair in PAIRS.items()}

def brackets(samples,n):
    require(n in (32,64,128,256,512) and len(samples)==n+1, 'Incomplete event samples')
    require(all(set(s)==set(PAIRS) for s in samples), 'Event pair roster differs')
    return [dict(pair=p,direction='enter' if samples[i][p] else 'leave',
                 left_s=(i-1)*.002/n,right_s=i*.002/n,width_s=.002/n)
            for i in range(1,n+1) for p in PAIRS if samples[i][p]!=samples[i-1][p]]

def event_agreement(a,b,tolerance):
    checks=[]
    for p in PAIRS:
        x=[r for r in a if r['pair']==p];y=[r for r in b if r['pair']==p]
        same=len(x)==len(y) and all(u['direction']==v['direction'] for u,v in zip(x,y))
        delta=max((abs((u['left_s']+u['right_s']-v['left_s']-v['right_s'])/2) for u,v in zip(x,y)),default=0.)
        checks.append(dict(pair=p,coarse_events=len(x),fine_events=len(y),ordered_directions_equal=same,
                           max_midpoint_difference_s=delta,limit_s=tolerance,passed=same and delta<=tolerance))
    return dict(passed=all(c['passed'] for c in checks),checks=checks)

def assess(a,b):
    rules=criteria();error=state_error(a['endpoint'],b['endpoint'])
    event=event_agreement(a['brackets'],b['brackets'],rules['local_event_agreement_s'])
    occupancy_equal=(a['event_samples'][0]==b['event_samples'][0] and a['event_samples'][-1]==b['event_samples'][-1])
    width=max((r['width_s'] for r in b['brackets']),default=0.)
    reasons=[]
    if not error['passed']:reasons.append('local_state_error')
    if not event['passed']:reasons.append('event_disagreement')
    if not occupancy_equal:reasons.append('event_occupancy_mismatch')
    if width>rules['accepted_event_bracket_width_s']:reasons.append('event_bracket_width')
    return dict(accepted=not reasons,reasons=reasons,state_error=error,event_agreement=event,
                initial_and_final_occupancy_equal=occupancy_equal,max_bracket_width_s=width)

def advance(run_trial,rollback,emit,replay_requested=False):
    """Callbacks own snapshots. At most 32+64+128+256+512+one replay per macro; global tier caps also apply.

    Exceptions propagate after preserving the last emitted trial; no implicit
    retry. A measured max-grid failure restores the starting state and stops.
    """
    a=run_trial(32,'estimate');emit('estimate_complete',dict(trial=a['id']))
    rollback('discard_estimator')
    b=run_trial(64,'candidate');decision=assess(a,b)
    emit('candidate_decision',dict(trial=b['id'],comparison_trial=a['id'],**decision))
    rejected=not decision['accepted'];replayed=False;retry_resolutions=[]
    if rejected:
        for n in (128,256,512):
            rollback('reject_candidate' if n==128 else 'reject_retry')
            c=run_trial(n,'retry');retry_resolutions.append(n);decision=assess(b,c)
            emit('retry_decision',dict(trial=c['id'],comparison_trial=b['id'],**decision))
            b=c
            if decision['accepted']:break
        if not decision['accepted']:
            rollback('max_grid_failure')
            raise RuntimeError('Adaptive acceptance failed at maximum resolution; starting Data restored; preserve evidence')
        if replay_requested:
            rollback('replay_from_identical_start')
            c=run_trial(b['resolution'],'replay')
            exact=c['state_hashes']==b['state_hashes'] and c['stage_sha256']==b['stage_sha256']
            emit('replay_comparison',dict(trial=c['id'],original_trial=b['id'],all_numeric_Data_exact=exact))
            if not exact:rollback('replay_mismatch')
            require(exact, 'Adaptive rejected-step replay differs')
            b=c;replayed=True
    emit('accept',dict(trial=b['id'],resolution=b['resolution'],rejected_candidate=rejected,replayed=replayed))
    return b,dict(rejected=rejected,replayed=replayed,retry_resolutions=retry_resolutions,decision=decision)


def merge_integral(target,source):
    target.stages+=source['stages'];target.duration+=source['quadrature_duration_s']
    for p in PAIRS:
        for key in ('normal_impulse_Ns','loaded_time_s'):target.pairs[p][key]+=source['pairs'][p][key]
        target.pairs[p]['peak_overlap_m']=max(target.pairs[p]['peak_overlap_m'],source['pairs'][p]['peak_overlap_m'])
