"""Whole singleton state seals and original-criterion historical comparisons."""
import copy,hashlib,json
import numpy as np
from lifecycle_support import require
from scene_evidence import array_meta
from ground_scalar_contract import assess
from convergence_contract import compare

def seal(record,model_metadata):
    groups={n:array_meta(record[n]) for n in ('data','owners','rng')}
    groups.update(scalars=record['scalars'],topology=record['topology'],model=model_metadata)
    if record.get('outputs') is not None:groups['outputs']=array_meta(record['outputs'])
    if record.get('delayed') is not None:groups['delayed']=array_meta({'BAM':record['delayed']})
    return hashlib.sha256(json.dumps(groups,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def global_timer_seal(record):
    key='RNGService.states.event_global_2/0'
    raw={key:record['owners'][key], 'timer':record['owners']['EventManager._interval_term_time_left.2'], 'events':record['owners']['env.events'][3]}
    value=dict(arrays=array_meta(raw),draws=record['scalars']['RNGService.draw_counts.event_global_2/0'],seed=record['scalars']['RNGService.seeds.event_global_2/0'])
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def historical_agreement(accepted,post,reference,logical_world,*,require_pass=True):
    require(len(accepted['worlds'])==1 and accepted['resolution']==128,'Historical comparison requires a complete singleton128 trial')
    a=accepted['worlds'][0];b=copy.deepcopy(reference['worlds'][logical_world])
    for n in ('qpos','qvel'):b['endpoint'][n]=np.asarray(b['endpoint'][n],np.float32)
    local=assess(a,b)
    current_time=float(post['data']['Data.time'][0]);reference_time=reference['native_times'][logical_world]
    require(current_time==reference_time,'Singleton/historical publication clocks differ')
    def point(trial,time):return dict(tick=1,native_time_s=time,qpos=trial['endpoint']['qpos'],qvel=trial['endpoint']['qvel'])
    agreement=compare([point(a,current_time)],[point(b,reference_time)],a['integral'],b['integral'],expected_macros=1,max_resolution=128)
    passed=bool(local['accepted'] and agreement['passed'])
    if require_pass:require(passed,'Singleton/previous paired physical reference exceeds original criteria')
    return dict(logical_world=logical_world,source_member=reference['source_member'],source_sha256=reference['source_sha256'],local_state_event=local,state_contact=agreement,passed=passed,original_criteria_unchanged=True,complete_opaque_history_equivalence_claimed=False)
