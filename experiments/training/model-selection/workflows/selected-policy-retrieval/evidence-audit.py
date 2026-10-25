import hashlib
import math
import statistics
import sys
def need(ok,message):
    if not ok:raise RuntimeError(message)

GATES = ('top_center', 'top_height', 'top_speed', 'lower_offset', 'lower_height_min',
         'lower_height_max', 'robot_tilt', 'lower_ball_speed', 'unassisted')

VALUES = ('top_center_m', 'top_height_error_m', 'top_speed_m_s', 'lower_offset_m',
          'lower_height_m', 'robot_tilt_rad', 'lower_ball_speed_m_s', 'hold')

THRESHOLDS = dict(top_center_m=.012, top_height_error_m=.006, top_speed_m_s=.08,
                  lower_offset_m=.06, lower_height_min_m=.20, lower_height_max_m=.29,
                  robot_tilt_deg=20., lower_ball_speed_m_s=.15, hold=1e-6,
                  consecutive_steps=250, control_dt=.02, horizon_seconds=10.)

def summary_audit(saved,recomputed):
    """Compare derived display statistics, never physical success thresholds."""
    differences=[];invalid=[]
    def compare(a,b,path):
        if isinstance(b,dict):
            if not isinstance(a,dict) or set(a)!=set(b):
                invalid.append(dict(path=path,kind='schema'));return
            for key in sorted(b):compare(a[key],b[key],path+'.'+key)
        elif isinstance(b,list):
            if not isinstance(a,list) or len(a)!=len(b):
                invalid.append(dict(path=path,kind='schema'));return
            for i,(x,y) in enumerate(zip(a,b)):compare(x,y,path+'[%d]'%i)
        elif type(b) is float:
            if type(a) is not float or not math.isfinite(a) or not math.isfinite(b):
                invalid.append(dict(path=path,kind='type_or_nonfinite',saved=repr(a),recomputed=repr(b)));return
            if a!=b:
                delta=abs(a-b);limit=max(1e-12,2e-12*max(abs(a),abs(b)))
                item=dict(path=path,saved=a,recomputed=b,absolute_difference=delta,allowed_difference=limit,
                          within_roundoff_tolerance=delta<=limit)
                differences.append(item)
                if delta>limit:invalid.append(item)
        elif type(a) is not type(b) or a!=b:
            invalid.append(dict(path=path,kind='exact_value',saved=repr(a),recomputed=repr(b)))
    compare(saved,recomputed,'summary')
    return dict(valid=not invalid,exact_equal=not differences and not invalid,
        float_abs_tolerance=1e-12,float_rel_tolerance=2e-12,
        scope='Derived summary float values only; all episode decisions/counts and physical thresholds remain exact',
        differences=differences,invalid=invalid,python=sys.version,executable=sys.executable)

def wilson(successes,count):
    need(0<=successes<=count and count>0,'Invalid episode count')
    p=successes/count;z=1.959963984540054;den=1+z*z/count
    center=(p+z*z/(2*count))/den
    radius=z*math.sqrt(p*(1-p)/count+z*z/(4*count*count))/den
    return [max(0.,center-radius),min(1.,center+radius)]

def summarize(rows):
    need(bool(rows),'No evaluation episodes')
    count=len(rows);success=sum(r['success'] for r in rows)
    steps=sum(r['steps'] for r in rows)
    lengths=sorted(r['longest_stable_steps']*.02 for r in rows)
    return dict(episodes=count,successes=success,success_fraction=success/count,
        success_wilson95=wilson(success,count),ever_5s_count=sum(r['ever_5s'] for r in rows),
        physically_terminated_count=sum(r['terminated'] for r in rows),
        mean_survival_seconds=statistics.mean(r['steps']*.02 for r in rows),
        longest_stable_seconds=dict(median=statistics.median(lengths),min=min(lengths),max=max(lengths)),
        stable_fraction_of_first_episode_steps=sum(r['stable_steps'] for r in rows)/steps,
        gate_violation_fraction={g:sum(r['gate_failure_counts'][i] for r in rows)/steps for i,g in enumerate(GATES)},
        run_break_counts={g:sum(r['run_break_counts'][i] for r in rows) for i,g in enumerate(GATES)},
        values_mean_over_first_episode_steps={v:sum(r['value_sums'][i] for r in rows)/steps for i,v in enumerate(VALUES)},
        termination_counts={t:sum(t in r['terminations'] for r in rows) for t in sorted({t for r in rows for t in r['terminations']})})

def audit_completed_evidence(folder):
    import json
    from pathlib import Path
    folder=Path(folder)
    result=json.loads((folder/'evaluation/evaluation-result.json').read_bytes())
    rows=[];groups=[];audits=[]
    for pin in result['group_evidence']:
        seed=pin['seed'];path=folder/('evaluation/group-%d.json'%seed)
        need(hashlib.sha256(path.read_bytes()).hexdigest()==pin['sha256'],'Completed episode file hash differs')
        group=json.loads(path.read_bytes())
        need(group['status']=='EVALUATED' and group['seed']==seed and group['model']=='5000' and
             group['checkpoint_sha256']==result['checkpoint']['sha256'] and group['policy_and_normalizer_unchanged'] is True,
             'Completed episode group metadata differs')
        episodes=group['episodes']
        need(len(episodes)==128 and sorted(r['env_id'] for r in episodes)==list(range(128)),
             'Completed group is not 128 unique first episodes')
        for row in episodes:
            need(row['seed']==seed and type(row['env_id']) is int,'Episode identity differs')
            for name in ('steps','stable_steps','longest_stable_steps'):
                need(type(row[name]) is int and 0<=row[name]<=500,'Invalid episode step count')
            need(0<row['steps'] and row['longest_stable_steps']<=row['stable_steps']<=row['steps'],
                 'Invalid consecutive step accounting')
            need(all(type(row[k]) is bool for k in ('success','ever_5s','terminated')),'Non-boolean episode decision')
            need(row['ever_5s']==(row['longest_stable_steps']>=250) and
                 row['success']==(row['ever_5s'] and not row['terminated']), 'Original success definition differs')
            for name in ('gate_failure_counts','run_break_counts'):
                need(len(row[name])==len(GATES) and all(type(v) is int and 0<=v<=row['steps'] for v in row[name]),
                     'Invalid episode gate counters')
            need(len(row['value_sums'])==len(VALUES) and all(type(v) in (int,float) and math.isfinite(v) for v in row['value_sums']),
                 'Invalid episode value sums')
            need(type(row['terminations']) is list and all(type(v) is str for v in row['terminations']),
                 'Invalid termination names')
        recomputed=summarize(episodes)
        for origin,saved in [('group',group['summary']),('evaluation_manifest',pin['summary'])]:
            audit=summary_audit(saved,recomputed);need(audit['valid'],'Retained group aggregate differs beyond float roundoff')
            audits.append(dict(seed=seed,source=origin,**audit))
        rows.extend(episodes)
        groups.append(dict(seed=seed,episodes=128,successes=recomputed['successes'],
                           longest_stable_seconds=recomputed['longest_stable_seconds'],
                           physically_terminated_count=recomputed['physically_terminated_count']))
    summary=summarize(rows);audit=summary_audit(result['summary'],summary)
    need(audit['valid'],'Retained aggregate differs beyond float roundoff');audits.append(dict(source='combined',**audit))
    need(summary['episodes']==256 and summary['successes']==253 and summary['physically_terminated_count']==0
         and summary['termination_counts']=={'time_out':256},'Selected first-episode counts differ')
    return dict(status='RETAINED_EVIDENCE_VERIFIED',summary=summary,groups=groups,summary_audits=audits,
                physical_thresholds_changed=False,selection_set_not_final_heldout_test=True,
                paired_comparison_allowed=False,added_PPO_updates=0,added_simulation_steps=0)

if __name__=='__main__':
    import json
    print(json.dumps(audit_completed_evidence(sys.argv[1]),ensure_ascii=False,allow_nan=False))

