"""Summarize existing timing/reward logs only; never import the simulator."""
import json
import math
from pathlib import Path
import re
import statistics

def stats(values):
    values=[float(x) for x in values if isinstance(x,(int,float)) and math.isfinite(x)]
    return dict(samples=len(values),mean=statistics.mean(values),median=statistics.median(values),minimum=min(values),maximum=max(values)) if values else None

def tail(path,limit=4*1024*1024):
    with Path(path).open('rb') as f:
        end=f.seek(0,2);start=max(0,end-limit);f.seek(start);data=f.read(end-start)
    if start:data=data.split(b'\n',1)[-1]
    return data.decode(errors='replace')

def parse_reward_log(text):
    text=re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]','',text)
    parts=re.split(r'Learning iteration\s+(\d+)\s*/\s*(\d+)',text)
    rows=[]
    for i in range(1,len(parts),3):
        body=parts[i+2]
        if 'Iteration time:' not in body:continue
        values={key:float(value) for key,value in re.findall(r'^\s*(Episode_Reward/[A-Za-z0-9_]+):\s*([-+0-9.eE]+)\s*$',body,re.M)}
        if values:rows.append(dict(iteration=int(parts[i]),values=values))
    rows=rows[-50:]
    return dict(samples=len(rows),first_iteration=rows[0]['iteration'] if rows else None,last_iteration=rows[-1]['iteration'] if rows else None,
        contributions={key:stats([row['values'][key] for row in rows if key in row['values']]) for key in sorted({key for row in rows for key in row['values']})},
        scope='Existing upstream printed Episode_Reward values, already weighted; four-decimal log precision; no simulator or reward recomputation')

def inspect_run(slot):
    slot=Path(slot);path=slot/'run/metrics.jsonl'
    result=dict(read_only=True,slot=str(slot),added_PPO_updates=0,added_simulation_steps=0,timing_scope='Existing runner collection/learning wall timers; excludes initialization and evaluation; not CUDA kernel timings')
    if not path.exists():return dict(result,available=False)
    rows=[];bad=0
    for line in tail(path,16*1024*1024).splitlines():
        try:
            value=json.loads(line)
            if isinstance(value,dict) and value.get('phase')=='refine_ppo' and isinstance(value.get('completed_updates'),int):rows.append(value)
        except (ValueError,TypeError):bad+=1
    result.update(available=bool(rows),parsed_rows=len(rows),invalid_lines=bad,segments=[])
    for segment in sorted({row['segment'] for row in rows}):
        selected=[row for row in rows if row['segment']==segment][-50:]
        wall=[]
        for left,right in zip(selected,selected[1:]):
            delta=right['completed_updates']-left['completed_updates'];elapsed=right['elapsed_seconds']-left['elapsed_seconds']
            if delta==1 and elapsed>0 and left['session']==right['session']:wall.append(elapsed)
        collection=[row['collection_seconds'] for row in selected];learning=[row['learning_seconds'] for row in selected]
        frames=[4096*24/(x+y) for x,y in zip(collection,learning) if x+y>0]
        rewards={key:stats([row['episode_metrics'][key] for row in selected if key in row.get('episode_metrics',{})])
            for key in sorted({k for row in selected for k in row.get('episode_metrics',{}) if k.startswith('Episode_Reward/')})}
        entry=dict(segment=segment,first_update=selected[0]['completed_updates'],last_update=selected[-1]['completed_updates'],
            collection_seconds=stats(collection),learning_seconds=stats(learning),environment_transitions_per_logged_compute_second=stats(frames),
            wall_seconds_between_updates=stats(wall),weighted_reward_metrics=rewards)
        logs=sorted((slot/'logs').glob('*-train-segment-%02d.log'%segment))
        if logs:entry['printed_reward_window']=parse_reward_log(tail(logs[-1]))
        result['segments'].append(entry)
    return result

def inspect_safe(slot):
    try:return inspect_run(slot)
    except Exception as exc:return dict(read_only=True,slot=str(slot),available=False,error=repr(exc),added_PPO_updates=0,added_simulation_steps=0)
