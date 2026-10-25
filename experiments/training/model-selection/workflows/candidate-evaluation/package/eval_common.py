"""Fixed, bounded comparison of two existing Stage12 checkpoints."""
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

BASE = Path('/root/autodl-tmp/microduck-double-balance')
TRAIN = BASE/'artifacts/double-balance-stage12/native-training-v2'
LEGACY = BASE/'artifacts/double-balance-stage12/native-training-v1'
SLOT = BASE/'artifacts/double-balance-stage12/native-evaluation-v1'
SOURCE = BASE/'artifacts/double-balance-stage12/finite-trial-v1/cloud-20261020-fp64-one-shot-v1/inputs/project'
PYTHON = BASE/'workspace/.venv/bin/python'
ORIGIN_ID = 'dd6d1e1e697bcaa348dbdd92dd8c7b30ea31ae9c32599338106859ae7657523a'
MODELS = {
    '4500': 'df03202acf010680b63d3277ec879f63e2d4808e5d1286653923a314f125ac6a',
    '6000': 'f89a16643e949bc0e79ae6be77c5e12635cd7538b5bfe0bdef9a1cb37c802ee3',
}
SEEDS = (20261021, 20261022)
NUM_ENVS = 128
HORIZON_STEPS = 500
MAX_SECONDS = 1800
GATES = ('top_center', 'top_height', 'top_speed', 'lower_offset', 'lower_height_min',
         'lower_height_max', 'robot_tilt', 'lower_ball_speed', 'unassisted')
VALUES = ('top_center_m', 'top_height_error_m', 'top_speed_m_s', 'lower_offset_m',
          'lower_height_m', 'robot_tilt_rad', 'lower_ball_speed_m_s', 'hold')
THRESHOLDS = dict(top_center_m=.012, top_height_error_m=.006, top_speed_m_s=.08,
                  lower_offset_m=.06, lower_height_min_m=.20, lower_height_max_m=.29,
                  robot_tilt_deg=20., lower_ball_speed_m_s=.15, hold=1e-6,
                  consecutive_steps=250, control_dt=.02, horizon_seconds=10.)

def need(ok, message):
    if not ok:
        raise RuntimeError(message)

def read(path):
    return json.loads(Path(path).read_bytes())

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024**2),b''):
            h.update(chunk)
    return h.hexdigest()

def write(path, value, exclusive=False):
    path=Path(path)
    data=(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
    target=path if exclusive else path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with target.open('xb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    if not exclusive:
        os.replace(target,path)

def identity(pid):
    try:
        row=(Path('/proc')/str(int(pid))/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),start=row[19],state=row[0],group=int(row[2]))
    except FileNotFoundError:
        return None

def alive(pin):
    if not pin:
        return False
    now=identity(pin['pid'])
    return bool(now and now['start']==pin['start'] and now['state'] not in ('Z','X'))

def model_path(key):
    return TRAIN/'run/checkpoints'/('update_%06d.pt'%int(key))

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

def paired_comparison(a,b):
    left={(r['seed'],r['env_id']):r for r in a};right={(r['seed'],r['env_id']):r for r in b}
    need(left.keys()==right.keys(),'Paired episode identities differ')
    pairs=[(left[k],right[k]) for k in sorted(left)]
    return dict(episodes=len(pairs),both_success=sum(x['success'] and y['success'] for x,y in pairs),
        only_4500_success=sum(x['success'] and not y['success'] for x,y in pairs),
        only_6000_success=sum(y['success'] and not x['success'] for x,y in pairs),
        neither_success=sum(not x['success'] and not y['success'] for x,y in pairs))
