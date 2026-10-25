"""New finite refinement campaign; all completed native-v2 evidence is retained."""
import hashlib
import json
import os
from pathlib import Path

BASE=Path('/root/autodl-tmp/microduck-double-balance')
TRAIN=BASE/'artifacts/double-balance-stage12/native-training-v2'
LEGACY=TRAIN.with_name('native-training-v1')
PRIOR=TRAIN.with_name('native-evaluation-v1')
RECOVERY1=TRAIN.with_name('native-evaluation-recovery-v1')
RECOVERY2=TRAIN.with_name('native-evaluation-recovery-v2')
SLOT=TRAIN.with_name('native-refine-4500-v1')
SOURCE=BASE/'artifacts/double-balance-stage12/finite-trial-v1/cloud-20261020-fp64-one-shot-v1/inputs/project'
PYTHON=BASE/'workspace/.venv/bin/python'
HERE=Path(__file__).resolve().parent
ORIGIN_ID='dd6d1e1e697bcaa348dbdd92dd8c7b30ea31ae9c32599338106859ae7657523a'
MODEL=TRAIN/'run/checkpoints/update_004500.pt'
MODEL_SHA='df03202acf010680b63d3277ec879f63e2d4808e5d1286653923a314f125ac6a'
MODEL6000_SHA='f89a16643e949bc0e79ae6be77c5e12635cd7538b5bfe0bdef9a1cb37c802ee3'
GROUP_SHAS={20261021:'7be7e52a27f9192626af8b9f2f7cf94ce06c99e3e86323e2f4055f4e1fdd48bf',
            20261022:'5df8d19a0c50869bdd65a06ac18293d458da6f4e46304bebee732e240b9e2edc'}
SEEDS=(20261021,20261022)
NUM_ENVS=4096
EVAL_ENVS=128
MAX_UPDATES=1000
SEGMENT_UPDATES=250
MAX_SEGMENTS=4
MAX_SECONDS=7200
LEARNING_RATE=1e-5
PATIENCE=2
TRAIN_SEED=20261023

def require(ok,message):
    if not ok:raise RuntimeError(message)

def read(path):return json.loads(Path(path).read_bytes())

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def write(path,value,exclusive=False):
    path=Path(path);data=(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False)+'\n').encode()
    target=path if exclusive else path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with target.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    if not exclusive:os.replace(target,path)

def identity(pid):
    try:
        row=(Path('/proc')/str(int(pid))/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=int(pid),start=row[19],state=row[0],group=int(row[2]))
    except FileNotFoundError:return None

def alive(pin):
    now=identity(pin['pid']) if pin else None
    return bool(now and now['start']==pin['start'] and now['state'] not in ('Z','X'))

def reserve(budget,now):
    require(now<budget['deadline_utc'],'Original two-hour deadline reached')
    require(not budget['pending'],'Earlier PPO attempt remains pending')
    require(budget['attempted_updates']<MAX_UPDATES,'New PPO attempt budget exhausted')
    budget['attempted_updates']+=1;budget['pending']=True

def commit(budget):
    require(budget['pending'],'PPO commit without reservation')
    budget['completed_updates']+=1;budget['pending']=False

def input_checkpoint():
    path=SLOT/'run/latest.json'
    if path.exists():return read(path)
    return dict(path=str(MODEL),sha256=MODEL_SHA,bytes=10036060,completed_updates=0,policy_updates=4500)

def baseline():
    from eval_metrics import summarize,summary_audit
    rows=[];audits={}
    for seed,digest in GROUP_SHAS.items():
        path=PRIOR/('group-4500-%d.json'%seed)
        require(sha(path)==digest,'Original 4500 evaluation file differs')
        g=read(path)
        require(g['model']=='4500' and g['seed']==seed and g['checkpoint_sha256']==MODEL_SHA and g['policy_and_normalizer_unchanged'],'Retained model identity differs')
        require(len(g['episodes'])==EVAL_ENVS and sorted(r['env_id'] for r in g['episodes'])==list(range(EVAL_ENVS)),'Retained episode identity differs')
        require(all(r['seed']==seed and r['success']==(r['ever_5s'] and not r['terminated']) and r['ever_5s']==(r['longest_stable_steps']>=250) for r in g['episodes']),'Retained success decisions differ')
        audit=summary_audit(g['summary'],summarize(g['episodes']))
        require(audit['valid'],'Retained aggregate has more than floating summary roundoff')
        audits[str(seed)]=audit;rows.extend(g['episodes'])
    summary=summarize(rows)
    require(summary['episodes']==256 and summary['successes']==212,'Original baseline is not 212/256')
    return dict(model='4500',checkpoint=dict(path=str(MODEL),sha256=MODEL_SHA,bytes=10036060,completed_updates=0,policy_updates=4500),
        summary=summary,reused_episodes=256,summary_audits=audits)

def choose(selection,candidate):
    """Only a higher success count replaces best. Ties preserve the earlier model."""
    require(candidate['segment']==len(selection['history'])+1,'Candidate order or duplicate selection differs')
    require(candidate['summary']['episodes']==256,'Candidate evaluation incomplete')
    better=candidate['summary']['successes']>selection['best']['summary']['successes']
    if better:selection['best']=candidate;selection['no_improvement']=0
    else:selection['no_improvement']+=1
    selection['history'].append(candidate)
    selection['early_stop']=selection['no_improvement']>=PATIENCE
    return better

def old_slots():return (LEGACY,TRAIN,PRIOR,RECOVERY1,RECOVERY2)

def protected_files():
    paths=[TRAIN/name for name in ('started.json','budget.json','config.json','run/latest.json','run/best-training-metric.json','run/progress.json','run/metrics.jsonl')]
    paths += [MODEL,TRAIN/'run/checkpoints/update_006000.pt']
    for folder in (PRIOR,RECOVERY1,RECOVERY2):
        paths += [folder/name for name in ('started.json','progress.json','exit.json','evaluation-result.json','original-files.json') if (folder/name).is_file()]
        paths += sorted(folder.glob('group-*.json'))
        paths += sorted(folder.glob('initial-*.json'))
    paths += [p for folder in old_slots() for p in sorted((folder/'package').glob('*')) if p.is_file()]
    return {str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths}

def preflight():
    require(PYTHON.is_file(),'Existing cloud project Python missing; no installer is run')
    require(sha(MODEL)==MODEL_SHA and MODEL.stat().st_size==10036060,'Selected 4500 checkpoint differs')
    require(sha(TRAIN/'run/checkpoints/update_006000.pt')==MODEL6000_SHA,'Retained 6000 checkpoint differs')
    b=read(TRAIN/'budget.json')
    require(b['completed_updates']==b['attempted_updates']==b['max_updates']==6000 and not b['pending'],'Old training campaign is incomplete')
    require(read(TRAIN/'config.json')['package_identity']==ORIGIN_ID,'Old training package identity differs')
    require(read(RECOVERY2/'exit.json')['status']=='EVALUATION_COMPLETE','Original comparison not completed')
    for folder in old_slots():
        for name in ('supervisor','child'):
            path=folder/(name+'.json');require(not path.exists() or not alive(read(path)),'An old GPU task is active')
    require(sha(HERE/'native_task.py')==sha(TRAIN/'package/native_task.py'),'Native task is not the original trained task')
    for name,digest in read(HERE/'origin-pins.json').items():require(sha(TRAIN/'package'/name)==digest,'Original package source changed: '+name)
    for rel,pin in read(HERE/'source-pins.json').items():
        path=SOURCE/rel
        require(path.is_file() and path.stat().st_size==pin['bytes'] and sha(path)==pin['sha256'],'Frozen project differs: '+rel)
    return baseline(),protected_files()
