"""Detached evaluation lifecycle. Existing training artifacts remain immutable."""
import argparse
from contextlib import ExitStack
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
from eval_common import *

HERE=Path(__file__).resolve().parent

def snapshot():
    if not (SLOT/'started.json').exists():
        return dict(status='NOT_STARTED',added_PPO_updates=0)
    progress=read(SLOT/'progress.json') if (SLOT/'progress.json').exists() else dict(status='STARTING')
    running={name:alive(read(SLOT/(name+'.json'))) if (SLOT/(name+'.json')).exists() else None
             for name in ('supervisor','child')}
    result=dict(status=progress['status'],progress=progress,processes_alive=running,added_PPO_updates=0,
                started=read(SLOT/'started.json'),slot=str(SLOT))
    if (SLOT/'exit.json').exists():
        result['exit']=read(SLOT/'exit.json')
        result['status']=result['exit']['status']
        if result['status']=='EVALUATION_COMPLETE':
            result['evaluation']=read(SLOT/'evaluation-result.json')
    elif not any(v is True for v in running.values()) and all(v is not None for v in running.values()):
        result['status']='EVALUATION_INTERRUPTED_PRESERVE'
    if result['status'] not in ('STARTING','INITIALIZING','EVALUATING','EVALUATION_COMPLETE'):
        log=SLOT/'worker.log'
        if log.exists():
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-16000));result['log_tail']=f.read().decode(errors='replace')
    return result

def protected_files():
    paths=[TRAIN/name for name in ('started.json','budget.json','config.json','run/latest.json',
        'run/best-training-metric.json','run/progress.json','run/metrics.jsonl')]
    paths += [model_path(key) for key in MODELS]
    return {str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in paths}

def preflight():
    need(PYTHON.is_file(),'Existing cloud Python is missing')
    b=read(TRAIN/'budget.json');p=read(TRAIN/'run/progress.json')
    need(b['completed_updates']==b['attempted_updates']==b['max_updates']==6000 and not b['pending'],
         'Original training budget is not fully completed')
    need(p['status']=='TRAINING_BUDGET_COMPLETE','Original training is not complete')
    need(read(TRAIN/'config.json')['package_identity']==ORIGIN_ID,'Original training identity changed')
    for slot in (LEGACY,TRAIN):
        for name in ('supervisor','child'):
            pin=slot/(name+'.json')
            need(not pin.exists() or not alive(read(pin)),'A prior training process is still active')
    origin=TRAIN/'package';manifest=read(origin/'manifest.json')
    need(manifest['identity']==ORIGIN_ID,'Original package manifest differs')
    pins=read(HERE/'origin-pins.json')
    for rel,digest in pins.items():need(sha(origin/rel)==digest,'Original package source differs: '+rel)
    for rel,pin in manifest['files'].items():need(sha(origin/rel)==pin['sha256'],'Original package file differs: '+rel)
    for rel,pin in read(origin/'source-pins.json').items():
        path=SOURCE/rel
        need(path.is_file() and path.stat().st_size==pin['bytes'] and sha(path)==pin['sha256'],'Frozen project differs: '+rel)
    for key,digest in MODELS.items():
        path=model_path(key)
        need(not path.is_symlink() and path.resolve().parent==(TRAIN/'run/checkpoints').resolve(), 'Unexpected model path')
        need(path.stat().st_size==10036060 and sha(path)==digest,'Selected model bytes differ: '+key)
    return protected_files()

def launch():
    with ExitStack() as stack:
        locks=[]
        # The original launchers use these exact lock inodes. Keep them across both processes.
        for folder in (LEGACY,TRAIN,SLOT):
            need(folder.is_dir() and not folder.is_symlink(),'Expected campaign directory missing or symlinked')
            lock=stack.enter_context((folder/'launch.lock').open('a'));locks.append(lock)
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                if (SLOT/'started.json').exists():return snapshot()
                raise RuntimeError('A training/evaluation operation holds the shared lock; nothing launched')
        if (SLOT/'started.json').exists():return snapshot()
        baseline=preflight();now=time.time()
        write(SLOT/'original-files.json',baseline,exclusive=True)
        write(SLOT/'started.json',dict(started_utc=now,deadline_utc=now+MAX_SECONDS,
            max_seconds=MAX_SECONDS,checkpoints=MODELS,seeds=SEEDS,num_envs=NUM_ENVS,
            total_first_episodes=512,maximum_new_PPO_updates=0,
            package_identity=read(HERE/'manifest.json')['identity']),exclusive=True)
        write(SLOT/'progress.json',dict(status='STARTING',phase='launch',completed_first_episodes=0,progress_percent=0.))
        fds=tuple(f.fileno() for f in locks)
        with (SLOT/'supervisor.log').open('xb') as log:
            child=subprocess.Popen([sys.executable,'-B','-u',str(HERE/'agent.py'),'supervise','--lock-fds',','.join(map(str,fds))],
                cwd=SLOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=fds,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        pin=identity(child.pid);need(pin is not None,'Evaluation supervisor launch uncertain; preserve started marker')
        write(SLOT/'supervisor.json',pin)
        return snapshot()

def supervise(fds):
    for fd in fds:os.fstat(fd)
    started=read(SLOT/'started.json');deadline=started['deadline_utc'];code=None;error=None
    try:
        cache=SLOT/'cache';cache.mkdir(exist_ok=True)
        temp=Path('/tmp')/('md12eval-'+started['package_identity'][:10]);temp.mkdir(mode=0o700,exist_ok=True)
        need(not temp.is_symlink() and temp.stat().st_uid==os.getuid(),'Unexpected evaluation temp directory')
        env={k:v for k,v in os.environ.items() if not k.startswith('MICRODUCK_') and k not in ('WORLD_SIZE','RANK','LOCAL_RANK')}
        env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',
            OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',WANDB_MODE='disabled',
            TMPDIR=str(temp),TEMP=str(temp),TMP=str(temp),MPLCONFIGDIR=str(cache/'mpl'))
        with (SLOT/'worker.log').open('xb') as log:
            worker=subprocess.Popen([str(PYTHON),'-B','-u',str(HERE/'evaluate.py')],cwd=SLOT,stdin=subprocess.DEVNULL,
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=tuple(fds),env=env)
            pin=identity(worker.pid);need(pin is not None,'Evaluation worker launch uncertain')
            write(SLOT/'child.json',pin);term_at=None
            while worker.poll() is None:
                if time.time()>=deadline and term_at is None and alive(pin):
                    os.killpg(pin['group'],signal.SIGTERM);term_at=time.monotonic()
                if term_at is not None and time.monotonic()-term_at>=90 and alive(pin):
                    os.killpg(pin['group'],signal.SIGKILL)
                try:worker.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
            code=worker.returncode
    except BaseException as exc:error=repr(exc)
    try:unchanged=protected_files()==read(SLOT/'original-files.json')
    except BaseException as exc:unchanged=False;error=(error or '')+' '+repr(exc)
    p=read(SLOT/'progress.json') if (SLOT/'progress.json').exists() else {}
    ok=code==0 and p.get('status')=='EVALUATION_COMPLETE' and unchanged
    write(SLOT/'exit.json',dict(status='EVALUATION_COMPLETE' if ok else 'EVALUATION_ERROR_PRESERVE',
        returncode=code,original_training_files_unchanged=unchanged,error=error,
        finished_utc=time.time(),added_PPO_updates=0,automatic_retry=False),exclusive=True)
    return snapshot()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('operation',choices=['start','status','collect','supervise'])
    ap.add_argument('--lock-fds',default='');a=ap.parse_args()
    if a.operation=='start':value=launch()
    elif a.operation=='supervise':value=supervise([int(x) for x in a.lock_fds.split(',') if x])
    else:value=snapshot()
    print('MD12_EVAL_STATE='+json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print('MD12_EVAL_ERROR='+json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()[-8000:])),flush=True)
        raise SystemExit(2)
