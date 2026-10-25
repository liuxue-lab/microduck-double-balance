"""Finite detached campaign. Neither disconnects nor resume reset the budget."""
import argparse
from contextlib import ExitStack
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
from common import *

def optional(name,default=None):
    path=SLOT/name
    return read(path) if path.exists() else default

def snapshot():
    started=optional('started.json')
    if started is None:return dict(status='NOT_STARTED',slot=str(SLOT),added_PPO_updates=0)
    session=optional('session.json',{})
    running={name:alive(optional(name+'.json')) for name in ('supervisor','child')}
    progress=optional('progress.json',dict(status='STARTING'))
    budget=optional('budget.json',{})
    value=dict(status=progress['status'],started=started,budget=budget,progress=progress,session=session,
        selection=optional('selection.json'),processes_alive=running,slot=str(SLOT),
        added_PPO_updates=budget.get('completed_updates',0),package_identity=read(HERE/'manifest.json')['identity'])
    ended=optional('exit.json')
    if ended and ended['session']==session.get('number'):
        value['exit']=ended;value['status']=ended['status']
    elif not any(running.values()) and time.time()-session.get('requested_utc',0)>30:
        value['status']='REFINEMENT_INTERRUPTED_PRESERVE'
    if value['status'].endswith('PRESERVE'):
        path=Path(optional('active-worker.json',{}).get('log',str(SLOT/'supervisor.log')))
        if path.is_file() and path.resolve().is_relative_to(SLOT.resolve()):
            with path.open('rb') as f:
                f.seek(max(0,path.stat().st_size-14000));value['log_tail']=f.read().decode(errors='replace')
    return value

def check_resume():
    budget=read(SLOT/'budget.json')
    require(time.time()<budget['deadline_utc'],'Original two-hour deadline expired; collect only, no extension')
    require(input_checkpoint()['completed_updates']==budget['completed_updates'],
        'Durable checkpoint lags committed updates; preserve without replaying spent updates')
    from evaluate import retained_group
    for folder in sorted((SLOT/'evaluations').glob('segment-*')):
        if (folder/'candidate.json').exists():
            checkpoint=read(folder/'candidate.json')
            for seed in SEEDS:retained_group(folder,seed,checkpoint)
    return budget

def launch(resume=False):
    with ExitStack() as stack:
        locks=[]
        for folder in (*old_slots(),SLOT):
            if folder in (LEGACY,RECOVERY1) and not folder.exists():continue
            require(folder.is_dir() and not folder.is_symlink(),'Campaign directory is missing or symlinked')
            lock=stack.enter_context((folder/'launch.lock').open('a'));locks.append(lock)
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                if (SLOT/'started.json').exists():return snapshot()
                raise RuntimeError('A previous GPU operation still holds the shared lock')
        exists=(SLOT/'started.json').exists()
        if exists and not resume:return snapshot()
        require(exists or not resume,'No refinement campaign exists; start with --train')
        ended=optional('exit.json',{})
        if exists and ended.get('status')=='REFINEMENT_COMPLETE':return snapshot()
        reference,original_files=preflight();now=time.time()
        if not exists:
            for name in ('run/checkpoints','evaluations','logs','cache'):(SLOT/name).mkdir(parents=True,exist_ok=True)
            started=dict(started_utc=now,deadline_utc=now+MAX_SECONDS,max_seconds=MAX_SECONDS,
                max_updates=MAX_UPDATES,num_envs=NUM_ENVS,segment_updates=SEGMENT_UPDATES,max_segments=MAX_SEGMENTS,
                learning_rate=LEARNING_RATE,rollout_steps=24,patience=PATIENCE,train_seed=TRAIN_SEED,evaluation_seeds=list(SEEDS),
                maximum_new_first_episodes=MAX_SEGMENTS*256,reused_baseline_first_episodes=256,
                package_identity=read(HERE/'manifest.json')['identity'],origin_checkpoint_sha256=MODEL_SHA,
                python=sys.version,executable=sys.executable,automatic_retry=False,physical_trajectory_resume=False,
                selection_rule='Higher success count only; ties keep the earlier model; two consecutive non-improvements stop training')
            budget=dict(started_utc=now,deadline_utc=started['deadline_utc'],max_updates=MAX_UPDATES,
                attempted_updates=0,completed_updates=0,pending=False,interrupted_attempts=0)
            write(SLOT/'original-files.json',original_files,exclusive=True)
            write(SLOT/'baseline.json',reference,exclusive=True)
            write(SLOT/'budget.json',budget,exclusive=True)
            write(SLOT/'selection.json',dict(best=reference,history=[],no_improvement=0,early_stop=False),exclusive=True)
            write(SLOT/'started.json',started,exclusive=True)
        else:
            require(read(SLOT/'started.json')['package_identity']==read(HERE/'manifest.json')['identity'],'Campaign package identity differs')
            require(original_files==read(SLOT/'original-files.json'),'Original training or evaluation evidence changed')
            budget=check_resume()
            if budget['pending']:
                budget['pending']=False;budget['interrupted_attempts']+=1
                write(SLOT/'budget.json',budget)
        number=optional('session.json',dict(number=0))['number']+1
        if (SLOT/'STOP.json').exists():os.rename(SLOT/'STOP.json',SLOT/('STOP-before-session-%03d.json'%number))
        session=dict(number=number,requested_utc=now,resume=resume,original_deadline_utc=budget['deadline_utc'])
        write(SLOT/('session-%03d.json'%number),session,exclusive=True);write(SLOT/'session.json',session)
        write(SLOT/'progress.json',dict(status='STARTING',phase='launch',completed_updates=budget['completed_updates'],attempted_updates=budget['attempted_updates']))
        fds=tuple(f.fileno() for f in locks)
        with (SLOT/'logs'/('supervisor-%03d.log'%number)).open('xb') as log:
            process=subprocess.Popen([str(PYTHON),'-B','-u',str(HERE/'agent.py'),'supervise','--session',str(number),'--lock-fds',','.join(map(str,fds))],
                cwd=SLOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,pass_fds=fds,
                env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        pin=identity(process.pid);require(pin is not None,'Supervisor launch uncertain; preserve started marker')
        write(SLOT/'supervisor.json',pin)
        return snapshot()

def worker_environment():
    temp=Path('/tmp')/('md12refine-'+read(HERE/'manifest.json')['identity'][:10]);temp.mkdir(mode=0o700,exist_ok=True)
    require(not temp.is_symlink() and temp.stat().st_uid==os.getuid(),'Unexpected campaign temp directory')
    env={k:v for k,v in os.environ.items() if not k.startswith('MICRODUCK_') and k not in ('WORLD_SIZE','RANK','LOCAL_RANK')}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='1',HF_HUB_OFFLINE='1',WANDB_MODE='disabled',TMPDIR=str(temp),TEMP=str(temp),TMP=str(temp),
        MPLCONFIGDIR=str(SLOT/'cache/mpl'))
    return env

def run_worker(kind,segment,fds,deadline,stopped):
    number=optional('worker-counter.json',dict(number=0))['number']+1
    write(SLOT/'worker-counter.json',dict(number=number))
    logfile=SLOT/'logs'/('%03d-%s-segment-%02d.log'%(number,kind,segment))
    argv=[str(PYTHON),'-B','-u',str(HERE/(kind+'.py')),'--segment',str(segment)]
    if kind=='train':argv += ['--session',str(number)]
    worker=None;pin=None;term_at=None
    try:
        with logfile.open('xb') as log:
            worker=subprocess.Popen(argv,cwd=SLOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=tuple(fds),env=worker_environment())
            pin=identity(worker.pid);require(pin is not None,'Worker launch uncertain; no retry')
            write(SLOT/'child.json',pin)
            write(SLOT/'active-worker.json',dict(number=number,kind=kind,segment=segment,log=str(logfile),pin=pin))
            while worker.poll() is None:
                stopping=stopped[0] or (SLOT/'STOP.json').exists() or time.time()>=deadline
                if stopping and term_at is None and alive(pin):
                    os.killpg(pin['group'],signal.SIGTERM);term_at=time.monotonic()
                if term_at is not None and time.monotonic()-term_at>=90 and alive(pin):os.killpg(pin['group'],signal.SIGKILL)
                try:worker.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
        return worker.returncode
    finally:
        # A supervisor exception must not leave an untracked worker running.
        if worker is not None and worker.poll() is None:
            if alive(pin):os.killpg(pin['group'],signal.SIGTERM)
            try:worker.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if alive(pin):os.killpg(pin['group'],signal.SIGKILL)
                worker.wait(timeout=15)

def supervise(fds,session):
    for fd in fds:os.fstat(fd)
    require(read(SLOT/'session.json')['number']==session,'Supervisor session differs')
    deadline=read(SLOT/'started.json')['deadline_utc'];stopped=[False];status='REFINEMENT_ERROR_PRESERVE';reason=None;error=None
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    try:
        while True:
            selection=read(SLOT/'selection.json');budget=read(SLOT/'budget.json')
            if selection['early_stop'] or len(selection['history'])>=MAX_SEGMENTS:
                status='REFINEMENT_COMPLETE';reason='TWO_CONSECUTIVE_NON_IMPROVEMENTS' if selection['early_stop'] else 'MAXIMUM_SEGMENTS';break
            if stopped[0] or (SLOT/'STOP.json').exists() or time.time()>=deadline:
                status='REFINEMENT_STOPPED_PRESERVE';reason='STOP_REQUEST_OR_ORIGINAL_DEADLINE';break
            segment=len(selection['history'])+1;target=min(segment*SEGMENT_UPDATES,MAX_UPDATES)
            folder=SLOT/'evaluations'/('segment-%02d'%segment);folder.mkdir(exist_ok=True)
            candidate_path=folder/'candidate.json'
            if not candidate_path.exists():
                if budget['completed_updates']<target and budget['attempted_updates']<MAX_UPDATES:
                    code=run_worker('train',segment,fds,deadline,stopped)
                    if code!=0:
                        status='REFINEMENT_STOPPED_PRESERVE' if code==3 or time.time()>=deadline or (SLOT/'STOP.json').exists() or stopped[0] else 'REFINEMENT_ERROR_PRESERVE'
                        reason='TRAIN_WORKER_RETURN_%s'%code;break
                budget=read(SLOT/'budget.json');checkpoint=input_checkpoint()
                require(checkpoint['completed_updates']==budget['completed_updates'] and not budget['pending'],'Segment checkpoint or budget is incomplete')
                previous=selection['history'][-1]['checkpoint']['completed_updates'] if selection['history'] else 0
                if checkpoint['completed_updates']<=previous:
                    status='REFINEMENT_COMPLETE';reason='ATTEMPT_BUDGET_EXHAUSTED';break
                require(sha(checkpoint['path'])==checkpoint['sha256'],'Segment checkpoint changed')
                write(candidate_path,checkpoint,exclusive=True)
            if stopped[0] or time.time()>=deadline or (SLOT/'STOP.json').exists():
                status='REFINEMENT_STOPPED_PRESERVE';reason='STOP_BEFORE_EVALUATION';break
            result_path=folder/'evaluation-result.json'
            if not result_path.exists():
                code=run_worker('evaluate',segment,fds,deadline,stopped)
                if code!=0:
                    status='REFINEMENT_STOPPED_PRESERVE' if time.time()>=deadline or (SLOT/'STOP.json').exists() or stopped[0] else 'REFINEMENT_ERROR_PRESERVE'
                    reason='EVALUATION_WORKER_RETURN_%s'%code;break
            result=read(result_path)
            require(result['status']=='EVALUATION_COMPLETE' and result['checkpoint']==read(candidate_path),'Candidate evaluation result identity differs')
            candidate={k:result[k] for k in ('segment','model','checkpoint','summary')}
            candidate.update(evaluation_path=str(result_path),evaluation_sha256=sha(result_path))
            choose(selection,candidate);write(SLOT/'selection.json',selection)
            write(SLOT/'progress.json',dict(status='SELECTING',phase='candidate_compared',segment=segment,
                candidate_successes=candidate['summary']['successes'],best_successes=selection['best']['summary']['successes'],
                no_improvement=selection['no_improvement']))
            if budget['attempted_updates']>=MAX_UPDATES:
                status='REFINEMENT_COMPLETE';reason='ATTEMPT_BUDGET_EXHAUSTED';break
    except BaseException as exc:error=dict(error=repr(exc),traceback=traceback.format_exc()[-10000:])
    try:
        unchanged=protected_files()==read(SLOT/'original-files.json')
        require(unchanged,'Original training or evaluation evidence changed')
    except BaseException as exc:
        unchanged=False;status='REFINEMENT_ERROR_PRESERVE';error=dict(error=repr(exc),prior_error=error)
    result=dict(status=status,session=session,reason=reason,error=error,finished_utc=time.time(),
        original_training_and_evaluation_files_unchanged=unchanged,original_deadline_utc=deadline,
        added_PPO_updates=read(SLOT/'budget.json')['completed_updates'],automatic_retry=False)
    write(SLOT/('exit-session-%03d.json'%session),result,exclusive=True);write(SLOT/'exit.json',result)
    return snapshot()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('operation',choices=['start','resume','status','collect','stop','supervise'])
    ap.add_argument('--lock-fds',default='');ap.add_argument('--session',type=int);a=ap.parse_args()
    if a.operation in ('start','resume'):value=launch(a.operation=='resume')
    elif a.operation=='supervise':value=supervise([int(x) for x in a.lock_fds.split(',') if x],a.session)
    elif a.operation=='stop':
        value=snapshot()
        if (SLOT/'started.json').exists() and value['status']!='REFINEMENT_COMPLETE':
            write(SLOT/'STOP.json',dict(requested_utc=time.time(),reason='Explicit user --stop; no power action'))
            value=snapshot()
    else:value=snapshot()
    print('MD12_REFINE_STATE='+json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        value=dict(error=repr(exc),traceback=traceback.format_exc()[-10000:],started_marker_exists=(SLOT/'started.json').exists())
        try:value['state']=snapshot()
        except Exception as other:value['snapshot_error']=repr(other)
        print('MD12_REFINE_ERROR='+json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)
        raise SystemExit(2)
