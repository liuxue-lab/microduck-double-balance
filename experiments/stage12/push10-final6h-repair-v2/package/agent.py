"""One six-hour detached training and validation campaign; never renews a budget."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

from common import *
import budget as account


def optional(name, default=None):
    path=SLOT/name
    return read(path) if path.is_file() else default


def process_pin(child):
    until=time.monotonic()+2.
    while child.poll() is None:
        pin=identity(child.pid)
        if pin is not None:return pin
        if time.monotonic()>=until:break
        time.sleep(.01)
    return None


def signal_owned(child,pin,sig):
    if child.poll() is not None:return
    current=identity(child.pid)
    require(pin and current and current['start']==pin['start']
            and current['group']==pin['group']==child.pid,
            'Owned worker process identity differs; preserve')
    if current['state'] not in ('Z','X'):os.killpg(pin['group'],sig)


def snapshot():
    started=optional('started.json')
    if started is None:return dict(status='NOT_STARTED',slot=str(SLOT),added_PPO_updates=0)
    running={name:alive(optional(name+'.json')) for name in ('supervisor','child')}
    progress=optional('progress.json',{'status':'STARTING'})
    ledger=optional('budget.json',{})
    ended=optional('exit.json')
    value=dict(status=progress['status'],slot=str(SLOT),started=started,budget=ledger,
        progress=progress,processes_alive=running,selected=optional('selected.json'),
        selection=optional('selection.json'),added_PPO_updates=ledger.get('completed_updates',0),
        package_identity=verify_package(),power_state='UNCONFIRMED',power_action='NONE',
        compute_active=ended is None and any(running.values()),
        transfer=optional('transfer-outcome.json'),cloud_return=optional('return.json'))
    value['repair']=optional('repair-applied.json')
    value['repair_resume']=optional('repair-resume-started.json')
    if ended is not None:
        value['exit']=ended
        value['status']='COMPUTE_COMPLETE_RETURNING' if running['supervisor'] else ended['status']
    elif not any(running.values()) and time.time()-started['started_utc']>30:
        value['status']='CAMPAIGN_INTERRUPTED_PRESERVE'
    if value['status'].endswith('PRESERVE'):
        info=optional('active-worker.json',{})
        if info.get('kind')=='train' and isinstance(info.get('number'),int):
            result=SLOT/'run'/('session-%03d'%info['number'])/'result.json'
        elif info.get('kind')=='evaluate' and '--run-dir' in info.get('args',[]):
            result=Path(info['args'][info['args'].index('--run-dir')+1])/'evaluation-result.json'
        else:result=None
        if result and result.is_file() and result.resolve().is_relative_to(SLOT.resolve()):
            value['worker_result']=read(result)
        log=Path(info.get('log',str(SLOT/'logs/supervisor.log')))
        if log.is_file() and log.resolve().is_relative_to(SLOT.resolve()):
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-16000));value['log_tail']=f.read().decode(errors='replace')
    return value


def launch_repair():
    """One explicit continuation of the exact zero-update failure; no new clock."""
    from repair import validate_resume, safe_path, matches
    lock_path=BASE/'artifacts/microduck-headroom-gpu.lock'
    with lock_path.open('a') as lease:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            require((SLOT/'repair-resume-started.json').exists(),'Another campaign holds the GPU lease')
            return snapshot()
        if (SLOT/'repair-resume-started.json').exists():return snapshot()
        package_id=verify_package();receipt=validate_resume(SLOT,package_id,time.time())
        require(not any(alive(optional(name+'.json')) for name in ('supervisor','child','transfer-worker')),
                'Old campaign or transfer process still alive; preserve')
        require(not (SLOT/'STOP.json').exists() and not (SLOT/'uncertain-worker.json').exists(),
                'Stop request or uncertain worker exists; preserve')
        checks=preflight();verify_protected(read(SLOT/'protected-records.json'))
        contract=read(HERE/'repair-contract.json')
        require(time.time()<contract['original_train_stop_utc'],'Original training window expired; no restart')
        root=SLOT/'repairs'/receipt['repair_id'];retired=root/'retired-control'
        retired.mkdir(exist_ok=True)
        names=('exit.json','progress.json','supervisor-started.json','supervisor.json','child.json',
               'active-worker.json','transfer-worker.json','transfer-outcome.json','return.json',
               'logs/supervisor.log','logs/automatic-transfer.log')
        pins={}
        for name in names:
            path=safe_path(SLOT,name)
            if path.exists():
                pins[name]=dict(bytes=path.stat().st_size,sha256=sha(path))
                saved=safe_path(retired,name);saved.parent.mkdir(parents=True,exist_ok=True)
                if not saved.exists():
                    with saved.open('xb') as f:f.write(path.read_bytes());f.flush();os.fsync(f.fileno())
                require(matches(saved,pins[name]),'Old control evidence differs: '+name)
        write(root/'resume-preflight.json',checks,exclusive=True)
        # Reserve this one continuation before removing markers or spawning.
        # A failure after this point is preserved, never automatically restarted.
        write(SLOT/'repair-resume-started.json',dict(repair_id=receipt['repair_id'],
            package_identity=package_id,requested_utc=time.time(),session_base=3,
            original_started_utc=contract['original_started_utc'],
            original_train_stop_utc=contract['original_train_stop_utc'],
            original_deadline_utc=contract['original_deadline_utc'],retired_control=pins,
            budget_renewed=False,automatic_retry=False),exclusive=True)
        for name,pin in pins.items():
            path=safe_path(SLOT,name);require(matches(path,pin),'Control changed before retirement: '+name)
            path.unlink()
        for name in ('budget.json','started.json'):
            require(matches(SLOT/name,contract['preserved_files'][name]),'Original budget/start changed')
        write(SLOT/'progress.json',dict(status='STARTING',phase='smoke',detail='repair_supervisor'))
        child=None
        try:
            with (SLOT/'logs/supervisor.log').open('xb') as log:
                child=subprocess.Popen([str(PYTHON),'-B','-u',str(HERE/'agent.py'),'supervise',
                    '--lock-fd',str(lease.fileno())],cwd=SLOT,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    pass_fds=(lease.fileno(),),env=worker_environment())
            pin=process_pin(child);require(pin is not None,'Repair launch uncertain; preserve')
            write(SLOT/'supervisor.json',pin,exclusive=True)
        except BaseException as exc:
            write(SLOT/'repair-launch-error.json',dict(error=repr(exc),automatic_retry=False,
                  original_budget_preserved=True),exclusive=True)
            raise
        return snapshot()


def launch():
    # Bootstrap creates only the isolated package, never the budget.
    require(SLOT.is_dir() and not SLOT.is_symlink(),'Campaign slot missing or symlink')
    lock_path=BASE/'artifacts/microduck-headroom-gpu.lock'
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open('a') as lease:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            require((SLOT/'started.json').is_file(),'Another GPU campaign holds the lease')
            return snapshot()
        if (SLOT/'started.json').exists():return snapshot()
        require(not any((SLOT/n).exists() for n in ('budget.json','supervisor.json','child.json','exit.json')),
                'Orphan campaign records exist; preserve, never reset')
        checks=preflight();now=time.time();ledger=account.new(now)
        require(ledger['deadline_utc']==now+MAX_SECONDS and ledger['train_stop_utc']==now+TRAIN_SECONDS,
                'New campaign deadlines differ')
        for folder in ('run/checkpoints','run/candidates','evaluations','logs','cache'):
            (SLOT/folder).mkdir(parents=True,exist_ok=True)
        started=dict(schema='stage12-push10-final6h-v1',started_utc=now,
            deadline_utc=ledger['deadline_utc'],train_stop_utc=ledger['train_stop_utc'],
            max_compute_seconds=MAX_SECONDS,max_training_seconds=TRAIN_SECONDS,
            max_updates=MAX_UPDATES,package_identity=checks['package_identity'],
            parent_checkpoint_sha256=MODEL_SHA,parent_policy_updates=START_POLICY_UPDATES,
            automatic_retry=False,automatic_budget_renewal=False,last_training_campaign=True,
            training_authorization='User requested the final six-hour training campaign',power_action='NONE')
        write(SLOT/'started.json',started,exclusive=True)
        write(SLOT/'budget.json',ledger,exclusive=True)
        write(SLOT/'protected-records.json',checks['protected_records'],exclusive=True)
        write(SLOT/'preflight.json',checks,exclusive=True)
        write(SLOT/'progress.json',dict(status='STARTING',phase='smoke',detail='supervisor'))
        child=None
        try:
            with (SLOT/'logs/supervisor.log').open('xb') as log:
                child=subprocess.Popen([str(PYTHON),'-B','-u',str(HERE/'agent.py'),'supervise',
                    '--lock-fd',str(lease.fileno())],cwd=SLOT,stdin=subprocess.DEVNULL,
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                    pass_fds=(lease.fileno(),),env=worker_environment())
            pin=process_pin(child);require(pin is not None,'Supervisor launch uncertain; preserve started marker')
            write(SLOT/'supervisor.json',pin,exclusive=True)
        except BaseException as exc:
            write(SLOT/'launch-error.json',dict(error=repr(exc),pid=None if child is None else child.pid,
                automatic_retry=False,started_budget_preserved=True),exclusive=True)
            raise
        return snapshot()


def worker_environment():
    temp=SLOT/'cache/tmp';temp.mkdir(parents=True,mode=0o700,exist_ok=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('MICRODUCK_')
         and k not in ('WORLD_SIZE','RANK','LOCAL_RANK','PYTHONPATH','PYTHONHOME')}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',
        TMPDIR=str(temp),TEMP=str(temp),TMP=str(temp),MPLCONFIGDIR=str(SLOT/'cache/mpl'))
    return env


def run_worker(kind,args,number,fd,deadline,stopped):
    logpath=SLOT/'logs'/('%03d-%s.log'%(number,kind))
    child=pin=None;term_at=None;stop_reason=None
    try:
        with logpath.open('xb') as log:
            child=subprocess.Popen([str(PYTHON),'-B','-u',str(HERE/(kind+'.py')),*args],
                cwd=SLOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=(fd,),env=worker_environment())
            pin=process_pin(child)
            if pin is None and child.poll() is None:
                write(SLOT/'uncertain-worker.json',dict(pid=child.pid,number=number,kind=kind,
                    log=str(logpath),reason='Process identity unavailable; do not claim quiescence'),exclusive=True)
            require(pin is not None,'Worker launch uncertain; preserve')
            write(SLOT/'child.json',pin)
            write(SLOT/'active-worker.json',dict(number=number,kind=kind,args=args,log=str(logpath),pin=pin))
            while child.poll() is None:
                if stopped[0] or (SLOT/'STOP.json').exists() or time.time()>=deadline:
                    if term_at is None:
                        stop_reason='user_stop' if stopped[0] or (SLOT/'STOP.json').exists() else 'compute_deadline'
                        signal_owned(child,pin,signal.SIGTERM);term_at=time.monotonic()
                    elif time.monotonic()-term_at>=90:
                        signal_owned(child,pin,signal.SIGKILL)
                try:child.wait(timeout=2)
                except subprocess.TimeoutExpired:pass
        return dict(worker_exit_code=child.returncode,stop_reason=stop_reason,worker_alive=False,log=str(logpath))
    finally:
        if child is not None and child.poll() is None:
            signal_owned(child,pin,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                signal_owned(child,pin,signal.SIGKILL);child.wait(timeout=15)


def candidate_receipts():
    paths=[];seen=set()
    for name in ('candidate-01.json','candidate-02.json','candidate-final.json'):
        path=SLOT/'run/candidates'/name
        if not path.exists():continue
        value=read(path);model=Path(value['path'])
        require(model.resolve().is_relative_to((SLOT/'run/checkpoints').resolve()) and not model.is_symlink(),
                'Candidate checkpoint outside this campaign')
        require(model.stat().st_size==value['bytes'] and sha(model)==value['sha256'],
                'Candidate checkpoint changed')
        require(value['policy_updates']==START_POLICY_UPDATES+value['completed_updates'],
                'Candidate policy accounting differs')
        if value['sha256'] not in seen:paths.append(path);seen.add(value['sha256'])
    require(0<len(paths)<=3,'No complete saved candidate, or too many candidates')
    return paths


def supervise(fd):
    os.fstat(fd);started=read(SLOT/'started.json')
    require(started['deadline_utc']==started['started_utc']+MAX_SECONDS
        and started['train_stop_utc']==started['started_utc']+TRAIN_SECONDS,
        'Original six-hour deadline changed')
    write(SLOT/'supervisor-started.json',dict(started_utc=time.time(),pid=os.getpid()),exclusive=True)
    stopped=[False];original_handlers={}
    for sig in (signal.SIGTERM,signal.SIGINT):
        original_handlers[sig]=signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    status='CAMPAIGN_ERROR_PRESERVE';reason=None;error=None;qualified=False;heldout_passed=False
    selection=None;heldout=None
    def boundary():
        require(not stopped[0] and not (SLOT/'STOP.json').exists()
                and time.time()<started['deadline_utc'],'STOP_REQUEST_OR_COMPUTE_DEADLINE')
    def train(phase,number):
        boundary()
        outcome=run_worker('train',['--phase',phase,'--session',str(number)],number,fd,
                           started['train_stop_utc'],stopped)
        path=SLOT/'run'/('session-%03d'%number)/'result.json';value=read(path)
        normal_window=(phase=='formal' and outcome['stop_reason']=='compute_deadline'
                       and value.get('reason')=='train_window_complete' and not (SLOT/'STOP.json').exists())
        require(outcome['worker_exit_code']==0 and (not outcome['stop_reason'] or normal_window)
            and value['status']=='TRAINING_PHASE_COMPLETE' and value['phase']==phase,
            'Training phase did not complete: '+phase)
        if phase=='smoke':
            require(value.get('smoke_export',{}).get('passed') is True,'New configuration smoke/export failed')
            require(read(SLOT/'budget.json')['completed_updates']==5,'Smoke must consume exactly five PPO updates')
        write(SLOT/('complete-'+phase+'.json'),dict(path=str(path),sha256=sha(path),process=outcome),exclusive=True)
        return value
    def evaluate(phase,receipt,number,index=None):
        boundary()
        folder=SLOT/'evaluations'/('heldout' if phase=='heldout' else 'selection-%03d'%index)
        write(SLOT/'progress.json',dict(status='EVALUATING',phase=phase,detail=folder.name,
                                       checkpoint=read(receipt),deadline_utc=started['deadline_utc']))
        outcome=run_worker('evaluate',['--phase',phase,'--checkpoint-json',str(receipt),
                         '--run-dir',str(folder)],number,fd,started['deadline_utc'],stopped)
        result_path=folder/'evaluation-result.json';value=read(result_path)
        require(outcome['worker_exit_code']==0 and not outcome['stop_reason']
            and value['status']=='EVALUATION_COMPLETE' and value['checkpoint']==read(receipt),
            'Evaluation incomplete or runtime failure: '+folder.name)
        from eval_criteria import summarize_phase
        require(value['summary']==summarize_phase(value['cases'],phase,value['checkpoint']['sha256']),
                'Evaluation summary differs from retained cases')
        for evidence in value.get('evidence',[]):
            path=Path(evidence['path'])
            require(path.resolve().is_relative_to(folder.resolve()) and sha(path)==evidence['sha256'],
                    'Evaluation case evidence differs')
        return value
    try:
        first=optional('repair-resume-started.json',{}).get('session_base',1)
        train('smoke',first);train('formal',first+1)
        results=[]
        for index,receipt in enumerate(candidate_receipts(),1):
            results.append(evaluate('selection',receipt,first+1+index,index))
        from eval_plan import choose_candidate
        selection=choose_candidate(results);qualified=bool(selection['qualified'])
        write(SLOT/'selection.json',selection,exclusive=True)
        if qualified:
            locked=SLOT/'selected-candidate.json';write(locked,selection['checkpoint'],exclusive=True)
            heldout=evaluate('heldout',locked,first+5)
            heldout_passed=bool(heldout['summary']['all_passed'])
            write(SLOT/'selected.json',dict(checkpoint=selection['checkpoint'],selection=selection,
                heldout=heldout,disturbance_goal_passed=heldout_passed,
                checked_baseline_goal_passed=heldout_passed,hardware_acceptance=False,
                selected_once=True,no_more_training_authorized=True),exclusive=True)
            reason='FINAL_HELDOUT_PASSED' if heldout_passed else 'FINAL_HELDOUT_FAILED_NO_MORE_TRAINING'
        else:reason='NO_QUALIFIED_CANDIDATE_NO_MORE_TRAINING'
        status='CAMPAIGN_COMPLETE'
    except BaseException as exc:
        error=dict(error=repr(exc),traceback=traceback.format_exc()[-12000:])
        if stopped[0] or (SLOT/'STOP.json').exists() or time.time()>=started['deadline_utc']:
            status='CAMPAIGN_STOPPED_PRESERVE';reason='STOP_REQUEST_OR_COMPUTE_DEADLINE'
    try:
        verify_inputs();verify_protected(read(SLOT/'protected-records.json'));unchanged=True
    except BaseException as exc:
        unchanged=False;status='CAMPAIGN_ERROR_PRESERVE';error=dict(error=repr(exc),earlier_error=error)
    ledger=read(SLOT/'budget.json')
    ended=dict(status=status,reason=reason,error=error,finished_utc=time.time(),
        deadline_utc=started['deadline_utc'],train_stop_utc=started['train_stop_utc'],
        qualified_candidate_found=qualified,disturbance_goal_passed=bool(heldout_passed and unchanged),
        checked_baseline_goal_passed=bool(heldout_passed and unchanged),
        original_inputs_and_records_unchanged=unchanged,added_PPO_updates=ledger['completed_updates'],
        attempted_updates=ledger['attempted_updates'],pending_attempt_preserved=ledger.get('pending'),
        automatic_retry=False,automatic_training_extension=False,last_training_campaign=True,
        power_action='NONE',instance_power_state='UNCONFIRMED',
        compute_workers_exited=not (SLOT/'uncertain-worker.json').exists() and not alive(optional('child.json')))
    require(ended['compute_workers_exited'],'A compute worker is still alive; do not package changing files')
    write(SLOT/'exit.json',ended,exclusive=True)
    write(SLOT/'progress.json',dict(status=status,phase='finished',detail='compute_finished',reason=reason))
    # CPU-only artifact return is detached too; closing the laptop never cancels it.
    transfer_outcome=None
    try:
        with (SLOT/'logs/automatic-transfer.log').open('xb') as log:
            child=subprocess.Popen([str(PYTHON),'-B','-u',str(HERE/'transfer.py')],cwd=SLOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                env=worker_environment())
            pin=process_pin(child);require(pin is not None,'Transfer launch uncertain')
            write(SLOT/'transfer-worker.json',pin)
            try:code=child.wait(timeout=1800)
            except subprocess.TimeoutExpired:
                signal_owned(child,pin,signal.SIGTERM)
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:signal_owned(child,pin,signal.SIGKILL);child.wait(timeout=15)
                raise RuntimeError('CPU artifact return timeout; retry --return only')
        transfer_outcome=dict(status='COMPLETE' if code==0 else 'RETURN_FAILED_PRESERVE',
            exit_code=code,finished_utc=time.time(),new_PPO_updates=0,automatic_training=False)
    except BaseException as exc:
        transfer_outcome=dict(status='RETURN_FAILED_PRESERVE',error=repr(exc),finished_utc=time.time(),
                              new_PPO_updates=0,automatic_training=False)
    write(SLOT/'transfer-outcome.json',transfer_outcome,exclusive=True)
    for sig,handler in original_handlers.items():signal.signal(sig,handler)
    return snapshot()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=['start','resume-repair','status','collect','stop','supervise'])
    parser.add_argument('--lock-fd',type=int)
    args=parser.parse_args()
    if args.operation=='start':value=launch()
    elif args.operation=='resume-repair':value=launch_repair()
    elif args.operation=='supervise':value=supervise(args.lock_fd)
    elif args.operation=='stop':
        if (SLOT/'started.json').exists() and not (SLOT/'exit.json').exists() and not (SLOT/'STOP.json').exists():
            write(SLOT/'STOP.json',dict(requested_utc=time.time(),reason='Explicit user stop; no power action'),exclusive=True)
        value=snapshot()
    else:value=snapshot()
    print('MD12_FINAL6H_STATE='+json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print('MD12_FINAL6H_ERROR='+json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()[-6000:])),flush=True)
        raise SystemExit(2)
