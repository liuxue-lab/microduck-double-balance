"""Detached finite orchestration. Status and collection never launch training."""
import argparse
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback
from common import *
import budget as account
from criteria import summarize


def optional(name,default=None):
    p=SLOT/name
    return read(p) if p.is_file() else default


def snapshot():
    started=optional('started.json')
    if started is None:return dict(status='NOT_STARTED',slot=str(SLOT),added_PPO_updates=0)
    session=optional('session.json',{})
    running={name:alive(optional(name+'.json')) for name in ('supervisor','child')}
    progress=optional('progress.json',{'status':'STARTING'})
    ledger=optional('budget.json',{})
    value=dict(status=progress['status'],slot=str(SLOT),started=started,budget=ledger,
        progress=progress,session=session,processes_alive=running,selection=optional('selection.json'),
        selected=optional('selected.json'),added_PPO_updates=ledger.get('completed_updates',0),
        package_identity=read(HERE/'manifest.json')['identity'],power_state='UNCONFIRMED',power_action='NONE')
    ended=optional('exit.json')
    if ended and ended['session']==session.get('number'):
        value.update(status=ended['status'],exit=ended)
    elif not any(running.values()) and time.time()-session.get('requested_utc',0)>30:
        value['status']='CAMPAIGN_INTERRUPTED_PRESERVE'
    if value['status'].endswith('PRESERVE'):
        info=optional('active-worker.json',{})
        log=Path(info.get('log',str(SLOT/'logs/supervisor.log')))
        if log.is_file() and log.resolve().is_relative_to(SLOT.resolve()):
            with log.open('rb') as f:
                f.seek(max(0,log.stat().st_size-16000));value['log_tail']=f.read().decode(errors='replace')
    return value


def launch(resume=False):
    lease=(BASE/'artifacts'/'microduck-headroom-gpu.lock').open('a')
    try:
        try:fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            require((SLOT/'started.json').is_file(),'Another headroom process holds the GPU lease')
            return snapshot()
        started=optional('started.json')
        if started is not None and not resume:return snapshot()
        require(started is not None or not resume,'No campaign exists to resume')
        if optional('exit.json',{}).get('status')=='CAMPAIGN_COMPLETE':return snapshot()
        require(not alive(optional('supervisor.json')) and not alive(optional('child.json')),
                'A campaign process is still alive')
        checks=preflight();now=time.time()
        if started is None:
            for name in ('run/checkpoints','evaluations','logs','cache'):
                (SLOT/name).mkdir(parents=True,exist_ok=True)
            ledger=account.new(now)
            started=dict(started_utc=now,deadline_utc=ledger['deadline_utc'],max_updates=505,
                max_compute_seconds=7200,max_cleanup_seconds=90,max_evaluation_episodes=8,
                smoke_updates=5,smoke_environments=64,formal_updates=500,formal_environments=4096,
                source_checkpoint_sha256=MODEL_SHA,package_identity=checks['package_identity'],
                approval=read(HERE/'approval.json'),old_budgets_frozen=True,
                automatic_retry=False,automatic_extension=False,disturbance_training=False)
            write(SLOT/'preflight.json',checks,exclusive=True)
            write(SLOT/'protected-records.json',checks['protected_records'],exclusive=True)
            write(SLOT/'budget.json',ledger,exclusive=True)
            write(SLOT/'selection.json',dict(history=[],candidate=None),exclusive=True)
            write(SLOT/'started.json',started,exclusive=True)
        else:
            require(started['package_identity']==checks['package_identity'],'Cannot replace code in an existing campaign')
            verify_protected(read(SLOT/'protected-records.json'))
            ledger=read(SLOT/'budget.json')
            account.resumable(ledger,input_checkpoint()['completed_updates'],now)
            require(optional('exit.json',{}).get('status')!='CAMPAIGN_ERROR_PRESERVE',
                    'Runtime error needs review; --resume does not repair or repeat it')
        number=optional('session.json',{'number':0})['number']+1
        if (SLOT/'STOP.json').exists():
            os.rename(SLOT/'STOP.json',SLOT/('STOP-before-session-%03d.json'%number))
        session=dict(number=number,requested_utc=now,resume=resume,original_deadline_utc=ledger['deadline_utc'])
        write(SLOT/('session-%03d.json'%number),session,exclusive=True);write(SLOT/'session.json',session)
        write(SLOT/'progress.json',dict(status='STARTING',phase='supervisor',completed_updates=ledger['completed_updates']))
        with (SLOT/'logs'/('supervisor-%03d.log'%number)).open('xb') as log:
            child=subprocess.Popen([str(PYTHON),'-B','-u',str(HERE/'agent.py'),'supervise',
                '--session',str(number),'--lock-fd',str(lease.fileno())],cwd=SLOT,
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,
                pass_fds=(lease.fileno(),),env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
        pin=identity(child.pid);require(pin is not None,'Supervisor launch uncertain; preserve started marker')
        write(SLOT/'supervisor.json',pin)
        return snapshot()
    finally:lease.close()


def worker_environment():
    temp=SLOT/'cache/tmp';temp.mkdir(mode=0o700,exist_ok=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('MICRODUCK_')
         and k not in ('WORLD_SIZE','RANK','LOCAL_RANK')}
    env.update(PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='0',
        OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',WANDB_MODE='disabled',HF_HUB_OFFLINE='1',
        TMPDIR=str(temp),TEMP=str(temp),TMP=str(temp),MPLCONFIGDIR=str(SLOT/'cache/mpl'))
    return env


def run_worker(kind,phase,fd,deadline,stopped):
    number=optional('worker-counter.json',{'number':0})['number']+1
    write(SLOT/'worker-counter.json',{'number':number})
    logpath=SLOT/'logs'/('%03d-%s-%s.log'%(number,kind,phase))
    argv=[str(PYTHON),'-B','-u',str(HERE/(kind+'.py')),'--phase',phase]
    if kind=='train':argv+=['--session',str(number)]
    child=pin=None;term_at=None
    try:
        with logpath.open('xb') as log:
            child=subprocess.Popen(argv,cwd=SLOT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,
                start_new_session=True,pass_fds=(fd,),env=worker_environment())
            pin=identity(child.pid);require(pin is not None,'Worker launch uncertain; preserve')
            write(SLOT/'child.json',pin)
            write(SLOT/'active-worker.json',dict(number=number,kind=kind,phase=phase,log=str(logpath),pin=pin))
            while child.poll() is None:
                if stopped[0] or (SLOT/'STOP.json').exists() or time.time()>=deadline:
                    if term_at is None and alive(pin):
                        os.killpg(pin['group'],signal.SIGTERM);term_at=time.monotonic()
                if term_at is not None and time.monotonic()-term_at>=90 and alive(pin):
                    os.killpg(pin['group'],signal.SIGKILL)
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
        require(child.returncode==0,'WORKER_RETURN_%s_%s_%s'%(kind,phase,child.returncode))
        return number
    finally:
        if child is not None and child.poll() is None:
            if alive(pin):os.killpg(pin['group'],signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                if alive(pin):os.killpg(pin['group'],signal.SIGKILL)
                child.wait(timeout=15)


def supervise(fd,session):
    os.fstat(fd);require(read(SLOT/'session.json')['number']==session,'Supervisor session mismatch')
    deadline=read(SLOT/'started.json')['deadline_utc'];stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    status='CAMPAIGN_ERROR_PRESERVE';reason=None;error=None;task_passed=False
    def boundary():
        require(not stopped[0] and not (SLOT/'STOP.json').exists() and time.time()<deadline,
                'STOP_REQUEST_OR_ORIGINAL_DEADLINE')
    def training_phase(phase):
        boundary();marker=SLOT/('complete-'+phase+'.json')
        if marker.is_file():
            pin=read(marker)
            require(sha(pin['path'])==pin['sha256'],'Training completion record changed')
            return
        ledger=read(SLOT/'budget.json')
        if ledger['completed_updates']<TARGETS[phase]:
            number=run_worker('train',phase,fd,deadline,stopped)
        else:
            active=read(SLOT/'active-worker.json')
            require(active['kind']=='train' and active['phase']==phase,
                    'Phase target consumed without matching completed worker; preserve')
            number=active['number']
        path=SLOT/'run'/('session-%03d'%number)/'result.json'
        result=read(path)
        require(result['status']=='TRAINING_PHASE_COMPLETE' and result['phase']==phase
                and result['completed_updates']==TARGETS[phase],'Training phase incomplete')
        if phase=='smoke':require(result.get('smoke_export',{}).get('passed') is True,'New-config export smoke did not pass')
        write(marker,dict(path=str(path),sha256=sha(path)),exclusive=True)
    def evaluate(phase,checkpoint):
        boundary();folder=SLOT/'evaluations'/phase;folder.mkdir(exist_ok=True)
        candidate=folder/'candidate.json'
        if not candidate.exists():write(candidate,checkpoint,exclusive=True)
        require(read(candidate)==checkpoint,'Candidate identity changed')
        require(not (folder/'error.json').exists(),'Prior evaluation runtime fault is preserved')
        path=folder/'evaluation-result.json'
        if not path.exists():run_worker('evaluate',phase,fd,deadline,stopped)
        value=read(path);ledger=read(SLOT/'budget.json');rows=[]
        for seed in EVAL_SEEDS[phase]:
            p=folder/str(seed)/'result.json';entry=ledger['evaluations'][phase+':'+str(seed)]
            require(entry['status']=='COMPLETE' and sha(p)==entry['result_sha256'],
                    'Evaluation completion hash differs')
            rows.append(read(p))
        summary=summarize(rows,EVAL_SEEDS[phase],checkpoint['sha256'])
        require(value['status']=='EVALUATION_COMPLETE' and value['checkpoint']==checkpoint
                and value['summary']==summary,'Evaluation aggregate differs')
        return dict(phase=phase,checkpoint=checkpoint,summary=summary,
                    evaluation_path=str(path),evaluation_sha256=sha(path))
    try:
        training_phase('smoke')
        selection=read(SLOT/'selection.json')
        chosen=selection['candidate']
        if chosen is None:
            for index in (1,2):
                phase='selection-'+str(index)
                retained=next((x for x in selection['history'] if x['phase']==phase),None)
                if retained is not None:
                    require(sha(retained['evaluation_path'])==retained['evaluation_sha256'],'Selection evidence changed')
                    if retained['summary']['all_passed']:chosen=retained;break
                    continue
                training_phase('segment-'+str(index))
                checkpoint=input_checkpoint()
                require(checkpoint['completed_updates']==TARGETS['segment-'+str(index)],'Candidate update count differs')
                candidate=evaluate(phase,checkpoint)
                selection['history'].append(candidate)
                if candidate['summary']['all_passed']:
                    chosen=candidate;selection['candidate']=candidate
                write(SLOT/'selection.json',selection)
                if chosen is not None:break
        if chosen is None:
            reason='NO_CANDIDATE_PASSED_CHECKED_BOUNDS';status='CAMPAIGN_COMPLETE'
        else:
            heldout=evaluate('heldout',chosen['checkpoint'])
            task_passed=heldout['summary']['all_passed']
            reason='HEADROOM_BASELINE_PASSED_THIS_SEED_SET' if task_passed else 'HELDOUT_FAILED_NO_FURTHER_TUNING'
            if task_passed and not (SLOT/'selected.json').exists():
                write(SLOT/'selected.json',dict(checkpoint=chosen['checkpoint'],selection=chosen,heldout=heldout,
                    baseline_goal_passed=True,disturbance_goal_passed=False,hardware_acceptance=False,
                    population_success_rate_claimed=False),exclusive=True)
            status='CAMPAIGN_COMPLETE'
    except BaseException as exc:
        error=dict(error=repr(exc),traceback=traceback.format_exc()[-10000:])
        if stopped[0] or (SLOT/'STOP.json').exists() or time.time()>=deadline:
            status='CAMPAIGN_STOPPED_PRESERVE';reason='STOP_REQUEST_OR_ORIGINAL_DEADLINE'
    try:
        verify_inputs();verify_protected(read(SLOT/'protected-records.json'));unchanged=True
    except BaseException as exc:
        unchanged=False;status='CAMPAIGN_ERROR_PRESERVE';error=dict(error=repr(exc),earlier_error=error)
    result=dict(status=status,reason=reason,error=error,session=session,finished_utc=time.time(),
        checked_baseline_goal_passed=bool(task_passed and unchanged),disturbance_goal_passed=False,
        original_inputs_and_records_unchanged=unchanged,original_deadline_utc=deadline,
        added_PPO_updates=read(SLOT/'budget.json')['completed_updates'],automatic_retry=False,
        power_action='NONE',instance_power_state='UNCONFIRMED')
    write(SLOT/('exit-session-%03d.json'%session),result,exclusive=True);write(SLOT/'exit.json',result)
    return snapshot()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('operation',choices=['start','resume','status','collect','stop','supervise'])
    parser.add_argument('--lock-fd',type=int);parser.add_argument('--session',type=int)
    args=parser.parse_args()
    if args.operation in ('start','resume'):value=launch(args.operation=='resume')
    elif args.operation=='supervise':value=supervise(args.lock_fd,args.session)
    elif args.operation=='stop':
        value=snapshot()
        if (SLOT/'started.json').exists() and value['status']!='CAMPAIGN_COMPLETE':
            write(SLOT/'STOP.json',dict(requested_utc=time.time(),reason='Explicit user stop; no instance power action'))
            value=snapshot()
    else:value=snapshot()
    print('MD12_HEADROOM_STATE='+json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        error=dict(error=repr(exc),traceback=traceback.format_exc()[-10000:])
        try:error['state']=snapshot()
        except Exception as other:error['snapshot_error']=repr(other)
        print('MD12_HEADROOM_ERROR='+json.dumps(error,ensure_ascii=False,allow_nan=False),flush=True)
        raise SystemExit(2)
