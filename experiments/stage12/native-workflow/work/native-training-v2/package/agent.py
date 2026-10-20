"""One native smoke run followed by formal training, under one durable budget."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback
from common import (SLOT, LEGACY_SLOT, SOURCE, BASE, PYTHON, PRIMARY_SHA, PRIMARY_REL,
    MAX_CHECKPOINT_BYTES, read, write, sha, require, identity, alive, has_campaign,
    abandon_partial_for_explicit_resume)
from probe import snapshot, discover, collect_slot
HERE=Path(__file__).resolve().parent

def find_checkpoint():
    if (SLOT/'initialization.json').exists():
        x=read(SLOT/'initialization.json')
        require(x['sha256']==PRIMARY_SHA and sha(x['path'])==PRIMARY_SHA,'Stage08 initialization bytes differ')
        return x
    candidates=[BASE/PRIMARY_REL,SLOT/'initial-checkpoint.pt']
    # Only selected checkpoint names in previous model/evidence trees, no WAL scans.
    for folder in ('double-balance-stage08','double-balance-stage09','double-balance-stage10'):
        root=BASE/'artifacts'/folder
        if root.is_dir():
            candidates.extend(root.rglob('update_004000.pt'))
            candidates.extend(root.rglob('*.pt'))
    seen=set()
    for path in candidates:
        if str(path) in seen:continue
        seen.add(str(path))
        if path.is_file() and 0<path.stat().st_size<=MAX_CHECKPOINT_BYTES and sha(path)==PRIMARY_SHA:
            x=dict(path=str(path),sha256=PRIMARY_SHA,kind='stage08_selected_E',bytes=path.stat().st_size)
            write(SLOT/'initialization.json',x,exclusive=True);return x
    return None

def check_sources():
    for rel,pin in read(HERE/'source-pins.json').items():
        p=SOURCE/rel
        require(p.is_file() and p.stat().st_size==pin['bytes'] and sha(p)==pin['sha256'],'Frozen project differs: '+rel)
    manifest=read(HERE/'manifest.json')
    for rel,pin in manifest['files'].items():
        require(sha(HERE/rel)==pin['sha256'],'Training source differs: '+rel)
    return manifest['identity']

def launch(resume=False):
    # Also hold v1's exact launch lock for the full v2 lifetime. Its existing
    # launcher already respects this lock. Never alter old started/budget files.
    with ExitStack() as stack:
        handles=[]
        for slot in (LEGACY_SLOT,SLOT):
            slot.mkdir(exist_ok=True)
            require(not slot.is_symlink(),'Unexpected campaign slot symlink')
            f=stack.enter_context((slot/'launch.lock').open('a'));handles.append(f)
            try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return discover()
        prior=discover()
        if prior.get('campaign_exists') and (not resume or prior.get('slot')!=str(SLOT)):
            return prior
        if resume:
            require(not prior.get('supervisor_alive') and not prior.get('training_alive'),'Existing task remains active')
            require((SLOT/'run/smoke-result.json').exists() and read(SLOT/'run/smoke-result.json')['status']=='SMOKE_PASSED',
                    'The one smoke attempt did not pass; preserve it, no automatic retry')
            require((SLOT/'run/latest.json').exists(),'No training checkpoint to resume')
        else:
            require(not (SLOT/'smoke-started.json').exists(),'Prior smoke marker preserved; no new smoke')
        package_identity=check_sources();cfg=read(SLOT/'config.json')
        require(cfg['package_identity']==package_identity,'Configuration identity differs')
        require(find_checkpoint() is not None,'Selected Stage08 E checkpoint required; no earlier fallback')
        require(PYTHON.is_file(),'Existing cloud Python missing')
        require(shutil.disk_usage(SLOT).free>=8*1024**3,'Less than 8 GiB free for training checkpoints')
        if resume:
            b=read(SLOT/'budget.json')
            require(time.time()<b['deadline_utc'] and b['attempted_updates']<b['max_updates'],'Original budget exhausted')
            old=read(SLOT/'session.json');session=old['session']+1
            abandon_partial_for_explicit_resume(b);write(SLOT/'budget.json',b)
            if (SLOT/'STOP.json').exists():os.rename(SLOT/'STOP.json',SLOT/('STOP-session-%02d.json'%old['session']))
        else:
            session=1;now=time.time()
            b=dict(started_utc=now,deadline_utc=now+cfg['max_hours']*3600,max_updates=cfg['max_updates'],
                attempted_updates=0,completed_updates=0,pending=False,interrupted_attempts=0,
                historical_validation_reservations_preserved=4,validation_runs_added=1)
            write(SLOT/'started.json',dict(started_utc=now,package_identity=package_identity,
                mode='ONE_SMOKE_THEN_NATIVE_FINETUNING'),exclusive=True)
            write(SLOT/'budget.json',b,exclusive=True)
        write(SLOT/'session.json',dict(session=session,requested_utc=time.time(),resume=resume))
        (SLOT/'run').mkdir(exist_ok=True)
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1')
        fds=[f.fileno() for f in handles]
        with (SLOT/('supervisor-%02d.log'%session)).open('xb') as f:
            proc=subprocess.Popen([str(PYTHON),'-B','-u',str(HERE/'agent.py'),'supervise','--session',str(session),
                '--lock-fds',','.join(map(str,fds))],cwd=SLOT,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,
                start_new_session=True,env=env,pass_fds=tuple(fds))
        pin=identity(proc.pid);require(pin is not None,'Supervisor launch uncertain; preserve markers')
        write(SLOT/'supervisor.json',pin)
        return snapshot(SLOT)

def smoke_allows_formal(returncode, result, budget):
    return (returncode==0 and result.get('status')=='SMOKE_PASSED'
            and result.get('completed_updates')==5 and result.get('reload_verified') is True
            and budget['completed_updates']==5 and budget['attempted_updates']==5 and not budget['pending'])

def supervise(session,lock_fds):
    # These inherited descriptors retain both launch locks even if the SSH client exits.
    for fd in lock_fds:os.fstat(fd)
    b=read(SLOT/'budget.json');cfg=read(SLOT/'config.json')
    temp=Path('/tmp')/('md12nt-'+cfg['package_identity'][:10]);temp.mkdir(mode=0o700,exist_ok=True)
    require(not temp.is_symlink() and temp.stat().st_uid==os.getuid(),'Unexpected temporary path')
    cache=SLOT/'cache';cache.mkdir(exist_ok=True);(cache/'warp').mkdir(exist_ok=True)
    env={k:v for k,v in os.environ.items() if not k.startswith('MICRODUCK_') and k not in ('WORLD_SIZE','RANK','LOCAL_RANK')}
    env.update(PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
        TMPDIR=str(temp),TEMP=str(temp),TMP=str(temp),MPLCONFIGDIR=str(cache/'mpl'),TORCH_HOME=str(cache/'torch'),
        HF_HUB_OFFLINE='1',WANDB_MODE='disabled',PYTHONUNBUFFERED='1')
    def run_phase(phase):
        require(time.time()<b['deadline_utc'] and not (SLOT/'STOP.json').exists(),'Campaign stopped before next phase')
        phase_deadline=min(b['deadline_utc'],time.time()+cfg['smoke_max_minutes']*60) if phase=='smoke' else b['deadline_utc']
        log=SLOT/'run'/('train-%02d-%s.log'%(session,phase))
        with log.open('xb') as f:
            proc=subprocess.Popen([str(PYTHON),'-B','-u',str(HERE/'train.py'),'--session',str(session),'--phase',phase],
                cwd=SLOT,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,
                pass_fds=tuple(lock_fds))
            pin=identity(proc.pid);require(pin is not None,'Worker launch uncertain; preserve markers')
            write(SLOT/'child.json',pin);term_at=None
            while proc.poll() is None:
                requested=(SLOT/'STOP.json').exists() or time.time()>=phase_deadline
                if requested and term_at is None and alive(pin):
                    os.killpg(pin['group'],signal.SIGTERM);term_at=time.monotonic()
                if term_at is not None and time.monotonic()-term_at>90 and alive(pin):os.killpg(pin['group'],signal.SIGKILL)
                try:proc.wait(timeout=5)
                except subprocess.TimeoutExpired:pass
        return proc.returncode
    phases=[]
    if session==1:
        write(SLOT/'smoke-started.json',dict(started_utc=time.time(),num_envs=64,updates=5,steps_per_env=24,maximum_attempts=1,max_minutes=cfg['smoke_max_minutes']),exclusive=True)
        code=run_phase('smoke');phases.append(dict(phase='smoke',returncode=code))
        result=read(SLOT/'run/smoke-result.json') if (SLOT/'run/smoke-result.json').exists() else {}
        if not smoke_allows_formal(code,result,read(SLOT/'budget.json')):
            write(SLOT/'run/progress.json',dict(status='SMOKE_FAILED_PRESERVE',phase='smoke',result=result))
            write(SLOT/('exit-%02d.json'%session),dict(phases=phases,finished_utc=time.time()),exclusive=True)
            return snapshot(SLOT)
    require(read(SLOT/'run/smoke-result.json')['status']=='SMOKE_PASSED','No passed smoke; cannot train')
    code=run_phase('formal');phases.append(dict(phase='formal',returncode=code))
    write(SLOT/('exit-%02d.json'%session),dict(phases=phases,finished_utc=time.time()),exclusive=True)
    return snapshot(SLOT)

def receive_checkpoint(size,offset=None,chunk_sha=None):
    require(not has_campaign(SLOT) and not has_campaign(LEGACY_SLOT),'Existing task preserved; upload disabled')
    require(0<size<=MAX_CHECKPOINT_BYTES,'Unexpected checkpoint size')
    target=SLOT/'initial-checkpoint.pt';partial=SLOT/'initial-checkpoint.pt.partial'
    meta=SLOT/'checkpoint-upload.json'
    if target.exists():
        require(target.stat().st_size==size and sha(target)==PRIMARY_SHA,'Existing model differs')
        return dict(status='CHECKPOINT_PRESENT',offset=size,initialization=find_checkpoint())
    with (SLOT/'checkpoint-upload.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        expected=dict(bytes=size,sha256=PRIMARY_SHA)
        if meta.exists():require(read(meta)==expected,'Prior partial upload preserved; metadata differs')
        else:write(meta,expected,exclusive=True)
        current=partial.stat().st_size if partial.exists() else 0
        require(current<=size,'Oversized partial file preserved')
        if offset is not None:
            data=sys.stdin.buffer.read(1024**2+1)
            require(0<len(data)<=1024**2 and hashlib.sha256(data).hexdigest()==chunk_sha,'Upload chunk differs')
            require(0<=offset and offset+len(data)<=size,'Invalid upload offset')
            if offset<current:
                with partial.open('rb') as f:f.seek(offset);existing=f.read(len(data))
                require(existing==data,'Repeated chunk differs; preserved')
            else:
                require(offset==current,'Noncontiguous upload; preserve received bytes')
                with partial.open('ab') as f:f.write(data);f.flush();os.fsync(f.fileno())
            current=partial.stat().st_size
        if current==size:
            require(sha(partial)==PRIMARY_SHA,'Completed upload hash differs; partial preserved')
            os.rename(partial,target)
            return dict(status='CHECKPOINT_PRESENT',offset=size,initialization=find_checkpoint())
        return dict(status='CHECKPOINT_PARTIAL',offset=current,total_bytes=size)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('operation',choices=['prepare','start','resume','status','stop','collect','receive','supervise'])
    ap.add_argument('--session',type=int);ap.add_argument('--size',type=int);ap.add_argument('--offset',type=int)
    ap.add_argument('--chunk-sha');ap.add_argument('--lock-fds',default='');a=ap.parse_args()
    prior=discover()
    if a.operation not in ('supervise','status','collect') and prior.get('campaign_exists') and prior.get('slot')!=str(SLOT):
        result=prior
    elif a.operation=='prepare':
        if prior.get('campaign_exists'):result=prior
        else:
            init=find_checkpoint();result=dict(status='READY' if init else 'NEEDS_INITIAL_CHECKPOINT',slot=str(SLOT),initialization=init)
    elif a.operation in ('start','resume'):result=launch(a.operation=='resume')
    elif a.operation=='status':result=prior if prior.get('campaign_exists') else snapshot(SLOT)
    elif a.operation=='collect':result=collect_slot(Path(prior['slot'])) if prior.get('slot') else prior
    elif a.operation=='supervise':result=supervise(a.session,[int(x) for x in a.lock_fds.split(',') if x])
    elif a.operation=='receive':result=receive_checkpoint(a.size,a.offset,a.chunk_sha)
    else:
        require(has_campaign(SLOT),'No new campaign exists')
        if not (SLOT/'STOP.json').exists():write(SLOT/'STOP.json',dict(requested_utc=time.time(),scope='native v2 only'),exclusive=True)
        result=snapshot(SLOT)
    print('MD12_NATIVE_STATE='+json.dumps(result,ensure_ascii=False,allow_nan=False),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print('MD12_NATIVE_ERROR='+json.dumps(dict(error=repr(exc),traceback=traceback.format_exc()[-12000:]),ensure_ascii=False),flush=True)
        raise SystemExit(2)
