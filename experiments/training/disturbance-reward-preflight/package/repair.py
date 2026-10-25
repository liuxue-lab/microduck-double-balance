"""Audited repair of the verified zero-update reward preflight failure; preserve the ledger."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

REPAIR_ID='reward-scaling-v2'


def require(ok,message):
    if not ok:raise RuntimeError(message)


def sha_bytes(raw):return hashlib.sha256(raw).hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def safe_path(root,name):
    path=Path(name)
    require(not path.is_absolute() and '..' not in path.parts,'Unsafe repair path')
    p=root
    require(not p.is_symlink(),'Symlink repair root')
    for part in path.parts:
        p=p/part;require(not p.is_symlink(),'Symlink repair path: '+str(p))
    return p


def matches(path,pin):
    return path.is_file() and not path.is_symlink() and path.stat().st_size==pin['bytes'] and sha_bytes(path.read_bytes())==pin['sha256']


def original_bytes(path,pin,append_allowed=False):
    require(path.is_file() and not path.is_symlink(),'Missing original evidence: '+str(path))
    raw=path.read_bytes()
    if append_allowed:
        require(len(raw)>=pin['bytes'],'Original log was truncated')
        raw=raw[:pin['bytes']]
    require(len(raw)==pin['bytes'] and sha_bytes(raw)==pin['sha256'],'Original evidence differs: '+str(path))
    return raw


def atomic(path,value):
    data=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode()
    temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temp.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)


def alive(pin):
    if not pin:return False
    try:
        fields=(Path('/proc')/str(int(pin['pid']))/'stat').read_text().rsplit(')',1)[1].split()
        return fields[19]==pin['start'] and fields[0] not in ('Z','X')
    except FileNotFoundError:return False


def validate_package(package,expected):
    manifest=read(package/'manifest.json')
    identity=sha_bytes(json.dumps(manifest['files'],sort_keys=True,separators=(',',':')).encode())
    require(identity==expected and manifest['identity']==expected,'Repair package identity differs')
    for name,pin in manifest['files'].items():
        require(Path(name).name==name and matches(safe_path(package,name),pin),'Repair package file differs: '+name)
    return manifest


def applied_receipt(slot,package_id):
    p=safe_path(slot,'repair-applied.json');receipt=read(p)
    require(receipt['repair_id']==REPAIR_ID and receipt['status']=='REPAIR_APPLIED'
            and receipt['package_identity']==package_id,'Repair receipt differs')
    origin=read(safe_path(slot,'source-commit.json'))
    require(origin==receipt['new_source'],'Repaired source commit differs')
    return receipt


def validate_resume(slot,package_id,now):
    receipt=applied_receipt(slot,package_id);contract=read(slot/'package/repair-contract.json')
    require(now<contract['original_train_stop_utc'],
            'Original training window expired; collect only; no automatic budget/clock renewal')
    for name in ('started.json','budget.json','exit.json','run/session-001/result.json'):
        require(matches(safe_path(slot,name),contract['preserved_files'][name]),
                'Known zero-update failure state changed; no repair resume: '+name)
    ledger=read(slot/'budget.json')
    require(ledger['attempted_updates']==ledger['completed_updates']==0 and ledger['pending'] is None
            and not ledger['evaluations'],'Repair is only for the verified zero-update preflight failure')
    require(not (slot/'run/latest.json').exists() and not list((slot/'run/checkpoints').glob('*.pt')),
            'Unexpected trained checkpoint; preserve')
    return receipt


def apply_repair(repo,slot,files,request):
    """Install in the same campaign, saving every old returned byte first."""
    contract=json.loads(files['repair-contract.json']);expected=request['package_identity']
    require(contract['repair_id']==REPAIR_ID,'Unknown repair')
    source=dict(commit=request['commit'],package_identity=expected,source_path=contract['repair_source_path'])
    repair_root=safe_path(slot,'repairs/'+REPAIR_ID)
    receipt_path=safe_path(slot,'repair-applied.json')
    if receipt_path.exists():
        receipt=applied_receipt(slot,expected);require(receipt['new_source']==source,'Repair commit changed')
        validate_package(slot/'package',expected);return receipt
    require(slot.is_dir(),'Original campaign is missing; never create a new budget')
    require(not (slot/'STOP.json').exists() and not (slot/'uncertain-worker.json').exists(),
            'Stop request or uncertain worker exists; preserve')
    lease=safe_path(repo.parent,'artifacts/microduck-headroom-gpu.lock')
    with lease.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('A campaign holds the GPU lease; repair must not replace running code')
        if receipt_path.exists():
            receipt=applied_receipt(slot,expected)
            require(receipt['new_source']==source,'Concurrent repair commit differs')
            validate_package(slot/'package',expected);return receipt
        for name in ('supervisor.json','child.json','transfer-worker.json'):
            p=safe_path(slot,name)
            require(not alive(read(p) if p.exists() else None),'Original worker is still alive; preserve')
        # These records are never replaced, including after an interrupted installation.
        for name in ('budget.json','started.json','exit.json','run/session-001/result.json'):
            require(matches(safe_path(slot,name),contract['preserved_files'][name]),'Original failure changed: '+name)
        result=read(slot/'run/session-001/result.json');ledger=read(slot/'budget.json')
        require(result['error']==contract['allowed_error'] and result['status']=='TRAINING_ERROR_PRESERVE'
                and ledger['pending'] is None and ledger['attempted_updates']==ledger['completed_updates']==0
                and not ledger['evaluations'],'Not the verified reward preflight failure; preserve')
        require(not (slot/'run/latest.json').exists() and not list((slot/'run/checkpoints').glob('*.pt')),
                'Unexpected trained checkpoint; preserve')
        repair_root.mkdir(parents=True,exist_ok=True)
        before=safe_path(repair_root,'before');before.mkdir(exist_ok=True)
        journal=repair_root/'journal.json'
        if journal.exists():
            require(read(journal)['new_source']==source,'Interrupted repair belongs to another source')
        # Verify the full old return inventory before any installation mutation.
        for name,pin in contract['preserved_files'].items():
            old=safe_path(slot,name);saved=safe_path(before,name)
            if not matches(saved,pin):
                original_bytes(old,pin,append_allowed=name=='logs/supervisor.log')
        for name,pin in contract['preserved_files'].items():
            saved=safe_path(before,name)
            if not saved.exists():
                saved.parent.mkdir(parents=True,exist_ok=True)
                raw=original_bytes(safe_path(slot,name),pin,append_allowed=name=='logs/supervisor.log')
                with saved.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
            require(matches(saved,pin),'Saved evidence differs: '+name)
        # The supervisor appends its final JSON only after transfer.py finishes,
        # so its archived prefix was empty. Preserve the completed log as well.
        tail=safe_path(slot,'logs/supervisor.log')
        complete=safe_path(repair_root,'completed-supervisor.log')
        if not complete.exists():
            with complete.open('xb') as f:f.write(tail.read_bytes());f.flush();os.fsync(f.fileno())
        require(complete.read_bytes()==tail.read_bytes(),'Completed supervisor log changed')
        old_return=slot/'return.json'
        if old_return.exists():
            require(not old_return.is_symlink(),'Unexpected old return receipt symlink')
            value=read(old_return)
            require(value['commit']==contract['returned_commit'],'Old return identity differs')
            saved=before/'return.json'
            if not saved.exists():saved.write_bytes(old_return.read_bytes())
            require(saved.read_bytes()==old_return.read_bytes(),'Old return receipt changed')
        atomic(journal,dict(status='EVIDENCE_SAVED',new_source=source,created_utc=time.time()))
        stage=safe_path(repair_root,'new-package');stage.mkdir(exist_ok=True)
        for name,value in files.items():
            require(Path(name).name==name,'Invalid staged source name')
            p=safe_path(stage,name)
            if p.exists():require(p.read_bytes()==value,'Staged repair file differs: '+name)
            else:
                with p.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
        validate_package(stage,expected)
        package=safe_path(slot,'package');retired=safe_path(repair_root,'retired-package')
        if package.exists():
            current=read(package/'manifest.json')['identity']
            if current==contract['original_package_identity']:
                require(not retired.exists(),'Ambiguous original package; preserve')
                validate_package(package,current);os.rename(package,retired)
            else:
                require(current==expected,'Another package is installed; preserve');validate_package(package,expected)
        else:require(retired.is_dir(),'Original package missing without repair backup')
        if not package.exists():os.rename(stage,package)
        origin=slot/'source-commit.json'
        original=dict(commit=contract['original_commit'],package_identity=contract['original_package_identity'],
                      source_path=contract['original_source_path'])
        require(read(origin) in (original,source),'Source commit changed during repair')
        if read(origin)!=source:atomic(origin,source)
        for name in ('budget.json','started.json'):
            require(matches(slot/name,contract['preserved_files'][name]),'Repair modified immutable campaign accounting')
        receipt=dict(schema='final6h-repair-applied-v2',repair_id=REPAIR_ID,status='REPAIR_APPLIED',
            package_identity=expected,original_package_identity=contract['original_package_identity'],new_source=source,
            original_started_utc=contract['original_started_utc'],original_deadline_utc=contract['original_deadline_utc'],
            old_return_commit=contract['returned_commit'],preserved_files=contract['preserved_files'],
            evidence_directory=str(before),applied_utc=time.time(),new_PPO_updates=0,
            completed_supervisor_log=dict(path=str(complete),bytes=complete.stat().st_size,
                                          sha256=sha_bytes(complete.read_bytes())),
            budget_renewed=False,deadline_extended=False,automatic_training=False,
            charged_unsaved_updates=0,parent_checkpoint_reload_required=True,remaining_max_attempts=10005,
            recovery_smoke_updates=5,remaining_formal_updates=10000,
            original_train_stop_utc=contract['original_train_stop_utc'])
        atomic(receipt_path,receipt);atomic(journal,dict(status='REPAIR_APPLIED',new_source=source))
        return receipt
