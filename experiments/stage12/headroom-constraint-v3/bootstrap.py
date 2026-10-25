"""Install the pinned committed package delivered over the existing SSH channel."""
import base64
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import zlib

BASE=Path('/root/autodl-tmp/microduck-double-balance')
CODE='experiments/stage12/headroom-constraint-v3/package'
BOOTSTRAP_REVISION='headroom-constraint-v3'
MAX_SOURCE_BYTES=4*1024*1024


def git_bytes(repo,*args):
    # Local metadata only: no GitHub fetch or dependency installation here.
    child=subprocess.Popen(['git','-C',str(repo),*args],stdin=subprocess.DEVNULL,
                           stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    try:
        out,err=child.communicate(timeout=15)
    except BaseException:
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
        child.communicate()
        raise
    if child.returncode:raise RuntimeError('Cannot read cloud Git metadata: '+err.decode(errors='replace')[-1000:])
    return out


def config_fingerprints(repo):
    result={}
    for name in ('config','config.worktree'):
        path=Path(os.fsdecode(git_bytes(repo,'rev-parse','--git-path',name)).strip())
        if not path.is_absolute():path=repo/path
        result[name]=hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return result


def verify_files(files,expected):
    raw=files['manifest.json'];manifest=json.loads(raw)
    calculated=hashlib.sha256(json.dumps(manifest['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if calculated!=expected or manifest['identity']!=expected:raise RuntimeError('Committed package identity differs')
    if set(files)!=set(manifest['files'])|{'manifest.json'}:raise RuntimeError('Unexpected or missing source files')
    for name,pin in manifest['files'].items():
        if not re.fullmatch(r'[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+',name) or name=='manifest.json':
            raise RuntimeError('Invalid package path')
        value=files[name]
        if len(value)!=pin['bytes'] or hashlib.sha256(value).hexdigest()!=pin['sha256']:
            raise RuntimeError('Committed source hash differs: '+name)
    return files


def decode_source(request):
    source=request.get('source')
    if not isinstance(source,dict) or source.get('encoding')!='zlib-base85-json-v1':
        raise RuntimeError('Pinned source payload missing; update the local launcher')
    if not isinstance(source.get('raw_bytes'),int) or not 0<source['raw_bytes']<=MAX_SOURCE_BYTES:
        raise RuntimeError('Source payload size exceeds limit')
    if len(source['data'])>2*MAX_SOURCE_BYTES:raise RuntimeError('Encoded source exceeds limit')
    compressed=base64.b85decode(source['data']);decoder=zlib.decompressobj()
    raw=decoder.decompress(compressed,MAX_SOURCE_BYTES+1)
    if len(raw)>MAX_SOURCE_BYTES or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise RuntimeError('Invalid or oversized compressed source')
    if len(raw)!=source['raw_bytes'] or hashlib.sha256(raw).hexdigest()!=source['sha256']:
        raise RuntimeError('Source transport hash differs')
    document=json.loads(raw)
    for name in ('commit','package_identity'):
        if document[name]!=request[name]:raise RuntimeError('Source provenance differs: '+name)
    if document['source_path']!=CODE:raise RuntimeError('Source provenance path differs')
    files={name:base64.b64decode(value,validate=True) for name,value in document['files'].items()}
    return verify_files(files,request['package_identity'])


def safe_directory(path):
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise RuntimeError('Unexpected directory path; preserve: '+str(path))
    path.mkdir(parents=True,exist_ok=True)


def install_source(repo,slot,request):
    before=config_fingerprints(repo)
    safe_directory(slot);audit=slot/'bootstrap-source';safe_directory(audit)
    receipt=audit/(str(time.time_ns())+'.json')
    source=request.get('source') or {}
    report=dict(schema='headroom-ssh-pinned-source-v1',bootstrap_revision=BOOTSTRAP_REVISION,
                commit=request['commit'],package_identity=request['package_identity'],source_path=CODE,
                transport='existing authenticated SSH stdin',payload_sha256=source.get('sha256'),
                source_bytes=source.get('raw_bytes'),config_before_sha256=before,
                requested_utc=time.time(),status='VERIFYING_SOURCE',new_PPO_updates=0,github_fetches=0)
    def save():
        temporary=receipt.with_suffix('.tmp')
        with temporary.open('x') as f:
            json.dump(report,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(temporary,receipt)
    save()
    try:
        print('HEADROOM_SOURCE_TRANSFER='+BOOTSTRAP_REVISION+'; verifying pinned package; GitHub fetches=0',flush=True)
        files=decode_source(request);package=slot/'package'
        if package.is_symlink() or (package.exists() and not package.is_dir()):
            raise RuntimeError('Unexpected package path; preserve')
        origin=slot/'source-commit.json'
        content={'commit':request['commit'],'package_identity':request['package_identity'],'source_path':CODE}
        if origin.is_symlink() or (origin.exists() and (not origin.is_file() or json.loads(origin.read_bytes())!=content)):
            raise RuntimeError('Campaign source commit differs; preserve')
        # Check every existing target before creating any package file.
        for name,value in files.items():
            target=package/name
            if target.is_symlink() or (target.exists() and (not target.is_file() or target.read_bytes()!=value)):
                raise RuntimeError('Installed campaign differs; preserve: '+name)
        safe_directory(package)
        for name,value in files.items():
            target=package/name
            if not target.exists():
                with target.open('xb') as f:f.write(value);f.flush();os.fsync(f.fileno())
        if not origin.exists():
            with origin.open('x') as f:json.dump(content,f);f.write('\n');f.flush();os.fsync(f.fileno())
        report.update(status='PINNED_SOURCE_INSTALLED',verified_files=len(files))
    except BaseException as exc:
        report.update(status='SOURCE_FAILED_PRESERVE',error=repr(exc));raise
    finally:
        after=config_fingerprints(repo)
        report.update(config_after_sha256=after,configuration_unchanged=before==after,finished_utc=time.time())
        if before!=after:report['status']='CONFIG_CHANGED_PRESERVE'
        save()
        if before!=after:raise RuntimeError('Repository configuration changed; preserve and stop')
    print('HEADROOM_SOURCE_VERIFIED='+request['package_identity']+'; separate bounded continuation, parent ledger retained',flush=True)


def main():
    raw=sys.stdin.read(2*MAX_SOURCE_BYTES+1)
    if len(raw)>2*MAX_SOURCE_BYTES:raise RuntimeError('Request size exceeds limit')
    request=json.loads(raw)
    commit=request['commit'];expected=request['package_identity'];operation=request['operation']
    if not re.fullmatch('[0-9a-f]{40}',commit) or not re.fullmatch('[0-9a-f]{64}',expected):
        raise RuntimeError('Invalid code identity')
    if operation not in ('start','resume','status','collect','stop','return','pack'):raise RuntimeError('Invalid operation')
    repo=BASE/'workspace';slot=BASE/'artifacts/double-balance-stage12/native-headroom-constraint-v3'
    package=slot/'package';python=repo/'.venv/bin/python'
    if slot.is_symlink() or package.is_symlink():raise RuntimeError('Unexpected campaign symlink')
    if operation in ('start','resume'):
        if not python.is_file():raise RuntimeError('Existing cloud project environment missing')
        install_source(repo,slot,request)
    if not package.is_dir():
        if operation in ('status','collect'):
            print('MD12_HEADROOM_STATE='+json.dumps({'status':'NOT_INSTALLED','added_PPO_updates':0}));return
        raise RuntimeError('Campaign package is not installed')
    manifest_path=package/'manifest.json'
    if manifest_path.is_symlink():raise RuntimeError('Unexpected manifest symlink')
    raw=manifest_path.read_bytes();manifest=json.loads(raw);files={'manifest.json':raw}
    for name in manifest['files']:
        if not re.fullmatch(r'[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+',name) or name=='manifest.json':
            raise RuntimeError('Invalid package path')
        target=package/name
        if target.is_symlink():raise RuntimeError('Installed package changed: '+name)
        files[name]=target.read_bytes()
    verify_files(files,expected)
    origin=slot/'source-commit.json'
    if origin.is_symlink() or json.loads(origin.read_bytes())!={'commit':commit,'package_identity':expected,'source_path':CODE}:
        raise RuntimeError('Installed source commit differs')
    script='transfer.py' if operation in ('return','pack') else 'agent.py'
    args=(['--pack-only'] if operation=='pack' else []) if operation in ('return','pack') else [operation]
    os.execv(str(python),[str(python),'-B','-u',str(package/script),*args])


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print('MD12_HEADROOM_ERROR='+json.dumps({'error':repr(exc),'bootstrap_failed':True}),flush=True)
        raise SystemExit(2)
