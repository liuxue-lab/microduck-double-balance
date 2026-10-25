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
CODE='experiments/stage12/push10-final6h-v1/package'
BOOTSTRAP_REVISION='committed-final6h-v1'
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


def main():
    raw=sys.stdin.read(2*MAX_SOURCE_BYTES+1)
    if len(raw)>2*MAX_SOURCE_BYTES:raise RuntimeError('Request size exceeds limit')
    request=json.loads(raw)
    commit=request['commit'];expected=request['package_identity'];operation=request['operation']
    if not re.fullmatch('[0-9a-f]{40}',commit) or not re.fullmatch('[0-9a-f]{64}',expected):
        raise RuntimeError('Invalid code identity')
    if operation not in ('start','status','collect','stop','return','pack'):raise RuntimeError('Invalid final campaign operation')
    repo=BASE/'workspace';slot=BASE/'artifacts/double-balance-stage12/native-push10-final6h-v1'
    package=slot/'package';python=repo/'.venv/bin/python'
    if slot.is_symlink() or package.is_symlink():raise RuntimeError('Unexpected campaign symlink')
    if not package.is_dir() and operation in ('status','collect','stop'):
        print('MD12_FINAL6H_STATE='+json.dumps(dict(status='NOT_INSTALLED',slot=str(slot),added_PPO_updates=0)),flush=True)
        return
    if operation=='start':
        if not python.is_file():raise RuntimeError('Existing cloud project environment missing')
        files=decode_source(request)
        before=config_fingerprints(repo)
        slot.mkdir(parents=True,exist_ok=True);package.mkdir(exist_ok=True)
        for name,data in files.items():
            target=package/name
            if target.is_symlink():raise RuntimeError('Symlink installed source: '+name)
            if target.exists():
                if target.read_bytes()!=data:raise RuntimeError('Installed final campaign source differs; never overwrite: '+name)
            else:
                with target.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
        origin=slot/'source-commit.json'
        wanted=dict(commit=commit,package_identity=expected,source_path=CODE)
        if origin.is_symlink():raise RuntimeError('Unexpected source provenance symlink')
        if origin.exists():
            if json.loads(origin.read_bytes())!=wanted:raise RuntimeError('Installed source provenance differs')
        else:
            with origin.open('x') as f:json.dump(wanted,f,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
        if config_fingerprints(repo)!=before:raise RuntimeError('Cloud Git configuration changed; preserve')
        print('FINAL6H_SOURCE_INSTALLED='+BOOTSTRAP_REVISION+'; GitHub fetches=0',flush=True)
    if not package.is_dir():raise RuntimeError('Final campaign package missing; start the approved task first')
    manifest_path=package/'manifest.json'
    if manifest_path.is_symlink():raise RuntimeError('Unexpected manifest symlink')
    raw=manifest_path.read_bytes();manifest=json.loads(raw);files={'manifest.json':raw}
    if manifest['identity']!=expected:raise RuntimeError('Installed package identity differs')
    for name in manifest['files']:
        if not re.fullmatch(r'[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+',name) or name=='manifest.json':
            raise RuntimeError('Invalid package path')
        target=package/name
        if target.is_symlink():raise RuntimeError('Installed package changed: '+name)
        files[name]=target.read_bytes()
    verify_files(files,expected)
    origin=slot/'source-commit.json'
    if origin.is_symlink() or json.loads(origin.read_bytes())!=dict(commit=commit,package_identity=expected,source_path=CODE):
        raise RuntimeError('Installed source commit differs')
    script='transfer.py' if operation in ('return','pack') else 'agent.py'
    args=['--pack-only'] if operation=='pack' else [] if operation=='return' else [operation]
    os.execv(str(python),[str(python),'-B','-u',str(package/script),*args])


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print('MD12_FINAL6H_ERROR='+json.dumps({'error':repr(exc),'bootstrap_failed':True}),flush=True)
        raise SystemExit(2)
