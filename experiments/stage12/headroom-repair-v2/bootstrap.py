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
CODE='experiments/stage12/headroom-repair-v2/package'
OLD_CODE='experiments/stage12/headroom-refine-v1/package'
OLD_COMMIT='cb6d279a875de7d9e0e88720ba591e0e1ff098d1'
OLD_ID='c0c233531214edfa72602540f979f9ab3c58b5f7008c52f07fefe0a744d27ec6'
BOOTSTRAP_REVISION='audited-import-repair-v2'
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
    if operation not in ('repair','resume','status','collect','stop','return'):raise RuntimeError('Invalid repair operation')
    repo=BASE/'workspace';slot=BASE/'artifacts/double-balance-stage12/native-headroom-refine-v1'
    package=slot/'package';python=repo/'.venv/bin/python'
    if slot.is_symlink() or package.is_symlink():raise RuntimeError('Unexpected campaign symlink')
    if not slot.is_dir():raise RuntimeError('Original campaign missing; repair cannot create a new campaign')
    if operation=='repair':
        if not python.is_file():raise RuntimeError('Existing cloud project environment missing')
        print('HEADROOM_SOURCE_TRANSFER='+BOOTSTRAP_REVISION+'; GitHub fetches=0',flush=True)
        files=decode_source(request)
        contract=json.loads(files['repair-contract.json'])
        if contract['original_commit']!=OLD_COMMIT or contract['original_package_identity']!=OLD_ID:
            raise RuntimeError('Repair lineage differs')
        before=config_fingerprints(repo)
        module={'__name__':'headroom_verified_repair','__file__':str(package/'repair.py')}
        exec(compile(files['repair.py'],'<verified headroom repair>','exec'),module)
        receipt=module['apply_repair'](repo,slot,files,request)
        if config_fingerprints(repo)!=before:raise RuntimeError('Cloud Git configuration changed; preserve')
        print('HEADROOM_REPAIR_APPLIED='+receipt['repair_id']+'; budget/deadline unchanged',flush=True)
    if not package.is_dir():raise RuntimeError('Original or repaired package missing; use the explicit --repair entry')
    manifest_path=package/'manifest.json'
    if manifest_path.is_symlink():raise RuntimeError('Unexpected manifest symlink')
    raw=manifest_path.read_bytes();manifest=json.loads(raw);active=manifest['identity'];files={'manifest.json':raw}
    if active not in (OLD_ID,expected):raise RuntimeError('Installed package identity differs')
    if operation in ('repair','resume') and active!=expected:raise RuntimeError('Run --repair before resume')
    for name in manifest['files']:
        if not re.fullmatch(r'[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+',name) or name=='manifest.json':
            raise RuntimeError('Invalid package path')
        target=package/name
        if target.is_symlink():raise RuntimeError('Installed package changed: '+name)
        files[name]=target.read_bytes()
    verify_files(files,active)
    origin=slot/'source-commit.json'
    wanted={'commit':OLD_COMMIT if active==OLD_ID else commit,'package_identity':active,
            'source_path':OLD_CODE if active==OLD_ID else CODE}
    if origin.is_symlink() or json.loads(origin.read_bytes())!=wanted:raise RuntimeError('Installed source commit differs')
    script='transfer.py' if operation=='return' else 'agent.py'
    args=[] if operation=='return' else ['resume-repaired' if operation=='repair' else operation]
    os.execv(str(python),[str(python),'-B','-u',str(package/script),*args])


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print('MD12_HEADROOM_ERROR='+json.dumps({'error':repr(exc),'bootstrap_failed':True}),flush=True)
        raise SystemExit(2)
