"""Install only the user-pushed pinned code commit; never install dependencies."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


def main():
    request=json.loads(sys.stdin.read())
    commit=request['commit'];expected=request['package_identity'];operation=request['operation']
    if not re.fullmatch('[0-9a-f]{40}',commit) or not re.fullmatch('[0-9a-f]{64}',expected):
        raise RuntimeError('Invalid code identity')
    if operation not in ('start','resume','status','collect','stop','return'):
        raise RuntimeError('Invalid operation')
    base=Path('/root/autodl-tmp/microduck-double-balance')
    repo=base/'workspace';slot=base/'artifacts/double-balance-stage12/native-headroom-refine-v1'
    package=slot/'package';python=repo/'.venv/bin/python'
    code='experiments/stage12/headroom-refine-v1/package'
    def git(*args):
        return subprocess.check_output(['git','-C',str(repo),*args])
    if operation in ('start','resume'):
        if not python.is_file():raise RuntimeError('Existing cloud project environment missing')
        if git('remote','get-url','origin').decode().strip()!='git@github.com:liuxue-lab/microduck-double-balance.git':
            raise RuntimeError('Cloud origin must be the existing Git SSH remote; preserve current config')
        # The user pushes from the canonical local repo; fetch is read-only to the cloud worktree.
        subprocess.run(['git','-C',str(repo),'fetch','--no-tags','origin','double-balance'],check=True)
        subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',commit,'FETCH_HEAD'],check=True)
        raw=git('show',commit+':'+code+'/manifest.json');manifest=json.loads(raw)
        calculated=hashlib.sha256(json.dumps(manifest['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
        if calculated!=expected or manifest['identity']!=expected:raise RuntimeError('Committed package identity differs')
        slot.mkdir(parents=True,exist_ok=True)
        if slot.is_symlink():raise RuntimeError('Unexpected campaign symlink')
        package.mkdir(exist_ok=True)
        if package.is_symlink():raise RuntimeError('Unexpected package symlink')
        files={}
        for name,pin in manifest['files'].items():
            if Path(name).name!=name:raise RuntimeError('Invalid package path')
            value=git('show',commit+':'+code+'/'+name)
            if len(value)!=pin['bytes'] or hashlib.sha256(value).hexdigest()!=pin['sha256']:
                raise RuntimeError('Committed source hash differs: '+name)
            files[name]=value
        files['manifest.json']=raw
        for name,value in files.items():
            target=package/name
            if target.exists():
                if target.is_symlink() or target.read_bytes()!=value:
                    raise RuntimeError('Installed campaign differs; preserve: '+name)
            else:
                with target.open('xb') as f:f.write(value)
        origin=slot/'source-commit.json'
        content={'commit':commit,'package_identity':expected,'source_path':code}
        if origin.exists():
            if json.loads(origin.read_text())!=content:raise RuntimeError('Campaign source commit differs')
        else:origin.write_text(json.dumps(content)+'\n')
    if not package.is_dir():
        if operation in ('status','collect'):
            print('MD12_HEADROOM_STATE='+json.dumps({'status':'NOT_INSTALLED','added_PPO_updates':0}));return
        raise RuntimeError('Campaign package is not installed')
    manifest=json.loads((package/'manifest.json').read_bytes())
    if manifest['identity']!=expected:raise RuntimeError('Installed package identity differs')
    for name,pin in manifest['files'].items():
        target=package/name
        if Path(name).name!=name or target.is_symlink() or hashlib.sha256(target.read_bytes()).hexdigest()!=pin['sha256']:
            raise RuntimeError('Installed package changed: '+name)
    script='transfer.py' if operation=='return' else 'agent.py'
    args=[] if operation=='return' else [operation]
    os.execv(str(python),[str(python),'-B','-u',str(package/script),*args])


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print('MD12_HEADROOM_ERROR='+json.dumps({'error':repr(exc),'bootstrap_failed':True}),flush=True)
        raise SystemExit(2)
