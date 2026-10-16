"""Small SSH bootstrap: fetch committed package through Git SSH, then detach."""
from pathlib import Path
import hashlib
import json
import os
import shlex
import subprocess
import sys

CONFIG = __CONFIG_LITERAL__


def need(ok,msg):
    if not ok:raise RuntimeError(msg)


def run(args,env=None):
    p=subprocess.run([str(x) for x in args],capture_output=True,text=True,env=env,timeout=240)
    need(p.returncode==0,p.stderr[-4000:]);return p.stdout.strip()


def same(path,raw):
    if path.exists():need(path.read_bytes()==raw,'Existing bootstrap file differs: '+str(path))
    else:
        with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())


def live(pid,token):
    p=Path('/proc')/str(pid)/'cmdline'
    return p.exists() and token.encode() in p.read_bytes()


def main():
    config=dict(CONFIG); base=Path(config['cloud_base']);base.mkdir(parents=True,exist_ok=True)
    need(not any(x.is_symlink() for x in (base,*base.parents)),'Linked cloud base refused')
    known=base/'github-known-hosts'
    same(known,config.pop('host_key').encode())
    actual=run(['ssh-keygen','-lf',known,'-E','sha256'])
    need(config.pop('host_fingerprint') in actual,'GitHub pinned key fingerprint differs')
    opts=['ssh','-F','/dev/null','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
          '-o','UserKnownHostsFile='+str(known),'-o','GlobalKnownHostsFile=/dev/null',
          '-o','HostKeyAlgorithms=ssh-ed25519','-o','ConnectTimeout=15']
    key=Path('/root/.ssh/id_ed25519_microduck_stage09')
    if key.is_file():opts+=['-o','IdentitiesOnly=yes','-i',str(key)]
    ssh=shlex.join(opts);config['github_ssh_command']=ssh
    env=dict(os.environ,GIT_SSH_COMMAND=ssh,GIT_TERMINAL_PROMPT='0')
    source=base/'input';branch=config['branch'];origin=config['origin']
    if not (source/'.git').exists():
        source.mkdir(exist_ok=True);run(['git','init',source],env)
        run(['git','-C',source,'remote','add','origin',origin],env)
    need(run(['git','-C',source,'remote','get-url','origin'],env)==origin,'Origin differs')
    # Reading HEAD does not start the campaign. A failed fetch can be continued.
    head=subprocess.run(['git','-C',str(source),'rev-parse','HEAD'],capture_output=True,text=True)
    if head.returncode:
        run(['git','-C',source,'fetch','--depth=1','origin','refs/heads/'+branch],env)
        need(run(['git','-C',source,'rev-parse','FETCH_HEAD'],env)==config['commit'],'Input branch changed after local commit')
        run(['git','-C',source,'-c','core.hooksPath=/dev/null','checkout','--detach',config['commit']],env)
    need(run(['git','-C',source,'rev-parse','HEAD'],env)==config['commit'],'Pinned input commit differs')
    package=source/'PACKAGE.json';need(hashlib.sha256(package.read_bytes()).hexdigest()==config['package_sha256'],'Manifest hash differs')
    for n,r in json.loads(package.read_bytes()).items():
        p=Path(n);need(not p.is_absolute() and '..' not in p.parts and '\\' not in n,'Unsafe package path')
        f=source/p;need(f.is_file() and not f.is_symlink(),'Missing/linked package member')
        need(f.stat().st_size==r['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==r['sha256'],'Package bytes differ '+n)
    work=Path(config['work']);work.mkdir(parents=True,exist_ok=True)
    path=base/'config.json';same(path,(json.dumps(config,sort_keys=True,indent=2)+'\n').encode())
    receipt=base/'launch.json'
    if receipt.exists():
        old=json.loads(receipt.read_bytes())
        print(json.dumps(dict(status='EXISTING_LAUNCH_NO_RESTART',alive=live(old['pid'],str(source/'cloud_run.py')),**old),indent=2))
        return
    with (work/'runner.log').open('xb') as log:
        p=subprocess.Popen([sys.executable,'-I','-B',str(source/'cloud_run.py'),'--config',str(path),'run'],
                           cwd=work,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    row=dict(pid=p.pid,config=str(path),source=str(source),work=str(work),commit=config['commit'])
    same(receipt,(json.dumps(row,sort_keys=True,indent=2)+'\n').encode())
    print(json.dumps(dict(status='DETACHED_CLOUD_CAMPAIGN_STARTED',**row),indent=2))


if __name__=='__main__':main()
