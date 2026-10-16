#!/usr/bin/env python3
"""User-laptop control: prepare -> manual push -> launch/status -> manual return push -> receive."""
from pathlib import Path
import argparse
import base64
import datetime
import hashlib
import json
import os
import secrets
import shlex
import shutil
import subprocess
import sys
import zipfile
import gzip

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT))
from common import *

REPO=Path('/home/lx/microduck-double-balance/workspace')
BASE=Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage12/cloud-execution-v1')
HOST='root@connect.bjb2.seetacloud.com';PORT='46171'


def ssh_args():
    return ['ssh','-T','-o','StrictHostKeyChecking=yes','-o','ConnectTimeout=15',
            '-o','ServerAliveInterval=20','-o','ServerAliveCountMax=3','-p',PORT,HOST]


def local_identity():
    return dict(head=command(['git','-C',REPO,'rev-parse','HEAD']),
                status=command(['git','-C',REPO,'status','--porcelain=v1']),
                mdp=info(REPO/'src/mjlab_microduck/tasks/mdp.py'),
                pyproject=info(REPO/'pyproject.toml'),lock=info(REPO/'uv.lock'))


def prepare():
    package=verify(ROOT);BASE.mkdir(parents=True,exist_ok=True)
    receipt=BASE/'input-receipt.json'
    if receipt.exists():
        row=read(receipt);require(row['package_sha256']==package,'Another package already owns this campaign path')
        require(command(['git','-C',BASE/'transfer','rev-parse','HEAD'])==row['commit'],'Input transfer HEAD differs')
        verify(BASE/'transfer');print_commands(row);return row
    identity_before=local_identity()
    require(identity_before['head']=='73ee01930a031d76a9d57419afdfd4464a547ab3','Local source HEAD differs; no source mutation performed')
    if (BASE/'started.json').exists():
        started=read(BASE/'started.json');require(started['package_sha256']==package,'Local preparation identity differs')
        run_id=started['run_id'];require(started['local_identity']==identity_before,'Local source changed since preparation started')
    else:
        run_id=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+secrets.token_hex(3)
        write(BASE/'started.json',dict(run_id=run_id,package_sha256=package,local_identity=identity_before,PPO=0,native=0))
    git_identity={k:command(['git','-C',REPO,'config','--get','user.'+k]) for k in ('name','email')}
    branch='stage12-cloud-execution-v1-input-'+run_id.lower();transfer=BASE/'transfer'
    prepare_git(transfer,branch,git_identity)
    for n in [*read(ROOT/'PACKAGE.json'),'PACKAGE.json']:
        same_or_new(transfer/n,(ROOT/n).read_bytes())
    verify(transfer)
    commit=commit_git(transfer,'Stage12 cloud bounded execution candidate; native only; PPO0')
    cloud_base='/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/cloud-execution-v1/'+run_id
    row=dict(run_id=run_id,branch=branch,commit=commit,package_sha256=package,identity=git_identity,
             origin=ORIGIN,cloud_base=cloud_base,work=cloud_base+'/work',runtime='/root/md12-runtime-'+run_id.lower())
    require(identity_before==local_identity(),'Local source changed during preparation')
    write(receipt,row);print_commands(row);return row


def receipt():
    row=read(BASE/'input-receipt.json')
    require(verify(ROOT)==row['package_sha256'],'Downloaded package changed')
    return row


def remote(row,action):
    cmd=['python3','-I','-B',row['cloud_base']+'/input/cloud_run.py','--config',row['cloud_base']+'/config.json',action]
    if action=='resume':
        # Reconnection never resends native steps. The remote controller takes
        # its process lock and verifies no retained worker is still live.
        script=('import subprocess,time\nfrom pathlib import Path\n'
                +'work=Path('+repr(row['work'])+')\n'
                +'log=work/("resume-"+str(time.time_ns())+".log")\n'
                +'with log.open("xb") as f:\n'
                +' p=subprocess.Popen('+repr(cmd)+',stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)\n'
                +'print("Stage12ResumePID="+str(p.pid)+"; log="+str(log))\n')
        return subprocess.run(ssh_args()+['python3 -I -B -'],input=script,text=True).returncode
    return subprocess.run(ssh_args()+[shlex.join(cmd)]).returncode


def print_commands(row):
    print('Stage12InputCommit='+row['commit'])
    print('Stage12InputBranch='+row['branch'])
    print('Stage12UploadCommand=git -C '+str(BASE/'transfer')+' push origin HEAD')
    print('Stage12LaunchCommand=python3 '+shlex.quote(str(ROOT/'local.py'))+' --launch')
    return_cmd='git -C '+shlex.quote(row['work']+'/return-transfer')+' push origin HEAD'
    print('Stage12ReturnUploadCommand='+shlex.join(ssh_args()+[return_cmd]))
    print('Stage12ReceiveCommand=python3 '+shlex.quote(str(ROOT/'local.py'))+' --receive')
    print('Stage12Deadline=2026-10-17T03:53:14Z; all attempts combined; PPO=0; cloud_training=0')


def receive(row):
    cache=BASE/'return-cache';out=BASE/'returned';branch='stage12-cloud-execution-v1-return-'+row['run_id'].lower()
    done=BASE/'return-verified.json'
    if done.exists():print(json.dumps(read(done),ensure_ascii=False,indent=2));return
    if not (cache/'.git').exists():
        cache.mkdir(exist_ok=True);command(['git','init',cache]);command(['git','-C',cache,'remote','add','origin',ORIGIN])
    require(command(['git','-C',cache,'remote','get-url','origin'])==ORIGIN,'Return origin differs')
    pin=BASE/'return-pin.json'
    if not pin.exists():
        command(['git','-C',cache,'fetch','--depth=1','origin','refs/heads/'+branch],timeout=900)
        commit=command(['git','-C',cache,'rev-parse','FETCH_HEAD']);write(pin,dict(branch=branch,commit=commit))
    commit=read(pin)['commit']
    def blob(name):
        safe_name(name)
        p=subprocess.run(['git','-C',str(cache),'show',commit+':'+name],capture_output=True)
        require(p.returncode==0,'Missing Git return member '+name);return p.stdout
    mraw=blob('MANIFEST.json');m=json.loads(mraw)
    tracked=command(['git','-C',cache,'ls-tree','-r','--name-only',commit]).splitlines()
    require(set(tracked)==set(m)|{'MANIFEST.json'},'Return tracked roster differs')
    for name,r in m.items():
        raw=blob(name);require({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}==r,'Return bytes differ '+name)
        same_or_new(out/'transport'/name,raw)
    same_or_new(out/'transport/MANIFEST.json',mraw)
    inp=read(out/'transport/input-identity.json');require(inp['commit']==row['commit'] and inp['package_sha256']==row['package_sha256'],'Return input identity differs')
    storage=read(out/'transport/evidence-index-storage.json');packed=[]
    require(storage['format']=='stage12-evidence-index-gzip-parts-v1' and storage['json_bytes']<=512*1024**2,'Evidence index format/size differs')
    for part in storage['parts']:
        raw=(out/'transport'/safe_name(part['path'])).read_bytes()
        require({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}=={k:part[k] for k in ('bytes','sha256')},'Index part hash differs')
        packed.append(base64.b64decode(raw,validate=True))
    compressed=b''.join(packed)
    require(len(compressed)==storage['gzip_bytes'] and hashlib.sha256(compressed).hexdigest()==storage['gzip_sha256'],'Compressed index differs')
    index_raw=gzip.decompress(compressed)
    require(len(index_raw)==storage['json_bytes'] and hashlib.sha256(index_raw).hexdigest()==storage['json_sha256'],'Full index differs')
    same_or_new(out/'transport/evidence-index.json',index_raw)
    index=json.loads(index_raw);archives=out/'archives';archives.mkdir(parents=True,exist_ok=True)
    seen=set()
    for a in index['archives']:
        target=archives/safe_name(a['filename'])
        if not target.exists():
            tmp=target.with_name(target.name+'.partial-'+secrets.token_hex(4))
            with tmp.open('xb') as f:
                for part in a['parts']:
                    raw=(out/'transport'/part['path']).read_bytes()
                    require({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}=={k:part[k] for k in ('bytes','sha256')},'Raw part hash differs')
                    f.write(base64.b64decode(raw,validate=True))
                f.flush();os.fsync(f.fileno())
            require(info(tmp)=={k:a[k] for k in ('bytes','sha256')},'Reassembled archive differs')
            os.link(tmp,target);tmp.unlink()
        require(info(target)=={k:a[k] for k in ('bytes','sha256')},'Retained archive differs')
        with zipfile.ZipFile(target) as z:
            for name in z.namelist():
                safe_name(name);require(name not in seen and name in index['members'],'Duplicate/unlisted raw member')
                seen.add(name);raw=z.read(name)
                require({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}=={k:index['members'][name][k] for k in ('bytes','sha256')},'Raw evidence differs')
                same_or_new(out/name,raw)
    require(seen==set(index['members']),'Incomplete native evidence segments')
    # Full episodes may contain >100k files. Group once; never scan the full
    # member index again for each original file (quadratic receive time).
    members_by_path={name:[] for name in index['files']}
    for name,r in index['members'].items():
        require(r['path'] in members_by_path,'Segment names an unlisted evidence file')
        members_by_path[r['path']].append((r['offset'],name,r))
    for name,expected in index['files'].items():
        safe_name(name);target=out/'evidence'/name;target.parent.mkdir(parents=True,exist_ok=True)
        if not target.exists():
            parts=sorted(members_by_path[name])
            tmp=target.with_name(target.name+'.partial-'+secrets.token_hex(4));offset=0
            with tmp.open('xb') as stream:
                for pos,n,r in parts:
                    require(pos==offset,'Evidence segment offset gap/overlap')
                    raw=(out/n).read_bytes();stream.write(raw);offset+=len(raw)
                stream.flush();os.fsync(stream.fileno())
            require(info(tmp)==expected,'Reconstructed raw file differs')
            os.link(tmp,target);tmp.unlink()
        require(info(target)==expected,'Retained evidence differs')
    final=dict(status='FULL_NEW_CLOUD_EVIDENCE_RETURN_VERIFIED',branch=branch,commit=commit,
               files=len(index['files']),bytes=index['raw_bytes'],directory=str(out),PPO=0,cloud_training=0,
               cloud_shutdown_confirmation='UNCONFIRMED',historical_Stage09_full_trace_returned=False)
    write(done,final);print(json.dumps(final,ensure_ascii=False,indent=2))
    print('云 GPU 本次必要工作及新证据回传已完成，请在平台手动关机并保留实例和数据。此消息不表示已经关机。')


def main():
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for flag in ('prepare','launch','status','resume','package-only','finalize','commands','receive'):g.add_argument('--'+flag,action='store_true')
    a=p.parse_args()
    if a.prepare:prepare();return 0
    row=receipt()
    if a.launch:
        require(time.time()<CAMPAIGN_DEADLINE-RETURN_RESERVE_SECONDS,'Original 24-hour deadline exhausted; no native launch')
        config=dict(row,host_key=HOST_KEY,host_fingerprint=HOST_FINGERPRINT)
        script=(ROOT/'remote_bootstrap.py').read_text().replace('__CONFIG_LITERAL__',repr(config))
        result=subprocess.run(ssh_args()+['python3 -I -B -'],input=script,text=True)
        print_commands(row);return result.returncode
    if a.commands:print_commands(row);return 0
    if a.receive:receive(row);return 0
    action='status' if a.status else 'resume' if a.resume else 'package-only' if a.package_only else 'finalize'
    return remote(row,action)


if __name__=='__main__':
    try:raise SystemExit(main())
    except (RuntimeError,OSError,ValueError,subprocess.SubprocessError) as exc:
        print('Stage12Cloud=STOP_PRESERVE: '+str(exc),file=sys.stderr);raise SystemExit(1)
