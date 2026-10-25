"""Return necessary NEW artifacts via an isolated Git SSH transfer branch."""
import argparse
import datetime
import fcntl
import shlex
import subprocess
import time
import zipfile
from common import *
from agent import snapshot

MAX_ARCHIVE_BYTES=49*1024*1024
MAX_GROUP_BYTES=45*1024*1024


def make_inventory():
    state=snapshot()
    require('exit' in state and not any(state['processes_alive'].values()),
            'Cloud computation is not yet quiescent; use --watch or --status')
    checkpoints={input_checkpoint()['path']}
    for p in (SLOT/'evaluations').glob('*/candidate.json'):
        checkpoints.add(read(p)['path'])
    smoke=SLOT/'run/checkpoints/headroom_000005.pt'
    if smoke.exists():checkpoints.add(str(smoke))
    needed_hashes={sha(p) for p in checkpoints if Path(p).is_file() and Path(p).is_relative_to(SLOT)}
    included={};omitted={};seen=set()
    for p in sorted(SLOT.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(SLOT)
        if p.is_symlink():raise RuntimeError('Unexpected artifact symlink: '+str(rel))
        if rel.parts[0] in ('cache','transfer') or '__pycache__' in rel.parts or p.name.endswith('.lock'):
            continue
        if p.name=='return.json':continue
        pin=dict(bytes=p.stat().st_size,sha256=sha(p))
        reason=None
        if p.suffix=='.pt':
            if pin['sha256'] not in needed_hashes:reason='Intermediate recovery checkpoint; retained on cloud with hash'
            elif pin['sha256'] in seen:reason='Duplicate model bytes; returned once'
            else:seen.add(pin['sha256'])
        elif 'tensorboard' in rel.parts:reason='TensorBoard event data; JSON training metrics and logs returned'
        if reason:omitted[rel.as_posix()]=dict(**pin,reason=reason)
        else:included[rel.as_posix()]=pin
    require(included,'No new artifacts to return')
    return dict(included=included,omitted=omitted,
        warm_start_checkpoint_sha256=MODEL_SHA,
        original_selected_model_already_local_sha256=SELECTED_BASE_SHA,
        original_entire_cloud_storage_backed_up=False,
        scope='New campaign: candidates/latest/smoke, optimizer checkpoints, all JSON ledgers/reports/logs/traces/code; intermediate model hashes retained',
        cloud_computation_status=state['status'],checked_baseline_goal_passed=state['exit']['checked_baseline_goal_passed'])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--pack-only',action='store_true')
    args=parser.parse_args()
    session_number=read(SLOT/'session.json')['number']
    directory=SLOT/'transfer'/('session-%03d'%session_number);directory.mkdir(parents=True,exist_ok=True)
    with (directory/'return.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        package_id=verify_package()
        index_path=directory/'return-index.json'
        if index_path.is_file():
            index=read(index_path)
            for name,pin in index['archives'].items():
                require(sha(directory/name)==pin['sha256'],'Existing return archive changed')
        else:
            inventory=make_inventory()
            inventory_path=directory/'inventory.json'
            if inventory_path.exists():
                require(read(inventory_path)==inventory,'Original packing inventory changed; preserve')
            else:write(inventory_path,inventory,exclusive=True)
            groups=[];group=[];size=0
            for name,pin in inventory['included'].items():
                require(pin['bytes']<MAX_GROUP_BYTES,'Oversized single file; preserve for explicit transfer review: '+name)
                if group and size+pin['bytes']>MAX_GROUP_BYTES:
                    groups.append(group);group=[];size=0
                group.append(name);size+=pin['bytes']
            if group:groups.append(group)
            archives={}
            for i,names in enumerate(groups,1):
                path=directory/('part-%04d.zip'%i)
                reusable=False
                if path.exists():
                    try:
                        with zipfile.ZipFile(path) as z:
                            reusable=z.namelist()==names and all(
                                hashlib.sha256(z.read(name)).hexdigest()==inventory['included'][name]['sha256']
                                for name in names)
                    except (OSError,zipfile.BadZipFile,KeyError):pass
                    if not reusable:path.rename(path.with_name(path.name+'.incomplete-'+str(time.time_ns())))
                if not reusable:
                    with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                        for name in names:
                            p=SLOT/name;require(sha(p)==inventory['included'][name]['sha256'],'Artifact changed while packaging')
                            z.write(p,name)
                require(path.stat().st_size<=MAX_ARCHIVE_BYTES,'Archive exceeds 49 MiB cap')
                archives[path.name]=dict(bytes=path.stat().st_size,sha256=sha(path),members=names)
            started=read(SLOT/'started.json')
            stamp=datetime.datetime.fromtimestamp(started['started_utc'],datetime.timezone.utc).strftime('%Y%m%dt%H%M%Sz').lower()
            branch='stage12-headroom-v3-return-'+stamp+'-'+package_id[:8]+'-s%03d'%session_number
            index=dict(schema='stage12-headroom-return-v3',branch=branch,archives=archives,
                inventory=inventory,package_identity=package_id,session=session_number,
                source_commit=read(SLOT/'source-commit.json')['commit'],created_utc=time.time(),
                power_state='UNCONFIRMED')
            write(index_path,index,exclusive=True)
        if args.pack_only:
            print('MD12_HEADROOM_PACK='+json.dumps(dict(status='CLOUD_PACK_COMPLETE',directory=str(directory),
                index=str(index_path),index_sha256=sha(index_path),
                archives=[dict(path=str(directory/n),bytes=p['bytes'],sha256=p['sha256']) for n,p in index['archives'].items()],
                package_identity=package_id,new_PPO_updates=0),ensure_ascii=False),flush=True)
            return
        repo=directory/'git';branch=index['branch']
        def git(*args,**kwargs):
            return subprocess.check_output(['git','-C',str(repo),*args],**kwargs)
        if not repo.exists():
            repo.mkdir();subprocess.run(['git','init',str(repo)],check=True,stdout=subprocess.DEVNULL)
            git('checkout','-b',branch)
            git('remote','add','origin',ORIGIN_SSH)
            # Reuse a configured successful Git SSH identity, without reading private key contents.
            command=subprocess.run(['git','-C',str(BASE/'workspace'),'config','--get','core.sshCommand'],
                                   capture_output=True,text=True).stdout.strip()
            known=Path('/root/.ssh/id_ed25519_microduck_stage09')
            if command:git('config','core.sshCommand',command)
            elif known.is_file():git('config','core.sshCommand','ssh -i '+shlex.quote(str(known))+' -o IdentitiesOnly=yes')
        require(git('remote','get-url','origin').decode().strip()==ORIGIN_SSH,'Transfer origin must remain Git SSH')
        for name in ['return-index.json',*index['archives']]:
            target=repo/name;data=(directory/name).read_bytes()
            if target.exists():require(target.read_bytes()==data,'Transfer Git worktree differs')
            else:target.write_bytes(data)
        git('add','--','return-index.json',*index['archives'])
        dirty=subprocess.run(['git','-C',str(repo),'diff','--cached','--quiet']).returncode
        if dirty:
            git('-c','user.name=Codex','-c','user.email=codex@openai.com','-c','commit.gpgsign=false',
                'commit','-m','Return bounded Stage12 headroom candidate evidence')
        commit=git('rev-parse','HEAD').decode().strip()
        print('Stage12HeadroomReturn=GitHub SSH artifact upload; this does not mark task acceptance',flush=True)
        subprocess.run(['git','-C',str(repo),'push','origin','HEAD:refs/heads/'+branch],check=True)
        result=dict(status='CLOUD_ARTIFACT_PUSH_COMPLETE',branch=branch,commit=commit,
            package_identity=index['package_identity'],index_sha256=sha(index_path),
            archive_count=len(index['archives']),total_archive_bytes=sum(v['bytes'] for v in index['archives'].values()),
            local_verified=False,power_state='UNCONFIRMED')
        write(SLOT/'return.json',result)
        print('MD12_HEADROOM_RETURN='+json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        print('MD12_HEADROOM_ERROR='+json.dumps(dict(error=repr(exc),transfer_only=True,new_PPO_updates=0)),flush=True)
        raise SystemExit(2)
