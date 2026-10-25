"""Local headroom campaign controller. Build injects bundle and pinned identities."""
import argparse
import base64
import datetime
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import signal
import stat
import subprocess
import sys
import time
import types
import zipfile
import zlib

PROJECT=Path.home()/'microduck-double-balance/workspace'
LOCAL=PROJECT.parent/'artifacts/double-balance-stage12/native-headroom-refine-v1-client'
REMOTE='git@github.com:liuxue-lab/microduck-double-balance.git'
BASE_COMMIT='cb6d279a875de7d9e0e88720ba591e0e1ff098d1'
ORIGINAL_PACKAGE_ID='c0c233531214edfa72602540f979f9ab3c58b5f7008c52f07fefe0a744d27ec6'
ORIGINAL_STARTED=1792757338.5239706
ORIGINAL_DEADLINE=1792764538.5239706
SCRIPT_NAME='microduck-stage12-headroom-repair-v2-20261023.py'
CLIENT_REVISION='audited-import-repair-v2'
TRANSPORT_REVISION='pinned-package-over-shared-ssh-repair-v2'
# RELEASE_DATA is supplied by the reproducible build, never by a network fetch.


def need(ok,message):
    if not ok:raise RuntimeError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def read(path):return json.loads(Path(path).read_bytes())


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temp.open('xb') as f:
        f.write((json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode());f.flush();os.fsync(f.fileno())
    os.replace(temp,path)


def git(*args):
    return subprocess.check_output(['git','-C',str(PROJECT),*args])


def parse_endpoint(command):
    words=shlex.split(command)
    need(words and words.pop(0)=='ssh','请输入平台当前 SSH 指令，例如 ssh -p 12345 root@host')
    host=None;port=None
    while words:
        item=words.pop(0)
        if item=='-p':
            need(words and port is None,'SSH 端口格式无效');port=int(words.pop(0))
        else:
            need(host is None and not item.startswith('-'),'只接受主机和 -p 端口，不执行额外 SSH 选项')
            host=item
    need(isinstance(port,int) and 1<=port<=65535,'SSH 端口必须在 1–65535 之间')
    need(host is not None and re.fullmatch(r'root@[A-Za-z0-9][A-Za-z0-9.-]*',host),
         '当前项目需要平台提供的 root@主机地址')
    return dict(host=host,port=port)


def local_edits():
    """Snapshot edits relative to HEAD; do not stage, stash, or clean files."""
    diff=('diff','--no-ext-diff','--no-textconv','--no-renames','--binary','--full-index')
    return {'status.bin':git('status','--porcelain=v1','-z','--untracked-files=all'),
            'unstaged.patch':git(*diff), 'staged.patch':git(*diff,'--cached')}


def incoming_conflicts(head,ref):
    """This bundle only adds files. Protect ignored files and symlink parents too."""
    rows=git('diff','--name-status','--no-renames','-z',head,ref,'--').split(b'\0')
    need(rows[-1]==b'' and len(rows[:-1])%2==0,'无法解析待导入路径；保留并停止')
    conflicts=[]
    for change,raw in zip(rows[0:-1:2],rows[1:-1:2]):
        name=os.fsdecode(raw);parts=PurePosixPath(name).parts
        need(change==b'A' and name.startswith('experiments/stage12/headroom-repair-v2/')
             and '..' not in parts,'安装包超出本轮新增目录：'+name)
        path=PROJECT
        for i,part in enumerate(parts):
            path=path/part
            if path.is_symlink() or (path.exists() and (i==len(parts)-1 or not path.is_dir())):
                conflicts.append(dict(incoming=name,existing=str(path.relative_to(PROJECT))))
                break
    return conflicts


def preserve_then_fast_forward(head,ref):
    need(not git('ls-files','--unmerged','-z'),'仓库存在未解决合并；保留并停止')
    for marker in ('MERGE_HEAD','CHERRY_PICK_HEAD','REVERT_HEAD','rebase-merge','rebase-apply'):
        path=Path(os.fsdecode(git('rev-parse','--git-path',marker)).strip())
        if not path.is_absolute():path=PROJECT/path
        need(not path.exists(),'仓库正在进行其他 Git 操作：'+marker+'；保留并停止')
    before=local_edits()
    saved=LOCAL/'install-preserved'/str(time.time_ns());saved.mkdir(parents=True)
    for name,raw in before.items():(saved/name).write_bytes(raw)
    report=dict(schema='headroom-local-install-preservation-v1',client_revision=CLIENT_REVISION,
                head_before=head,target_commit=RELEASE_DATA['commit'],snapshot=str(saved),
                snapshot_sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in before.items()},
                local_status=before['status.bin'].decode('utf-8','backslashreplace').replace('\0','\n'),
                cloud_connections=0,PPO_updates=0)
    report_path=LOCAL/'install-report.json'
    conflicts=incoming_conflicts(head,ref)
    if conflicts:
        report.update(status='PATH_CONFLICT_PRESERVED',conflicts=conflicts);write(report_path,report)
        raise RuntimeError('待导入路径与本机文件冲突，未移动或覆盖：'+
                           ', '.join(x['existing'] for x in conflicts[:12])+'；报告：'+str(report_path))
    report.update(status='FAST_FORWARD_READY',conflicts=[]);write(report_path,report)
    if before['status.bin']:
        print('本机改动与新增代码不冲突；保留现状导入。改动快照：'+str(saved),flush=True)
    try:
        subprocess.run(['git','-C',str(PROJECT),'-c','merge.autoStash=false','merge',
                        '--ff-only','--no-autostash','--no-overwrite-ignore',ref],check=True)
    except BaseException as exc:
        report.update(status='FAST_FORWARD_FAILED_PRESERVE',error=repr(exc));write(report_path,report)
        raise
    after=local_edits()
    preserved=before==after
    report.update(status='INSTALLED_EDITS_PRESERVED' if preserved else 'EDIT_COMPARISON_FAILED_PRESERVE',
                  head_after=git('rev-parse','HEAD').decode().strip(),local_edits_preserved=preserved,
                  after_sha256={name:hashlib.sha256(raw).hexdigest() for name,raw in after.items()})
    write(report_path,report)
    need(preserved,'导入后本机改动记录发生变化；停止后续推送/训练，请提供报告：'+str(report_path))


def install_code():
    print('HEADROOM_INSTALLER='+CLIENT_REVISION,flush=True)
    need(PROJECT.is_dir(),'找不到固定本机仓库：'+str(PROJECT))
    need(git('remote','get-url','origin').decode().strip()==REMOTE,'保留原 Git SSH 远端；当前 origin 不匹配')
    need(git('branch','--show-current').decode().strip()=='double-balance','请先回到 double-balance 分支')
    LOCAL.mkdir(parents=True,exist_ok=True)
    path=LOCAL/'microduck-stage12-headroom-repair-v2-from-v1.bundle'
    raw=zlib.decompress(base64.b85decode(RELEASE_DATA['bundle']))
    need(hashlib.sha256(raw).hexdigest()==RELEASE_DATA['bundle_sha256'],'内嵌 bundle 校验失败')
    if path.exists():need(path.read_bytes()==raw,'已存在不同 bundle；保留并停止')
    else:path.write_bytes(raw)
    subprocess.run(['git','-C',str(PROJECT),'bundle','verify',str(path)],check=True)
    ref='refs/remotes/stage12-headroom-repair-v2/double-balance'
    subprocess.run(['git','-C',str(PROJECT),'fetch',str(path),'refs/heads/double-balance:'+ref],check=True)
    need(git('rev-parse',ref).decode().strip()==RELEASE_DATA['commit'],'bundle 提交不匹配')
    head=git('rev-parse','HEAD').decode().strip()
    if head!=RELEASE_DATA['commit']:
        ancestor=subprocess.run(['git','-C',str(PROJECT),'merge-base','--is-ancestor',RELEASE_DATA['commit'],'HEAD']).returncode==0
        if not ancestor:
            need(head==BASE_COMMIT,'本机分支有新提交或分歧；不覆盖、不强制合并，请保留输出')
            preserve_then_fast_forward(head,ref)
    if (LOCAL/'code-installed.json').exists() and not (LOCAL/'code-installed-before-repair-v2.json').exists():
        write(LOCAL/'code-installed-before-repair-v2.json',read(LOCAL/'code-installed.json'))
    write(LOCAL/'code-installed.json',dict(commit=RELEASE_DATA['commit'],package_identity=RELEASE_DATA['package_identity'],
                                         client_revision=CLIENT_REVISION))
    print('HEADROOM_CODE_INSTALLED；未推送、未连接云主机、未训练。')
    print('下一条：git -C '+shlex.quote(str(PROJECT))+' push origin double-balance')


def initialize_ssh(argument):
    existing=read(LOCAL/'endpoint.json') if (LOCAL/'endpoint.json').exists() else None
    endpoint=parse_endpoint(argument) if argument else existing
    if endpoint is None:
        endpoint=parse_endpoint(input('粘贴本阶段当前云 5090 的完整 SSH 指令（不输入密码）：').strip())
    ssh=types.ModuleType('headroom_shared_ssh')
    exec(compile(zlib.decompress(base64.b85decode(RELEASE_DATA['ssh'])), '<headroom_shared_ssh>','exec'),ssh.__dict__)
    ssh.HOST,ssh.PORT=endpoint['host'],endpoint['port']
    return ssh,endpoint


def source_payload():
    """Read the already imported commit, never the possibly edited worktree."""
    commit=RELEASE_DATA['commit'];expected=RELEASE_DATA['package_identity']
    code='experiments/stage12/headroom-repair-v2/package'
    raw=git('show',commit+':'+code+'/manifest.json');manifest=json.loads(raw)
    calculated=hashlib.sha256(json.dumps(manifest['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    need(calculated==expected and manifest['identity']==expected,'本机固定提交的训练包身份不匹配')
    files={'manifest.json':base64.b64encode(raw).decode()}
    for name,pin in manifest['files'].items():
        need(re.fullmatch(r'[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+',name) is not None
             and name!='manifest.json','无效训练包文件名')
        value=git('show',commit+':'+code+'/'+name)
        need(len(value)==pin['bytes'] and hashlib.sha256(value).hexdigest()==pin['sha256'],
             '本机已提交训练代码校验失败：'+name)
        files[name]=base64.b64encode(value).decode()
    raw=json.dumps(dict(commit=commit,package_identity=expected,source_path=code,files=files),
                   sort_keys=True,separators=(',',':')).encode()
    need(len(raw)<=4*1024*1024,'小型代码包超出上限；不自动改用大型文件直传')
    return dict(encoding='zlib-base85-json-v1',raw_bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),data=base64.b85encode(zlib.compress(raw,9)).decode())


def rpc(ssh,operation,timeout=360):
    boot=RELEASE_DATA['bootstrap']
    remote='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b85decode('+repr(boot)+')))')
    request=dict(operation=operation,commit=RELEASE_DATA['commit'],package_identity=RELEASE_DATA['package_identity'])
    if operation=='repair':
        request['source']=source_payload()
        print('HEADROOM_SOURCE_TRANSPORT='+TRANSPORT_REVISION+
              '; committed source bytes='+str(request['source']['raw_bytes']),flush=True)
    args=ssh.argv(remote)  # Authentication occurs on the terminal, outside the captured log.
    logs=LOCAL/'requests';logs.mkdir(parents=True,exist_ok=True)
    logpath=logs/(str(time.time_ns())+'-'+operation+'.log')
    requestpath=logpath.with_suffix('.request.json')
    with requestpath.open('xb') as f:
        f.write(json.dumps(request,separators=(',',':')).encode())
    print('云端操作：'+operation,flush=True)
    print('当前启动日志： '+str(logpath),flush=True)
    started=time.monotonic();next_heartbeat=started+15
    with logpath.open('xb') as log,requestpath.open('rb') as input_file,logpath.open('rb') as progress:
        # File stdin avoids blocking on pipe capacity before the timeout loop starts.
        # This group contains only the request slave, not the shared SSH master.
        child=subprocess.Popen(args,stdin=input_file,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        pending=b'';remote_error=False
        def show_new_lines(final=False):
            nonlocal pending,remote_error
            pending+=progress.read()
            lines=pending.split(b'\n');pending=lines.pop()
            if final and pending:lines.append(pending);pending=b''
            for raw in lines:
                line=raw.decode('utf-8','replace')
                if line.startswith('MD12_HEADROOM_ERROR='):remote_error=True
                if line and not line.startswith(('MD12_HEADROOM_STATE=','MD12_HEADROOM_RETURN=')):
                    print(line[:4000],flush=True)
        try:
            while child.poll() is None:
                show_new_lines()
                if remote_error:break
                now=time.monotonic()
                if now-started>timeout:
                    value=dict(status='SSH_REQUEST_OUTCOME_UNCERTAIN_PRESERVE',operation=operation,
                               log=str(logpath),request=str(requestpath),elapsed_seconds=now-started)
                    write(LOCAL/'last-error.json',value)
                    raise TimeoutError('SSH 请求超时，结果未确认；仅用 --status 查看，不能新开预算。日志：'+str(logpath))
                if now>=next_heartbeat:
                    print('等待 '+operation+' 返回，已 '+str(int(now-started))+' 秒；不会重复启动。',flush=True)
                    next_heartbeat=now+15
                try:child.wait(timeout=1)
                except subprocess.TimeoutExpired:pass
        finally:
            if child.poll() is None:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:os.killpg(child.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    child.wait()
            show_new_lines(final=True)
    content=logpath.read_text(errors='replace')
    values=[]
    for line in content.splitlines():
        for prefix in ('MD12_HEADROOM_STATE=','MD12_HEADROOM_RETURN=','MD12_HEADROOM_ERROR='):
            if line.startswith(prefix):values.append((prefix,json.loads(line[len(prefix):])))
    need(values,'云端未返回完整状态。日志：'+str(logpath)+'\n'+content[-7000:])
    prefix,value=values[-1]
    if prefix=='MD12_HEADROOM_ERROR=' or child.returncode!=0:
        write(LOCAL/'last-error.json',value)
        raise RuntimeError('云端保留停止：'+json.dumps(value,ensure_ascii=False)+'\n日志：'+str(logpath))
    if prefix=='MD12_HEADROOM_RETURN=':write(LOCAL/'cloud-return.json',value)
    return value


def save_report(value):
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    p=LOCAL/'reports'/('microduck-stage12-headroom-'+stamp+'.json')
    value=dict(value,local_report_utc=time.time(),controller_sha256=sha(Path(__file__)),
               controller_package_identity=RELEASE_DATA['package_identity'])
    write(p,value);write(LOCAL/'latest-report.json',value)
    print('HEADROOM_REPORT_SAVED='+str(p),flush=True)
    return p


def display(value):
    b=value.get('budget',{});p=value.get('progress',{})
    print('状态 '+value['status']+' | '+str(p.get('phase',''))+'/'+str(p.get('detail',''))
        +' | PPO '+str(b.get('completed_updates',0))+'/'+str(b.get('attempted_updates',0))+'/505（完成/消耗/上限）'
        +' | 评估 '+str(len(b.get('evaluations',{})))+'/8'
        +' | 控制步 '+str(p.get('control_steps',p.get('rollout_steps','—'))),flush=True)
    if 'exit' in value:
        print('本组无外力保持与模型边界：'+('通过' if value['exit']['checked_baseline_goal_passed'] else '未通过')
              +'；10% 抗扰尚未训练。',flush=True)


def validate_same_campaign(value,endpoint):
    local=read(LOCAL/'campaign.json') if (LOCAL/'campaign.json').exists() else None
    if local is not None:
        need(value.get('started',{}).get('started_utc')==local['started_utc']
             and value['started']['deadline_utc']==local['deadline_utc'],
             '这台实例缺少原预算账本或截止时间不同；不能另开一轮 505 次')
    intent=read(LOCAL/'launch-intent.json') if (LOCAL/'launch-intent.json').exists() else None
    if local is None and intent is not None:
        need(intent['endpoint']==endpoint,'上次启动结果未确认，不能换主机新开预算；先找回原状态')
    if value.get('started'):
        need(value['started']['package_identity']==ORIGINAL_PACKAGE_ID
             and value['started']['started_utc']==ORIGINAL_STARTED
             and value['started']['deadline_utc']==ORIGINAL_DEADLINE,'原任务身份或预算时间不同')
        active=value.get('package_identity')
        need(active in (ORIGINAL_PACKAGE_ID,RELEASE_DATA['package_identity']),'云端训练包身份不同')
        if active==RELEASE_DATA['package_identity']:
            repair=value.get('repair',{})
            need(repair.get('package_identity')==active and repair.get('original_package_identity')==ORIGINAL_PACKAGE_ID
                 and repair.get('original_deadline_utc')==ORIGINAL_DEADLINE,'缺少明确的代码修复回执')
        write(LOCAL/'campaign.json',dict(started_utc=value['started']['started_utc'],
            deadline_utc=value['started']['deadline_utc'],package_identity=RELEASE_DATA['package_identity']))


def watch(ssh,value):
    while True:
        display(value)
        if value['status'] in ('NOT_STARTED','NOT_INSTALLED') or 'exit' in value or value['status'].endswith('PRESERVE'):
            save_report(value);return value
        time.sleep(15)
        value=rpc(ssh,'status',timeout=60)


def download_return():
    receipt=read(LOCAL/'cloud-return.json')
    need(receipt['status']=='CLOUD_ARTIFACT_PUSH_COMPLETE','缺少已完成云端推送回执')
    branch,commit=receipt['branch'],receipt['commit']
    need(re.fullmatch(r'stage12-headroom-v1-return-[a-z0-9-]+',branch) is not None
         and re.fullmatch('[0-9a-f]{40}',commit) is not None,'Invalid returned Git identity')
    cache=LOCAL/'return-cache.git'
    if not cache.exists():subprocess.run(['git','init','--bare',str(cache)],check=True)
    ref='refs/remotes/headroom/'+branch
    subprocess.run(['git','-C',str(cache),'fetch','--no-tags',REMOTE,'refs/heads/'+branch+':'+ref],check=True)
    def obj(path):
        return subprocess.check_output(['git','-C',str(cache),'show',commit+':'+path])
    index_raw=obj('return-index.json')
    need(hashlib.sha256(index_raw).hexdigest()==receipt['index_sha256'],'回传索引 SHA256 不匹配')
    index=json.loads(index_raw)
    need(index['package_identity'] in (ORIGINAL_PACKAGE_ID,RELEASE_DATA['package_identity']) and index['branch']==branch,
         '回传索引身份不同')
    root=LOCAL/'returned'/commit
    root.mkdir(parents=True,exist_ok=True)
    needed=index['inventory']['included'];seen=set()
    for i,(name,pin) in enumerate(index['archives'].items(),1):
        need(re.fullmatch(r'part-[0-9]{4}\.zip',name) is not None,'Invalid archive name')
        archive=root/name
        if not archive.exists():
            data=obj(name)
            need(len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['sha256'],'回传压缩包 SHA256 不匹配')
            temporary=archive.with_suffix('.zip.part')
            with temporary.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
            os.replace(temporary,archive)
        need(archive.stat().st_size==pin['bytes'] and sha(archive)==pin['sha256'],'已有压缩包不匹配；保留停止')
        with zipfile.ZipFile(archive) as z:
            need(z.namelist()==pin['members'] and len(z.namelist())==len(set(z.namelist())), '压缩包成员清单不匹配')
            for info in z.infolist():
                path=PurePosixPath(info.filename)
                need(not path.is_absolute() and '..' not in path.parts and info.filename in needed,
                     '不安全的回传文件路径')
                need(not stat.S_ISLNK(info.external_attr>>16),'回传不接受符号链接')
                meta=needed[info.filename];need(info.file_size==meta['bytes'],'回传文件大小不匹配')
                target=root/'files'/info.filename
                need(target.parent.resolve().is_relative_to((root/'files').resolve()),'不安全的回传父目录')
                target.parent.mkdir(parents=True,exist_ok=True)
                need(not target.is_symlink() and target.resolve().is_relative_to((root/'files').resolve()),
                     '回传目标路径异常')
                if target.exists():need(sha(target)==meta['sha256'],'已有回传文件不同；不覆盖')
                else:
                    data=z.read(info)
                    need(hashlib.sha256(data).hexdigest()==meta['sha256'],'解包文件 SHA256 不匹配')
                    temporary=target.with_name(target.name+'.part')
                    need(not temporary.is_symlink(),'不安全的临时回传路径')
                    with temporary.open('wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
                    os.replace(temporary,target)
                seen.add(info.filename)
        print('回传校验 '+str(i)+'/'+str(len(index['archives']))+' 包完成。',flush=True)
    need(seen==set(needed),'回传文件不完整')
    write(root/'return-index.json',index)
    local=dict(status='LOCAL_RETURN_VERIFIED',branch=branch,commit=commit,verified_files=len(seen),
        verified_utc=time.time(),path=str(root),cloud_power_state='UNCONFIRMED',
        original_entire_cloud_storage_backed_up=False)
    write(root/'return-verified.json',local);write(LOCAL/'return-verified.json',local)
    print('HEADROOM_LOCAL_RETURN_VERIFIED='+str(root),flush=True)
    print('本轮任务状态：'+index['inventory']['cloud_computation_status']+'；受检双球目标通过：'+str(index['inventory']['checked_baseline_goal_passed']),flush=True)
    if (root/'files/progress.json').is_file():
        result=read(root/'files/progress.json')
        if result.get('error'):print('本轮原始错误：'+result['error'],flush=True)
    print('本轮必要计算与回传已结束且 SHA256 校验通过。云 GPU 现在不再需要，请在平台手动关机并保留所需数据。',flush=True)
    print('本轮未包含的中间模型已列在索引中；旧云端完整历史数据未宣称全部备份。',flush=True)
    write(LOCAL/'shutdown-reminder.json',dict(reminded_utc=time.time(),actual_shutdown_confirmation='UNCONFIRMED'))
    return local


def main():
    parser=argparse.ArgumentParser(description='Stage12 已审查的缺模块修复；保留同一 505 次预算和原截止时间。')
    actions=parser.add_mutually_exclusive_group(required=True)
    for action in ('install-code','repair','resume','watch','status','collect','stop','return','download'):
        actions.add_argument('--'+action,action='store_true')
    parser.add_argument('--ssh',help='当前阶段平台 SSH 指令；首次未提供时在终端粘贴')
    args=parser.parse_args()
    LOCAL.mkdir(parents=True,exist_ok=True)
    if args.install_code:install_code();return
    if args.download:
        receipt=download_return()
        if (LOCAL/'latest-report.json').exists():
            value=read(LOCAL/'latest-report.json');value['local_return']=receipt;save_report(value)
        return
    if args.repair or args.resume:
        need((LOCAL/'code-installed.json').exists(),'先运行 --install-code，再由你 git push origin double-balance')
        need(read(LOCAL/'code-installed.json')['commit']==RELEASE_DATA['commit'],'本机安装的代码提交不同')
    ssh,endpoint=initialize_ssh(args.ssh)
    if args.repair:
        need((LOCAL/'campaign.json').is_file() and (LOCAL/'launch-intent.json').is_file(),
             '缺少原本机任务记录；不能新建预算')
        campaign=read(LOCAL/'campaign.json');intent=read(LOCAL/'launch-intent.json')
        need(campaign['started_utc']==ORIGINAL_STARTED and campaign['deadline_utc']==ORIGINAL_DEADLINE,
             '原本机账本身份不同；保留停止')
        need(intent['endpoint']==endpoint,'只能修复原实例；不能换主机重开预算')
        current=None
    else:
        current=rpc(ssh,'status',timeout=60)
        validate_same_campaign(current,endpoint)
    write(LOCAL/'endpoint.json',endpoint)
    if args.repair or args.resume:
        need((LOCAL/'launch-intent.json').exists(),'原启动意向缺失；保留停止')
        value=rpc(ssh,'resume' if args.resume else 'repair')
        validate_same_campaign(value,endpoint)
        value=watch(ssh,value)
        if 'exit' in value:
            # A terminal status can precede supervisor exit by a fraction of a second.
            for _ in range(10):
                if not any(value['processes_alive'].values()):break
                time.sleep(1);value=rpc(ssh,'status',timeout=60)
            rpc(ssh,'return',timeout=1800)
            value['local_return']=download_return();save_report(value)
    elif args.watch:watch(ssh,current)
    elif args.status or args.collect:display(current);save_report(current)
    elif args.stop:display(rpc(ssh,'stop'));save_report(rpc(ssh,'status',timeout=60))
    elif getattr(args,'return'):
        rpc(ssh,'return',timeout=1800)
        current['local_return']=download_return();save_report(current)


if __name__=='__main__':
    try:main()
    except KeyboardInterrupt:
        print('\n已退出本机监控；云端任务不会因此停止。查看用 --status，明确停止作业用 --stop。')
        raise SystemExit(130)
    except BaseException as exc:
        if isinstance(exc,SystemExit):raise
        print('HEADROOM_PRESERVED: '+repr(exc),file=sys.stderr)
        if '--install-code' in sys.argv:
            print('本机安装未完成；本次没有推送、连接云端或启动训练。请保留以上错误和 install-report.json（如有）。',file=sys.stderr)
        else:
            print('不会自动续训、续预算或关闭实例。状态用 --status；仅重试回传用 --return；已有云推送回执仅下载用 --download。',file=sys.stderr)
        raise SystemExit(2)
