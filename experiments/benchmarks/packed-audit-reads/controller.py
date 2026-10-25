"""Humble deployment preparation; source is read from the pinned Git commit."""
import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import time
import zlib

PROJECT=Path.home()/'microduck-double-balance/workspace'
LOCAL=PROJECT.parent/'artifacts/double-balance-stage12/native-pc-audit-v1'
REMOTE='git@github.com:liuxue-lab/microduck-double-balance.git'
BASE_COMMIT='f7a95ee1b89f5c123c3e3281bc23f69660213669'
SCRIPT_NAME='microduck-stage12-pc-audit-v1-20261024.py'
CLIENT_REVISION='pc-audit-v1'
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
        need(change==b'A' and name.startswith('experiments/stage12/pc-audit-v1/')
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
    report=dict(schema='final6h-local-install-preservation-v1',client_revision=CLIENT_REVISION,
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
    need(preserved,'导入后本机改动记录发生变化；停止后续操作，请提供报告：'+str(report_path))


def install_code():
    print('PC_AUDIT_INSTALLER='+CLIENT_REVISION,flush=True)
    need(PROJECT.is_dir(),'找不到固定本机仓库：'+str(PROJECT))
    need(git('remote','get-url','origin').decode().strip()==REMOTE,'保留原 Git SSH 远端；当前 origin 不匹配')
    need(git('branch','--show-current').decode().strip()=='double-balance','请先回到 double-balance 分支')
    LOCAL.mkdir(parents=True,exist_ok=True)
    path=LOCAL/'microduck-stage12-pc-audit-v1-from-graph-v1.bundle'
    raw=zlib.decompress(base64.b85decode(RELEASE_DATA['bundle']))
    need(hashlib.sha256(raw).hexdigest()==RELEASE_DATA['bundle_sha256'],'内嵌 bundle 校验失败')
    if path.exists():need(path.read_bytes()==raw,'已存在不同 bundle；保留并停止')
    else:path.write_bytes(raw)
    subprocess.run(['git','-C',str(PROJECT),'bundle','verify',str(path)],check=True)
    ref='refs/remotes/stage12-pc-audit-v1/double-balance'
    subprocess.run(['git','-C',str(PROJECT),'fetch',str(path),'refs/heads/double-balance:'+ref],check=True)
    need(git('rev-parse',ref).decode().strip()==RELEASE_DATA['commit'],'bundle 提交不匹配')
    head=git('rev-parse','HEAD').decode().strip()
    if head!=RELEASE_DATA['commit']:
        ancestor=subprocess.run(['git','-C',str(PROJECT),'merge-base','--is-ancestor',RELEASE_DATA['commit'],'HEAD']).returncode==0
        if not ancestor:
            need(head==BASE_COMMIT,'本机分支有新提交或分歧；不覆盖、不强制合并，请保留输出')
            preserve_then_fast_forward(head,ref)
    if (LOCAL/'code-installed.json').exists() and not (LOCAL/'code-installed-before-local-eval.json').exists():
        write(LOCAL/'code-installed-before-local-eval.json',read(LOCAL/'code-installed.json'))
    write(LOCAL/'code-installed.json',dict(commit=RELEASE_DATA['commit'],package_identity=RELEASE_DATA['package_identity'],
                                         client_revision=CLIENT_REVISION))
    print('PC_AUDIT_CODE_INSTALLED；下一步由用户通过已有 Git SSH 身份推送。')
    print('下一条：git -C '+shlex.quote(str(PROJECT))+' push origin double-balance')


def package_files():
    commit=RELEASE_DATA['commit'];expected=RELEASE_DATA['package_identity']
    prefix='experiments/stage12/pc-audit-v1/package/'
    raw=git('show',commit+':'+prefix+'manifest.json');manifest=json.loads(raw)
    digest=hashlib.sha256(json.dumps(manifest['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    need(digest==expected and manifest['identity']==expected,'固定提交的验证代码身份不符')
    result={'manifest.json':raw}
    for name,pin in manifest['files'].items():
        need(re.fullmatch(r'[A-Za-z0-9_-]+\.[A-Za-z0-9_.-]+',name) is not None,'Invalid package member')
        raw=git('show',commit+':'+prefix+name)
        need(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256'],'验证代码校验失败：'+name)
        result[name]=raw
    return result


def install_package():
    need((LOCAL/'code-installed.json').is_file() and read(LOCAL/'code-installed.json')['commit']==RELEASE_DATA['commit'],
         '请先运行 --install-code，再按固定规范 git push origin double-balance')
    need(not LOCAL.is_symlink(),'Unexpected validation output symlink')
    folder=LOCAL/'package';need(not folder.is_symlink(),'Unexpected package symlink')
    folder.mkdir(exist_ok=True)
    for name,data in package_files().items():
        path=folder/name;need(not path.is_symlink(),'Unexpected source symlink')
        if path.exists():need(path.read_bytes()==data,'已有验证代码不同；保留停止：'+name)
        else:
            with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    return folder


def installed_package():
    folder=LOCAL/'package'
    need(folder.is_dir() and not folder.is_symlink(),'尚未安装运行包；首次请用 --run')
    manifest=read(folder/'manifest.json')
    digest=hashlib.sha256(json.dumps(manifest['files'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    need(digest==RELEASE_DATA['package_identity']==manifest['identity'],'已安装验证代码身份不同')
    for name,pin in manifest['files'].items():
        need(Path(name).name==name,'Invalid source member')
        path=folder/name
        need(not path.is_symlink() and path.is_file() and path.stat().st_size==pin['bytes'] and sha(path)==pin['sha256'],
             '已安装验证代码有变化：'+name)
    return folder


def main():
    parser=argparse.ArgumentParser(description='电脑端已验证 BAM Graph 与合并检查读取各160步对照；保留500Hz边界与外力检查，不训练。')
    actions=parser.add_mutually_exclusive_group(required=True)
    for name in ('install-code','run','retry-run','status','watch','stop'):
        actions.add_argument('--'+name,action='store_true')
    args=parser.parse_args()
    if args.install_code:install_code();return 0
    operation=next(name for name in ('run','retry-run','status','watch','stop') if getattr(args,name.replace('-','_')))
    folder=install_package() if operation in ('run','retry-run') else installed_package()
    if operation in ('run','retry-run'):
        print('本轮只使用电脑 RTX5060；保持主终端开启。5090 保持关闭，无需连接 Orin。',flush=True)
    env={k:v for k,v in os.environ.items() if k not in ('PYTHONPATH','PYTHONHOME')}
    os.chdir(PROJECT)
    os.execve('/usr/bin/python3',['/usr/bin/python3','-B','-u',str(folder/'audit_ops.py'),operation],env)


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print('操作已中断；保留最近报告。',flush=True)
        raise SystemExit(130)
    except Exception as exc:
        print('PC_AUDIT_PRESERVED='+repr(exc),file=sys.stderr)
        raise SystemExit(2)
