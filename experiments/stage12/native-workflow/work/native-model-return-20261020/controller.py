#!/usr/bin/env python3
"""Retrieve the selected Stage12 model and completed evidence, without simulation."""
import argparse
import base64
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import zlib

SSH_B85=@@SSH_B85@@
SSH_SHA=@@SSH_SHA@@
REMOTE_B85=@@REMOTE_B85@@
REMOTE_SHA=@@REMOTE_SHA@@
FILES=@@FILES@@
CHUNK=1024*1024

def need(ok,message):
    if not ok:raise RuntimeError(message)

def decode(value,digest):
    data=zlib.decompress(base64.b85decode(value))
    need(hashlib.sha256(data).hexdigest()==digest,'Embedded source identity differs')
    return data

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda:f.read(CHUNK),b''):h.update(data)
    return h.hexdigest()

def request(ssh,args):
    source=decode(REMOTE_B85,REMOTE_SHA).decode()
    command=' '.join(map(shlex.quote,['python3','-B','-c',source,*map(str,args)]))
    p=subprocess.run(ssh['argv'](command),capture_output=True,timeout=180)
    need(p.returncode==0,'只读取回失败；已保留断点。'+p.stderr.decode(errors='replace')[-5000:])
    if args[0]!='chunk':return json.loads(p.stdout)
    header,separator,data=p.stdout.partition(b'\n');parts=header.split()
    need(separator and len(parts)==3 and parts[0]==b'MD12_DATA','Chunk response header is incomplete')
    need(len(data)==int(parts[1])==int(args[3]) and hashlib.sha256(data).hexdigest()==parts[2].decode(),
         'Chunk size or SHA256 mismatch; this chunk was not written')
    return data

def local_files(folder):
    found={}
    for key,item in FILES.items():
        path=folder/item['local']
        if not path.exists():continue
        need(path.is_file() and not path.is_symlink(),'Unexpected local artifact: '+str(path))
        need(sha(path)==item['sha256'],'Existing local artifact differs; preserved without overwrite: '+str(path))
        found[key]=dict(path=str(path),bytes=path.stat().st_size,sha256=item['sha256'])
    return found

def download(folder,ssh):
    found=local_files(folder)
    if len(found)==len(FILES):
        print('本机已有全部文件且SHA256一致：100%；无需SSH。',flush=True)
        return found
    print('只读读取4500模型和已完成证据；不启动仿真或PPO。',flush=True)
    inventory=request(ssh,['inventory'])
    need(set(inventory)==set(FILES),'Remote artifact inventory differs')
    for key,item in inventory.items():
        need(item['sha256']==FILES[key]['sha256'] and 0<item['bytes']<=FILES[key]['max_bytes'],'Remote artifact identity differs')
    total=sum(item['bytes'] for item in inventory.values());completed=sum(inventory[k]['bytes'] for k in found)
    started=time.monotonic()
    for key,item in FILES.items():
        if key in found:continue
        target=folder/item['local'];target.parent.mkdir(parents=True,exist_ok=True)
        partial=target.with_name(target.name+'.partial');size=inventory[key]['bytes']
        need(not partial.is_symlink(),'Unexpected partial symlink')
        if partial.exists():
            need(partial.is_file() and partial.stat().st_size<=size,'Unexpected partial file')
            offset=partial.stat().st_size;prefix=request(ssh,['prefix',key,offset])
            need(prefix['bytes']==offset and prefix['sha256']==sha(partial),'Partial prefix differs; preserved without truncation')
            print('续传 %s：%.1f%%。'%(key,100*offset/size),flush=True)
        else:offset=0
        mode='ab' if partial.exists() else 'xb'
        with partial.open(mode) as f:
            while offset<size:
                data=request(ssh,['chunk',key,offset,min(CHUNK,size-offset)])
                f.write(data);f.flush();os.fsync(f.fileno());offset+=len(data)
                print('取回 %.1f/%.1f MiB（%.1f%%）；%s %.1f%%；耗时 %.1f 秒。'%(
                    (completed+offset)/CHUNK,total/CHUNK,100*(completed+offset)/total,
                    key,100*offset/size,time.monotonic()-started),flush=True)
        need(sha(partial)==item['sha256'],'Completed file SHA256 mismatch; partial preserved')
        # Publish only verified bytes. link() refuses to overwrite an existing target.
        os.link(partial,target);partial.unlink()
        found[key]=dict(path=str(target),bytes=size,sha256=item['sha256']);completed+=size
    return found

def local_replay_inventory(folder):
    project=Path.home()/'microduck-double-balance/workspace'
    pins=json.loads((folder/'task/source-pins.json').read_bytes())
    matched=[];missing=[];different=[]
    for rel,pin in pins.items():
        path=Path(rel);need(not path.is_absolute() and '..' not in path.parts,'Invalid frozen source relative path')
        actual=project/path
        if not actual.is_file():missing.append(rel)
        elif actual.stat().st_size==pin['bytes'] and sha(actual)==pin['sha256']:matched.append(rel)
        else:different.append(rel)
    runtime=project/'.venv/bin/python';packages={}
    if runtime.is_file():
        probe="import importlib.metadata as m,json,sys; packages={}; names=['mjlab','mujoco','mujoco-warp','warp-lang','rsl-rl-lib','torch'];\nfor name in names:\n try: packages[name]=m.version(name)\n except m.PackageNotFoundError: packages[name]=None\nprint(json.dumps(dict(python=sys.version,executable=sys.executable,packages=packages)))"
        try:
            p=subprocess.run([str(runtime),'-B','-c',probe],capture_output=True,timeout=30)
            packages=json.loads(p.stdout) if p.returncode==0 else dict(error=p.stderr.decode(errors='replace')[-2000:])
        except Exception as exc:packages=dict(error=repr(exc))
    return dict(project=str(project),frozen_source_count=len(pins),matching_files=len(matched),
        missing_count=len(missing),different_count=len(different),missing=missing,different=different,
        existing_python=str(runtime) if runtime.is_file() else None,runtime_metadata=packages,
        modifications=0,dependency_installations=0,simulation_steps=0,PPO_updates=0,
        meaning='Read-only preparation inventory, not an inference or physical validation result')

def save_report(folder,value):
    reports=folder/'reports';reports.mkdir(exist_ok=True)
    value=dict(value,created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),added_PPO_updates=0,
               added_simulation_steps=0,remote_writes=0,selected_model='4500',power_action='NONE')
    path=reports/('microduck-stage12-fetch-4500-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'-'+str(os.getpid())+'.json')
    data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    print('SELECTED_MODEL_REPORT_SAVED='+str(path),flush=True)
    print('报告 %.1f KiB；SHA256=%s'%(len(data)/1024,hashlib.sha256(data).hexdigest()),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--download',action='store_true',required=True);ap.parse_args()
    folder=Path.home()/'microduck-double-balance/artifacts/double-balance-stage12/native-selected-4500'
    folder.mkdir(parents=True,exist_ok=True);need(not folder.is_symlink(),'Unexpected destination symlink')
    print('候选4500：212/256成功；取回约10 MiB；不计算GPU、不改云端文件。',flush=True)
    ssh={'__name__':'shared_ssh'}
    exec(compile(decode(SSH_B85,SSH_SHA),'<shared_ssh>','exec'),ssh)
    with (folder/'download.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('已有本机下载正在运行，请接回原终端。',flush=True);return 2
        try:
            files=download(folder,ssh)
            print('取回100%，正在只读核对本机源码与依赖；不导入Torch、不运行仿真。',flush=True)
            inventory=local_replay_inventory(folder)
            save_report(folder,dict(status='SELECTED_MODEL_RETURN_VERIFIED',files=files,local_replay_inventory=inventory))
            print('4500模型已核对并保存在：'+files['model']['path'],flush=True)
            print('请只回传上述小JSON；不上传模型。本次取回完成，如无其他云任务可关闭5090并保留存储。',flush=True)
            return 0
        except (Exception,KeyboardInterrupt) as exc:
            save_report(folder,dict(status='SELECTED_MODEL_RETURN_PRESERVED',error=repr(exc)))
            print('具体错误：'+repr(exc)+'；已保留下载字节，重复同一命令续传。',flush=True)
            return 2

if __name__=='__main__':raise SystemExit(main())
