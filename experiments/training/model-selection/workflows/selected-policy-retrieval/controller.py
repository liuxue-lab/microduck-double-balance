#!/usr/bin/env python3
"""Fetch speed-refine +500 and record two bounded local RTX5060 examples."""
import argparse
import base64
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback
import zlib

SOURCES_B85=@@SOURCES_B85@@
SOURCES_SHA=@@SOURCES_SHA@@
payload=zlib.decompress(base64.b85decode(SOURCES_B85))
if hashlib.sha256(payload).hexdigest()!=SOURCES_SHA:raise RuntimeError('Embedded source bundle differs')
SOURCES=json.loads(payload)
SELECTED=Path.home()/'microduck-double-balance/artifacts/double-balance-stage12/native-selected-speed500-v1'
REPLAY=SELECTED.parent/'native-replay-speed500-v1'
fetch={'__name__':'selected_fetch'};exec(compile(SOURCES['fetch'],'<selected_fetch>','exec'),fetch)
replay={'__name__':'selected_replay'};exec(compile(SOURCES['replay'],'<selected_replay>','exec'),replay)
context={}

def save(value):
    folder=SELECTED/'reports';folder.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    report=dict(status=value.get('status','UNKNOWN'),selected_model='speed_refine_+500',policy_updates=5000,
                selected_checkpoint_sha256=fetch['FIXED_FILES']['model']['sha256'],fetch=context.get('fetch'),
                local_replay=value if 'started' in value or 'result' in value else None,
                added_PPO_updates=0,ssh_read_requests=fetch['READ_REQUESTS'],cloud_GPU_jobs_started=0,
                cloud_artifact_writes=0,power_action='NONE',source_bundle_sha256=SOURCES_SHA,local_return_utc=stamp)
    if value.get('error'):report['error']=value['error']
    if value.get('traceback'):report['traceback']=value['traceback']
    if 'local_return_utc' in value:report['local_replay']=value
    path=folder/('microduck-stage12-selected-speed500-'+stamp+'-'+str(os.getpid())+'.json')
    data=(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as stream:stream.write(data);stream.flush();os.fsync(stream.fileno())
    print('SELECTED_REPLAY_REPORT_SAVED='+str(path),flush=True)
    print('报告 %.1f KiB；SHA256=%s'%(len(data)/1024,hashlib.sha256(data).hexdigest()),flush=True)
    if value.get('status')=='LOCAL_VIDEO_COMPLETE' and value.get('result',{}).get('status')=='LOCAL_VIDEO_COMPLETE':
        print('视频总览：'+str(REPLAY/'index.html'),flush=True)
        for clip in value['result']['clips']:print('视频：'+clip['video']['path'],flush=True)
        print('请回传上述小JSON及两段MP4；模型不回传。',flush=True)
        print('本次必要云端取回已完成；若两小时内仍需用5090则保留开机，否则可手动关闭并保留存储。',flush=True)
    elif value.get('error'):
        print('已保留现场：'+str(value['error']),flush=True)
    return path

replay['REPORT_CALLBACK']=save

def download_and_verify():
    ssh={'__name__':'shared_ssh'}
    exec(compile(fetch['decode'](fetch['SSH_B85'],fetch['SSH_SHA']),'<shared_ssh>','exec'),ssh)
    fetch['FILES']=dict(fetch['FIXED_FILES'])
    files=fetch['download'](SELECTED,ssh)
    evaluation=json.loads((SELECTED/'evaluation/evaluation-result.json').read_bytes())
    fetch['FILES']=fetch['expanded_files'](evaluation)
    files=fetch['download'](SELECTED,ssh)
    print('全部文件已核对；正在只读复核已有256回合与本机源码、依赖。',flush=True)
    inventory=fetch['local_replay_inventory'](SELECTED)
    context['fetch']=dict(status='FILES_VERIFIED_AUDIT_PENDING',files=files,local_replay_inventory=inventory,
                          added_PPO_updates=0,added_simulation_steps=0,remote_artifact_writes=0)
    python=inventory['existing_python'];fetch['need'](python is not None,'Existing local project Python is missing')
    proc=subprocess.run([python,'-B','-c',SOURCES['audit'],str(SELECTED)],capture_output=True,timeout=60)
    fetch['need'](proc.returncode==0,'Retained evidence audit failed: '+proc.stderr.decode(errors='replace')[-5000:])
    audit=json.loads(proc.stdout)
    receipt=dict(status='SELECTED_MODEL_RETURN_VERIFIED',files=files,local_replay_inventory=inventory,evidence_audit=audit,
                 added_PPO_updates=0,added_simulation_steps=0,remote_artifact_writes=0)
    replay['write'](SELECTED/'fetch-verified.json',receipt)
    context['fetch']=receipt
    return receipt

def existing_receipt():
    path=SELECTED/'fetch-verified.json'
    if path.is_file():context['fetch']=json.loads(path.read_bytes())

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run',action='store_true',help='取回并校验，再进行本机两段有限录像')
    mode.add_argument('--download',action='store_true',help='只取回核验，不运行仿真')
    mode.add_argument('--watch',action='store_true',help='只读本机录像进度，不连接云端')
    args=parser.parse_args(argv)
    SELECTED.mkdir(parents=True,exist_ok=True)
    try:
        fetch['need'](not SELECTED.is_symlink(),'Unexpected selected destination symlink')
        if args.watch:
            existing_receipt();return replay['watch']()
        with (SELECTED/'pipeline.lock').open('a+b') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:
                print('此入口已在运行；请保留原终端。录像阶段可用 --watch 只读查看。',flush=True);return 2
            if args.run and (REPLAY/'started.json').exists():
                existing_receipt();return replay['launch']()
            print('选定速度奖励微调+500（策略累计5000）：253/256成功。取回约10 MiB。'+('只下载核验，不运行仿真。' if args.download else '随后使用本机5060录像。'),flush=True)
            receipt=download_and_verify()
            if args.download:save({'status':'SELECTED_MODEL_RETURN_VERIFIED'});return 0
            inventory=receipt['local_replay_inventory']
            fetch['need'](inventory['matching_files']==365 and inventory['missing_count']==inventory['different_count']==0,
                          'Local source differs; download retained, no visual simulation started')
            print('云端必要取回已完成；两小时内还要用5090则保留开机，否则可手动关机并保留存储。录像使用本机5060。',flush=True)
            return replay['launch']()
    except (Exception,KeyboardInterrupt) as exc:
        save(dict(status='SELECTED_MODEL_OR_REPLAY_ERROR_PRESERVE',error=repr(exc),traceback=traceback.format_exc()[-10000:]))
        print('下载中断可重复 --run 续传；录像一旦启动不自动重录。请回传上述小JSON。',flush=True)
        return 2

if __name__=='__main__':raise SystemExit(main())
