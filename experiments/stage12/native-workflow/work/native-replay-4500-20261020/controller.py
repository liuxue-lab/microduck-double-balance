#!/usr/bin/env python3
"""Local RTX5060 visual rollout of the already selected Stage12 checkpoint."""
import argparse
import base64
from datetime import datetime,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import zlib

PACKAGE_B85=@@PACKAGE_B85@@
PACKAGE_SHA=@@PACKAGE_SHA@@
PACKAGE_ID=@@PACKAGE_ID@@
payload=zlib.decompress(base64.b85decode(PACKAGE_B85))
if hashlib.sha256(payload).hexdigest()!=PACKAGE_SHA:raise RuntimeError('Embedded package differs')
PACKAGE=json.loads(payload)
exec(compile(PACKAGE['replay_common.py'],'<embedded replay_common>','exec'),globals())

def install():
    folder=SLOT/'package';folder.mkdir(exist_ok=True)
    for name,source in PACKAGE.items():
        need(Path(name).name==name and name.endswith('.py'),'Invalid package member')
        target=folder/name;data=source.encode()
        if target.exists():need(not target.is_symlink() and target.read_bytes()==data,'Existing replay package differs: '+name)
        else:
            with target.open('xb') as f:f.write(data)
    return folder

def snapshot():
    value=read(SLOT/'progress.json') if (SLOT/'progress.json').exists() else {'status':'PREPARED'}
    if (SLOT/'started.json').exists():value['started']=read(SLOT/'started.json')
    if (SLOT/'worker.json').exists():
        pin=read(SLOT/'worker.json');value['worker']=pin;value['worker_alive']=alive(pin)
    if (SLOT/'result.json').exists():value['result']=read(SLOT/'result.json')
    if (SLOT/'finished.json').exists():
        value['finished']=read(SLOT/'finished.json')
        if value['finished']['worker_exit_code']!=0 or value.get('result',{}).get('status')!='LOCAL_VIDEO_COMPLETE':
            value['status']='LOCAL_VIDEO_ERROR_PRESERVE'
    elif 'worker_alive' in value and not value['worker_alive'] and not value.get('result'):
        value['status']='LOCAL_VIDEO_ERROR_PRESERVE'
        value.setdefault('error','Local worker is no longer alive; partial files preserved')
    return value

def show(value):
    print('本机录像 %s；完成 %s/2 段；当前种子 %s；步数 %s/500；阶段 %s；新增PPO=0。'%(
        value.get('status','UNKNOWN'),value.get('completed_clips',0),value.get('seed','—'),
        value.get('control_steps_in_clip',0),value.get('phase','—')),flush=True)
    if value.get('started'):
        print('原15分钟上限剩 %.1f 分钟。'%max(0,(value['started']['deadline_utc']-time.time())/60),flush=True)

def save_report(value):
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    folder=SLOT/'reports';folder.mkdir(exist_ok=True)
    report=dict(value,controller_package_identity=PACKAGE_ID,local_return_utc=stamp,
                added_PPO_updates=0,cloud_connections=0,power_action='NONE')
    logs={}
    for path in [SLOT/'worker.log',*sorted(SLOT.glob('ffmpeg-*.log'))]:
        if path.is_file():
            with path.open('rb') as f:f.seek(max(0,path.stat().st_size-9000));tail=f.read().decode('utf-8',errors='replace')
            logs[path.name]=tail
    if logs:report['log_tails']=logs
    out=folder/('microduck-stage12-replay-4500-'+stamp+'-'+str(os.getpid())+'.json')
    write(out,report,exclusive=True)
    print('LOCAL_VIDEO_REPORT_SAVED='+str(out),flush=True)
    print('报告 %.1f KiB；SHA256=%s'%(out.stat().st_size/1024,sha(out)),flush=True)
    result=value.get('result',{})
    if value.get('status')=='LOCAL_VIDEO_COMPLETE' and result.get('status')=='LOCAL_VIDEO_COMPLETE':
        print('视频总览：'+str(SLOT/'index.html'),flush=True)
        for clip in result['clips']:print('视频：'+clip['video']['path'],flush=True)
        print('请回传小JSON及两段MP4；不用回传模型。5090无需开启。',flush=True)
    else:
        error=value.get('error')
        if not error:error=value.get('finished',{}).get('error')
        if error:print('错误：'+str(error),flush=True)
        print('现场已保留；请回传上述小JSON，不要删除started.json或重开录像预算。',flush=True)
    return out

def halt(process,pin):
    # Never signal an unrelated process or a reused PID.
    if process.poll() is not None:return process.returncode
    need(alive(pin) and pin['group']==pin['pid'],'Worker process identity changed; refuse unrelated signal')
    try:os.killpg(pin['group'],signal.SIGTERM)
    except ProcessLookupError:pass
    try:return process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        need(alive(pin),'Worker identity changed during shutdown')
        os.killpg(pin['group'],signal.SIGKILL)
        return process.wait(timeout=10)

def monitor(process,pin):
    stop_reason=None;last=0
    try:
        while process.poll() is None:
            value=snapshot();now=time.time()
            if now-last>=20:show(value);last=now
            if now>=value['started']['deadline_utc']:
                stop_reason='Original 15-minute wall-clock limit reached';halt(process,pin);break
            time.sleep(1)
    except KeyboardInterrupt:
        stop_reason='Interrupted by user; original budget and partial files retained'
        print('正在停止本机录像并保留现场……',flush=True);halt(process,pin)
    except BaseException:
        if process.poll() is None:halt(process,pin)
        raise
    code=process.wait();value=snapshot()
    finished=dict(worker_exit_code=code,worker_alive=alive(pin),finished_utc=time.time())
    if stop_reason:finished['error']=stop_reason
    write(SLOT/'finished.json',finished,exclusive=True);value['finished']=finished
    if code!=0 or value.get('result',{}).get('status')!='LOCAL_VIDEO_COMPLETE':
        value['status']='LOCAL_VIDEO_ERROR_PRESERVE'
        value.setdefault('error',stop_reason or 'Local worker exited without complete videos; see log_tails')
    save_report(value)
    return 0 if value['status']=='LOCAL_VIDEO_COMPLETE' and code==0 else 2

def watch():
    need((SLOT/'started.json').exists(),'No local replay has been started')
    try:
        while True:
            value=snapshot();show(value)
            if not value.get('worker_alive',False):save_report(value);return 0 if value.get('result') else 2
            time.sleep(20)
    except KeyboardInterrupt:
        print('只退出只读监看；本机录像由原启动终端管理。',flush=True);return 0

def launch():
    SLOT.mkdir(parents=True,exist_ok=True)
    with (SLOT/'launch.lock').open('a+b') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('已有本机录像入口运行中；不重复启动。可用本脚本 --watch 只读查看。',flush=True);show(snapshot());return 2
        if (SLOT/'started.json').exists():
            value=snapshot();show(value);save_report(value)
            print('本次录像已经启动过；不重录、不重置原截止时间。',flush=True)
            return 0 if value.get('result',{}).get('status')=='LOCAL_VIDEO_COMPLETE' else 2
        process=None;pin=None
        try:
            need(PYTHON.is_file(),'Existing project Python is missing; no environment installation')
            verify_inputs();folder=install()
            env=dict(os.environ)
            env.update(PYTHONDONTWRITEBYTECODE='1',MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',CUDA_VISIBLE_DEVICES='0',
                OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='4',HF_HUB_OFFLINE='1',WANDB_MODE='disabled',
                PYTHONUNBUFFERED='1')
            env.pop('PYTHONPATH',None);env.pop('PYTHONHOME',None)
            now=time.time()
            write(SLOT/'started.json',dict(started_utc=now,deadline_utc=now+MAX_SECONDS,maximum_control_steps=1000,
                maximum_clips=2,seeds=list(SEEDS),package_identity=PACKAGE_ID,added_PPO_updates=0),exclusive=True)
            print('本机RTX5060：固定两段，每段最多10秒；15分钟墙钟上限含首次初始化。',flush=True)
            print('这是前台录像；保持本机开机及此终端。Ctrl+C会停止本次录像，结果与预算保留。无需SSH或网络。',flush=True)
            with (SLOT/'worker.log').open('xb') as log:
                process=subprocess.Popen([str(PYTHON),'-B',str(folder/'record.py')],cwd=str(PROJECT),env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            pin=identity(process.pid);need(pin and pin['group']==process.pid,'Cannot identify local worker')
            write(SLOT/'worker.json',pin,exclusive=True)
            return monitor(process,pin)
        except BaseException as exc:
            if process is not None and process.poll() is None and pin is not None:halt(process,pin)
            value=snapshot();value.update(status='LOCAL_VIDEO_ERROR_PRESERVE',error=repr(exc),traceback=traceback.format_exc()[-10000:])
            save_report(value);return 2

def main():
    parser=argparse.ArgumentParser(description='4500模型本机有限录像；无新增PPO，无云连接，无自动重录。')
    mode=parser.add_mutually_exclusive_group(required=True);mode.add_argument('--record',action='store_true');mode.add_argument('--watch',action='store_true')
    args=parser.parse_args()
    try:return launch() if args.record else watch()
    except Exception as exc:print('已停止：'+repr(exc),flush=True);return 2

if __name__=='__main__':raise SystemExit(main())
