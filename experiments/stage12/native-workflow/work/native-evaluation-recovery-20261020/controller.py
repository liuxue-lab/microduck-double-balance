#!/usr/bin/env python3
"""Reuse completed 4500 episodes, evaluate only 6000; original deadline retained."""
import argparse
import base64
import datetime as dt
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
BOOT_B85=@@BOOT_B85@@
BOOT_SHA=@@BOOT_SHA@@
PAYLOAD_B85=@@PAYLOAD_B85@@
PAYLOAD_SHA=@@PAYLOAD_SHA@@
PACKAGE_ID=@@PACKAGE_ID@@
SLOT='/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/native-evaluation-recovery-v1'

def decode(value,digest,compressed=True):
    data=base64.b85decode(value)
    if compressed:data=zlib.decompress(data)
    if hashlib.sha256(data).hexdigest()!=digest:raise RuntimeError('Embedded package hash differs')
    return data

def rpc(ssh,args,data=None):
    command=' '.join(map(shlex.quote,args))
    p=subprocess.run(ssh['argv'](command),input=data,capture_output=True,timeout=180)
    raw=p.stdout.decode(errors='replace');err=p.stderr.decode(errors='replace')
    rows=[line.split('=',1)[1] for line in raw.splitlines() if line.startswith('MD12_EVAL_STATE=')]
    if p.returncode or not rows:raise RuntimeError('SSH返回不完整；不自动重试启动。'+err[-1800:]+raw[-8000:])
    return json.loads(rows[-1])

def request(ssh,operation):
    if operation=='start':
        boot=decode(BOOT_B85,BOOT_SHA).decode()
        payload=decode(PAYLOAD_B85,PAYLOAD_SHA,compressed=False)
        print('发送评估代码约 %.1f KiB；复用云端项目、模型和环境。'%(len(payload)/1024),flush=True)
        return rpc(ssh,['python3','-B','-c',boot,PACKAGE_ID],payload)
    code="from pathlib import Path; import os,sys; p=Path(sys.argv[1])/'package/agent.py'; print('MD12_EVAL_STATE={\\\"status\\\":\\\"NOT_STARTED\\\"}') if not p.is_file() else os.execv(sys.executable,[sys.executable,'-B',str(p),'status'])"
    return rpc(ssh,['python3','-B','-c',code,SLOT])

def save_report(value):
    value=dict(value,local_return_utc=dt.datetime.now(dt.timezone.utc).isoformat(),controller_package_identity=PACKAGE_ID)
    directory=Path.home()/'microduck-double-balance/artifacts/double-balance-stage12/native-evaluation-recovery-v1-reports'
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/('microduck-stage12-native-eval-recovery-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'-'+str(os.getpid())+'.json')
    data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    print('NATIVE_EVALUATION_REPORT_SAVED='+str(path),flush=True)
    print('报告 %.1f KiB；SHA256=%s'%(len(data)/1024,hashlib.sha256(data).hexdigest()),flush=True)

def show(value):
    p=value.get('progress',{});start=value.get('started',{})
    print('模型评估 %s；首回合 %d/512（%.1f%%）；模型 %s；分组 %s/4；采样 %s/500；阶段 %s；新增PPO=0。'%(
        value['status'],p.get('completed_first_episodes',0),p.get('progress_percent',0.),p.get('model','—'),
        p.get('group','—'),p.get('control_steps_in_group',0),p.get('phase','等待')),flush=True)
    if start:print('评估原截止时间剩 %.1f 分钟。'%max(0.,(start['deadline_utc']-time.time())/60),flush=True)

def main():
    ap=argparse.ArgumentParser(description=__doc__);mode=ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--resume',action='store_true');mode.add_argument('--watch',action='store_true');mode.add_argument('--collect',action='store_true')
    args=ap.parse_args()
    helper=decode(SSH_B85,SSH_SHA);ssh={'__name__':'shared_ssh'};exec(compile(helper,'<shared_ssh>','exec'),ssh)
    print('ssh -p 52763 root@connect.bjb2.seetacloud.com',flush=True)
    print('保留4500已完成256回合，只补6000的256回合；原评估截止时间不变；新增PPO=0。',flush=True)
    print('使用相同配置作独立批次比较，不再声称初始状态严格配对；原指纹不一致记录保留。',flush=True)
    try:
        value=request(ssh,'start' if args.resume else 'status')
        while True:
            show(value)
            pending=any(v is True for v in value.get('processes_alive',{}).values()) or value['status'] in ('STARTING','INITIALIZING','EVALUATING') or (
                value['status']=='EVALUATION_COMPLETE' and 'exit' not in value)
            if args.collect or not pending:
                save_report(value)
                if value['status']=='EVALUATION_COMPLETE' and 'evaluation' in value:
                    for key,s in value['evaluation']['summaries'].items():
                        print('模型%s：5秒成功 %s/%s（%.1f%%），连续稳定时长中位数 %.2f 秒。'%(key,s['successes'],s['episodes'],100*s['success_fraction'],s['longest_stable_seconds']['median']),flush=True)
                print('请回传上述小JSON；模型和逐回合数据保留云端。',flush=True)
                return 0
            time.sleep(20);value=request(ssh,'status')
    except KeyboardInterrupt:
        print('\n只退出本机监看，云端评估继续；用同一文件 --watch 接回。',flush=True);return 0
    except Exception as exc:
        save_report(dict(status='CLIENT_ERROR_PRESERVE',error=repr(exc),cloud_evaluation_state='UNKNOWN',added_PPO_updates=0))
        print('已保留现场。只读接回：python3 '+shlex.quote(str(Path(__file__).resolve()))+' --watch',flush=True);return 2

if __name__=='__main__':raise SystemExit(main())
