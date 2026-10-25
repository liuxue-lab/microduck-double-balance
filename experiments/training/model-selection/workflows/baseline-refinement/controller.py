#!/usr/bin/env python3
"""Finite native refinement from 4500; one SSH authentication, durable cloud run."""
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
SLOT='/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/native-refine-4500-v1'
RUNTIME='/root/autodl-tmp/microduck-double-balance/workspace/.venv/bin/python'

def decode(value,digest,compressed=True):
    data=base64.b85decode(value)
    if compressed:data=zlib.decompress(data)
    if hashlib.sha256(data).hexdigest()!=digest:raise RuntimeError('Embedded package hash differs')
    return data

def rpc(ssh,args,data=None):
    p=subprocess.run(ssh['argv'](' '.join(map(shlex.quote,args))),input=data,capture_output=True,timeout=180)
    raw=p.stdout.decode(errors='replace');err=p.stderr.decode(errors='replace')
    rows=[line.split('=',1)[1] for line in raw.splitlines() if line.startswith('MD12_REFINE_STATE=')]
    errors=[line.split('=',1)[1] for line in raw.splitlines() if line.startswith('MD12_REFINE_ERROR=')]
    if errors:
        error=json.loads(errors[-1]);return dict(error.get('state',{}),status='REMOTE_ERROR_PRESERVE',remote_error=error)
    if p.returncode or not rows:raise RuntimeError('SSH返回不完整；不自动重试启动。'+err[-1800:]+raw[-8000:])
    return json.loads(rows[-1])

def request(ssh,operation):
    if operation in ('start','resume'):
        boot=decode(BOOT_B85,BOOT_SHA).decode();payload=decode(PAYLOAD_B85,PAYLOAD_SHA,compressed=False)
        print('发送微调代码约 %.1f KiB；复用云端项目、模型和环境。'%(len(payload)/1024),flush=True)
        return rpc(ssh,['python3','-B','-c',boot,PACKAGE_ID,operation],payload)
    code="from pathlib import Path; import os,sys; p=Path(sys.argv[1])/'package/agent.py'; print('MD12_REFINE_STATE={\"status\":\"NOT_STARTED\"}') if not p.is_file() else os.execv(sys.argv[2],[sys.argv[2],'-B',str(p),sys.argv[3]])"
    return rpc(ssh,['python3','-B','-c',code,SLOT,RUNTIME,operation])

def save_report(value):
    value=dict(value,local_return_utc=dt.datetime.now(dt.timezone.utc).isoformat(),controller_package_identity=PACKAGE_ID)
    directory=Path.home()/'microduck-double-balance/artifacts/double-balance-stage12/native-refine-4500-v1-reports'
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/('microduck-stage12-refine-4500-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'-'+str(os.getpid())+'.json')
    data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    print('NATIVE_REFINEMENT_REPORT_SAVED='+str(path),flush=True)
    print('报告 %.1f KiB；SHA256=%s'%(len(data)/1024,hashlib.sha256(data).hexdigest()),flush=True)

def show(value):
    b=value.get('budget',{});p=value.get('progress',{});selection=value.get('selection') or {};best=selection.get('best',{})
    print('4500后续微调 %s；新增完成 %d；已用尝试 %d/1000；分段 %s/4；阶段 %s。'%(
        value['status'],b.get('completed_updates',0),b.get('attempted_updates',0),p.get('segment','—'),p.get('phase','等待')),flush=True)
    if p.get('status') in ('EVALUATING','EVALUATION_SEGMENT_COMPLETE'):
        print('本段评估首回合 %d/256；分组 %s/2；采样 %s/500。'%(p.get('completed_first_episodes',0),p.get('group','—'),p.get('control_steps_in_group',0)),flush=True)
    if 'joint_geometric_current_fraction' in p:
        print('当前脚下稳定 %.1f%%；双平衡几何稳定 %.1f%%；采样 %.2f 秒，PPO %.2f 秒/更新。'%(
            p['lower_unassisted_current_fraction']*100,p['joint_geometric_current_fraction']*100,p['collection_seconds'],p['learning_seconds']),flush=True)
    if best:
        s=best['summary'];print('当前最佳模型 %s：5秒成功 %s/256（%.1f%%）；连续未改善 %d/2。'%(best['model'],s['successes'],100*s['success_fraction'],selection['no_improvement']),flush=True)
    if b:print('本轮原截止时间剩 %.1f 分钟；最多1000次更新，可提前结束。'%max(0,(b['deadline_utc']-time.time())/60),flush=True)
    if 'remote_error' in value:print('具体错误：'+value['remote_error'].get('error','未知'),flush=True)

def pending(value):
    if value['status']=='REMOTE_ERROR_PRESERVE':return False
    return any(value.get('processes_alive',{}).values()) or (
        'exit' not in value and value['status'] in ('STARTING','INITIALIZING','TRAINING','TRAINING_SEGMENT_COMPLETE','EVALUATING','EVALUATION_SEGMENT_COMPLETE','SELECTING'))

def main():
    ap=argparse.ArgumentParser(description=__doc__);mode=ap.add_mutually_exclusive_group(required=True)
    for flag in ('train','resume','watch','collect','stop'):mode.add_argument('--'+flag,action='store_true')
    args=ap.parse_args();helper=decode(SSH_B85,SSH_SHA);ssh={'__name__':'shared_ssh'};exec(compile(helper,'<shared_ssh>','exec'),ssh)
    print('ssh -p 52763 root@connect.bjb2.seetacloud.com',flush=True)
    if args.train or args.resume:
        print('从4500恢复完整模型、归一化和优化器；4096环境，固定学习率1e-5；原任务与成功判定不变。',flush=True)
        print('最多新增1000次PPO尝试、总墙钟2小时；每250次评估256首回合；连续两段未提高成功数即停。',flush=True)
        print('复用4500原256回合；独立批次比较，不声称初始状态严格配对；不重复短测。',flush=True)
    operation='start' if args.train else 'resume' if args.resume else 'stop' if args.stop else 'status'
    try:
        value=request(ssh,operation)
        while True:
            show(value)
            if args.collect or not pending(value):
                save_report(value)
                if value['status']=='REFINEMENT_COMPLETE':
                    best=value['selection']['best'];s=best['summary']
                    print('本轮完成；保留最佳模型 %s：%s/256（%.1f%%）。'%(best['model'],s['successes'],100*s['success_fraction']),flush=True)
                    print('最佳检查点：'+best['checkpoint']['path'],flush=True)
                print('请只回传上述小JSON；模型、逐回合记录和旧证据保留云端。',flush=True)
                if value['status'].endswith('PRESERVE'):
                    print('已保留现场。--watch 只读接回；--resume 仅在预算未过期且检查点完整时恢复，不重置预算。',flush=True)
                return 2 if value['status'].endswith('ERROR_PRESERVE') else 0
            time.sleep(20);value=request(ssh,'status')
    except KeyboardInterrupt:
        print('\n只退出本机监看，云端任务继续；重连后用同一文件 --watch。',flush=True);return 0
    except Exception as exc:
        print('具体错误：'+str(exc),flush=True)
        save_report(dict(status='CLIENT_ERROR_PRESERVE',error=repr(exc),cloud_training_state='UNKNOWN'))
        print('已保留现场。只读接回：python3 '+shlex.quote(str(Path(__file__).resolve()))+' --watch',flush=True);return 2

if __name__=='__main__':raise SystemExit(main())
