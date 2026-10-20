#!/usr/bin/env python3
"""Wait for the existing Stage12 campaign and collect a read-only final report."""
import argparse
import base64
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
import zlib

SSH_B85=@@SSH_B85@@
SSH_SHA=@@SSH_SHA@@
REMOTE_B85=@@REMOTE_B85@@
REMOTE_SHA=@@REMOTE_SHA@@
SLOT='/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/native-training-v2'


def decode(value, digest):
    source=zlib.decompress(base64.b85decode(value))
    if hashlib.sha256(source).hexdigest()!=digest:
        raise RuntimeError('Embedded read-only source hash mismatch')
    return source


def request(ssh, operation):
    source=decode(REMOTE_B85,REMOTE_SHA).decode()
    command=' '.join(map(shlex.quote,['python3','-B','-c',source,SLOT,operation]))
    r=subprocess.run(ssh['argv'](command),capture_output=True,timeout=120)
    output=r.stdout.decode(errors='replace');err=r.stderr.decode(errors='replace')
    rows=[line.split('=',1)[1] for line in output.splitlines() if line.startswith('MD12_FINISH_STATE=')]
    if r.returncode or not rows:
        raise RuntimeError('只读连接返回不完整；云端任务状态未确认。'+err[-1500:]+output[-1500:])
    return json.loads(rows[-1])


def save_report(value):
    value=dict(value,local_return_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    folder=Path.home()/'microduck-double-balance/artifacts/double-balance-stage12/native-training-v2-reports'
    folder.mkdir(parents=True,exist_ok=True)
    path=folder/('microduck-stage12-finish-readonly-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')+'-'+str(os.getpid())+'.json')
    data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with path.open('xb') as f:
        f.write(data);f.flush();os.fsync(f.fileno())
    print('FINISH_READONLY_REPORT_SAVED='+str(path),flush=True)
    print('报告 %.1f KiB；SHA256=%s'%(len(data)/1024,hashlib.sha256(data).hexdigest()),flush=True)
    return path


def show(value):
    b=value.get('files',{}).get('budget.json') or {}
    p=value.get('files',{}).get('run/progress.json') or {}
    maximum=b.get('max_updates',6000)
    count=b.get('completed_updates',0)
    text='只读监看 %s；完成 %s/%s（%.1f%%）；阶段 %s。'%(value['status'],count,maximum,100*count/maximum if maximum else 0,p.get('phase','未知'))
    if 'joint_geometric_current_fraction' in p:
        text+=' 当前双平衡几何稳定 %.1f%%。'%(100*p['joint_geometric_current_fraction'])
    print(text,flush=True)


def run(watch, ssh):
    while True:
        value=request(ssh,'status');show(value)
        if value['status']=='STOPPED':
            print('后台任务已退出；正在读取训练曲线并核对最后/训练期最佳检查点SHA256。',flush=True)
            value=request(ssh,'collect')
            if value['status']=='RUNNING' and watch:
                print('检测到任务重新处于运行状态，继续只读等待。',flush=True)
            else:
                save_report(value)
                summary=value.get('summary',{})
                print('训练结束记录确认：%s；可用模型字节核对：%s。'%(summary.get('finished_training_run_confirmed'),summary.get('available_checkpoint_bytes_verified')),flush=True)
                print('只回传这个小JSON。模型和完整日志仍保留云端；尚未执行独立策略评估。',flush=True)
                return value
        elif not watch or value['status']!='RUNNING':
            save_report(value)
            return value
        time.sleep(20)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--watch',action='store_true',help='只读等待当前任务结束，然后自动收尾')
    mode.add_argument('--collect',action='store_true',help='只查一次；若任务已结束，收集完整收尾报告')
    args=parser.parse_args()
    print('ssh -p 52763 root@connect.bjb2.seetacloud.com',flush=True)
    print('只读收尾：不停止/启动/续训，不新增短测、PPO或仿真，不操作云机电源。',flush=True)
    helper=decode(SSH_B85,SSH_SHA)
    ssh={'__name__':'shared_ssh'}
    exec(compile(helper,'<shared_ssh>','exec'),ssh)
    try:
        run(args.watch,ssh)
    except KeyboardInterrupt:
        print('\n只退出本地收尾监控；云端训练不受影响。用同文件 --watch 接回。',flush=True)
    except Exception as exc:
        save_report(dict(status='READONLY_CLIENT_ERROR_PRESERVE',error=repr(exc),
                         cloud_training_state='UNKNOWN',added_PPO_updates=0,added_simulation_steps=0,
                         traceback=traceback.format_exc()[-4000:]))
        print('未发送任何训练操作。重新连接：python3 '+shlex.quote(str(Path(__file__).resolve()))+' --watch',flush=True)
        return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
