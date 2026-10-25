#!/usr/bin/env python3
"""Stage08 E -> one native 64x24x5 smoke -> 2048-environment fine-tuning.
--train starts only if no prior campaign exists; otherwise attaches read-only.
--status/--watch/--collect are read-only. Ctrl+C ends local monitoring only.
--resume is explicit formal-only recovery under the original deadline/budget.
"""
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
HOST='root@connect.bjb2.seetacloud.com'
PORT=52763
SLOT='/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/native-training-v2'
PYTHON='/root/autodl-tmp/microduck-double-balance/workspace/.venv/bin/python'
REPORTS=Path.home()/'microduck-double-balance/artifacts/double-balance-stage12/native-training-v2-reports'
PACKAGE_B85=@@PACKAGE_B85@@
PACKAGE_SHA=@@PACKAGE_SHA@@
BOOTSTRAP_B85=@@BOOTSTRAP_B85@@
SSH_B85=@@SSH_B85@@
SSH_SHA=@@SSH_SHA@@
PACKAGE_ID=@@PACKAGE_ID@@
PROBE_B85=@@PROBE_B85@@
PRIMARY_SHA='86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97'
PRIMARY_REL='artifacts/double-balance-stage08/returned/5d998fc322345102/evidence-20260930T093645-8955/extension/E-20260929/segment-001/checkpoints/update_004000.pt'

def require(ok,why):
    if not ok:raise RuntimeError(why)

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()

def save_report(value):
    REPORTS.mkdir(parents=True,exist_ok=True)
    value=dict(value,local_return_utc=dt.datetime.now(dt.timezone.utc).isoformat(),controller_package_identity=PACKAGE_ID)
    p=REPORTS/('microduck-stage12-native-v2-report-'+dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid())+'-'+str(time.time_ns()%1000000)+'.json')
    data=(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
    with p.open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
    print('NATIVE_TRAINING_REPORT_SAVED='+str(p),flush=True)
    print('报告 %.1f KiB；SHA256=%s'%(len(data)/1024,hashlib.sha256(data).hexdigest()),flush=True)
    return p

def rpc(ssh,args,data=None,timeout=120):
    command=' '.join(shlex.quote(x) for x in args)
    try:p=subprocess.run(ssh['argv'](command),input=data,capture_output=True,timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError('SSH 超时；任务可能继续。用 --watch 接回。若发生在模型补传，重复 --train 会从已收到的位置继续。') from exc
    raw=p.stdout.decode(errors='replace');err=p.stderr.decode(errors='replace')
    states=[x.split('=',1)[1] for x in raw.splitlines() if x.startswith('MD12_NATIVE_STATE=')]
    errors=[x.split('=',1)[1] for x in raw.splitlines() if x.startswith('MD12_NATIVE_ERROR=')]
    if errors:raise RuntimeError(json.loads(errors[-1]).get('error',errors[-1]))
    require(p.returncode==0 and states,'SSH 返回不完整；不自动重试启动。'+err[-1500:]+raw[-1500:])
    return json.loads(states[-1])

def probe(ssh,collect=False):
    code=zlib.decompress(base64.b85decode(PROBE_B85.encode())).decode()
    code+='\nx=discover()\n'
    if collect:code+="if x.get('slot'): x=collect_slot(Path(x['slot']))\n"
    code+="if not x.get('campaign_exists'):\n    missing=[str(p) for p in (PYTHON,SOURCE) if not p.exists()]\n    if missing: x.update(status='CLONED_DATA_MISSING',missing_paths=missing)\n"
    code+="print('MD12_NATIVE_STATE='+json.dumps(x,ensure_ascii=False,allow_nan=False))\n"
    return rpc(ssh,['python3','-B','-c',code])

def agent(ssh,operation,extra=(),data=None):
    return rpc(ssh,[PYTHON,'-B',SLOT+'/package/agent.py',operation,*map(str,extra)],data)

def install(ssh):
    blob=base64.b85decode(PACKAGE_B85.encode());require(hashlib.sha256(blob).hexdigest()==PACKAGE_SHA,'Embedded package damaged')
    code=zlib.decompress(base64.b85decode(BOOTSTRAP_B85.encode())).decode()
    print('发送代码约 %.1f KiB；复用云端项目和环境。'%(len(blob)/1024+len(code.encode())/1024),flush=True)
    return rpc(ssh,[PYTHON,'-B','-c',code,PACKAGE_SHA],blob)

def ensure_initial_checkpoint(ssh,state,explicit):
    if state['status']!='NEEDS_INITIAL_CHECKPOINT':return
    root=Path.home()/'microduck-double-balance'
    paths=[Path(explicit).expanduser()] if explicit else [root/PRIMARY_REL]
    if not explicit:
        for stage in ('double-balance-stage08','double-balance-stage09','double-balance-stage10'):
            folder=root/'artifacts'/stage
            if folder.is_dir():paths.extend(folder.rglob('update_004000.pt'))
    for path in paths:
        if not path.is_file() or not 0<path.stat().st_size<=512*1024**2:continue
        print('核对已有Stage08候选：'+str(path),flush=True)
        if digest(path)!=PRIMARY_SHA:continue
        size=path.stat().st_size
        state=agent(ssh,'receive',('--size',size));offset=state['offset']
        print('云端未找到指定模型；补传 %.2f MiB，已保留 %.2f MiB。'%(size/1024**2,offset/1024**2),flush=True)
        with path.open('rb') as f:
            f.seek(offset)
            while offset<size:
                data=f.read(min(1024**2,size-offset));require(data,'Model changed during transfer')
                state=agent(ssh,'receive',('--size',size,'--offset',offset,'--chunk-sha',hashlib.sha256(data).hexdigest()),data)
                offset=state['offset']
                print('模型补传 %.1f%%（%.2f/%.2f MiB）'%(100*offset/size,offset/1024**2,size/1024**2),flush=True)
        require(state['status']=='CHECKPOINT_PRESENT','Model transfer incomplete');return
    raise RuntimeError('未找到固定SHA的Stage08 E主模型，未开训。用 --checkpoint 指定本机已有的 update_004000.pt；不会回退到早期模型。')

def show(x):
    if x.get('status')=='CLONED_DATA_MISSING':
        print('新实例缺少既有运行环境或冻结项目，未启动训练；缺失路径：'+', '.join(x.get('missing_paths',[])),flush=True);return
    if x.get('status')=='MULTIPLE_CAMPAIGNS_PRESERVED':
        print('发现多个既有训练记录；全部保留，未启动。请回传小报告。',flush=True);return
    p=x.get('progress',{});b=x.get('new_training_budget') or {}
    count=b.get('completed_updates',p.get('completed_updates',0));attempt=b.get('attempted_updates',0)
    if x.get('legacy_attached'):
        print('已接回旧v1，未安装/启动v2；状态 %s，已完成PPO %s。'%(x['status'],count),flush=True);return
    phase=p.get('campaign_phase','smoke' if not x.get('smoke') else 'formal')
    if phase=='smoke':
        text='短测 %s；更新 %s/5（%.1f%%）；采样 %s/24。'%(x['status'],min(count,5),min(count,5)*20,p.get('rollout_steps',0))
    else:
        text='正式微调 %s；总完成 %s/%s（%.1f%%，含短测5次）；已用尝试 %s。'%(x['status'],count,b.get('max_updates',6000),100*count/b.get('max_updates',6000),attempt)
    text+=' 当前阶段 '+p.get('phase','等待启动')+'。'
    if b:text+=' 耗时 %.1f 分钟；原截止时间剩 %.1f 分钟。'%((time.time()-b['started_utc'])/60,max(0,b['deadline_utc']-time.time())/60)
    if 'joint_geometric_current_fraction' in p:
        text+=' 当前脚下稳定 %.1f%%；双平衡几何稳定 %.1f%%。'%(100*p['lower_unassisted_current_fraction'],100*p['joint_geometric_current_fraction'])
    print(text,flush=True)

def collect(ssh):
    x=probe(ssh,collect=True)
    for name,item in x.get('small_receipts',{}).items():
        data=item['text'].encode();require(len(data)==item['bytes'] and hashlib.sha256(data).hexdigest()==item['sha256'],'Small receipt differs: '+name)
    show(x);save_report(x)
    print('只需回传上述小JSON；模型、完整日志和旧证据保留云端。',flush=True)
    if x.get('slot') and not x.get('supervisor_alive') and not x.get('training_alive'):
        print('该任务进程已退出；若无其他必要云任务，可在平台关闭5090并保留存储。脚本不操作电源。',flush=True)
    return x

def watch(ssh,state=None):
    while True:
        x=state or probe(ssh);state=None;show(x)
        if not x.get('supervisor_alive') and not x.get('training_alive'):return collect(ssh)
        time.sleep(20)

def main():
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
    for name in ('train','status','watch','collect','stop','resume'):m.add_argument('--'+name,action='store_true')
    p.add_argument('--checkpoint',help='仅用于补传固定SHA的已有Stage08 E主模型');a=p.parse_args()
    require(not a.checkpoint or a.train,'--checkpoint is for --train only')
    print('云端连接：ssh -p %s %s'%(PORT,HOST),flush=True)
    print('计划：仅1次64环境×24步×5更新短测，通过后自动接续2048环境微调。',flush=True)
    print('总上限6000次PPO尝试（含短测）或8小时；旧4次预约保留；本机不运行PPO。',flush=True)
    source=zlib.decompress(base64.b85decode(SSH_B85.encode()));require(hashlib.sha256(source).hexdigest()==SSH_SHA,'SSH helper differs')
    ssh={'__name__':'shared_ssh'};exec(compile(source,'<shared-ssh>','exec'),ssh);ssh['HOST']=HOST;ssh['PORT']=PORT
    try:
        if a.train:
            prior=probe(ssh)
            if prior.get('campaign_exists'):return watch(ssh,prior)
            if prior.get('status')=='CLONED_DATA_MISSING':
                show(prior);save_report(prior);return 2
            prepared=install(ssh)
            if prepared.get('campaign_exists'):return watch(ssh,prepared)
            ensure_initial_checkpoint(ssh,prepared,a.checkpoint)
            return watch(ssh,agent(ssh,'start'))
        if a.watch:return watch(ssh)
        if a.collect:return collect(ssh)
        if a.stop or a.resume:
            prior=probe(ssh)
            if prior.get('slot')!=SLOT:return collect(ssh)
            return watch(ssh,agent(ssh,'stop' if a.stop else 'resume'))
        x=probe(ssh);show(x);save_report(x);return x
    except KeyboardInterrupt:
        print('\n仅退出本地监看；云端按原预算继续。用同文件 --watch 接回。',flush=True)
    except BaseException as exc:
        save_report(dict(status='CLIENT_RETURN_INCOMPLETE_PRESERVE',error=repr(exc),cloud_training_state='UNKNOWN',traceback=traceback.format_exc()[-5000:]))
        print('已保留现场。只读接回：python3 '+shlex.quote(str(Path(__file__).resolve()))+' --watch',flush=True)
        return 2
    return 0

if __name__=='__main__':
    result=main();raise SystemExit(result if isinstance(result,int) else 0)
