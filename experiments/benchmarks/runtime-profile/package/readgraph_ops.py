"""PC-only bounded Inference-only/captured audit-read comparison. No SSH, package install, or PPO."""
import argparse
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
import uuid

import common as old
import readgraph_contract as contract
from deploy_contract import require,sha,REPORT_SHA
from loop_contract import frozen

HERE=Path(__file__).resolve().parent
ROOT=old.BASE/'artifacts/double-balance-stage12/native-pc-readgraph-v2'
EXPORT=old.BASE/'artifacts/double-balance-stage12/native-ros2-humble-v1/frozen-export'


def mkdir(path):
    current=Path(path.anchor)
    for part in path.parts[1:]:
        current/=part;require(not current.is_symlink(),'Unexpected directory symlink: '+str(current))
    path.mkdir(parents=True,exist_ok=True)


def stop_owned(process):
    if process is None or process.poll() is not None:return
    try:os.killpg(process.pid,signal.SIGTERM)
    except ProcessLookupError:return
    try:process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=5)


def recommendation(baseline,cached,parity):
    base=baseline['full_environment_step_ms'];fast=cached['full_environment_step_ms']
    require(base['mean']>0 and base['p95']>0 and fast['mean']>0 and fast['p95']>0,'Invalid timing sample')
    mean_gain=1-fast['mean']/base['mean'];p95_gain=1-fast['p95']/base['p95']
    faster=parity['passed'] and mean_gain>=.05 and p95_gain>=.05
    return dict(choice='READ_GRAPH_CANDIDATE' if faster else 'KEEP_INFERENCE_ONLY',
        mean_reduction_fraction=mean_gain,p95_reduction_fraction=p95_gain,
        candidate_environment_p95_below_20ms=fast['p95']<20.,production_activated=False,
        full_loop_realtime_acceptance=False,hardware_acceptance=False,
        scope='One matched pair; PC environment-step time excludes policy inference, network and comparison-trace recording',
        next_step='Review parity and phase times before any full ROS/plant test; keep accepted FP32 model')


def execute(retry=False):
    mkdir(ROOT)
    lock_path=ROOT.parent/'native-ros2-humble-operation.lock'
    require(not lock_path.is_symlink(),'Unexpected operation lock symlink')
    with lock_path.open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('另一个本机 Humble 任务仍在运行；不并发启动。')
        if (ROOT/'report.json').exists() and not retry:
            print('已有本轮记录，保留且不自动重跑；显式重跑使用 --retry-run。',flush=True)
            return show()
        run=ROOT/'runs'/(time.strftime('%Y%m%dT%H%M%SZ',time.gmtime())+'-'+uuid.uuid4().hex[:8]);mkdir(run)
        report=dict(schema='stage12-pc-readgraph-v2',status='RUNNING',directory=str(run),
            controller=old.identity(os.getpid()),started_utc=time.time(),added_PPO_updates=0,
            cloud_connections=0,orin_connections=0,motor_io=False,automatic_retry=False,
            maximum_new_control_steps=contract.STEPS*2,maximum_new_physics_steps=contract.STEPS*20,
            hardware_acceptance=False,full_loop_realtime_acceptance=False,production_activated=False,
            cloud_power=dict(state='OFF_USER_CONFIRMED',evidence='用户消息：5090已关机',
                confirmation_date='2026-10-24',timezone='Asia/Shanghai',platform_verified=False,exact_shutdown_time=None),
            arms={},errors=[])
        before=None;process=None;streams=[]
        def update(**values):
            report.update(values);old.write(run/'report.json',report);old.write(ROOT/'report.json',report)
        def boundary():require(not (run/'STOP.json').exists(),'Stop requested; preserve partial results')
        try:
            update(phase='verify_accepted_inputs',package_identity=old.verify_package())
            prior=contract.reviewed();report['reviewed_runtime_v1_report_sha256']=contract.REPORT_SHA
            report['prior_runtime_v1_status']=prior['status']
            failed=contract.reviewed_failure()
            report['reviewed_readgraph_v1_failure_sha256']=contract.FAILED_REPORT_SHA
            report['prior_readgraph_v1_status']=failed['status']
            before=old.verify_inputs();frozen(EXPORT)
            require(sha(old.SLOT/'report.json')==REPORT_SHA,'Original validation report changed')
            require(shutil.disk_usage(ROOT).free>=256*1024**2,'Less than 256 MiB free; no files deleted')
            for mode in ('inference-only','graph-reads'):
                boundary();folder=run/mode;mkdir(folder)
                log=run/(mode+'.log');stream=log.open('xb');streams.append(stream)
                update(phase=mode,active_directory=str(folder),active_log=str(log))
                process=subprocess.Popen([str(old.PYTHON),'-B','-u',str(HERE/'readgraph_worker.py'),
                    '--folder',str(folder),'--mode',mode,'--cancel-file',str(run/'STOP.json')],
                    cwd=old.PROJECT,env=old.clean_environment(),stdout=stream,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True)
                update(owned_worker=old.identity(process.pid));begun=last=time.monotonic()
                with log.open(errors='replace') as view:
                    while process.poll() is None:
                        boundary();require(time.monotonic()-begun<230,'PC worker wall-clock limit reached')
                        for line in view.readlines():
                            if line.startswith('PC_READGRAPH_'):print(line.rstrip(),flush=True);last=time.monotonic()
                        if time.monotonic()-last>=5:
                            print('%s：进行中 %.0f 秒 | %s' % (mode,time.monotonic()-begun,log),flush=True);last=time.monotonic()
                        time.sleep(.2)
                    for line in view.readlines():
                        if line.startswith('PC_READGRAPH_'):print(line.rstrip(),flush=True)
                if process.returncode!=0:
                    failed=old.read(folder/'worker-report.json') if (folder/'worker-report.json').is_file() else {}
                    detail=failed.get('episode_result',{}).get('error') or failed.get('status') or 'worker exited'
                    raise RuntimeError(mode+' 未完成：'+str(detail)+'；日志 '+str(log))
                row=old.read(folder/'worker-report.json')
                report['arms'][mode]=row
                require(row['status']=='READGRAPH_ARM_COMPLETE' and contract.valid_partial(row['episode_result'])
                        and row['runtime_bindings_restored'],'Invalid/incomplete bounded arm')
                update()
            boundary();update(phase='compare_paired_states')
            parity=contract.compare_traces(run/'inference-only/state-trace.jsonl',run/'graph-reads/state-trace.jsonl')
            parity.update(contract.compare_audits(report['arms']['inference-only'],report['arms']['graph-reads']))
            parity['passed']=parity['passed'] and parity['full_audit_equal'] and parity['trial_equal'] and parity['disturbance_equal']
            report['comparison']=parity
            report['recommendation']=recommendation(report['arms']['inference-only'],report['arms']['graph-reads'],parity)
            update(status='PC_READGRAPH_MEASURED' if parity['passed'] else 'PC_READGRAPH_PARITY_FAILED_PRESERVE')
        except BaseException as exc:
            update(status='PC_READGRAPH_FAILED_PRESERVE',errors=[repr(exc)],traceback=traceback.format_exc()[-12000:])
        finally:
            old.write(run/'STOP.json',dict(reason='controller_finalization',utc=time.time()))
            stop_owned(process)
            for stream in streams:stream.close()
            for mode in ('inference-only','graph-reads'):
                path=run/mode/'worker-report.json'
                if path.is_file():report['arms'][mode]=old.read(path)
            if before is not None:
                try:
                    report['original_records_unchanged']=old.verify_inputs()==before and sha(old.SLOT/'report.json')==REPORT_SHA
                    contract.reviewed();contract.reviewed_failure();frozen(EXPORT)
                    require(report['original_records_unchanged'],'Frozen source/model/records changed')
                except Exception as exc:
                    report['status']='PC_READGRAPH_FAILED_PRESERVE';report['errors'].append(repr(exc))
            report['worker_exited']=process is None or process.poll() is not None
            report['last_worker_exit_code']=None if process is None else process.returncode
            report['log_tails']={p.name:p.read_text(errors='replace')[-16000:] for p in run.glob('*.log')}
            report['trace_files']={str(p.relative_to(run)):dict(bytes=p.stat().st_size,sha256=sha(p))
                                   for p in run.glob('*/state-trace.jsonl')}
            update(phase='finished',finished_utc=time.time())
            print('PC_READGRAPH_STATUS='+report['status'],flush=True)
            print('PC_READGRAPH_REPORT='+str(ROOT/'report.json'),flush=True)
            if report['errors']:print('待处理：'+'; '.join(report['errors']),flush=True)
        return 0 if report['status']=='PC_READGRAPH_MEASURED' else 2


def show(watch=False):
    while True:
        path=ROOT/'report.json'
        if not path.is_file():print('尚未运行电脑端对照。',flush=True);code=2
        else:
            r=old.read(path);status=r['status']
            if status=='RUNNING' and not old.alive(r.get('controller')):status='CONTROLLER_EXITED_PRESERVE'
            progress_path=Path(r.get('active_directory',r['directory']))/'progress.json'
            p=old.read(progress_path) if progress_path.is_file() else {}
            print('%s | %s | %d/%d | %.1f%%' % (status,r.get('phase',''),p.get('completed',0),
                p.get('total',contract.STEPS),p.get('percent',0.)),flush=True)
            if not watch:print('PC_READGRAPH_REPORT='+str(path),flush=True)
            code=0 if status=='PC_READGRAPH_MEASURED' else 2
        if not watch:return code
        time.sleep(2)


def stop():
    r=old.read(ROOT/'report.json');run=Path(r['directory'])
    require(run.parent.resolve()==(ROOT/'runs').resolve() and not run.is_symlink(),'Unexpected run directory')
    require(r['status']=='RUNNING' and old.alive(r.get('controller')),'No live owned PC controller')
    old.write(run/'STOP.json',dict(reason='user_requested_stop',utc=time.time()))
    print('已请求停止本轮电脑端对照并保留结果。',flush=True);return 0


def main():
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=('run','retry-run','status','watch','stop'))
    args=parser.parse_args()
    if args.operation in ('status','watch'):return show(args.operation=='watch')
    if args.operation=='stop':return stop()
    def interrupt(signum,frame):raise KeyboardInterrupt('Controller signal '+str(signum))
    for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP):signal.signal(sig,interrupt)
    return execute(args.operation=='retry-run')


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:raise SystemExit(130)
    except Exception as exc:
        print('PC_READGRAPH_PRESERVED='+repr(exc),flush=True);raise SystemExit(2)
