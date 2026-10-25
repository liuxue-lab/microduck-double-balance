"""PC-only bounded unprofiled baseline and CPU/CUDA timeline diagnostics. No SSH, package install, or PPO."""
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
import timeline_contract as contract
from deploy_contract import require,sha,REPORT_SHA
from loop_contract import frozen

HERE=Path(__file__).resolve().parent
ROOT=old.BASE/'artifacts/double-balance-stage12/native-pc-timeline-v1'
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


def recommendation(baseline,candidate,parity):
    summary=(candidate.get('cuda_timeline') or {}).get('summary') or {}
    return dict(choice='KEEP_INFERENCE_ONLY',diagnostic_only=True,
        baseline_environment_step_ms=baseline['full_environment_step_ms'],
        timeline_capture_status=summary.get('status'),
        numerical_parity_passed=parity['passed'],production_activated=False,
        full_loop_realtime_acceptance=False,hardware_acceptance=False,
        physical_robot_in_scope=False,real_robot_demonstration_required=False,
        scope='Profiled timings include overhead; no speedup ratio or adoption decision',
        next_step='Review kernel, memory-copy, launch and synchronization traces before choosing an optimization; then real-time ROS2/Orin HIL acceptance')


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
        report=dict(schema='stage12-pc-timeline-v1',status='RUNNING',directory=str(run),
            controller=old.identity(os.getpid()),started_utc=time.time(),added_PPO_updates=0,
            cloud_connections=0,orin_connections=0,motor_io=False,automatic_retry=False,
            maximum_new_control_steps=contract.STEPS*2,maximum_new_physics_steps=contract.STEPS*20,
            hardware_acceptance=False,full_loop_realtime_acceptance=False,production_activated=False,
            cprofile_enabled=False,all_500Hz_checks_retained=True,
            diagnostic_only=True,physical_robot_in_scope=False,real_robot_demonstration_required=False,
            cloud_power=dict(state='OFF_USER_CONFIRMED',evidence='用户消息：5090已关机',
                confirmation_date='2026-10-24',timezone='Asia/Shanghai',platform_verified=False,exact_shutdown_time=None),
            arms={},errors=[])
        before=None;process=None;streams=[]
        def update(**values):
            report.update(values);old.write(run/'report.json',report);old.write(ROOT/'report.json',report)
        def boundary():require(not (run/'STOP.json').exists(),'Stop requested; preserve partial results')
        try:
            update(phase='verify_accepted_inputs',package_identity=old.verify_package())
            prior=contract.reviewed();report['reviewed_auditfast_v1_report_sha256']=contract.REPORT_SHA
            report['prior_auditfast_v1_status']=prior['status']
            before=old.verify_inputs();frozen(EXPORT)
            require(sha(old.SLOT/'report.json')==REPORT_SHA,'Original validation report changed')
            require(shutil.disk_usage(ROOT).free>=512*1024**2,'Less than 512 MiB free; no files deleted')
            for mode in ('inference-only','timeline'):
                boundary();folder=run/mode;mkdir(folder)
                log=run/(mode+'.log');stream=log.open('xb');streams.append(stream)
                update(phase=mode,active_directory=str(folder),active_log=str(log))
                process=subprocess.Popen([str(old.PYTHON),'-B','-u',str(HERE/'timeline_worker.py'),
                    '--folder',str(folder),'--mode',mode,'--cancel-file',str(run/'STOP.json')],
                    cwd=old.PROJECT,env=old.clean_environment(),stdout=stream,stderr=subprocess.STDOUT,
                    stdin=subprocess.DEVNULL,start_new_session=True)
                update(owned_worker=old.identity(process.pid));begun=last=time.monotonic()
                with log.open(errors='replace') as view:
                    while process.poll() is None:
                        boundary();require(time.monotonic()-begun<230,'PC worker wall-clock limit reached')
                        for line in view.readlines():
                            if line.startswith('PC_TIMELINE_'):print(line.rstrip(),flush=True);last=time.monotonic()
                        if time.monotonic()-last>=5:
                            print('%s：进行中 %.0f 秒 | %s' % (mode,time.monotonic()-begun,log),flush=True);last=time.monotonic()
                        time.sleep(.2)
                    for line in view.readlines():
                        if line.startswith('PC_TIMELINE_'):print(line.rstrip(),flush=True)
                if process.returncode!=0:
                    failed=old.read(folder/'worker-report.json') if (folder/'worker-report.json').is_file() else {}
                    detail=failed.get('episode_result',{}).get('error') or failed.get('status') or 'worker exited'
                    raise RuntimeError(mode+' 未完成：'+str(detail)+'；日志 '+str(log))
                row=old.read(folder/'worker-report.json')
                report['arms'][mode]=row
                require(row['status']=='TIMELINE_ARM_COMPLETE' and contract.valid_partial(row['episode_result'])
                        and row['runtime_bindings_restored'],'Invalid/incomplete bounded arm')
                update()
            boundary();update(phase='compare_paired_states')
            parity=contract.compare_traces(run/'inference-only/state-trace.jsonl',run/'timeline/state-trace.jsonl')
            parity.update(contract.compare_audits(report['arms']['inference-only'],report['arms']['timeline']))
            parity['passed']=parity['passed'] and parity['full_audit_equal'] and parity['trial_equal'] and parity['disturbance_equal']
            parity['all_numeric_state_differences_zero']=all(v==0 for v in parity['max_absolute_error_by_field'].values())
            parity['passed']=parity['passed'] and parity['all_numeric_state_differences_zero']
            report['comparison']=parity
            report['recommendation']=recommendation(report['arms']['inference-only'],report['arms']['timeline'],parity)
            update(status='PC_TIMELINE_MEASURED' if parity['passed'] else 'PC_TIMELINE_PARITY_FAILED_PRESERVE')
        except BaseException as exc:
            update(status='PC_TIMELINE_FAILED_PRESERVE',errors=[repr(exc)],traceback=traceback.format_exc()[-12000:])
        finally:
            old.write(run/'STOP.json',dict(reason='controller_finalization',utc=time.time()))
            stop_owned(process)
            for stream in streams:stream.close()
            for mode in ('inference-only','timeline'):
                path=run/mode/'worker-report.json'
                if path.is_file():report['arms'][mode]=old.read(path)
            if before is not None:
                try:
                    report['original_records_unchanged']=old.verify_inputs()==before and sha(old.SLOT/'report.json')==REPORT_SHA
                    contract.reviewed();frozen(EXPORT)
                    require(report['original_records_unchanged'],'Frozen source/model/records changed')
                except Exception as exc:
                    report['status']='PC_TIMELINE_FAILED_PRESERVE';report['errors'].append(repr(exc))
            report['worker_exited']=process is None or process.poll() is not None
            report['last_worker_exit_code']=None if process is None else process.returncode
            report['log_tails']={p.name:p.read_text(errors='replace')[-16000:] for p in run.glob('*.log')}
            report['trace_files']={str(p.relative_to(run)):dict(bytes=p.stat().st_size,sha256=sha(p))
                                   for p in run.glob('*/state-trace.jsonl')}
            update(phase='finished',finished_utc=time.time())
            print('PC_TIMELINE_STATUS='+report['status'],flush=True)
            print('PC_TIMELINE_REPORT='+str(ROOT/'report.json'),flush=True)
            if report['errors']:print('待处理：'+'; '.join(report['errors']),flush=True)
        return 0 if report['status']=='PC_TIMELINE_MEASURED' else 2


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
            if not watch:print('PC_TIMELINE_REPORT='+str(path),flush=True)
            code=0 if status=='PC_TIMELINE_MEASURED' else 2
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
        print('PC_TIMELINE_PRESERVED='+repr(exc),flush=True);raise SystemExit(2)
