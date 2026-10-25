"""One bounded run compares both synchronization changes and their combination."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
import uuid

from bootstrap import load_base, require, sha, STAGE
old = load_base()
import action_contract as contract
import timeline_contract as previous
from deploy_contract import REPORT_SHA as VALIDATION_SHA
from loop_contract import frozen

HERE = Path(__file__).resolve().parent
ROOT = STAGE / 'native-pc-actiongraph-v1'
FIXED = STAGE / 'native-pc-timeline-fixed-v2'
REPAIR_SHA = '1c12ea0d8602dfc3aeb83b9662936b5145a1ef659f936ae1e52e8d1fc2f8e114'
CUPTI_SHA = '2fdab19dc3fccdd4b2f5eba137aabefc68e685f6497bb38eea649866c5672a2a'
FIX_COMMIT = 'f33210ad7c474161b85e9f4cb5dc6db65f877db7'
EXPORT = STAGE / 'native-ros2-humble-v1/frozen-export'


def mkdir(path):
    for parent in (path, *path.parents):
        require(not parent.is_symlink(), 'Unexpected directory link: ' + str(parent))
    path.mkdir(parents=True, exist_ok=True)


def repair_evidence():
    receipt = FIXED / 'fix-report.json'
    evidence = FIXED / 'evidence/reviewed-repair-report.json'
    require(receipt.is_file() and not receipt.is_symlink(), 'Fixed CUPTI validation receipt missing')
    row = old.read(receipt)
    require(row.get('status') == 'TIMELINE_FIX_INSTALLED_VERIFIED' and not row.get('errors') and
            row.get('minimal_passes') == {'torch-only': True, 'warp-before-profiler': True} and
            row.get('original_inputs_unchanged') is True and row.get('reviewed_report_sha256') == REPAIR_SHA and
            row.get('fixed_library_sha256') == CUPTI_SHA, 'Fixed profiler validation incomplete')
    require(sha(evidence) == REPAIR_SHA and sha(FIXED / 'runtime/cupti-12.9.79/libcupti.so.12') == CUPTI_SHA,
            'Successful profiling evidence or fixed library changed')
    require(old.read(FIXED / 'code-installed.json')['commit'] == FIX_COMMIT, 'Unexpected profiler fix commit')
    return dict(fix_receipt_sha256=sha(receipt), reviewed_timeline_sha256=REPAIR_SHA,
                fixed_CUPTI_sha256=CUPTI_SHA, fix_commit=FIX_COMMIT)


def stop_owned(process):
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)


def run(retry=False):
    mkdir(ROOT)
    lock_path = ROOT.parent / 'native-ros2-humble-operation.lock'
    require(not lock_path.is_symlink(), 'Unexpected operation lock link')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('另一个本机任务仍在运行；不并发启动。')
        if (ROOT / 'report.json').exists() and not retry:
            print('已有记录，不自动重跑。', flush=True)
            return show()
        return execute()


def execute():
    folder = ROOT / 'runs' / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8])
    mkdir(folder)
    report = dict(schema='stage12-pc-actiongraph-v1', status='RUNNING', phase='verify',
        directory=str(folder), controller=old.identity(os.getpid()), started_utc=time.time(),
        maximum_arms=5, maximum_new_control_steps=800, maximum_new_physics_steps=8000,
        controls_per_arm=160, physics_steps_per_arm=1600, arms={}, comparisons={}, errors=[],
        added_PPO_updates=0, cloud_connections=0, orin_connections=0, motor_io=False,
        full_loop_realtime_acceptance=False, production_activated=False, hardware_acceptance=False,
        physical_robot_in_scope=False, cloud_power='OFF_USER_CONFIRMED',
        profiler_enabled=False, all_500Hz_checks_retained=True,
        accepted_pipeline='InferenceOnly + PhaseCache + BAM Graph v2 + BatchedAuditReads',
        proposed_changes=['Original action writes in one CUDA graph per physics substep',
                          'Cumulative action graph + static quaternion + exact scalar audit + packed checks'],
        original_delay_RNG_sampling=True, normal_unprofiled_timing=True)
    before = reviewed = process = None
    streams = []
    def save(**values):
        report.update(values)
        old.write(folder / 'report.json', report)
        old.write(ROOT / 'report.json', report)
    def boundary():
        require(not (folder / 'STOP.json').exists(), 'Stop requested; preserve results')
    def arm(name, mode):
        nonlocal process
        boundary()
        out = folder / name
        mkdir(out)
        log = folder / (name + '.log')
        stream = log.open('xb')
        streams.append(stream)
        env = old.clean_environment()
        require(not env.get('LD_PRELOAD'), 'Unprofiled comparison requires no inherited LD_PRELOAD')
        save(phase=name, active_directory=str(out), active_log=str(log), active_mode=mode,
             completed_arms=len(report['arms']))
        print('PC_ACTIONGRAPH_PHASE=%s | mode=%s | completed=%d | maximum=5' % (name, mode, len(report['arms'])), flush=True)
        process = subprocess.Popen([str(old.PYTHON), '-B', '-u', str(HERE / 'action_worker.py'),
            '--folder', str(out), '--mode', mode, '--cancel-file', str(folder / 'STOP.json')],
            cwd=old.PROJECT, env=env, stdin=subprocess.DEVNULL, stdout=stream,
            stderr=subprocess.STDOUT, start_new_session=True)
        save(owned_worker=old.identity(process.pid))
        begun = last = time.monotonic()
        with log.open(errors='replace') as view:
            while process.poll() is None:
                boundary()
                require(time.monotonic() - begun < 230, 'Worker wall-clock limit reached')
                for line in view.readlines():
                    if line.startswith('PC_ACTIONGRAPH_'):
                        print(line.rstrip(), flush=True)
                        last = time.monotonic()
                if time.monotonic() - last >= 5:
                    print('%s：进行中 %.0f 秒 | %s' % (name, time.monotonic() - begun, log), flush=True)
                    last = time.monotonic()
                time.sleep(.2)
            for line in view.readlines():
                if line.startswith('PC_ACTIONGRAPH_'):
                    print(line.rstrip(), flush=True)
        path = out / 'worker-report.json'
        if path.is_file():
            report['arms'][name] = old.read(path)
        save()
        require(process.returncode == 0 and name in report['arms'] and contract.arm_valid(report['arms'][name]),
                'Incomplete arm ' + name + '; log ' + str(log))
        require(report['arms'][name]['mode'] == mode, 'Worker mode differs')
        stream.close()
        return report['arms'][name]
    def compare(a, b):
        boundary()
        save(phase='compare-' + b)
        report['comparisons'][a + '__' + b] = contract.compare_folders(folder / a, folder / b)
        save()
    try:
        save(package_identity=old.read(HERE / 'manifest.json')['identity'],
             baseline_package_identity=old.verify_package())
        previous.reviewed()
        reviewed = repair_evidence()
        sync_path=STAGE/'native-pc-sync-v1/report.json'
        require(sha(sync_path)=='afe6ba189a35733df2470d8e95f4f7600070e5752eab99669e851cccdd8a167a',
                'Reviewed PC synchronization report differs')
        report['reviewed_sync_report_sha256']=sha(sync_path)
        report['reviewed_repair'] = reviewed
        before = old.verify_inputs()
        frozen(EXPORT)
        require(sha(old.SLOT / 'report.json') == VALIDATION_SHA, 'Original model validation changed')
        require(shutil.disk_usage(ROOT).free > 1024**3, 'Need 1 GiB free; no files deleted')
        for mode in contract.MODES:
            arm(mode, mode)
            if mode != 'baseline':
                compare('baseline', mode)
        selected = contract.preliminary(report['arms'])
        report['preliminary_selection'] = selected
        save()
        if selected is not None:
            # Independent fresh processes; reverse the order for the confirmation.
            arm('candidate-confirm', selected)
            compare('baseline', 'candidate-confirm')
            arm('baseline-confirm', 'baseline')
            compare('baseline', 'baseline-confirm')
            compare('baseline-confirm', 'candidate-confirm')
        report['recommendation'] = contract.recommendation(report['arms'], selected)
        save(status='PC_ACTIONGRAPH_MEASURED')
    except BaseException as exc:
        report['errors'].append(repr(exc))
        save(status='PC_ACTIONGRAPH_FAILED_PRESERVE', traceback=traceback.format_exc()[-12000:])
    finally:
        old.write(folder / 'STOP.json', dict(reason='controller_finalization', utc=time.time()))
        stop_owned(process)
        for stream in streams:
            stream.close()
        for out in folder.iterdir():
            if out.is_dir() and (out / 'worker-report.json').is_file():
                report['arms'][out.name] = old.read(out / 'worker-report.json')
        if before is not None:
            try:
                unchanged = (old.verify_inputs() == before and repair_evidence() == reviewed and
                             sha(old.SLOT / 'report.json') == VALIDATION_SHA)
                previous.reviewed()
                require(sha(sync_path)==report['reviewed_sync_report_sha256'], 'Historical synchronization report changed')
                frozen(EXPORT)
                require(unchanged, 'Frozen model, sources, profiler fix or historical reports changed')
                report['original_records_unchanged'] = True
            except BaseException as exc:
                report['errors'].append(repr(exc))
                report['status'] = 'PC_ACTIONGRAPH_FAILED_PRESERVE'
        report['worker_exited'] = process is None or process.poll() is not None
        report['last_worker_exit_code'] = None if process is None else process.returncode
        report['completed_arms'] = len(report['arms'])
        report['actual_control_steps'] = sum(a.get('control_steps', 0) for a in report['arms'].values())
        report['actual_physics_steps'] = sum(a.get('physics_steps', 0) or 0 for a in report['arms'].values())
        report['log_tails'] = {p.name: p.read_text(errors='replace')[-16000:] for p in folder.glob('*.log')}
        report['trace_files'] = {str(p.relative_to(folder)): dict(bytes=p.stat().st_size, sha256=sha(p))
                                 for p in folder.glob('*/*-trace.jsonl')}
        report['auxiliary_files'] = {str(p.relative_to(folder)): dict(bytes=p.stat().st_size, sha256=sha(p))
                                     for p in folder.glob('*/aux-state.jsonl')}
        save(phase='finished', finished_utc=time.time())
        print('PC_ACTIONGRAPH_STATUS=' + report['status'], flush=True)
        print('PC_ACTIONGRAPH_REPORT=' + str(ROOT / 'report.json'), flush=True)
        if report.get('recommendation'):
            print('PC_ACTIONGRAPH_RECOMMENDATION=' + report['recommendation']['choice'], flush=True)
        print('全部方案汇总在这一份 report.json；请上传此文件，不需要逐项发日志。', flush=True)
    return 0 if report['status'] == 'PC_ACTIONGRAPH_MEASURED' else 2


def show(watch=False):
    while True:
        path = ROOT / 'report.json'
        if not path.is_file():
            print('尚无本轮结果。', flush=True)
            code = 2
        else:
            r = old.read(path)
            status = r['status']
            if status == 'RUNNING' and not old.alive(r.get('controller')):
                status = 'CONTROLLER_EXITED_PRESERVE'
            progress_path = Path(r.get('active_directory', r['directory'])) / 'progress.json'
            p = old.read(progress_path) if progress_path.is_file() else {}
            print('%s | %s | %d/%d | %.1f%% | completed arms %d' % (status, r.get('phase', ''),
                p.get('completed', 0), p.get('total', 160), p.get('percent', 0.), r.get('completed_arms', 0)), flush=True)
            if not watch:
                print('PC_ACTIONGRAPH_REPORT=' + str(path), flush=True)
            code = 0 if status == 'PC_ACTIONGRAPH_MEASURED' else 2
        if not watch:
            return code
        time.sleep(2)


def stop():
    r = old.read(ROOT / 'report.json')
    folder = Path(r['directory'])
    require(folder.parent.resolve() == (ROOT / 'runs').resolve() and not folder.is_symlink(), 'Unexpected run path')
    require(r['status'] == 'RUNNING' and old.alive(r.get('controller')), 'No live owned task')
    old.write(folder / 'STOP.json', dict(reason='user_requested_stop', utc=time.time()))
    print('已请求停止本轮对照并保留结果。', flush=True)
    return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument('operation', choices=('run', 'retry-run', 'status', 'watch', 'stop'))
    args = p.parse_args()
    if args.operation in ('status', 'watch'):
        return show(args.operation == 'watch')
    if args.operation == 'stop':
        return stop()
    def interrupt(signum, frame):
        raise KeyboardInterrupt('Controller signal ' + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupt)
    return run(args.operation == 'retry-run')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print('PC_ACTIONGRAPH_PRESERVED=' + repr(exc), flush=True)
        raise SystemExit(2)
