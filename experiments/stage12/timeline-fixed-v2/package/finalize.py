"""Finalize an already validated repair, then test the stable launch path once."""
import argparse
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import sys
import time
import traceback
import uuid

import fixed_runtime as fixed
import probe_helper as helper
import timeline_service as service

HERE = Path(__file__).resolve().parent


def apply():
    fixed.mkdir(fixed.ROOT)
    lock_path = fixed.ROOT.parent / 'native-ros2-humble-operation.lock'
    fixed.require(not lock_path.is_symlink(), 'Unexpected operation lock link')
    with lock_path.open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('已有本机任务运行；未启动重复验证。')
        prior = fixed.ROOT / 'fix-report.json'
        if prior.exists():
            row = json.loads(fixed.regular(prior).read_bytes())
            if row.get('status') == 'TIMELINE_FIX_INSTALLED_VERIFIED':
                fixed.verify_installed()
                print('TIMELINE_FIX_INSTALLED_VERIFIED：已固化，不重复 GPU 验证。', flush=True)
                return 0
            raise RuntimeError('上次固化未成功；已保留记录，不自动重复。报告：' + str(prior))
        return execute()


def execute():
    folder = fixed.ROOT / 'finalize-runs' / (datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    fixed.mkdir(folder)
    row = dict(schema='microduck-stage12-timeline-fix-finalization-v2', status='RUNNING', phase='verify',
               directory=str(folder), controller=helper.process_identity(os.getpid()), started_utc=time.time(),
               reviewed_report_sha256=fixed.REVIEW_SHA, fixed_library_sha256=fixed.NEW_SHA,
               fixed_library=str(fixed.LIBRARY), original_environment_modified=False,
               historical_reports_modified=False, model_modified=False, new_training_steps=0,
               new_project_physics_steps=0, full_loop_realtime_acceptance=False,
               cloud_power='OFF_USER_CONFIRMED', cloud_connections=0, orin_connections=0,
               processes={}, minimal_passes={}, errors=[])
    def save(**changes):
        row.update(changes)
        fixed.write(folder / 'report.json', row)
        fixed.write(fixed.ROOT / 'fix-report.json', row)
    def child(name, command, limit, env):
        save(phase=name)
        print('TIMELINE_FIX_PHASE=' + name, flush=True)
        result = helper.run_child(command, folder / name, limit, env)
        row['processes'][name] = result
        save()
        fixed.require(result.get('exit_code') == 0 and not result.get('timed_out'), 'Validation failed: ' + name)
        return result
    before = clean = None
    save()
    try:
        evidence = fixed.install_runtime()
        row['reviewed_project_proof'] = dict(
            status=evidence['status'], cuda_kernel_events=35522, captured_steps=list(range(101, 109)),
            compared_steps=160, compared_scalars=73440, all_differences_zero=True,
            source='Reviewed RTX 5060 run; project simulation is not repeated by --apply',
            baseline_environment_step_ms=evidence['arms']['inference-only']['full_environment_step_ms'])
        clean = service.clean_environment()
        env = fixed.fixed_environment(clean)
        before = child('before', [str(service.PYTHON), '-B', str(HERE / 'repair_entry.py'),
                       '--mode', 'inspect', '--folder', str(folder / 'before')], 60, clean)['result']
        # GPU smoke tests exercise the final directory, not the previous run directory.
        for mode in ('torch-only', 'warp-before-profiler'):
            name = 'probe-' + mode
            proc = child(name, [str(service.PYTHON), '-B', str(HERE / 'probe_helper.py'),
                         '--worker', mode, '--folder', str(folder / name)], 90, env)
            row['minimal_passes'][mode] = service.minimal_ok(proc, mode, fixed.LIBRARY)
            save()
            fixed.require(row['minimal_passes'][mode], 'Fixed-path GPU trace missing, mixed or incomplete: ' + mode)
        fixed.verify_libraries(fixed.LIB_DIR)
        fixed.reviewed(fixed.EVIDENCE)
    except BaseException as exc:
        row['errors'].append(repr(exc))
        row['traceback'] = traceback.format_exc()[-12000:]
    finally:
        if before is not None:
            try:
                after = child('after', [str(service.PYTHON), '-B', str(HERE / 'repair_entry.py'),
                              '--mode', 'inspect', '--folder', str(folder / 'after')], 60, clean)
                fixed.require(after['result'] == before, 'Frozen project inputs changed')
                fixed.require(fixed.sha(fixed.OLD_LIBRARY) == fixed.OLD_SHA, 'Original CUPTI changed')
                fixed.require(fixed.sha(fixed.SOURCE_REPORT) == fixed.REVIEW_SHA, 'Successful source report changed')
                row['original_inputs_unchanged'] = True
            except BaseException as exc:
                row['errors'].append(repr(exc))
        for name in ('before', 'probe-torch-only', 'probe-warp-before-profiler', 'after'):
            path = folder / name / 'process.json'
            if path.is_file():
                row['processes'][name] = json.loads(path.read_bytes())
        passed = (not row['errors'] and row.get('original_inputs_unchanged') is True and
                  row['minimal_passes'] == {'torch-only': True, 'warp-before-profiler': True})
        save(status='TIMELINE_FIX_INSTALLED_VERIFIED' if passed else 'TIMELINE_FIX_FAILED_PRESERVE',
             phase='finished', finished_utc=time.time())
        print('TIMELINE_FIX_STATUS=' + row['status'], flush=True)
        print('TIMELINE_FIX_REPORT=' + str(fixed.ROOT / 'fix-report.json'), flush=True)
        if passed:
            print('固定修复已生效。无需为同一个 CUPTI 错误重复上传；旧报告保留。', flush=True)
        else:
            print('固化未完成，报告包含原因；未回退到旧 CUPTI，也未标记通过。', flush=True)
    return 0 if passed else 2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true', required=True)
    parser.parse_args()
    def stop(*_):
        raise KeyboardInterrupt('User interrupted fixed-path validation')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        return apply()
    except Exception as exc:
        print('TIMELINE_FIX_PRESERVED=' + repr(exc), flush=True)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
