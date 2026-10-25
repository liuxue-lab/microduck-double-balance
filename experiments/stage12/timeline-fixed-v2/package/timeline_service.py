"""Permanent per-profiler CUPTI loader with bounded original project validation."""
import argparse
import base64
import datetime
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback
import uuid
import zipfile
import zlib

ROOT = Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-timeline-fixed-v2')
PROJECT = Path('/home/lx/microduck-double-balance/workspace')
PYTHON = PROJECT / '.venv/bin/python'
PRIOR = ROOT.parent / 'native-pc-cupti-probe-v1/report.json'
TIMELINE_PRIOR = ROOT.parent / 'native-pc-timeline-v1/report.json'
PRIOR_SHA = '1a2de50135766c622b3511f388f35fefc10cdf8270bec17da200b6a9d318614e'
TIMELINE_SHA = 'e9922c52e59acb75a26e67ce97d7c4a107fb7f6023643f97825ed937a85dd0fe'
OLD_LIB = PROJECT / '.venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12'
OLD_LIB_SHA = 'fccdc5fcc7c32f8cf4b3ed9b17f7f187a185a6fa0a5bcc06e3434cbd1687c808'
NEW_LIB_SHA = '2fdab19dc3fccdd4b2f5eba137aabefc68e685f6497bb38eea649866c5672a2a'
WHEEL = 'nvidia_cuda_cupti_cu12-12.9.79-py3-none-manylinux_2_25_x86_64.whl'
WHEEL_SHA = '096bcf334f13e1984ba36685ad4c1d6347db214de03dbb6eebb237b41d9d934f'
WHEEL_SOURCE = 'https://files.pythonhosted.org/packages/c1/2e/b84e32197e33f39907b455b83395a017e697c07a449a2b15fd07fc1c9981/' + WHEEL

import fixed_runtime as fixed
import probe_helper
HERE = Path(__file__).resolve().parent

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def mkdir(path):
    current = Path(path.anchor)
    for name in path.parts[1:]:
        current /= name
        require(not current.is_symlink(), 'Unexpected symlink: ' + str(current))
    path.mkdir(parents=True, exist_ok=True)

def write(path, data):
    require(not path.is_symlink(), 'Unexpected report symlink')
    tmp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with tmp.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    os.replace(tmp, path)

def candidate_environment(clean, library):
    require(not clean.get('LD_PRELOAD'), 'Unexpected existing LD_PRELOAD; preserve for review')
    env = dict(clean)
    # Keep the same SONAME libcupti.so.12; never alias a CUDA 13 library to 12.
    env['LD_PRELOAD'] = str(library)
    env['LD_LIBRARY_PATH'] = str(library.parent) + (':' + clean['LD_LIBRARY_PATH'] if clean.get('LD_LIBRARY_PATH') else '')
    return env

def clean_environment():
    env = {k: v for k, v in os.environ.items() if k not in ('PYTHONPATH', 'PYTHONHOME', 'WORLD_SIZE', 'RANK', 'LOCAL_RANK')
           and not k.startswith('MICRODUCK_')}
    require(not env.get('LD_PRELOAD'), 'Existing LD_PRELOAD differs from reviewed diagnostic')
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', CUDA_VISIBLE_DEVICES='0',
               OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1',
               WANDB_MODE='disabled', HF_HUB_OFFLINE='1')
    return env

def library_loaded(row, path, expected_sha, version):
    libs = row.get('loaded_cupti', [])
    return (len(libs) == 1 and libs[0].get('path') == str(path)
            and libs[0].get('sha256') == expected_sha and libs[0].get('api_version') == version
            and libs[0].get('return_code') == 0)

def minimal_ok(process, mode, library):
    r = process.get('result', {})
    summaries = r.get('summaries') or []
    minimum = 6 if mode == 'torch-only' else 9
    return (process.get('exit_code') == 0 and not process.get('timed_out')
            and r.get('status') == 'MINIMAL_GPU_TRACE_OBSERVED'
            and r.get('gpu_compute_ok') is True and r.get('graph_ok') is True
            and len(summaries) == 1 and summaries[0].get('cuda_kernel_events', 0) >= minimum
            and (mode == 'torch-only' or summaries[0].get('warp_probe_kernel_events', 0) >= 3)
            and library_loaded(r, library, NEW_LIB_SHA, 28))

def project_ok(report):
    comparison = report.get('comparison', {})
    if not (comparison.get('passed') is True and comparison.get('compared_steps') == 160
            and comparison.get('scalar_comparisons') == 73440 and comparison.get('values_outside_tolerance') == 0
            and all(comparison.get(k) is True for k in ('full_audit_equal', 'trial_equal', 'disturbance_equal',
                                                      'all_numeric_state_differences_zero', 'valid_bounded_arms'))):
        return False
    if not all(report.get('minimal_passes', {}).get(m) is True for m in ('torch-only', 'warp-before-profiler')):
        return False
    arms = report.get('arms', {})
    if set(arms) != {'inference-only', 'timeline'}:
        return False
    for mode, a in arms.items():
        process = report.get('processes', {}).get(mode, {})
        if (process.get('exit_code') != 0 or process.get('timed_out') or
            a.get('status') != 'TIMELINE_ARM_COMPLETE' or a.get('control_steps') != 160 or
            a.get('physics_steps') != 1600 or a.get('runtime_bindings_restored') is not True):
            return False
    c = arms['timeline'].get('cuda_timeline') or {}
    s = c.get('summary') or {}
    return (s.get('status') == 'CUDA_TIMELINE_CAPTURED' and s.get('cuda_kernel_events', 0) > 0
            and s.get('required_scope_counts_complete') is True and s.get('every_step_has_cuda_kernel') is True
            and c.get('completed_steps') == list(range(101, 109))
            and report.get('application_libraries_verified') is True
            and report.get('original_records_unchanged') is True)

def status(watch=False):
    while True:
        p = ROOT / 'report.json'
        if not p.exists():
            p = ROOT / 'fix-report.json'
        if p.is_file():
            r = json.loads(p.read_text())
            state = r.get('status')
            if state == 'RUNNING' and not controller_alive(r.get('controller')):
                state = 'CONTROLLER_EXITED_PRESERVE'
            print('%s | %s | %s' % (state, r.get('phase'), p), flush=True)
        else:
            print('尚无本轮报告：' + str(p), flush=True)
        if not watch:
            return 0 if p.is_file() and r.get('status') in ('TIMELINE_FIX_INSTALLED_VERIFIED', 'PC_TIMELINE_FIXED_MEASURED') else 2
        time.sleep(2)


def controller_alive(identity):
    if not identity or not isinstance(identity.get('pid'), int) or identity['pid'] <= 1:
        return False
    now = probe_helper.process_identity(identity['pid'])
    return bool(now and now.get('start_ticks') == identity.get('start_ticks') and now.get('state') != 'Z')


def stop_active():
    for name in ('report.json', 'fix-report.json'):
        path = ROOT / name
        if not path.is_file():
            continue
        row = json.loads(fixed.regular(path).read_bytes())
        if row.get('status') != 'RUNNING' or not controller_alive(row.get('controller')):
            continue
        folder = Path(row['directory'])
        require(folder.parent in (ROOT / 'runs', ROOT / 'finalize-runs') and not folder.is_symlink(), 'Unexpected active run')
        write(folder / 'STOP.json', {'reason': 'user_requested_stop', 'utc': time.time()})
        os.kill(row['controller']['pid'], signal.SIGTERM)
        print('已请求本入口停止并保留结果。', flush=True)
        return 0
    print('没有本入口所属的活动任务；未终止其他进程。', flush=True)
    return 0

def run(retry=False):
    mkdir(ROOT)
    locks = []
    try:
        for path in (ROOT / 'operation.lock', ROOT.parent / 'native-ros2-humble-operation.lock'):
            require(not path.is_symlink(), 'Unexpected operation lock symlink')
            lock = path.open('a')
            locks.append(lock)
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('另一个本机诊断/仿真任务仍在运行；保留停止。')
        if (ROOT / 'report.json').exists() and not retry:
            print('已有记录，不自动重复执行。', flush=True)
            return status()
        return execute()
    finally:
        for lock in reversed(locks):
            lock.close()

def execute():
    folder = ROOT / 'runs' / (datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    mkdir(folder)
    r = dict(schema='microduck-stage12-timeline-fixed-v2', status='RUNNING', phase='verify',
             started_utc=time.time(), directory=str(folder), script_sha256=sha(Path(__file__)),
             candidate_version='12.9.79', candidate_library_sha256=NEW_LIB_SHA,
             NVIDIA_wheel_sha256=WHEEL_SHA, NVIDIA_wheel_source=WHEEL_SOURCE,
             reviewed_probe_sha256=PRIOR_SHA, reviewed_timeline_sha256=TIMELINE_SHA,
             original_cupti_version='12.8.90', original_cupti_sha256=OLD_LIB_SHA,
             candidate_loading='fixed verified CUPTI 12.9.79; profiler child only',
             original_environment_modified=False, project_files_modified=False,
             installation_performed=False, production_activated=False,
             cloud_connections=0, orin_connections=0, added_PPO_updates=0, motor_io=False,
             physical_robot_in_scope=False, real_robot_demonstration_required=False,
             cloud_power='OFF_USER_CONFIRMED', diagnostic_only=True,
             maximum_new_control_steps=320, maximum_new_physics_steps=3200,
             full_loop_realtime_acceptance=False, profiling_repair_verified=False,
             processes={}, probes={}, arms={}, minimal_passes={}, errors=[],
             protocol_references=['docs/handoffs/microduck-local-ssh-push-protocol.md',
                                  'docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md'])
    def save(**values):
        r.update(values)
        write(folder / 'report.json', r)
        write(ROOT / 'report.json', r)
    save()
    helper = clean = candidate = assets = library = None
    before = None
    def child(name, command, limit, env):
        save(phase=name)
        print('PC_TIMELINE_FIXED_PHASE=' + name, flush=True)
        result = helper.run_child(command, folder / name, limit, env)
        r['processes'][name] = result
        save()
        return result
    try:
        require(sys.platform == 'linux' and PYTHON.is_file(), 'Use the existing Linux project environment')
        require(sha(PRIOR) == PRIOR_SHA and sha(TIMELINE_PRIOR) == TIMELINE_SHA, 'Reviewed reports changed; preserve')
        prior = json.loads(PRIOR.read_text())
        require(prior['status'] == 'PC_CUPTI_PROBE_COLLECTED', 'Expected the reviewed completed probe')
        require(sha(OLD_LIB) == OLD_LIB_SHA, 'Installed CUPTI changed since diagnosis')
        require(shutil.disk_usage(ROOT).free > 512 * 1024**2, 'Need 512 MiB free; no files deleted')
        clean = clean_environment()
        receipt = fixed.verify_installed()
        assets = HERE
        library = fixed.LIBRARY
        helper = probe_helper
        r['minimal_passes'] = receipt['minimal_passes']
        r['minimal_validation_source'] = str(ROOT / 'fix-report.json')
        r['reviewed_repair_sha256'] = fixed.REVIEW_SHA
        r['nvidia_driver'] = helper.read_text('/proc/driver/nvidia/version', 8000)
        r['profiling_permission_setting'] = [line for line in helper.read_text('/proc/driver/nvidia/params', 24000).splitlines()
                                             if 'ProfilingAdminOnly' in line]
        r['controller'] = helper.process_identity(os.getpid())
        r['git_head'] = helper.capture_command(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'])
        r['candidate_library'] = str(library)
        candidate = candidate_environment(clean, library)
        # CPU-only inspection has no candidate preload and does not load a model.
        pre = folder / 'before'
        before_process = child('before', [str(PYTHON), '-B', str(assets / 'repair_entry.py'),
                              '--mode', 'inspect', '--folder', str(pre)], 60, clean)
        require(before_process.get('exit_code') == 0, 'Original inputs did not verify')
        before = before_process['result']
        r['before_inputs'] = before
        for mode in ('inference-only', 'timeline'):
            env = clean if mode == 'inference-only' else candidate
            proc = child(mode, [str(PYTHON), '-B', str(assets / 'repair_entry.py'), '--mode', mode,
                               '--folder', str(folder / mode), '--run', str(folder)], 230, env)
            path = folder / mode / 'worker-report.json'
            if path.is_file():
                r['arms'][mode] = json.loads(path.read_text())
            save()
            require(proc.get('exit_code') == 0, 'Project arm failed: ' + mode)
        r['application_libraries_verified'] = (
            library_loaded(r['processes']['inference-only']['result'], OLD_LIB, OLD_LIB_SHA, 26) and
            library_loaded(r['processes']['timeline']['result'], library, NEW_LIB_SHA, 28))
        require(r['application_libraries_verified'], 'Application used an unexpected or mixed CUPTI library')
        parity = child('compare', [str(PYTHON), '-B', str(assets / 'repair_entry.py'), '--mode', 'compare',
                                 '--folder', str(folder / 'compare'), '--run', str(folder)], 30, clean)
        r['comparison'] = parity.get('result', {})
        require(parity.get('exit_code') == 0 and r['comparison'].get('passed'), 'State/audit/force parity failed')
        save(status='PC_TIMELINE_FIXED_PENDING_FINAL_CHECKS')
    except BaseException as exc:
        r['errors'].append(repr(exc))
        save(status='PC_TIMELINE_FIXED_FAILED_PRESERVE', traceback=traceback.format_exc()[-12000:])
    finally:
        write(folder / 'STOP.json', {'reason': 'repair controller finalization'})
        # run_child kills and waits for only the subprocess group it created.
        for name in ('before', 'probe-torch-only', 'probe-warp-before-profiler', 'inference-only', 'timeline', 'compare'):
            path = folder / name / 'process.json'
            if path.is_file():
                r['processes'][name] = json.loads(path.read_text())
        for mode in ('inference-only', 'timeline'):
            path = folder / mode / 'worker-report.json'
            if path.is_file():
                r['arms'][mode] = json.loads(path.read_text())
        if before is not None:
            try:
                after = child('after', [str(PYTHON), '-B', str(assets / 'repair_entry.py'), '--mode', 'inspect',
                                       '--folder', str(folder / 'after')], 60, clean)
                unchanged = (after.get('exit_code') == 0 and after.get('result') == before
                             and sha(OLD_LIB) == OLD_LIB_SHA and sha(PRIOR) == PRIOR_SHA
                             and sha(TIMELINE_PRIOR) == TIMELINE_SHA)
                r['original_records_unchanged'] = unchanged
                require(unchanged, 'Original inputs, installed CUPTI or historical reports changed')
            except BaseException as exc:
                r['errors'].append(repr(exc))
        r['profiling_repair_verified'] = not r['errors'] and project_ok(r)
        r['status'] = 'PC_TIMELINE_FIXED_MEASURED' if r['profiling_repair_verified'] else 'PC_TIMELINE_FIXED_FAILED_PRESERVE'
        r['recommendation'] = dict(choice='KEEP_INFERENCE_ONLY', production_activated=False,
            full_loop_realtime_acceptance=False,
            next_step='CUPTI issue fixed; performance remains a separate unprofiled optimization and 60-second HIL acceptance task' if r['profiling_repair_verified']
                      else 'Review preserved errors; do not promote CPU-only or partial traces to a pass')
        save(phase='finished', finished_utc=time.time())
        print('PC_TIMELINE_FIXED_STATUS=' + r['status'], flush=True)
        print('PC_TIMELINE_FIXED_REPORT=' + str(ROOT / 'report.json'), flush=True)
        print('新的时间线已留存；成功无需为 CUPTI 错误重复上传。性能优化时使用本报告。', flush=True)
    return 0 if r['profiling_repair_verified'] else 2

def main():
    p = argparse.ArgumentParser(description=__doc__)
    a = p.add_mutually_exclusive_group(required=True)
    a.add_argument('--run', action='store_true')
    a.add_argument('--retry-run', action='store_true')
    a.add_argument('--status', action='store_true')
    a.add_argument('--watch', action='store_true')
    a.add_argument('--stop', action='store_true')
    args = p.parse_args()
    def stop(*_):
        raise KeyboardInterrupt('User interrupted repair trial')
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, stop)
    try:
        if args.stop:
            return stop_active()
        if args.status or args.watch:
            return status(args.watch)
        return run(args.retry_run)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print('PC_TIMELINE_FIXED_PRESERVED=' + repr(exc), flush=True)
        return 2

if __name__ == '__main__':
    raise SystemExit(main())
