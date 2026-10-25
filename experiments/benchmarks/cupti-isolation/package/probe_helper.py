#!/usr/bin/env python3
"""Bounded, offline CUPTI diagnostic. No project edits, model loads or simulation.

Run with system Python; workers use the existing project's Python 3.12 venv.
All three probes are fresh processes. This is a diagnostic, not a repair or
performance acceptance. CUDA events measure the tiny test only, not Microduck.
"""
import argparse
import collections
import ctypes
import datetime
import fcntl
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import traceback
import uuid

SCHEMA = 'microduck-stage12-cupti-probe-v1'
PROJECT = Path('/home/lx/microduck-double-balance/workspace')
DEFAULT_OUTPUT = Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage12/native-pc-cupti-probe-v1')
MODES = ('torch-only', 'warp-before-profiler', 'profiler-before-warp')
PACKAGES = ('torch', 'nvidia-cuda-cupti-cu12', 'nvidia-cuda-cupti-cu13',
            'nvidia-cuda-runtime-cu12', 'nvidia-cuda-runtime-cu13',
            'nvidia-cudnn-cu12', 'nvidia-cudnn-cu13', 'triton', 'warp-lang',
            'mjlab', 'mujoco', 'mujoco-warp')
ENV_KEYS = ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER', 'CUDA_MODULE_LOADING',
            'CUDA_INJECTION64_PATH', 'CUPTI_INJECTION64_PATH', 'LD_LIBRARY_PATH',
            'LD_PRELOAD', 'PYTHONPATH', 'KINETO_LOG_LEVEL', 'CUDA_LAUNCH_BLOCKING')
SOURCES = [
    'https://docs.pytorch.org/docs/2.9/profiler.html',
    'https://docs.pytorch.org/docs/2.9/generated/torch.cuda.Event.html',
    'https://docs.nvidia.com/cupti/13.1.0/main/main.html',
    'https://docs.nvidia.com/cupti/12.9/api/group__CUPTI__VERSION__API.html',
    'https://nvidia.github.io/warp/v1.12/api_reference/warp.html',
]
WARP_SOURCE = '''import warp as wp
@wp.kernel
def probe_increment(x: wp.array(dtype=wp.float32)):
    i = wp.tid()
    x[i] = x[i] + 1.0
'''


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write('\n')
    os.replace(temporary, path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_text(path, limit=24000):
    try:
        with Path(path).open('rb') as f:
            try:
                size = f.seek(0, 2)
                f.seek(max(0, size - limit))
            except OSError:
                # procfs files need not support SEEK_END; read them sequentially.
                f.close()
                with Path(path).open('rb') as sequential:
                    return sequential.read(limit).decode('utf-8', errors='replace')
            return f.read(limit).decode('utf-8', errors='replace')
    except OSError as exc:
        return repr(exc)


def loaded_libraries(text=None):
    """Record actual mapped paths, never guess a library from its filename."""
    if text is None:
        text = Path('/proc/self/maps').read_text()
    paths = set()
    for line in text.splitlines():
        fields = line.split(None, 5)
        if len(fields) == 6 and fields[5].startswith('/'):
            p = fields[5]
            if any(s in Path(p).name.lower() for s in (
                'cupti', 'libcuda', 'libnvrtc', 'libnvjitlink', 'libkineto', 'libtorch_cuda', 'warp.so')):
                paths.add(p)
    return sorted(paths)


def metadata_inventory():
    versions, candidates = {}, set()
    for name in PACKAGES:
        try:
            dist = importlib.metadata.distribution(name)
            versions[name] = {'version': dist.version, 'location': str(dist.locate_file(''))}
            for entry in dist.files or ():
                if 'libcupti.so' in Path(str(entry)).name:
                    p = Path(dist.locate_file(entry))
                    candidates.add(str(p))
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    # Bounded, known CUDA installation patterns; no recursive filesystem scan.
    for pattern in ('cuda*/extras/CUPTI/lib64/libcupti.so*',
                    'cuda*/targets/x86_64-linux/lib/libcupti.so*'):
        candidates.update(str(p) for p in Path('/usr/local').glob(pattern))
    for directory in os.environ.get('LD_LIBRARY_PATH', '').split(':'):
        if directory and Path(directory).is_dir():
            candidates.update(str(p) for p in Path(directory).glob('libcupti.so*'))
    rows = []
    for name in sorted(candidates)[:60]:
        p = Path(name)
        try:
            rows.append({'path': str(p), 'resolved_path': str(p.resolve()), 'bytes': p.stat().st_size})
        except OSError as exc:
            rows.append({'path': str(p), 'error': repr(exc)})
    return {'python': sys.version, 'executable': sys.executable, 'prefix': sys.prefix,
            'packages': versions, 'cupti_candidates': rows,
            'environment_allowlist': {k: os.environ[k] for k in ENV_KEYS if k in os.environ},
            'candidate_paths_are_not_proof_of_loading': True}


def loaded_cupti_versions():
    """Query already loaded handles only, after the profiling test has ended."""
    rows = []
    for name in loaded_libraries():
        if 'libcupti.so' not in Path(name).name:
            continue
        row = {'path': name}
        try:
            # RTLD_NOLOAD prevents loading another competing CUPTI version.
            lib = ctypes.CDLL(name, mode=os.RTLD_NOLOAD | os.RTLD_LOCAL | os.RTLD_NOW)
            fn = lib.cuptiGetVersion
            fn.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
            fn.restype = ctypes.c_int
            value = ctypes.c_uint32()
            row.update(return_code=int(fn(ctypes.byref(value))), api_version=int(value.value),
                       sha256=digest(name))
        except Exception as exc:
            row['error'] = repr(exc)
        rows.append(row)
    return rows


def trace_summary(trace):
    counts = collections.Counter()
    kernels = []
    for event in trace.get('traceEvents', []):
        if not isinstance(event, dict):
            continue
        category = str(event.get('cat', ''))
        counts[category] += 1
        if event.get('ph') == 'X' and category in ('kernel', 'cuda_kernel'):
            kernels.append(event)
    return {'categories': dict(counts), 'cuda_kernel_events': len(kernels),
            'warp_probe_kernel_events': sum('probe_increment' in str(e.get('name', '')) for e in kernels),
            'kernel_names': sorted({str(e.get('name', '')) for e in kernels})[:30],
            'cuda_runtime_events': counts['cuda_runtime'],
            'cuda_driver_events': counts['cuda_driver'],
            'device_properties': trace.get('deviceProperties', []),
            'missing_gpu_activity_is_not_zero_gpu_time': True}


def probe_result(compute_ok, graph_ok, summaries):
    if not compute_ok:
        return 'GPU_COMPUTE_FAILED_PRESERVE'
    if not graph_ok:
        return 'GRAPH_TEST_FAILED_PRESERVE'
    if len(summaries) != 1 or summaries[0].get('cuda_kernel_events', 0) <= 0:
        return 'GPU_TRACE_MISSING_PRESERVE'
    return 'MINIMAL_GPU_TRACE_OBSERVED'


def gpu_worker(mode, folder):
    folder = Path(folder)
    row = {'mode': mode, 'started': stamp(), 'status': 'STARTED', 'stages': [],
           'snapshots': {}, 'summaries': [], 'gpu_compute_ok': False, 'graph_ok': False,
           'size_float32_elements': 4096, 'uses_microduck_model': False}
    output = folder / 'worker.json'

    def checkpoint(stage):
        row['stages'].append(stage)
        row['snapshots'][stage] = loaded_libraries()
        write_json(output, row)
        print('PROBE_STAGE=' + stage, flush=True)

    profiler = None
    try:
        checkpoint('before_import')
        import torch
        row['torch'] = {'version': torch.__version__, 'path': torch.__file__,
                        'built_cuda': torch.version.cuda,
                        'cuda_available': torch.cuda.is_available()}
        checkpoint('torch_imported')
        if not torch.cuda.is_available():
            row['status'] = 'CUDA_UNAVAILABLE_PRESERVE'
            return 2
        torch.cuda.set_device(0)
        row['devices'] = [
            {'ordinal': i, 'name': torch.cuda.get_device_name(i),
             'capability': list(torch.cuda.get_device_capability(i)),
             'total_memory': torch.cuda.get_device_properties(i).total_memory}
            for i in range(torch.cuda.device_count())]
        row['torch']['compiled_arch_list'] = torch.cuda.get_arch_list()
        x = torch.zeros(4096, device='cuda:0', dtype=torch.float32)
        for _ in range(3):
            x.add_(1.0)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(4):
            x.add_(1.0)
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end))
        row['cuda_event_test'] = {'elapsed_ms': elapsed, 'valid': math.isfinite(elapsed) and elapsed >= 0,
                                  'scope': 'four small tensor adds only; not project performance'}
        row['gpu_compute_ok'] = bool(torch.all(x == 7.0).item())
        checkpoint('torch_cuda_and_events_tested')

        def start_profiler():
            api = torch.profiler
            row['supported_activities'] = sorted(str(x) for x in api.supported_activities())
            p = api.profile(activities=[api.ProfilerActivity.CPU, api.ProfilerActivity.CUDA],
                            record_shapes=False, profile_memory=False, with_stack=False)
            p.start()
            return p

        if mode == 'profiler-before-warp':
            profiler = start_profiler()
            checkpoint('profiler_started_before_warp')
        wp = kernel = array = stream = None
        if mode != 'torch-only':
            import warp as wp
            # Confine this diagnostic's JIT cache to its own output directory.
            wp.config.kernel_cache_dir = str(folder / 'warp-cache')
            wp.init()
            source = folder / 'probe_warp_kernel.py'
            source.write_text(WARP_SOURCE)
            spec = importlib.util.spec_from_file_location('microduck_cupti_test_kernel', source)
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
            kernel = module.probe_increment
            stream = wp.stream_from_torch(torch.cuda.current_stream(0))
            array = wp.from_torch(x)
            row['warp'] = {'version': wp.__version__, 'path': wp.__file__,
                           'device': str(array.device), 'same_torch_stream': True}
            wp.launch(kernel, dim=4096, inputs=[array], stream=stream)
            torch.cuda.synchronize()
            row['warp']['warmup_correct'] = bool(torch.all(x == 8.0).item())
            row['gpu_compute_ok'] &= row['warp']['warmup_correct']
            checkpoint('warp_kernel_warmed')

        # Exercise both ordinary launches and replay, since Microduck uses graphs.
        # Prepare outside the measured markers. Separate xg avoids shared state.
        xg = torch.zeros_like(x)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            xg.add_(2.0)
        graph.replay()
        torch.cuda.synchronize()
        row['graph_ok'] = bool(torch.all(xg == 2.0).item())
        checkpoint('graph_prepared')
        if profiler is None:
            profiler = start_profiler()
            checkpoint('profiler_started_after_warmup')
        initial = 8.0 if wp is not None else 7.0
        for i in range(3):
            with torch.profiler.record_function('md12_probe_torch_and_graph_%d' % i):
                x.add_(1.0)
                graph.replay()
            if wp is not None:
                with torch.profiler.record_function('md12_probe_warp_%d' % i):
                    wp.launch(kernel, dim=4096, inputs=[array], stream=stream)
            torch.cuda.synchronize()
        checkpoint('minimal_work_finished')
        p, profiler = profiler, None  # No automatic retry if stop fails.
        p.stop()
        trace_path = folder / 'trace.json'
        p.export_chrome_trace(str(trace_path))
        row['summaries'].append(trace_summary(json.loads(trace_path.read_text())))
        row['warp_kernel_seen_in_trace'] = (wp is None or
            row['summaries'][0]['warp_probe_kernel_events'] > 0)
        row['trace'] = {'path': str(trace_path), 'bytes': trace_path.stat().st_size,
                        'sha256': digest(trace_path)}
        row['gpu_compute_ok'] &= bool(torch.all(x == initial + (6 if wp else 3)).item())
        row['graph_ok'] &= bool(torch.all(xg == 8.0).item())
        row['status'] = probe_result(row['gpu_compute_ok'], row['graph_ok'], row['summaries'])
        if row['status'] == 'MINIMAL_GPU_TRACE_OBSERVED' and not row['warp_kernel_seen_in_trace']:
            row['status'] = 'WARP_GPU_TRACE_MISSING_PRESERVE'
        checkpoint('profiler_stopped_and_trace_saved')
    except Exception as exc:
        row.update(status='WORKER_ERROR_PRESERVE', error=repr(exc), traceback=traceback.format_exc())
    finally:
        # Persist partial results before querying native libraries (which can crash).
        write_json(output, row)
        if profiler is not None:
            try:
                profiler.stop()
            except Exception as exc:
                row['cleanup_error'] = repr(exc)
        row['snapshots']['at_exit'] = loaded_libraries()
        write_json(output, row)
        row['loaded_cupti'] = loaded_cupti_versions()
        row['finished'] = stamp()
        write_json(output, row)
        print('PROBE_RESULT=' + row['status'], flush=True)
    return 0 if row['status'] == 'MINIMAL_GPU_TRACE_OBSERVED' else 2


def capture_command(command, timeout=5):
    try:
        done = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=timeout, text=True, errors='replace')
        return {'command': command, 'exit_code': done.returncode, 'output': done.stdout[-16000:]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {'command': command, 'error': repr(exc)}


def run_child(command, folder, seconds, env=None):
    """Fresh process group with bounded runtime; collect even partial/crashed output."""
    folder = Path(folder)
    folder.mkdir()
    begun = time.monotonic()
    logfile = folder / 'worker.log'
    proc = None
    outcome = {'command': command, 'timeout_seconds': seconds, 'timed_out': False}
    try:
        with logfile.open('xb') as log:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                    stderr=subprocess.STDOUT, start_new_session=True, env=env)
            outcome['owned_worker_pid'] = proc.pid
            last = begun
            while proc.poll() is None:
                elapsed = time.monotonic() - begun
                if elapsed >= seconds:
                    outcome['timed_out'] = True
                    break
                if time.monotonic() - last >= 5:
                    print('仍在检查 %s，已用 %.0f 秒（上限 %.0f 秒）' % (folder.name, elapsed, seconds), flush=True)
                    last = time.monotonic()
                try:
                    proc.wait(timeout=min(1.0, max(.01, seconds - elapsed)))
                except subprocess.TimeoutExpired:
                    pass
    finally:
        if proc is not None and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=2)
        outcome['exit_code'] = None if proc is None else proc.returncode
        outcome['elapsed_seconds'] = time.monotonic() - begun
        outcome['log_tail'] = read_text(logfile)
        result = folder / 'worker.json'
        if result.is_file():
            try:
                outcome['result'] = json.loads(result.read_text())
            except Exception as exc:
                outcome['result_parse_error'] = repr(exc)
        write_json(folder / 'process.json', outcome)
    return outcome


def assess(probes):
    observed = {m: probes.get(m, {}).get('result', {}).get('status') == 'MINIMAL_GPU_TRACE_OBSERVED'
                and probes.get(m, {}).get('exit_code') == 0 for m in MODES}
    logs = '\n'.join(str(x.get('log_tail', '')) for x in probes.values())
    codes = sorted(set(re.findall(r'CUPTI_ERROR_[A-Z_]+', logs)))
    # Explicitly avoid treating the generic privileges hint as an actual error.
    actual = sorted(set(re.findall(r'(?:with error|error:?)\s+(CUPTI_ERROR_[A-Z_]+)', logs)))
    if all(observed.values()):
        conclusion = 'MINIMAL_PROBES_PASS_PROJECT_TIMELINE_STILL_UNVERIFIED'
    elif observed['torch-only'] and not observed['warp-before-profiler']:
        conclusion = ('EARLY_PROFILER_WORKAROUND_CANDIDATE_NEEDS_PROJECT_VALIDATION'
                      if observed['profiler-before-warp'] else 'MIXED_RUNTIME_FAILURE_NEEDS_REVIEW')
    elif not observed['torch-only']:
        conclusion = 'BASE_TORCH_PROBE_FAILED_REVIEW_VERSIONS_MAPS_AND_ERRORS'
    else:
        conclusion = 'PARTIAL_PROBE_RESULTS_NEED_REVIEW'
    return {'observed_gpu_trace_by_case': observed, 'conclusion': conclusion,
            'actual_cupti_error_codes': actual, 'all_cupti_code_mentions': codes,
            'automatic_repair': False, 'project_timeline_fixed': False,
            'full_loop_realtime_acceptance': False,
            'note': 'Minimal probes isolate failure conditions; not a project-level pass or unique root-cause proof.'}


def process_identity(pid):
    try:
        fields = Path('/proc/%d/stat' % pid).read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'state': fields[0], 'start_ticks': fields[19]}
    except (OSError, IndexError):
        return None


def parent(args):
    root = Path(args.output).absolute()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / 'probe.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print('已有诊断正在运行，请用 --status 查看。')
        lock.close()
        return 2
    latest = root / 'report.json'
    if latest.exists() and not args.retry_run:
        print('已有报告，保留且不重复执行：' + str(latest))
        show_status(root)
        lock.close()
        return 0
    run = root / 'runs' / (datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
    run.mkdir(parents=True)
    script = Path(__file__).resolve()
    interpreter = Path(args.python or PROJECT / '.venv/bin/python')
    report = {'schema': SCHEMA, 'status': 'RUNNING', 'started': stamp(), 'directory': str(run),
              'controller': process_identity(os.getpid()),
              'script_sha256': digest(script), 'project': str(PROJECT), 'python_requested': str(interpreter),
              'probes': {}, 'phase': 'inventory', 'diagnostic_only': True,
              'added_PPO_updates': 0, 'physics_steps': 0, 'control_steps': 0,
              'cloud_connections': 0, 'orin_connections': 0, 'motor_io': False,
              'production_activated': False, 'project_files_modified': False,
              'dependency_installation': False, 'cloud_power': 'OFF_USER_CONFIRMED',
              'official_api_references': SOURCES,
              'protocols': ['docs/handoffs/microduck-local-ssh-push-protocol.md',
                            'docs/handoffs/microduck-cloud-gpu-and-transfer-protocol.md']}

    def save():
        write_json(run / 'report.json', report)
        write_json(latest, report)

    save()
    deadline = time.monotonic() + 300
    try:
        print('仅进行小型 GPU 诊断；保留现有环境。总时间上限约 5 分钟。', flush=True)
        report['nvidia_smi'] = capture_command(['nvidia-smi', '--query-gpu=index,name,driver_version,compute_cap', '--format=csv,noheader'])
        report['nsys_version'] = capture_command(['nsys', '--version']) if shutil.which('nsys') else {'installed_on_PATH': False}
        report['nvidia_driver'] = read_text('/proc/driver/nvidia/version', 4000)
        params = read_text('/proc/driver/nvidia/params', 20000)
        report['profiling_permission_setting'] = [l for l in params.splitlines() if 'ProfilingAdminOnly' in l]
        if PROJECT.is_dir():
            report['git_head'] = capture_command(['git', '-C', str(PROJECT), 'rev-parse', 'HEAD'])
            report['git_branch'] = capture_command(['git', '-C', str(PROJECT), 'branch', '--show-current'])
        prior = DEFAULT_OUTPUT.parent / 'native-pc-timeline-v1/report.json'
        if prior.is_file():
            report['prior_timeline_report_sha256'] = digest(prior)
        if not interpreter.is_file():
            raise FileNotFoundError('项目 Python 不存在：' + str(interpreter))
        worker_env = os.environ.copy()
        worker_env['PYTHONDONTWRITEBYTECODE'] = '1'
        worker_env['PYTHONUNBUFFERED'] = '1'
        # Preserve CUDA visibility and library paths; do not silently "fix" them.
        inv = run / 'inventory'
        report['inventory'] = run_child([str(interpreter), str(script), '--worker', 'inventory', '--folder', str(inv)], inv, 30, worker_env)
        save()
        for index, mode in enumerate(MODES, 1):
            report['phase'] = mode
            save()
            print('[%d/3] %s' % (index, mode), flush=True)
            folder = run / mode
            remaining = deadline - time.monotonic()
            if remaining <= 3:
                raise TimeoutError('总诊断时间上限到达')
            report['probes'][mode] = run_child(
                [str(interpreter), str(script), '--worker', mode, '--folder', str(folder)],
                folder, min(75 if mode == 'torch-only' else 90, remaining - 2), worker_env)
            save()
        report['assessment'] = assess(report['probes'])
        report['status'] = 'PC_CUPTI_PROBE_COLLECTED'
        if (report['inventory'].get('exit_code') != 0 or any(
                p.get('timed_out') or 'result' not in p or p.get('exit_code') not in (0, 2)
                for p in [report['inventory'], *report['probes'].values()])):
            report['status'] = 'PC_CUPTI_PROBE_PARTIAL_PRESERVE'
    except KeyboardInterrupt:
        report['status'] = 'PC_CUPTI_PROBE_INTERRUPTED_PRESERVE'
    except Exception as exc:
        report.update(status='PC_CUPTI_PROBE_PARTIAL_PRESERVE', error=repr(exc), traceback=traceback.format_exc())
    finally:
        # Collect snapshots even when a worker was interrupted or its native code crashed.
        for mode in MODES:
            p = run / mode / 'process.json'
            if mode not in report['probes'] and p.is_file():
                report['probes'][mode] = json.loads(p.read_text())
        report['assessment'] = assess(report['probes'])
        report.update(finished=stamp(), phase='finished')
        save()
        print('PC_CUPTI_PROBE_STATUS=' + report['status'], flush=True)
        print('PC_CUPTI_PROBE_REPORT=' + str(latest), flush=True)
        print('请只上传上面的 report.json；采集结束不代表故障已修复或实时性能通过。', flush=True)
        lock.close()
    return 0 if report['status'] == 'PC_CUPTI_PROBE_COLLECTED' else 2


def show_status(root):
    path = Path(root) / 'report.json'
    if not path.is_file():
        print('尚无报告：' + str(path))
        return
    try:
        row = json.loads(path.read_text())
        if row.get('status') == 'RUNNING':
            saved = row.get('controller') or {}
            live = process_identity(saved.get('pid', -1))
            if not live or live['state'] == 'Z' or live['start_ticks'] != saved.get('start_ticks'):
                print('控制进程已退出；下面是保留的旧进度，不能视为仍在运行。', flush=True)
        print('状态=%s；阶段=%s；已收集=%d/3；报告=%s' % (
            row.get('status'), row.get('phase'), len(row.get('probes', {})), path), flush=True)
    except Exception as exc:
        print('报告读取失败：' + repr(exc))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--run', action='store_true')
    actions.add_argument('--retry-run', action='store_true', help='显式新建一轮诊断；旧报告留在 runs 中')
    actions.add_argument('--status', action='store_true')
    actions.add_argument('--watch', action='store_true')
    actions.add_argument('--worker', choices=('inventory',) + MODES, help=argparse.SUPPRESS)
    parser.add_argument('--folder', help=argparse.SUPPRESS)
    parser.add_argument('--output', default=str(DEFAULT_OUTPUT))
    parser.add_argument('--python', help='仅覆盖工作进程解释器，默认使用项目 .venv/bin/python')
    args = parser.parse_args()
    if args.worker:
        if not args.folder or not Path(args.folder).is_dir():
            parser.error('internal worker folder missing')
        if args.worker == 'inventory':
            write_json(Path(args.folder) / 'worker.json', metadata_inventory())
            return 0
        return gpu_worker(args.worker, args.folder)
    if args.status:
        show_status(args.output)
        return 0
    if args.watch:
        try:
            while True:
                show_status(args.output)
                time.sleep(2)
        except KeyboardInterrupt:
            return 0
    # SIGTERM also preserves results and stops only the current owned worker.
    def interrupted(*_):
        raise KeyboardInterrupt()
    signal.signal(signal.SIGTERM, interrupted)
    return parent(args)


if __name__ == '__main__':
    raise SystemExit(main())
