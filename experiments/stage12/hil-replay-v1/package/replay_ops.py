"""Foreground replay supervisor with bounded workers and preserved attempts."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
import uuid
from replay_common import (HERE, MAX_SECONDS, PYTHON, ROOT, SOURCE_RUN, SOURCE_SHA,
    alive, input_path, need, package_identity, process_identity, read, sha, verify_source, write)


def stop_process(process):
    if process is None or process.poll() is not None:
        return
    # These processes always have their own session; never target a shared group.
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        process.wait(timeout=3)
        return
    try:
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=3)


def clean_environment():
    removed = ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'AMENT_PREFIX_PATH',
               'CMAKE_PREFIX_PATH', 'COLCON_PREFIX_PATH', 'ROS_DISTRO', 'ROS_VERSION', 'ROS_PYTHON_VERSION')
    env = {key:value for key,value in os.environ.items() if key not in removed}
    env.update(PYTHONNOUSERSITE='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def candidates(env):
    selected = env.get('MUJOCO_GL')
    if selected:
        need(selected in ('egl', 'glfw', 'osmesa'), 'Unsupported explicit MUJOCO_GL: '+selected)
        return [selected]
    return ['egl']+(['glfw'] if env.get('DISPLAY') else [])+['osmesa']


def status():
    if not (ROOT/'report.json').exists():
        print('HIL_REPLAY_STATUS=NOT_STARTED', flush=True)
        return None
    report = read(ROOT/'report.json')
    state = report['status']
    if state == 'REPLAY_RUNNING' and not alive(report.get('controller')):
        state = 'INTERRUPTED_PRESERVE'
    print('HIL_REPLAY_STATUS='+state, flush=True)
    print('HIL_REPLAY_REPORT='+str(ROOT/'report.json'), flush=True)
    progress = Path(report['directory'])/'progress.json'
    if progress.is_file() and state == 'REPLAY_RUNNING':
        row = read(progress)
        print('回放生成：%s，%s / 6000 帧；阶段 %s。'%(row.get('case_id', ''),
              row.get('completed_frames', 0), row.get('phase')), flush=True)
    if report.get('index'):
        print('HIL_REPLAY_PAGE='+report['index'], flush=True)
    if report.get('errors'):
        print('待处理：'+'; '.join(report['errors']), flush=True)
    return report


def run(retry=False):
    ROOT.mkdir(parents=True, exist_ok=True)
    need(not ROOT.is_symlink(), 'Replay output symlink')
    with (ROOT/'controller.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('已有回放生成进程，未重复启动。', flush=True)
            status()
            return 0
        if (ROOT/'report.json').exists():
            previous = read(ROOT/'report.json')
            need(not alive(previous.get('worker')), 'A previous owned render worker is still alive; use --stop')
            if not retry:
                status()
                print('保留已有结果；如需重新生成视频，显式使用 --retry-run。不会重跑HIL。', flush=True)
                return 0 if previous['status'] == 'HIL_REPLAY_COMPLETE' else 2
        output = ROOT/'runs'/(time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())+'-'+uuid.uuid4().hex[:8])
        output.mkdir(parents=True)
        started = dict(started_utc=time.time(), deadline_utc=time.time()+MAX_SECONDS)
        write(output/'started.json', started)
        report = dict(schema='stage12-hil-replay-supervisor-v1', status='REPLAY_RUNNING',
            directory=str(output), controller=process_identity(os.getpid()), source_report_sha256=SOURCE_SHA,
            source_run=str(SOURCE_RUN), package_identity=package_identity(HERE),
            new_control_ticks=0, new_physics_steps=0, policy_inferences=0, Orin_requests=0,
            PPO_updates=0, cloud_connections=0, motor_io=False, errors=[], backend_attempts=[])
        process = None
        logs = []
        def update(**values):
            report.update(values)
            write(output/'report.json', report)
            write(ROOT/'report.json', report)
        def stop(*_):
            raise KeyboardInterrupt('Replay controller interrupted')
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            signal.signal(sig, stop)
        def wait(limit):
            deadline = min(time.monotonic()+limit, time.monotonic()+max(0, started['deadline_utc']-time.time()))
            previous = None
            while process.poll() is None:
                need(time.monotonic() < deadline, 'Replay worker time limit reached')
                need(not (output/'STOP.json').exists(), 'Replay stop requested')
                progress_path = output/'progress.json'
                if progress_path.exists():
                    row = read(progress_path)
                    stamp = (row.get('phase'), row.get('completed_frames', 0)//250)
                    if stamp != previous:
                        print('回放生成 %s/6000 帧（%.1f%%）；%s。'%(row.get('completed_frames', 0),
                            row.get('completed_frames', 0)/60, row.get('phase')), flush=True)
                        previous = stamp
                time.sleep(.2)
            return process.returncode
        try:
            update(phase='verify_source')
            need(PYTHON.is_file(), 'Existing project Python is missing: '+str(PYTHON))
            source = verify_source()
            env = clean_environment()
            selected = None
            for backend in candidates(env):
                output_log = output/('probe-'+backend+'.log')
                log = output_log.open('xb'); logs.append(log)
                env['MUJOCO_GL'] = backend
                process = subprocess.Popen([str(PYTHON), '-B', '-u', str(HERE/'replay_render.py'), 'probe',
                    str(input_path(source, 'case-01/native-model.mjb'))], env=env,
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                update(phase='probe_render_backend', worker=process_identity(process.pid))
                try:
                    code = wait(45)
                except RuntimeError as exc:
                    if (output/'STOP.json').exists() or time.time() >= started['deadline_utc']:
                        raise
                    stop_process(process)
                    code = process.returncode
                log.flush()
                tail = output_log.read_text(errors='replace')[-5000:]
                report['backend_attempts'].append(dict(backend=backend, returncode=code, log_tail=tail))
                if code == 0:
                    selected = backend
                    break
                print('绘图后端 %s 不可用，检查下一个已安装后端。'%backend, flush=True)
            need(selected is not None, 'No working existing render backend; see backend_attempts in report')
            env['MUJOCO_GL'] = selected
            log = (output/'render.log').open('xb'); logs.append(log)
            process = subprocess.Popen([str(PYTHON), '-B', '-u', str(HERE/'replay_render.py'), 'render', str(output)],
                env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            update(phase='render_saved_states', backend=selected, worker=process_identity(process.pid))
            code = wait(MAX_SECONDS)
            need((output/'result.json').is_file(), 'Renderer exited without result.json')
            result = read(output/'result.json')
            report['result'] = result
            need(code == 0 and result['status'] == 'HIL_REPLAY_COMPLETE', result.get('error', 'Replay failed'))
            need(result['completed_frames'] == 6000 and result['original_inputs_unchanged'] is True,
                 'Replay completeness/preservation missing')
            update(status='HIL_REPLAY_COMPLETE', phase='finished', index=result['index'], worker_exit_code=code)
        except BaseException as exc:
            report.update(status='HIL_REPLAY_FAILED_PRESERVE', phase='stopping_owned_worker',
                          errors=[repr(exc)], traceback=traceback.format_exc()[-9000:])
        finally:
            # Cleanup cannot be interrupted by repeated Ctrl+C before its evidence is saved.
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
                signal.signal(sig, signal.SIG_IGN)
            stop_process(process)
            for log in logs:
                log.close()
            report['worker_exit_code'] = process.poll() if process is not None else None
            report['owned_worker_exited'] = process is None or process.poll() is not None
            report['log_tails'] = {p.name:p.read_text(errors='replace')[-10000:] for p in output.glob('*.log')}
            report['finished_utc'] = time.time()
            report['phase'] = 'finished'
            if (output/'result.json').is_file():
                report['result'] = read(output/'result.json')
            report['outputs'] = {p.name:dict(bytes=p.stat().st_size, sha256=sha(p))
                for p in output.iterdir() if p.suffix in ('.mp4', '.png', '.html') and p.is_file()}
            update()
        status()
        return 0 if report['status'] == 'HIL_REPLAY_COMPLETE' else 2


def stop():
    report = status()
    if not report:
        return 0
    controller, worker = report.get('controller'), report.get('worker')
    if alive(controller):
        write(Path(report['directory'])/'STOP.json', dict(reason='explicit_user_stop', utc=time.time()))
        try:
            os.kill(controller['pid'], signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif alive(worker):
        need(worker['group'] == worker['pid'], 'Unexpected owned worker process group')
        try:
            os.killpg(worker['pid'], signal.SIGTERM)
        except ProcessLookupError:
            pass
    else:
        print('本轮进程已退出，数据保留。', flush=True)
    return 0


def main():
    package_identity(HERE)
    operation = sys.argv[1]
    if operation in ('run', 'retry-run'):
        return run(retry=operation == 'retry-run')
    if operation == 'status':
        status(); return 0
    if operation == 'watch':
        while True:
            status(); time.sleep(2)
    if operation == 'stop':
        return stop()
    if operation == 'open':
        report = status()
        need(report and report['status'] == 'HIL_REPLAY_COMPLETE', 'Complete replay page not ready')
        return subprocess.call(['xdg-open', report['index']])
    raise RuntimeError('Unknown replay operation')


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception as exc:
        print('HIL_REPLAY_PRESERVED='+repr(exc), flush=True)
        raise SystemExit(2)
