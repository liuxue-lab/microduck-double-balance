"""Local preparation and bounded LAN ROS test; all writes live in new stage paths."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import traceback
import uuid

import common as old
from deploy_contract import (DOMAIN, MODEL_SHA, PC_IP, ORIN_IP, REPORT_SHA, require,
                             sha, validate_report, verify_export, wired_xml)
from probe import probe, readiness, shell_argv, clean_env
import shared_ssh

HERE = Path(__file__).resolve().parent
ROOT = old.BASE / 'artifacts/double-balance-stage12/native-ros2-humble-v1'
shared_ssh.HOST = 'jetson@192.168.50.2'
shared_ssh.PORT = 22


def save(path, value):
    old.write(path, value)


def protected_mkdir(path):
    path = Path(path)
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        require(not current.is_symlink(), 'Unexpected directory symlink: ' + str(current))
    path.mkdir(parents=True, exist_ok=True)


def run_logged(argv, log, timeout, env, label):
    """Own one process group; bounded wait and cancellation, no detached worker."""
    with Path(log).open('xb') as stream:
        p = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=env,
                             cwd=old.PROJECT, start_new_session=True)
        started = last = time.monotonic()
        try:
            while p.poll() is None:
                now = time.monotonic()
                require(now - started < timeout, label + ' deadline exceeded')
                if now - last >= 5:
                    print('%s：进行中 %.0f 秒 | %s' % (label, now - started, log), flush=True)
                    last = now
                time.sleep(.2)
            require(p.returncode == 0, label + ' failed; see ' + str(log))
        finally:
            if p.poll() is None:
                # This Popen child is still unreaped, so its PID cannot be reused.
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait(timeout=5)


def remote(command, stdin=None, timeout=25):
    p = subprocess.run(shared_ssh.argv(command), input=stdin, capture_output=True, timeout=timeout)
    require(p.returncode == 0, 'Jetson command failed: ' + p.stderr.decode('utf-8', 'replace')[-2500:])
    return p.stdout


def remote_json(script, args=(), stdin=None):
    cmd = shlex.join(['/usr/bin/python3', '-B', '-c', script, *map(str, args)])
    return json.loads(remote(cmd, stdin=stdin))


def ros_link(run, export_dir, pc, jetson):
    session = uuid.uuid4().hex
    destination = '/home/jetson/microduck-stage12/ros2-humble-v1/runs/' + session
    create = '''import json,pathlib,sys
p=pathlib.Path(sys.argv[1]);cur=pathlib.Path('/')
for part in p.parts[1:]:
 cur=cur/part
 if cur.is_symlink():raise RuntimeError('Unexpected remote directory symlink')
p.parent.mkdir(parents=True,exist_ok=True);p.mkdir(mode=0o700)
print(json.dumps({'path':str(p)}))
'''
    remote_json(create, [destination])
    files = [export_dir / n for n in ('final6h_005304.onnx', 'golden.json', 'contract.json', 'export-report.json')]
    files += [HERE / n for n in ('ros_check.py', 'deploy_contract.py')]
    orin_xml = run / 'orin-fastdds.xml'; orin_xml.write_text(wired_xml(ORIN_IP))
    pc_xml = run / 'pc-fastdds.xml'; pc_xml.write_text(wired_xml(PC_IP))
    files.append(orin_xml)
    control = shared_ssh.connect()
    print('ROS：通过原有网线/SSH传送冻结 ONNX 和本次检查代码', flush=True)
    subprocess.run(['scp', '-P', '22', '-o', 'ControlPath=' + str(control), '-o', 'ControlMaster=no',
                    '-o', 'BatchMode=yes', '-o', 'ProxyCommand=false', *map(str, files),
                    'jetson@192.168.50.2:' + destination + '/'], check=True, timeout=60)
    pins = {p.name: sha(p) for p in files}
    check = '''import pathlib,json,hashlib,sys
p=pathlib.Path(sys.argv[1]);pins=json.load(sys.stdin)
for n,h in pins.items():
 f=p/n
 if f.is_symlink() or hashlib.sha256(f.read_bytes()).hexdigest()!=h:raise RuntimeError('Transfer SHA mismatch: '+n)
print(json.dumps({'verified':len(pins)}))
'''
    receipt = remote_json(check, [destination], json.dumps(pins).encode())
    save(run / 'jetson-transfer.json', dict(destination=destination, files=pins, **receipt))
    env = clean_env()
    env['FASTRTPS_DEFAULT_PROFILES_FILE'] = str(pc_xml)
    server_args = [destination + '/ros_check.py', 'server', '--folder', destination,
                   '--session', session, '--report', destination + '/server-report.json']
    server_cmd = shlex.join(['timeout', '--signal=TERM', '--kill-after=5s', '90s',
                            'env', 'PYTHONPATH=', 'PYTHONHOME=', 'LD_LIBRARY_PATH=',
                            'AMENT_PREFIX_PATH=', 'CMAKE_PREFIX_PATH=', 'COLCON_PREFIX_PATH=',
                            'ROS_DISCOVERY_SERVER=', 'FASTDDS_DEFAULT_PROFILES_FILE=',
                            'RMW_FASTRTPS_USE_QOS_FROM_XML=0', 'ROS_DOMAIN_ID=62', 'ROS_LOCALHOST_ONLY=0',
                            'RMW_IMPLEMENTATION=rmw_fastrtps_cpp',
                            'FASTRTPS_DEFAULT_PROFILES_FILE=' + destination + '/orin-fastdds.xml',
                            *shell_argv(jetson['python'], server_args)])
    with (run / 'jetson-server.log').open('xb') as stream:
        process = subprocess.Popen(shared_ssh.argv(server_cmd), stdout=stream, stderr=subprocess.STDOUT)
        try:
            run_logged(shell_argv(pc['python'], [HERE / 'ros_check.py', 'client', '--folder', export_dir,
                       '--session', session, '--report', run / 'ros-link-report.json']),
                       run / 'ros-client.log', 65, env, 'ROS2 推理链路检查')
            process.wait(timeout=5)
            require(process.returncode == 0, 'Jetson server did not exit cleanly')
            server = remote_json('import json,sys;print(open(sys.argv[1]).read())',
                                 [destination + '/server-report.json'])
            save(run / 'jetson-server-report.json', server)
            require(server['status'] == 'LINK_SERVER_COMPLETE' and server['completed_samples'] == 128,
                    'Jetson completion report differs')
        finally:
            if process.poll() is None:
                try:
                    remote_json("import pathlib,json,sys;pathlib.Path(sys.argv[1]).touch(exist_ok=False);print('{}')",
                                [destination + '/cancel'])
                    process.wait(timeout=5)
                except Exception:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill(); process.wait(timeout=5)
                    print('Jetson 退出尚未确认；远端测试有独立 90 秒上限。', flush=True)
    return json.loads((run / 'ros-link-report.json').read_bytes())


def execute(probe_only=False):
    protected_mkdir(ROOT)
    with (ROOT / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = ROOT / 'runs' / (time.strftime('%Y%m%dT%H%M%SZ', time.gmtime()) + '-' + uuid.uuid4().hex[:8])
        protected_mkdir(run)
        report = dict(schema='stage12-humble-preparation-v1', status='RUNNING', started_utc=time.time(),
                      directory=str(run), ros_distribution='humble', ros_domain_id=DOMAIN,
                      pc_address=PC_IP, jetson_address=ORIN_IP, ssh_user='jetson',
                      checkpoint_sha256=MODEL_SHA, validation_report_sha256=REPORT_SHA,
                      added_PPO_updates=0, cloud_connections=0, network_configuration_changes=0,
                      package_installations=0, hardware_acceptance=False, realtime_acceptance=False,
                      plant_simulation_run=False, motor_io=False, errors=[], blockers=[])
        def update(**values):
            report.update(values)
            save(run / 'report.json', report); save(ROOT / 'report.json', report)
        update(phase='read_only_preflight')
        try:
            package = old.verify_package()
            inputs = old.verify_inputs()
            validate_report(old.SLOT / 'report.json')
            pc = probe(); save(run / 'pc-preflight.json', pc)
            pc_ready = readiness(pc)
            jetson = jetson_ready = None
            print('检查原有 Orin 连接 jetson@192.168.50.2；只读环境信息', flush=True)
            try:
                jetson = json.loads(remote('/usr/bin/python3 -B -', (HERE / 'probe.py').read_bytes(), timeout=75))
                save(run / 'jetson-preflight.json', jetson)
                jetson_ready = readiness(jetson, jetson=True)
            except Exception as exc:
                jetson_ready = dict(ready=False, python=None, blockers=['Jetson read-only probe failed: ' + repr(exc)])
            blockers = pc_ready['blockers'] + jetson_ready['blockers']
            update(package_identity=package, frozen_inputs=inputs, pc_readiness=pc_ready,
                   jetson_readiness=jetson_ready, blockers=blockers)
            if probe_only:
                update(status='PREFLIGHT_READY' if not blockers else 'PREFLIGHT_BLOCKED', phase='finished')
                return
            update(phase='freeze_and_export')
            export_dir = ROOT / 'frozen-export'
            if export_dir.exists():
                export_receipt = verify_export(export_dir)
                print('复用已通过 SHA256 与一致性检查的导出模型', flush=True)
            else:
                staging = run / 'export-staging'; staging.mkdir()
                run_logged([str(old.PYTHON), '-B', '-u', str(HERE / 'export_final.py'), '--output', str(staging)],
                           run / 'export.log', 1200, old.clean_environment(), '最终模型导出')
                export_receipt = verify_export(staging)
                staging.rename(export_dir)
            require(old.verify_inputs() == inputs and sha(old.SLOT / 'report.json') == REPORT_SHA,
                    'Original frozen evidence changed')
            update(export_directory=str(export_dir), export=export_receipt,
                   original_training_records_unchanged=True)
            if blockers:
                update(status='EXPORT_PASSED_ENVIRONMENT_PENDING', phase='finished')
                print('ONNX 已导出；ROS2 环境待补齐，具体项目已写入报告。', flush=True)
                return
            update(phase='ros2_inference_link')
            link = ros_link(run, export_dir, pc_ready, jetson_ready)
            require(old.verify_inputs() == inputs and sha(old.SLOT / 'report.json') == REPORT_SHA,
                    'Original frozen input changed')
            verify_export(export_dir)
            update(status='EXPORT_AND_ROS_LINK_PASSED', phase='finished', ros_link=link,
                   original_training_records_unchanged=True,
                   next_step='PC simulated plant to Orin inference closed loop; real-time deadline audit still pending')
        except BaseException as exc:
            update(status='PREPARATION_INTERRUPTED' if isinstance(exc, KeyboardInterrupt) else 'PREPARATION_FAILED_PRESERVE',
                   phase='finished', errors=[repr(exc)], traceback=traceback.format_exc()[-10000:])
            if not isinstance(exc, KeyboardInterrupt):
                print('停止并保留现场：' + str(exc), flush=True)
        finally:
            update(finished_utc=time.time())
            print('HUMBLE_STATUS=' + report['status'], flush=True)
            print('HUMBLE_REPORT=' + str(ROOT / 'report.json'), flush=True)
            for blocker in report['blockers']:
                print('待处理：' + blocker, flush=True)


def main():
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Received signal ' + str(signum))
    signal.signal(signal.SIGTERM, interrupted)
    signal.signal(signal.SIGHUP, interrupted)
    p = argparse.ArgumentParser()
    p.add_argument('operation', choices=('prepare', 'probe', 'status'))
    args = p.parse_args()
    if args.operation == 'status':
        print((ROOT / 'report.json').read_text() if (ROOT / 'report.json').exists() else '尚未运行')
        return 0
    execute(probe_only=args.operation == 'probe')
    r = json.loads((ROOT / 'report.json').read_bytes())
    return 0 if r['status'] in ('EXPORT_AND_ROS_LINK_PASSED', 'EXPORT_PASSED_ENVIRONMENT_PENDING',
                                'PREFLIGHT_READY', 'PREFLIGHT_BLOCKED') else 2


if __name__ == '__main__':
    raise SystemExit(main())
