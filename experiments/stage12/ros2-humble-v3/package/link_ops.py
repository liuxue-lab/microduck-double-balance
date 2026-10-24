"""Bounded ROS link operations reused from the checked v1; no export entry point."""
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
from deploy_contract import DOMAIN, MODEL_SHA, PC_IP, ORIN_IP, REPORT_SHA, require, sha, validate_report, verify_export, wired_xml
from probe import probe, readiness, shell_argv, clean_env
import shared_ssh
HERE = Path(__file__).resolve().parent
ROOT = old.BASE / 'artifacts/double-balance-stage12/native-ros2-humble-v3'
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
        p = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=env, cwd=old.PROJECT, start_new_session=True)
        started = last = time.monotonic()
        try:
            while p.poll() is None:
                now = time.monotonic()
                require(now - started < timeout, label + ' deadline exceeded')
                if now - last >= 5:
                    print('%s：进行中 %.0f 秒 | %s' % (label, now - started, log), flush=True)
                    last = now
                time.sleep(0.2)
            require(p.returncode == 0, label + ' failed; see ' + str(log))
        finally:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    p.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL)
                    p.wait(timeout=5)

def remote(command, stdin=None, timeout=25):
    p = subprocess.run(shared_ssh.argv(command), input=stdin, capture_output=True, timeout=timeout)
    require(p.returncode == 0, 'Jetson command failed: ' + p.stderr.decode('utf-8', 'replace')[-2500:])
    return p.stdout

def remote_json(script, args=(), stdin=None):
    cmd = shlex.join(['/usr/bin/python3', '-B', '-c', script, *map(str, args)])
    return json.loads(remote(cmd, stdin=stdin))

def ros_link(run, export_dir, pc, jetson):
    session = uuid.uuid4().hex
    destination = '/home/jetson/microduck-stage12/ros2-humble-v3/runs/' + session
    create = "import json,pathlib,sys\np=pathlib.Path(sys.argv[1]);cur=pathlib.Path('/')\nfor part in p.parts[1:]:\n cur=cur/part\n if cur.is_symlink():raise RuntimeError('Unexpected remote directory symlink')\np.parent.mkdir(parents=True,exist_ok=True);p.mkdir(mode=0o700)\nprint(json.dumps({'path':str(p)}))\n"
    remote_json(create, [destination])
    files = [export_dir / n for n in ('final6h_005304.onnx', 'golden.json', 'contract.json', 'export-report.json')]
    files += [HERE / n for n in ('ros_check.py', 'deploy_contract.py')]
    orin_xml = run / 'orin-fastdds.xml'
    orin_xml.write_text(wired_xml(ORIN_IP))
    pc_xml = run / 'pc-fastdds.xml'
    pc_xml.write_text(wired_xml(PC_IP))
    files.append(orin_xml)
    control = shared_ssh.connect()
    print('ROS：通过原有网线/SSH传送冻结 ONNX 和本次检查代码', flush=True)
    subprocess.run(['scp', '-P', '22', '-o', 'ControlPath=' + str(control), '-o', 'ControlMaster=no', '-o', 'BatchMode=yes', '-o', 'ProxyCommand=false', *map(str, files), 'jetson@192.168.50.2:' + destination + '/'], check=True, timeout=60)
    pins = {p.name: sha(p) for p in files}
    check = "import pathlib,json,hashlib,sys\np=pathlib.Path(sys.argv[1]);pins=json.load(sys.stdin)\nfor n,h in pins.items():\n f=p/n\n if f.is_symlink() or hashlib.sha256(f.read_bytes()).hexdigest()!=h:raise RuntimeError('Transfer SHA mismatch: '+n)\nprint(json.dumps({'verified':len(pins)}))\n"
    receipt = remote_json(check, [destination], json.dumps(pins).encode())
    save(run / 'jetson-transfer.json', dict(destination=destination, files=pins, **receipt))
    env = clean_env()
    env['FASTRTPS_DEFAULT_PROFILES_FILE'] = str(pc_xml)
    server_args = [destination + '/ros_check.py', 'server', '--folder', destination, '--session', session, '--report', destination + '/server-report.json']
    server_cmd = shlex.join(['timeout', '--signal=TERM', '--kill-after=5s', '90s', 'env', 'PYTHONPATH=', 'PYTHONHOME=', 'LD_LIBRARY_PATH=', 'AMENT_PREFIX_PATH=', 'CMAKE_PREFIX_PATH=', 'COLCON_PREFIX_PATH=', 'ROS_DISCOVERY_SERVER=', 'FASTDDS_DEFAULT_PROFILES_FILE=', 'RMW_FASTRTPS_USE_QOS_FROM_XML=0', 'ROS_DOMAIN_ID=62', 'ROS_LOCALHOST_ONLY=0', 'RMW_IMPLEMENTATION=rmw_fastrtps_cpp', 'FASTRTPS_DEFAULT_PROFILES_FILE=' + destination + '/orin-fastdds.xml', *shell_argv(jetson['python'], server_args)])
    with (run / 'jetson-server.log').open('xb') as stream:
        process = subprocess.Popen(shared_ssh.argv(server_cmd), stdout=stream, stderr=subprocess.STDOUT)
        try:
            run_logged(shell_argv(pc['python'], [HERE / 'ros_check.py', 'client', '--folder', export_dir, '--session', session, '--report', run / 'ros-link-report.json']), run / 'ros-client.log', 65, env, 'ROS2 推理链路检查')
            process.wait(timeout=5)
            require(process.returncode == 0, 'Jetson server did not exit cleanly')
            server = remote_json('import json,sys;print(open(sys.argv[1]).read())', [destination + '/server-report.json'])
            save(run / 'jetson-server-report.json', server)
            require(server['status'] == 'LINK_SERVER_COMPLETE' and server['completed_samples'] == 128, 'Jetson completion report differs')
        finally:
            if process.poll() is None:
                try:
                    remote_json("import pathlib,json,sys;pathlib.Path(sys.argv[1]).touch(exist_ok=False);print('{}')", [destination + '/cancel'])
                    process.wait(timeout=5)
                except Exception:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=5)
                    print('Jetson 退出尚未确认；远端测试有独立 90 秒上限。', flush=True)
    return json.loads((run / 'ros-link-report.json').read_bytes())
