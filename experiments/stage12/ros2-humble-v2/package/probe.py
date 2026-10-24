"""Read-only two-host probe. This script is also sent to SSH Python stdin."""
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys


def clean_env():
    skip = ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'CONDA_PREFIX', 'AMENT_PREFIX_PATH',
            'CMAKE_PREFIX_PATH', 'COLCON_PREFIX_PATH', 'LD_LIBRARY_PATH', 'ROS_DISTRO',
            'ROS_VERSION', 'ROS_DOMAIN_ID', 'ROS_LOCALHOST_ONLY', 'RMW_IMPLEMENTATION',
            'FASTRTPS_DEFAULT_PROFILES_FILE', 'FASTDDS_DEFAULT_PROFILES_FILE',
            'ROS_DISCOVERY_SERVER', 'RMW_FASTRTPS_USE_QOS_FROM_XML', 'CYCLONEDDS_URI')
    env = {k: v for k, v in os.environ.items() if k not in skip}
    env.update(PATH='/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin',
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               ROS_DOMAIN_ID='62', ROS_LOCALHOST_ONLY='0', RMW_IMPLEMENTATION='rmw_fastrtps_cpp')
    return env


def command(args, timeout=20):
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=timeout, env=clean_env())
        return dict(returncode=p.returncode, stdout=p.stdout[-20000:], stderr=p.stderr[-4000:])
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(returncode=-1, error=repr(exc))


def shell_argv(python, script_args):
    return ['bash', '--noprofile', '--norc', '-c',
            'source /opt/ros/humble/setup.bash && exec ' + shlex.join([str(python), '-B', *map(str, script_args)])]


def probe():
    os_release = {}
    for line in Path('/etc/os-release').read_text().splitlines():
        if '=' in line:
            k, v = line.split('=', 1); os_release[k] = v.strip('"')
    result = dict(hostname=platform.node(), architecture=platform.machine(), os=os_release,
                  system_python=command(['/usr/bin/python3', '--version']),
                  wired_addresses=command(['ip', '-j', '-4', 'address', 'show']),
                  routes=command(['ip', '-j', '-4', 'route', 'show']),
                  route_to_peer=command(['ip', '-j', '-4', 'route', 'get',
                                        '192.168.50.1' if platform.machine() == 'aarch64' else '192.168.50.2']),
                  humble_setup_present=Path('/opt/ros/humble/setup.bash').is_file(),
                  ros_candidates=[], changes_made=False)
    for path, name in [('/proc/device-tree/model', 'board_model'), ('/etc/nv_tegra_release', 'l4t')]:
        p = Path(path)
        result[name] = p.read_text().strip('\x00\n') if p.is_file() else None
    if platform.machine() == 'aarch64':
        result['power_query'] = command(['nvpmodel', '-q'])
    candidates = ['/usr/bin/python3']
    if platform.machine() == 'aarch64':
        candidates.insert(0, '/home/jetson/venvs/microduck-stage12-humble-v2/bin/python')
        candidates.append('/home/jetson/venvs/robomimic-hil/bin/python')
    code = '''import sys,json,os
r={'python':sys.executable,'version':list(sys.version_info[:3]),'ros_distro':os.environ.get('ROS_DISTRO')}
if sys.version_info[:2] != (3,10):
 r['error']='Humble binary ABI requires the existing Python 3.10';print(json.dumps(r));raise SystemExit(0)
try:
 import rclpy
 from rclpy.utilities import get_rmw_implementation_identifier
 from std_msgs.msg import String
 r.update(rclpy_file=rclpy.__file__,rmw=get_rmw_implementation_identifier(),ros_ok=True)
except Exception as e:r.update(ros_ok=False,ros_error=repr(e))
try:
 import onnxruntime as ort
 import numpy
 r.update(ort_version=ort.__version__,ort_providers=ort.get_available_providers(),numpy_version=numpy.__version__)
except Exception as e:r['ort_error']=repr(e)
print(json.dumps(r))
'''
    for candidate in candidates:
        if not Path(candidate).is_file() or not result['humble_setup_present']:
            continue
        row = command(shell_argv(candidate, ['-c', code]))
        try:
            row['details'] = json.loads(row.get('stdout', '').strip().splitlines()[-1])
        except (ValueError, IndexError):
            pass
        row['requested_python'] = candidate
        result['ros_candidates'].append(row)
    return result


def readiness(result, jetson=False):
    failures = []
    target = '192.168.50.2' if jetson else '192.168.50.1'
    iface = 'enP8p1s0' if jetson else 'enp4s0'
    if result['os'].get('ID') != 'ubuntu' or result['os'].get('VERSION_ID') != '22.04':
        failures.append('Expected the established Ubuntu 22.04; do not install Jammy packages on another OS')
    if result['architecture'] != ('aarch64' if jetson else 'x86_64'):
        failures.append('Host architecture differs')
    try:
        rows = json.loads(result['wired_addresses']['stdout'])
        if not any(r['ifname'] == iface and any(a.get('local') == target and a.get('prefixlen') == 24
                   for a in r.get('addr_info', [])) for r in rows):
            failures.append('Existing wired address/interface is not active; network settings preserved')
        route = json.loads(result['route_to_peer']['stdout'])[0]
        if route.get('dev') != iface or route.get('prefsrc') != target:
            failures.append('Route to peer is not the established wired link')
    except (KeyError, ValueError, IndexError):
        failures.append('Cannot verify existing wired address and route')
    ready = [r for r in result['ros_candidates'] if r.get('details', {}).get('ros_ok') is True
             and r['details'].get('ros_distro') == 'humble'
             and r['details'].get('rmw') == 'rmw_fastrtps_cpp'
             and (not jetson or 'CPUExecutionProvider' in r['details'].get('ort_providers', []))]
    if not ready:
        failures.append('Need an existing Humble Python 3.10 + Fast DDS environment' +
                        (' with ONNX Runtime CPU provider on Orin' if jetson else ' on PC'))
    return dict(ready=not failures, blockers=failures,
                python=ready[0]['requested_python'] if ready else None)


if __name__ == '__main__':
    print(json.dumps(probe(), ensure_ascii=False))
