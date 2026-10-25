"""Humble setup on an existing Ubuntu 22.04 host; no training or network rewrites."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback

from probe import clean_env, probe

BOOTSTRAP = 'ros2-apt-source_1.3.0.jammy_all.deb'
BOOTSTRAP_SHA = '110b9a462d55252decb8b7c816f61c2ba0d9890ce5fb93ac504e97cae5860d76'
ORT_VERSION = '1.23.2'  # 1.24.4 requires >=3.11; Humble uses the existing 3.10.
VENV = Path.home() / 'venvs/microduck-stage12-humble-v2'
OWNER = 'microduck-stage12-humble-v2-owned-environment'
GIB = 1024 ** 3


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def resources():
    info = {}
    for row in Path('/proc/meminfo').read_text().splitlines():
        key, value = row.split(':', 1)
        info[key] = int(value.strip().split()[0]) * 1024
    disks = {}
    for path in ('/', str(Path.home())):
        usage = shutil.disk_usage(path)
        vfs = os.statvfs(path)
        disks[path] = dict(total_bytes=usage.total, free_bytes=usage.free, free_inodes=vfs.f_favail)
    return dict(memory={k:info.get(k) for k in ('MemTotal','MemAvailable','SwapTotal','SwapFree')}, disks=disks)


def check_resources(r, disk_required, memory_required):
    memory_ok = (r['memory'].get('MemAvailable') or 0) >= memory_required
    disk_ok = all(d['free_bytes'] >= disk_required and d['free_inodes'] >= 10000 for d in r['disks'].values())
    return memory_ok, disk_ok


def check_apt_plan(text):
    """Never permit package removals or unrelated GPU/kernel replacements."""
    need(not re.search(r'^Remv\s', text, re.M), 'APT proposes removing packages; preserve and stop')
    names = re.findall(r'^Inst\s+([^\s:]+)(?::\S+)?\s', text, re.M)
    protected = ('nvidia-', 'libnvidia-', 'cuda-', 'libcudnn', 'cudnn', 'libnvinfer',
                 'tensorrt', 'linux-image-', 'linux-headers-', 'linux-modules-',
                 'python3.11', 'python3.12', 'python3.13', 'python3.14')
    need(not any(name.startswith(protected) for name in names),
         'APT proposes changing protected GPU/kernel/other-Python packages; preserve and stop')
    need(not re.search(r'\bdowngraded\b', text, re.I), 'APT downgrade proposed')
    return names


class Setup:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.actions = []
        self.sudo_ready = False
        self.counter = 0

    def command(self, argv, timeout=300, capture=True):
        self.counter += 1
        env = clean_env(); env.update(LC_ALL='C', LANG='C.UTF-8')
        print('SETUP: ' + ' '.join(map(str, argv)), flush=True)
        if capture:
            log=self.directory / ('command-%03d.log' % self.counter)
            with log.open('wb') as stream:
                p=subprocess.Popen(list(map(str,argv)),env=env,stdout=stream,stderr=subprocess.STDOUT)
                started=last=time.monotonic()
                try:
                    while p.poll() is None:
                        now=time.monotonic()
                        need(now-started<timeout,'Command time limit reached: '+str(argv))
                        if now-last>=5:
                            print('环境检查进行中 %.0f 秒 | %s' % (now-started,log),flush=True);last=now
                        time.sleep(.2)
                finally:
                    if p.poll() is None:
                        p.terminate()
                        try:p.wait(timeout=5)
                        except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
            output=log.read_text(errors='replace')
        else:
            p = subprocess.run(list(map(str, argv)), env=env, timeout=timeout)
            output = ''
        self.actions.append(dict(command=list(map(str, argv)), returncode=p.returncode))
        need(p.returncode == 0, 'Command failed; inspect ' +
             ('command-%03d.log' % self.counter if capture else 'terminal output') + ': ' + str(argv))
        return output

    def sudo(self):
        if not self.sudo_ready:
            print('需要当前机器的 sudo 权限来补齐 ROS 依赖；按终端提示输入密码。', flush=True)
            self.command(['sudo', '-v'], capture=False)
            self.sudo_ready = True
        return ['sudo', '-n']

    def room(self, install=False):
        r = resources()
        threshold = (2 * GIB if platform.machine() == 'x86_64' else GIB) if install else 128 * 1024 ** 2
        ram = 512 * 1024 ** 2
        memory_ok, disk_ok = check_resources(r, threshold, ram)
        # The user allowed deleting old things if storage is insufficient. Only
        # apt's re-downloadable package cache is automatically removed.
        if not disk_ok and install:
            print('磁盘余量不足：仅清理可重新下载的 APT 安装包缓存。', flush=True)
            self.command(self.sudo() + ['apt-get', 'clean'])
            r = resources(); memory_ok, disk_ok = check_resources(r, threshold, ram)
        need(memory_ok, 'RAM_AVAILABLE_LOW: close unused applications; deleting disk files does not free RAM')
        need(disk_ok, 'DISK_OR_INODES_LOW: apt cache cleanup was insufficient; models, environments and records preserved')
        return r

    def apt_update(self):
        self.command(self.sudo() + ['apt-get', '-o', 'Acquire::Retries=0', '-o', 'Acquire::http::Timeout=20',
                     '-o', 'Acquire::https::Timeout=20', '-o', 'APT::Update::Error-Mode=any', 'update'], timeout=300)

    def install(self, packages):
        self.room(install=True)
        plan = self.command(['apt-get', '--simulate', '--no-install-recommends', '--no-remove', 'install', *packages])
        planned = check_apt_plan(plan)
        print('APT 检查通过，待处理软件包数量：' + str(len(planned)), flush=True)
        if not planned:
            return
        self.command(self.sudo() + ['apt-get', '-o', 'DPkg::Lock::Timeout=60', '-o', 'Acquire::Retries=1',
                     '--yes', '--no-remove', '--no-install-recommends', 'install', *packages],
                     timeout=1200, capture=False)

    def ensure_ros(self):
        p = probe()
        if any(x.get('details',{}).get('ros_ok') and x['details'].get('ros_distro')=='humble'
               and x['details'].get('rmw')=='rmw_fastrtps_cpp' for x in p['ros_candidates']):
            return
        self.room(install=True)
        self.apt_update()
        policy = self.command(['apt-cache','policy','ros-humble-ros-base'])
        if not re.search(r'Candidate:\s+(?!\(none\))\S+', policy):
            # Only enable Ubuntu's official Universe component if unavailable.
            if 'c=universe' not in self.command(['apt-cache','policy']):
                if not shutil.which('add-apt-repository'):
                    self.install(['software-properties-common'])
                self.command(self.sudo()+['add-apt-repository','--yes','--no-update','universe'])
            package = Path(__file__).parent / BOOTSTRAP
            need(hashlib.sha256(package.read_bytes()).hexdigest()==BOOTSTRAP_SHA, 'ROS source bootstrap SHA mismatch')
            fields = self.command(['dpkg-deb','--field',str(package),'Package','Version','Architecture'])
            need('Package: ros2-apt-source' in fields and 'Version: 1.3.0~jammy' in fields
                 and 'Architecture: all' in fields, 'Unexpected official ROS source package')
            self.install([str(package.resolve())])
            self.apt_update()
        self.install(['ros-humble-ros-base','ros-humble-rclpy','ros-humble-std-msgs','ros-humble-rmw-fastrtps-cpp'])

    def ensure_ort(self):
        # Keep working historical environments untouched; create a dedicated
        # Humble/ORT environment only when no usable existing one is available.
        p = probe()
        ready = [x for x in p['ros_candidates'] if x.get('details',{}).get('ros_ok')
                 and x['details'].get('ros_distro')=='humble'
                 and x['details'].get('rmw')=='rmw_fastrtps_cpp'
                 and 'CPUExecutionProvider' in x['details'].get('ort_providers',[])]
        if ready:
            return ready[0]['requested_python']
        self.room(install=True)
        available = subprocess.run(['/usr/bin/python3','-c','import ensurepip'], capture_output=True).returncode == 0
        if not available:
            self.apt_update(); self.install(['python3.10-venv'])
        need(not VENV.is_symlink() and not VENV.parent.is_symlink(), 'Unexpected environment symlink')
        marker = VENV / 'microduck-owner.json'
        if VENV.exists():
            need(marker.is_file() and not marker.is_symlink() and json.loads(marker.read_text()).get('owner')==OWNER,
                 'Existing environment is not owned by this setup; preserve')
        else:
            VENV.mkdir(parents=True)
            marker.write_text(json.dumps(dict(owner=OWNER, ort=ORT_VERSION, numpy='1.26.4'))+'\n')
        self.command(['/usr/bin/python3','-m','venv','--system-site-packages',str(VENV)])
        python = VENV / 'bin/python'
        self.command([python,'-m','pip','install','--disable-pip-version-check','--no-cache-dir',
                      '--only-binary=:all:','--index-url','https://pypi.org/simple',
                      'onnxruntime=='+ORT_VERSION,'numpy==1.26.4'], timeout=600, capture=False)
        self.command([python,'-c',"import onnxruntime as o,numpy as n; assert o.__version__=='1.23.2' and n.__version__=='1.26.4'; assert 'CPUExecutionProvider' in o.get_available_providers(); print(o.__version__,n.__version__)"])
        return str(python)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--role',choices=('pc','jetson'),required=True)
    parser.add_argument('--directory',required=True)
    args=parser.parse_args()
    directory=Path(args.directory)
    need(directory.is_dir() and not directory.is_symlink(),'Existing private run directory required')
    job=Setup(directory)
    report=dict(status='HOST_SETUP_FAILED',role=args.role,started_utc=time.time(),actions=job.actions,
                training=False,network_configuration_changes=0,user_data_deleted=False,
                resources_before=resources(),python_environment_policy='preserve existing; isolated new environment only')
    try:
        need(sys.version_info[:2]==(3,10),'Use the existing /usr/bin/python3 Python 3.10')
        expected='x86_64' if args.role=='pc' else 'aarch64'
        need(platform.machine()==expected,'Wrong host architecture')
        p=probe();report['probe_before']=p
        need(p['os'].get('ID')=='ubuntu' and p['os'].get('VERSION_ID')=='22.04','Only existing Ubuntu 22.04 is supported')
        job.room()
        need(not job.command(['dpkg','--audit']).strip(), 'DPKG audit is not clean; preserve existing package state')
        job.ensure_ros()
        if args.role=='jetson':report['ort_python']=job.ensure_ort()
        report['probe_after']=probe()
        need(any(x.get('details',{}).get('ros_ok') for x in report['probe_after']['ros_candidates']), 'ROS import still fails')
        report['status']='HOST_SETUP_COMPLETE'
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc()[-8000:])
    finally:
        report.update(finished_utc=time.time(),resources_after=resources(),actions=job.actions)
        (directory/'host-setup-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('HOST_SETUP_STATUS='+report['status'],flush=True)
    return 0 if report['status']=='HOST_SETUP_COMPLETE' else 2


if __name__=='__main__':
    raise SystemExit(main())
