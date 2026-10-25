"""Humble setup on an existing Ubuntu 22.04 host; no training or network rewrites."""
import argparse
import codecs
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
from apt_scope import AptSources
import live_progress
from operation_guard import wait_package_manager

BOOTSTRAP = 'ros2-apt-source_1.3.0.jammy_all.deb'
BOOTSTRAP_SHA = '110b9a462d55252decb8b7c816f61c2ba0d9890ce5fb93ac504e97cae5860d76'
ORT_VERSION = '1.23.2'  # 1.24.4 requires >=3.11; Humble uses the existing 3.10.
VENV = Path.home() / 'venvs/microduck-stage12-humble-v2'
OWNER = 'microduck-stage12-humble-v2-owned-environment'
GIB = 1024 ** 3


def log_evidence(log, action):
    if not log.exists(): return ''
    raw = log.read_bytes()
    output = raw.decode('utf-8', errors='replace')
    clean = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', output)
    lines = clean.splitlines()
    indices = {j for i,line in enumerate(lines)
               if re.match(r'^(?:Err:|E:|W:|ERROR|Traceback)',line)
               for j in range(max(0,i-1),min(len(lines),i+4))}
    action.update(output_bytes=len(raw), output_sha256=hashlib.sha256(raw).hexdigest(),
                  output_tail=clean[-65536:], output_truncated=len(clean)>65536,
                  error_context='\n'.join(lines[i] for i in sorted(indices))[-32768:])
    return output


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
        self.apt_scope = None
        self.role = 'Orin' if platform.machine() == 'aarch64' else 'PC'

    def command(self, argv, timeout=300, capture=True, live=False, progress=None, installing=False):
        self.counter += 1
        env = clean_env(); env.update(LC_ALL='C', LANG='C.UTF-8')
        if 'apt-get' in argv and 'install' in argv:
            env['DEBIAN_FRONTEND'] = 'noninteractive'
        print('SETUP: ' + ' '.join(map(str, argv)), flush=True)
        action = dict(command=list(map(str, argv)), returncode=None)
        self.actions.append(action)
        output = ''
        if progress:
            log = self.directory / ('command-%03d.log' % self.counter)
            action.update(log=str(log), progress_display='native_pty', phase=progress,
                          package_configuration=installing, timeout_seconds=timeout)
            try:
                action['returncode'] = live_progress.run(argv, env, log, timeout,
                                                        self.role + ' | ' + progress, installing=installing)
            except BaseException as exc:
                action['error'] = repr(exc)
                raise
            finally:
                output = log_evidence(log, action)
            need(action['returncode']==0, 'Command failed; full output included in report: ' + str(log))
            return output
        try:
            if capture:
                log=self.directory / ('command-%03d.log' % self.counter)
                action['log'] = str(log)
                p = None
                try:
                    with log.open('wb') as stream, log.open('rb') as display:
                        decoder = codecs.getincrementaldecoder('utf-8')('replace')
                        p=subprocess.Popen(list(map(str,argv)),env=env,stdout=stream,stderr=subprocess.STDOUT)
                        started=last=time.monotonic()
                        try:
                            while p.poll() is None:
                                now=time.monotonic()
                                need(now-started<timeout,'Command time limit reached: '+str(argv))
                                if live:
                                    chunk = display.read()
                                    if chunk: print(decoder.decode(chunk),end='',flush=True)
                                if now-last>=5:
                                    print('环境检查进行中 %.0f 秒 | %s' % (now-started,log),flush=True);last=now
                                time.sleep(.2)
                        finally:
                            if p.poll() is None:
                                p.terminate()
                                try:p.wait(timeout=5)
                                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
                            if live: print(decoder.decode(display.read(),final=True),end='',flush=True)
                finally:
                    if p is not None: action['returncode'] = p.returncode
                    if log.exists():
                        raw = log.read_bytes(); output = raw.decode('utf-8',errors='replace')
                        lines = output.splitlines()
                        error_indices = {j for i,line in enumerate(lines)
                                         if re.match(r'^(?:Err:|E:|W:|ERROR|Traceback)',line)
                                         for j in range(max(0,i-1),min(len(lines),i+4))}
                        action.update(output_bytes=len(raw),output_sha256=hashlib.sha256(raw).hexdigest(),
                                      output_tail=output[-65536:],output_truncated=len(output)>65536,
                                      error_context='\n'.join(lines[i] for i in sorted(error_indices))[-32768:])
            else:
                # Only interactive sudo authentication is uncaptured. Passwords stay in the terminal.
                p = subprocess.run(list(map(str, argv)), env=env, timeout=timeout)
                action['returncode'] = p.returncode
        except BaseException as exc:
            action['error'] = repr(exc)
            raise
        if action['returncode'] != 0 and capture:
            print('安装步骤失败，具体输出已收录到报告：\n' + output[-12000:], flush=True)
        need(action['returncode'] == 0, 'Command failed; inspect ' +
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

    def apt_options(self, refresh=False):
        if self.apt_scope is None:
            self.apt_scope = AptSources(self.directory, clean_env())
            refresh = True
        return self.apt_scope.refresh() if refresh else self.apt_scope.options()

    def apt_update(self):
        options = self.apt_options(refresh=True)
        count = self.apt_scope.current['excluded_code_entries']
        print('APT 本次使用 IPv4；临时排除 VS Code 源条目 %d 个，原源文件保留。' % count, flush=True)
        wait_package_manager()
        self.command(self.sudo() + ['apt-get', *options, '-o', 'Acquire::Retries=2', '-o', 'quiet=0',
                     '-o', 'APT::Color=0', '-o', 'APT::Update::Error-Mode=any', 'update'], timeout=1800,
                     progress='刷新软件索引：总量动态变化，百分比仅为 APT 估计')

    def install(self, packages):
        wait_package_manager()
        self.room(install=True)
        def simulate():
            return self.command(['apt-get', *self.apt_options(), '--simulate', '--no-install-recommends', '--no-remove', 'install', *packages])
        plan = simulate()
        planned = check_apt_plan(plan)
        print('APT 检查通过，待处理软件包数量：' + str(len(planned)), flush=True)
        if not planned:
            print(self.role + ' | 所需软件包已安装，本批无需下载。', flush=True)
            return
        print(self.role + ' | 先下载并校验，随后安装；完整缓存复用，部分下载由 APT 尝试续传。', flush=True)
        self.command(self.sudo() + ['apt-get', *self.apt_options(), '-o', 'DPkg::Lock::Timeout=60',
                     '-o', 'Acquire::Retries=2', '-o', 'quiet=0', '-o', 'APT::Color=0',
                     '-o', 'APT::Keep-Downloaded-Packages=true',
                     '--download-only', '--yes', '--no-remove', '--no-install-recommends', 'install', *packages],
                     timeout=7200,
                     progress='下载软件包：百分比 | 当前包已下载/总量 | 下载速度 B/s | 预计剩余时间')
        print('\n' + self.role + ' | 本批下载完成 100%，继续核对安装计划。', flush=True)
        after = simulate(); check_apt_plan(after)
        need(re.findall(r'^(?:Inst|Conf)\s.*',plan,re.M)==re.findall(r'^(?:Inst|Conf)\s.*',after,re.M),
             'Package installation plan changed during download; preserve cache and stop')
        self.command(self.sudo() + ['env', 'DEBIAN_FRONTEND=noninteractive', 'apt-get', *self.apt_options(), '-o', 'DPkg::Lock::Timeout=60',
                     '-o', 'quiet=0', '-o', 'APT::Color=0', '-o', 'Dpkg::Progress-Fancy=1',
                     '-o', 'Dpkg::Options::=--force-confold', '-o', 'APT::Keep-Downloaded-Packages=true', '--no-download',
                     '--yes', '--no-remove', '--no-install-recommends', 'install', *packages],
                     timeout=None, progress='安装软件包：下载已完成，下面显示安装进度', installing=True)

    def ensure_ros(self):
        p = probe()
        if any(x.get('details',{}).get('ros_ok') and x['details'].get('ros_distro')=='humble'
               and x['details'].get('rmw')=='rmw_fastrtps_cpp' for x in p['ros_candidates']):
            print(self.role + ' | 已有 Humble 环境可用，直接复用。', flush=True)
            return
        self.room(install=True)
        self.apt_update()
        policy = self.command(['apt-cache',*self.apt_options(),'policy','ros-humble-ros-base'])
        if not re.search(r'Candidate:\s+(?!\(none\))\S+', policy):
            # Only enable Ubuntu's official Universe component if unavailable.
            if 'c=universe' not in self.command(['apt-cache',*self.apt_options(),'policy']):
                if not shutil.which('add-apt-repository'):
                    self.install(['software-properties-common'])
                self.command(self.sudo()+['add-apt-repository','--yes','--no-update','universe'])
                self.apt_update()
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
        self.command([python,'-m','pip','install','--disable-pip-version-check','--progress-bar','on',
                      '--cache-dir',str(Path.home()/'.cache/microduck-stage12-pip'),
                      '--only-binary=:all:','--index-url','https://pypi.org/simple',
                      'onnxruntime=='+ORT_VERSION,'numpy==1.26.4'], timeout=7200,
                      progress='下载推理依赖：按当前文件显示已下载/总量、速度和剩余时间')
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
        wait_package_manager()
        job.room()
        need(not job.command(['dpkg','--audit']).strip(), 'DPKG audit is not clean; preserve existing package state')
        job.ensure_ros()
        if args.role=='jetson':report['ort_python']=job.ensure_ort()
        report['probe_after']=probe()
        need(any(x.get('details',{}).get('ros_ok') for x in report['probe_after']['ros_candidates']), 'ROS import still fails')
        report['status']='HOST_SETUP_COMPLETE'
        print(job.role + ' | 环境检查通过。', flush=True)
    except BaseException as exc:
        report.update(error=repr(exc),traceback=traceback.format_exc()[-8000:])
    finally:
        report.update(finished_utc=time.time(),resources_after=resources(),actions=job.actions,
                      apt_source_scopes=job.apt_scope.generations if job.apt_scope else [])
        (directory/'host-setup-report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print('HOST_SETUP_STATUS='+report['status'],flush=True)
    return 0 if report['status']=='HOST_SETUP_COMPLETE' else 2


if __name__=='__main__':
    raise SystemExit(main())
