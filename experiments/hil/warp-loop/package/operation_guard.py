"""Wait for earlier Microduck runs and active package-manager locks; never kill them."""
from contextlib import contextmanager, ExitStack
import fcntl
import os
from pathlib import Path
import time

LOCK_PATHS = ('/var/lib/dpkg/lock-frontend', '/var/lib/dpkg/lock',
              '/var/lib/apt/lists/lock', '/var/cache/apt/archives/lock')


def acquire_wait(stream, label):
    last = -1e9
    while True:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            now = time.monotonic()
            if now-last >= 5:
                print('等待已有任务结束：' + label + '；结束后自动继续，下载尚未开始。', flush=True)
                last = now
            time.sleep(.5)


@contextmanager
def previous_runs(root):
    root = Path(root)
    with ExitStack() as stack:
        shared = root.parent / 'native-ros2-humble-operation.lock'
        if shared.is_symlink(): raise RuntimeError('Unexpected operation lock symlink')
        handle = stack.enter_context(shared.open('a'))
        acquire_wait(handle, '另一个 Humble 安装任务')
        for version in ('v2', 'v3'):
            path = root.parent / ('native-ros2-humble-' + version) / 'repair.lock'
            if path.exists():
                if path.is_symlink(): raise RuntimeError('Unexpected previous lock symlink')
                handle = stack.enter_context(path.open('a'))
                acquire_wait(handle, 'Humble ' + version)
        yield


def held_package_locks(paths=LOCK_PATHS, proc_locks=Path('/proc/locks')):
    identities = {}
    for name in paths:
        p = Path(name)
        if p.exists():
            st = p.stat()
            identities[(os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)] = str(p)
    result = []
    for line in proc_locks.read_text().splitlines():
        fields = line.split()
        if len(fields) < 6 or fields[1] == '->': continue
        try:
            major, minor, inode = fields[5].split(':')
            key = int(major,16), int(minor,16), int(inode)
        except ValueError:
            continue
        if key in identities:
            result.append(dict(pid=int(fields[4]), path=identities[key], kind=fields[1]))
    return result


def wait_package_manager():
    last = -1e9
    while True:
        held = held_package_locks()
        if not held: return
        now = time.monotonic()
        if now-last >= 5:
            print('等待已有 APT/DPKG 完成：' + ', '.join(str(x['pid'])+' '+x['path'] for x in held), flush=True)
            last = now
        time.sleep(.5)
