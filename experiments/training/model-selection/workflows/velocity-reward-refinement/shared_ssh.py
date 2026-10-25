"""One explicit interactive SSH authentication, then mux-only commands.

The control socket is shared by host/port/user, not a package or run directory.
No passwords are read/stored and no keys or sshd/client config are changed.
"""
import fcntl
import hashlib
import os
from pathlib import Path
import stat
import subprocess
import time

HOST = 'root@connect.bjb2.seetacloud.com'
PORT = 52763
_SOCKET = None
_REUSED = False


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def _directory():
    token = hashlib.sha256((HOST + ':' + str(PORT)).encode()).hexdigest()[:12]
    path = Path('/tmp') / ('md12-ssh-shared-%d-%s' % (os.getuid(), token))
    if not path.exists():
        path.mkdir(mode=0o700)
    st = path.lstat()
    require(stat.S_ISDIR(st.st_mode) and st.st_uid == os.getuid() and stat.S_IMODE(st.st_mode) == 0o700,
            'Unsafe SSH control directory')
    return path


def _old_socket():
    base = '/home/lx/microduck-double-balance/artifacts/double-balance-stage12/finite-trial-v1/client-20261019-v1'
    token = hashlib.sha256((base + HOST + str(PORT)).encode()).hexdigest()[:12]
    folder = Path('/tmp') / ('md12-ssh-%d-%s' % (os.getuid(), token))
    if not folder.is_dir() or folder.is_symlink():
        return None
    st = folder.stat()
    if st.st_uid != os.getuid() or stat.S_IMODE(st.st_mode) != 0o700:
        return None
    return folder / 's'


def _live(sock):
    if sock is None:
        return False
    try:
        st = sock.lstat()
        if not stat.S_ISSOCK(st.st_mode) or st.st_uid != os.getuid():
            return False
        r = subprocess.run(['ssh', '-p', str(PORT), '-S', str(sock), '-O', 'check', HOST],
                           stdin=subprocess.DEVNULL, capture_output=True, timeout=10)
        return r.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def connect():
    global _SOCKET, _REUSED
    if _SOCKET is not None:
        require(_live(_SOCKET), 'SSH shared connection ended; rerun the status command to reconnect once')
        return _SOCKET
    directory = _directory()
    with (directory / 'connect.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        sock = directory / 's'
        if _live(sock):
            _SOCKET = sock
        else:
            old = _old_socket()
            if _live(old):
                # Attach to an already authenticated v1 connection while it is live.
                # Periodic commands keep its existing idle timer from expiring.
                _SOCKET, _REUSED = old, True
            else:
                if sock.exists() or sock.is_symlink():
                    st = sock.lstat()
                    require(stat.S_ISSOCK(st.st_mode) and st.st_uid == os.getuid(),
                            'Unexpected SSH control path; preserve it')
                    # The owned socket failed an OpenSSH control check. Preserve
                    # it under a new name instead of deleting unrelated paths.
                    sock.rename(directory / ('s.stale.%d' % time.time_ns()))
                print('SSH 首次认证；后续状态、停止和回传复用此会话。', flush=True)
                subprocess.run(['ssh', '-p', str(PORT), '-M', '-N', '-f', '-S', str(sock),
                    '-o', 'ControlMaster=yes', '-o', 'ControlPersist=yes',
                    '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15',
                    '-o', 'ServerAliveCountMax=3', '-o', 'NumberOfPasswordPrompts=1', HOST], check=True)
                require(_live(sock), 'SSH master connection was not established')
                _SOCKET = sock
    print('SSH_SHARED_SESSION=READY; subsequent_password_prompts=0', flush=True)
    return _SOCKET


def argv(remote_command):
    sock = connect()
    # ProxyCommand=false makes direct-connect fallback fail closed if the master
    # disappears between -O check and command execution. BatchMode prevents any
    # password/key prompt. The user explicitly reconnects after an actual outage.
    return ['ssh', '-p', str(PORT), '-S', str(sock), '-o', 'ControlMaster=no',
            '-o', 'BatchMode=yes', '-o', 'ProxyCommand=false',
            '-o', 'ConnectTimeout=15', HOST, remote_command]
