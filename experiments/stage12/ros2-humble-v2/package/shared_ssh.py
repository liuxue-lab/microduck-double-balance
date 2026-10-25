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

HOST = None
PORT = None
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
    require(HOST is not None and PORT is not None, "Current user-supplied SSH endpoint is required")
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
            if sock.exists() or sock.is_symlink():
                st = sock.lstat()
                require(stat.S_ISSOCK(st.st_mode) and st.st_uid == os.getuid(),
                        'Unexpected SSH control path; preserve it')
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
