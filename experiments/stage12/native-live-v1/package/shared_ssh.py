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
DIAGNOSTICS = {'attempts': []}


def direct_options():
    require(HOST == 'jetson@192.168.50.2' and PORT == 22, 'Expected the verified wired Orin endpoint')
    return ['-4', '-b', '192.168.50.1', '-o', 'ProxyCommand=none', '-o', 'ProxyJump=none',
            '-o', 'HostName=192.168.50.2']


def record_attempt(log, returncode, mode):
    raw = log.read_text(errors='replace') if log.exists() else ''
    authenticated = 'Authenticated to ' in raw or 'Authentication succeeded' in raw
    if returncode == 0: category = 'SSH_COMMAND_PASSED'
    elif 'REMOTE HOST IDENTIFICATION HAS CHANGED' in raw or 'Host key verification failed' in raw:
        category = 'HOST_KEY_CHECK_FAILED'
    elif 'Too many authentication failures' in raw: category = 'TOO_MANY_AUTHENTICATION_ATTEMPTS'
    elif 'Permission denied' in raw: category = 'AUTHENTICATION_REJECTED'
    elif authenticated: category = 'FAILED_AFTER_AUTHENTICATION'
    else: category = 'CONNECTION_FAILED_WITHOUT_CONFIRMED_AUTHENTICATION'
    row = dict(mode=mode, returncode=returncode, category=category, authenticated=authenticated,
               log=str(log), output_tail=raw[-60000:], output_truncated=len(raw)>60000,
               password_recorded=False, automatic_authentication_retries=0)
    DIAGNOSTICS['attempts'].append(row)
    return row


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
            print('SSH：通过网线连接 jetson@192.168.50.2；请输入 Orin 的 jetson 登录密码。', flush=True)
            print('先执行只读连接检查，成功后后续操作复用本次认证。', flush=True)
            log = directory / ('connect-%d.log' % time.time_ns()); log.touch(mode=0o600)
            result = subprocess.run(['ssh', *direct_options(), '-p', str(PORT), '-M', '-T', '-S', str(sock),
                '-vv', '-E', str(log),
                '-o', 'ControlMaster=yes', '-o', 'ControlPersist=yes',
                '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=15',
                '-o', 'ServerAliveCountMax=3', '-o', 'NumberOfPasswordPrompts=1', HOST, '/usr/bin/true'])
            evidence = record_attempt(log, result.returncode, 'command_then_controlpersist')
            require(result.returncode == 0, 'SSH connection failed: ' + evidence['category'] + '; detailed log included in report')
            require(_live(sock), 'SSH master connection was not established')
            _SOCKET = sock
        DIAGNOSTICS.update(status='SHARED_SESSION_READY', socket=str(_SOCKET))
    print('SSH_SHARED_SESSION=READY; subsequent_password_prompts=0', flush=True)
    return _SOCKET


def check_only():
    """One ordinary non-multiplexed read-only command, only when explicitly requested."""
    directory = _directory()
    log = directory / ('check-%d.log' % time.time_ns()); log.touch(mode=0o600)
    print('SSH 单独诊断：请输入 Orin 上 jetson 用户的登录密码。', flush=True)
    result = subprocess.run(['ssh', *direct_options(), '-p', str(PORT), '-T', '-S', 'none',
                            '-o', 'ControlMaster=no', '-o', 'ControlPersist=no',
                            '-o', 'ConnectTimeout=15', '-o', 'NumberOfPasswordPrompts=1',
                            '-vv', '-E', str(log), HOST, '/usr/bin/true'])
    return record_attempt(log, result.returncode, 'ordinary_no_multiplexing')


def argv(remote_command):
    sock = connect()
    # ProxyCommand=false makes direct-connect fallback fail closed if the master
    # disappears between -O check and command execution. BatchMode prevents any
    # password/key prompt. The user explicitly reconnects after an actual outage.
    return ['ssh', '-p', str(PORT), '-S', str(sock), '-o', 'ControlMaster=no',
            '-o', 'BatchMode=yes', '-o', 'ProxyCommand=false',
            '-o', 'ConnectTimeout=15', HOST, remote_command]
