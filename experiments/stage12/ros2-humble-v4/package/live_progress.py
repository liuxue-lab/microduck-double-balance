"""Retain native APT/pip progress through a PTY while saving complete output."""
import errno
import fcntl
import os
from pathlib import Path
import pty
import select
import shutil
import signal
import struct
import subprocess
import sys
import termios
import time
import codecs
import re


class PipText:
    """Plain per-file pip progress, adding a percentage to pip's byte totals."""
    def __init__(self): self.pending = ''

    def feed(self, text, final=False):
        self.pending += text
        rows = re.split(r'[\r\n]', self.pending)
        self.pending = '' if final else rows.pop()
        for row in rows:
            row = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', row).strip()
            if not row: continue
            match = re.search(r'(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\s*(?:kB|MB|GB|B)\b', row)
            if match and float(match.group(2)) > 0:
                percentage = min(100, 100 * float(match.group(1))/float(match.group(2)))
                row = '当前文件 %5.1f%% | ' % percentage + row
            print(row, flush=True)


def run(argv, env, log, timeout, label, installing=False):
    master, slave = pty.openpty()
    width = max(80, shutil.get_terminal_size(fallback=(120, 24)).columns)
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack('HHHH', 24, width, 0, 0))
    env = dict(env, TERM='xterm', COLUMNS=str(width), NO_COLOR='1', PIP_PROGRESS_BAR='on')
    started = last_output = last_notice = time.monotonic()
    decoder = codecs.getincrementaldecoder('utf-8')('replace')
    pip_text = PipText() if '-m' in argv and 'pip' in argv else None
    def display(text, final=False):
        if pip_text: pip_text.feed(text, final=final)
        else: print(text, end='', flush=True)
    p = None
    stop_requested = False
    print('\n' + label, flush=True)
    try:
        with Path(log).open('wb') as output:
            p = subprocess.Popen(list(map(str, argv)), env=env, stdin=subprocess.DEVNULL,
                                 stdout=slave, stderr=subprocess.STDOUT, start_new_session=True)
            os.close(slave); slave = None
            ended = False
            while not ended:
                try:
                    now = time.monotonic()
                    if timeout is not None and now-started > timeout:
                        raise TimeoutError('Download/check timeout; cached files preserved: ' + str(log))
                    ready, _, _ = select.select([master], [], [], .2)
                    if ready:
                        try:
                            chunk = os.read(master, 65536)
                        except OSError as exc:
                            if exc.errno != errno.EIO: raise
                            chunk = b''
                        if chunk:
                            output.write(chunk); output.flush()
                            display(decoder.decode(chunk))
                            last_output = now
                        else:
                            ended = True
                    elif p.poll() is not None:
                        ended = True
                    if now-last_output >= 5 and now-last_notice >= 5:
                        print('\n%s | 已用 %.0f 秒 | 等待当前步骤输出' % (label, now-started), flush=True)
                        last_notice = now
                except KeyboardInterrupt:
                    if not installing:
                        raise
                    stop_requested = True
                    print('\n正在配置软件包，当前步骤结束后停止后续操作。', flush=True)
            display(decoder.decode(b'', final=True), final=True)
            # No wall-clock kill during dpkg configuration.
            while p.poll() is None:
                try:
                    p.wait(timeout=.5)
                except subprocess.TimeoutExpired:
                    pass
                except KeyboardInterrupt:
                    if not installing: raise
                    stop_requested = True
            returncode = p.returncode
        if stop_requested:
            raise KeyboardInterrupt('Stopped after package configuration completed')
        return returncode
    finally:
        if slave is not None: os.close(slave)
        os.close(master)
        if p is not None and p.poll() is None:
            # This path is for an interrupted download or an unexpected reader
            # failure. Never intentionally time out the dpkg installation phase.
            if installing:
                print('\n安装进程仍在运行，等待其退出并保留日志。', flush=True)
                while p.poll() is None:
                    try: p.wait(timeout=1)
                    except (subprocess.TimeoutExpired, KeyboardInterrupt): pass
            else:
                try: os.killpg(p.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: p.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, signal.SIGKILL); p.wait(timeout=10)

