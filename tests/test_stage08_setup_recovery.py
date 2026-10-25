"""Exercise recovery against disposable processes, never a GPU or real installer."""
from contextlib import contextmanager
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import pytest

REPO = Path(__file__).resolve().parents[1]


def controller(root):
    script = (REPO / 'scripts/recover_stage08_setup_local.sh').read_text()
    code = script.split(".venv/bin/python - <<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
    return code.replace('/root/autodl-tmp/microduck-double-balance', str(root))


@contextmanager
def installer(root, executable, *, parent='setup_stage08_autodl.sh'):
    repo = root / 'workspace'
    (repo / 'scripts').mkdir(parents=True, exist_ok=True)
    executable.parent.mkdir(parents=True, exist_ok=True)
    executable.touch()
    ready = repo / f'ready-{parent}'
    # argv[0] models both pip's console entry point and uv's packaged executable.
    (repo / 'sync').write_text(
        'from pathlib import Path\nimport os,time\n'
        f'Path({str(ready)!r}).write_text(str(os.getpid()))\n'
        'time.sleep(60)\n')
    worker = repo / 'scripts' / parent
    worker.write_text('#!/bin/bash\n'
        'bash -c \'exec -a "$1" "$2" sync\' fake '
        f'{shlex.quote(str(executable))} {shlex.quote(sys.executable)}\n'
        'exit "$?"\n')
    process = subprocess.Popen(['bash', str(worker), '--worker'], cwd=repo,
                               start_new_session=True, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
    try:
        deadline = time.monotonic() + 5
        while not ready.exists():
            assert process.poll() is None
            assert time.monotonic() < deadline
            time.sleep(.02)
        yield process, int(ready.read_text())
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        process.wait(timeout=5)
        ready.unlink(missing_ok=True)


@pytest.mark.parametrize('layout', ['bin/uv', 'lib/python3.10/site-packages/uv/uv'])
def test_recovery_stops_only_the_setup_download(tmp_path, layout):
    with installer(tmp_path, tmp_path / 'tools/bootstrap' / layout) as (target, _):
        with installer(tmp_path, tmp_path / 'unrelated/bin/uv', parent='other.sh') as (other, _):
            result = subprocess.run([sys.executable, '-c', controller(tmp_path)],
                                    capture_output=True, text=True, timeout=10)
            assert result.returncode == 0, result.stderr
            assert 'STOP_REQUESTED' in result.stdout
            assert target.wait(timeout=5) != 0
            assert other.poll() is None


def test_recovery_refuses_an_unrecognized_parent(tmp_path):
    with installer(tmp_path, tmp_path / 'tools/bootstrap/bin/uv', parent='other.sh') as (other, _):
        result = subprocess.run([sys.executable, '-c', controller(tmp_path)],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode != 0
        assert 'Unrecognized installer parent' in result.stderr
        assert other.poll() is None
