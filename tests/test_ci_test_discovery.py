"""Keep host-specific historical suites out of default pytest discovery.

Exercise the real pytest collector in an isolated repository fixture. Archived
stages intentionally reuse module names and can change process-wide settings
at import time; they must be selected explicitly in their own environment.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def _fixture(tmp_path: Path) -> None:
    (tmp_path / 'pytest.ini').write_bytes((ROOT / 'pytest.ini').read_bytes())
    (tmp_path / 'tests').mkdir()
    (tmp_path / 'tests/test_contract.py').write_text(
        'import os\ndef test_portable_contract():\n'
        '    assert "MICRODUCK_ARCHIVE_IMPORTED" not in os.environ\n'
    )
    for stage in ('older-stage', 'newer-stage'):
        folder = tmp_path / 'experiments' / stage / 'tests'
        folder.mkdir(parents=True)
        (folder / 'test_contract.py').write_text(
            'import os\nos.environ["MICRODUCK_ARCHIVE_IMPORTED"] = "1"\n'
            'raise RuntimeError("HOST_ARCHIVE_MUST_NOT_BE_IMPORTED")\n'
        )
    portable = tmp_path / 'experiments/explicit-portable/tests'
    portable.mkdir(parents=True)
    (portable / 'test_contract.py').write_text('def test_explicit_stage():\n    assert True\n')


def _pytest(folder: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop('PYTEST_ADDOPTS', None)
    env.pop('MICRODUCK_ARCHIVE_IMPORTED', None)
    env['PYTEST_DISABLE_PLUGIN_AUTOLOAD'] = '1'
    return subprocess.run(
        [sys.executable, '-m', 'pytest', '-q', *args], cwd=folder, env=env,
        capture_output=True, text=True, timeout=30, check=False,
    )


def test_default_discovery_does_not_import_archived_stages(tmp_path: Path) -> None:
    _fixture(tmp_path)
    result = _pytest(tmp_path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout
    assert 'HOST_ARCHIVE_MUST_NOT_BE_IMPORTED' not in result.stdout


def test_explicit_stage_selection_is_still_available(tmp_path: Path) -> None:
    _fixture(tmp_path)
    result = _pytest(tmp_path, 'experiments/explicit-portable/tests')
    assert result.returncode == 0, result.stdout + result.stderr
    assert '1 passed' in result.stdout


def test_unbounded_collection_reproduces_the_original_failure(tmp_path: Path) -> None:
    _fixture(tmp_path)
    # This intentionally overrides the configured search roots in a disposable
    # fixture, proving that the guard catches the actual import-time failure.
    result = _pytest(tmp_path, '.')
    assert result.returncode == 2, result.stdout + result.stderr
    assert ('HOST_ARCHIVE_MUST_NOT_BE_IMPORTED' in result.stdout
            or 'import file mismatch' in result.stdout)
