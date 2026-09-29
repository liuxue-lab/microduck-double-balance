"""Verify evidence retention and budget continuity without CUDA or SSH."""
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import time

import pytest
from mjlab_microduck.double_balance_stage08_budget import BudgetLedger, create_ledger

spec = importlib.util.spec_from_file_location('migration', Path(__file__).resolve().parents[1] /
                                              'scripts/migrate_stage08_setup.py')
migration = importlib.util.module_from_spec(spec)
spec.loader.exec_module(migration)


def source_tree(root):
    stage = root / 'artifacts/double-balance-stage08'
    (stage / 'setup').mkdir(parents=True)
    create_ledger(stage / 'budget.json', '5090',
                  datetime.fromtimestamp(time.time() - 7200, timezone.utc).isoformat())
    log = stage / 'setup/failed.log'
    log.write_text('RuntimeError: Download cancelled after another piece failed\n')
    (stage / 'setup/latest.log').symlink_to(log)
    return stage


def test_verified_archive_preserves_logs_and_original_budget(tmp_path):
    stage = source_tree(tmp_path)
    original = json.loads((stage / 'budget.json').read_text())
    archive = tmp_path / 'old.tar.gz'
    digest = migration.pack(tmp_path, archive)
    output = migration.verify(archive, digest, tmp_path / 'local-verified')
    assert (output / 'setup/failed.log').read_bytes() == (stage / 'setup/failed.log').read_bytes()
    destination = tmp_path / 'clone/budget.json'
    migration.carry_budget(output / 'budget.json', destination)
    current = BudgetLedger(destination).snapshot()
    assert current['campaign_id'] == original['campaign_id']
    assert current['started_epoch'] == original['started_epoch']
    assert current['elapsed_seconds'] >= original['elapsed_seconds'] >= 7200
    migration.carry_budget(output / 'budget.json', destination)  # retry keeps later clock
    assert json.loads(destination.read_text()) == current


def test_budget_conflict_cannot_overwrite_a_campaign(tmp_path):
    stage = source_tree(tmp_path)
    other = tmp_path / 'clone/budget.json'
    create_ledger(other, '5090', datetime.now(timezone.utc).isoformat())
    before = other.read_bytes()
    with pytest.raises(ValueError, match='campaign_id'):
        migration.carry_budget(stage / 'budget.json', other)
    assert other.read_bytes() == before


def test_refuses_running_setup_and_unarchived_training(tmp_path):
    stage = source_tree(tmp_path)
    with (stage / 'setup/install.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            migration.pack(tmp_path, tmp_path / 'busy.tar.gz')
    (stage / 'screening').mkdir()
    with pytest.raises(ValueError, match='additional Stage 08 artifacts'):
        migration.pack(tmp_path, tmp_path / 'training.tar.gz')


def test_corruption_and_unsafe_members_fail_before_extraction(tmp_path):
    source_tree(tmp_path)
    archive = tmp_path / 'old.tar.gz'
    migration.pack(tmp_path, archive)
    with pytest.raises(ValueError, match='SHA-256'):
        migration.verify(archive, '0' * 64, tmp_path / 'invalid')
    unsafe = tmp_path / 'unsafe.tar.gz'
    with tarfile.open(unsafe, 'w:gz') as tar:
        info = tarfile.TarInfo('../escaped'); info.size = 1
        tar.addfile(info, io.BytesIO(b'x'))
    with pytest.raises(ValueError, match='Unsafe'):
        migration.verify(unsafe, hashlib.sha256(unsafe.read_bytes()).hexdigest(), tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()
    assert not (tmp_path / 'escaped').exists()
