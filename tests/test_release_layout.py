"""Publication renames must validate both historical and current source bytes."""
import hashlib
import json
from pathlib import Path

import pytest

from mjlab_microduck.release_layout import (
    BASELINE_COMMIT, SCHEMA, resolve_source_path, verify_source_file,
    verify_source_inventory,
)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def fixture(tmp_path):
    original, current = b'original source\n', b'published imports\n'
    target = tmp_path/'src/control.py'
    target.parent.mkdir()
    target.write_bytes(current)
    manifest = tmp_path/'docs/reference/source-layout.json'
    manifest.parent.mkdir(parents=True)
    data = {'schema': SCHEMA, 'baseline_commit': BASELINE_COMMIT, 'files': {
        'src/old_control.py': {'path': 'src/control.py', 'original_sha256': digest(original),
                               'sha256': digest(current), 'bytes': len(current)}}}
    manifest.write_text(json.dumps(data))
    return original, current, target, manifest, data


def test_renamed_source_requires_original_pin_and_current_hash(tmp_path):
    original, current, target, _, _ = fixture(tmp_path)
    assert resolve_source_path(tmp_path, 'src/old_control.py') == target
    assert verify_source_file(tmp_path, 'src/old_control.py', digest(original), len(original)) == target
    with pytest.raises(ValueError, match='Recorded source identity'):
        verify_source_file(tmp_path, 'src/old_control.py', digest(current))
    target.write_bytes(b'changed after publication')
    with pytest.raises(ValueError, match='Published source SHA-256'):
        verify_source_file(tmp_path, 'src/old_control.py', digest(original))


def test_original_unchanged_file_is_checked_without_manifest(tmp_path):
    target = tmp_path/'source.py'
    target.write_bytes(b'stable')
    assert verify_source_file(tmp_path, 'source.py', digest(b'stable'), 6) == target
    with pytest.raises(ValueError, match='size'):
        verify_source_file(tmp_path, 'source.py', digest(b'stable'), 5)


def test_manifest_cannot_redirect_outside_source_tree(tmp_path):
    original, _, _, manifest, data = fixture(tmp_path)
    for path in ('../outside.py', '/tmp/outside.py', 'src/../../outside.py'):
        data['files']['src/old_control.py']['path'] = path
        manifest.write_text(json.dumps(data))
        with pytest.raises(ValueError):
            verify_source_file(tmp_path, 'src/old_control.py', digest(original))


def test_bad_manifest_schema_or_baseline_is_rejected(tmp_path):
    _, _, _, manifest, data = fixture(tmp_path)
    data['baseline_commit'] = '0'*40
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='Unrecognized'):
        resolve_source_path(tmp_path, 'src/old_control.py')


def test_inventory_maps_paths_and_rejects_duplicate_rows(tmp_path):
    original, _, target, _, _ = fixture(tmp_path)
    row = {'path': 'old_control.py', 'sha256': digest(original), 'size_bytes': len(original)}
    assert verify_source_inventory(tmp_path, [row], prefix='src') == [target]
    with pytest.raises(ValueError, match='Duplicate'):
        verify_source_inventory(tmp_path, [row, row], prefix='src')


def test_symlink_and_actual_size_mismatch_are_rejected(tmp_path):
    original, current, target, manifest, data = fixture(tmp_path)
    data['files']['src/old_control.py']['bytes'] += 1
    manifest.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='size'):
        verify_source_file(tmp_path, 'src/old_control.py', digest(original))
    target.unlink()
    outside = tmp_path/'outside.py'
    outside.write_bytes(current)
    target.symlink_to(outside)
    with pytest.raises(ValueError, match='escapes'):
        resolve_source_path(tmp_path, 'src/old_control.py')
