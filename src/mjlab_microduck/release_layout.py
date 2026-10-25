"""Resolve functional source paths while retaining recorded source identities.

The layout manifest describes publication edits. Original evidence keeps its
original path and SHA-256; a mapped source is accepted only when both its recorded
identity and the current published bytes match the explicit manifest entry.
External artifacts are intentionally outside this source-only adapter.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re


LAYOUT_PATH = 'docs/reference/source-layout.json'
SCHEMA = 'microduck-source-layout-v1'
BASELINE_COMMIT = '776ad40b1a379c2fc44132cc2ff47d2529fb6061'


def _relative(value):
    path = PurePosixPath(str(value))
    if path.is_absolute() or not path.parts or '..' in path.parts or '\\' in str(value):
        raise ValueError('Expected a safe repository-relative source path: ' + str(value))
    return path.as_posix()


def _inside(repo, relative):
    root = Path(repo).resolve()
    path = root / _relative(relative)
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError('Source path escapes the repository: ' + str(relative))
    return path


def _layout(repo):
    path = _inside(repo, LAYOUT_PATH)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if data.get('schema') != SCHEMA or data.get('baseline_commit') != BASELINE_COMMIT:
        raise ValueError('Unrecognized source layout manifest')
    records = data.get('files')
    if not isinstance(records, dict):
        raise ValueError('Invalid source layout records')
    return records


def resolve_source_path(repo, original_relative):
    """Map a recorded repository path to its current public source path."""
    original = _relative(original_relative)
    record = _layout(repo).get(original)
    return _inside(repo, record['path'] if record is not None else original)


def verify_source_file(repo, original_relative, expected_original_sha256,
                       expected_original_bytes=None):
    """Verify the historical pin and current source bytes, returning its Path.

    For an unmapped path, the expected original SHA and size apply directly. For
    a mapped path, the original SHA must equal the recorded pre-publication SHA;
    the current SHA and size are independently checked against the release map.
    The original size is checked for unchanged bytes. Historical size records are
    otherwise retained by the caller rather than reinterpreted as current size.
    """
    original = _relative(original_relative)
    if not re.fullmatch(r'[0-9a-f]{64}', str(expected_original_sha256)):
        raise ValueError('Invalid original source SHA-256: ' + original)
    record = _layout(repo).get(original)
    if record is None:
        path = _inside(repo, original)
        expected_sha = expected_original_sha256
        expected_size = expected_original_bytes
    else:
        if record.get('original_sha256') != expected_original_sha256:
            raise ValueError('Recorded source identity differs: ' + original)
        path = _inside(repo, record['path'])
        expected_sha = record.get('sha256')
        expected_size = record.get('bytes')
        if not re.fullmatch(r'[0-9a-f]{64}', str(expected_sha)) or type(expected_size) is not int or expected_size < 0:
            raise ValueError('Invalid published source identity: ' + original)
        if expected_sha == expected_original_sha256 and expected_original_bytes is not None:
            if expected_size != expected_original_bytes:
                raise ValueError('Recorded source size differs: ' + original)
    if not path.is_file():
        raise ValueError('Missing published source: ' + str(path))
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha:
        raise ValueError('Published source SHA-256 differs: ' + original)
    if expected_size is not None and len(content) != expected_size:
        raise ValueError('Published source size differs: ' + original)
    return path


def verify_source_inventory(repo, rows, prefix=''):
    """Verify recorded inventory rows against mapped current repository files."""
    base = _relative(prefix) if prefix else ''
    paths = []
    seen = set()
    for row in rows:
        relative = _relative(row['path'])
        original = str(PurePosixPath(base) / relative) if base else relative
        if original in seen:
            raise ValueError('Duplicate source inventory entry: ' + original)
        seen.add(original)
        paths.append(verify_source_file(repo, original, row['sha256'], row.get('size_bytes')))
    return paths
