"""Frozen local safety and precision-reading helpers from corrected Batch A."""

from __future__ import annotations

import argparse

from datetime import datetime, timezone

import hashlib

import itertools

import json

import os

from pathlib import Path

import shlex

import shutil

import subprocess

import sys

import time

import uuid

import zipfile

REPO = Path('/home/lx/microduck-double-balance/workspace')

BASE = '00e34c2038771c5d4ad49fe45dff828c0232e60c'

PRIMARY = '86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97'

MANIFEST = 'b61850afbd6fa0deccb3a23a098a910aae74bc0e9b01cca2ee0634c2d1f6ecfd'

INITIAL = '253123669696a953830caeaf587e1300e023c12293d608e38e655a38cdc00f58'

ART = REPO.parent / 'artifacts/double-balance-stage10'

VIDEO09 = REPO.parent / 'artifacts/double-balance-stage09/video-review/20260930T210037Z-5a95cdc3'

def require(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
        temp.replace(path)
    finally:
        temp.unlink(missing_ok=True)

def utc():
    return datetime.now(timezone.utc).isoformat()

def checked(path, checksum):
    path = Path(path)
    require(path.is_file() and not path.is_symlink(), 'Missing or linked input: ' + str(path))
    require(sha(path) == checksum, 'SHA256 differs: ' + str(path))
    return path

def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], text=True).strip()

def verify_source():
    require(REPO.is_dir(), 'Canonical laptop repository is missing: ' + str(REPO))
    require(git('rev-parse', '--show-toplevel') == str(REPO), 'Unexpected repository root')
    require(git('branch', '--show-current') == 'double-balance', 'Expected double-balance branch')
    require(git('rev-parse', 'HEAD') == BASE, 'Source HEAD changed; preserve it and return the message for review')
    require(not git('diff', 'HEAD', '--name-only'), 'Tracked local edits detected; preserve them and return the message')
    manifest = checked(REPO / 'docs/audits/stage-09-runtime-hashes.json', MANIFEST)
    rows = read(manifest)
    require(len(rows) == 101, 'Expected 101 frozen runtime files')
    for relative, checksum in rows.items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'Unsafe manifest path')
        checked(REPO / relative, checksum)
    return {'source_head': BASE, 'frozen_files_verified': 101, 'runtime_manifest_sha256': MANIFEST}

def resolve_inputs():
    receipt = read(REPO / 'docs/audits/stage-08-return-verified.json')
    checkpoint = Path(receipt['extracted_directory']) / 'extension/E-20260929/segment-001/checkpoints/update_004000.pt'
    checked(checkpoint, PRIMARY)
    # Reuse exactly the Stage 09 local replay dataset, not a new random draw.
    dataset = checked(VIDEO09 / 'initial-states/dev.pt', INITIAL)
    sidecar = dataset.with_suffix('.json')
    metadata = read(sidecar)
    require(metadata['sha256'] == INITIAL and len(metadata['state_ids']) == 128,
            'Stage 09 dev dataset receipt differs')
    return checkpoint, dataset, sidecar

def supervised(command, log, timeout_s=900):
    """Supervise only the local child launched here; never manage other processes."""
    print('Stage10Command=' + shlex.join(command), flush=True)
    log = Path(log)
    started = time.monotonic()
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=REPO, stdout=stream, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    code = child.wait(timeout=15)
                    break
                except subprocess.TimeoutExpired:
                    elapsed = time.monotonic() - started
                    print(f'Stage10Heartbeat=local PID {child.pid}; {elapsed:.0f}s; log={log}', flush=True)
                    require(elapsed < timeout_s, 'Local job exceeded its 15-minute limit; stopping this child only')
        except BaseException:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
            raise
    require(code == 0, f'Local child exited {code}; see {log}')

def inventory(folder):
    return [{'path': str(p.relative_to(folder)), 'sha256': sha(p), 'size_bytes': p.stat().st_size}
            for p in sorted(Path(folder).rglob('*')) if p.is_file()]

def verify_inventory(folder, rows):
    for row in rows:
        rel = Path(row['path'])
        require(not rel.is_absolute() and '..' not in rel.parts, 'Unsafe receipt path')
        p = checked(Path(folder) / rel, row['sha256'])
        require(p.stat().st_size == row['size_bytes'], 'File size differs: ' + str(p))

_PRECISION_OWNER = None
_PRECISION_HANDLES = None


def precision_handles(torch):
    """Keep live getter objects before LSTM imports can shadow cudnn.rnn.

    This retains objects, never precision values. Every snapshot still calls
    their fp32_precision getter and observes the current underlying setting.
    """
    global _PRECISION_OWNER, _PRECISION_HANDLES
    if _PRECISION_OWNER is not torch:
        handles = {'matmul': torch.backends.cuda.matmul,
                   'conv': torch.backends.cudnn.conv, 'rnn': torch.backends.cudnn.rnn}
        for name, handle in handles.items():
            require(isinstance(handle.fp32_precision, str), 'Missing precision getter: ' + name)
        _PRECISION_OWNER, _PRECISION_HANDLES = torch, handles
    return _PRECISION_HANDLES


def precision_reader_audit(torch):
    import inspect
    handles = precision_handles(torch)
    result = {'cached_values': False, 'getter_objects_retained': True, 'settings_written': False,
              'rnn_namespace_shadowed': torch.backends.cudnn.rnn is not handles['rnn'], 'getter_types': {}, 'getter_sources': {}}
    for name, handle in handles.items():
        cls = type(handle)
        result['getter_types'][name] = cls.__module__ + '.' + cls.__qualname__
        try:
            result['getter_sources'][name] = inspect.getsource(cls)
        except (TypeError, OSError):
            result['getter_sources'][name] = 'SOURCE_UNAVAILABLE'
    return result


def backend_snapshot(torch):
    """Read current settings using live getter objects; do not set precision."""
    handles = precision_handles(torch)
    return {'precision_api': 'fp32_precision',
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'global_fp32_precision': torch.backends.fp32_precision,
            'matmul_fp32_precision': handles['matmul'].fp32_precision,
            'cudnn_fp32_precision': torch.backends.cudnn.fp32_precision,
            'cudnn_conv_fp32_precision': handles['conv'].fp32_precision,
            'cudnn_rnn_fp32_precision': handles['rnn'].fp32_precision}
