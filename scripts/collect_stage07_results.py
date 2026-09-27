#!/usr/bin/env python3
"""Read-only cloud collection; run this file with Python 3 on the laptop.

No project imports, dependency installation, training, or cloud source changes.
Selected checkpoints remain outside Git. All other periodic models stay on cloud.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import tempfile

CLOUD_ROOT = Path('/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage07')
DOWNLOADS = Path('/home/lx/下载')
LOCAL_ARTIFACTS = Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage07')
PROTOCOL = 'stage04-frozen-play-10s-first-episode-v1'
INITIAL_SHA = '548758afc546e11d8f3f446a4bcc846283a669ff20f4048da7a3a7bb1332b98e'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else digest_stream(stream)


def digest_stream(stream):
    digest = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        digest.update(block)
    return digest.hexdigest()


def rank(report):
    require(report['status'] == 'PASS' and report['protocol'] == PROTOCOL, 'Evaluation protocol/status mismatch')
    require(report['num_envs'] == 64 and report['horizon_seconds'] == 10.0 and report['assistance'] == 0.0,
            'Evaluation configuration mismatch')
    values = (report['success_fraction'], report['mean_survival_seconds'],
              report['mean_stable_fraction'], -report['mean_top_center_error_m'])
    require(all(math.isfinite(x) for x in values), 'Nonfinite selection metric')
    return values


def collect_plan(root):
    root = root.resolve(strict=True)
    current = (root / 'latest-training').resolve(strict=True)
    require(current.parent == root, 'Unexpected latest-training location')
    training = read_json(current / 'training.json')
    require(training['status'] == 'TRAINING_COMPLETE', f"Training has not finished: {training['status']}")
    require(training['finite_checks'] == 'PASS' and training['completed_updates'] == training['target_updates'] == 6000,
            'Expected the completed, finite 6000-update segment')
    runs = sorted(p for p in root.glob('train-*') if p.is_dir() and not p.is_symlink())
    evaluations = []
    files = {}
    for run in runs:
        if not (run / 'training.json').is_file():
            continue
        require(read_json(run / 'training.json')['initialization_sha256'] == INITIAL_SHA,
                f'Unexpected initialization lineage: {run.name}')
        for path in sorted(run.glob('evaluations/update_*/evaluation.json')):
            report = read_json(path)
            if report.get('status') == 'PASS':
                rank(report)
                evaluations.append((path, report))
        for path in run.rglob('*'):
            if path.is_file() and not path.is_symlink() and 'tensorboard' not in path.relative_to(run).parts:
                if path.suffix in ('.json', '.jsonl', '.yaml', '.yml', '.log'):
                    files[path.relative_to(root).as_posix()] = path
        console = run.with_suffix('.console.log')
        if console.is_file() and not console.is_symlink():
            files[console.name] = console
    require(bool(evaluations), 'No passing evaluations found')
    # Stable max over sorted paths; earlier update wins a complete metric tie.
    best = max(evaluations, key=lambda item: (rank(item[1]), -item[1]['completed_updates']))
    final_path = current / 'evaluations/update_006000/evaluation.json'
    final = next((item for item in evaluations if item[0] == final_path), None)
    require(final is not None, 'Final evaluation is absent or failed')
    selected = {}
    for role, (evaluation_path, report) in (('best_nominal', best), ('final_state', final)):
        update = report['completed_updates']
        path = evaluation_path.parents[2] / 'checkpoints' / f'update_{update:06d}.pt'
        require(not path.is_symlink() and path.is_file(), 'Checkpoint missing or symbolic link')
        require(Path(report['checkpoint']).resolve(strict=True) == path.resolve(strict=True), 'Evaluation checkpoint path mismatch')
        receipt_path = path.with_suffix('.json')
        receipt = read_json(receipt_path)
        checksum = sha256(path)
        require(checksum == receipt['sha256'] == report['checkpoint_sha256'], 'Checkpoint SHA-256 mismatch')
        require(receipt['completed_updates'] == update and receipt['size_bytes'] == path.stat().st_size,
                'Checkpoint receipt progress/size mismatch')
        relative = path.relative_to(root).as_posix()
        files[relative] = path
        files[receipt_path.relative_to(root).as_posix()] = receipt_path
        selected[role] = dict(path=relative, sha256=checksum, completed_updates=update,
                              evaluation=evaluation_path.relative_to(root).as_posix(), rank=rank(report))
    require(Path(training['latest_checkpoint']).resolve(strict=True) == files[selected['final_state']['path']].resolve(),
            'Final training report points to a different checkpoint')
    environment = root.parent / 'stage07-setup/environment.json'
    if environment.is_file():
        files['setup/environment.json'] = environment
    for path in sorted(root.glob('capacity-*/*.json')):
        if path.is_file() and not path.is_symlink():
            files[path.relative_to(root).as_posix()] = path
    manifest = dict(schema_version=1, stage=7, stage07_complete=False,
                    created_utc=datetime.now(timezone.utc).isoformat(), cloud_root=str(root),
                    current_run=current.name, training_status=training['status'], selected=selected,
                    scope='Raw run metadata/logs/receipts plus best nominal and final checkpoints; other periodic models remain on cloud',
                    evaluation_count=len(evaluations), files={})
    return files, manifest


def write_archive(root, output):
    files, manifest = collect_plan(root)
    print('Stage07CloudCheckpointReceipts=PASS', file=sys.stderr, flush=True)
    for role, entry in manifest['selected'].items():
        print(f"Selected[{role}]={entry['path']}", file=sys.stderr, flush=True)
    with tarfile.open(fileobj=output, mode='w|gz') as archive:
        for name, path in sorted(files.items()):
            manifest['files'][name] = dict(sha256=sha256(path), size_bytes=path.stat().st_size)
            archive.add(path, arcname=name, recursive=False)
        payload = (json.dumps(manifest, indent=2) + '\n').encode()
        info = tarfile.TarInfo('collection-manifest.json')
        info.size = len(payload)
        info.mode = 0o600
        archive.addfile(info, io.BytesIO(payload))


def verify_extract(archive_path, destination):
    require(destination.is_dir() and not any(destination.iterdir()), 'Extraction requires an empty new directory')
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        require(len(names) == len(set(names)), 'Duplicate archive member')
        for member in members:
            parts = PurePosixPath(member.name)
            require(member.isfile() and not parts.is_absolute() and '..' not in parts.parts and '\\' not in member.name,
                    'Unsafe archive member')
        manifest = json.load(archive.extractfile('collection-manifest.json'))
        require(manifest['schema_version'] == 1 and manifest['stage'] == 7, 'Unknown archive format')
        require(set(names) == set(manifest['files']) | {'collection-manifest.json'}, 'Archive file list mismatch')
        for member in members:
            path = destination / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with archive.extractfile(member) as source, path.open('xb') as target:
                shutil.copyfileobj(source, target)
            if member.name != 'collection-manifest.json':
                receipt = manifest['files'][member.name]
                require(path.stat().st_size == receipt['size_bytes'] and sha256(path) == receipt['sha256'],
                        f'Transfer checksum mismatch: {member.name}')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=26497)
    parser.add_argument('--cloud', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.cloud:
        write_archive(CLOUD_ROOT, sys.stdout.buffer)
        return
    require(1 <= args.port <= 65535, 'Invalid SSH port')
    require(Path('/home/lx').is_dir(), 'Run this script on the lx laptop')
    DOWNLOADS.mkdir(parents=True, exist_ok=True)
    LOCAL_ARTIFACTS.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    with tempfile.NamedTemporaryFile(prefix=f'microduck-stage07-results-{stamp}-', suffix='.partial',
                                     dir=DOWNLOADS, delete=False) as stream:
        partial = Path(stream.name)
        print(f'Stage07Downloading={partial}', flush=True)
        result = subprocess.run([
            'ssh', '-F', '/dev/null', '-p', str(args.port), '-o', 'ConnectTimeout=20',
            '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=3',
            'root@connect.bjb1.seetacloud.com',
            '/root/autodl-tmp/microduck-double-balance/workspace/.venv/bin/python - --cloud',
        ], input=Path(__file__).read_bytes(), stdout=stream)
    require(result.returncode == 0, f'SSH collection failed; partial file retained: {partial}')
    with tempfile.TemporaryDirectory(prefix='.collect-', dir=LOCAL_ARTIFACTS) as temporary:
        manifest = verify_extract(partial, Path(temporary))
        destination = LOCAL_ARTIFACTS / partial.stem
        require(not destination.exists(), 'Destination already exists')
        Path(temporary).rename(destination)
    archive = partial.with_suffix('.tar.gz')
    require(not archive.exists(), 'Archive destination already exists')
    partial.rename(archive)
    checksum = sha256(archive)
    archive.with_suffix('.gz.sha256').write_text(f'{checksum}  {archive.name}\n')
    print('Stage07ResultsDownload=PASS')
    print(f'Archive={archive}\nArchiveMiB={archive.stat().st_size / 1024**2:.2f}')
    print(f'SHA256={checksum}\nLocalArtifacts={destination}')
    print('Selected=' + json.dumps(manifest['selected'], ensure_ascii=False))
    print('请回传上述 tar.gz 供诊断；本机归档已校验，云端原件全部保留。')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError, tarfile.TarError, StopIteration) as exc:
        print(f'Stage07ResultsDownload=FAIL: {exc}', file=sys.stderr)
        sys.exit(1)
