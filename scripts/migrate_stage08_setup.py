#!/usr/bin/env python3
"""Carry a failed setup's evidence and original budget to an existing clone.

Stdlib only; no package installation or CUDA computation. Offline recovery
retains the laptop's original deployment clock and labels unavailable evidence.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile

ROOT = Path('/root/autodl-tmp/microduck-double-balance')
KNOWN = {
    '001000': 'c667f96607b68383047f23956ba58805920434465245ef8e32d1148b17fc65b7',
    '006000': 'a5ad0aedb500555b649c5d4b11aded833cbc0f932c1fff1a747e1492534c495c',
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def pack(root, output):
    stage = root / 'artifacts/double-balance-stage08'
    # A setup archive must never stand in for an archive of training results.
    unexpected = {p.name for p in stage.iterdir()} - {
        'budget.json', 'budget.lock', 'budget.job.lock', 'setup', 'references'}
    if unexpected:
        raise ValueError(f'Inspect additional Stage 08 artifacts before migration: {sorted(unexpected)}')
    if not (stage / 'budget.json').is_file():
        raise ValueError('Original budget is missing; do not create a new budget')
    with ExitStack() as stack:
        for path in (stage / 'budget.job.lock', stage / 'setup/install.lock'):
            lock = stack.enter_context(path.open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        files = {'budget.json': stage / 'budget.json'}
        links = []
        for base, prefix in ((stage / 'setup', 'setup'),
                             (root / 'cache/stage08-wheels', 'download-diagnostics')):
            if not base.exists():
                continue
            for path in sorted(base.rglob('*')):
                name = f'{prefix}/{path.relative_to(base).as_posix()}'
                if path.is_symlink():
                    if not path.resolve().is_relative_to(base.resolve()):
                        raise ValueError(f'Unexpected external evidence symlink: {name}')
                    links.append({'path': name, 'target': str(path.readlink())})
                    continue
                if not path.is_file() or path.suffix in ('.lock', '.tmp'):
                    continue
                if prefix == 'download-diagnostics' and path.suffix not in ('.json', '.txt', '.log', '.headers'):
                    continue
                files[name] = path
        if sum(p.stat().st_size for p in files.values()) > 64 * 1024**2:
            raise ValueError('Setup evidence exceeds 64 MiB; inspect before transfer')
        manifest = {'stage': 8, 'purpose': 'failed setup migration',
                    'formal_training_started': False, 'symlinks_recorded': links,
                    'excluded': 'Reproducible dependency wheels/cache; Stage 07 models retained on clone and laptop',
                    'files': [{'path': n, 'sha256': sha(p), 'size_bytes': p.stat().st_size}
                              for n, p in sorted(files.items())]}
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, 'x:gz') as tar:
            for name, path in sorted(files.items()):
                tar.add(path, arcname=name, recursive=False)
            content = (json.dumps(manifest, indent=2) + '\n').encode()
            info = tarfile.TarInfo('MANIFEST.json'); info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return sha(output)


def verify(archive, expected_sha, output):
    if sha(archive) != expected_sha:
        raise ValueError('Archive SHA-256 mismatch')
    if output.exists():
        raise ValueError('Extraction destination already exists')
    with tarfile.open(archive, 'r:gz') as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        if len(set(names)) != len(names) or any(
            not m.isfile() or str(PurePosixPath(m.name)) != m.name or
            PurePosixPath(m.name).is_absolute() or '..' in PurePosixPath(m.name).parts
            for m in members):
            raise ValueError('Unsafe archive paths or members')
        manifest = json.load(tar.extractfile('MANIFEST.json'))
        expected = {v['path']: v for v in manifest['files']}
        if set(expected) | {'MANIFEST.json'} != set(names):
            raise ValueError('Archive membership differs')
        if sum(m.size for m in members) > 65 * 1024**2:
            raise ValueError('Archive exceeds setup evidence limit')
        for name, value in expected.items():
            data = tar.extractfile(name).read()
            if len(data) != value['size_bytes'] or hashlib.sha256(data).hexdigest() != value['sha256']:
                raise ValueError(f'Archive member SHA-256 mismatch: {name}')
        output.mkdir(parents=True)
        for member in members:
            path = output / member.name
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open('xb') as stream:
                shutil.copyfileobj(tar.extractfile(member), stream)
    for name, value in expected.items():
        if sha(output / name) != value['sha256']:
            raise ValueError('Extracted evidence SHA-256 mismatch')
    return output


def carry_budget(source, destination):
    original = json.loads(source.read_text())
    if original['gpu'] != '5090' or original['schema_version'] != 1:
        raise ValueError('Expected the existing Stage 08 5090 budget')
    if destination.exists():
        current = json.loads(destination.read_text())
        for key in ('campaign_id', 'started_at', 'started_epoch', 'gpu',
                    'maximum_cloud_hours', 'finalization_reserve_hours'):
            if current[key] != original[key]:
                raise ValueError(f'Existing clone budget conflicts: {key}')
        for key in ('elapsed_seconds', 'last_observed_epoch'):
            if current[key] < original[key]:
                raise ValueError(f'Existing clone budget is behind the source: {key}')
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open('xb') as stream:
            stream.write(source.read_bytes())


def offline_pack(start_record, retained_root, archive):
    """Retain elapsed time when the old host/UUID cannot be retrieved.

    No original UUID is invented. A replacement identity is persisted locally once,
    explicitly marked reconstructed, and never represents recovered cloud evidence.
    This is only for the setup failure before any formal Stage 08 training.
    """
    from datetime import datetime
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    from mjlab_microduck.double_balance_stage08_budget import create_ledger
    if not start_record.is_file():
        raise ValueError(f'Original local deployment start is missing: {start_record}; do not reset it to now')
    started = start_record.read_text().strip()
    parsed = datetime.fromisoformat(started)
    if parsed.tzinfo is None:
        raise ValueError('Original deployment start must include a timezone')
    stage = retained_root / 'artifacts/double-balance-stage08'
    (stage / 'setup').mkdir(parents=True, exist_ok=True)
    ledger = stage / 'budget.json'
    if ledger.exists():
        state = json.loads(ledger.read_text())
        if state['started_epoch'] != parsed.timestamp() or state['gpu'] != '5090':
            raise ValueError('Retained offline budget conflicts with the original local start')
    else:
        state = create_ledger(ledger, '5090', started)
    evidence = {
        'status': 'OLD_HOST_UNAVAILABLE', 'ssh_observation': 'Connection refused on old port 45743',
        'old_cloud_logs_returned': False, 'old_cloud_ledger_returned': False,
        'old_campaign_id': None, 'replacement_campaign_id': state['campaign_id'],
        'campaign_identity': 'RECONSTRUCTED_ONCE_LOCALLY_NOT_RECOVERED',
        'deployment_start_source': str(start_record), 'deployment_start_sha256': sha(start_record),
        'started_at': state['started_at'], 'budget_reset_to_now': False,
        'formal_training_started': False,
        'formal_training_evidence': 'User-provided preflight says zero Stage 08 PPO updates; old setup failed during dependency download',
        'old_disk_contents': 'UNKNOWN; no shutdown/data-loss inference from refused SSH',
        'follow_up': 'Collect old setup logs/ledger later if accessible; missing evidence stays explicit',
    }
    (stage / 'setup/offline-recovery.json').write_text(json.dumps(evidence, indent=2) + '\n')
    shutil.copyfile(start_record, stage / 'setup/original-deployment-start.txt')
    return pack(retained_root, archive)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('pack', 'offline-pack', 'verify', 'adopt'))
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--sha256')
    p.add_argument('--output', type=Path)
    p.add_argument('--original-start', type=Path)
    args = p.parse_args()
    if args.mode == 'offline-pack':
        if args.original_start is None:
            p.error('Offline recovery requires the retained --original-start file')
        print(offline_pack(args.original_start, args.root, args.archive))
        return
    if args.mode == 'pack':
        if subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                    '--format=csv,noheader'], text=True).strip():
            raise ValueError('Old instance still has GPU compute processes')
        print(pack(args.root, args.archive))
        return
    if not args.sha256 or args.output is None:
        p.error('Verification requires --sha256 and --output')
    evidence = verify(args.archive, args.sha256, args.output)
    if args.mode == 'adopt':
        stage = args.root / 'artifacts/double-balance-stage08'
        carry_budget(evidence / 'budget.json', stage / 'budget.json')
        for update, digest in KNOWN.items():
            destination = stage / 'references' / f'update_{update}.pt'
            if not destination.exists():
                for source in (args.root / 'artifacts/double-balance-stage07').rglob(destination.name):
                    if sha(source) == digest:
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(source, destination)
                        break
            if not destination.is_file() or sha(destination) != digest:
                raise ValueError(f'Archived Stage 07 checkpoint differs: {destination.name}')
        if (evidence / 'setup/offline-recovery.json').exists():
            print('Stage08Budget=ORIGINAL_START_RETAINED_IDENTITY_RECONSTRUCTED')
            print('OldInstanceEvidence=UNAVAILABLE')
        else:
            print('Stage08Budget=ORIGINAL_RETAINED')
        print('Stage08References=SHA256_PASS')
    print('Stage08SetupEvidence=SHA256_PASS')


if __name__ == '__main__':
    main()
