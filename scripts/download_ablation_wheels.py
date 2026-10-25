#!/usr/bin/env python3
"""Fetch the eight blocked wheels with bounded HTTP Range requests and lock hashes.

No dependencies, environment changes, CUDA imports or PPO. Run under setup's
install lock and timeout. Completed 8 MiB pieces survive retries and SSH loss.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import tomllib
from urllib.parse import urlsplit

PINS = {
    'torch': '2.9.1',
    'nvidia-cublas-cu12': '12.8.4.1',
    'nvidia-cudnn-cu12': '9.10.2.21',
    'nvidia-cusparse-cu12': '12.5.8.93',
    'nvidia-cusolver-cu12': '11.7.3.90',
    'nvidia-cusparselt-cu12': '0.7.1',
    'nvidia-nccl-cu12': '2.27.5',
    'warp-lang': '1.12.0',
}
CHUNK = 8 * 1024**2


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def manifest(lock_path):
    data = tomllib.loads(Path(lock_path).read_text())
    result = []
    for name, version in PINS.items():
        packages = [p for p in data['package'] if p['name'] == name
                    and p['version'] == version
                    and p['source'] == {'registry': 'https://pypi.org/simple'}]
        if len(packages) != 1:
            raise ValueError(f'Ambiguous or changed locked package: {name}')
        wheels = []
        for wheel in packages[0]['wheels']:
            url = urlsplit(wheel['url'])
            filename = Path(url.path).name
            python, abi, platform = filename.removesuffix('.whl').rsplit('-', 3)[-3:]
            if (url.scheme == 'https' and url.hostname == 'files.pythonhosted.org'
                and filename.endswith('.whl') and 'manylinux' in platform
                and all(tag.endswith('_x86_64') for tag in platform.split('.'))
                and ((python == 'cp312' and abi == 'cp312')
                     or ('py3' in python.split('.') and abi == 'none'))):
                wheels.append({**wheel, 'name': name, 'version': version, 'filename': filename})
        if len(wheels) != 1:
            raise ValueError(f'Ambiguous CPython 3.12 Linux x86_64 wheel: {name}')
        wheel = wheels[0]
        if not re.fullmatch(r'sha256:[0-9a-f]{64}', wheel['hash']) or wheel['size'] <= 0:
            raise ValueError(f'Invalid lock hash or size: {name}')
        result.append(wheel)
    return result


def fetch_piece(wheel, start, end, directory, stop, *, curl='curl', retries=3):
    expected = end - start + 1
    piece = directory / f'{start:012d}.chunk'
    if piece.exists() and piece.stat().st_size == expected:
        return  # Full-wheel SHA-256 is mandatory before any installation.
    partial = piece.with_suffix('.partial')
    headers = piece.with_suffix('.headers')
    for attempt in range(retries):
        if stop.is_set():
            raise RuntimeError('Download cancelled after another piece failed')
        result = subprocess.run([
            curl, '-q', '--silent', '--show-error', '--fail', '--location',
            '--noproxy', '*', '--proto', '=https', '--proto-redir', '=https',
            '--connect-timeout', '10', '--max-time', '45',
            '--max-filesize', str(expected), '--header', 'Accept-Encoding: identity',
            '--range', f'{start}-{end}', '--dump-header', str(headers),
            '--output', str(partial), wheel['url'],
        ], capture_output=True, text=True, timeout=50)
        response = headers.read_text(errors='replace') if headers.exists() else ''
        ranges = re.findall(r'^content-range:\s*bytes (\d+)-(\d+)/(\d+)\s*$',
                            response, re.I | re.M)
        codes = re.findall(r'^HTTP/\S+\s+(\d+)', response, re.M)
        if (result.returncode == 0 and codes and codes[-1] == '206'
            and ranges and tuple(map(int, ranges[-1])) == (start, end, wheel['size'])
            and partial.exists() and partial.stat().st_size == expected):
            os.replace(partial, piece)
            headers.unlink(missing_ok=True)
            return
        print(f'Stage08RangeRetry={wheel["name"]}:{start} attempt={attempt+1}'
              f' curl_exit={result.returncode}', flush=True)
        partial.unlink(missing_ok=True)
    stop.set()
    raise RuntimeError(f'Range response failed validation: {wheel["name"]} bytes {start}-{end}')


def download(wheels, output, *, chunk=CHUNK, workers=4, fetch=fetch_piece, report=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    total = sum(w['size'] for w in wheels)
    verified = set()
    stop = threading.Event()
    directories = {w['name']: output / 'pieces' / w['hash'].split(':')[1] for w in wheels}
    previous_bytes, previous_time = 0, time.monotonic()

    def progress(phase, current='', *, force=False):
        nonlocal previous_bytes, previous_time
        now = time.monotonic()
        if not force and now - previous_time < 5:
            return
        received = 0
        for w in wheels:
            if w['name'] in verified:
                received += w['size']
                continue
            subtotal = 0
            for p in directories[w['name']].glob('*'):
                if p.suffix in ('.chunk', '.partial'):
                    try:
                        subtotal += p.stat().st_size
                    except FileNotFoundError:
                        pass  # An atomic partial -> completed-piece rename.
            received += min(subtotal, w['size'])
        speed = (max(0, received-previous_bytes) / max(now-previous_time, .001) / 1024**2
                 if phase != 'START' else 0.)
        value = {'phase': phase, 'percent': received/total*100, 'received_bytes': received,
                 'total_bytes': total, 'effective_mib_per_second': speed,
                 'verified_wheels': len(verified), 'wheel_count': len(wheels),
                 'current_wheel': current, 'updated_epoch': time.time()}
        if report:
            temporary = Path(str(report)+'.tmp')
            temporary.write_text(json.dumps(value, indent=2)+'\n')
            os.replace(temporary, report)
        print(f'Stage08Download={value["percent"]:.1f}%'
              f' | {received/1024**2:.1f}/{total/1024**2:.1f} MiB'
              f' | 有效写入 {speed:.2f} MiB/s'
              f' | SHA256通过 {len(verified)}/{len(wheels)} | {phase} {current}', flush=True)
        previous_bytes, previous_time = received, now

    # Discard only unfinished pieces from a prior stopped attempt, never .chunks.
    for w in wheels:
        directories[w['name']].mkdir(parents=True, exist_ok=True)
        for p in directories[w['name']].glob('*.partial'):
            p.unlink()
        destination = output / w['filename']
        if destination.exists():
            if destination.stat().st_size != w['size'] or sha(destination) != w['hash'][7:]:
                raise ValueError(f'Existing wheel failed lock SHA-256: {destination.name}')
            verified.add(w['name'])
    progress('START', force=True)
    for w in wheels:
        if w['name'] in verified:
            continue
        directory = directories[w['name']]
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(fetch, w, start, min(start+chunk, w['size'])-1,
                                       directory, stop)
                           for start in range(0, w['size'], chunk)}
                while futures:
                    done, futures = wait(futures, timeout=1, return_when=FIRST_COMPLETED)
                    for future in done:
                        try:
                            future.result()
                        except BaseException:
                            stop.set()
                            for pending in futures:
                                pending.cancel()
                            raise
                    progress('DOWNLOADING', w['name'])
            temporary = output / (w['filename']+'.assembling')
            with temporary.open('wb') as stream:
                for start in range(0, w['size'], chunk):
                    with (directory / f'{start:012d}.chunk').open('rb') as piece:
                        shutil.copyfileobj(piece, stream, 1024**2)
            if temporary.stat().st_size != w['size'] or sha(temporary) != w['hash'][7:]:
                raise ValueError(f'Assembled wheel failed lock SHA-256: {w["name"]}')
            os.replace(temporary, output / w['filename'])
            verified.add(w['name'])
            shutil.rmtree(directory)
            progress('VERIFIED', w['name'], force=True)
        except BaseException:
            progress('FAILED', w['name'], force=True)
            raise
    requirements = output / 'requirements-locked.txt'
    requirements.write_text(''.join(f'{w["name"]}=={w["version"]} --hash={w["hash"]}\n' for w in wheels))
    progress('ALL_HASHES_VERIFIED', force=True)
    return requirements


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    wheels = manifest(args.lock)
    args.output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    # Retain space for assembled files and the subsequent expanded installation.
    required = sum(w['size'] for w in wheels if not (args.output/w['filename']).exists()) + 12*1024**3
    if shutil.disk_usage(args.output).free < required:
        raise SystemExit('Insufficient free disk space for wheel staging and installation')
    record = {'lock_sha256': sha(args.lock), 'chunk_bytes': CHUNK, 'workers': 4, 'wheels': wheels}
    (args.output/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
    download(wheels, args.output, report=args.report)


if __name__ == '__main__':
    main()
