"""Local HTTPS Range, resume/integrity and offline locked-sync integration."""
from contextlib import contextmanager
from functools import partial
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import ssl
import subprocess
import sys
import threading
import zipfile

import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('stage08_download', REPO/'scripts/download_stage08_wheels.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@contextmanager
def range_server(tmp_path, payload, mode='valid'):
    key, certificate = tmp_path/'key.pem', tmp_path/'cert.pem'
    subprocess.run(['openssl', 'req', '-x509', '-newkey', 'rsa:2048', '-nodes', '-days', '1',
                    '-subj', '/CN=localhost', '-addext', 'subjectAltName=IP:127.0.0.1',
                    '-keyout', str(key), '-out', str(certificate)],
                   check=True, capture_output=True, timeout=10)
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_GET(self):
            request_range = self.headers.get('Range', '')
            requests.append(request_range)
            start, end = map(int, request_range.removeprefix('bytes=').split('-'))
            data = payload[start:end+1]
            self.send_response(200 if mode == 'ignored' else 206)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Content-Range',
                             f'bytes {start+(mode=="wrong_range")}-{end}/{len(payload)}')
            self.end_headers()
            self.wfile.write(data)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    tls.load_cert_chain(certificate, key)
    server.socket = tls.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    curl = tmp_path/'curl-with-test-ca'
    curl.write_text('#!/bin/sh\nexec '+shlex.quote(shutil.which('curl'))+
                    ' -q --cacert '+shlex.quote(str(certificate))+' "$@"\n')
    curl.chmod(0o755)
    wheel = {'name': 'fixture', 'version': '1.0', 'filename': 'fixture-1.0-py3-none-any.whl',
             'size': len(payload), 'hash': 'sha256:'+hashlib.sha256(payload).hexdigest(),
             'url': f'https://127.0.0.1:{server.server_port}/fixture.whl'}
    try:
        yield wheel, requests, str(curl)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_range_resume_hash_and_progress(tmp_path):
    payload = b'0123456789ABCDEF'*301
    output = tmp_path/'wheels'
    with range_server(tmp_path, payload) as (wheel, requests, curl):
        directory = output/'pieces'/wheel['hash'][7:]
        directory.mkdir(parents=True)
        (directory/'000000000000.chunk').write_bytes(payload[:1024])
        requirements = module.download([wheel], output, chunk=1024,
            fetch=partial(module.fetch_piece, curl=curl), report=tmp_path/'progress.json')
        assert 'bytes=0-1023' not in requests
        assert all(int(r.split('-')[1])-int(r.split('=')[1].split('-')[0])+1 <= 1024 for r in requests)
        assert (output/wheel['filename']).read_bytes() == payload
        assert wheel['hash'] in requirements.read_text()
        assert json.loads((tmp_path/'progress.json').read_text())['percent'] == 100
        previous = len(requests)
        module.download([wheel], output)
        assert len(requests) == previous  # A verified wheel is not downloaded again.


@pytest.mark.parametrize('mode', ['ignored', 'wrong_range'])
def test_range_response_must_match_request(tmp_path, mode):
    with range_server(tmp_path, b'x'*512, mode) as (wheel, _, curl):
        with pytest.raises(RuntimeError, match='failed validation'):
            module.fetch_piece(wheel, 0, 255, tmp_path, threading.Event(), curl=curl, retries=1)
        assert not list(tmp_path.glob('*.chunk'))


def test_bad_resumed_piece_cannot_be_installed(tmp_path):
    payload = b'0123456789ABCDEF'*128
    output = tmp_path/'wheels'
    with range_server(tmp_path, payload) as (wheel, _, curl):
        directory = output/'pieces'/wheel['hash'][7:]
        directory.mkdir(parents=True)
        (directory/'000000000000.chunk').write_bytes(b'X'*1024)
        with pytest.raises(ValueError, match='SHA-256'):
            module.download([wheel], output, chunk=1024, fetch=partial(module.fetch_piece, curl=curl))
        assert not (output/wheel['filename']).exists()
        assert not (output/'requirements-locked.txt').exists()


def test_manifest_uses_exact_locked_x86_64_wheels():
    wheels = module.manifest(REPO/'uv.lock')
    assert {w['name']: w['version'] for w in wheels} == module.PINS
    assert len(wheels) == 8
    assert all('aarch64' not in w['url'] and '+cu129' not in w['url'] for w in wheels)


def test_registry_seed_is_accepted_by_offline_locked_sync(tmp_path):
    cloud_uv = Path('/root/autodl-tmp/microduck-double-balance/tools/bootstrap/bin/uv')
    uv = str(cloud_uv) if cloud_uv.is_file() else shutil.which('uv')
    if not uv:
        pytest.skip('uv executable is required for the offline installer integration')
    project = tmp_path/'project'
    project.mkdir()
    wheelhouse = tmp_path/'wheels'
    wheelhouse.mkdir()
    name = 'stage08_range_fixture'
    wheel = wheelhouse/f'{name}-1.0-py3-none-any.whl'
    metadata = f'{name}-1.0.dist-info'
    files = {
        f'{name}/__init__.py': 'VALUE = 1\n',
        f'{metadata}/METADATA': 'Metadata-Version: 2.1\nName: stage08-range-fixture\nVersion: 1.0\n',
        f'{metadata}/WHEEL': 'Wheel-Version: 1.0\nGenerator: stage08-test\nRoot-Is-Purelib: true\nTag: py3-none-any\n',
    }
    files[f'{metadata}/RECORD'] = ''.join(f'{p},,\n' for p in [*files, f'{metadata}/RECORD'])
    with zipfile.ZipFile(wheel, 'w') as archive:
        for path, content in files.items():
            archive.writestr(path, content)
    digest = module.sha(wheel)
    (project/'pyproject.toml').write_text('[project]\nname="range-seed-probe"\nversion="0.1.0"\n'
        'requires-python=">=3.12"\ndependencies=["stage08-range-fixture==1.0"]\n')
    (project/'uv.lock').write_text('version = 1\nrevision = 3\nrequires-python = ">=3.12"\n'
        '[[package]]\nname = "range-seed-probe"\nversion = "0.1.0"\nsource = { virtual = "." }\n'
        'dependencies = [{ name = "stage08-range-fixture" }]\n[package.metadata]\n'
        'requires-dist = [{ name = "stage08-range-fixture", specifier = "==1.0" }]\n'
        '[[package]]\nname = "stage08-range-fixture"\nversion = "1.0"\n'
        'source = { registry = "https://pypi.org/simple" }\n'
        f'wheels = [{{ url = "https://files.pythonhosted.org/test/{wheel.name}", '
        f'hash = "sha256:{digest}", size = {wheel.stat().st_size} }}]\n')
    requirements = wheelhouse/'requirements.txt'
    requirements.write_text(f'stage08-range-fixture==1.0 --hash=sha256:{digest}\n')
    env = {**os.environ, 'UV_CACHE_DIR': str(tmp_path/'cache'), 'UV_PYTHON_DOWNLOADS': 'never', 'UV_NO_CONFIG': '1'}
    python = project/'.venv/bin/python'
    commands = [
        ['venv', str(project/'.venv'), '--python', sys.executable, '--offline'],
        ['pip', 'install', '--python', str(python), '--no-deps', '--no-index',
         '--find-links', str(wheelhouse), '--require-hashes', '-r', str(requirements)],
        ['sync', '--locked', '--offline', '--python', str(python)],
    ]
    original_lock = (project/'uv.lock').read_bytes()
    for command in commands:
        result = subprocess.run([uv, *command], cwd=project, env=env,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
    assert (project/'uv.lock').read_bytes() == original_lock
    assert not list((project/'.venv').rglob('direct_url.json'))
