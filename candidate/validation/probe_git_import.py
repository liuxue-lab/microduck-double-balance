"""Developer-only real upstream import test; no Torch, simulator, model or PPO.

Requires an already downloaded source fixture with source-manifest.json for
GitPython 3.1.46 / gitdb 4.0.12 / smmap v5.0.2. Does not download or install it.
Run in a separate Python process because Python audit hooks cannot be removed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage12.git_import import GitImportProbe, GIT
from stage12.host_probe import audit_boundary
from stage12.optional_imports import FontconfigFallback
from stage12.source_capture import file_identity
from stage12.contracts import require


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fixture', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--source-form', choices=['upstream', 'release'], default='release')
    args = ap.parse_args()
    fixture, output = args.fixture.resolve(), args.output.resolve()
    require(not output.exists(), 'Use a new developer-test output directory')
    manifest = fixture / 'source-manifest.json'
    rows = json.loads(manifest.read_bytes())
    require(len(rows) == 59 and {(row['repo'], row['ref']) for row in rows} == {
        ('gitpython-developers/GitPython', '3.1.46'),
        ('gitpython-developers/gitdb', '4.0.12'),
        ('gitpython-developers/smmap', 'v5.0.2')}, 'Wrong source fixture roster')
    for row in rows:
        actual = file_identity(fixture / row['path'])
        require(actual['bytes'] == row['size'] and actual['git_blob_sha1'] == row['sha'],
                'Fixture source differs: ' + row['path'])
    sys.dont_write_bytecode = True
    for key in tuple(os.environ):
        if key.startswith('GIT_'):
            os.environ.pop(key)
    output.mkdir(parents=True)
    if args.source_form == 'release':
        derived = output / 'release-source'
        for row in rows:
            path = derived / row['path']
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(fixture / row['path'], path)
        init = derived / 'git/__init__.py'
        raw = init.read_bytes()
        require(raw.count(b'__version__ = "git"') == 1, 'Unexpected version stamp location')
        init.write_bytes(raw.replace(b'__version__ = "git"', b"__version__ = '3.1.46'"))
        fixture = derived
    sys.path.insert(0, str(fixture))
    os.chdir(output)
    guard = GitImportProbe()
    audit_boundary(output, FontconfigFallback(), guard)
    dist = SimpleNamespace(version='3.1.46', locate_file=lambda name: fixture / name)
    guard.configure(ROOT, output, lambda name: dist)
    guard.preload()  # Real import: includes gitdb, smmap and two real git version children.
    require(guard.report['completed_queries'] == 2 and not guard.active, 'Import window did not close')
    denied = []
    for argv in ([GIT, 'version'], [GIT, 'status'], [GIT, 'push'], ['/bin/true'], ['/bin/sh', '-c', 'git version']):
        try:
            subprocess.run(argv, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.DEVNULL)
        except RuntimeError:
            denied.append(argv)
        else:
            raise RuntimeError('Unexpected subprocess permitted after import: ' + repr(argv))
    require(len(guard.report['permitted_queries']) == 2, 'Extra process escaped the boundary')
    guard.verify_unchanged()
    result = dict(status='PASS_REAL_UPSTREAM_GITPYTHON_IMPORT', fixture_files_verified=59,
                  source_form=args.source_form,
                  fixture_manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
                  git_import=guard.report, denied_after_import=denied,
                  host_installed_stack_tested=False, native_integrations=0, PPO=0, cloud_training=0)
    (output / 'git-import-integration.json').write_text(json.dumps(result, indent=2, sort_keys=True)+'\n')
    print(json.dumps(dict(status=result['status'], real_git_version_queries=2,
                         other_processes_denied=len(denied), native=0, PPO=0)))


if __name__ == '__main__':
    main()
