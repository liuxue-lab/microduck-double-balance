"""Read the exact installed baseline package without editing its files."""
import hashlib
import importlib
import json
from pathlib import Path
import sys

PROJECT = Path('/home/lx/microduck-double-balance/workspace')
STAGE = PROJECT.parent / 'artifacts/double-balance-stage12'
BASE_PACKAGE = STAGE / 'native-pc-timeline-v1/package'
BASE_MANIFEST_SHA = 'c320743d583532893a7d7069770f94cfaa6a0818afcc6dbcaa161dc9a61a3bc1'
BASE_PACKAGE_ID = 'ba0b16e7acde7bc8484d9c9519c490110a3556fe528d9276f0cf611732456b54'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_base(package=BASE_PACKAGE):
    require(package.is_dir() and not package.is_symlink(), 'Pinned baseline package missing')
    require(sha(package / 'manifest.json') == BASE_MANIFEST_SHA, 'Baseline manifest changed')
    manifest = json.loads((package / 'manifest.json').read_bytes())
    require(manifest['identity'] == BASE_PACKAGE_ID, 'Baseline identity changed')
    for name, pin in manifest['files'].items():
        path = package / name
        require(Path(name).name == name and path.is_file() and not path.is_symlink() and
                path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'],
                'Changed baseline source: ' + name)
    # Own package stays first; no baseline module with one of our names exists.
    sys.path.insert(1, str(package))
    common = importlib.import_module('common')
    require(Path(common.__file__).parent == package, 'Another baseline module is already imported')
    require(common.verify_package() == BASE_PACKAGE_ID, 'Baseline package incomplete')
    return common
