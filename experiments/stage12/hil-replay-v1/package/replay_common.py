"""Pinned read-only inputs and owned replay outputs; standard library only."""
import hashlib
import json
import os
from pathlib import Path

PROJECT = Path('/home/lx/microduck-double-balance/workspace')
STAGE = PROJECT.parent / 'artifacts/double-balance-stage12'
ROOT = STAGE / 'native-hil-replay-v1'
SOURCE_RUN = STAGE / 'native-mujoco-v3/runs/20261025T044453Z-d194fd47'
SOURCE_SHA = '513c9add8b0029dfb039efba38fc1d1390a97ed310766b8c184d42d87abfd079'
SOURCE_PACKAGE = STAGE / 'native-mujoco-v3/package'
SOURCE_ID = 'a057b01ab6a3bdf2b33a825331fdc631ab39b094010d7df296858a54bb0b8ce4'
PYTHON = PROJECT / '.venv/bin/python'
HERE = Path(__file__).resolve().parent
CASES = ('quiet-2407', 'regression-2500')
MAX_SECONDS = 900


def need(value, message):
    if not value:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024**2), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path = Path(path)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    temporary = path.with_name(path.name+'.tmp-'+str(os.getpid()))
    with temporary.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def package_identity(folder, expected=None):
    folder = Path(folder)
    need(not folder.is_symlink(), 'Package symlink: '+str(folder))
    manifest = read(folder/'manifest.json')
    identity = hashlib.sha256(json.dumps(manifest['files'], sort_keys=True,
                                        separators=(',', ':')).encode()).hexdigest()
    need(identity == manifest['identity'] and (expected is None or identity == expected),
         'Package identity differs: '+str(folder))
    for name, pin in manifest['files'].items():
        path = folder/name
        need(Path(name).name == name and not path.is_symlink() and path.is_file() and
             path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'],
             'Changed package source: '+name)
    return identity


def process_identity(pid):
    try:
        fields = (Path('/proc')/str(int(pid))/'stat').read_text().rsplit(')', 1)[1].split()
        return dict(pid=int(pid), start=fields[19], state=fields[0], group=int(fields[2]))
    except (FileNotFoundError, ProcessLookupError):
        return None


def alive(pin):
    current = process_identity(pin['pid']) if pin else None
    return bool(current and current['start'] == pin['start'] and current['state'] not in ('Z', 'X'))


def input_path(report, name, run=SOURCE_RUN):
    relative = Path(name)
    need(not relative.is_absolute() and '..' not in relative.parts, 'Invalid exported path')
    path = Path(run)
    need(not path.is_symlink(), 'Source run symlink')
    for part in relative.parts:
        path /= part
        need(not path.is_symlink(), 'Export symlink: '+name)
    pin = report['exported_data'][name]
    need(path.is_file() and path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'],
         'Original export changed or missing: '+str(path))
    return path


def verify_source(run=SOURCE_RUN, report_sha=SOURCE_SHA, package=SOURCE_PACKAGE):
    import sys
    run = Path(run)
    need(not run.is_symlink() and sha(run/'report.json') == report_sha, 'Accepted v3 report differs')
    report = read(run/'report.json')
    need(report.get('status') == 'NATIVE_MUJOCO_REALTIME_PASSED' and
         report.get('package_identity') == SOURCE_ID and not report.get('errors') and
         report.get('local_workers_exited') is True and report.get('remote_worker_exit_confirmed') is True and
         all(value == 0 for value in report['local_worker_exit_codes'].values()),
         'Source is not the reviewed completed HIL run')
    package_identity(package, SOURCE_ID)
    sys.path.insert(0, str(package))
    try:
        from loop_evidence import verify_reports
        need(Path(sys.modules['loop_evidence'].__file__).parent == Path(package), 'Unexpected HIL verifier')
        acceptance = verify_reports(report['simulator'], report['bridge'], report['orin_server'])
    finally:
        sys.path.pop(0)
    need(acceptance == report['acceptance'] and acceptance['full_loop_realtime_acceptance'] is True,
         'Recorded acceptance disagrees with original criteria')
    need(report['simulator']['prior_state_comparison']['all_bit_identical'] is True and
         report['original_records_unchanged'] is True and report['added_PPO_updates'] == 0 and
         report['motor_io'] is False, 'Preservation or scope differs')
    for name in report['exported_data']:
        input_path(report, name, run)
    return report
