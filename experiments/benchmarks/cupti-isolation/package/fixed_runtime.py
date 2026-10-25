"""Persist the exact CUPTI build already validated on the user's RTX 5060.

Only profiler subprocesses preload it. The original venv and historical reports
are immutable inputs. This module has no network, training or robot operations.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

PROJECT = Path('/home/lx/microduck-double-balance/workspace')
STAGE = PROJECT.parent / 'artifacts/double-balance-stage12'
ROOT = STAGE / 'native-pc-timeline-fixed-v2'
LIB_DIR = ROOT / 'runtime/cupti-12.9.79'
LIBRARY = LIB_DIR / 'libcupti.so.12'
EVIDENCE = ROOT / 'evidence/reviewed-repair-report.json'
SOURCE_REPORT = STAGE / 'native-pc-cupti-repair-v1/report.json'
REVIEW_SHA = '1c12ea0d8602dfc3aeb83b9662936b5145a1ef659f936ae1e52e8d1fc2f8e114'
SOURCE_RUN = STAGE / 'native-pc-cupti-repair-v1/runs/20261024T132900Z-205d6d0f'
SOURCE_LIB_DIR = SOURCE_RUN / 'assets/cupti-12.9.79'
OLD_LIBRARY = PROJECT / '.venv/lib/python3.12/site-packages/nvidia/cuda_cupti/lib/libcupti.so.12'
OLD_SHA = 'fccdc5fcc7c32f8cf4b3ed9b17f7f187a185a6fa0a5bcc06e3434cbd1687c808'
NEW_SHA = '2fdab19dc3fccdd4b2f5eba137aabefc68e685f6497bb38eea649866c5672a2a'
LIB_FILES = {
    'libcheckpoint.so': (1407432, '54960e5de63d4e0bf3a852693c3b3f0e21d3e8c69316a1aef566cf9ca48b9ebd'),
    'libcupti.so.12': (7604800, NEW_SHA),
    'libnvperf_host.so': (28578704, '74518220de5c00a7ddd9668c764f2f6891c1e6cf821ee214d24612509a4482bc'),
    'libnvperf_target.so': (5070928, '9d37b772ce043335be27e84a6f79da3f6eb375218172445d8e72d44c4c52ea3e'),
    'libpcsamplingutil.so': (703904, '87746cacc10424695e1878c08adf8ee69db1ea3b955a033dc11a055560495623'),
    'NVIDIA-License.txt': (59262, 'ad6f5853fba0ca0d159d0f58d49ae49830c2f8c93f7a92648b9ce90adb4c6ccd'),
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def regular(path):
    path = Path(path)
    for parent in (path, *path.parents):
        require(not parent.is_symlink(), 'Unexpected symbolic link: ' + str(parent))
    require(path.is_file(), 'Required file missing: ' + str(path))
    return path


def mkdir(path):
    for parent in (path, *path.parents):
        require(not parent.is_symlink(), 'Unexpected symbolic link: ' + str(parent))
    path.mkdir(parents=True, exist_ok=True)


def write(path, row):
    mkdir(path.parent)
    require(not path.is_symlink(), 'Unexpected report link')
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    with temporary.open('x', encoding='utf-8') as stream:
        json.dump(row, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def reviewed(path):
    require(sha(regular(path)) == REVIEW_SHA, 'This is not the reviewed successful GPU report')
    row = json.loads(path.read_bytes())
    require(row['status'] == 'PC_CUPTI_REPAIR_TIMELINE_MEASURED' and
            row['profiling_repair_verified'] is True and not row['errors'] and
            row['application_libraries_verified'] is True and row['original_records_unchanged'] is True,
            'Reviewed profiling repair is incomplete')
    require(row['candidate_library'] == str(SOURCE_LIB_DIR / 'libcupti.so.12') and
            row['candidate_library_sha256'] == NEW_SHA, 'Unexpected reviewed runtime')
    comparison = row['comparison']
    require(comparison['passed'] and comparison['scalar_comparisons'] == 73440 and
            comparison['compared_steps'] == 160 and comparison['all_numeric_state_differences_zero'] and
            all(comparison[k] for k in ('full_audit_equal', 'trial_equal', 'disturbance_equal')),
            'Reviewed project parity did not pass')
    timeline = row['arms']['timeline']['cuda_timeline']
    require(timeline['summary']['status'] == 'CUDA_TIMELINE_CAPTURED' and
            timeline['summary']['cuda_kernel_events'] == 35522 and
            timeline['completed_steps'] == list(range(101, 109)), 'Reviewed GPU coverage incomplete')
    return row


def verify_libraries(directory):
    require(directory.is_dir() and not directory.is_symlink(), 'Missing fixed CUPTI directory')
    for name, (size, digest) in LIB_FILES.items():
        path = regular(directory / name)
        require(path.stat().st_size == size and sha(path) == digest, 'CUPTI file changed: ' + name)
    return directory / 'libcupti.so.12'


def install_runtime():
    """Copy verified bytes to a stable location; never replace a conflicting file."""
    source = EVIDENCE if EVIDENCE.exists() else SOURCE_REPORT
    evidence = reviewed(source)
    require(sha(regular(OLD_LIBRARY)) == OLD_SHA, 'Original installed CUPTI changed')
    if LIB_DIR.exists():
        verify_libraries(LIB_DIR)
    else:
        verify_libraries(SOURCE_LIB_DIR)
        mkdir(LIB_DIR.parent)
        require(shutil.disk_usage(LIB_DIR.parent).free > 128 * 1024**2, 'Need 128 MiB free; no data deleted')
        staging = LIB_DIR.parent / ('cupti-12.9.79.staging-' + uuid.uuid4().hex)
        staging.mkdir()
        for name in LIB_FILES:
            with (staging / name).open('xb') as output:
                with (SOURCE_LIB_DIR / name).open('rb') as input_file:
                    shutil.copyfileobj(input_file, output)
                output.flush()
                os.fsync(output.fileno())
        verify_libraries(staging)
        require(not LIB_DIR.exists() and not LIB_DIR.is_symlink(), 'Fixed directory appeared during installation')
        staging.rename(LIB_DIR)
    mkdir(EVIDENCE.parent)
    if EVIDENCE.exists():
        reviewed(EVIDENCE)
    else:
        raw = regular(source).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == REVIEW_SHA, 'Source report changed during copy')
        with EVIDENCE.open('xb') as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
    return evidence


def fixed_environment(clean):
    require(not clean.get('LD_PRELOAD'), 'Unexpected pre-existing LD_PRELOAD')
    verify_libraries(LIB_DIR)
    env = dict(clean)
    env['LD_PRELOAD'] = str(LIBRARY)
    env['LD_LIBRARY_PATH'] = str(LIB_DIR) + (':' + env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return env


def verify_installed():
    reviewed(EVIDENCE)
    verify_libraries(LIB_DIR)
    require(sha(regular(OLD_LIBRARY)) == OLD_SHA, 'Original installed CUPTI changed')
    receipt = ROOT / 'fix-report.json'
    require(receipt.is_file(), 'Run --apply to finalize the fixed entry first')
    row = json.loads(regular(receipt).read_bytes())
    require(row.get('status') == 'TIMELINE_FIX_INSTALLED_VERIFIED' and
            row.get('reviewed_report_sha256') == REVIEW_SHA and
            row.get('fixed_library_sha256') == NEW_SHA and
            row.get('minimal_passes') == {'torch-only': True, 'warp-before-profiler': True},
            'Fixed runtime has not passed its launch validation')
    return row
