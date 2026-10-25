#!/usr/bin/env python3
"""Repair the reviewed, zero-physics Stage 10 Batch B startup failure, then resume.

Only diagnostic metadata code and its hash receipts change. Keep strict backend
comparison, all old physics/reward/acceptance, 3 x 0.2 s scope and zero PPO.
Old tool, manifest, plan and stopped-run evidence are backed up first.
No installs, network/Git writes, cloud access or power operations.
"""
from __future__ import annotations

from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
import uuid

SCRIPT = Path('/home/lx/下载/microduck-stage10-substep-diagnostics/microduck-stage10-substep.py')
ART = Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage10')
FOLDER = ART / 'substep/20261001T071355Z-6050d852'
NAME = 'microduck-stage10-substep.py'
ORIGINAL_SHA = '3fd020b9b1ec316bf11d84927978554e9dbc38cc55349d901d51bf2529986630'
REPAIRED_SHA = '7b4df9209da214b2b90556b3c69da8477f42ecde8964cfe98cc18d4f44c879f7'
PLAN_SHA = '1b443e015c11b58b399c1bb8003c49358975a4a3beff3c25b306bdabf865b398'
RUN_SHA = '67c809661072ab63112da8080d1c5d3c11126bb689af9eed7fa15c444a85996f'
RESULT_SHA = '6ba226dafc64ff4be2696664547b62cec7e87483d6026eb9e77a6344a70a9351'
LOG_SHA = '97744f21ae6c46225de8940019ff7226ec42fee239956e341a5dcd5becfe800a'
HELPERS = {
    'stage10_core.py': '92a2d6a908ca06392306115a715dd4f507c5897cbf73ea6cd0610096357dfd75',
    'stage10_trace.py': '6c65c166aa5d6457c9082d2c4edad2e6355b5fe8cec9b8c687832e0daf4848ee',
    'stage10_capture.py': '67bc77de9ae1b75dab3f694bfad70b511101c2d0af120d56148d9799c20ba579',
    'reference-runtime.json': '06ffe5398ce827c9b9e44442c5d393a30c9cfa9409d31a7a7b8ca24667d44ab1',
}
OLD_RUNTIME = "def runtime_identity(torch):\n    names = ('mjlab', 'mujoco', 'mujoco-warp', 'warp-lang', 'rsl-rl-lib', 'torch')\n    identity = {'versions': {name: version(name) for name in names}, 'cuda': torch.version.cuda,\n                'gpu': torch.cuda.get_device_name(0), 'num_threads': torch.get_num_threads(),\n                'backend': core.backend_snapshot(torch),\n                'environment': {name: os.environ.get(name) for name in\n                                ('MUJOCO_GL', 'PYTHONHASHSEED', 'CUBLAS_WORKSPACE_CONFIG', 'CUDA_LAUNCH_BLOCKING')}}\n    expected = read(HERE / 'reference-runtime.json')\n    require(identity == {name: expected[name] for name in identity},\n            'Runtime/backend differs from Batch A; no setting was changed to force a match')\n    return identity"
NEW_RUNTIME = "def runtime_differences(actual, expected, prefix=''):\n    if isinstance(actual, dict) and isinstance(expected, dict):\n        result = []\n        for key in sorted(set(actual) | set(expected)):\n            path = prefix + '.' + key if prefix else key\n            if key not in actual or key not in expected:\n                result.append({'field': path, 'actual': actual.get(key), 'expected': expected.get(key),\n                               'missing_from': 'actual' if key not in actual else 'expected'})\n            else:\n                result.extend(runtime_differences(actual[key], expected[key], path))\n        return result\n    return [] if actual == expected else [{'field': prefix, 'actual': actual, 'expected': expected}]\n\n\ndef runtime_identity(torch, audit_path):\n    # Same version source as Batch A: the loaded torch build, including +cu128.\n    # Distribution metadata (e.g. 2.9.1) is separate evidence, not that build ID.\n    names = ('mjlab', 'mujoco', 'mujoco-warp', 'warp-lang', 'rsl-rl-lib')\n    identity = {'versions': {**{name: version(name) for name in names}, 'torch': str(torch.__version__)},\n                'cuda': torch.version.cuda,\n                'gpu': torch.cuda.get_device_name(0), 'num_threads': torch.get_num_threads(),\n                'backend': core.backend_snapshot(torch),\n                'environment': {name: os.environ.get(name) for name in\n                                ('MUJOCO_GL', 'PYTHONHASHSEED', 'CUBLAS_WORKSPACE_CONFIG', 'CUDA_LAUNCH_BLOCKING')}}\n    reference = read(HERE / 'reference-runtime.json')\n    expected = {name: reference[name] for name in identity}\n    differences = runtime_differences(identity, expected)\n    write(audit_path, {'status': 'MISMATCH' if differences else 'MATCH',\n                      'actual': identity, 'expected': expected, 'differences': differences,\n                      'torch_distribution_version': version('torch'),\n                      'torch_build_version': str(torch.__version__),\n                      'comparison_version_source': 'torch.__version__, identical to Batch A',\n                      'backend_settings_changed_by_check': False, 'new_ppo_updates': 0})\n    summary = '; '.join(str(row['field']) + ': expected=' + repr(row['expected']) +\n                        ', actual=' + repr(row['actual']) for row in differences)\n    require(not differences, 'Runtime/backend differs from Batch A: ' + summary +\n            '; full values saved in ' + str(audit_path))\n    return identity"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    require(path.is_file() and not path.is_symlink(), 'Missing or linked input: ' + str(path))
    return path.read_bytes()


def encode(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode()


def atomic(path, data):
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if path.exists():
            temporary.chmod(path.stat().st_mode & 0o777)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def backup(path, data):
    if path.exists():
        require(load(path) == data, 'Backup differs: ' + str(path))
    else:
        atomic(path, data)


def patch_script(original):
    require(digest(original) == ORIGINAL_SHA, 'Unknown original tool version')
    source = original.decode()
    old_call = "report['runtime_identity'] = runtime_identity(torch)"
    new_call = "report['runtime_identity'] = runtime_identity(torch, output / 'runtime-check.json')"
    require(source.count(OLD_RUNTIME) == 1 and source.count(old_call) == 1, 'Patch anchors differ')
    repaired = source.replace(OLD_RUNTIME, NEW_RUNTIME, 1).replace(old_call, new_call, 1).encode()
    require(digest(repaired) == REPAIRED_SHA, 'Patched code hash differs')
    compile(repaired, NAME, 'exec')
    return repaired


def reviewed_failure(folder):
    require(digest(load(folder / 'plan.json')) == PLAN_SHA, 'Plan differs from reviewed upload')
    require(digest(load(folder / 'run.json')) == RUN_SHA, 'Run state differs from reviewed upload')
    attempt = folder / '01-substep/attempt-37b2fc74'
    require(digest(load(attempt / 'result.json')) == RESULT_SHA, 'Failure result differs from reviewed upload')
    require(digest(load(folder / '01-substep/attempt-37b2fc74.log')) == LOG_SHA, 'Failure log differs')
    require(sorted(p.name for p in attempt.iterdir()) == ['result.json'], 'Unexpected simulation evidence; return it first')
    require(sorted(p.name for p in (folder / '01-substep').iterdir()) == ['attempt-37b2fc74', 'attempt-37b2fc74.log'],
            'Another attempt or completion receipt exists')
    require(not (folder / '02-substep').exists() and not (folder / '03-substep').exists()
            and not (folder / 'comparison.json').exists(), 'Later jobs exist; this narrow migration does not apply')


def repair(script, folder):
    require(folder.is_dir() and not folder.is_symlink(), 'Missing or linked output directory')
    require(not script.parent.is_symlink(), 'Linked diagnostic package directory')
    for name, checksum in HELPERS.items():
        require(digest(load(script.parent / name)) == checksum, 'Diagnostic helper changed: ' + name)
        require(digest(load(folder / 'tool' / name)) == checksum, 'Original tool snapshot changed: ' + name)
    current = load(script)
    require(digest(current) in (ORIGINAL_SHA, REPAIRED_SHA), 'Unknown tool version; no changes made')
    receipt_path = folder / 'runtime-check-fix-v1.json'
    backups = folder / 'runtime-check-fix-v1'
    require(not backups.is_symlink() and not receipt_path.is_symlink(), 'Linked repair path')
    if receipt_path.exists():
        receipt = json.loads(load(receipt_path))
        require(receipt['original_script_sha256'] == ORIGINAL_SHA and receipt['repaired_script_sha256'] == REPAIRED_SHA
                and receipt['script_path'] == str(script) and receipt['output_directory'] == str(folder),
                'Migration receipt identity differs')
        original = load(backups / 'script.original.py')
        old_plan_bytes = load(backups / 'plan.original.json')
        old_manifest_bytes = load(backups / 'manifest.original.json')
        require(digest(old_plan_bytes) == PLAN_SHA and digest(load(backups / 'run.original.json')) == RUN_SHA,
                'Original evidence backup changed')
        require(digest(old_manifest_bytes) == receipt['original_manifest_sha256'], 'Original manifest backup changed')
        require(load(backups / 'snapshot.original.py') == original, 'Original snapshot backup changed')
    else:
        reviewed_failure(folder)
        require(digest(current) == ORIGINAL_SHA, 'Missing migration receipt for changed code')
        original = current
        old_plan_bytes = load(folder / 'plan.json')
        old_manifest_bytes = load(script.parent / 'manifest.json')
        require(load(folder / 'tool' / NAME) == original, 'Original snapshot differs')
        receipt = {'schema_version': 1, 'status': 'PREPARED', 'created_utc': datetime.now(timezone.utc).isoformat(),
                   'script_path': str(script), 'output_directory': str(folder),
                   'original_script_sha256': ORIGINAL_SHA, 'repaired_script_sha256': REPAIRED_SHA,
                   'original_plan_sha256': PLAN_SHA, 'original_manifest_sha256': digest(old_manifest_bytes),
                   'review_zip_sha256': '1d9ff89817eb7e34c6306ed42d2763caa49ad9733cc9f251dc58e6eecc531cee',
                   'reason': 'Use the same torch build-version source as Batch A and save field-level mismatch evidence before raising',
                   'plan_fields_changed': ['tools.microduck-stage10-substep.py'],
                   'strict_backend_comparison_retained': True, 'backend_settings_changed': False,
                   'repository_edited': False, 'cloud_contacted': False, 'new_ppo_updates': 0,
                   'reviewed_attempt_physics_steps': 0, 'stage10_complete': False}
    repaired = patch_script(original)
    old_plan = json.loads(old_plan_bytes)
    require(old_plan['tools'] == {NAME: ORIGINAL_SHA, **HELPERS}, 'Original plan tool map differs')
    new_plan = {**old_plan, 'tools': {**old_plan['tools'], NAME: REPAIRED_SHA}}
    old_manifest = json.loads(old_manifest_bytes)
    matches = [row for row in old_manifest if row['path'] == NAME]
    require(len(matches) == 1 and matches[0]['sha256'] == ORIGINAL_SHA and matches[0]['size_bytes'] == len(original),
            'Original package manifest differs')
    new_manifest = [{**row, 'sha256': REPAIRED_SHA, 'size_bytes': len(repaired)} if row['path'] == NAME else row
                    for row in old_manifest]
    targets = [
        (script, original, repaired),
        (folder / 'tool' / NAME, original, repaired),
        (folder / 'plan.json', old_plan_bytes, encode(new_plan)),
        (script.parent / 'manifest.json', old_manifest_bytes, encode(new_manifest)),
    ]
    # Validate all target bytes before preparing or resuming the transaction.
    for target, before, after in targets:
        require(load(target) in (before, after), 'File changed outside the recorded migration: ' + str(target))
    if not receipt_path.exists():
        backups.mkdir(exist_ok=True)
        for name, content in [('script.original.py', original), ('snapshot.original.py', original),
                              ('plan.original.json', old_plan_bytes), ('manifest.original.json', old_manifest_bytes),
                              ('run.original.json', load(folder / 'run.json'))]:
            backup(backups / name, content)
        backup(backups / 'repair-script.py', load(Path(__file__).resolve()))
        atomic(receipt_path, encode(receipt))
    for target, before, after in targets:
        if load(target) == before:
            atomic(target, after)
        require(load(target) == after, 'Repair verification failed: ' + str(target))
    if receipt['status'] != 'APPLIED':
        receipt.update(status='APPLIED', applied_utc=datetime.now(timezone.utc).isoformat(),
                       repaired_plan_sha256=digest(encode(new_plan)))
        atomic(receipt_path, encode(receipt))
    print('Stage10RuntimeCheckFix=APPLIED; strict backend check retained; zero PPO', flush=True)
    print('Stage10FixReceipt=' + str(receipt_path), flush=True)
    return json.loads(load(folder / 'run.json'))['status']


def main():
    require(not sys.argv[1:], 'Run this repair without arguments')
    require(ART.is_dir(), 'Run on the original laptop')
    # This helper is hash-pinned and stdlib-only. Recheck source before writing.
    core_path = SCRIPT.parent / 'stage10_core.py'
    require(digest(load(core_path)) == HELPERS['stage10_core.py'], 'Unknown source-check helper')
    spec = importlib.util.spec_from_file_location('stage10_repair_core', core_path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    with (ART / 'local-diagnostic.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another local diagnostic controller is running; no files changed')
        core.verify_source()
        status = repair(SCRIPT, FOLDER)
    command = [sys.executable, str(SCRIPT), '--resume', str(FOLDER)]
    print('Stage10Resume=' + shlex.join(command), flush=True)
    if status == 'SUBSTEP_BATCH_COMPLETE_REVIEW_REQUIRED':
        print('Stage10AlreadyComplete=Batch awaits review; no replay relaunched', flush=True)
        return
    # A preflight failure consumed no physics budget. The original controller
    # still blocks replay of any later partial-physics attempt and checks inputs.
    os.execv(sys.executable, command)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print('Stage10RuntimeCheckFixStopped=' + type(exc).__name__ + ': ' + str(exc), flush=True)
        raise SystemExit(1)
