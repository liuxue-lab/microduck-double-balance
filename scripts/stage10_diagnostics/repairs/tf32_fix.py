#!/usr/bin/env python3
"""Repair only the known Stage 10 TF32 metadata-reader failure, then resume.

Usage: python3 /home/lx/下载/microduck-stage10-tf32-fix.py
No dependency installation, repository edits, precision-setting changes, PPO,
network access, Git writes, cloud access or power operations. Original tool,
plan and stopped-run record are backed up before the narrow tool migration.
"""
from __future__ import annotations
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from datetime import datetime, timezone
import uuid

SCRIPT = Path('/home/lx/下载/microduck-stage10-diagnostics/microduck-stage10-diagnostics.py')
ART = Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage10')
FOLDER = ART / 'replay/20260930T221521Z-e410f3f1'
ORIGINAL_SHA = '2c8b6c8ff33115ed755e9789e3a0e56002263e34dd725c8ed4b6799286fd1e05'
REPAIRED_SHA = 'b6fe8d329f616e63b3e679efc4f58ee96de3aa49022fe2a103575f3719ca6262'
BASE = '00e34c2038771c5d4ad49fe45dff828c0232e60c'
PRIMARY = '86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97'
INITIAL = '253123669696a953830caeaf587e1300e023c12293d608e38e655a38cdc00f58'
OLD_BACKEND = "            'backend': {'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),\n                        'cudnn_benchmark': torch.backends.cudnn.benchmark,\n                        'cudnn_deterministic': torch.backends.cudnn.deterministic,\n                        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,\n                        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32},\n"
NEW_BACKEND = "            # Read the PyTorch 2.9 precision API used by mjlab. Never write\n            # backend settings or access legacy allow_tf32 getters here.\n            'backend': backend_snapshot(torch),\n"
HELPER = 'def backend_snapshot(torch):\n    """Read current settings without switching precision or configuring backends."""\n    return {\'precision_api\': \'fp32_precision\',\n            \'deterministic_algorithms\': torch.are_deterministic_algorithms_enabled(),\n            \'cudnn_benchmark\': torch.backends.cudnn.benchmark,\n            \'cudnn_deterministic\': torch.backends.cudnn.deterministic,\n            \'global_fp32_precision\': torch.backends.fp32_precision,\n            \'matmul_fp32_precision\': torch.backends.cuda.matmul.fp32_precision,\n            \'cudnn_fp32_precision\': torch.backends.cudnn.fp32_precision,\n            \'cudnn_conv_fp32_precision\': torch.backends.cudnn.conv.fp32_precision,\n            \'cudnn_rnn_fp32_precision\': torch.backends.cudnn.rnn.fp32_precision}\n\n\n'


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load(path):
    require(path.is_file() and not path.is_symlink(), 'Missing or linked file: ' + str(path))
    return path.read_bytes()


def encode(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def atomic(path, data, mode=None):
    temporary = path.with_name(path.name + '.tmp-' + uuid.uuid4().hex)
    try:
        with temporary.open('xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if mode is not None:
            temporary.chmod(mode)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def backup(path, data):
    if path.exists():
        require(load(path) == data, 'Backup differs; preserve evidence: ' + str(path))
    else:
        atomic(path, data)


def patch_script(original):
    require(digest(original) == ORIGINAL_SHA, 'Unknown diagnostic version; no changes made')
    text = original.decode('utf-8')
    require(text.count(OLD_BACKEND) == 1 and text.count('def contact_inventory(output):') == 1,
            'Patch anchors differ')
    text = text.replace(OLD_BACKEND, NEW_BACKEND, 1)
    text = text.replace('def contact_inventory(output):', HELPER + 'def contact_inventory(output):', 1)
    result = text.encode('utf-8')
    require(digest(result) == REPAIRED_SHA, 'Patched tool SHA256 differs')
    compile(result, str(SCRIPT), 'exec')
    return result


def check_plan(plan):
    jobs = [{'name': f'{2*i-1:02d}-headless-{i}', 'video_env_id': None, 'repeat': i}
            if j == 0 else {'name': f'{2*i:02d}-video99-{i}', 'video_env_id': 99, 'repeat': i}
            for i in (1, 2, 3) for j in (0, 1)]
    require(plan['script_sha256'] == ORIGINAL_SHA and plan['source_head'] == BASE
            and plan['checkpoint_sha256'] == PRIMARY and plan['dataset_sha256'] == INITIAL
            and plan['jobs'] == jobs and plan['new_ppo_updates'] == 0
            and plan['cloud_gpu_hours'] == 0 and plan['stage10_complete'] is False,
            'Original plan does not match the approved zero-PPO batch')


def repair(script, folder):
    require(folder.is_dir() and not folder.is_symlink(), 'Original output directory missing or linked')
    current = load(script)
    current_sha = digest(current)
    require(current_sha in (ORIGINAL_SHA, REPAIRED_SHA), 'Unknown tool version; no files changed')
    plan_path, state_path = folder / 'plan.json', folder / 'run.json'
    plan_bytes, state_bytes = load(plan_path), load(state_path)
    plan = json.loads(plan_bytes)
    state = json.loads(state_bytes)
    backups = folder / 'tf32-api-fix-v1'
    receipt_path = folder / 'tf32-api-fix-v1.json'
    require(not backups.is_symlink() and not receipt_path.is_symlink(), 'Linked repair path')
    if receipt_path.exists():
        receipt = json.loads(load(receipt_path))
        require(receipt['original_script_sha256'] == ORIGINAL_SHA
                and receipt['repaired_script_sha256'] == REPAIRED_SHA
                and receipt['output_directory'] == str(folder)
                and receipt['script_path'] == str(script), 'Repair receipt differs')
        original = load(backups / 'diagnostics.original.py')
        original_plan_bytes = load(backups / 'plan.original.json')
        require(digest(original_plan_bytes) == receipt['original_plan_sha256'], 'Original plan backup differs')
        require(digest(load(backups / 'run.original.json')) == receipt['original_run_sha256'],
                'Original stopped-run backup differs')
    else:
        require(current_sha == ORIGINAL_SHA, 'Missing migration receipt; preserve evidence')
        require(state['status'] == 'STOPPED' and state['completed_jobs'] == []
                and state['new_ppo_updates'] == 0 and state['cloud_contacted'] is False,
                'Repair requires the stopped preflight with zero completed jobs')
        require(not (folder / 'runtime.json').exists()
                and not (folder / 'contact-inventory.json').exists()
                and not (folder / 'comparison.json').exists(),
                'This repair is only for failure before evaluation')
        check_plan(plan)
        require(not any((folder / job['name']).exists() for job in plan['jobs']),
                'Evaluation attempts exist; preserve them for review')
        probe = folder / 'probe-6c823b.log'
        probe_bytes = load(probe)
        error = probe_bytes.decode('utf-8', errors='replace')
        require('allow_tf32_new' in error and 'mix of the legacy and new APIs' in error,
                'The original probe does not contain the reported TF32 API error')
        original, original_plan_bytes = current, plan_bytes
        # Validate every input and generated byte before any file changes.
        patch_script(original)
        backups.mkdir(exist_ok=True)
        backup(backups / 'diagnostics.original.py', original)
        backup(backups / 'plan.original.json', original_plan_bytes)
        backup(backups / 'run.original.json', state_bytes)
        receipt = {'schema_version': 1, 'status': 'PREPARED',
                   'created_utc': datetime.now(timezone.utc).isoformat(),
                   'script_path': str(script), 'output_directory': str(folder),
                   'original_script_sha256': ORIGINAL_SHA, 'repaired_script_sha256': REPAIRED_SHA,
                   'original_plan_sha256': digest(original_plan_bytes),
                   'original_run_sha256': digest(state_bytes), 'original_probe_sha256': digest(probe_bytes),
                   'original_plan': plan, 'original_stopped_run': state,
                   'reason': 'Replace legacy TF32 metadata getters with read-only fp32_precision getters',
                   'plan_fields_changed': ['script_sha256'], 'new_ppo_updates': 0,
                   'backend_settings_changed_by_fix': False, 'repository_edited': False,
                   'cloud_contacted': False, 'stage10_complete': False}
        atomic(receipt_path, encode(receipt))
    original_plan = json.loads(original_plan_bytes)
    check_plan(original_plan)
    repaired = patch_script(original)
    new_plan = {**original_plan, 'script_sha256': REPAIRED_SHA}
    require(plan == original_plan or plan == new_plan, 'Plan changed outside the recorded migration')
    require(receipt['original_plan'] == original_plan, 'Receipt original plan differs')
    if current_sha == ORIGINAL_SHA:
        atomic(script, repaired, script.stat().st_mode & 0o777)
    if plan != new_plan:
        atomic(plan_path, encode(new_plan))
    require(digest(load(script)) == REPAIRED_SHA and json.loads(load(plan_path)) == new_plan,
            'Repair verification failed; preserve backups')
    if receipt['status'] != 'APPLIED':
        receipt.update(status='APPLIED', applied_utc=datetime.now(timezone.utc).isoformat(),
                       repaired_plan_sha256=digest(load(plan_path)))
        atomic(receipt_path, encode(receipt))
    print('Stage10TF32Fix=APPLIED; read-only precision metadata; zero PPO', flush=True)
    print('Stage10TF32FixReceipt=' + str(receipt_path), flush=True)
    return state.get('status')


def main():
    require(not sys.argv[1:], 'Run this repair without arguments')
    require(ART.is_dir(), 'Run on the original laptop; Stage 10 artifacts are missing')
    with (ART / 'local-diagnostic.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('A diagnostic controller is running; no files changed')
        status = repair(SCRIPT, FOLDER)
    command = [sys.executable, str(SCRIPT), '--resume', str(FOLDER)]
    print('Stage10Resume=' + shlex.join(command), flush=True)
    if status == 'REPLAY_BATCH_COMPLETE_REVIEW_REQUIRED':
        print('Stage10AlreadyComplete=Batch awaits review; no replay relaunched', flush=True)
        return
    # The original controller still enforces all source/checkpoint/input guards,
    # acquires its own lock and re-execs the existing project venv interpreter.
    os.execv(sys.executable, command)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print('Stage10TF32FixStopped=' + type(exc).__name__ + ': ' + str(exc), flush=True)
        raise SystemExit(1)
