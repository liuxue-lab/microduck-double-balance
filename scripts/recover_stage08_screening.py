#!/usr/bin/env python3
"""Verify the completed A endpoint, renew capacity, then resume B--E once."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from mjlab_microduck.double_balance_stage08_budget import BudgetLedger, atomic_json
from mjlab_microduck.double_balance_stage08_plan import SOURCE_SHA256, TRAIN_SEEDS
from mjlab_microduck.double_balance_stage08_review import checksum

BASE_HEAD = 'c63fa4a4a5a804875288972b38abdbeb57a7eb16'


def verify_completed_a(root, campaign_id):
    """Read-only verification; no conversion of the old FAIL into a PASS."""
    from mjlab_microduck.double_balance_stage08_state import load_checked, require
    manifest_path = root / 'screening/batch.json'
    manifest = json.loads(manifest_path.read_text())
    contract = manifest['contract']
    require(manifest['status'] == 'NEEDS_REVIEW', 'Expected the stopped screening batch')
    require(contract['phase'] == 'screening' and contract['gpu'] == '5090' and
            contract['campaign_id'] == campaign_id and contract['source_sha256'] == SOURCE_SHA256,
            'Screening campaign differs')
    require(contract['datasets'] == str(root / 'initial-states'), 'Initial-state directory differs')
    jobs = manifest['jobs']
    require([j['key'] for j in jobs] == [f'{p}-{TRAIN_SEEDS[0]}' for p in 'ABCDE'], 'Unexpected jobs')
    require(all(j['target_updates'] == 500 for j in jobs) and
            len(jobs[0]['segments']) == 1 and all(not j['segments'] for j in jobs[1:]),
            'This recovery is only for completed A followed by untouched B--E')
    directory = Path(jobs[0]['segments'][0]['directory'])
    require(directory == root / f'screening/A-{TRAIN_SEEDS[0]}/segment-001', 'A directory differs')
    report = json.loads((directory / 'training.json').read_text())
    require(report['status'] == 'FAIL' and not report.get('error') and not report.get('pause_reason') and
            report['experiment_completed_updates'] == 500 and report['git_head'] == BASE_HEAD and
            'timed out after 5 seconds' in report.get('sampler_error', '') and
            'nvidia-smi' in report.get('sampler_error', ''), 'Failure is not the known sampler-only endpoint')
    watchdog = json.loads((directory / 'watchdog.json').read_text())
    require(watchdog['exit_code'] == 1 and watchdog['watchdog_reason'] is None and
            watchdog['child_exited'] and not watchdog['shutdown_performed'], 'Unexpected watchdog failure')
    old_capacity = Path(report['measured_capacity']['summary'])
    require(old_capacity.is_relative_to(root) and checksum(old_capacity) == contract['capacity_sha256'],
            'Original capacity evidence differs')
    require(json.loads(old_capacity.read_text())['status'] == 'PASS', 'Original capacity was not PASS')
    files = [manifest_path, directory / 'training.json', directory / 'watchdog.json', old_capacity]
    hashes = {}
    for completed in (0, 100, 200, 250, 300, 400, 500):
        path = directory / f'checkpoints/update_{completed:06d}.pt'
        receipt = path.with_suffix('.json')
        hashes[completed] = checksum(path)
        require(json.loads(receipt.read_text())['sha256'] == hashes[completed], 'Checkpoint receipt SHA mismatch')
        files.extend((path, receipt))
    endpoint = directory / 'checkpoints/update_000500.pt'
    require(report['latest_checkpoint'] == str(endpoint), 'A endpoint differs')
    _, state, _ = load_checked(endpoint, resume=True, profile='A', seed=TRAIN_SEEDS[0], campaign_id=campaign_id)
    require(state['experiment_completed_updates'] == 500, 'Checkpoint has not completed 500 updates')
    evaluations = []
    for completed in (0, 250, 500):
        for protocol in ('nominal', 'dev'):
            path = directory / f'evaluations/update_{completed:06d}/{protocol}/evaluation.json'
            data = json.loads(path.read_text())
            require(data['status'] == 'PASS' and not data.get('error') and data['protocol'] == protocol and
                    data['checkpoint_sha256'] == hashes[completed], 'Evaluation identity/completion differs')
            files.append(path)
            evaluations.append({'update':completed, 'protocol':protocol,
                **{k:data.get(k) for k in ('successes', 'episodes_count', 'success_fraction', 'policy_objective_status')}})
    return {'status':'PASS', 'old_capacity_summary':str(old_capacity), 'endpoint':str(endpoint),
            'endpoint_sha256':hashes[500], 'completed_updates':500,
            'learning_rate':state['learning_rate'], 'expected_adam_steps':state['expected_adam_steps'],
            'common_step_counter':state['common_step_counter'], 'evaluations':evaluations,
            'evidence_files':[{'path':str(p), 'sha256':checksum(p)} for p in files]}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--attempt', type=Path, required=True)
    args = p.parse_args()
    from mjlab_microduck.double_balance_stage08 import measured_capacity
    from mjlab_microduck.double_balance_stage08_state import ARTIFACTS, environment_preflight, require, utc_now
    root, attempt = ARTIFACTS, args.attempt.resolve()
    require(attempt.is_relative_to(root) and attempt.is_dir(), 'Use the reserved recovery directory')
    status_path = attempt / 'recovery.json'
    require(not status_path.exists(), 'Recovery attempt already exists; inspect its report')
    report = {'status':'VERIFYING_A', 'created_utc':utc_now(), 'stage08_complete':False}
    atomic_json(status_path, report)
    ledger = BudgetLedger(root / 'budget.json')
    try:
        with ledger.job_lock():
            head = environment_preflight('5090')
            evidence = verify_completed_a(root, ledger.identity)
            # Raw original records remain in place, plus an immutable batch snapshot.
            (attempt / 'batch-before.json').write_bytes((root / 'screening/batch.json').read_bytes())
            atomic_json(attempt / 'a-endpoint-verification.json', evidence)
            require(ledger.remaining(training=True) > 2100, 'Insufficient retained budget for bounded capacity')
            report.update(git_head=head, campaign_id=ledger.identity, status='RECHECKING_CAPACITY')
            atomic_json(status_path, report)
        print('Stage08AEndpoint=SHA256_AND_STATE_PASS', flush=True)
        print('Stage08Recovery=RECHECKING_CAPACITY', flush=True)
        subprocess.run([sys.executable, '-u', '-m', 'mjlab_microduck.double_balance_stage08', 'capacity',
            '--gpu', '5090', '--ledger', str(ledger.path), '--checkpoint', str(root / 'references/update_001000.pt'),
            '--datasets', str(root / 'initial-states'), '--output', str(attempt / 'capacity')], check=True)
        capacity_path = attempt / 'capacity/capacity-summary.json'
        with ledger.job_lock():
            measured = measured_capacity(capacity_path, '5090', head)
            for item in evidence['evidence_files']:
                require(checksum(item['path']) == item['sha256'], 'Original recovery evidence changed')
            cases = json.loads(capacity_path.read_text())['cases'].values()
            seconds = max(measured['seconds_per_update'], 3.1)
            evaluation = max(t for c in cases for t in c['evaluation_seconds'])
            saving = max(t for c in cases for t in c['save_seconds'])
            # A: reload + evaluation only. B--E: 4 x 500 updates, 12 eval blocks.
            estimate = 1.2 * (2000 * seconds + 13 * evaluation + 29 * saving + 5 * 30)
            require(ledger.remaining(training=True) >= estimate, 'Remaining screening exceeds retained budget')
            report.update(status='RESUMING_SCREENING', estimated_hours_with_margin=estimate / 3600,
                          remaining_training_hours=ledger.remaining(training=True) / 3600,
                          capacity_summary=str(capacity_path), a_additional_ppo_updates=0,
                          remaining_formal_updates=2000)
            atomic_json(status_path, report)
        print('BudgetAdmission=PASS', flush=True)
        print('Stage08Recovery=A_ENDPOINT_RECHECK_THEN_B_C_D_E', flush=True)
        result = subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('run_stage08_batch.py')),
            'screening', '--gpu', '5090', '--ledger', str(ledger.path),
            '--source', str(root / 'references/update_001000.pt'), '--capacity-summary', str(capacity_path),
            '--previous-capacity-summary', evidence['old_capacity_summary'],
            '--datasets', str(root / 'initial-states'), '--output', str(root / 'screening'), '--resume-batch'])
        report.update(batch_exit_code=result.returncode,
                      status=json.loads((root / 'screening/batch.json').read_text())['status'])
        atomic_json(status_path, report)
        raise SystemExit(result.returncode)
    except Exception as exc:
        report.update(status='NEEDS_REVIEW', error=f'{type(exc).__name__}: {exc}')
        atomic_json(status_path, report)
        raise


if __name__ == '__main__':
    main()
