#!/usr/bin/env python3
"""Check returned Stage 08 evidence locally without GPU, SSH or training."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

REPO = Path(__file__).resolve().parents[1]
CLOUD_ROOT = Path('/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage08')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def mapped(root, source):
    relative = Path(source).relative_to(CLOUD_ROOT)
    require('..' not in relative.parts, 'Unsafe cloud-relative path')
    target = root / relative
    require(target.resolve().is_relative_to(root.resolve()), 'Returned evidence path escapes root')
    return target


def main():
    audits = REPO / 'docs/audits'
    acceptance = read(audits / 'stage-08-final-acceptance.json')
    require(acceptance['technical_work_complete'] is True, 'Stage 08 technical review incomplete')
    for item in acceptance['evidence_files']:
        require(sha(REPO / item['path']) == item['sha256'], 'Repository evidence changed: ' + item['path'])
    receipt = read(audits / 'stage-08-return-verified.json')
    require(all(receipt[k] is True for k in ('ready_to_shutdown', 'no_remaining_cloud_work',
                'necessary_files_returned_and_sha256_verified')), 'No verified return receipt')
    root = Path(receipt['extracted_directory'])
    require(root.is_dir(), 'Returned evidence directory missing: ' + str(root))
    returned = []
    for item in receipt['verification']['models']:
        path = mapped(root, item['path'])
        require(sha(path) == item['sha256'], 'Selected checkpoint checksum differs: ' + str(path))
        returned.append({'path': str(path), 'sha256': item['sha256']})
    final = read(audits / 'stage-08-final-evaluation-review.json')
    require(sha(root / 'final-selection.json') == final['selection_sha256'], 'Frozen selection differs')
    require(sha(root / 'final-evaluation/batch-evaluation.json') == final['batch_reference']['sha256'],
            'Final evaluation batch differs')
    for item in final['selection']['development_reports']:
        require(sha(mapped(root, item['path'])) == item['sha256'], 'Development selection evidence differs')
    count = 0
    for model in final['models']:
        for protocol, summary in model['protocols'].items():
            ref = summary['report_reference']
            path = mapped(root, ref['path'])
            require(sha(path) == ref['sha256'], 'Final report checksum differs')
            report = read(path)
            require(report['status'] == 'PASS' and report['protocol'] == protocol and
                    report['checkpoint_sha256'] == model['identity']['sha256'], 'Final report/model mismatch')
            require(sha(mapped(root, report['initial_states']['path'])) == report['initial_states']['sha256'],
                    'Initial-state checksum differs')
            require(sha(path.parent / 'trace.pt') == report['trace_sha256'], 'Cloud metric trace differs')
            if protocol.startswith('test-'):
                require(report['selection_sha256'] == final['selection_sha256'], 'Held-out selection mismatch')
            count += 1
    require(count == 25, 'Missing final evaluation protocols')
    local = read(audits / 'stage-08-local-render-report.json')
    package = Path(local['output_directory']) / 'microduck-stage08-local-review.zip'
    require(sha(package) == acceptance['local_video_review']['archive_sha256'], 'Local video package differs')
    with zipfile.ZipFile(package) as archive:
        manifest = json.loads(archive.read('review-files.json'))
        require(len(manifest) == 52 and len(archive.namelist()) == 53, 'Local review package membership differs')
        for item in manifest:
            data = archive.read(item['path'])
            require(len(data) == item['size_bytes'] and hashlib.sha256(data).hexdigest() == item['sha256'],
                    'Local review package member differs: ' + item['path'])
    require(len(local['videos']) == 9 and local['new_ppo_updates'] == 0, 'Local review scope differs')
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    record = {'status': 'PASS', 'checked_utc': datetime.now(timezone.utc).isoformat(),
              'git_head': head, 'selected_models': returned, 'final_protocols': count,
              'local_video_package': str(package), 'local_video_package_sha256': sha(package),
              'new_ppo_updates': 0, 'gpu_compute_executed': False, 'cloud_contacted': False,
              'archive_scope': 'Prior 1353-file return receipt retained; critical selected models, selection, 25 final reports/traces/datasets and 52 video-package members rechecked.',
              'policy_status': acceptance['policy_objective_status'], 'head_posture': 'OBSERVED_UNRESOLVED'}
    output = root.parents[2] / 'finalization' / f'local-finalization-{head[:12]}.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + '\n')
    print('Stage08LocalArchive=PASS')
    print('Stage08FinalProtocols=25; LocalVideos=9; PPOUpdates=0')
    print('Stage08HeadPosture=OBSERVED_UNRESOLVED')
    print('Stage08LocalFinalizationReport=' + str(output))


if __name__ == '__main__':
    main()
