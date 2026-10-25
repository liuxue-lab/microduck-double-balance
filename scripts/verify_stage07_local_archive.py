"""Verify the already-downloaded Stage 07 deliverables before local tagging.

Standard library only; does not run PPO, simulation, tests, downloads, or Git.
"""
import hashlib
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def verify_file(path, expected):
    require(path.is_file(), f'Missing local artifact: {path}')
    require(digest(path) == expected, f'Local artifact SHA-256 mismatch: {path}')


def verify(report, downloads):
    require(report['technical_work_complete'] is True, 'Technical review not complete')
    require(report['policy_objective_passed'] is False, 'Unexpected policy conclusion; inspect before tagging')
    for spec in report['archives'].values():
        verify_file(downloads / spec['filename'], spec['sha256'])
    training = Path(report['archives']['training']['local_directory'])
    for spec in report['selected_checkpoints'].values():
        path = training / spec['path']
        verify_file(path, spec['sha256'])
        receipt = json.loads(path.with_suffix('.json').read_text())
        require(receipt['sha256'] == spec['sha256'] and receipt['size_bytes'] == path.stat().st_size,
                'Local model receipt mismatch')
        require(receipt['completed_updates'] == spec['completed_updates'], 'Local model update count mismatch')
    diagnostic = Path(report['archives']['diagnostic']['local_directory'])
    for role, spec in report['video_review']['files'].items():
        verify_file(diagnostic / spec['relative_path'], spec['sha256'])
        evidence = json.loads((diagnostic / role / 'diagnostic.json').read_text())
        require(evidence['status'] == evidence['finite_checks'] == evidence['cached_metric_parity'] == 'PASS',
                'Local diagnostic did not pass')
        require(evidence['checkpoint_sha256'] == report['selected_checkpoints'][role]['sha256'],
                'Diagnostic model mismatch')
        require(evidence['success_fraction'] == 0 and evidence['checkpoint_unchanged'] is True,
                'Diagnostic conclusion mismatch')


if __name__ == '__main__':
    repo = Path('/home/lx/microduck-double-balance/workspace')
    report = json.loads((repo / 'docs/audits/stage-07-final-acceptance.json').read_text())
    verify(report, Path('/home/lx/下载'))
    print('Stage07LocalArchive=PASS')
    print('Stage07TechnicalWork=COMPLETE')
    print('Stage07PolicyObjective=UNMET')
