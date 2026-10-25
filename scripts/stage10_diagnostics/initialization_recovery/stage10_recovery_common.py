"""Hash-bound recovery of one reviewed trace; original evidence stays immutable."""
from pathlib import Path
import stage10_core as core
from stage10_core import read, require, checked

HERE = Path(__file__).resolve().parent
REFERENCE = read(HERE / 'recovery-reference.json')
GAP_STATUS = 'INITIALIZATION_TRACES_COMPLETE_WITH_POSTCHECK_GAP_REVIEW_REQUIRED'


def expected_folder():
    return core.ART / 'initialization' / REFERENCE['batch_directory']


def verify_package():
    manifest = read(HERE / 'manifest.json')
    core.verify_inventory(HERE, manifest)
    require({p.name for p in HERE.glob('*.py')} <= {r['path'] for r in manifest},
            'Unlisted Python file in recovery package')


def validate_saved_plan(folder):
    folder = Path(folder)
    require(folder == expected_folder() and folder.is_dir() and not folder.is_symlink(),
            'Recovery is restricted to the reviewed initialization batch')
    verify_package()
    core.verify_inventory(folder, REFERENCE['original_inventory'])
    plan = read(folder / 'plan.json')
    require(plan == REFERENCE['original_plan'], 'Original initialization plan changed')
    checked(plan['checkpoint'], core.PRIMARY)
    checked(folder / 'initial-states/dev.pt', core.INITIAL)
    checked(folder / 'initial-states/dev.json', plan['dataset_receipt_sha256'])
    active = read(folder.parent / 'active-batch.json')
    require(active == {'folder': str(folder), 'tools': plan['tools']},
            'Original active batch registration changed')
    return plan


def validate_approved_first(output):
    output = Path(output)
    require(output == expected_folder() / REFERENCE['approved_attempt'],
            'Only the explicitly reviewed first attempt can be reused')
    first = output.parent
    require(sorted(p.name for p in first.glob('attempt-*')) ==
            [output.name, output.name + '.log'], 'Unexpected extra first-job attempt')
    require(not (first / 'completed.json').exists(), 'Original failed attempt must not be marked complete')
    core.verify_inventory(expected_folder(), REFERENCE['original_inventory'])
    checked(output / 'result.json', REFERENCE['original_result_sha256'])
    report = read(output / 'result.json')
    require(report['status'] == 'FAILED_REVIEW_REQUIRED' and report['recorded_events'] == 33 and
            report['error'] == 'ValueError: Residual assistance cache: _force',
            'Unreviewed first-run failure')
    require('backend_after' not in report and 'source_recheck_after' not in report,
            'Original missing checks must remain missing')
    return report
