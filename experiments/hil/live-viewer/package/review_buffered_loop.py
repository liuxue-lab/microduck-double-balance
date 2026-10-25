"""Read the exact completed v2 run; never rerun or rewrite its evidence."""
import csv
from pathlib import Path

from bootstrap import STAGE, require, sha
from loop_evidence import read, verify_reports

V2_RUN = STAGE/'native-mujoco-v2/runs/20261025T041428Z-9b2774b5'
V2_REPORT_SHA = '39b91caa9b165ba3522ac7eef27f890824baed5a65a0fc52a71e4ff025efc6bd'
V2_PACKAGE = '641b44af2da16030684244fd3d525a8b6ba081137727dddbb7fa162528a166f0'


def report_v2():
    path = V2_RUN/'report.json'
    require(not path.is_symlink() and sha(path) == V2_REPORT_SHA, 'Reviewed v2 report changed')
    report = read(path)
    require(report.get('package_identity') == V2_PACKAGE and
            report.get('status') == 'NATIVE_MUJOCO_FUNCTIONAL_PASSED_REALTIME_PENDING' and
            report.get('original_records_unchanged') is True and not report.get('errors'),
            'Unexpected v2 result')
    return report


def checked_export(report, name):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts, 'Invalid v2 export path')
    path = V2_RUN
    for part in relative.parts:
        path /= part
        require(not path.is_symlink(), 'Unexpected v2 export symlink')
    pin = report['exported_data'][name]
    require(path.is_file() and path.stat().st_size == pin['bytes'] and sha(path) == pin['sha256'],
            'Reviewed v2 export changed: '+name)
    return path


def timing_details(rows, expected):
    require(len(rows) == 3000 and [int(r['step']) for r in rows] == list(range(1,3001)),
            'Reviewed v2 timing is incomplete')
    epoch = int(rows[0]['release_ns'])
    require(all(int(r['release_ns']) == epoch+i*20_000_000 and
                int(r['deadline_ns']) == int(r['release_ns'])+20_000_000
                for i,r in enumerate(rows)), 'Reviewed v2 release schedule differs')
    late = [r for r in rows if int(r['finished_ns']) > int(r['deadline_ns'])]
    require(len(late) == expected['deadline_misses'], 'V2 CSV and deadline summary disagree')
    peak = max((int(r['finished_ns'])-int(r['release_ns']))/1e6 for r in rows)
    require(abs(peak-expected['all_steps']['release_to_finish_ms']['max']) < 1e-9,
            'V2 CSV and maximum duration disagree')
    def event(row):
        return dict(step=int(row['step']), **{k:float(v) for k,v in row.items()
                    if k.endswith('_ms') and v not in ('',None)})
    return dict(deadline_misses=len(late), deadline_events=[event(r) for r in late],
        peak_reference_tick=event(max(rows,key=lambda r:float(r['reference_inference_ms']))),
        first_tick=event(rows[0]), prior_deadlines_unchanged=True)


def review(destination, write):
    report = report_v2()
    accepted = verify_reports(report['simulator'], report['bridge'], report['orin_server'])
    cases = []
    for i, case in enumerate(report['simulator']['cases'], 1):
        with checked_export(report, 'case-%02d/timing.csv'%i).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        checked_export(report, 'case-%02d/state-data.npz'%i)
        cases.append(dict(case_id=case['case_id'], **timing_details(rows, case['timing'])))
    result = dict(schema='stage12-native-v2-readonly-review-v1', report_sha256=V2_REPORT_SHA,
        package_identity=V2_PACKAGE, functional_closed_loop=accepted['functional_closed_loop'],
        full_loop_realtime_acceptance=accepted['full_loop_realtime_acceptance'], cases=cases,
        old_records_unchanged=True, new_control_ticks=0, new_physics_steps=0, new_Orin_requests=0)
    path = Path(destination)/'reviewed-v2.json'
    if path.exists():
        require(read(path) == result, 'Existing v2 review differs; preserve')
    else:
        write(path, result, exclusive=True)
    return dict(path=str(path), sha256=sha(path), **result)


def compare_states(run, rows):
    """After both cases, compare every published state against frozen v2 data."""
    import numpy as np
    report = report_v2()
    checks = []
    for i, row in enumerate(rows, 1):
        name = 'case-%02d/state-data.npz'%i
        previous = checked_export(report, name)
        current = Path(run)/name
        with np.load(previous, allow_pickle=False) as a, np.load(current, allow_pickle=False) as b:
            require(set(a.files) == set(b.files), 'State export schema differs from v2')
            fields = {}
            for key in a.files:
                left, right = a[key], b[key]
                equal = (left.shape == right.shape and left.dtype == right.dtype and
                         left.tobytes() == right.tobytes())
                fields[key] = dict(shape=list(left.shape), bit_identical=equal)
            checks.append(dict(case_id=row['case_id'], fields=fields,
                               bit_identical=all(x['bit_identical'] for x in fields.values())))
    return dict(previous_report_sha256=V2_REPORT_SHA, comparison='every exported element, dtype and shape',
        cases=checks, all_bit_identical=all(c['bit_identical'] for c in checks),
        previous_run_reexecuted=False)
