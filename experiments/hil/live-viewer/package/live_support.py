"""Pure-standard-library startup and completion checks for the live extension."""
import json
import os
from pathlib import Path
from bootstrap import STAGE, require, sha

V3_RUN = STAGE / 'native-mujoco-v3/runs/20261025T044453Z-d194fd47'
V3_SHA = '513c9add8b0029dfb039efba38fc1d1390a97ed310766b8c184d42d87abfd079'
MODEL_SHA = '2e481a6dfa1508d2345de0821897278d403b80535d7d0514b571724d6091c645'


def read(path):
    return json.loads(Path(path).read_bytes())


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp-' + str(os.getpid()))
    with temporary.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    os.replace(temporary, path)


def accepted_source():
    path = V3_RUN / 'report.json'
    require(not path.is_symlink() and sha(path) == V3_SHA, 'Accepted v3 report changed')
    report = read(path)
    require(report['status'] == 'NATIVE_MUJOCO_REALTIME_PASSED' and
            report['full_loop_realtime_acceptance'] is True, 'Accepted v3 status differs')
    names = ('case-01/native-model.mjb', 'case-01/state-data.npz', 'case-02/state-data.npz')
    paths = {}
    for name in names:
        member = V3_RUN
        for part in Path(name).parts:
            member /= part
            require(not member.is_symlink(), 'Unexpected accepted source symlink')
        pin = report['exported_data'][name]
        require(member.stat().st_size == pin['bytes'] and sha(member) == pin['sha256'],
                'Accepted v3 input changed: ' + name)
        paths[name] = str(member)
    require(report['exported_data'][names[0]]['sha256'] == MODEL_SHA, 'Visual model identity differs')
    return dict(report_path=str(path), report_sha256=V3_SHA, paths=paths,
                model_sha256=MODEL_SHA, old_records_unchanged=True)


def viewer_cpu(placement):
    rows = placement.get('candidates', [])
    reserved = {(r['package'], r['core']) for r in rows
                if r['cpu'] in (placement.get('control_cpu'), placement.get('bridge_cpu'))}
    eligible = [r for r in rows if (r['package'], r['core']) not in reserved]
    require(placement.get('enabled') and eligible and len(reserved) == 2,
            'Live display needs a third eligible physical CPU core; accepted v3 stays available')
    chosen = min(eligible, key=lambda r: (r['busy_fraction'], -r['maximum_frequency_kHz'], r['cpu']))
    return chosen['cpu']


def visual_result(viewer, simulator):
    cases = viewer.get('cases', {})
    checks = dict(clean_viewer_exit=viewer.get('status') == 'LIVE_VIEWER_COMPLETE',
                  no_visual_physics=viewer.get('physics_steps') == 0,
                  no_visual_policy=viewer.get('policy_inferences') == 0,
                  both_cases_seen=set(cases) == {'1', '2'})
    for number in (1, 2):
        seen = cases.get(str(number), {})
        checks['case%d_visible' % number] = bool(
            seen.get('fresh_frames', 0) >= 500 and seen.get('first_step', 3001) <= 10 and
            seen.get('last_step', 0) >= 2990 and seen.get('max_age_ms', 1000) <= 500 and
            seen.get('max_step_gap', 3001) <= 25)
    rows = simulator.get('cases', [])
    checks['both_publishers_valid'] = len(rows) == 2 and all(
        r.get('runtime_adapters', {}).get('state_export', {}).get('live_display', {}).get('send_errors') == 0
        and r['runtime_adapters']['state_export']['live_display'].get('deadline_includes_publication') is True
        for r in rows)
    return dict(complete=all(checks.values()), checks=checks,
                criteria='per case >=500 fresh frames, first<=10, last>=2990, age<=500ms, step gaps<=25',
                realtime_acceptance_is_separate=True)


def compare_v3(run, rows):
    import numpy as np
    source = accepted_source()
    cases = []
    for i, row in enumerate(rows, 1):
        name = 'case-%02d/state-data.npz' % i
        with np.load(source['paths'][name], allow_pickle=False) as a, np.load(Path(run)/name, allow_pickle=False) as b:
            require(set(a.files) == set(b.files), 'Export fields differ from accepted v3')
            checks = {key: a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
                      and a[key].tobytes() == b[key].tobytes() for key in a.files}
            cases.append(dict(case_id=row['case_id'], fields=checks, bit_identical=all(checks.values())))
    return dict(previous_report_sha256=V3_SHA, cases=cases,
                all_bit_identical=all(r['bit_identical'] for r in cases), old_run_reexecuted=False)
