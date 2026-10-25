"""Approved six-case, original-material CPU diagnostic. No policy or PPO.

The archived validator and every material parameter remain frozen. Floor arrival
censors a case; it never clears the old floor failures or completes the stage.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import time
import traceback
import zipfile

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_loader = importlib.util.spec_from_file_location('stage11_friction_frozen_floor', ROOT/'scripts/diagnose_head_contact_floor_impact.py')
floor = importlib.util.module_from_spec(_loader); _loader.loader.exec_module(floor)
v2 = floor.v2
require, save, digest = floor.require, floor.save, floor.digest
SOURCE = floor.SOURCE
REFINEMENT = Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage11/floor-refinement/20261003T173938Z')
REFINEMENT_ZIP_SHA256 = '5085c2ec989361d0523cbad27cbadb7927cb928f5f39175bb5f10ef8a81c394a'
REFINEMENT_COMMIT = '234030e0f701e788f2d649c31b98ba76a5cd8934'
FROZEN = {
    'src/mjlab_microduck/robot/head_contact.py': '8a2446aa27d8e420e6e26e5e9985ab24cb774036936f393de680857edfd0af64',
    'scripts/validate_head_contact_physics.py': 'd5f88ffc2a65c3d98a6714d8aa67b8bffe8144f81be6e1d1fd6d89dcb908067c',
    'scripts/diagnose_head_contact_floor_impact.py': '130dee8548e9a1127380162254d6a2d70bf710e30622927bafc5bdefe7d620a7',
    'scripts/diagnose_head_contact_floor_stiffness.py': 'e2f52491480582fbdd754fc18a5e5a5b86e7a59b01afa97d6e34327c8163b54a',
    'scripts/diagnose_head_contact_floor_refinement.py': '7e4005e965e2ff725b29534221689c719eab6299441874b298c28bbb9560abab',
}
CASE_NAMES = ('slide-x-plus', 'slide-x-minus', 'slide-y-plus', 'slide-y-minus', 'initial-no-slip-roll', 'normal-spin')
DT, DURATION, MAX_STEPS, TOTAL_STEPS, WALL_SECONDS = .002, 3., 1500, 9000, 300
PENETRATION = .001
CASE_COMPLETE = 'FULL_HORIZON_PHYSICAL_GATES_REVIEW_REQUIRED'
FLOOR_ENDPOINTS = ('FLOOR_EVENT_CENSORED', 'FLOOR_PENETRATION_BLOCKED')
ERROR = 'ERROR_STOP_MATRIX'
EXTRA_INPUTS = ('local-v2/material-geometry.npz', 'local-v2/cpu/release-center/initial.json',
                'local-v2/static-geometry.json', 'local-v2/structure-inertia.json')
PRIOR_INPUTS = ('run-summary.json', 'diagnostic/preflight.json', 'diagnostic/diagnostic-review.json', 'diagnostic/input-provenance.json')


def pinned_files(directory, zip_sha, names):
    directory = Path(directory); archive = directory.with_suffix('.zip')
    require(archive.is_file() and digest(archive) == zip_sha, 'Pinned archive missing or changed: '+str(archive))
    result = {}
    with zipfile.ZipFile(archive) as z:
        require(len(z.namelist()) == len(set(z.namelist())), 'Duplicate archive paths')
        inventory = json.loads(z.read('evidence-inventory.json'))
        require((directory/'evidence-inventory.json').read_bytes() == z.read('evidence-inventory.json'), 'Inventory changed')
        for name in names:
            path = directory/name
            require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(directory.resolve()), 'Missing or unsafe input: '+name)
            expected = hashlib.sha256(z.read(name)).hexdigest()
            require(digest(path) == expected == inventory[name]['sha256'], 'Input SHA mismatch: '+name)
            result[name] = {'sha256': expected, 'bytes': path.stat().st_size}
    return result


def verify_inputs(current):
    for name, sha in FROZEN.items(): require(digest(ROOT/name) == sha, 'Frozen source changed: '+name)
    original = floor.verify_source(SOURCE)
    extra = pinned_files(SOURCE, floor.SOURCE_ZIP_SHA256, EXTRA_INPUTS)
    previous = pinned_files(REFINEMENT, REFINEMENT_ZIP_SHA256, PRIOR_INPUTS)
    run = floor.load_json(REFINEMENT/'run-summary.json')
    prior = floor.load_json(REFINEMENT/'diagnostic/diagnostic-review.json')
    pre = floor.load_json(REFINEMENT/'diagnostic/preflight.json')
    require(all(pre[k] == current[k] for k in ('head', 'versions', 'python')), 'Runtime differs from canonical C3/C4')
    require(run['head'] == run['head_after'] == v2.BASELINE and run['canonical_total_known_completed_steps'] == 6211
            and run['physics_integration_steps'] == run['physics_attempted_steps'] == 1200
            and run['execution_invariants_unchanged'], 'Unexpected prior baseline/count/invariants')
    require(prior['status'] == 'STEP_REFINEMENT_SENSITIVITY_BLOCKED_REVIEW_REQUIRED'
            and prior['diagnostic_source_sha256'] == FROZEN['scripts/diagnose_head_contact_floor_refinement.py']
            and not prior['comparison']['sensitivity']['all_four_pass'], 'Prior sensitivity failure must be preserved')
    old_source = floor.load_json(REFINEMENT/'diagnostic/input-provenance.json')['original_material']
    require(old_source['zip_sha256'] == original['zip_sha256'] and old_source['files'] == original['files'], 'Original input chain differs from C3/C4')
    geometry = floor.load_json(SOURCE/'local-v2/static-geometry.json')
    require(geometry['v2_status'] == 'PASS' and geometry['count'] == 49, 'Original v2 static gate not verified')
    require(floor.load_json(SOURCE/'local-v2/structure-inertia.json')['status'] == 'PASS', 'Original inertia audit not verified')
    old_cases = floor.load_json(SOURCE/'local-v2/physics-review.json')['cases']
    require(len(old_cases) == 4 and not any(c['case'] in CASE_NAMES for c in old_cases), 'Six friction cases were already attempted in source archive')
    return {'original_material': {**original, 'files': {**original['files'], **extra}},
            'refinement': {'directory': str(REFINEMENT), 'artifact_commit': REFINEMENT_COMMIT,
                           'zip_sha256': REFINEMENT_ZIP_SHA256, 'files': previous}, 'frozen_sources': FROZEN,
            'prior_canonical_steps': 6211, 'old_experiments_replayed': False}


def selected_cases(triangles, center_xy, archived_center):
    matrix = v2.dynamic_matrix(triangles, center_xy)
    require(len(matrix) == 13 and len({c['name'] for c in matrix}) == 13, 'Original case matrix changed')
    center = matrix[0]
    for key in ('center', 'witness', 'normal', 'velocity', 'omega'):
        require(np.array_equal(center[key], archived_center[key]), 'Original center initialization differs: '+key)
    selected = [c for c in matrix if c['name'] in CASE_NAMES]
    require(tuple(c['name'] for c in selected) == CASE_NAMES, 'Unapproved case selection/order')
    return selected


def configured_copy(original):
    signature = floor.model_signature(original); model = copy.deepcopy(original)
    require(floor.model_signature(model) == signature, 'Copied material model differs')
    sphere, ground = model.geom('sphere').id, model.geom('floor').id
    require(model.nq == 7 and model.nv == 6 and model.nu == model.na == model.neq == 0, 'Not the fixed material sphere fixture')
    require(model.opt.timestep == DT and model.opt.disableflags == 0 and np.array_equal(model.opt.gravity, [0, 0, -9.81]), 'Original timestep/options required')
    require(np.array_equal(model.geom_solref[sphere], [.003, 1.]) and model.geom_condim[sphere] == 4
            and np.array_equal(model.geom_friction[sphere], [.6, .005, .0001]), 'Original sphere material required')
    require(model.geom_type[ground] == mujoco.mjtGeom.mjGEOM_PLANE and model.geom_bodyid[ground] == 0
            and np.array_equal(model.geom_quat[ground], [1, 0, 0, 0]) and
            model.geom_type[sphere] == mujoco.mjtGeom.mjGEOM_SPHERE and np.array_equal(model.geom_pos[sphere], [0, 0, 0]),
            'Expected horizontal floor and centered sphere')
    return model, signature


def post_geometry(model, probe, data, triangles):
    """Collision/geometry only in separate data; no force attributed to post-state."""
    probe.qpos[:] = data.qpos; probe.qvel[:] = data.qvel; probe.time = data.time
    probe.qacc_warmstart[:] = 0
    mujoco.mj_forward(model, probe)
    sphere, head, ground = [model.geom(n).id for n in ('sphere', 'head', 'floor')]
    rows = []
    for contact in probe.contact:
        a, b = map(int, contact.geom)
        normal = np.asarray(contact.frame[:3]) * (1 if b == sphere else -1)
        other = b if a == sphere else a
        item = {'geom': [a, b], 'other': model.geom(other).name, 'dist': float(contact.dist),
                'allowed': sphere in (a, b) and other in (head, ground),
                'pos': contact.pos.copy(), 'normal_toward_ball': normal}
        if item['allowed'] and other == head:
            witness = contact.pos - .5*contact.dist*normal
            item['geometry'] = v2.geometry_metrics(data.qpos[:3], witness, normal, contact.dist, triangles)
        rows.append(item)
    _, distance, _ = v2.nearest_surface(data.qpos[:3], triangles)
    floor_gap = float(data.qpos[2]-model.geom_pos[ground, 2]-model.geom_size[sphere, 0])
    return {'phase': 'post_integration_geometry_only_no_force_claim', 'time_s': float(data.time), 'contacts': rows,
            'floor_gap_m': floor_gap, 'floor_overlap_m': max(0., -floor_gap),
            'original_head_surface_gap_m': float(distance-model.geom_size[sphere, 0]),
            'head_overlap_m': max([0.] + [-c['dist'] for c in rows if c['other'] == 'head'])}


def classify_step(rows, post):
    """Head/unexpected errors take precedence over an otherwise censorable floor."""
    for row in rows:
        require(row['allowed'] and row['other'] in ('head', 'floor'), 'Unexpected pre-step contact')
        require(row['dim'] == 4, 'Effective condim changed')
        for key, expected in [('friction', [.6, .6, .005, .0001, .0001]), ('solref', [.003, 1.]), ('solimp', [.95, .99, .001, .5, 2.])]:
            require(np.allclose(row[key], expected, rtol=1e-6, atol=1e-9), 'Effective '+key+' changed')
    for row in list(rows)+post['contacts']:
        require(row['allowed'] and row['other'] in ('head', 'floor'), 'Unexpected post-step contact')
        if row['other'] == 'head':
            require(-row['dist'] <= PENETRATION, 'Head penetration exceeds 1 mm')
            require(v2.selected_geometry_pass(row['geometry'], 'v2'), 'Head v2 geometry gate failed')
    require(post['head_overlap_m'] <= PENETRATION, 'Post-state head penetration exceeds 1 mm')
    floor_rows = [r for r in list(rows)+post['contacts'] if r['other'] == 'floor']
    if floor_rows or post['floor_overlap_m'] > 0:
        maximum = max([post['floor_overlap_m']] + [-r['dist'] for r in floor_rows])
        return FLOOR_ENDPOINTS[1] if maximum > PENETRATION else FLOOR_ENDPOINTS[0]
    return None


def new_summary(case):
    return {'name': case['name'], 'status': 'STARTED', 'steps': 0, 'attempted_steps': 0,
            'horizon_completed': False, 'fixed_material_fixture': True, 'policy_success': None,
            'new_ppo_updates': 0, 'full_robot_steps': 0, 'head_contact_steps': 0,
            'floor_contact_steps': 0, 'max_head_slip_m_s': 0., 'max_head_penetration_m': 0.,
            'max_post_head_overlap_m': 0., 'max_floor_penetration_m': 0.,
            'head_normal_impulse_Ns': 0., 'head_tangential_world_impulse_Ns': np.zeros(3),
            'head_torsional_impulse_Nms': 0., 'head_tangential_work_J': 0., 'head_torsional_work_J': 0.,
            'max_positive_head_tangential_power_W': 0., 'max_positive_head_torsional_power_W': 0.,
            'head_tangential_positive_power_samples': 0, 'head_torsional_positive_power_samples': 0,
            'max_head_cone_utilization': 0., 'legacy_v1_failed_contact_count': 0,
            'head_contact_intervals': [], 'no_head_contact_intervals': [],
            'critical_frame_steps': [], 'physical_friction_calibration': False}


def accumulate(summary, rows, post, step):
    head = [r for r in rows if r['other'] == 'head']
    summary['head_contact_steps'] += bool(head)
    summary['floor_contact_steps'] += any(r['other'] == 'floor' for r in rows)
    summary['max_post_head_overlap_m'] = max(summary['max_post_head_overlap_m'], post['head_overlap_m'])
    summary['max_floor_penetration_m'] = max([summary['max_floor_penetration_m'], post['floor_overlap_m']]+[-r['dist'] for r in rows if r['other'] == 'floor'])
    for row in head:
        if row['slip_speed_m_s'] > summary['max_head_slip_m_s']:
            summary['max_head_slip_m_s'] = row['slip_speed_m_s']; summary['max_head_slip_step'] = step
        if -row['dist'] > summary['max_head_penetration_m']:
            summary['max_head_penetration_m'] = -row['dist']; summary['max_head_penetration_step'] = step
        summary['legacy_v1_failed_contact_count'] += not row['geometry']['legacy_v1_pass']
        summary['head_normal_impulse_Ns'] += row['normal_force_N']*DT
        # The original contact frame gives force on geom2; orient to sphere.
        sign = 1 if np.dot(np.asarray(row['frame'])[:3], row['normal_toward_ball']) > 0 else -1
        tangent = sign*(np.asarray(row['frame']).reshape(3, 3).T@np.r_[0., np.asarray(row['force'])[1:3]])
        summary['head_tangential_world_impulse_Ns'] += tangent*DT
        summary['head_torsional_impulse_Nms'] += row['torsional_torque_Nm']*DT
        for kind in ('tangential', 'torsional'):
            power = row[kind+'_power_W']
            summary['head_'+kind+'_work_J'] += power*DT
            summary['max_positive_head_'+kind+'_power_W'] = max(summary['max_positive_head_'+kind+'_power_W'], power)
            summary['head_'+kind+'_positive_power_samples'] += power > 0
        if row['pyramidal_cone_utilization'] is not None:
            summary['max_head_cone_utilization'] = max(summary['max_head_cone_utilization'], row['pyramidal_cone_utilization'])
    return bool(head)


def contact_intervals(flags):
    result = []
    edges = np.flatnonzero(np.diff(np.r_[False, flags, False]))
    for first, end in edges.reshape(-1, 2):
        result.append({'first_pre_step': int(first), 'end_step_exclusive': int(end),
                       'start_s': float(first*DT), 'end_s': float(end*DT),
                       'right_censored_at_record_end': bool(end == len(flags)),
                       'interpretation': 'Sampled pre-step contact intervals, not continuous-time retention proof'})
    return result


def run_case(original, triangles, case, destination, deadline, progress):
    require(case['name'] in CASE_NAMES, 'Unapproved friction case')
    destination.mkdir(parents=True, exist_ok=False)
    summary = new_summary(case); model = None; signature = None
    poses, velocities, times, head_flags = [], [], [], []
    started = time.monotonic()
    try:
        model, signature = configured_copy(original)
        data = v2.set_ball(model, case['center'], case['velocity'], case['omega']); probe = mujoco.MjData(model)
        require(all(c.dist >= -1e-8 for c in data.contact), 'Initial surface intersection')
        save(destination/'initial.json', {**case, 'qpos': data.qpos, 'qvel': data.qvel,
              'qacc_warmstart': data.qacc_warmstart, 'time_s': data.time,
              'method': 'Original set_ball initialization; new data per case, no replay or discarded warmup'})
        poses.append(data.qpos.copy()); velocities.append(data.qvel.copy()); times.append(float(data.time))
        warnings = data.warning.number.copy()
        with gzip.open(destination/'contacts.jsonl.gz', 'wt') as flat, \
             gzip.open(destination/'raw-contacts.jsonl.gz', 'wt') as raw, \
             gzip.open(destination/'step-log.jsonl.gz', 'wt') as log:
            for step in range(MAX_STEPS):
                require(time.monotonic() < deadline, 'Approved 300 second physics wall limit reached')
                require(abs(data.time-step*DT) < 1e-11, 'Pre-step clock drift/reset')
                before_q, before_v = data.qpos.copy(), data.qvel.copy()
                summary['attempted_steps'] += 1; progress(summary)
                mujoco.mj_step(model, data)
                summary['steps'] += 1; progress(summary)
                poses.append(data.qpos.copy()); velocities.append(data.qvel.copy()); times.append(float(data.time))
                require(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(), 'Nonfinite physical state')
                require(abs(data.time-(step+1)*DT) < 1e-11, 'Post-step clock drift/reset')
                require(np.array_equal(warnings, data.warning.number), 'New runtime warning')
                records = v2.cpu_records(model, data)
                raw.write(json.dumps({'step': step, 'phase': 'pre_integration_contacts_and_solver_force', 'contacts': records}, default=v2.serial, allow_nan=False)+'\n'); raw.flush()
                rows = v2.contact_rows(model, before_q, before_v, records, step, 'cpu', triangles)
                for row in rows: flat.write(json.dumps(row, default=v2.serial, allow_nan=False)+'\n')
                flat.flush()
                require(len(rows) == len(records), 'Unexpected non-sphere contact')
                post = post_geometry(model, probe, data, triangles)
                log.write(json.dumps({'step': step, 'time_s': step*DT, 'qpos_before': before_q, 'qvel_before': before_v,
                    'qpos_after': data.qpos, 'qvel_after': data.qvel, 'contacts': rows, 'post_state_geometry': post}, default=v2.serial, allow_nan=False)+'\n'); log.flush()
                head_flags.append(accumulate(summary, rows, post, step))
                endpoint = classify_step(rows, post)
                if endpoint:
                    summary['status'] = endpoint; summary['floor_endpoint'] = {'step': step, 'pre_time_s': step*DT, 'post_time_s': float(data.time), 'post': post}
                    break
            else:
                require(summary['head_contact_steps'] > 0, 'No head contact in full horizon')
                summary['horizon_completed'] = True; summary['status'] = CASE_COMPLETE
        require(time.monotonic() <= deadline, 'Approved physics wall limit exceeded')
    except Exception as exc:
        summary['status'] = ERROR; summary['error'] = str(exc)
        (destination/'error.txt').write_text(traceback.format_exc())
    finally:
        if model is not None and signature is not None:
            summary['model_unchanged'] = floor.model_signature(model) == signature == floor.model_signature(original)
            if not summary['model_unchanged']: summary.update(status=ERROR, error='Material model changed')
        summary['elapsed_simulation_s'] = summary['steps']*DT
        summary['completed_step_count_exact'] = summary['steps'] == summary['attempted_steps']
        summary['wall_seconds'] = time.monotonic()-started
        summary['head_contact_intervals'] = contact_intervals(np.asarray(head_flags, dtype=bool))
        summary['no_head_contact_intervals'] = contact_intervals(~np.asarray(head_flags, dtype=bool))
        summary['classified_pre_step_samples'] = len(head_flags)
        for interval in summary['head_contact_intervals']:
            summary['critical_frame_steps'].extend([interval['first_pre_step'], interval['end_step_exclusive']])
        for key in ('max_head_slip_step', 'max_head_penetration_step'):
            if key in summary: summary['critical_frame_steps'].append(summary[key])
        summary['power_and_cone_metrics_are_observations_not_pass_thresholds'] = True
        np.savez_compressed(destination/'states.npz', qpos=np.asarray(poses), qvel=np.asarray(velocities), time=np.asarray(times), dt=DT)
        save(destination/'summary.json', summary); progress(summary)
    v2.emit(case['name']+' '+summary['status']+' completed='+str(summary['steps']))
    return summary


def run_matrix(model, triangles, cases, out, report, deadline):
    require(tuple(c['name'] for c in cases) == CASE_NAMES, 'Exactly the six approved cases required')
    for case in cases:
        def progress(current):
            completed = sum(c['steps'] for c in report['cases'])+current['steps']
            attempted = sum(c['attempted_steps'] for c in report['cases'])+current['attempted_steps']
            require(0 <= completed <= attempted <= TOTAL_STEPS and current['attempted_steps'] <= MAX_STEPS, 'Approved global step budget exceeded')
            save(out/'progress.json', {'completed_steps': completed, 'attempted_steps': attempted, 'current_case': case['name'], 'current_status': current['status']})
        result = run_case(model, triangles, case, out/'cases'/case['name'], deadline, progress)
        report['cases'].append(result); save(out/'diagnostic-review.json', report)
        if result['status'] == ERROR:
            report['status'] = 'HEAD_FRICTION_ERROR_STOP_REVIEW_REQUIRED'; break
        require(result['status'] in (CASE_COMPLETE, *FLOOR_ENDPOINTS), 'Unknown case status')
    else:
        report['status'] = ('HEAD_FRICTION_EVIDENCE_REVIEW_REQUIRED' if all(c['status'] == CASE_COMPLETE for c in report['cases'])
                            else 'HEAD_FRICTION_FLOOR_ENDPOINTS_REVIEW_REQUIRED')
    report['not_run_cases'] = [n for n in CASE_NAMES if n not in [c['name'] for c in report['cases']]]


def physics_main(out):
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); deadline = started+WALL_SECONDS
    report = {'status': 'STARTED', 'cases': [], 'not_run_cases': list(CASE_NAMES),
        'stage11_status': 'MATERIAL_BLOCKED_REVIEW_REQUIRED', 'stage11_complete': False,
        'new_ppo_updates': 0, 'cloud_hours': 0, 'full_robot_steps': 0, 'prior_canonical_steps': 6211,
        'formal_task_migration': False, 'old_floor_and_sensitivity_failures_retained': True,
        'approval': {'user_message': '批准并继续', 'time': '2026-10-04T01:59:56+08:00'},
        'head_neck_fixed_only_in_material_fixture': True, 'policy_success': None,
        'no_automatic_retry_or_extra_cases': True, 'videos': 'PENDING_SAVED_STATE_RENDER'}
    provenance = {}
    try:
        v2.local_preflight(out); provenance = verify_inputs(floor.load_json(out/'preflight.json'))
        save(out/'input-provenance.json', provenance)
        for name in ('material-fixture.mjb', 'material-geometry.npz'):
            shutil.copyfile(SOURCE/'local-v2'/name, out/name)
        model = mujoco.MjModel.from_binary_path(str(out/'material-fixture.mjb'))
        with np.load(out/'material-geometry.npz', allow_pickle=False) as geometry:
            triangles = geometry['vertices'][geometry['faces']]; center_xy = geometry['center_xy'].copy()
        cases = selected_cases(triangles, center_xy, floor.load_json(SOURCE/'local-v2/cpu/release-center/initial.json'))
        save(out/'selected-cases.json', cases)
        run_matrix(model, triangles, cases, out, report, deadline)
    except Exception as exc:
        report['status'] = 'HEAD_FRICTION_ERROR_STOP_REVIEW_REQUIRED'; report['error'] = str(exc)
        (out/'error.txt').write_text(traceback.format_exc())
    finally:
        report['diagnostic_completed_steps'] = sum(c['steps'] for c in report['cases'])
        report['diagnostic_attempted_steps'] = sum(c['attempted_steps'] for c in report['cases'])
        if (out/'progress.json').exists():
            progress = floor.load_json(out/'progress.json')
            report['diagnostic_completed_steps'] = max(report['diagnostic_completed_steps'], progress['completed_steps'])
            report['diagnostic_attempted_steps'] = max(report['diagnostic_attempted_steps'], progress['attempted_steps'])
        report['canonical_total_completed_steps'] = 6211+report['diagnostic_completed_steps']
        report['physics_wall_seconds'] = time.monotonic()-started
        report['diagnostic_source_sha256'] = digest(Path(__file__))
        for key, directory in [('original_material', SOURCE), ('refinement', REFINEMENT)]:
            if key in provenance:
                report[key+'_unchanged'] = all((directory/n).is_file() and digest(directory/n) == meta['sha256'] for n, meta in provenance[key]['files'].items())
                if not report[key+'_unchanged']: report['status'] = 'SOURCE_CHANGED_BLOCKED'
        report['frozen_sources_unchanged'] = all((ROOT/n).is_file() and digest(ROOT/n) == sha for n, sha in FROZEN.items())
        if not report['frozen_sources_unchanged']: report['status'] = 'SOURCE_CHANGED_BLOCKED'
        report['not_run_cases'] = [n for n in CASE_NAMES if n not in [c['name'] for c in report['cases']]]
        save(out/'diagnostic-review.json', report)
    print('Stage11Diagnostic='+report['status'], flush=True)
    print('Stage11DiagnosticSteps='+str(report['diagnostic_completed_steps']), flush=True)
    return 0 if report['status'] in ('HEAD_FRICTION_EVIDENCE_REVIEW_REQUIRED', 'HEAD_FRICTION_FLOOR_ENDPOINTS_REVIEW_REQUIRED') else 2


def select_videos(cases):
    available = [c for c in cases if c['steps'] > 0]
    selected = []
    slides = [c for c in available if c['name'].startswith('slide-')]
    if slides: selected.append(max(slides, key=lambda c: c['max_head_slip_m_s'])['name'])
    if any(c['name'] == 'normal-spin' for c in available): selected.append('normal-spin')
    blocked = next((c['name'] for c in available if c['status'] != CASE_COMPLETE), None)
    if blocked and blocked not in selected: selected.append(blocked)
    return selected


def render_main(out):
    """Separate CLI path: read saved states only; never call physics_main/step."""
    report = floor.load_json(out/'diagnostic-review.json')
    require(report['diagnostic_source_sha256'] == digest(Path(__file__)), 'Render source differs from executed diagnostic')
    render_path = out/'video-review.json'
    require(not render_path.exists(), 'Video pass already recorded; no automatic rerender')
    result = {'new_integration_steps': 0, 'fixed_material_fixture': True, 'policy_success': None,
              'assistant_visual_review': 'PENDING_RETURN', 'videos': [], 'errors': [],
              'requested_cases': select_videos(report['cases']), 'normal_spin_not_run': 'normal-spin' in report['not_run_cases']}
    try:
        provenance = floor.load_json(out/'input-provenance.json')
        require(digest(out/'material-fixture.mjb') == provenance['original_material']['files']['local-v2/material-fixture.mjb']['sha256'], 'Render MJB changed')
        for name in result['requested_cases']:
            summary = floor.load_json(out/'cases'/name/'summary.json')
            try:
                model = mujoco.MjModel.from_binary_path(str(out/'material-fixture.mjb'))
                video = render_case(model, out/'cases'/name, out/(name+'.mp4'), summary)
                result['videos'].append({**video, 'case_status': summary['status'], 'horizon_completed': summary['horizon_completed'], 'recorded_duration_s': summary['elapsed_simulation_s']})
            except Exception as exc:
                result['errors'].append({'case': name, 'error': str(exc)})
                (out/(name+'-render-error.txt')).write_text(traceback.format_exc())
                break
    except Exception as exc:
        result['errors'].append({'error': str(exc)})
    result['status'] = ('RENDER_BLOCKED' if result['errors'] else
                        'RENDERED_PENDING_VISUAL_REVIEW' if result['videos'] else 'NO_REPLAYABLE_SAVED_CASES')
    save(render_path, result)
    print('Stage11SavedStateVideos='+result['status'], flush=True)
    return 2 if result['errors'] else 0


def render_case(model, directory, target, summary):
    # Frozen renderer mechanics, but explicit endpoint labels and critical states.
    from PIL import Image, ImageDraw, ImageFont
    import subprocess
    require(shutil.which('ffmpeg') is not None, 'ffmpeg missing; no automatic install')
    with np.load(directory/'states.npz', allow_pickle=False) as z: poses, velocities = z['qpos'].copy(), z['qvel'].copy()
    require(len(poses) == summary['steps']+1 and len(poses) > 1 and np.isfinite(poses).all() and np.isfinite(velocities).all(), 'No finite saved trajectory to render')
    rows = {}
    with gzip.open(directory/'contacts.jsonl.gz', 'rt') as log:
        for line in log:
            row = json.loads(line); rows.setdefault(row['step'], []).append(row)
    samples = set(range(0, len(poses)-1, 20)) | {len(poses)-2, len(poses)-1}
    samples.update(i for i in summary['critical_frame_steps'] if 0 <= i < len(poses))
    frame_steps = sorted(samples)
    width, height = 960, 720
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, width); model.vis.global_.offheight = max(model.vis.global_.offheight, height)
    renderer = mujoco.Renderer(model, height=height, width=width)
    data = mujoco.MjData(model); camera = mujoco.MjvCamera(); camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = poses[0, :3]-[0, 0, .045]; camera.distance = .42; camera.azimuth = 130; camera.elevation = -22
    process = subprocess.Popen(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '25', '-i', '-', '-an', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)], stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    font = ImageFont.load_default(size=18)
    try:
        for step in frame_steps:
            data.qpos[:] = poses[step]; data.qvel[:] = velocities[step]; data.time = step*DT
            mujoco.mj_forward(model, data); renderer.update_scene(data, camera=camera)
            picture = Image.fromarray(renderer.render()); draw = ImageDraw.Draw(picture)
            final = step == len(poses)-1
            draw.rectangle((0, 0, width, 54), fill='black')
            draw.text((7, 3), f'FIXED MATERIAL | {directory.name} | t={step*DT:.3f}s | '+('FINAL POST STATE' if final else 'PRE STEP'), font=font, fill='white')
            draw.text((7, 28), summary['status']+' | recorded '+f'{summary["elapsed_simulation_s"]:.3f}s / 3.000s', font=font, fill='white')
            current = [] if final else rows.get(step, [])
            if current:
                text = ' | '.join(f"{r['other']}: pen={max(0.,-r['dist'])*1000:.3f}mm slip={r['slip_speed_m_s']:.4f}m/s n="+','.join(f'{x:+.3f}' for x in r['normal_toward_ball']) for r in current)
                draw.rectangle((0, height-29, width, height), fill='black'); draw.text((7, height-25), text, font=font, fill='white')
            process.stdin.write(np.asarray(picture).tobytes())
        process.stdin.close(); error = process.stderr.read().decode(errors='replace'); code = process.wait(timeout=30)
        require(code == 0, 'Video encoding failed: '+error)
    finally:
        renderer.close()
        if process.poll() is None: process.kill(); process.wait()
    require(subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(target), '-f', 'null', '-'], capture_output=True).returncode == 0, 'Video decode failed')
    return {'file': target.name, 'sha256': digest(target), 'frames': len(frame_steps), 'fps': 25,
            'frame_steps': frame_steps, 'frame_times_s': [s*DT for s in frame_steps],
            'sampling': 'Saved-state montage with critical and terminal frames; displayed timestamps are authoritative',
            'no_new_simulation_steps': True, 'visual_review': 'PENDING'}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--approved-six-case-friction', action='store_true')
    mode.add_argument('--render-saved', action='store_true')
    args = parser.parse_args()
    return render_main(args.output.resolve()) if args.render_saved else physics_main(args.output.resolve())


if __name__ == '__main__': raise SystemExit(main())
