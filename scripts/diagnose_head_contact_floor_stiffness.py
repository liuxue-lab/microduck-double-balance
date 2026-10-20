"""Explicitly approved C1/C2 CPU contact-stiffness diagnostic; zero PPO.

Only fresh material-model copies are stepped, never the formal task. Reuse the
archived A0 check without replaying it. A short candidate is not Stage 11 success.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import time
import traceback
import zipfile

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_loader = importlib.util.spec_from_file_location("stage11_frozen_floor", ROOT / "scripts/diagnose_head_contact_floor_impact.py")
floor = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(floor)
v2 = floor.v2
require, save, digest = floor.require, floor.save, floor.digest

SOURCE = floor.SOURCE
PRIOR = Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage11/floor-impact/20261003T141958Z')
PRIOR_ZIP_SHA256 = 'dd42f05c240f45127731c87d9ad96079ced3c204588b505d762be804467f4f52'
PRIOR_COMMIT = 'fe2db584e9b8fc391f9014e0e3401b4ceef9f5e5'
FLOOR_SCRIPT_SHA256 = '130dee8548e9a1127380162254d6a2d70bf710e30622927bafc5bdefe7d620a7'
FACTORY_SHA256 = '8a2446aa27d8e420e6e26e5e9985ab24cb774036936f393de680857edfd0af64'
CASES = (
    {'name': 'C1', 'dt': .0005, 'solref0': .001, 'max_steps': 100},
    {'name': 'C2', 'dt': .00025, 'solref0': .001, 'max_steps': 200},
)
DURATION, PENETRATION_MAX, WALL_SECONDS, TOTAL_STEPS_MAX = .05, .001, 300, 300
PRIOR_INPUTS = ('run-summary.json', 'local-versions.json', 'diagnostic/diagnostic-review.json',
                'diagnostic/input-provenance.json', 'diagnostic/preflight.json',
                'diagnostic/cases/A0/initialization.json', 'diagnostic/cases/A0/states.npz')
PASS_CASE = 'SHORT_IMPACT_CASE_GATES_PASS_REVIEW_REQUIRED'
PASS_MATRIX = 'SHORT_IMPACT_CANDIDATE_REVIEW_REQUIRED'


def verify_prior(prior, source_provenance, current_preflight):
    """Trust the pinned ZIP, then verify local inputs and the already tested A0."""
    prior = Path(prior)
    require(digest(ROOT / 'scripts/diagnose_head_contact_floor_impact.py') == FLOOR_SCRIPT_SHA256,
            'Frozen five-case diagnostic changed')
    require(digest(ROOT / 'src/mjlab_microduck/robot/head_contact.py') == FACTORY_SHA256,
            'Physical factory changed')
    archive = prior.with_suffix('.zip')
    require(archive.is_file() and digest(archive) == PRIOR_ZIP_SHA256,
            'Pinned five-case ZIP missing or changed; do not rerun old diagnostics')
    hashes = {}
    with zipfile.ZipFile(archive) as z:
        require(len(z.namelist()) == len(set(z.namelist())), 'Duplicate paths in five-case ZIP')
        inventory = json.loads(z.read('evidence-inventory.json'))
        require((prior / 'evidence-inventory.json').read_bytes() == z.read('evidence-inventory.json'),
                'Five-case inventory differs from pinned archive')
        for name in PRIOR_INPUTS:
            p = prior / name
            require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(prior.resolve()),
                    'Missing or unsafe five-case input: ' + name)
            expected = floor.hashlib.sha256(z.read(name)).hexdigest()
            require(digest(p) == expected == inventory[name]['sha256'], 'Five-case input changed: ' + name)
            hashes[name] = {'sha256': expected, 'bytes': p.stat().st_size}
    previous_source = floor.load_json(prior / 'diagnostic/input-provenance.json')
    require(previous_source['zip_sha256'] == source_provenance['zip_sha256'] and
            previous_source['files'] == source_provenance['files'], 'A0 and candidate source inputs differ')
    old_preflight = floor.load_json(prior / 'diagnostic/preflight.json')
    for key in ('head', 'versions', 'python'):
        require(current_preflight[key] == old_preflight[key], 'Runtime/baseline differs from archived A0: ' + key)
    require(floor.load_json(prior / 'local-versions.json') == current_preflight['versions'], 'Version records differ')
    run = floor.load_json(prior / 'run-summary.json')
    require(run['head'] == run['head_after'] == v2.BASELINE and run['physics_integration_steps'] == 14
            and run['stage11_status'] == 'MATERIAL_BLOCKED_REVIEW_REQUIRED', 'Unexpected five-case run state')
    report = floor.load_json(prior / 'diagnostic/diagnostic-review.json')
    require(report['diagnostic_source_sha256'] == FLOOR_SCRIPT_SHA256 and
            report['diagnostic_completed_steps'] == report['diagnostic_attempted_steps'] == 14 and
            report['original_model_unchanged'] and report['source_files_unchanged'], 'Invalid archived diagnostic invariants')
    require([c['name'] for c in report['cases']] == ['A0', 'A1', 'A2', 'B1', 'B2'] and
            all(c['status'] == 'PENETRATION_BLOCKED' for c in report['cases']), 'Unexpected prior case outcomes')
    match = report['baseline_reproduction']
    require(match['pass'] and match['time_error_s'] == match['distance_error_m'] == 0 and
            np.array_equal(match['force_component_absolute_error'], np.zeros(6)), 'Archived A0 is not the reviewed exact match')
    return {'artifact_commit': PRIOR_COMMIT, 'zip_sha256': PRIOR_ZIP_SHA256,
            'files': hashes, 'archived_A0_reused_without_integration': True,
            'baseline_reproduction': match, 'runtime_and_original_inputs_match_A0': True}


def verify_restart(prior, qpos, qvel, expected_contact):
    init = floor.load_json(prior / 'diagnostic/cases/A0/initialization.json')
    require(init['time_s'] == floor.START_TIME and init['initial_contacts'] == 0 and
            np.array_equal(qpos, init['qpos']) and np.array_equal(qvel, init['qvel']),
            'Restart differs from archived A0 initialization')
    with np.load(prior / 'diagnostic/cases/A0/states.npz', allow_pickle=False) as states:
        require(np.array_equal(qpos, states['qpos'][0]) and np.array_equal(qvel, states['qvel'][0]),
                'Restart differs from saved A0 state')
    a0 = floor.load_json(prior / 'diagnostic/diagnostic-review.json')['cases'][0]
    require(floor.baseline_match(a0['first_floor_contact'], expected_contact)['pass'], 'Archived A0/source contact mismatch')
    return np.asarray(init['qacc_warmstart'])


def configured_copy(original, case):
    require(case in CASES, 'Case is outside the approved two-case matrix')
    before = floor.model_signature(original)
    model = copy.deepcopy(original)
    model.opt.timestep = case['dt']
    sphere = model.geom('sphere').id
    model.geom_solref[sphere, 0] = case['solref0']
    after = floor.model_signature(model)
    expected_options = copy.deepcopy(before['options']); expected_options['timestep'] = case['dt']
    expected_solref = original.geom_solref.copy(); expected_solref[sphere, 0] = case['solref0']
    require(before['dimensions'] == after['dimensions'] and after['options'] == expected_options,
            'Undeclared model dimensions/options change')
    require(np.array_equal(model.geom_solref, expected_solref), 'Undeclared solref change')
    require({k: v for k, v in after['arrays'].items() if k != 'geom_solref'} ==
            {k: v for k, v in before['arrays'].items() if k != 'geom_solref'},
            'Geometry/mass/inertia/friction/masks changed')
    require(floor.model_signature(original) == before, 'Original in-memory model changed')
    require(model.opt.disableflags == 0 and max(case['solref0'], 2 * case['dt']) == .001,
            'Expected refsafe enabled and effective time constant 1 ms')
    ground = model.geom('floor').id
    require(model.geom_bodyid[ground] == 0 and model.geom_type[ground] == mujoco.mjtGeom.mjGEOM_PLANE
            and np.array_equal(model.geom_quat[ground], [1, 0, 0, 0]) and
            np.array_equal(model.geom_pos[sphere], [0, 0, 0]) and
            model.geom_type[sphere] == mujoco.mjtGeom.mjGEOM_SPHERE,
            'Post-step overlap formula requires the archived horizontal floor and centered sphere')
    return model, {'timestep': [original.opt.timestep, model.opt.timestep],
                   'sphere_solref0': [float(original.geom_solref[sphere, 0]), case['solref0']],
                   'all_other_model_arrays_and_options_unchanged': True}


def overlap(model, qpos):
    value = float(model.geom_pos[model.geom('floor').id, 2] +
                  model.geom_size[model.geom('sphere').id, 0] - qpos[2])
    require(np.isfinite(value), 'Nonfinite geometric overlap')
    return max(0., value)


def post_state_check(model, probe, data):
    """Nonintegrating collision check on separate data; preserve main warm-start."""
    probe.qpos[:] = data.qpos; probe.qvel[:] = data.qvel; probe.time = data.time
    probe.qacc_warmstart[:] = 0
    mujoco.mj_forward(model, probe)
    wanted = {model.geom('floor').id, model.geom('sphere').id}
    contacts = []
    for contact in probe.contact:
        require(set(map(int, contact.geom)) == wanted, 'Unexpected contact in post-integration state')
        require(np.isfinite(np.r_[contact.dist, contact.pos, contact.frame]).all(), 'Nonfinite post-state contact')
        contacts.append({'geom': contact.geom.copy(), 'dist': float(contact.dist),
                         'pos': contact.pos.copy(), 'frame': contact.frame.copy()})
    return {'phase': 'post_integration_geometry_only_no_force_claim', 'time_s': float(data.time),
            'floor_overlap_m': overlap(model, data.qpos), 'contacts': contacts}


def run_case(original, qpos, qvel, expected_warmstart, case, destination, deadline, progress):
    destination.mkdir(parents=True, exist_ok=False)
    summary = {'name': case['name'], 'case': case, 'status': 'RUNNING', 'steps': 0, 'attempted_steps': 0,
               'horizon_completed': False, 'max_pre_step_penetration_m': 0., 'max_post_step_overlap_m': 0.,
               'normal_impulse_Ns': 0., 'max_normal_force_N': 0., 'first_floor_contact': None,
               'new_ppo_updates': 0, 'full_robot_steps': 0}
    poses, velocities, times = [], [], []
    try:
        model, changes = configured_copy(original, case)
        data = floor.initialize(model, qpos, qvel); probe = mujoco.MjData(model)
        require(np.array_equal(data.qacc_warmstart, expected_warmstart), 'Reconstructed warm-start differs from A0')
        require(overlap(model, data.qpos) == 0., 'Restart intersects floor')
        summary['model_changes'] = changes; summary['effective_normal_timeconst_s'] = .001
        save(destination / 'initialization.json', {'qpos': data.qpos, 'qvel': data.qvel, 'time_s': data.time,
             'qacc_warmstart': data.qacc_warmstart, 'initial_contacts': data.ncon,
             'method': 'Same saved qpos/qvel and free-flight derived/warm-start initialization as archived A0',
             'complete_historical_state_restore': False})
        poses.append(data.qpos.copy()); velocities.append(data.qvel.copy()); times.append(float(data.time))
        initial_warnings = np.asarray(data.warning.number).copy()
        with gzip.open(destination / 'contacts.jsonl.gz', 'wt') as log:
            for step in range(case['max_steps']):
                require(time.monotonic() < deadline, 'Approved five-minute wall limit reached')
                before_qpos, before_qvel = data.qpos.copy(), data.qvel.copy()
                before_time = floor.START_TIME + step * case['dt']
                require(abs(data.time - before_time) < 1e-11, 'Pre-step clock drift')
                summary['attempted_steps'] += 1; progress(summary)
                mujoco.mj_step(model, data)
                summary['steps'] += 1
                poses.append(data.qpos.copy()); velocities.append(data.qvel.copy()); times.append(float(data.time))
                require(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(), 'Nonfinite state')
                require(abs(data.time - (before_time + case['dt'])) < 1e-11, 'Post-step clock drift or automatic reset')
                require(np.array_equal(data.warning.number, initial_warnings), 'New MuJoCo runtime warning')
                records = v2.cpu_records(model, data)
                rows = v2.contact_rows(model, before_qpos, before_qvel, records, step, 'cpu')
                for row in rows: row['time_s'] = before_time
                require(len(rows) == len(records), 'Unexpected non-sphere contact')
                pre_blocked = floor.contact_failure(rows, case['solref0'])
                post = post_state_check(model, probe, data)
                log.write(json.dumps({'step': step, 'time_s': before_time, 'qpos_before': before_qpos,
                    'qvel_before': before_qvel, 'qpos_after': data.qpos, 'qvel_after': data.qvel,
                    'contacts': rows, 'raw_contacts': records, 'post_state_geometry': post},
                    default=v2.serial, allow_nan=False) + '\n'); log.flush()
                summary['max_post_step_overlap_m'] = max(summary['max_post_step_overlap_m'], post['floor_overlap_m'])
                for row in rows:
                    summary['max_pre_step_penetration_m'] = max(summary['max_pre_step_penetration_m'], -float(row['dist']))
                    summary['normal_impulse_Ns'] += float(row['normal_force_N']) * case['dt']
                    summary['max_normal_force_N'] = max(summary['max_normal_force_N'], float(row['normal_force_N']))
                    if summary['first_floor_contact'] is None: summary['first_floor_contact'] = row
                summary['last_post_step_overlap_m'] = post['floor_overlap_m']
                progress(summary)
                if pre_blocked or post['floor_overlap_m'] > PENETRATION_MAX:
                    summary['status'] = 'PENETRATION_BLOCKED'
                    summary['stop_gate'] = 'pre_integration_contact' if pre_blocked else 'post_integration_geometry'
                    break
            else:
                require(summary['first_floor_contact'] is not None, 'No floor impact within approved horizon')
                summary['horizon_completed'] = True; summary['status'] = PASS_CASE
        require(time.monotonic() <= deadline, 'Approved wall limit exceeded')
    except Exception as exc:
        summary['status'] = 'ERROR_STOP_MATRIX'; summary['error'] = str(exc)
        (destination / 'error.txt').write_text(traceback.format_exc())
    finally:
        summary['completed_step_count_exact'] = summary['steps'] == summary['attempted_steps']
        summary['elapsed_simulation_s'] = summary['steps'] * case['dt']
        if velocities and np.isfinite(velocities).all():
            summary['final_linear_velocity_m_s'] = velocities[-1][:3]
            summary['final_angular_velocity_joint_coordinates_rad_s'] = velocities[-1][3:]
            summary['max_upward_velocity_m_s'] = max(0., max(float(v[2]) for v in velocities))
        np.savez_compressed(destination / 'states.npz', qpos=np.asarray(poses), qvel=np.asarray(velocities),
                            time=np.asarray(times), dt=case['dt'])
        save(destination / 'summary.json', summary); progress(summary)
    return summary


def compare_saved(out):
    with np.load(out / 'cases/C1/states.npz', allow_pickle=False) as a, np.load(out / 'cases/C2/states.npz', allow_pickle=False) as b:
        ta, tb = a['time'], b['time']
        require(len(ta) and len(tb), 'Missing saved comparison states')
        indices = np.rint((ta - floor.START_TIME) / CASES[1]['dt']).astype(int)
        mask = (indices >= 0) & (indices < len(tb))
        ia, ib = np.flatnonzero(mask), indices[mask]
        require(np.all(np.abs(ta[ia] - tb[ib]) < 1e-11), 'Common comparison times differ')
        result = {'common_saved_samples': len(ia), 'last_common_time_s': float(ta[ia[-1]]),
                  'max_position_difference_m': float(np.linalg.norm(a['qpos'][ia, :3] - b['qpos'][ib, :3], axis=1).max()),
                  'max_linear_velocity_difference_m_s': float(np.linalg.norm(a['qvel'][ia, :3] - b['qvel'][ib, :3], axis=1).max()),
                  'interpretation': 'Common saved-time sensitivity only; two timesteps do not prove convergence or material calibration',
                  'new_integration_steps': 0}
    save(out / 'time-aligned-comparison.json', result)
    return result


def run_matrix(original, qpos, qvel, warmstart, out, report, deadline):
    before = floor.model_signature(original)
    for case in CASES:
        prior_steps = sum(c['steps'] for c in report['cases'])
        prior_attempts = sum(c['attempted_steps'] for c in report['cases'])
        def progress(current):
            require(prior_attempts + current['attempted_steps'] <= TOTAL_STEPS_MAX, 'Approved global step budget exceeded')
            save(out / 'progress.json', {'completed_cases': report['cases'], 'active_case': current,
                 'completed_steps': prior_steps + current['steps'], 'attempted_steps': prior_attempts + current['attempted_steps'],
                 'new_ppo_updates': 0})
        result = run_case(original, qpos, qvel, warmstart, case, out / 'cases' / case['name'], deadline, progress)
        report['cases'].append(result); save(out / 'diagnostic-review.json', report)
        require(result['status'] in ('PENETRATION_BLOCKED', PASS_CASE), result.get('error', 'Unknown case outcome'))
        require(floor.model_signature(original) == before, 'Original model changed between cases')
    report['original_model_unchanged'] = True
    report['status'] = PASS_MATRIX if all(c['status'] == PASS_CASE and c['horizon_completed'] for c in report['cases']) else 'STIFFNESS_DIAGNOSTIC_PENETRATION_BLOCKED_REVIEW_REQUIRED'
    report['time_aligned_comparison'] = compare_saved(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--approved-two-case-stiffness-diagnostic', action='store_true', required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); deadline = started + WALL_SECONDS
    report = {'status': 'STARTED', 'cases': [], 'stage11_status': 'MATERIAL_BLOCKED_REVIEW_REQUIRED',
              'stage11_complete': False, 'new_ppo_updates': 0, 'cloud_hours': 0, 'full_robot_steps': 0,
              'formal_task_migration': False, 'prior_canonical_steps': 4711, 'new_A0_steps': 0,
              'approval': {'user_message': '是继续', 'time': '2026-10-03T23:16:14+08:00'},
              'sampling_phase': 'Original pre-integration contact gate plus all saved post-integration geometric overlaps',
              'remaining_full_matrix_automatically_run': False}
    source_provenance = prior_provenance = None
    try:
        v2.local_preflight(out)
        source_provenance = floor.verify_source(SOURCE)
        prior_provenance = verify_prior(PRIOR, source_provenance, floor.load_json(out / 'preflight.json'))
        save(out / 'input-provenance.json', {'original_material': source_provenance, 'archived_five_case': prior_provenance})
        model = mujoco.MjModel.from_binary_path(str(SOURCE / 'local-v2/material-fixture.mjb'))
        require(model.opt.timestep == floor.SOURCE_DT and model.opt.disableflags == 0 and
                np.array_equal(model.opt.gravity, [0, 0, -9.81]), 'Original model options changed')
        qpos, qvel, expected = floor.read_source_states(SOURCE, model, out)
        warmstart = verify_restart(PRIOR, qpos, qvel, expected)
        run_matrix(model, qpos, qvel, warmstart, out, report, deadline)
    except Exception as exc:
        report['status'] = 'STIFFNESS_DIAGNOSTIC_ERROR_STOP_REVIEW_REQUIRED'; report['error'] = str(exc)
        (out / 'error.txt').write_text(traceback.format_exc())
    finally:
        report['diagnostic_completed_steps'] = sum(c['steps'] for c in report['cases'])
        report['diagnostic_attempted_steps'] = sum(c['attempted_steps'] for c in report['cases'])
        report['canonical_total_completed_steps'] = 4711 + report['diagnostic_completed_steps']
        report['wall_seconds'] = time.monotonic() - started
        for label, root, provenance in [('source_files_unchanged', SOURCE, source_provenance),
                                         ('prior_files_unchanged', PRIOR, prior_provenance)]:
            if provenance:
                report[label] = all((root / n).is_file() and digest(root / n) == m['sha256'] for n, m in provenance['files'].items())
                if not report[label]: report['status'] = 'SOURCE_CHANGED_BLOCKED'
        report['diagnostic_source_sha256'] = digest(Path(__file__))
        save(out / 'diagnostic-review.json', report)
    print('Stage11Diagnostic=' + report['status'], flush=True)
    print('Stage11DiagnosticSteps=' + str(report['diagnostic_completed_steps']), flush=True)
    return 0 if report['status'] in (PASS_MATRIX, 'STIFFNESS_DIAGNOSTIC_PENETRATION_BLOCKED_REVIEW_REQUIRED') else 2


if __name__ == '__main__': raise SystemExit(main())
