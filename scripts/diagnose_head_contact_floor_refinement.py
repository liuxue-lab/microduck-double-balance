"""Approved C3/C4 step refinement of the isolated floor impact; zero PPO.

Fresh copies only, at most 1200 steps/300 seconds. Reuse archived A0 and C2;
do not replay them, adopt formal physics, or automatically continue beyond C4.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import importlib.util
import json
from pathlib import Path
import shutil
import time
import traceback
import zipfile
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
_loader=importlib.util.spec_from_file_location('stage11_frozen_stiffness', ROOT/'scripts/diagnose_head_contact_floor_stiffness.py')
stiff=importlib.util.module_from_spec(_loader); _loader.loader.exec_module(stiff)
floor, v2 = stiff.floor, stiff.v2
require, save, digest = stiff.require, stiff.save, stiff.digest
overlap, post_state_check = stiff.overlap, stiff.post_state_check
SOURCE, PRIOR = stiff.SOURCE, stiff.PRIOR
STIFFNESS=Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage11/floor-stiffness/20261003T170744Z')
STIFFNESS_ZIP_SHA256='0ba55b83e4523ba636f7e8f4a948f1694ed4b9f3de13890c317da6e9bf3976cf'
STIFFNESS_COMMIT='bf2a9e8eabd2a31513b8e55504f6e7a84a77d276'
STIFFNESS_SCRIPT_SHA256='e2f52491480582fbdd754fc18a5e5a5b86e7a59b01afa97d6e34327c8163b54a'
CASES=(
    {'name':'C3','dt':.000125,'solref0':.001,'max_steps':400},
    {'name':'C4','dt':.0000625,'solref0':.001,'max_steps':800},
)
DURATION, PENETRATION_MAX, WALL_SECONDS, TOTAL_STEPS_MAX=.05,.001,300,1200
PASS_CASE='REFINEMENT_CASE_GATES_PASS_REVIEW_REQUIRED'
PASS_MATRIX='STEP_REFINEMENT_CANDIDATE_REVIEW_REQUIRED'
BUDGETS={
    'abs_difference_peak_sampled_overlap_m':.0001,
    'max_common_time_position_difference_m':.0002,
    'max_common_time_linear_velocity_difference_m_s':.05,
    'abs_difference_peak_upward_velocity_m_s':.02,
}
STIFFNESS_INPUTS=('run-summary.json','local-versions.json','diagnostic/preflight.json',
    'diagnostic/diagnostic-review.json','diagnostic/input-provenance.json',
    'diagnostic/cases/C2/summary.json','diagnostic/cases/C2/initialization.json',
    'diagnostic/cases/C2/states.npz','diagnostic/cases/C2/contacts.jsonl.gz')


def verify_stiffness(source, material_provenance, five_provenance, current_preflight):
    source=Path(source); archive=source.with_suffix('.zip')
    require(digest(ROOT/'scripts/diagnose_head_contact_floor_stiffness.py')==STIFFNESS_SCRIPT_SHA256,
            'Frozen C1/C2 source changed')
    require(archive.is_file() and digest(archive)==STIFFNESS_ZIP_SHA256, 'Pinned C1/C2 ZIP missing or changed')
    hashes={}
    with zipfile.ZipFile(archive) as z:
        require(len(z.namelist())==len(set(z.namelist())), 'Duplicate archive paths')
        inventory=json.loads(z.read('evidence-inventory.json'))
        require((source/'evidence-inventory.json').read_bytes()==z.read('evidence-inventory.json'), 'C1/C2 inventory changed')
        for name in STIFFNESS_INPUTS:
            path=source/name
            require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(source.resolve()), 'Missing/unsafe C1/C2 input: '+name)
            expected=floor.hashlib.sha256(z.read(name)).hexdigest()
            require(digest(path)==expected==inventory[name]['sha256'], 'C1/C2 input SHA mismatch: '+name)
            hashes[name]={'sha256':expected,'bytes':path.stat().st_size}
    prior_inputs=floor.load_json(source/'diagnostic/input-provenance.json')
    for key,actual in [('original_material',material_provenance),('archived_five_case',five_provenance)]:
        require(prior_inputs[key]['zip_sha256']==actual['zip_sha256'] and prior_inputs[key]['files']==actual['files'], 'C2 provenance differs: '+key)
    before=floor.load_json(source/'diagnostic/preflight.json')
    for key in ('head','versions','python'):
        require(current_preflight[key]==before[key], 'Runtime differs from C2: '+key)
    require(floor.load_json(source/'local-versions.json')==current_preflight['versions'], 'C2 runtime version records differ')
    run=floor.load_json(source/'run-summary.json')
    require(run['head']==run['head_after']==v2.BASELINE and run['physics_integration_steps']==run['physics_attempted_steps']==300
            and run['canonical_total_known_completed_steps']==5011 and run['execution_invariants_unchanged'], 'Invalid C1/C2 run evidence')
    report=floor.load_json(source/'diagnostic/diagnostic-review.json')
    require(report['status']==stiff.PASS_MATRIX and report['diagnostic_source_sha256']==STIFFNESS_SCRIPT_SHA256
            and report['diagnostic_completed_steps']==report['diagnostic_attempted_steps']==300
            and report['original_model_unchanged'] and report['source_files_unchanged'] and report['prior_files_unchanged'], 'Invalid C1/C2 physical evidence')
    require([c['name'] for c in report['cases']]==['C1','C2'] and all(c['horizon_completed'] and c['status']==stiff.PASS_CASE for c in report['cases']), 'C1/C2 did not complete')
    require(floor.load_json(source/'diagnostic/cases/C2/summary.json')==report['cases'][1], 'C2 summaries differ')
    return {'artifact_commit':STIFFNESS_COMMIT,'zip_sha256':STIFFNESS_ZIP_SHA256,'files':hashes,
            'C1_C2_reused_without_replay':True,'runtime_and_inputs_match_C2':True}


def load_series(path, dt):
    with np.load(path,allow_pickle=False) as z:
        t,q,v=z['time'].copy(),z['qpos'].copy(),z['qvel'].copy()
        require(float(z['dt'])==dt,'Saved timestep mismatch')
    require(len(t)>0 and q.shape==(len(t),7) and v.shape==(len(t),6), 'Saved trajectory shape mismatch')
    require(np.isfinite(t).all() and np.isfinite(q).all() and np.isfinite(v).all(), 'Nonfinite saved trajectory')
    require(np.max(np.abs(t-(floor.START_TIME+dt*np.arange(len(t)))))<1e-11, 'Saved trajectory clock mismatch')
    return {'time':t,'qpos':q,'qvel':v,'dt':dt}


def copy_archived_C2(source,out,qpos,qvel,warmstart):
    previous=source/'diagnostic/cases/C2'
    initial=floor.load_json(previous/'initialization.json')
    require(initial['time_s']==floor.START_TIME and initial['initial_contacts']==0 and
            np.array_equal(initial['qpos'],qpos) and np.array_equal(initial['qvel'],qvel) and
            np.array_equal(initial['qacc_warmstart'],warmstart),'Archived C2 initialization differs from A0/source')
    series=load_series(previous/'states.npz',.00025)
    require(len(series['time'])==201 and np.array_equal(series['qpos'][0],qpos) and np.array_equal(series['qvel'][0],qvel),'Archived C2 initial state or length differs')
    dest=out/'archived-C2';dest.mkdir(parents=True,exist_ok=False)
    for name in ('states.npz','summary.json','initialization.json'):
        shutil.copyfile(previous/name,dest/name)
    save(dest/'source.json',{'artifact_commit':STIFFNESS_COMMIT,'copied_without_integration':True,
         'files':{n:digest(dest/n) for n in ('states.npz','summary.json','initialization.json')}})


def aligned_metrics(a,b,times=None):
    grid=a['time'] if times is None else np.asarray(times)
    ia=np.rint((grid-floor.START_TIME)/a['dt']).astype(int)
    ib=np.rint((grid-floor.START_TIME)/b['dt']).astype(int)
    mask=(ia>=0)&(ib>=0)&(ia<len(a['time']))&(ib<len(b['time']))
    grid,ia,ib=grid[mask],ia[mask],ib[mask]
    require(len(grid)>0,'No common saved times')
    require(np.max(abs(a['time'][ia]-grid))<1e-11 and np.max(abs(b['time'][ib]-grid))<1e-11,'Common comparison clocks differ')
    pd=np.linalg.norm(a['qpos'][ia,:3]-b['qpos'][ib,:3],axis=1)
    vd=np.linalg.norm(a['qvel'][ia,:3]-b['qvel'][ib,:3],axis=1)
    return {'common_saved_samples':len(grid),'last_common_time_s':float(grid[-1]),
            'max_common_time_position_difference_m':float(pd.max()),
            'max_common_time_linear_velocity_difference_m_s':float(vd.max()),
            'max_position_difference_time_s':float(grid[pd.argmax()]),
            'max_velocity_difference_time_s':float(grid[vd.argmax()])}


def sensitivity_gates(metrics):
    require(set(metrics)==set(BUDGETS),'Four sensitivity metrics required')
    require(all(np.isfinite(v) and v>=0 for v in metrics.values()),'Invalid sensitivity metric')
    checks={name:{'value':metrics[name],'limit':limit,'pass':bool(metrics[name]<=limit)} for name,limit in BUDGETS.items()}
    return {'all_four_pass':all(c['pass'] for c in checks.values()),'checks':checks,
            'interpretation':'Approved event-level diagnostic error budgets; not a convergence theorem or formal task acceptance'}


def compare_saved(out,cases):
    c2=load_series(out/'archived-C2/states.npz',.00025)
    c3=load_series(out/'cases/C3/states.npz',CASES[0]['dt'])
    c4=load_series(out/'cases/C4/states.npz',CASES[1]['dt'])
    require(all(np.array_equal(c2[k][0],x[k][0]) for x in (c3,c4) for k in ('qpos','qvel')),'Comparison initial states differ')
    pair23=aligned_metrics(c2,c3);pair34=aligned_metrics(c3,c4)
    metrics={k:pair34[k] for k in BUDGETS if k.startswith('max_common')}
    maxima=[max(c['max_pre_step_penetration_m'],c['max_post_step_overlap_m']) for c in cases]
    metrics['abs_difference_peak_sampled_overlap_m']=abs(maxima[0]-maxima[1])
    metrics['abs_difference_peak_upward_velocity_m_s']=abs(cases[0]['max_upward_velocity_m_s']-cases[1]['max_upward_velocity_m_s'])
    end=min(c2['time'][-1],c3['time'][-1],c4['time'][-1])
    grid=c2['time'][c2['time']<=end+1e-11]
    full=all(c['horizon_completed'] and c['status']==PASS_CASE for c in cases)
    result={'C2_C3':pair23,'C3_C4':pair34,'on_C2_grid':{
        'C2_C3':aligned_metrics(c2,c3,grid),'C3_C4':aligned_metrics(c3,c4,grid),'C2_C4':aligned_metrics(c2,c4,grid)},
        'sensitivity':sensitivity_gates(metrics),'full_physical_horizons_completed':full,
        'new_integration_steps':0,'C2_replayed':False,
        'limits':'Partial comparisons cannot qualify a candidate; equal final impulse does not establish transient convergence.'}
    save(out/'time-aligned-comparison.json',result)
    return result


def matrix_status(cases,comparison):
    physical= len(cases)==2 and all(c['status']==PASS_CASE and c['horizon_completed'] for c in cases)
    if not physical:return 'STEP_REFINEMENT_PENETRATION_BLOCKED_REVIEW_REQUIRED'
    return PASS_MATRIX if comparison['sensitivity']['all_four_pass'] else 'STEP_REFINEMENT_SENSITIVITY_BLOCKED_REVIEW_REQUIRED'


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
    report['comparison']=compare_saved(out,report['cases'])
    report['status']=matrix_status(report['cases'],report['comparison'])



def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--approved-two-case-refinement',action='store_true',required=True)
    args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+WALL_SECONDS
    report={'status':'STARTED','cases':[],'stage11_status':'MATERIAL_BLOCKED_REVIEW_REQUIRED',
            'stage11_complete':False,'new_ppo_updates':0,'cloud_hours':0,'full_robot_steps':0,
            'formal_task_migration':False,'prior_canonical_steps':5011,'replayed_A0_C1_C2_steps':0,
            'approval':{'user_message':'批准','time':'2026-10-04T01:23:44+08:00'},
            'sampling_phase':'Original pre-integration contacts plus all saved post-integration geometric overlaps',
            'approved_sensitivity_budgets':BUDGETS,'remaining_full_matrix_automatically_run':False}
    provenance={}
    try:
        v2.local_preflight(out);current=floor.load_json(out/'preflight.json')
        provenance['original_material']=floor.verify_source(SOURCE)
        provenance['archived_five_case']=stiff.verify_prior(PRIOR,provenance['original_material'],current)
        provenance['archived_C1_C2']=verify_stiffness(STIFFNESS,provenance['original_material'],provenance['archived_five_case'],current)
        save(out/'input-provenance.json',provenance)
        model=mujoco.MjModel.from_binary_path(str(SOURCE/'local-v2/material-fixture.mjb'))
        require(model.opt.timestep==floor.SOURCE_DT and model.opt.disableflags==0 and np.array_equal(model.opt.gravity,[0,0,-9.81]),'Original model options changed')
        qpos,qvel,expected=floor.read_source_states(SOURCE,model,out)
        warmstart=stiff.verify_restart(PRIOR,qpos,qvel,expected)
        copy_archived_C2(STIFFNESS,out,qpos,qvel,warmstart)
        run_matrix(model,qpos,qvel,warmstart,out,report,deadline)
    except Exception as exc:
        report['status']='STEP_REFINEMENT_ERROR_STOP_REVIEW_REQUIRED';report['error']=str(exc)
        (out/'error.txt').write_text(traceback.format_exc())
    finally:
        report['diagnostic_completed_steps']=sum(c['steps'] for c in report['cases'])
        report['diagnostic_attempted_steps']=sum(c['attempted_steps'] for c in report['cases'])
        report['canonical_total_completed_steps']=5011+report['diagnostic_completed_steps']
        report['wall_seconds']=time.monotonic()-started
        for key,directory in [('original_material',SOURCE),('archived_five_case',PRIOR),('archived_C1_C2',STIFFNESS)]:
            if key in provenance:
                report[key+'_unchanged']=all((directory/n).is_file() and digest(directory/n)==m['sha256'] for n,m in provenance[key]['files'].items())
                if not report[key+'_unchanged']:report['status']='SOURCE_CHANGED_BLOCKED'
        report['diagnostic_source_sha256']=digest(Path(__file__))
        save(out/'diagnostic-review.json',report)
    print('Stage11Diagnostic='+report['status'],flush=True)
    print('Stage11DiagnosticSteps='+str(report['diagnostic_completed_steps']),flush=True)
    return 0 if report['status'] in (PASS_MATRIX,'STEP_REFINEMENT_PENETRATION_BLOCKED_REVIEW_REQUIRED','STEP_REFINEMENT_SENSITIVITY_BLOCKED_REVIEW_REQUIRED') else 2


if __name__=='__main__':raise SystemExit(main())
