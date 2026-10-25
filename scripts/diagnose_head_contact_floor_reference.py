"""Bounded C5/C6 CPU numerical reference of one saved sphere-floor impact.

No production physics adoption, task migration, PPO, cloud or old-case replay.
Stop the entire matrix on any physical failure. CPU preserves the C3/C4 backend.
"""
from __future__ import annotations
import argparse
import copy
import gzip
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
import mujoco
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
_loader=importlib.util.spec_from_file_location('stage11_frozen_refinement', ROOT/'scripts/diagnose_head_contact_floor_refinement.py')
refine=importlib.util.module_from_spec(_loader); _loader.loader.exec_module(refine)
stiff, floor, v2 = refine.stiff, refine.floor, refine.v2
require, digest = refine.require, refine.digest
overlap, post_state_check = refine.overlap, refine.post_state_check
SOURCE, PRIOR, STIFFNESS = refine.SOURCE, refine.PRIOR, refine.STIFFNESS
REFINEMENT=Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage11/floor-refinement/20261003T173938Z')
REFINEMENT_ZIP_SHA256='5085c2ec989361d0523cbad27cbadb7927cb928f5f39175bb5f10ef8a81c394a'
REFINEMENT_COMMIT='234030e0f701e788f2d649c31b98ba76a5cd8934'
REFINEMENT_SCRIPT_SHA256='7e4005e965e2ff725b29534221689c719eab6299441874b298c28bbb9560abab'
CASES=(
    {'name':'C5','dt':.00003125,'solref0':.001,'max_steps':1600},
    {'name':'C6','dt':.000015625,'solref0':.001,'max_steps':3200},
)
DURATION, PENETRATION_MAX, WALL_SECONDS, TOTAL_STEPS_MAX=.05,.001,300,4800
PASS_CASE='FLOOR_REFERENCE_CASE_GATES_PASS_REVIEW_REQUIRED'
PASS_MATRIX='FLOOR_REFERENCE_FINEST_PAIR_REVIEW_REQUIRED'
BLOCKED_PHYSICAL='FLOOR_REFERENCE_PHYSICAL_BLOCKED_REVIEW_REQUIRED'
BLOCKED_SENSITIVITY='FLOOR_REFERENCE_SENSITIVITY_BLOCKED_REVIEW_REQUIRED'
BUDGETS=copy.deepcopy(refine.BUDGETS)
load_series, sensitivity_gates = refine.load_series, refine.sensitivity_gates
REFINEMENT_INPUTS=('run-summary.json','local-versions.json','diagnostic/preflight.json',
    'diagnostic/diagnostic-review.json','diagnostic/input-provenance.json',
    'diagnostic/time-aligned-comparison.json','diagnostic/cases/C4/summary.json',
    'diagnostic/cases/C4/initialization.json','diagnostic/cases/C4/states.npz',
    'diagnostic/cases/C4/contacts.jsonl.gz')


def save(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj,default=v2.serial,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def local_preflight(out):
    plan=floor.load_json(ROOT/'docs/audits/stage-11-floor-reference-plan.json')
    manifest=floor.load_json(out.parent/'package-manifest.json')
    actual={name:importlib.metadata.version(name) for name in ('mujoco','mujoco-warp','mjlab','warp-lang')}
    expected={'mujoco':'3.10.0','mujoco-warp':'3.8.1','mjlab':'1.3.0','warp-lang':'1.12.0'}
    result={'head':git('rev-parse','HEAD'),'versions':actual,'expected_versions':expected,
        'python':sys.version,'working_tree':git('status','--short','--branch'),
        'tag':git('rev-parse','stage-10-complete^{}'),'tag_type':git('cat-file','-t','refs/tags/stage-10-complete'),
        'branch':git('branch','--show-current'),'tracked_diff':git('diff','--name-status'),
        'staged_diff':git('diff','--cached','--name-status'),'new_ppo_updates':0,'network_used_by_validator':False}
    save(out/'preflight.json',result)
    require(result['head']==v2.BASELINE==manifest['baseline'],'HEAD differs from reviewed baseline')
    require(result['branch']=='double-balance' and not result['staged_diff'],'Branch/index changed')
    require(result['tag']=='1a11f30cc7264d00dec7bce8f11449ded9164169' and result['tag_type']=='tag','Old annotated tag changed')
    require(result['tracked_diff']=='M\tsrc/mjlab_microduck/tasks/mdp.py','Unexpected tracked changes')
    require(actual==expected,'Runtime versions changed')
    flags={k:v for k,v in os.environ.items() if k.startswith('MICRODUCK_')}
    save(out/'environment-overrides.json',flags);require(not flags,'MICRODUCK environment overrides are present')
    for name,sha in plan['frozen_files'].items():
        p=ROOT/name
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(ROOT.resolve()) and digest(p)==sha,'Reviewed frozen source changed: '+name)
    for name,meta in manifest['files'].items():
        p=ROOT/name
        require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(ROOT.resolve()) and digest(p)==meta['sha256'],'Package source changed: '+name)
    save(out/'source-snapshot.json',manifest['files'])
    return result,manifest['files']


def verify_refinement(source, provenance, current):
    source=Path(source);archive=source.with_suffix('.zip')
    require(digest(ROOT/'scripts/diagnose_head_contact_floor_refinement.py')==REFINEMENT_SCRIPT_SHA256,'Frozen C3/C4 source changed')
    require(archive.is_file() and digest(archive)==REFINEMENT_ZIP_SHA256,'Pinned C3/C4 ZIP missing or changed')
    hashes={}
    with zipfile.ZipFile(archive) as z:
        require(len(z.namelist())==len(set(z.namelist())),'Duplicate archive paths')
        inventory=json.loads(z.read('evidence-inventory.json'))
        require((source/'evidence-inventory.json').read_bytes()==z.read('evidence-inventory.json'),'C3/C4 inventory changed')
        for name in REFINEMENT_INPUTS:
            path=source/name
            require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(source.resolve()),'Missing/unsafe C3/C4 input: '+name)
            expected=floor.hashlib.sha256(z.read(name)).hexdigest()
            require(digest(path)==expected==inventory[name]['sha256'],'C3/C4 input SHA mismatch: '+name)
            hashes[name]={'sha256':expected,'bytes':path.stat().st_size}
    old=floor.load_json(source/'diagnostic/input-provenance.json')
    for key in ('original_material','archived_five_case','archived_C1_C2'):
        require(old[key]['zip_sha256']==provenance[key]['zip_sha256'] and old[key]['files']==provenance[key]['files'],'C4 provenance differs: '+key)
    before=floor.load_json(source/'diagnostic/preflight.json')
    for key in ('head','versions','python'):require(before[key]==current[key],'Runtime differs from C4: '+key)
    require(floor.load_json(source/'local-versions.json')==current['versions'],'C4 runtime version records differ')
    run=floor.load_json(source/'run-summary.json');report=floor.load_json(source/'diagnostic/diagnostic-review.json')
    require(run['head']==run['head_after']==v2.BASELINE and run['physics_integration_steps']==run['physics_attempted_steps']==1200 and
            run['canonical_total_known_completed_steps']==6211 and run['execution_invariants_unchanged'],'Invalid C3/C4 run evidence')
    require(report['status']=='STEP_REFINEMENT_SENSITIVITY_BLOCKED_REVIEW_REQUIRED' and
            report['diagnostic_source_sha256']==REFINEMENT_SCRIPT_SHA256 and
            report['diagnostic_completed_steps']==report['diagnostic_attempted_steps']==1200 and
            report['original_model_unchanged'] and all(report[k+'_unchanged'] for k in provenance),'Invalid C3/C4 physical evidence')
    require([(c['name'],c['steps'],c['attempted_steps']) for c in report['cases']]==[('C3',400,400),('C4',800,800)] and
            all(c['horizon_completed'] and c['status']==refine.PASS_CASE for c in report['cases']),'C3/C4 incomplete')
    require(floor.load_json(source/'diagnostic/cases/C4/summary.json')==report['cases'][1],'C4 summaries differ')
    comparison=floor.load_json(source/'diagnostic/time-aligned-comparison.json')
    require(comparison==report['comparison'] and not comparison['sensitivity']['all_four_pass'],'Historical C3/C4 failure changed')
    return {'artifact_commit':REFINEMENT_COMMIT,'zip_sha256':REFINEMENT_ZIP_SHA256,'files':hashes,
        'C3_C4_reused_without_replay':True,'runtime_and_inputs_match_C4':True,'historical_sensitivity_failure_retained':True}


def copy_archived_C4(source,out,qpos,qvel,warmstart):
    previous=source/'diagnostic/cases/C4';initial=floor.load_json(previous/'initialization.json')
    require(initial['time_s']==floor.START_TIME and initial['initial_contacts']==0 and
        np.array_equal(initial['qpos'],qpos) and np.array_equal(initial['qvel'],qvel) and
        np.array_equal(initial['qacc_warmstart'],warmstart),'Archived C4 initialization differs from A0/source')
    series=load_series(previous/'states.npz',.0000625)
    require(len(series['time'])==801 and np.array_equal(series['qpos'][0],qpos) and np.array_equal(series['qvel'][0],qvel),'Archived C4 initial state or length differs')
    dest=out/'archived-C4';dest.mkdir(parents=True,exist_ok=False)
    for name in ('states.npz','summary.json','initialization.json'):shutil.copyfile(previous/name,dest/name)
    shutil.copyfile(source/'diagnostic/time-aligned-comparison.json',dest/'historical-C3-C4-comparison.json')
    save(dest/'source.json',{'artifact_commit':REFINEMENT_COMMIT,'copied_without_integration':True,
        'files':{p.name:digest(p) for p in dest.iterdir() if p.is_file()}})


def aligned_metrics(a,b,times=None):
    grid=a['time'] if times is None else np.asarray(times)
    for s in (a,b):
        idx=np.rint((grid-floor.START_TIME)/s['dt']).astype(int)
        require(len(grid)>0 and np.all(idx>=0) and np.all(idx<len(s['time'])),'Comparison lacks full requested coverage')
    return refine.aligned_metrics(a,b,grid)


def complete_series(path,case):
    series=load_series(path,case['dt'])
    require(len(series['time'])==case['max_steps']+1,'Comparison requires complete 50ms horizon')
    return series


def pair_metrics(a,b,sa,sb):
    aligned=aligned_metrics(a,b)
    metrics={k:aligned[k] for k in BUDGETS if k.startswith('max_common')}
    peak=lambda s:max(s['max_pre_step_penetration_m'],s['max_post_step_overlap_m'])
    metrics['abs_difference_peak_sampled_overlap_m']=abs(peak(sa)-peak(sb))
    metrics['abs_difference_peak_upward_velocity_m_s']=abs(sa['max_upward_velocity_m_s']-sb['max_upward_velocity_m_s'])
    return {**aligned,'sensitivity':sensitivity_gates(metrics)}


def compare_saved(out,cases):
    require(len(cases)==2 and all(c['horizon_completed'] and c['status']==PASS_CASE for c in cases),'Physical horizons incomplete')
    c4=complete_series(out/'archived-C4/states.npz',{'dt':.0000625,'max_steps':800})
    c5,c6=[complete_series(out/'cases'/c['name']/'states.npz',c) for c in CASES]
    require(all(np.array_equal(c4[k][0],s[k][0]) for s in (c5,c6) for k in ('qpos','qvel')),'Comparison initial states differ')
    archived=floor.load_json(out/'archived-C4/summary.json')
    pair45=pair_metrics(c4,c5,archived,cases[0]);pair56=pair_metrics(c5,c6,cases[0],cases[1])
    result={'C4_C5':pair45,'C5_C6':pair56,'sensitivity':pair56['sensitivity'],
        'full_physical_horizons_completed':True,'new_integration_steps':0,'C4_replayed':False,
        'historical_C3_C4_failure_retained':True,'time_shift_applied_s':0,
        'limits':'Exact common times over complete 50ms horizons; no event shift, truncation or formal material/task release.'}
    save(out/'time-aligned-comparison.json',result);return result


def matrix_status(cases,comparison):
    if len(cases)!=2 or not all(c['status']==PASS_CASE and c['horizon_completed'] for c in cases):return BLOCKED_PHYSICAL
    return PASS_MATRIX if comparison['sensitivity']['all_four_pass'] else BLOCKED_SENSITIVITY


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
        with (destination/'states.partial.jsonl').open('x') as state_log, gzip.open(destination / 'contacts.jsonl.gz', 'wt') as log:
            def record_state():
                # Incremental uncompressed records survive an outer-timeout kill.
                # NaN/Infinity tokens deliberately preserve invalid raw states if encountered.
                state_log.write(json.dumps({'returned_steps':summary['steps'],'time_s':float(data.time),
                    'qpos':data.qpos,'qvel':data.qvel},default=v2.serial,allow_nan=True)+'\n');state_log.flush()
            record_state()
            for step in range(case['max_steps']):
                require(time.monotonic() < deadline, 'Approved five-minute wall limit reached')
                before_qpos, before_qvel = data.qpos.copy(), data.qvel.copy()
                before_time = floor.START_TIME + step * case['dt']
                require(abs(data.time - before_time) < 1e-11, 'Pre-step clock drift')
                summary['attempted_steps'] += 1; progress(summary)
                mujoco.mj_step(model, data)
                summary['steps'] += 1; progress(summary); record_state()
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
        if 'data' in locals():
            np.savez_compressed(destination/'failure-primary-state.npz',qpos=data.qpos,qvel=data.qvel,time=data.time,
                qacc_warmstart=data.qacc_warmstart,attempted_steps=summary['attempted_steps'],returned_steps=summary['steps'])
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
    before=floor.model_signature(original)
    for case in CASES:
        prior_steps=sum(c['steps'] for c in report['cases']);prior_attempts=sum(c['attempted_steps'] for c in report['cases'])
        def progress(current):
            require(prior_attempts+current['attempted_steps']<=TOTAL_STEPS_MAX,'Approved global step budget exceeded')
            save(out/'progress.json',{'completed_cases':report['cases'],'active_case':current,
                'completed_steps':prior_steps+current['steps'],'attempted_steps':prior_attempts+current['attempted_steps'],
                'count_semantics':'Returned mj_step calls are lower bound; in-flight or raised call may have advanced state.',
                'new_ppo_updates':0})
        result=run_case(original,qpos,qvel,warmstart,case,out/'cases'/case['name'],deadline,progress)
        report['cases'].append(result);save(out/'diagnostic-review.json',report)
        require(floor.model_signature(original)==before,'Original model changed between cases')
        if result['status']!=PASS_CASE or not result['horizon_completed']:
            report['status']=BLOCKED_PHYSICAL
            report['not_run_cases']=[c['name'] for c in CASES[len(report['cases']):]]
            report['original_model_unchanged']=True
            return
    report['original_model_unchanged']=True
    report['comparison']=compare_saved(out,report['cases'])
    report['status']=matrix_status(report['cases'],report['comparison'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--approved-floor-reference',action='store_true',required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+WALL_SECONDS
    report={'status':'STARTED','cases':[],'stage11_status':'MATERIAL_BLOCKED_REVIEW_REQUIRED',
        'stage11_complete':False,'new_ppo_updates':0,'cloud_hours':0,'full_robot_steps':0,
        'formal_task_migration':False,'formal_physical_parameter_adoption':False,'prior_canonical_steps':7895,
        'replayed_prior_case_steps':0,'max_attempted_steps':TOTAL_STEPS_MAX,'max_wall_seconds':WALL_SECONDS,
        'sampling_phase':'Pre-integration contacts/forces plus every post-integration geometric overlap including terminal state',
        'approved_sensitivity_budgets':BUDGETS,'automatically_expanded_cases':False}
    provenance={};snapshot={};current=None
    # Persist zero counters before any preflight or import of historical data.
    save(out/'progress.json',{'completed_cases':[],'active_case':None,'completed_steps':0,'attempted_steps':0,'new_ppo_updates':0})
    try:
        current,snapshot=local_preflight(out)
        provenance['original_material']=floor.verify_source(SOURCE)
        provenance['archived_five_case']=stiff.verify_prior(PRIOR,provenance['original_material'],current)
        provenance['archived_C1_C2']=refine.verify_stiffness(STIFFNESS,provenance['original_material'],provenance['archived_five_case'],current)
        provenance['archived_C3_C4']=verify_refinement(REFINEMENT,provenance,current)
        save(out/'input-provenance.json',provenance)
        model=mujoco.MjModel.from_binary_path(str(SOURCE/'local-v2/material-fixture.mjb'))
        require(model.opt.timestep==floor.SOURCE_DT and model.opt.disableflags==0 and np.array_equal(model.opt.gravity,[0,0,-9.81]),'Original model options changed')
        qpos,qvel,expected=floor.read_source_states(SOURCE,model,out)
        warmstart=stiff.verify_restart(PRIOR,qpos,qvel,expected)
        copy_archived_C4(REFINEMENT,out,qpos,qvel,warmstart)
        require(time.monotonic()<deadline,'Approved wall budget exhausted before integration')
        run_matrix(model,qpos,qvel,warmstart,out,report,deadline)
    except Exception as exc:
        report['status']='FLOOR_REFERENCE_ERROR_STOP_REVIEW_REQUIRED';report['error']=str(exc)
        (out/'error.txt').write_text(traceback.format_exc())
    finally:
        progress=floor.load_json(out/'progress.json')
        report['diagnostic_completed_steps']=progress['completed_steps']
        report['diagnostic_attempted_steps']=progress['attempted_steps']
        report['step_count_exact']=progress['completed_steps']==progress['attempted_steps']
        report['canonical_total_known_completed_steps']=7895+progress['completed_steps']
        report['canonical_total_attempted_upper_bound']=7895+progress['attempted_steps']
        report['wall_seconds']=time.monotonic()-started
        for key,directory in [('original_material',SOURCE),('archived_five_case',PRIOR),('archived_C1_C2',STIFFNESS),('archived_C3_C4',REFINEMENT)]:
            if key in provenance:
                report[key+'_unchanged']=all((directory/n).is_file() and digest(directory/n)==m['sha256'] for n,m in provenance[key]['files'].items())
                if not report[key+'_unchanged']:report['status']='SOURCE_CHANGED_BLOCKED'
        if snapshot:
            report['installed_source_files_unchanged']=all((ROOT/n).is_file() and digest(ROOT/n)==m['sha256'] for n,m in snapshot.items())
            report['git_unchanged']=all(git(*args)==current[key] for key,args in [('head',('rev-parse','HEAD')),('tag',('rev-parse','stage-10-complete^{}')),('tag_type',('cat-file','-t','refs/tags/stage-10-complete')),('branch',('branch','--show-current')),('tracked_diff',('diff','--name-status')),('staged_diff',('diff','--cached','--name-status'))])
            if not report['installed_source_files_unchanged'] or not report['git_unchanged']:report['status']='SOURCE_CHANGED_BLOCKED'
        report['diagnostic_source_sha256']=digest(Path(__file__))
        save(out/'diagnostic-review.json',report)
    print('Stage11Diagnostic='+report['status'],flush=True)
    print('Stage11DiagnosticSteps='+str(report['diagnostic_completed_steps']),flush=True)
    return 0 if report['status'] in (PASS_MATRIX,BLOCKED_PHYSICAL,BLOCKED_SENSITIVITY) else 2


if __name__=='__main__':raise SystemExit(main())
