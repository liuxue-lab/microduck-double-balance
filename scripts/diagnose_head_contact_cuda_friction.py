"""Six fresh original-parameter CUDA friction prefixes; no CPU replay or PPO.

One world at a time, max9000 attempted/returned calls and600s across all cases.
First floor event ends a case; other failures stop the batch. Original material
matrix and floor sensitivity blockers remain, even if every prefix is recorded.
"""
from __future__ import annotations
import argparse,copy,gzip,importlib.util,json,shutil,time,traceback,warnings
from pathlib import Path
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('cuda_friction_center',ROOT/'scripts/diagnose_head_contact_cuda_center.py')
cc=importlib.util.module_from_spec(spec);spec.loader.exec_module(cc)
ref,fr,floor,v2=cc.ref,cc.fr,cc.floor,cc.v2
require,save,digest=cc.require,cc.save,cc.digest
analysis_rows,validate_record,check_clock,validate_device=cc.analysis_rows,cc.validate_record,cc.check_clock,cc.validate_device
DT,MAX_STEPS,TOTAL_STEPS,WALL_SECONDS=.002,1500,9000,600
CASE_NAMES=fr.CASE_NAMES
PASS='CUDA_FRICTION_FULL_HORIZON_REVIEW_REQUIRED'
ERROR='CUDA_FRICTION_ERROR_STOP_REVIEW_REQUIRED'
SOURCE=floor.SOURCE
CPU=Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage11/head-friction/20261003T183812Z')
CPU_SHA='f692ff89cfdd0b1036570a2eb5cc4a7f679eb49bc407af37e5e3410478dd343d'
LATEST=Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage11/cuda-center-v2/20261004T101459Z')
LATEST_SHA='3a040814b969e94eabe8533a11e45ea21835f34441e1b75978787018f3b218dc'
PRIOR_STEPS=12983
CPU_CASE_FILES=('initial.json','summary.json','states.npz','contacts.jsonl.gz','raw-contacts.jsonl.gz')


def source_inputs(out,current):
 original=floor.verify_source(SOURCE)
 extra=fr.pinned_files(SOURCE,floor.SOURCE_ZIP_SHA256,fr.EXTRA_INPUTS)
 names=('run-summary.json','diagnostic/preflight.json','diagnostic/diagnostic-review.json','diagnostic/input-provenance.json')+tuple('diagnostic/cases/'+n+'/'+f for n in CASE_NAMES for f in CPU_CASE_FILES)
 cpu=fr.pinned_files(CPU,CPU_SHA,names)
 latest=fr.pinned_files(LATEST,LATEST_SHA,('run-summary.json','diagnostic/preflight.json','diagnostic/diagnostic-review.json'))
 for directory in (CPU,LATEST):
  old=floor.load_json(directory/'diagnostic/preflight.json')
  require(all(old[k]==current[k] for k in ('head','versions','python')),'Runtime differs from pinned '+str(directory))
 previous=floor.load_json(LATEST/'run-summary.json')
 require(previous['canonical_total_known_completed_steps']==PRIOR_STEPS and previous['physics_integration_steps']==previous['physics_attempted_steps']==282 and previous['execution_invariants_unchanged'],'Latest CUDA count/invariants mismatch')
 cpu_report=floor.load_json(CPU/'diagnostic/diagnostic-review.json')
 require(tuple(c['name'] for c in cpu_report['cases'])==CASE_NAMES and cpu_report['diagnostic_completed_steps']==cpu_report['diagnostic_attempted_steps']==1684,'Unexpected historical CPU cases/count')
 require(all(c['status'] in fr.FLOOR_ENDPOINTS and not c['horizon_completed'] for c in cpu_report['cases']),'Historical CPU censoring changed')
 prior_original=floor.load_json(CPU/'diagnostic/input-provenance.json')['original_material']
 require(prior_original['zip_sha256']==original['zip_sha256'] and all(prior_original['files'][n]==m for n,m in original['files'].items()),'CPU source fixture differs')
 with np.load(SOURCE/'local-v2/material-geometry.npz',allow_pickle=False) as z:triangles=z['vertices'][z['faces']].copy();center_xy=z['center_xy'].copy()
 cases=fr.selected_cases(triangles,center_xy,floor.load_json(SOURCE/'local-v2/cpu/release-center/initial.json'))
 for case in cases:
  initial=floor.load_json(CPU/'diagnostic/cases'/case['name']/'initial.json')
  for key in ('center','witness','normal','velocity','omega'):require(np.array_equal(case[key],initial[key]),'CPU initial '+key+' differs: '+case['name'])
  require(initial['name']==case['name'] and initial['time_s']==0,'CPU initialization metadata differs')
  dest=out/'archived-CPU-friction'/case['name'];dest.mkdir(parents=True)
  for name in CPU_CASE_FILES:shutil.copyfile(CPU/'diagnostic/cases'/case['name']/name,dest/name)
 for n in ('material-fixture.mjb','material-geometry.npz'):shutil.copyfile(SOURCE/'local-v2'/n,out/n)
 save(out/'selected-cases.json',cases)
 provenance={'original_material':{**original,'files':{**original['files'],**extra}},'CPU_friction':{'directory':str(CPU),'zip_sha256':CPU_SHA,'files':cpu},'latest_CUDA_center':{'directory':str(LATEST),'zip_sha256':LATEST_SHA,'files':latest},'CPU_replayed_steps':0}
 return cases,triangles,provenance


def run_case(model,triangles,case,out,deadline,progress):
 require(case['name'] in CASE_NAMES,'Only six declared friction cases authorized')
 dest=out/'cases'/case['name'];dest.mkdir(parents=True,exist_ok=False)
 summary=fr.new_summary(case);summary['backend']='RTX5060_MUJOCO_WARP';summary['status']='STARTED'
 summary['analyzed_steps']=0
 signature=floor.model_signature(model);poses=[];velocities=[];clocks=[];head_flags=[];capture_verified=False
 try:
  require(model.opt.timestep==DT and model.opt.disableflags==0,'Original dt/flags changed')
  require(np.array_equal(model.geom_solref[model.geom('sphere').id],[.003,1.]),'Original solref changed')
  import warp as wp
  validate_device(wp.get_device('cuda:0'))
  initial=v2.set_ball(model,case['center'],case['velocity'],case['omega'])
  require(initial.time==0 and all(c.dist>=-1e-8 for c in initial.contact),'Initial intersection or time changed')
  summary['unexpected_setup_advance_possible']=True;progress(summary)
  with warnings.catch_warnings(record=True) as capture_warnings:
   warnings.simplefilter('always')
   stepper=v2.WarpStepper(model,initial,'cuda:0',True)
  require(not capture_warnings,'Python warning during GPU setup: '+repr([str(w.message) for w in capture_warnings]))
  q=stepper.data.qpos.numpy()[0].copy();v=stepper.data.qvel.numpy()[0].copy();clock=float(stepper.data.time.numpy()[0])
  summary['unexpected_setup_advance_possible']=not (np.array_equal(q,initial.qpos.astype(np.float32)) and np.array_equal(v,initial.qvel.astype(np.float32)) and clock==0)
  progress(summary)
  require(not summary['unexpected_setup_advance_possible'],'Graph capture changed primary state; no launch authorized')
  capture_verified=True
  save(dest/'initialization.json',{'case':case,'cpu_qpos':initial.qpos,'cpu_qvel':initial.qvel,'cpu_qacc_warmstart':initial.qacc_warmstart,
   'device_qpos':q,'device_qvel':v,'device_qacc_warmstart':stepper.data.qacc_warmstart.numpy()[0],
   'device':stepper.device.name,'device_time':clock,'dtype':str(q.dtype),'graph_capture_primary_state_unchanged':True,
   'ls_parallel':True,'warmup_steps':0,'fixed_material_fixture':True,'original_dt_s':DT,'original_solref':[.003,1.]})
  poses.append(q.copy());velocities.append(v.copy());clocks.append(clock)
  probe=mujoco.MjData(model);post_data=mujoco.MjData(model);cpu_warnings=probe.warning.number.copy()
  with gzip.open(dest/'contacts.jsonl.gz','wt') as flat,gzip.open(dest/'raw-contacts.jsonl.gz','wt') as raw,(dest/'step-log.jsonl').open('x') as log,(dest/'states.partial.jsonl').open('x') as states:
   def state():
    states.write(json.dumps({'returned_steps':summary['steps'],'device_time_s':clock,'nominal_time_s':summary['steps']*DT,'qpos':q,'qvel':v},default=v2.serial,allow_nan=True)+'\n');states.flush()
   state()
   for i in range(MAX_STEPS):
    require(time.monotonic()<deadline,'600 second material budget exhausted')
    before_q,before_v,before_clock=q.copy(),v.copy(),clock
    summary['attempted_steps']+=1;require(summary['attempted_steps']<=MAX_STEPS,'Step budget exceeded');progress(summary)
    with warnings.catch_warnings(record=True) as caught:
     warnings.simplefilter('always');q,v,records=stepper.step()
    summary['steps']+=1;progress(summary)
    clock=float(stepper.data.time.numpy()[0]);poses.append(q.copy());velocities.append(v.copy());clocks.append(clock);state()
    require(not caught,'Python warning during GPU step: '+repr([str(w.message) for w in caught]))
    check_clock(before_clock,clock)
    require(np.isfinite(q).all() and np.isfinite(v).all(),'Nonfinite GPU state')
    require(int(stepper.data.nacon.numpy()[0])<256 and int(stepper.data.nefc.numpy()[0])<1024,'GPU contact/constraint capacity reached')
    raw.write(json.dumps({'step':i,'phase':'pre_integration','contacts':records},default=v2.serial,allow_nan=False)+'\n');raw.flush()
    rows=analysis_rows(v2.contact_rows(model,before_q,before_v,records,i,'warp',triangles))
    require(len(rows)==len(records),'Unexpected non-sphere contact')
    for r in rows:flat.write(json.dumps(r,default=v2.serial,allow_nan=False)+'\n')
    flat.flush()
    post_data.qpos[:]=q;post_data.qvel[:]=v;post_data.time=clock
    post=fr.post_geometry(model,probe,post_data,triangles)
    require(np.array_equal(probe.warning.number,cpu_warnings),'New warning in separate post-state geometry probe')
    log.write(json.dumps({'step':i,'nominal_pre_time_s':i*DT,'device_pre_time_s':before_clock,'device_post_time_s':clock,
      'qpos_before':before_q,'qvel_before':before_v,'qpos_after':q,'qvel_after':v,'contacts':rows,'post_geometry':post},default=v2.serial,allow_nan=False)+'\n');log.flush()
    endpoint=validate_record(rows,post);head_flags.append(fr.accumulate(summary,rows,post,i))
    summary['analyzed_steps']+=1
    summary['max_clock_quantization_error_s']=max(summary.get('max_clock_quantization_error_s',0.),abs(clock-summary['steps']*DT))
    progress(summary)
    if endpoint:
     summary['status']=endpoint;summary['critical_frame_steps'] += [i,i+1];break
    if i and head_flags[-1]!=head_flags[-2]:summary['critical_frame_steps'].append(i)
    if i%100==0:print(case['name']+' CUDA returned='+str(summary['steps'])+'/'+str(MAX_STEPS),flush=True)
   else:
    require(summary['head_contact_steps']>0,'No real head contact')
    summary['horizon_completed']=True;summary['status']=PASS
  require(time.monotonic()<=deadline,'600 second material wall budget exceeded')
 except Exception as exc:
  summary['status']=ERROR;summary['error']=str(exc)
  (dest/'error.txt').write_text(traceback.format_exc())
 finally:
  summary['capture_primary_state_verified']=capture_verified
  summary['first_unanalyzed_pre_step']=summary['analyzed_steps'] if summary['analyzed_steps']<summary['steps'] else None
  summary['completed_step_count_exact']=summary['steps']==summary['attempted_steps'] and not summary.get('unexpected_setup_advance_possible',False)
  summary['elapsed_simulation_s']=summary['steps']*DT
  summary['head_contact_intervals']=fr.contact_intervals(head_flags)
  summary['no_head_contact_intervals']=fr.contact_intervals([not x for x in head_flags])
  summary['model_arrays_and_options_unchanged']=floor.model_signature(model)==signature
  if not summary['model_arrays_and_options_unchanged']:summary['status']='MODEL_MUTATION_BLOCKED'
  np.savez_compressed(dest/'states.npz',qpos=np.asarray(poses),qvel=np.asarray(velocities),device_time=np.asarray(clocks),time=DT*np.arange(len(poses)),dt=DT)
  save(dest/'summary.json',summary);progress(summary)
 return summary



def run_matrix(model,triangles,cases,out,report,deadline):
 require(tuple(c['name'] for c in cases)==CASE_NAMES,'Exactly six named cases in declared order required')
 for case in cases:
  require(time.monotonic()<deadline,'600 second global material budget exhausted before next case')
  before_done=sum(c['steps'] for c in report['cases']);before_attempts=sum(c['attempted_steps'] for c in report['cases'])
  def progress(current):
   done,attempts=before_done+current['steps'],before_attempts+current['attempted_steps']
   require(0<=done<=attempts<=TOTAL_STEPS and current['attempted_steps']<=MAX_STEPS,'Six-case/global budget exceeded')
   save(out/'progress.json',{'completed_steps':done,'attempted_steps':attempts,'active_case':current,'completed_case_names':[c['name'] for c in report['cases']],
    'unexpected_setup_advance_possible':current.get('unexpected_setup_advance_possible',False),'new_ppo_updates':0})
  progress(fr.new_summary(case))
  result=run_case(model,triangles,case,out,deadline,progress);report['cases'].append(result)
  report['not_run_cases']=[n for n in CASE_NAMES if n not in [c['name'] for c in report['cases']]]
  save(out/'diagnostic-review.json',report)
  if result['status'] not in (PASS,*fr.FLOOR_ENDPOINTS) or not result['completed_step_count_exact']:
   report['status']=ERROR;break
 else:
  report['status']='CUDA_FRICTION_PREFIXES_REVIEW_REQUIRED'
 return report


def compare_saved(out,cases):
 results=[]
 for case in cases:
  if not case['capture_primary_state_verified'] or case['steps']==0:continue
  cpu=out/'archived-CPU-friction'/case['name'];gpu=out/'cases'/case['name']
  with np.load(cpu/'states.npz',allow_pickle=False) as z:cq,cv=z['qpos'].copy(),z['qvel'].copy();require(float(z['dt'])==DT,'CPU dt changed')
  with np.load(gpu/'states.npz',allow_pickle=False) as z:gq,gv=z['qpos'].copy(),z['qvel'].copy();require(float(z['dt'])==DT,'GPU dt changed')
  old=floor.load_json(cpu/'summary.json');require(len(cq)==old['steps']+1 and len(gq)==case['steps']+1,'Saved state length mismatch')
  n=min(len(cq),len(gq));pd=np.linalg.norm(cq[:n,:3]-gq[:n,:3],axis=1);vd=np.linalg.norm(cv[:n,:3]-gv[:n,:3],axis=1)
  cf=gzip.open(cpu/'contacts.jsonl.gz','rt');gf=gzip.open(gpu/'contacts.jsonl.gz','rt')
  with cf,gf:
   cpu_contact_steps={json.loads(line)['step'] for line in cf};gpu_contact_steps={json.loads(line)['step'] for line in gf}
  results.append({'case':case['name'],'status':'DESCRIPTIVE_ONLY_NO_PARITY_PASS_GATE','common_nominal_step_samples':n,'common_prefix_s':(n-1)*DT,
   'CPU_status':old['status'],'GPU_status':case['status'],'CPU_steps':old['steps'],'GPU_steps':case['steps'],
   'max_position_difference_m':float(pd.max()),'max_linear_velocity_difference_m_s':float(vd.max()),
   'initial_upload_position_error_m':float(pd[0]),'contact_presence_different_steps_in_common_prefix':sorted((cpu_contact_steps^gpu_contact_steps)&set(range(n-1))),
   'CPU_replayed_steps':0,'floor_or_head_contact_intervals_are_sampled_not_continuous_retention':True})
 save(out/'saved-CPU-GPU-comparison.json',{'status':'DESCRIPTIVE_BACKEND_COMPARISON_ONLY','cases':results,'no_floor_convergence_or_policy_acceptance_claim':True})
 return results


def select_videos(cases):
 available=[c for c in cases if c['steps']>0 and c.get('capture_primary_state_verified',False)]
 names=[];slides=[c for c in available if c['name'].startswith('slide-')]
 if slides:names.append(max(slides,key=lambda c:c['max_head_slip_m_s'])['name'])
 if any(c['name']=='normal-spin' for c in available):names.append('normal-spin')
 blocked=next((c['name'] for c in available if c['status'] not in (PASS,'FLOOR_EVENT_CENSORED')),None)
 if blocked and blocked not in names:names.append(blocked)
 return names


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--bounded-cuda-friction',required=True,action='store_true');args=parser.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+WALL_SECONDS
 report={'status':'STARTED','cases':[],'not_run_cases':list(CASE_NAMES),'stage11_status':'MATERIAL_BLOCKED_REVIEW_REQUIRED','stage11_complete':False,
  'new_ppo_updates':0,'full_robot_steps':0,'cloud_hours':0,'formal_task_migration':False,'formal_physical_parameter_adoption':False,
  'prior_canonical_steps':PRIOR_STEPS,'max_attempted_steps':TOTAL_STEPS,'max_steps_per_case':MAX_STEPS,'wall_budget_s':WALL_SECONDS,
  'original_material_matrix_omissions_retained':True,'CPU_cases_replayed':False,'policy_success':None}
 save(out/'progress.json',{'completed_steps':0,'attempted_steps':0,'active_case':None,'new_ppo_updates':0})
 snapshot={};provenance={};model=None
 try:
  current,snapshot=ref.local_preflight(out)
  plan=floor.load_json(ROOT/'docs/audits/stage-11-cuda-friction-plan.json')
  for name,sha in plan['frozen_files'].items():require(digest(ROOT/name)==sha,'Frozen source changed: '+name)
  import mujoco_warp
  deps=Path(mujoco_warp.__file__).parent
  for name,sha in plan['pinned_dependency_sources'].items():require(digest(deps/name)==sha,'Pinned MuJoCo Warp source changed: '+name)
  save(out/'dependency-sources.json',plan['pinned_dependency_sources'])
  cases,triangles,provenance=source_inputs(out,current);save(out/'input-provenance.json',provenance)
  model=mujoco.MjModel.from_binary_path(str(out/'material-fixture.mjb'));model,signature=fr.configured_copy(model);save(out/'model-signature-before.json',signature)
  run_matrix(model,triangles,cases,out,report,deadline)
  report['comparison']=compare_saved(out,report['cases'])
  save(out/'model-signature-after.json',floor.model_signature(model));require(floor.model_signature(model)==signature,'Original model changed')
 except Exception as exc:
  report['status']=ERROR;report['error']=str(exc);(out/'error.txt').write_text(traceback.format_exc())
 finally:
  progress=floor.load_json(out/'progress.json');done,attempts=progress['completed_steps'],progress['attempted_steps']
  report['diagnostic_completed_steps']=done;report['diagnostic_attempted_steps']=attempts
  report['unexpected_setup_advance_possible']=progress.get('unexpected_setup_advance_possible',False)
  report['step_count_exact']=done==attempts and not report['unexpected_setup_advance_possible']
  report['canonical_total_known_completed_steps']=None if report['unexpected_setup_advance_possible'] else PRIOR_STEPS+done
  report['physics_wall_seconds']=time.monotonic()-started
  report['not_run_cases']=[n for n in CASE_NAMES if n not in [c['name'] for c in report['cases']]]
  if snapshot:
   report['installed_sources_unchanged']=all((ROOT/n).is_file() and digest(ROOT/n)==m['sha256'] for n,m in snapshot.items())
   if not report['installed_sources_unchanged']:report['status']='SOURCE_CHANGED_BLOCKED'
  for key,directory in (('original_material',SOURCE),('CPU_friction',CPU),('latest_CUDA_center',LATEST)):
   if key in provenance:
    report[key+'_files_unchanged']=all((directory/n).is_file() and digest(directory/n)==m['sha256'] for n,m in provenance[key]['files'].items())
    if not report[key+'_files_unchanged']:report['status']='SOURCE_CHANGED_BLOCKED'
  report['diagnostic_source_sha256']=digest(Path(__file__));save(out/'diagnostic-review.json',report)
 # Saved-state videos only; no integration or retry after an error.
 report['videos']=[];report['video_errors']=[]
 if model is not None:
  for name in select_videos(report['cases']):
   try:
    case=next(c for c in report['cases'] if c['name']==name)
    report['videos'].append(fr.render_case(copy.deepcopy(model),out/'cases'/name,out/(name+'-saved.mp4'),case))
   except Exception as exc:report['video_errors'].append({'case':name,'error':str(exc)})
 save(out/'diagnostic-review.json',report);print('Stage11Diagnostic='+report['status'],flush=True)
 return 0 if report['status']=='CUDA_FRICTION_PREFIXES_REVIEW_REQUIRED' else 2


if __name__=='__main__':raise SystemExit(main())
