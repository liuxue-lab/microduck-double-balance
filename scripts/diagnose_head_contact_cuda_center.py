"""One independent RTX5060 fixed-head material release; max1500 steps/600s.

Original 2ms/solref/friction/mesh unchanged. Stop at first floor arrival or any
physical gate failure. A censored head prefix is not a 3s pass or policy result.
"""
from __future__ import annotations
import argparse,copy,gzip,hashlib,importlib.util,json,shutil,time,traceback,warnings
from pathlib import Path
import mujoco
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def module(name,file):
 spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ref=module('cuda_center_ref','diagnose_head_contact_floor_reference.py')
fr=module('cuda_center_friction','diagnose_head_contact_friction.py')
floor,v2=fr.floor,fr.v2
require,save,digest=ref.require,ref.save,ref.digest
SOURCE=floor.SOURCE
DT,MAX_STEPS,WALL_SECONDS=.002,1500,600
PASS='CUDA_CENTER_FULL_HORIZON_REVIEW_REQUIRED'
INPUTS=fr.EXTRA_INPUTS+('local-v2/cpu/release-center/states.npz','local-v2/cpu/release-center/summary.json','local-v2/cpu/release-center/contacts.jsonl.gz','local-v2/cpu/release-center/raw-contacts.jsonl.gz')

def source_inputs(out):
 base=floor.verify_source(SOURCE);extra=fr.pinned_files(SOURCE,floor.SOURCE_ZIP_SHA256,INPUTS)
 case=floor.load_json(SOURCE/'local-v2/cpu/release-center/initial.json')
 require(case['name']=='release-center' and np.array_equal(case['velocity'],[0,0,0]) and np.array_equal(case['omega'],[0,0,0]),'Wrong archived release')
 cpu=floor.load_json(SOURCE/'local-v2/cpu/release-center/summary.json')
 require(cpu['status']=='PASS' and cpu['steps']==1500 and cpu['backend']=='cpu','Archived CPU case incomplete')
 # CPU PASS included floor contacts: never interpret it as head retention.
 require(cpu['floor_contacts']>0,'Unexpected historical CPU floor record')
 with np.load(SOURCE/'local-v2/material-geometry.npz',allow_pickle=False) as z:
  triangles=z['vertices'][z['faces']].copy();center_xy=z['center_xy'].copy()
 expected=next(c for c in v2.dynamic_matrix(triangles,center_xy) if c['name']=='release-center')
 require(json.loads(json.dumps(expected,default=v2.serial))==case,'Archived center differs from frozen matrix')
 dest=out/'archived-CPU-center';dest.mkdir()
 for n in ('initial.json','summary.json','states.npz','contacts.jsonl.gz','raw-contacts.jsonl.gz'):
  shutil.copyfile(SOURCE/'local-v2/cpu/release-center'/n,dest/n)
 shutil.copyfile(SOURCE/'local-v2/material-geometry.npz',out/'material-geometry.npz')
 shutil.copyfile(SOURCE/'local-v2/material-fixture.mjb',out/'material-fixture.mjb')
 return case,triangles,{'original_material':base,'center_files':extra,'CPU_replayed_steps':0,'historical_CPU_PASS_includes_floor_contacts':True}


def analysis_rows(rows):
 """Copy contact rows into the flat frame convention of the frozen CPU analyzer.

 Warp records use (3,3), CPU records use (9,). Row-major flattening preserves
 the three frame axes and forces; native raw records remain untouched.
 """
 result=[]
 for row in rows:
  frame=np.asarray(row['frame'])
  require(frame.shape in ((9,),(3,3)),'Unexpected contact frame shape')
  result.append({**row,'frame':frame.reshape(9).copy()})
 return result


def validate_record(rows,post):
 for row in rows:
  for key,shapes in (('dist',((),)),('pos',((3,),)),('frame',((9,),(3,3))),('force',((6,),))):
   value=np.asarray(row[key])
   require(value.shape in shapes,'Unexpected contact '+key+' shape')
   require(np.isfinite(value).all(),'Nonfinite contact '+key)
  require(-float(row['dist'])<=.001,'Pre-step penetration exceeds 1 mm')
 return fr.classify_step(rows,post)


def check_clock(previous,current):
 require(np.isfinite(current),'Nonfinite GPU clock')
 expected=float(np.float32(np.float32(previous)+np.float32(DT)))
 require(float(current)==expected,'GPU clock differs from one float32 2ms addition')
 return expected


def validate_device(device):
 require(device.is_cuda and '5060' in device.name,'Only local RTX5060 CUDA authorized; no CPU or cloud fallback')


def run_case(model,triangles,case,out,deadline,progress):
 dest=out/'cases/release-center';dest.mkdir(parents=True)
 summary=fr.new_summary(case);summary['backend']='RTX5060_MUJOCO_WARP';summary['status']='STARTED'
 summary['analyzed_steps']=0
 signature=floor.model_signature(model);poses=[];velocities=[];clocks=[];head_flags=[];capture_verified=False
 try:
  require(case['name']=='release-center','Only one center case authorized')
  require(model.opt.timestep==DT and model.opt.disableflags==0,'Original dt/flags changed')
  require(np.array_equal(model.geom_solref[model.geom('sphere').id],[.003,1.]),'Original solref changed')
  import warp as wp
  validate_device(wp.get_device('cuda:0'))
  initial=v2.set_ball(model,case['center'],case['velocity'],case['omega'])
  require(initial.time==0 and all(c.dist>=-1e-8 for c in initial.contact),'Initial intersection or time changed')
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
    if i%100==0:print('CUDA center returned='+str(summary['steps'])+'/'+str(MAX_STEPS),flush=True)
   else:
    require(summary['head_contact_steps']>0,'No real head contact')
    summary['horizon_completed']=True;summary['status']=PASS
  require(time.monotonic()<=deadline,'600 second material wall budget exceeded')
 except Exception as exc:
  summary['status']='CUDA_CENTER_ERROR_STOP_REVIEW_REQUIRED';summary['error']=str(exc)
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


def compare_saved(out,case):
 with np.load(out/'archived-CPU-center/states.npz',allow_pickle=False) as z:
  cpuq,cpuv=z['qpos'].copy(),z['qvel'].copy();require(float(z['dt'])==DT,'CPU dt mismatch')
 with np.load(out/'cases/release-center/states.npz',allow_pickle=False) as z:
  gpuq,gpuv=z['qpos'].copy(),z['qvel'].copy();require(float(z['dt'])==DT,'GPU dt mismatch')
 require(len(gpuq)==case['steps']+1 and len(cpuq)==1501,'Saved state coverage mismatch')
 n=len(gpuq);pd=np.linalg.norm(cpuq[:n,:3]-gpuq[:,:3],axis=1);vd=np.linalg.norm(cpuv[:n,:3]-gpuv[:,:3],axis=1)
 result={'status':'DESCRIPTIVE_BACKEND_COMPARISON_ONLY','common_nominal_step_samples':n,
  'recorded_prefix_s':case['steps']*DT,'max_position_difference_m':float(pd.max()),'max_linear_velocity_difference_m_s':float(vd.max()),
  'position_peak_step':int(pd.argmax()),'velocity_peak_step':int(vd.argmax()),'CPU_replayed_steps':0,
  'initial_position_upload_error_m':float(pd[0]),'finest_floor_pair_gates_applied':False,
  'limits':'Different backend/precision, no parity pass threshold adopted. Prefix may be floor-censored and cannot qualify a three-second case. Historical CPU full case included floor contacts.'}
 save(out/'saved-CPU-GPU-comparison.json',result);return result


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--bounded-cuda-center',action='store_true',required=True);args=parser.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+WALL_SECONDS
 report={'status':'STARTED','stage11_status':'MATERIAL_BLOCKED_REVIEW_REQUIRED','stage11_complete':False,
  'new_ppo_updates':0,'full_robot_steps':0,'cloud_hours':0,'formal_task_migration':False,'formal_physical_parameter_adoption':False,
  'prior_canonical_steps':12701,'prior_cuda_checker_error_steps':6,'diagnostic_revision':2,'original_CPU9_and_CUDA13_matrix_omissions_retained':True,
  'one_independent_CUDA_case_only':True,'max_attempted_steps':MAX_STEPS,'wall_budget_s':WALL_SECONDS,'cases':[]}
 def progress(s):
  require(0<=s['steps']<=s['attempted_steps']<=MAX_STEPS,'Step accounting outside one-case budget')
  save(out/'progress.json',{'completed_steps':s['steps'],'attempted_steps':s['attempted_steps'],'active_case':s,
   'unexpected_setup_advance_possible':s.get('unexpected_setup_advance_possible',False),'new_ppo_updates':0})
 save(out/'progress.json',{'completed_steps':0,'attempted_steps':0,'active_case':None,'new_ppo_updates':0})
 provenance={};snapshot={};model=None
 try:
  current,snapshot=ref.local_preflight(out)
  plan=floor.load_json(ROOT/'docs/audits/stage-11-cuda-center-plan.json')
  for n,sha in plan['frozen_files'].items():require(digest(ROOT/n)==sha,'Prior source changed: '+n)
  repair=floor.load_json(ROOT/'docs/audits/stage-11-cuda-center-v2-plan.json')
  for n,sha in repair['frozen_files'].items():require(digest(ROOT/n)==sha,'Prior CUDA source changed: '+n)
  import mujoco_warp
  dep_root=Path(mujoco_warp.__file__).parent
  dependencies={}
  for n,sha in plan['pinned_dependency_sources'].items():
   dependencies[n]={'sha256':digest(dep_root/n),'reviewed_sha256':sha}
   save(out/'dependency-sources.json',dependencies)
   require(dependencies[n]['sha256']==sha,'MuJoCo Warp source changed: '+n)
  # Same versions/Python as the six-step CUDA return. Fresh original initialization,
  # not a qpos/qvel splice: the complete solver state was not saved for resumption.
  previous=Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage11/cuda-center/20261004T095858Z')
  pinned=fr.pinned_files(previous,'de6c865a0785902b344317f477b822ab4b164c37fcbf786a223df144472652de',('diagnostic/preflight.json','diagnostic/diagnostic-review.json','run-summary.json'))
  old=floor.load_json(previous/'diagnostic/preflight.json')
  prior=floor.load_json(previous/'run-summary.json')
  require(prior['physics_integration_steps']==prior['physics_attempted_steps']==6 and prior['canonical_total_known_completed_steps']==12701,'Unexpected prior CUDA accounting')
  require(all(current[k]==old[k] for k in ('head','versions','python')),'Runtime differs from reviewed CUDA return')
  case,triangles,provenance=source_inputs(out);provenance['prior_cuda_checker_error_files']=pinned;save(out/'input-provenance.json',provenance)
  model=mujoco.MjModel.from_binary_path(str(out/'material-fixture.mjb'))
  signature=floor.model_signature(model);save(out/'model-signature-before.json',signature)
  result=run_case(model,triangles,case,out,deadline,progress);report['cases']=[result];report['status']=result['status']
  if result['steps']>0:report['comparison']=compare_saved(out,result)
  save(out/'model-signature-after.json',floor.model_signature(model))
  require(floor.model_signature(model)==signature,'Original fixture changed')
 except Exception as exc:
  report['status']='CUDA_CENTER_SETUP_OR_REVIEW_ERROR';report['error']=str(exc);(out/'error.txt').write_text(traceback.format_exc())
 finally:
  p=floor.load_json(out/'progress.json')
  report['diagnostic_completed_steps']=p['completed_steps'];report['diagnostic_attempted_steps']=p['attempted_steps']
  report['step_count_exact']=p['completed_steps']==p['attempted_steps'] and not p.get('unexpected_setup_advance_possible',False)
  report['canonical_total_known_completed_steps']=12701+p['completed_steps'];report['physics_wall_seconds']=time.monotonic()-started
  if snapshot:
   report['installed_sources_unchanged']=all((ROOT/n).is_file() and digest(ROOT/n)==m['sha256'] for n,m in snapshot.items())
   if not report['installed_sources_unchanged']:report['status']='SOURCE_CHANGED_BLOCKED'
  if provenance:
   frozen={**provenance['original_material']['files'],**provenance['center_files']}
   report['original_material_files_unchanged']=all((SOURCE/n).is_file() and digest(SOURCE/n)==m['sha256'] for n,m in frozen.items())
   if not report['original_material_files_unchanged']:report['status']='SOURCE_CHANGED_BLOCKED'
  report['diagnostic_source_sha256']=digest(Path(__file__));save(out/'diagnostic-review.json',report)
 # Rendering only saved states with copied display model; no extra step.
 if model is not None and report['cases'] and report['cases'][0]['steps']>0:
  try:
   report['video']=fr.render_case(copy.deepcopy(model),out/'cases/release-center',out/'cuda-center-saved.mp4',report['cases'][0])
  except Exception as exc:report['video_error']=str(exc)
  save(out/'diagnostic-review.json',report)
 print('Stage11Diagnostic='+report['status'],flush=True)
 return 0 if report['status'] in (PASS,*fr.FLOOR_ENDPOINTS) else 2


if __name__=='__main__':raise SystemExit(main())
