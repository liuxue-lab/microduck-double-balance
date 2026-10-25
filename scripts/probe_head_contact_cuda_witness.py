"""Four read-only CUDA CCD/GJK witness queries on archived kinematics.

No dynamics, no collision-normal replacement, no installed dependency patch.
Original native functions are invoked in an independent instrumentation kernel.
Only outputs matching the archived native contact can support that contact's
mechanism claim. Other outcomes remain a diagnostic mismatch for review.
"""
from __future__ import annotations
import argparse,importlib.util,json,shutil,time,traceback,warnings
from pathlib import Path
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('witness_normal',ROOT/'scripts/probe_head_contact_cuda_normal.py');normal=importlib.util.module_from_spec(sp);sp.loader.exec_module(normal)
require,save,digest=normal.require,normal.save,normal.digest
ref,fr,floor,v2=normal.ref,normal.fr,normal.floor,normal.v2
SOURCE=Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage11/cuda-normal/20261004T112308Z')
SOURCE_SHA='3699710443bb05d5b5349fc26f05f46ab4162ac8c2226fe2b8b88e3e0bc5936c'
CASES=normal.CASES
WALL_SECONDS=300
PASS='CUDA_WITNESS_STATIC_REVIEW_REQUIRED'
ERROR='CUDA_WITNESS_STATIC_BLOCKED_REVIEW_REQUIRED'
INPUTS=('run-summary.json','diagnostic/preflight.json','diagnostic/diagnostic-review.json','diagnostic/selected-states.json','diagnostic/material-fixture.mjb','diagnostic/material-geometry.npz')+tuple('diagnostic/cases/'+c+'/collision/'+n for c in CASES for n in ('input.json','raw-result.json','result.json'))


def angle(a,b):
 a=np.asarray(a,float);b=np.asarray(b,float)
 require(np.linalg.norm(a)>0 and np.linalg.norm(b)>0,'Cannot compare zero normal')
 return float(np.degrees(np.arccos(np.clip((a@b)/(np.linalg.norm(a)*np.linalg.norm(b)),-1,1))))


def analyze(result,raw,reference):
 s,v,f=np.asarray(result['scalars']),np.asarray(result['vectors']),np.asarray(result['frame'])
 require(s.shape==(9,) and v.shape==(12,3) and f.shape==(3,3),'Unexpected witness result shape')
 require(np.isfinite(s).all() and np.isfinite(v).all() and np.isfinite(f).all(),'Nonfinite witness result')
 native=raw['records'];require(len(native)<=1,'Unexpected multiple saved contacts')
 present=bool(s[0]<0 and int(s[1])>0)
 match={'contact_presence':present==bool(native),'dist':None,'midpoint':None,'frame':None}
 if native:
  c=native[0]
  match.update(dist=float(s[0])==c['dist'],midpoint=np.array_equal(v[9],np.asarray(c['pos'],np.float32)),frame=np.array_equal(f,np.asarray(c['frame'],np.float32)))
 all_match=all(x is None or x for x in match.values())
 shallow=bool(s[5]==1)
 branch_match=shallow and float(s[0])==float(s[6]) and np.array_equal(v[0],v[4]) and np.array_equal(v[1],v[5])
 out={'native_contact_match':match,'all_native_contact_fields_match':all_match,'shallow_finite_branch':shallow,'separate_shrink_inflate_matches_native_CCD_witnesses':branch_match,'physical_acceptance':False,'normal_replacement_applied':False,'policy_success':None}
 if shallow and native:
  expected=np.asarray(reference['contacts'][0]['geometry']['expected_normal'])
  before=-v[6];after=-v[8]
  out.update(original_normal_error_deg=angle(after,expected),preinflate_direction_error_deg=angle(before,expected),preinflate_candidate_toward_ball=before,
   stored_inflated_witness_difference=v[10],stored_inflated_difference_norm_m=float(np.linalg.norm(v[10].astype(float))),
   x_and_z_witness_differences_zero=bool(v[10,0]==0 and v[10,2]==0),
   supporting_case_mechanism=all_match and branch_match and angle(after,expected)>5 and angle(before,expected)<=5)
 else:out['supporting_case_mechanism']=False
 out['interpretation']='Read-only candidate uses pre-inflation sphere-center-to-head direction. It is neither written to contact data nor integrated. Matching the static saved contact does not prove every historical initialization difference or all convex contact branches.'
 return out


def saved_cases(out,current):
 files=fr.pinned_files(SOURCE,SOURCE_SHA,INPUTS)
 old=floor.load_json(SOURCE/'diagnostic/preflight.json')
 require(all(old[k]==current[k] for k in ('head','versions','python')),'Runtime differs from saved CUDA normal probe')
 previous=floor.load_json(SOURCE/'run-summary.json')
 require(previous['queries_completed']==previous['query_attempts']==8 and previous['physics_integration_steps']==previous['physics_attempted_steps']==0 and previous['canonical_total_known_completed_steps']==13354 and previous['execution_invariants_unchanged'],'Prior static run counts/invariants changed')
 for name in ('material-fixture.mjb','material-geometry.npz'):shutil.copyfile(SOURCE/'diagnostic'/name,out/name)
 states=floor.load_json(SOURCE/'diagnostic/selected-states.json');require(tuple(s['name'] for s in states)==CASES,'Saved four-state scope changed')
 cases=[]
 for state in states:
  base=SOURCE/'diagnostic/cases'/state['name']/'collision'
  raw,initial,reference=(floor.load_json(base/n) for n in ('raw-result.json','input.json','result.json'))
  require(initial['case']==state,'Primary saved state mismatch')
  for k in initial['primary_before']:require(np.array_equal(initial['primary_before'][k],raw['primary_after'][k]),'Original static query changed '+k)
  require(np.asarray(raw['geom_xpos']).shape==(1,3,3) and np.asarray(raw['geom_xmat']).shape==(1,3,3,3),'Wrong saved kinematic shapes')
  require(np.isfinite(raw['geom_xpos']).all() and np.isfinite(raw['geom_xmat']).all(),'Nonfinite saved kinematics')
  dest=out/'saved-inputs'/state['name'];dest.mkdir(parents=True)
  for name in ('input.json','raw-result.json','result.json'):shutil.copyfile(base/name,dest/name)
  cases.append({'state':state,'raw':raw,'reference':reference})
 save(out/'input-provenance.json',{'source':str(SOURCE),'zip_sha256':SOURCE_SHA,'files':files,'kinematics':'exact archived CUDA float32 geom_xpos/xmat; no recomputation or integration'})
 return cases


def run_queries(wm,cases,kernel_module,out,report,deadline):
 kernel=kernel_module.make_kernel(wm.opt.ccd_iterations)
 require(tuple(c['state']['name'] for c in cases)==CASES,'Only declared four witnesses permitted')
 for case in cases:
  require(time.monotonic()<deadline,'300second witness budget exhausted')
  name=case['state']['name'];dest=out/'cases'/name;dest.mkdir(parents=True,exist_ok=False)
  report['query_attempts']+=1
  save(out/'progress.json',{'query_attempts':report['query_attempts'],'queries_completed':report['queries_completed'],'integration_steps':0,'attempted_integration_steps':0,'active_case':name})
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always');result=kernel_module.query(wm,case['raw']['geom_xpos'],case['raw']['geom_xmat'],kernel)
  save(dest/'witness.json',result);require(not caught,'Python warning in CUDA witness query: '+repr([str(w.message) for w in caught]))
  result_review=analyze(result,case['raw'],case['reference'])
  save(dest/'analysis.json',result_review)
  report['results'].append({'case':name,**result_review});report['queries_completed']+=1
  save(out/'progress.json',{'query_attempts':report['query_attempts'],'queries_completed':report['queries_completed'],'integration_steps':0,'attempted_integration_steps':0,'active_case':name})
  save(out/'diagnostic-review.json',report)
 require(time.monotonic()<=deadline,'300second witness budget exhausted')


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--zero-integration-witness-probe',required=True,action='store_true');args=parser.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+WALL_SECONDS
 report={'status':'STARTED','query_attempts':0,'queries_completed':0,'results':[],'diagnostic_completed_steps':0,'diagnostic_attempted_steps':0,'canonical_total_known_completed_steps':13354,'stage11_status':'MATERIAL_BLOCKED_REVIEW_REQUIRED','stage11_complete':False,'new_ppo_updates':0,'full_robot_steps':0,'cloud_hours':0,'formal_task_migration':False,'formal_physical_parameter_adoption':False,'policy_success':None,'collision_normal_replacement':False,'installed_dependency_patch':False}
 save(out/'progress.json',{'query_attempts':0,'queries_completed':0,'integration_steps':0,'attempted_integration_steps':0})
 snapshot={};model=None;wm=None;signature=None;wm_before=None
 try:
  current,snapshot=ref.local_preflight(out)
  plan=floor.load_json(ROOT/'docs/audits/stage-11-cuda-witness-plan.json')
  for name,sha in plan['frozen_files'].items():require(digest(ROOT/name)==sha,'Frozen source changed: '+name)
  import warp as wp
  import mujoco_warp as mw
  from mujoco_warp._src import forward as fw
  deps=Path(mw.__file__).parent
  for name,sha in plan['pinned_dependency_sources'].items():require(digest(deps/name)==sha,'Pinned runtime source changed: '+name)
  save(out/'dependency-sources.json',plan['pinned_dependency_sources'])
  sp=importlib.util.spec_from_file_location('head_witness_kernel',ROOT/'scripts/head_contact_witness_kernel.py');w=importlib.util.module_from_spec(sp);sp.loader.exec_module(w)
  cases=saved_cases(out,current)
  model=mujoco.MjModel.from_binary_path(str(out/'material-fixture.mjb'));model,signature=fr.configured_copy(model)
  require(model.ngeom==3 and model.npair==0 and np.array_equal(model.geom_type,[0,7,2]) and not np.any(model.geom_margin) and not np.any(model.geom_gap),'Instrumentation requires unchanged zero-margin sphere2/head1 pair')
  save(out/'model-signature-before.json',signature)
  device=wp.get_device('cuda:0');normal.cc.validate_device(device)
  with wp.ScopedDevice(device),normal.forbid_integration(mw,fw):
   wm=mw.put_model(model);wm.opt.ls_parallel=True;wm_before=normal.model_fingerprint(wm);save(out/'device-model-before.json',wm_before)
   run_queries(wm,cases,w,out,report,deadline)
  report['status']=PASS
 except Exception as exc:
  report['status']=ERROR;report['error']=str(exc);(out/'error.txt').write_text(traceback.format_exc())
 finally:
  if model is not None and signature is not None:
   after=floor.model_signature(model);save(out/'model-signature-after.json',after);report['CPU_model_unchanged']=after==signature
  if wm is not None and wm_before is not None:
   after=normal.model_fingerprint(wm);save(out/'device-model-after.json',after);report['GPU_model_unchanged']=after==wm_before
  if snapshot:report['source_files_unchanged']=all(digest(ROOT/n)==m['sha256'] for n,m in snapshot.items())
  if any(report.get(k) is False for k in ('CPU_model_unchanged','GPU_model_unchanged','source_files_unchanged')):report['status']=ERROR;report['invariant_failure']=True
  report['not_completed_cases']=[c for c in CASES if c not in [r['case'] for r in report['results']]]
  report['wall_seconds']=time.monotonic()-started;report['diagnostic_source_sha256']=digest(Path(__file__))
  report['limits']=['Static instrumented original functions may compile differently: exact saved-contact matching is mandatory for native-case mechanism support','Separate shrink/inflate must match original CCD witnesses before attributing that branch','No restored direction is applied to a simulation contact','This does not clear original floor/sensitivity failures, missing material cases, Stage10causality gaps, task migration or policy objective']
  save(out/'diagnostic-review.json',report)
 print('Stage11WitnessResult='+report['status'],flush=True)
 return 0 if report['status']==PASS else 2


if __name__=='__main__':raise SystemExit(main())
