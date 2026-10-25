"""Isolated static normal-candidate validation; eleven pair queries, no dynamics.

The candidate preserves the sphere-center-to-mesh direction in the shallow
zero-margin sphere-mesh branch. It is not wired into the installed engine,
robot, task, solver or policy. Deep penetration is a fallback control only.
"""
from __future__ import annotations
import argparse,copy,importlib.util,json,shutil,time,traceback,warnings
from pathlib import Path
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('candidate_witness',ROOT/'scripts/probe_head_contact_cuda_witness.py');previous=importlib.util.module_from_spec(sp);sp.loader.exec_module(previous)
normal=previous.normal
require,save,digest=previous.require,previous.save,previous.digest
ref,fr,floor,v2=previous.ref,previous.fr,previous.floor,previous.v2
SOURCE=Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage11/cuda-witness/20261004T114251Z')
SOURCE_SHA='dd000bdae7f7d0a9db700e383e6855d14801fdcfbf714a4ccd45ee8019a6e007'
BASE_CASES=previous.CASES
CASE_NAMES=BASE_CASES+('z-ulp-minus','z-ulp-plus','z-1um-minus','z-1um-plus','z-100um-minus','z-100um-plus','deep-fallback-control')
WALL_SECONDS=300
PASS='CUDA_NORMAL_CANDIDATE_STATIC_REVIEW_REQUIRED'
ERROR='CUDA_NORMAL_CANDIDATE_STATIC_BLOCKED_REVIEW_REQUIRED'
INPUTS=('run-summary.json','diagnostic/preflight.json','diagnostic/diagnostic-review.json','diagnostic/material-fixture.mjb','diagnostic/material-geometry.npz')+tuple('diagnostic/saved-inputs/'+c+'/'+n for c in BASE_CASES for n in ('input.json','raw-result.json','result.json'))


def select_cases(source,vertices):
 cases=[]
 for name in BASE_CASES:
  raw=floor.load_json(source/'diagnostic/saved-inputs'/name/'raw-result.json')
  cases.append({'name':name,'kind':'saved_native_control','geom_xpos':np.asarray(raw['geom_xpos'],np.float32),'geom_xmat':np.asarray(raw['geom_xmat'],np.float32),'native_reference':raw,'physical_gate_expected':name!='CPU50-rounded32'})
 base=next(c for c in cases if c['name']=='GPU50')
 z=np.float32(base['geom_xpos'][0,2,2])
 offsets=[('z-ulp-minus',np.nextafter(z,np.float32(-np.inf))),('z-ulp-plus',np.nextafter(z,np.float32(np.inf))),
  ('z-1um-minus',np.float32(float(z)-1e-6)),('z-1um-plus',np.float32(float(z)+1e-6)),('z-100um-minus',np.float32(float(z)-1e-4)),('z-100um-plus',np.float32(float(z)+1e-4))]
 for name,new_z in offsets:
  c=copy.deepcopy(base);c.update(name=name,kind='static_pair_z_offset',native_reference=None,requested_ball_z_m=float(new_z),actual_float32_z_offset_m=float(new_z)-float(z),physical_gate_expected=True)
  c['geom_xpos'][0,2,2]=new_z;cases.append(c)
 c=copy.deepcopy(base);c.update(name='deep-fallback-control',kind='deep_branch_parity_only',native_reference=None,physical_gate_expected=False)
 # Vertex average lies in the convex hull: diagnostic deep branch input only.
 c['geom_xpos'][0,2]=np.mean(vertices,axis=0).astype(np.float32);cases.append(c)
 require(tuple(c['name'] for c in cases)==CASE_NAMES,'Static candidate scope changed')
 for c in cases:
  require(c['geom_xpos'].shape==(1,3,3) and c['geom_xmat'].shape==(1,3,3,3),'Kinematic input shape changed')
  require(np.isfinite(c['geom_xpos']).all() and np.isfinite(c['geom_xmat']).all(),'Nonfinite kinematic input')
 return cases


def analyze(case,result,triangles):
 s,v,f,cf=(np.asarray(result[k]) for k in ('scalars','vectors','frame','candidate_frame'))
 require(s.shape==(12,) and v.shape==(17,3) and f.shape==cf.shape==(3,3),'Candidate output shape changed')
 require(all(np.isfinite(x).all() for x in (s,v,f,cf)),'Nonfinite candidate output')
 old_contact=bool(s[0]<0 and s[1]>0);new_contact=bool(s[9]<0 and s[10]>0)
 preserved={'distance':float(s[0])==float(s[9]),'contact_count':s[1]==s[10],'witness1':np.array_equal(v[0],v[12]),'witness2':np.array_equal(v[1],v[13]),'midpoint':np.array_equal(v[9],v[15]),'contact_presence':old_contact==new_contact}
 require(all(preserved.values()),'Candidate changed distance, witnesses or contact presence')
 native_match=None
 if case['native_reference'] is not None:
  raw=case['native_reference']['records'];native_match=old_contact==bool(raw)
  if raw:native_match=native_match and float(s[0])==raw[0]['dist'] and np.array_equal(v[9],np.asarray(raw[0]['pos'],np.float32)) and np.array_equal(f,np.asarray(raw[0]['frame'],np.float32))
 out={'name':case['name'],'kind':case['kind'],'preserved':preserved,'original_fields_match_saved_CUDA':native_match,'candidate_branch_used':bool(s[11]),'original_contact_present':old_contact,'candidate_contact_present':new_contact,'formal_physics_adoption':False,'policy_success':None,'material_acceptance':False}
 if case['kind']=='deep_branch_parity_only':
  require(not bool(s[11]) and np.array_equal(f,cf),'Deep fallback changed')
  out.update(status='DEEP_BRANCH_PARITY_ONLY_NOT_PHYSICAL_PASS',deep_penetration_m=max(0.,-float(s[9])))
 elif new_contact:
  require(float(s[9])>=-.001,'Static candidate exceeds original1mm penetration gate')
  np.testing.assert_allclose(cf@cf.T,np.eye(3),rtol=0,atol=5e-7)
  require(np.linalg.det(cf)>.999999,'Candidate frame is not right handed')
  center=case['geom_xpos'][0,2]
  ng=-cf[0];og=-f[0]
  cg=v2.geometry_metrics(center,v[15]-.5*float(s[9])*ng,ng,float(s[9]),triangles)
  old=v2.geometry_metrics(center,v[9]-.5*float(s[0])*og,og,float(s[0]),triangles)
  out.update(candidate_geometry=cg,original_geometry=old,candidate_frame=cf,original_frame=f,
   candidate_original_geometry_gate=v2.selected_geometry_pass(cg,'v2'),original_geometry_gate=v2.selected_geometry_pass(old,'v2'))
  require(out['candidate_original_geometry_gate'],'Candidate violates original curved-surface gate')
  out['status']='CANDIDATE_STATIC_CONTACT_GATE_PASS_REVIEW_REQUIRED'
 else:out['status']='NO_CONTACT_CONTROL_PRESERVED'
 if native_match is False:out['status']='INSTRUMENTED_BASELINE_MISMATCH_REVIEW_REQUIRED'
 return out


def run_queries(wm,cases,kernel_module,triangles,out,report,deadline):
 require(tuple(c['name'] for c in cases)==CASE_NAMES,'Exactly eleven declared static queries required')
 kernel=kernel_module.make_kernel(wm.opt.ccd_iterations)
 for case in cases:
  require(time.monotonic()<deadline,'Static candidate300second budget exhausted')
  dest=out/'cases'/case['name'];dest.mkdir(parents=True,exist_ok=False);save(dest/'input.json',case)
  report['query_attempts']+=1
  def progress():save(out/'progress.json',{'query_attempts':report['query_attempts'],'queries_completed':report['queries_completed'],'integration_steps':0,'attempted_integration_steps':0,'active_case':case['name']})
  progress()
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always');result=kernel_module.query(wm,case['geom_xpos'],case['geom_xmat'],kernel)
  save(dest/'raw-result.json',result);require(not caught,'Warning in candidate kernel: '+repr([str(w.message) for w in caught]))
  reviewed=analyze(case,result,triangles);save(dest/'analysis.json',reviewed)
  report['results'].append(reviewed);report['queries_completed']+=1;progress();save(out/'diagnostic-review.json',report)
 require(time.monotonic()<=deadline,'Static candidate300second budget exhausted')


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--isolated-static-normal-candidate',required=True,action='store_true');args=parser.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+WALL_SECONDS
 report={'status':'STARTED','query_attempts':0,'queries_completed':0,'results':[],'diagnostic_completed_steps':0,'diagnostic_attempted_steps':0,'canonical_total_known_completed_steps':13354,'stage11_status':'MATERIAL_BLOCKED_REVIEW_REQUIRED','stage11_complete':False,'new_ppo_updates':0,'full_robot_steps':0,'cloud_hours':0,'formal_task_migration':False,'formal_physical_parameter_adoption':False,'policy_success':None,'installed_dependency_patch':False,'candidate_used_by_integrator_or_solver':False}
 save(out/'progress.json',{'query_attempts':0,'queries_completed':0,'integration_steps':0,'attempted_integration_steps':0})
 snapshot={};model=None;wm=None;signature=None;wm_before=None
 try:
  current,snapshot=ref.local_preflight(out)
  plan=floor.load_json(ROOT/'docs/audits/stage-11-cuda-candidate-plan.json')
  for name,sha in plan['frozen_files'].items():require(digest(ROOT/name)==sha,'Frozen source changed: '+name)
  import warp as wp
  import mujoco_warp as mw
  from mujoco_warp._src import forward as fw
  deps=Path(mw.__file__).parent
  for name,sha in plan['pinned_dependency_sources'].items():require(digest(deps/name)==sha,'Pinned runtime changed: '+name)
  save(out/'dependency-sources.json',plan['pinned_dependency_sources'])
  files=fr.pinned_files(SOURCE,SOURCE_SHA,INPUTS)
  old=floor.load_json(SOURCE/'diagnostic/preflight.json');require(all(old[k]==current[k] for k in ('head','versions','python')),'Runtime differs from witness source')
  prior=floor.load_json(SOURCE/'run-summary.json');require(prior['queries_completed']==prior['query_attempts']==4 and prior['physics_integration_steps']==prior['physics_attempted_steps']==0 and prior['canonical_total_known_completed_steps']==13354 and prior['execution_invariants_unchanged'],'Prior counts/invariants changed')
  for n in ('material-fixture.mjb','material-geometry.npz'):shutil.copyfile(SOURCE/'diagnostic'/n,out/n)
  save(out/'input-provenance.json',{'source':str(SOURCE),'zip_sha256':SOURCE_SHA,'files':files})
  with np.load(out/'material-geometry.npz') as z:vertices=z['vertices'].copy();triangles=vertices[z['faces']]
  cases=select_cases(SOURCE,vertices);save(out/'selected-cases.json',cases)
  model=mujoco.MjModel.from_binary_path(str(out/'material-fixture.mjb'));model,signature=fr.configured_copy(model)
  require(model.ngeom==3 and model.npair==0 and np.array_equal(model.geom_type,[0,7,2]) and not np.any(model.geom_margin) and not np.any(model.geom_gap),'Candidate only covers unchanged zero-margin sphere2/head1 pair')
  save(out/'model-signature-before.json',signature)
  sp=importlib.util.spec_from_file_location('head_normal_candidate',ROOT/'scripts/head_contact_candidate_kernel.py');kernel_module=importlib.util.module_from_spec(sp);sp.loader.exec_module(kernel_module)
  device=wp.get_device('cuda:0');normal.cc.validate_device(device)
  with wp.ScopedDevice(device),normal.forbid_integration(mw,fw):
   wm=mw.put_model(model);wm.opt.ls_parallel=True;wm_before=normal.model_fingerprint(wm);save(out/'device-model-before.json',wm_before)
   run_queries(wm,cases,kernel_module,triangles,out,report,deadline)
  report['status']=PASS
 except Exception as exc:
  report['status']=ERROR;report['error']=str(exc);(out/'error.txt').write_text(traceback.format_exc())
 finally:
  if model is not None and signature is not None:
   after=floor.model_signature(model);save(out/'model-signature-after.json',after);report['CPU_model_unchanged']=after==signature
  if wm is not None and wm_before is not None:
   after=normal.model_fingerprint(wm);save(out/'device-model-after.json',after);report['GPU_model_unchanged']=after==wm_before
  if snapshot:report['source_files_unchanged']=all(digest(ROOT/n)==v['sha256'] for n,v in snapshot.items())
  if any(report.get(k) is False for k in ('CPU_model_unchanged','GPU_model_unchanged','source_files_unchanged')):report['status']=ERROR;report['invariant_failure']=True
  report['not_completed_cases']=[c for c in CASE_NAMES if c not in [r['name'] for r in report['results']]]
  report['wall_seconds']=time.monotonic()-started;report['diagnostic_source_sha256']=digest(Path(__file__))
  report['limits']=['Only zero-margin ordered sphere-mesh pair in a standalone diagnostic kernel','No solver/integration, full-robot or policy evidence','Deep state only checks unchanged native fallback; not a physical penetration pass','Native baseline mismatch remains inconclusive','Floor2ms and sensitivity failures/missing cases/formal task migration remain blocked']
  save(out/'diagnostic-review.json',report)
 print('Stage11CandidateResult='+report['status'],flush=True)
 return 0 if report['status']==PASS else 2


if __name__=='__main__':raise SystemExit(main())
