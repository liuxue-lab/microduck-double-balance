"""Four saved-state RTX5060 collision/forward probes; zero integration or PPO.

Queries native collision then fresh-data native forward for each saved state.
Known bad normals are recorded, never accepted or repaired. No graph capture,
dependency patch, parameter sweep, new rollout, or missing-case continuation.
"""
from __future__ import annotations
import argparse,contextlib,dataclasses,gzip,hashlib,importlib.util,json,shutil,time,traceback,warnings
from pathlib import Path
from unittest.mock import patch
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('normal_frozen_cuda',''+str(ROOT/'scripts/diagnose_head_contact_cuda_friction.py'))
cf=importlib.util.module_from_spec(sp);sp.loader.exec_module(cf)
cc,fr,floor,v2,ref=cf.cc,cf.fr,cf.floor,cf.v2,cf.ref
require,save,digest=cf.require,cf.save,cf.digest
SOURCE=Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage11/cuda-friction/20261004T104109Z')
SOURCE_SHA='b253028776a653328f334a93b5f7d9b74bd0dfdaad812d3eef3f3ca556a91733'
PRIOR_STEPS=13354
CASES=('GPU49','GPU50','GPU51','CPU50-rounded32')
ROUTES=('collision','forward')
WALL_SECONDS=300
PASS='CUDA_NORMAL_STATIC_PROBE_REVIEW_REQUIRED'
ERROR='CUDA_NORMAL_STATIC_PROBE_BLOCKED_REVIEW_REQUIRED'
INPUTS=('run-summary.json','diagnostic/preflight.json','diagnostic/diagnostic-review.json','diagnostic/material-fixture.mjb','diagnostic/material-geometry.npz','diagnostic/model-signature-before.json','diagnostic/cases/slide-x-minus/states.npz','diagnostic/cases/slide-x-minus/raw-contacts.jsonl.gz','diagnostic/cases/slide-x-minus/step-log.jsonl','diagnostic/cases/slide-x-minus/summary.json','diagnostic/archived-CPU-friction/slide-x-minus/states.npz')


def selected_states(source):
 with np.load(source/'diagnostic/cases/slide-x-minus/states.npz',allow_pickle=False) as z:
  q,v,t=z['qpos'].copy(),z['qvel'].copy(),z['device_time'].copy()
 require(q.shape==(52,7) and v.shape==(52,6) and q.dtype==v.dtype==np.float32,'Saved failed prefix shape/dtype changed')
 with np.load(source/'diagnostic/archived-CPU-friction/slide-x-minus/states.npz',allow_pickle=False) as z:cq,cv=z['qpos'].copy(),z['qvel'].copy()
 result=[{'name':'GPU'+str(i),'qpos':q[i],'qvel':v[i],'time':float(t[i]),'source_state_index':i,'source':'saved_CUDA','rounding':'none'} for i in (49,50,51)]
 result.append({'name':'CPU50-rounded32','qpos':cq[50].astype(np.float32),'qvel':cv[50].astype(np.float32),'time':float(np.float32(.1)),'source_state_index':50,'source':'saved_CPU','rounding':'explicit_float32_upload'})
 require(tuple(r['name'] for r in result)==CASES,'Unexpected states')
 for r in result:require(np.isfinite(r['qpos']).all() and np.isfinite(r['qvel']).all() and np.isfinite(r['time']),'Nonfinite input')
 return result


def primary(data):
 return {k:getattr(data,k).numpy().copy() for k in ('qpos','qvel','time','act','ctrl','qfrc_applied','xfrc_applied')}


def equal_primary(before,after):
 require(before.keys()==after.keys(),'Primary field set changed')
 for k in before:require(np.array_equal(before[k],after[k]),'Unexpected primary state mutation: '+k)


def model_fingerprint(value):
 """Hash device arrays plus scalar/nested model settings; no numerical edits."""
 if hasattr(value,'numpy'):
  a=value.numpy();return {'shape':list(a.shape),'dtype':str(a.dtype),'sha256':hashlib.sha256(a.tobytes()).hexdigest()}
 if dataclasses.is_dataclass(value):return {f.name:model_fingerprint(getattr(value,f.name)) for f in dataclasses.fields(value)}
 if isinstance(value,(dict,)):return {str(k):model_fingerprint(v) for k,v in value.items()}
 if isinstance(value,(list,tuple)):return [model_fingerprint(v) for v in value]
 if isinstance(value,(str,int,float,bool,type(None))):return value
 return repr(value)


@contextlib.contextmanager
def forbid_integration(mw,fw):
 def blocked(*args,**kwargs):raise RuntimeError('Integration is forbidden in static normal probe')
 with contextlib.ExitStack() as stack:
  for obj,names in ((mujoco,('mj_step','mj_step1','mj_step2')),(mw,('step','step1','step2')),(fw,('step','step1','step2','euler','implicit','rungekutta4','_advance'))):
   for name in names:
    if hasattr(obj,name):stack.enter_context(patch.object(obj,name,blocked))
  yield


def read_contacts(data,forces=None):
 count=int(data.nacon.numpy()[0]);require(0<=count<256,'GPU contact capacity reached')
 arrays={k:getattr(data.contact,k).numpy()[:count].copy() for k in ('geom','dist','pos','frame','dim','friction','solref','solimp','efc_address')}
 result=[]
 for i in range(count):
  row={k:v[i] for k,v in arrays.items()};row['efc_address']=int(arrays['efc_address'][i,0])
  row['force']=np.zeros(6,dtype=np.float32) if forces is None else forces[i].copy()
  result.append(row)
 return result


def assess(model,state,records,triangles):
 rows=cc.analysis_rows(v2.contact_rows(model,state['qpos'],state['qvel'],records,state['source_state_index'],'warp',triangles))
 require(len(rows)==len(records),'Unexpected contact not involving sphere')
 for row in rows:
  for k in ('dist','pos','frame','force'):require(np.isfinite(np.asarray(row[k])).all(),'Nonfinite '+k)
  require(row['other']=='head','Unexpected contact outside saved head prefix')
 # Retain the original geometry/material gate outcome without an integration.
 try:fr.classify_step(rows,{'head_overlap_m':0.,'floor_overlap_m':0.,'contacts':[]})
 except (RuntimeError,KeyError) as exc:
  if isinstance(exc,KeyError) or str(exc)!='Head v2 geometry gate failed':raise
  return rows,{'status':'ORIGINAL_HEAD_GATE_FAILED','error':str(exc),'physical_acceptance':False}
 return rows,{'status':'ORIGINAL_HEAD_GATE_PASS' if rows else 'NO_HEAD_CONTACT','physical_acceptance':False}


def run_query(model,mw_model,triangles,state,route,wp,mw,deadline,out,report):
 require(route in ROUTES and state['name'] in CASES,'Undeclared static query')
 require(time.monotonic()<deadline,'Static wall budget exhausted')
 cpu=mujoco.MjData(model);cpu.qpos[:]=state['qpos'];cpu.qvel[:]=state['qvel'];cpu.time=state['time']
 cpu_before=(cpu.qpos.copy(),cpu.qvel.copy(),cpu.time)
 mujoco.mj_forward(model,cpu)
 require(np.array_equal(cpu.qpos,cpu_before[0]) and np.array_equal(cpu.qvel,cpu_before[1]) and cpu.time==cpu_before[2],'CPU forward advanced primary state')
 require(not np.any(cpu.warning.number),'CPU geometry warning')
 cpu_rows=v2.cpu_records(model,cpu)
 # No saved warmstart exists at these checkpoints: explicit zero; solver force
 # is a fresh forward diagnostic, not an exact replay of historic solver state.
 cpu.qacc_warmstart[:]=0
 data=mw.put_data(model,cpu,nworld=1,nconmax=256,njmax=1024)
 before=primary(data)
 np.testing.assert_array_equal(before['qpos'][0],state['qpos']);np.testing.assert_array_equal(before['qvel'][0],state['qvel'])
 require(float(before['time'][0])==state['time'],'Upload time changed')
 init={'case':state,'route':route,'primary_before':before,'warmstart_before':data.qacc_warmstart.numpy(),'CPU_forward_contacts':cpu_rows,'force_semantics':'not computed for collision route; fresh zero-warmstart solve for forward route'}
 dest=out/'cases'/state['name']/route;dest.mkdir(parents=True,exist_ok=False);save(dest/'input.json',init)
 report['query_attempts']+=1;save(out/'progress.json',{'query_attempts':report['query_attempts'],'queries_completed':report['queries_completed'],'integration_steps':0,'attempted_integration_steps':0,'active_case':state['name'],'active_route':route})
 forces=None
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter('always')
  if route=='collision':mw.kinematics(mw_model,data);mw.collision(mw_model,data)
  else:
   mw.forward(mw_model,data)
   ids=wp.array(np.arange(256,dtype=np.int32),dtype=wp.int32);buffer=wp.zeros(256,dtype=wp.spatial_vector)
   mw.contact_force(mw_model,data,ids,False,buffer);forces=buffer.numpy()
  wp.synchronize()
 after=primary(data);records=read_contacts(data,forces)
 raw={'records':records,'primary_after':after,'qacc':data.qacc.numpy(),'qacc_warmstart_after':data.qacc_warmstart.numpy(),'geom_xpos':data.geom_xpos.numpy(),'geom_xmat':data.geom_xmat.numpy(),'warnings':[str(w.message) for w in caught],'nefc':data.nefc.numpy(),'nacon':data.nacon.numpy()}
 save(dest/'raw-result.json',raw) # Preserve evidence before any gate can raise.
 equal_primary(before,after)
 require(not caught,'Python warning during static CUDA query')
 require(int(data.nefc.numpy()[0])<1024,'Constraint capacity reached')
 rows,gate=assess(model,state,records,triangles)
 if route=='collision':
  for r in rows:
   for k in ('force','normal_force_N','tangent_force_N','torsional_torque_Nm','tangential_power_W','torsional_power_W','pyramidal_cone_utilization'):r[k]=None
 result={'case':state['name'],'route':route,'gate':gate,'contacts':rows,'primary_unchanged':True,'integration_steps':0,'force_computed':route=='forward','warning_counter_limitation':'Native Warp Data has no MuJoCo CPU warning counter; Python warnings/capacity/finite checks retained.'}
 save(dest/'result.json',result);report['queries_completed']+=1
 save(out/'progress.json',{'query_attempts':report['query_attempts'],'queries_completed':report['queries_completed'],'integration_steps':0,'attempted_integration_steps':0,'active_case':state['name'],'active_route':route})
 return result


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--zero-integration-normal-probe',required=True,action='store_true');args=parser.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);started=time.monotonic();deadline=started+WALL_SECONDS
 report={'status':'STARTED','query_attempts':0,'queries_completed':0,'results':[],'diagnostic_completed_steps':0,'diagnostic_attempted_steps':0,'canonical_total_known_completed_steps':PRIOR_STEPS,'stage11_status':'MATERIAL_BLOCKED_REVIEW_REQUIRED','stage11_complete':False,'new_ppo_updates':0,'full_robot_steps':0,'cloud_hours':0,'formal_task_migration':False,'formal_physical_parameter_adoption':False,'policy_success':None}
 save(out/'progress.json',{'query_attempts':0,'queries_completed':0,'integration_steps':0,'attempted_integration_steps':0})
 snapshot={};model=None;wm=None;signature=None;wm_before=None
 try:
  current,snapshot=ref.local_preflight(out)
  plan=floor.load_json(ROOT/'docs/audits/stage-11-cuda-normal-probe-plan.json')
  for name,sha in plan['frozen_files'].items():require(digest(ROOT/name)==sha,'Frozen source changed: '+name)
  import warp as wp
  import mujoco_warp as mw
  from mujoco_warp._src import forward as fw
  deps=Path(mw.__file__).parent
  for name,sha in plan['pinned_dependency_sources'].items():require(digest(deps/name)==sha,'Pinned dependency source changed: '+name)
  save(out/'dependency-sources.json',plan['pinned_dependency_sources'])
  files=fr.pinned_files(SOURCE,SOURCE_SHA,INPUTS)
  old=floor.load_json(SOURCE/'diagnostic/preflight.json');require(all(old[k]==current[k] for k in ('head','versions','python')),'Runtime differs from saved friction batch')
  previous=floor.load_json(SOURCE/'run-summary.json');require(previous['canonical_total_known_completed_steps']==PRIOR_STEPS and previous['physics_integration_steps']==previous['physics_attempted_steps']==371 and previous['execution_invariants_unchanged'],'Latest count/invariants changed')
  for name in ('material-fixture.mjb','material-geometry.npz'):shutil.copyfile(SOURCE/'diagnostic'/name,out/name)
  save(out/'input-provenance.json',{'source':str(SOURCE),'zip_sha256':SOURCE_SHA,'files':files,'integration_replayed':False})
  states=selected_states(SOURCE);save(out/'selected-states.json',states)
  model=mujoco.MjModel.from_binary_path(str(out/'material-fixture.mjb'));model,signature=fr.configured_copy(model);save(out/'model-signature-before.json',signature)
  with np.load(out/'material-geometry.npz',allow_pickle=False) as z:triangles=z['vertices'][z['faces']].copy()
  device=wp.get_device('cuda:0');cc.validate_device(device)
  with wp.ScopedDevice(device),forbid_integration(mw,fw):
   wm=mw.put_model(model);wm.opt.ls_parallel=True
   wm_before=model_fingerprint(wm);save(out/'device-model-before.json',wm_before)
   for state in states:
    for route in ROUTES:
     result=run_query(model,wm,triangles,state,route,wp,mw,deadline,out,report);report['results'].append(result);save(out/'diagnostic-review.json',report)
   require(time.monotonic()<=deadline,'Static wall budget exhausted')
  report['status']=PASS
 except Exception as exc:
  report['status']=ERROR;report['error']=str(exc);(out/'error.txt').write_text(traceback.format_exc())
 finally:
  if model is not None and signature is not None:
   after=floor.model_signature(model);save(out/'model-signature-after.json',after);report['CPU_model_unchanged']=after==signature
  if wm is not None and wm_before is not None:
   after=model_fingerprint(wm);save(out/'device-model-after.json',after);report['GPU_model_unchanged']=after==wm_before
  if snapshot:report['source_files_unchanged']=all(digest(ROOT/n)==m['sha256'] for n,m in snapshot.items())
  if any(report.get(k) is False for k in ('CPU_model_unchanged','GPU_model_unchanged','source_files_unchanged')):report['status']=ERROR;report['invariant_failure']=True
  report['wall_seconds']=time.monotonic()-started;report['not_completed_queries']=[{'case':s,'route':r} for s in CASES for r in ROUTES if not any(a['case']==s and a['route']==r for a in report['results'])]
  report['known_geometry_failure_is_recorded_not_waived']=True
  report['limitations']=['Fresh zero qacc_warmstart, not full historical solver-state replay','No native CCD witnesses captured; cannot prove full float32-cancellation causal chain','No graph capture; historical graph-specific effect remains possible if fresh calls differ','Original floor/sensitivity and matrix omissions retained; no physical acceptance or policy result']
  report['diagnostic_source_sha256']=digest(Path(__file__))
  save(out/'diagnostic-review.json',report)
 print('Stage11NormalProbeResult='+report['status'],flush=True)
 return 0 if report['status']==PASS else 2


if __name__=='__main__':raise SystemExit(main())
