"""Independent native-witness candidate v2: static guards then six prefixes.

No installed dependency patch or formal task adoption. Original fixed material,
2ms, friction/solver settings, six initial conditions and geometry gates remain.
Stop each prefix at the first floor event; other failures stop the batch.
"""
from __future__ import annotations
import argparse,copy,gzip,importlib.util,json,shutil,time,traceback,warnings
from pathlib import Path
import mujoco,numpy as np
ROOT=Path(__file__).resolve().parents[1]
def module(name,file):
 sp=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);return m
static=module('runtime_static','validate_head_contact_cuda_candidate.py')
normal=static.normal;cf=normal.cf;cc,fr,floor,v2,ref=cf.cc,cf.fr,cf.floor,cf.v2,cf.ref
require,save,digest=cf.require,cf.save,cf.digest
SOURCE=Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage11/candidate-runtime/20261004T123613Z')
STATIC_SOURCE=Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage11/cuda-candidate/20261004T120651Z')
STATIC_SOURCE_SHA='beabbcf9b71be7c1569ac3b20997ad68734967bbfe0a224f89128e5d98461672'
SOURCE_SHA='58c345d4f6f7d958aba4b3619161bb722b867100c96de778033a494802121546'
CASE_NAMES=('slide-y-plus','slide-x-minus','slide-x-plus','slide-y-minus','initial-no-slip-roll','normal-spin')
DT,MAX_STEPS,TOTAL_STEPS,WALL_SECONDS=.002,1500,9000,600
PASS='NATIVE_WITNESS_RUNTIME_V2_PREFIXES_REVIEW_REQUIRED'
ERROR='NATIVE_WITNESS_RUNTIME_V2_BLOCKED_REVIEW_REQUIRED'
PRIOR_STEPS=13932
KERNEL_CACHE={}
CONTACT_FIELDS=('geom','dist','pos','frame','dim','friction','solref','solimp','efc_address')


def runtime_modules():
 return module('head_runtime','head_contact_candidate_runtime.py'),module('head_kernel_v2','head_contact_native_witness_kernel.py')


def contract(model):
 require(model.nq==7 and model.nv==6 and model.nu==0 and model.nsensor==0 and model.nbody==3,'Only fixed material fixture authorized')
 require(model.ngeom==3 and model.npair==0 and np.array_equal(model.geom_type,[0,7,2]),'Only floor0/head1/sphere2 fixture authorized')
 require(not np.any(model.geom_margin) and not np.any(model.geom_gap),'Zero-margin candidate scope changed')
 require(model.opt.timestep==DT and int(model.opt.integrator)==3 and model.opt.disableflags==0,'Original2ms/implicitfast/flags changed')
 require(np.array_equal(model.geom_solref[2],[.003,1.]),'Original solref changed')


def parity(a,b):
 require(len(a)==len(b),'Contact count differs')
 for left,right in zip(a,b):
  for key in (*CONTACT_FIELDS,'force'):require(np.array_equal(left[key],right[key]),'Forward parity differs: '+key)


def hook_decision(native,result,xpos,xmat,triangles):
 heads=[r for r in native if set(np.asarray(r['geom']).tolist())=={1,2}]
 require(len(heads)<=1 and all(np.array_equal(r['geom'],[2,1]) for r in heads),'Unsupported head contact order/count')
 for r in native:require(set(np.asarray(r['geom']).tolist()) in ({1,2},{0,2}),'Unexpected collision pair')
 case={'name':'runtime-pair','kind':'runtime_pair','native_reference':{'records':heads},'geom_xpos':xpos,'geom_xmat':xmat}
 analysis=static.analyze(case,result,triangles)
 require(analysis['original_fields_match_saved_CUDA'],'Independent pair does not exactly match live native collision')
 return analysis


class FrameHook:
 def __init__(self,kernel_module,runtime,wm,triangles,log):
  self.kernel_module=kernel_module;self.runtime=runtime
  key=(kernel_module,wm.opt.ccd_iterations)
  if key not in KERNEL_CACHE:KERNEL_CACHE[key]=kernel_module.make_kernel(wm.opt.ccd_iterations)
  self.kernel=KERNEL_CACHE[key]
  self.triangles=triangles;self.log=log;self.calls=0;self.replacements=0;self.last=None
 def __call__(self,wm,data):
  self.calls+=1
  native=normal.read_contacts(data);xpos=data.geom_xpos.numpy();xmat=data.geom_xmat.numpy()
  result=self.kernel_module.query(wm,xpos,xmat,self.kernel)
  raw={'query':self.calls,'time':data.time.numpy(),'geom_xpos':xpos,'geom_xmat':xmat,'native_contacts':native,'pair_result':result}
  if self.log:self.log.write(json.dumps({'raw':raw},default=v2.serial,allow_nan=False)+'\n');self.log.flush()
  analysis=hook_decision(native,result,xpos,xmat,self.triangles)
  indices=[i for i,r in enumerate(native) if np.array_equal(r['geom'],[2,1])]
  if indices and analysis['candidate_branch_used']:
   self.runtime.replace_contact_frame(data,indices[0],result['candidate_frame']);self.replacements+=1
  after=normal.read_contacts(data)
  require(len(after)==len(native),'Frame replacement changed contact count')
  for i,(old,new) in enumerate(zip(native,after)):
   for key in CONTACT_FIELDS:
    expected=result['candidate_frame'] if key=='frame' and i in indices and analysis['candidate_branch_used'] else old[key]
    require(np.array_equal(new[key],expected),'Frame replacement changed '+key)
  self.last={'analysis':analysis,'replacement_count':self.replacements,'contacts_after':after}
  if self.log:self.log.write(json.dumps({'checked':self.last},default=v2.serial,allow_nan=False)+'\n');self.log.flush()


def forward_once(wm,data,route,runtime,hook,wp,mw):
 from mujoco_warp._src import forward as fw
 before=normal.primary(data)
 if route=='native':mw.forward(wm,data)
 else:runtime.forward(wm,data,hook=hook)
 ids=wp.array(np.arange(256,dtype=np.int32),dtype=wp.int32);forces=wp.zeros(256,dtype=wp.spatial_vector)
 mw.contact_force(wm,data,ids,False,forces)
 normal.equal_primary(before,normal.primary(data))
 require(int(data.nefc.numpy()[0])<1024,'Constraint capacity reached')
 require(np.isfinite(data.qacc.numpy()).all(),'Nonfinite forward acceleration')
 return normal.read_contacts(data,forces.numpy()),data.qacc.numpy().copy()


def select_pair_cases(static_source,runtime_source):
 cases=[]
 for name in static.CASE_NAMES:
  c=floor.load_json(static_source/'diagnostic/cases'/name/'input.json')
  c['geom_xpos']=np.asarray(c['geom_xpos'],np.float32);c['geom_xmat']=np.asarray(c['geom_xmat'],np.float32);cases.append(c)
 with gzip.open(runtime_source/'diagnostic/cases/slide-y-plus/normal-candidate.jsonl.gz','rt') as stream:raw=[json.loads(line)['raw'] for line in stream if '"raw"' in line]
 require(len(raw)==6 and raw[-1]['query']==6,'Failed raw query changed')
 h=raw[-1];cases.append({'name':'YPLUS5','kind':'saved_runtime_blocker','native_reference':{'records':h['native_contacts']},'geom_xpos':np.asarray(h['geom_xpos'],np.float32),'geom_xmat':np.asarray(h['geom_xmat'],np.float32)})
 require(tuple(c['name'] for c in cases)==static.CASE_NAMES+('YPLUS5',),'Exactly twelve static cases required')
 return cases


def pair_controls(wm,triangles,cases,out,report,deadline,kernel_module):
 require(tuple(c['name'] for c in cases)==static.CASE_NAMES+('YPLUS5',),'Static pair scope changed')
 key=(kernel_module,wm.opt.ccd_iterations)
 if key not in KERNEL_CACHE:KERNEL_CACHE[key]=kernel_module.make_kernel(wm.opt.ccd_iterations)
 for c in cases:
  require(time.monotonic()<deadline,'600second budget exhausted in static pairs')
  dest=out/'pair-controls'/c['name'];dest.mkdir(parents=True);save(dest/'input.json',c)
  report['pair_query_attempts']+=1;save(out/'pair-progress.json',{'attempts':report['pair_query_attempts'],'completed':report['pair_queries_completed']})
  with warnings.catch_warnings(record=True) as caught:
   warnings.simplefilter('always');r=kernel_module.query(wm,c['geom_xpos'],c['geom_xmat'],KERNEL_CACHE[key])
  save(dest/'raw-result.json',r);require(not caught,'Warning in native-witness pair query')
  a=static.analyze(c,r,triangles);save(dest/'analysis.json',a);report['pair_controls'].append(a);report['pair_queries_completed']+=1
  save(out/'pair-progress.json',{'attempts':report['pair_query_attempts'],'completed':report['pair_queries_completed']})
 require(report['pair_queries_completed']==report['pair_query_attempts']==12,'Pair controls incomplete')


def saved_forward_controls(model,wm,triangles,states,out,report,deadline,wp,mw,runtime,kernel_module):
 require(tuple(s['name'] for s in states)==normal.CASES+('YPLUS5',),'Saved forward controls changed')
 for state in states:
  results={};dest=out/'forward-controls'/state['name'];dest.mkdir(parents=True)
  for route in ('native','passthrough','candidate'):
   require(time.monotonic()<deadline,'600second budget exhausted in forward controls')
   cpu=mujoco.MjData(model);cpu.qpos[:]=state['qpos'];cpu.qvel[:]=state['qvel'];cpu.time=state['time'];mujoco.mj_forward(model,cpu)
   data=mw.put_data(model,cpu,nworld=1,nconmax=256,njmax=1024);data.qacc_warmstart.zero_()
   before=normal.primary(data)
   with (dest/(route+'-hook.jsonl')).open('x') as log:
    hook=FrameHook(kernel_module,runtime,wm,triangles,log) if route=='candidate' else None
    report['forward_query_attempts']+=1;save(out/'forward-progress.json',{'attempts':report['forward_query_attempts'],'completed':report['forward_queries_completed']})
    with warnings.catch_warnings(record=True) as caught:
     warnings.simplefilter('always');records,qacc=forward_once(wm,data,route,runtime,hook,wp,mw)
    record={'state':state,'route':route,'records':records,'qacc':qacc,'primary_unchanged':True,'hook':hook.last if hook else None}
    save(dest/(route+'.json'),record);require(not caught,'Warning in saved forward control')
    results[route]=record;report['forward_queries_completed']+=1
    if route=='candidate' and hook.replacements:report['candidate_used_by_integrator_or_solver']=True
    save(out/'forward-progress.json',{'attempts':report['forward_query_attempts'],'completed':report['forward_queries_completed']})
   normal.equal_primary(before,normal.primary(data))
  parity(results['native']['records'],results['passthrough']['records'])
  require(np.array_equal(results['native']['qacc'],results['passthrough']['qacc']),'Copied forward acceleration differs from native')
  rows,status=normal.assess(model,state,results['candidate']['records'],triangles)
  require(status['status'] in ('ORIGINAL_HEAD_GATE_PASS','NO_HEAD_CONTACT'),'Candidate solved forward violates original geometry')
  for row in rows:
   require(row['normal_force_N']>=-1e-8,'Negative normal force')
   if row['pyramidal_cone_utilization'] is not None:require(row['pyramidal_cone_utilization']<=1.0001,'Solved friction cone violated')
  review={'name':state['name'],'passthrough_exact':True,'candidate_geometry_status':status,'candidate_rows':rows,'no_integration':True,'fresh_zero_warmstart':True,'not_historical_force_replay':True}
  report['forward_controls'].append(review);save(dest/'review.json',review)
 require(report['forward_queries_completed']==report['forward_query_attempts']==15,'Forward control count incomplete')


def run_case(model,wm,triangles,case,out,deadline,progress,wp,mw,runtime,kernel_module):
 from mujoco_warp._src import forward as fw
 require(case['name'] in CASE_NAMES,'Undeclared case');contract(model)
 dest=out/'cases'/case['name'];dest.mkdir(parents=True,exist_ok=False)
 summary=fr.new_summary(case);summary.update(backend='RTX5060_ISOLATED_CANDIDATE_NO_GRAPH',analyzed_steps=0,forward_attempts=0,forward_completed=0,frame_replacements=0)
 signature=floor.model_signature(model);poses=[];velocities=[];clocks=[];flags=[]
 try:
  initial=v2.set_ball(model,case['center'],case['velocity'],case['omega']);require(initial.time==0 and all(c.dist>=-1e-8 for c in initial.contact),'Initial intersection/time changed')
  data=mw.put_data(model,initial,nworld=1,nconmax=256,njmax=1024)
  q=data.qpos.numpy()[0].copy();v=data.qvel.numpy()[0].copy();clock=float(data.time.numpy()[0])
  require(np.array_equal(q,initial.qpos.astype(np.float32)) and np.array_equal(v,initial.qvel.astype(np.float32)) and clock==0,'Primary state changed during upload')
  summary['primary_upload_verified']=True;summary['unexpected_setup_advance_possible']=False
  save(dest/'initialization.json',{'case':case,'device_qpos':q,'device_qvel':v,'device_qacc_warmstart':data.qacc_warmstart.numpy(),'cpu_qacc_warmstart':initial.qacc_warmstart,'capture':False,'warmup_steps':0,'ls_parallel':True,'fixed_material_fixture':True,'normal_replacement_is_diagnostic_only':True})
  poses.append(q.copy());velocities.append(v.copy());clocks.append(clock)
  probe=mujoco.MjData(model);post_data=mujoco.MjData(model);cpu_warnings=probe.warning.number.copy()
  with gzip.open(dest/'normal-candidate.jsonl.gz','wt') as hooks,gzip.open(dest/'contacts.jsonl.gz','wt') as flat,gzip.open(dest/'raw-contacts.jsonl.gz','wt') as raw,(dest/'step-log.jsonl').open('x') as log,(dest/'states.partial.jsonl').open('x') as states:
   hook=FrameHook(kernel_module,runtime,wm,triangles,hooks)
   def state():
    states.write(json.dumps({'returned_steps':summary['steps'],'device_time_s':clock,'qpos':q,'qvel':v},default=v2.serial,allow_nan=True)+'\n');states.flush()
   state()
   for i in range(MAX_STEPS):
    require(time.monotonic()<deadline,'600second runtime budget exhausted')
    before_q,before_v,before_clock=q.copy(),v.copy(),clock
    summary['forward_attempts']+=1;progress(summary)
    with warnings.catch_warnings(record=True) as caught:
     warnings.simplefilter('always');records,qacc=forward_once(wm,data,'candidate',runtime,hook,wp,mw)
    summary['forward_completed']+=1;summary['frame_replacements']=hook.replacements;progress(summary)
    raw.write(json.dumps({'step':i,'phase':'pre_integration','contacts':records,'qacc':qacc,'qfrc_constraint':data.qfrc_constraint.numpy(),'qfrc_smooth':data.qfrc_smooth.numpy(),'qacc_smooth':data.qacc_smooth.numpy()},default=v2.serial,allow_nan=False)+'\n');raw.flush()
    require(not caught,'Warning in candidate forward')
    rows=cc.analysis_rows(v2.contact_rows(model,before_q,before_v,records,i,'warp',triangles))
    require(len(rows)==len(records),'Unexpected non-sphere contact')
    for row in rows:flat.write(json.dumps(row,default=v2.serial,allow_nan=False)+'\n')
    flat.flush()
    # Check the original contact gates before permitting any integration.
    require(cf.validate_record(rows,{'head_overlap_m':0.,'floor_overlap_m':0.,'contacts':[]}) is None,'Unexpected pre-step floor contact')
    for row in rows:
     require(row['normal_force_N']>=-1e-8,'Negative normal force')
     if row['pyramidal_cone_utilization'] is not None:require(row['pyramidal_cone_utilization']<=1.0001,'Solved friction cone violated')
    require(time.monotonic()<deadline,'600second runtime budget exhausted before integration')
    summary['attempted_steps']+=1;progress(summary)
    with warnings.catch_warnings(record=True) as caught:
     warnings.simplefilter('always');fw.implicit(wm,data);wp.synchronize()
    summary['steps']+=1;progress(summary)
    q=data.qpos.numpy()[0].copy();v=data.qvel.numpy()[0].copy();clock=float(data.time.numpy()[0])
    poses.append(q.copy());velocities.append(v.copy());clocks.append(clock);state()
    require(not caught,'Warning in candidate integration');cc.check_clock(before_clock,clock)
    require(np.isfinite(q).all() and np.isfinite(v).all(),'Nonfinite dynamic state')
    require(not np.any(data.xfrc_applied.numpy()) and not np.any(data.qfrc_applied.numpy()),'External assistance detected')
    post_data.qpos[:]=q;post_data.qvel[:]=v;post_data.time=clock;post=fr.post_geometry(model,probe,post_data,triangles)
    require(np.array_equal(probe.warning.number,cpu_warnings),'Warning in separate geometry check')
    log.write(json.dumps({'step':i,'qpos_before':before_q,'qvel_before':before_v,'qpos_after':q,'qvel_after':v,'contacts':rows,'post_geometry':post,'device_pre_time_s':before_clock,'device_post_time_s':clock},default=v2.serial,allow_nan=False)+'\n');log.flush()
    endpoint=cf.validate_record(rows,post);flags.append(fr.accumulate(summary,rows,post,i));summary['analyzed_steps']+=1;progress(summary)
    if endpoint:summary['status']=endpoint;summary['critical_frame_steps'] += [i,i+1];break
    if i and flags[-1]!=flags[-2]:summary['critical_frame_steps'].append(i)
    if i%100==0:print(case['name']+' candidate returned='+str(summary['steps'])+'/1500',flush=True)
   else:
    require(summary['head_contact_steps']>0,'No head contact');summary['horizon_completed']=True;summary['status']='NATIVE_WITNESS_RUNTIME_V2_FULL_HORIZON_REVIEW_REQUIRED'
  require(time.monotonic()<=deadline,'600second runtime budget exhausted')
 except Exception as exc:
  summary['status']=ERROR;summary['error']=str(exc);(dest/'error.txt').write_text(traceback.format_exc())
 finally:
  summary['completed_step_count_exact']=summary['steps']==summary['attempted_steps']
  summary['elapsed_simulation_s']=summary['steps']*DT;summary['head_contact_intervals']=fr.contact_intervals(flags);summary['no_head_contact_intervals']=fr.contact_intervals([not x for x in flags])
  summary['model_arrays_and_options_unchanged']=floor.model_signature(model)==signature
  if not summary['model_arrays_and_options_unchanged']:summary['status']=ERROR
  np.savez_compressed(dest/'states.npz',qpos=np.asarray(poses),qvel=np.asarray(velocities),device_time=np.asarray(clocks),time=DT*np.arange(len(poses)),dt=DT)
  save(dest/'summary.json',summary);progress(summary)
 return summary


def run_matrix(cases,run,out,report,deadline):
 require(tuple(c['name'] for c in cases)==CASE_NAMES,'Exactly six declared cases required')
 for case in cases:
  require(time.monotonic()<deadline,'Global600second budget exhausted')
  done=sum(c['steps'] for c in report['cases']);attempts=sum(c['attempted_steps'] for c in report['cases'])
  def progress(current):
   require(0<=current['steps']<=current['attempted_steps']<=MAX_STEPS and attempts+current['attempted_steps']<=TOTAL_STEPS,'Integration budget exceeded')
   save(out/'progress.json',{'completed_steps':done+current['steps'],'attempted_steps':attempts+current['attempted_steps'],'active_case':current,'new_ppo_updates':0})
  progress(fr.new_summary(case));result=run(case,progress);report['cases'].append(result);save(out/'diagnostic-review.json',report)
  if result['status'] not in (*fr.FLOOR_ENDPOINTS,'NATIVE_WITNESS_RUNTIME_V2_FULL_HORIZON_REVIEW_REQUIRED') or not result['completed_step_count_exact']:
   report['status']=ERROR;break
 else:report['status']=PASS


def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--bounded-native-witness-runtime-v2',required=True,action='store_true');args=parser.parse_args()
 out=args.output.resolve();out.mkdir(parents=True,exist_ok=False);start=time.monotonic();deadline=start+WALL_SECONDS
 report={'status':'STARTED','cases':[],'pair_controls':[],'pair_queries_completed':0,'pair_query_attempts':0,'forward_controls':[],'forward_query_attempts':0,'forward_queries_completed':0,'new_ppo_updates':0,'full_robot_steps':0,'cloud_hours':0,'stage11_status':'MATERIAL_BLOCKED_REVIEW_REQUIRED','stage11_complete':False,'formal_task_migration':False,'formal_physical_parameter_adoption':False,'policy_success':None,'installed_dependency_patch':False,'candidate_used_by_integrator_or_solver':False,'candidate_used_by_integrator':False,'prior_canonical_steps':PRIOR_STEPS,'original_material_omissions_retained':True}
 save(out/'progress.json',{'completed_steps':0,'attempted_steps':0,'active_case':None,'new_ppo_updates':0})
 model=wm=None;snapshot={};signature=wm_before=None;provenance={};deps=None;dependency_hashes={}
 try:
  current,snapshot=ref.local_preflight(out);plan=floor.load_json(ROOT/'docs/audits/stage-11-native-witness-runtime-v2-plan.json')
  for n,h in plan['frozen_files'].items():require(digest(ROOT/n)==h,'Frozen source changed: '+n)
  import warp as wp,mujoco_warp as mw
  from mujoco_warp._src import forward as fw
  deps=Path(mw.__file__).parent;dependency_hashes=plan['pinned_dependency_sources']
  for n,h in plan['pinned_dependency_sources'].items():require(digest(deps/n)==h,'Pinned dependency changed: '+n)
  save(out/'dependency-sources.json',plan['pinned_dependency_sources'])
  files=fr.pinned_files(SOURCE,SOURCE_SHA,('run-summary.json','diagnostic/preflight.json','diagnostic/diagnostic-review.json','diagnostic/cases/slide-y-plus/states.npz','diagnostic/cases/slide-y-plus/normal-candidate.jsonl.gz'))
  old=floor.load_json(SOURCE/'diagnostic/preflight.json');require(all(old[k]==current[k] for k in ('head','versions','python')),'Runtime differs from runtime-v1 evidence')
  prior=floor.load_json(SOURCE/'run-summary.json');pr=floor.load_json(SOURCE/'diagnostic/diagnostic-review.json')
  require(prior['physics_integration_steps']==prior['physics_attempted_steps']==578 and prior['canonical_total_known_completed_steps']==13932 and prior['execution_invariants_unchanged'],'Prior runtime counts/invariants changed')
  require(pr['status']=='CANDIDATE_RUNTIME_BLOCKED_REVIEW_REQUIRED' and pr['cases'][-1]['name']=='slide-y-plus' and pr['cases'][-1]['steps']==5 and pr['cases'][-1]['error']=='Candidate changed distance, witnesses or contact presence','Prior preservation block changed')
  cases,triangles,provenance=cf.source_inputs(out,current);cases=[next(c for c in cases if c['name']==n) for n in CASE_NAMES]
  provenance['latest_runtime_v1']={'directory':str(SOURCE),'zip_sha256':SOURCE_SHA,'files':files};save(out/'input-provenance.json',provenance);save(out/'selected-cases.json',cases)
  saved_files=fr.pinned_files(normal.SOURCE,normal.SOURCE_SHA,normal.INPUTS)
  provenance['saved_native_CUDA']={'directory':str(normal.SOURCE),'zip_sha256':normal.SOURCE_SHA,'files':saved_files};save(out/'input-provenance.json',provenance)
  static_files=fr.pinned_files(STATIC_SOURCE,STATIC_SOURCE_SHA,tuple('diagnostic/cases/'+n+'/input.json' for n in static.CASE_NAMES))
  provenance['prior_static_candidate']={'directory':str(STATIC_SOURCE),'zip_sha256':STATIC_SOURCE_SHA,'files':static_files};save(out/'input-provenance.json',provenance)
  states=normal.selected_states(normal.SOURCE)
  with np.load(SOURCE/'diagnostic/cases/slide-y-plus/states.npz') as z:
   require(len(z['qpos'])==6,'PositiveY failed prefix length changed')
   states.append({'name':'YPLUS5','qpos':z['qpos'][5].copy(),'qvel':z['qvel'][5].copy(),'time':float(z['device_time'][5]),'source_state_index':5,'source':'saved_runtime_v1','rounding':'none'})
  pair_cases=select_pair_cases(STATIC_SOURCE,SOURCE);save(out/'selected-pair-cases.json',pair_cases);save(out/'selected-forward-states.json',states)
  model=mujoco.MjModel.from_binary_path(str(out/'material-fixture.mjb'));model,signature=fr.configured_copy(model);contract(model);save(out/'model-signature-before.json',signature)
  runtime,kernel_module=runtime_modules();device=wp.get_device('cuda:0');cc.validate_device(device)
  with wp.ScopedDevice(device):
   wm=mw.put_model(model);wm.opt.ls_parallel=True;wm_before=normal.model_fingerprint(wm);save(out/'device-model-before.json',wm_before)
   with normal.forbid_integration(mw,fw):
    pair_controls(wm,triangles,pair_cases,out,report,deadline,kernel_module)
    require(all(r['original_fields_match_saved_CUDA'] is not False for r in report['pair_controls']),'Native static baseline mismatch; no dynamics permitted')
    saved_forward_controls(model,wm,triangles,states,out,report,deadline,wp,mw,runtime,kernel_module)
   run_matrix(cases,lambda c,p:run_case(model,wm,triangles,c,out,deadline,p,wp,mw,runtime,kernel_module),out,report,deadline)
 except Exception as exc:
  report['status']=ERROR;report['error']=str(exc);(out/'error.txt').write_text(traceback.format_exc())
 finally:
  progress=floor.load_json(out/'progress.json');done,attempts=progress['completed_steps'],progress['attempted_steps']
  report.update(diagnostic_completed_steps=done,diagnostic_attempted_steps=attempts,step_count_exact=done==attempts,canonical_total_known_completed_steps=PRIOR_STEPS+done,canonical_total_attempted_upper_bound=PRIOR_STEPS+attempts,not_run_cases=[n for n in CASE_NAMES if n not in [r['name'] for r in report['cases']]],physics_wall_seconds=time.monotonic()-start,diagnostic_source_sha256=digest(Path(__file__)))
  if model is not None and signature is not None:
   after=floor.model_signature(model);save(out/'model-signature-after.json',after);report['CPU_model_unchanged']=after==signature
  if wm is not None and wm_before is not None:
   after=normal.model_fingerprint(wm);save(out/'device-model-after.json',after);report['GPU_model_unchanged']=after==wm_before
  if provenance:
   report['input_files_unchanged']=all(digest(Path(v['directory'])/n)==meta['sha256'] for key,v in provenance.items() if isinstance(v,dict) and 'directory' in v and 'files' in v for n,meta in v['files'].items())
  if deps is not None:report['dependency_sources_unchanged']=all(digest(deps/n)==h for n,h in dependency_hashes.items())
  if snapshot:report['source_files_unchanged']=all(digest(ROOT/n)==v['sha256'] for n,v in snapshot.items())
  if any(report.get(k) is False for k in ('CPU_model_unchanged','GPU_model_unchanged','source_files_unchanged','input_files_unchanged','dependency_sources_unchanged')):report['status']=ERROR
  save(out/'diagnostic-review.json',report)
 report['candidate_used_by_integrator']=any(c.get('frame_replacements',0)>0 and c['steps']>0 for c in report['cases'])
 report['candidate_used_by_integrator_or_solver']=report['candidate_used_by_integrator_or_solver'] or report['candidate_used_by_integrator']
 report['videos']=[];report['video_errors']=[]
 if model is not None:
  # One failure/endpoint and the spin case, sampled saved states only.
  available=[c for c in report['cases'] if c['steps']>0 and c.get('primary_upload_verified')]
  names=[available[0]['name']] if available else []
  if any(c['name']=='normal-spin' for c in available):names.append('normal-spin')
  for n in names:
   try:report['videos'].append(fr.render_case(copy.deepcopy(model),out/'cases'/n,out/(n+'-candidate-saved.mp4'),next(c for c in available if c['name']==n)))
   except Exception as exc:report['video_errors'].append({'case':n,'error':str(exc)})
 report['limits']=['Fixed head material only; no unassisted policy success','Original floor2ms and sensitivity failures not waived','Candidate normal changes constraint forces only in isolated runtime; no installed engine/formal adoption','First floor endpoints are censored/blocked, not3s material passes','Uncaptured runtime differs from historical captured execution; saved native/passthrough controls check forward equivalence only']
 save(out/'diagnostic-review.json',report);print('Stage11RuntimeV2Result='+report['status'],flush=True)
 return 0 if report['status']==PASS else 2

if __name__=='__main__':raise SystemExit(main())
