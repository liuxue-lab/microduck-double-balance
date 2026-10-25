"""Independent review of returned initialization evidence. No simulator imports."""
from pathlib import Path
from itertools import combinations
from collections import Counter
import hashlib,io,json,zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'stage10-initialization-c-review';OUT=Path(__file__).resolve().parent
JOBS=['01-initialization/attempt-772ee431','02-initialization/attempt-8181eb75','03-initialization/attempt-41fdfd96']
INPUTS=('time','qpos','qvel','act','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','eq_active','mocap_pos','mocap_quat')
EXPECTED={'raw.reset':1,'raw._reset_idx':1,'sim.reset':1,'scene.reset':1,'event.apply':1,'sim.forward':2,'scene.write_data_to_sim':1,'bam.compute':1,'command.compute':1,'sim.sense':1,'observation.compute':2,'policy.reset':1}
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def inventory(folder,rows):
 for row in rows:
  rel=Path(row['path']);assert not rel.is_absolute() and '..' not in rel.parts
  p=folder/rel;assert not p.is_symlink() and p.stat().st_size==row['size_bytes'] and sha(p)==row['sha256']
manifest=read(SRC/'review-files.json');inventory(SRC,manifest)
reference=read(SRC/'initialization-recovery-v1/tool/recovery-reference.json')
inventory(SRC,reference['original_inventory'])
local=ROOT/'stage10-delivery/microduck-stage10-initialization-recovery'
for row in read(local/'manifest.json'):
 assert sha(local/row['path'])==sha(SRC/'initialization-recovery-v1/tool'/row['path'])==row['sha256']
blobs={};traces=[];reports=[];perrun=[]
for name in JOBS:
 p=SRC/name;r=read(p/'result.json');reports.append(r)
 e=[json.loads(line) for line in (p/'trace/events.jsonl').read_text().splitlines()];traces.append(e)
 assert len(e)==33 and all(row['index']==i and row['control']==row['substep']==0 for i,row in enumerate(e))
 assert r['boundary_counts']==r['observed_boundary_counts']==EXPECTED
 assert read(p/'boundary-counts.json')=={'counts':EXPECTED,'stack':[]}
 assert r['new_ppo_updates']==r['rollout_physics_steps']==r['policy_inference_calls_after_load']==0
 assert read(p/'runtime-check.json')['status']=='MATCH'
 assert sha(p/'params/env.yaml')==r['task_config_sha256']
 assert sha(p/'dependency-sources.json')==r['dependency_sources_sha256']
 boot=read(p/'bootstrap-calls.json');assert boot['rollout_calls']==0
 if name!=JOBS[0]:
  assert r['status']=='INITIALIZATION_TRACE_COMPLETE_REVIEW_REQUIRED'
  assert r['backend_before']==r['backend_after'] and r['source_recheck_after']=='PASS_101_FROZEN_FILES'
  assert all(r[k] for k in ('model_parameters_unchanged','model_buffers_unchanged','adam_state_unchanged'))
  assert r['initialization_assistance_check']['status']=='PASS_UNPROCESSED_INITIALIZATION_ONLY'
  inventory(p,read(p.parent/'completed.json')['files'])
  assert r['backend_reader_after']['cached_values']==False
  assert r['backend_reader_after']['settings_written']==False
 else:
  assert r['status']=='FAILED_REVIEW_REQUIRED' and 'backend_after' not in r
  assert not (p.parent/'completed.json').exists()
 unique={};refs=0;archives={};rawbytes=npybytes=0
 for archive in sorted((p/'trace').glob('arrays-*.zip')):
  with zipfile.ZipFile(archive) as z:
   assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))
   for entry in z.namelist():
    data=z.read(entry);a=np.load(io.BytesIO(data),allow_pickle=False)
    digest=hashlib.sha256(json.dumps([a.dtype.str,list(a.shape)],separators=(',',':')).encode()+b'\0'+a.tobytes()).hexdigest()
    assert entry==digest+'.npy' and digest not in unique
    unique[digest]=(archive.name,entry);blobs[digest]=a;rawbytes+=a.nbytes;npybytes+=len(data)
 refs_used=set();nonfinite=Counter();zeros=Counter();nefc=set();contacts=set()
 for row in e:
  f=row['fields']
  for key,d in f.items():
   if not isinstance(d,dict) or 'blob' not in d:continue
   refs+=1;refs_used.add(d['blob']);a=blobs[d['blob']]
   assert unique[d['blob']]==(d['file'],d['entry']) and a.dtype.str==d['dtype'] and list(a.shape)==d['shape']
   if a.dtype.kind in 'buif' and not np.isfinite(a).all():nonfinite[key]+=int((~np.isfinite(a)).sum())
  assert f['hold_action/_force'] is None
  assert all('hold_action/'+k not in f for k in ('_torque','_duck_force','_duck_torque'))
  for key in ('context/assistance_hold','physical/xfrc_applied','physical/qfrc_applied','physical/time'):
   a=blobs[f[key]['blob']];assert np.isfinite(a).all() and not a.any();zeros[key]+=1
  nefc.update(np.unique(blobs[f['physical/nefc']['blob']]).tolist());contacts.add(f['contact/count'])
 assert refs_used==set(unique)
 assert not any(k.startswith(('physical/','observations_returned/','returned_torque')) for k in nonfinite)
 final=e[-1]['fields'];assert final['context/common_step_counter']==final['context/sim_step_counter']==0
 assert not blobs[final['context/episode_length_buf']['blob']].any()
 perrun.append({'job':name,'events':len(e),'array_references':refs,'unique_arrays':len(unique),
 'raw_array_bytes':rawbytes,'npy_array_bytes':npybytes,'nonfinite_fields_and_counts':dict(nonfinite),
 'zero_fields_at_all_events':dict(zeros),'recorded_nefc_values':sorted(nefc),'contact_counts':sorted(contacts),
 'bootstrap_calls':boot,'original_status':r['status'],'postchecks_complete':name!=JOBS[0]})
 if name!=JOBS[0]:assert r['unique_trace_arrays']==len(unique) and r['raw_unique_array_bytes']==npybytes
for r in reports[1:]:
 for k in ('scope','runtime_identity','model_loaded','source_head','checkpoint_sha256','dataset_sha256','task_config_sha256','dependency_sources_sha256','boundary_counts'):
  assert r[k]==reports[0][k]
def differing(a,b):
 if isinstance(a,dict) and isinstance(b,dict) and 'blob' in a and 'blob' in b:
  return a['blob']!=b['blob'] or a['env_axis']!=b['env_axis']
 return a!=b
def detail(da,db):
 if not isinstance(da,dict) or 'blob' not in da or not isinstance(db,dict) or 'blob' not in db:return {'metadata_changed':True}
 a,b=blobs[da['blob']],blobs[db['blob']]
 if a.shape!=b.shape or a.dtype!=b.dtype:return {'layout_changed':True}
 d=np.abs(a.astype('f8')-b.astype('f8'));finite=np.isfinite(a)&np.isfinite(b)
 out={'max_abs':float(d[finite].max()) if finite.any() else None,'numeric_changed_elements':int(np.not_equal(a,b).sum()),'shape':list(a.shape),'dtype':a.dtype.str}
 if a.ndim and a.shape[0]==128:
  envs=np.flatnonzero(np.any(a!=b,axis=tuple(range(1,a.ndim))) if a.ndim>1 else a!=b).tolist();out['env_ids']=envs
 return out
pairs=[];returned=read(SRC/'initialization-recovery-v1/comparison.json')
for pairnum,(i,j) in enumerate(combinations(range(3),2)):
 first={};phases=[];boundaries=[]
 for a,b in zip(traces[i],traces[j],strict=True):
  assert {k:a[k] for k in ('index','phase','control','substep')}=={k:b[k] for k in ('index','phase','control','substep')}
  assert a['fields'].keys()==b['fields'].keys()
  changed=[k for k in sorted(a['fields']) if differing(a['fields'][k],b['fields'][k])]
  for k in changed:
   if k not in first:first[k]={'index':a['index'],'phase':a['phase'],**detail(a['fields'][k],b['fields'][k])}
  phases.append({'index':a['index'],'phase':a['phase'],'changed_fields':changed})
  if a['phase'] in ('sim.forward#1.before','sim.forward#1.after','sim.forward#2.before','sim.forward#2.after','bam.compute#1.before','bam.compute#1.after','reset_ready_after_original_observer_setup'):
   selected=['physical/qacc','physical/qfrc_bias','physical/qfrc_constraint','physical/qM','physical/subtree_com','model/wp/dof_frictionloss','workspace/efc/force','returned_torque']
   boundaries.append({'phase':a['phase'],'different_inputs':[k for k in INPUTS if 'physical/'+k in changed],
    'different_model_fields':[k for k in changed if k.startswith('model/')],
    'different_derived_fields':[k for k in changed if k.startswith('physical/') and k[9:] not in INPUTS],
    'details':{k:detail(a['fields'][k],b['fields'][k]) for k in selected if k in changed}})
 expected=returned['pairs'][pairnum]['first_difference_by_field']
 assert set(first)==set(expected)
 for k,v in first.items():
  assert v['index']==expected[k]['index'] and v['phase']==expected[k]['phase']
  if 'max_abs' in v:assert v['max_abs']==expected[k]['max_abs_finite']
 pairs.append({'left':JOBS[i],'right':JOBS[j],'first_differences':first,'phases':phases,'boundaries':boundaries,
 'all_recorded_integrator_inputs_equal_at_all_events':not any('physical/'+k in first for k in INPUTS),
 'captured_observations_equal':not any(k.startswith('observations_returned/') for k in first),
 'captured_rng_equal':not any(k.startswith('rng/') for k in first),
 'captured_bam_state_and_arguments_equal':not any(k.startswith(('bam/','call_args/')) for k in first),
 'contact_prefixes_equal':not any(k.startswith('contact/') for k in first)})
summary={'status':'INITIALIZATION_C_REVIEWED_WITH_FIRST_RUN_POSTCHECK_GAP','manifest_files':len(manifest),
 'original_files_retained':len(reference['original_inventory']),'total_events':sum(x['events'] for x in perrun),
 'sum_run_unique_arrays':sum(x['unique_arrays'] for x in perrun),'global_unique_arrays':len(blobs),
 'returned_comparison_independently_reproduced':True,'per_run':perrun,'pairs':pairs,
 'new_ppo_updates':0,'rollout_physics_steps':0,'stage10_complete':False,'root_kernel_identified':False}
write(OUT/'independent-review.json',summary)
for p in pairs:
 print('PAIR',p['left'].split('/')[0],p['right'].split('/')[0],{k:v for k,v in p.items() if k.startswith(('all_','captured_','contact_'))})
 for b in p['boundaries']:
  print(b['phase'],'inputs',b['different_inputs'],'model',b['different_model_fields'],'maxima',{k:v['max_abs'] for k,v in b['details'].items()})
print('SUMMARY',json.dumps({k:v for k,v in summary.items() if k not in ('pairs','per_run')},ensure_ascii=False))
print('PER-RUN',json.dumps(perrun,ensure_ascii=False))
