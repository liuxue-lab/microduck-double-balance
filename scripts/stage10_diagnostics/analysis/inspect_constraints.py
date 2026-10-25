from pathlib import Path
import sys,json
from itertools import combinations
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'stage10-delivery/microduck-stage10-initialization-recovery'))
from stage10_trace import rows,TraceReader
from stage10_core import write
base=ROOT/'stage10-initialization-c-review';jobs=['01-initialization/attempt-772ee431','02-initialization/attempt-8181eb75','03-initialization/attempt-41fdfd96'];allruns=[]
for job in jobs:
 reader=TraceReader(base/job/'trace');events=rows(base/job/'trace');summary=[]
 for phase in ('bam.compute#1.before','bam.compute#1.after','sim.forward#2.after'):
  f=next(e['fields'] for e in events if e['phase']==phase)
  def arr(k):return reader.array(f[k])
  nefc=arr('physical/nefc');types=arr('workspace/efc/type');ids=arr('workspace/efc/id');forces=arr('workspace/efc/force')
  active=np.arange(types.shape[1])[None,:]<nefc[:,None]
  assert np.all(nefc==14) and np.all(arr('physical/nf')==14) and not arr('physical/ne').any() and not arr('physical/nl').any()
  assert f['model/cpu/ntendon']==0 and f['contact/count']==0
  dofs=arr('bam/_dof_ids')
  for env in range(128):assert np.array_equal(np.sort(ids[env,active[env]]),np.sort(dofs))
  assert np.unique(types[active]).tolist()==[1]
  changed=[]
  if phase=='bam.compute#1.after':
   before=next(e['fields'] for e in events if e['phase']=='bam.compute#1.before')
   x=reader.array(before['model/wp/dof_frictionloss']);y=arr('model/wp/dof_frictionloss')
   bad=np.argwhere(x!=y);changed=[{'env':int(i),'dof':int(j),'before':float(x[i,j]),'after':float(y[i,j]),'delta':float(y[i,j]-x[i,j])} for i,j in bad]
  summary.append({'phase':phase,'nefc_each':14,'nf_each':14,'ne_each':0,'nl_each':0,'aggregate_contact_count':0,
  'active_type_codes':[1],'active_ids_match_14_bam_dofs':True,'each_active_dof_occurs_once':True,'runtime_friction_changes_this_call':changed})
 allruns.append({'job':job,'checks':summary});reader.close()
write(Path(__file__).parent/'constraint-and-bam-audit.json',{'runs':allruns,
 'source_interpretation':'The captured Data source describes nf as friction constraints; ntendon is zero; valid rows match the 14 BAM DOFs. Padding is excluded.',
 'bam_friction_budget_recomputed':False,'reason':'Captured direct BAM state does not include all parameter objects; no exact CPU emulation is claimed.',
 'root_cause_not_proved':True})
for r in allruns:
 print(r['job'],[(s['phase'],len(s['runtime_friction_changes_this_call'])) for s in r['checks']])
 print(r['checks'][1]['runtime_friction_changes_this_call'][:8])
