"""CPU arithmetic-only counterfactual; not a simulation or root-cause proof."""
from pathlib import Path
from itertools import permutations,product,combinations
import sys,json
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'stage10-delivery/microduck-stage10-initialization-recovery'))
from stage10_trace import TraceReader,rows
from stage10_core import write
base=ROOT/'stage10-initialization-c-review'
jobs=['01-initialization/attempt-772ee431','02-initialization/attempt-8181eb75','03-initialization/attempt-41fdfd96']
phases=['sim.forward#1.after','sim.forward#2.after'];data={};readers=[]
for job in jobs:
 reader=TraceReader(base/job/'trace');readers.append(reader)
 es=rows(base/job/'trace')
 data[job]={}
 for phase in phases:
  f=next(e['fields'] for e in es if e['phase']==phase)
  data[job][phase]={k:reader.array(f[k]) for k in ('model/wp/body_parentid','model/wp/body_mass','model/wp/body_subtreemass','physical/xipos','physical/subtree_com')}
results=[]
for phase in phases:
 first=data[jobs[0]][phase]
 for j in jobs[1:]:
  for k in first:
   if k!='physical/subtree_com':assert np.array_equal(first[k],data[j][phase][k])
 par=first['model/wp/body_parentid'];mass=first['model/wp/body_mass'][0];submass=first['model/wp/body_subtreemass'][0];xipos=first['physical/xipos']
 children={b:np.flatnonzero((par==b)&(np.arange(len(par))!=b)).tolist() for b in range(len(par))}
 observed=np.stack([data[j][phase]['physical/subtree_com'] for j in jobs])
 diff=np.any(observed!=observed[0:1],axis=0)
 changed=np.argwhere(diff);matches=[]
 for env,body,axis in changed:
  # Scalar component enumeration allows independent component atomic order.
  def sums(b):
   acc0=np.float32(xipos[env,b,axis]*mass[b]);ch=children[b]
   if not ch:return {acc0.tobytes():acc0}
   values=[list(sums(c).values()) for c in ch];out={}
   for vals in product(*values):
    for order in permutations(range(len(ch))):
     v=acc0
     for i in order:v=np.float32(v+vals[i])
     out[v.tobytes()]=v
   assert len(out)<1000
   return out
  candidates=[np.float32(x/submass[body]) if submass[body]!=0 else x for x in sums(int(body)).values()]
  bits={x.tobytes() for x in candidates}
  targets=observed[:,env,body,axis]
  matches.append({'env':int(env),'body':int(body),'axis':int(axis),'observed':[float(x) for x in targets],
    'candidate_values':sorted(set(float(x) for x in candidates)),
    'each_observed_in_candidate_set':[x.tobytes() in bits for x in targets]})
 results.append({'phase':phase,'cross_run_changed_components':len(matches),
                 'all_changed_observations_explainable_by_scalar_float32_order':all(all(x['each_observed_in_candidate_set']) for x in matches),'components':matches})
for r in readers:r.close()
report={'status':'OFFLINE_ARITHMETIC_PROBE','new_simulation_calls':0,'new_ppo_updates':0,'results':results,
 'interpretation':'Membership demonstrates arithmetic compatibility with atomic sibling-addition ordering, not the actual GPU schedule or a causal proof.',
 'limitations':['Only differing subtree_com scalar components at two recorded post-forward boundaries are considered.',
 'Scalar orders are enumerated independently; this is not a reconstruction of one joint vector/kernel schedule.',
 'The check does not isolate crb/solver, reproduce qacc, or explain 10-second labels.']}
write(Path(__file__).parent/'roundoff-probe.json',report)
for x in results:print(x['phase'],x['cross_run_changed_components'],x['all_changed_observations_explainable_by_scalar_float32_order'])
