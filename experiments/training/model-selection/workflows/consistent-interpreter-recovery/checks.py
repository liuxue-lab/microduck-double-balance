"""Focused offline checks for reuse, protocol disclosure and original deadline."""
import ast
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from unittest.mock import patch
import zlib

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE/'package'))
import eval_common as c
import agent
checks=[]
def ok(name,value):assert value,name;checks.append(name)
def row(i):
    return dict(seed=c.SEEDS[0],env_id=i,steps=500,stable_steps=300,longest_stable_steps=250,
        ever_5s=True,terminated=i==0,success=i!=0,terminations=['fell'] if i==0 else ['time_out'],
        gate_failure_counts=[0]*len(c.GATES),run_break_counts=[0]*len(c.GATES),value_sums=[0.]*len(c.VALUES))
with tempfile.TemporaryDirectory() as tmp:
    prior=Path(tmp)/'prior';prior.mkdir();rows=[row(i) for i in range(c.NUM_ENVS)]
    initial=dict(sha256='fixture',fields=['fixture'],scope='fixture')
    g=dict(status='EVALUATED',model='4500',seed=c.SEEDS[0],checkpoint_sha256=c.MODELS['4500'],
        policy_and_normalizer_unchanged=True,episodes=rows,summary=c.summarize(rows),initial=initial)
    path=prior/('group-4500-%d.json'%c.SEEDS[0]);path.write_text(json.dumps(g))
    (prior/('initial-seed-%d.json'%c.SEEDS[0])).write_text(json.dumps(initial))
    before=path.read_bytes()
    with patch.object(c,'PRIOR',prior):
        ok('completed first episodes are loaded without mutation',c.completed_group(c.SEEDS[0])['summary']==g['summary'] and path.read_bytes()==before)
        broken=deepcopy(g);broken['episodes'][1]['env_id']=0;path.write_text(json.dumps(broken))
        try:c.completed_group(c.SEEDS[0]);raised=False
        except RuntimeError:raised=True
        ok('duplicate retained environment identifiers rejected',raised)
        broken=deepcopy(g);broken['episodes'][0]['success']=True;path.write_text(json.dumps(broken))
        try:c.completed_group(c.SEEDS[0]);raised=False
        except RuntimeError:raised=True
        ok('retained physical failure cannot become success',raised)
        broken=deepcopy(g);broken['summary']['successes']=0;path.write_text(json.dumps(broken))
        try:c.completed_group(c.SEEDS[0]);raised=False
        except RuntimeError:raised=True
        ok('retained summaries must match original episode rows',raised)
    # Expired old budget must return existing results without writes or spawning.
    legacy=Path(tmp)/'legacy';train=Path(tmp)/'train';slot=Path(tmp)/'recovery'
    for p in (legacy,train,slot):p.mkdir()
    (prior/'started.json').write_text(json.dumps(dict(started_utc=0,deadline_utc=1)))
    with patch.multiple(agent,LEGACY=legacy,TRAIN=train,PRIOR=prior,SLOT=slot),patch.object(agent,'preflight',return_value={}),patch.object(agent,'snapshot',return_value={'status':'NOT_STARTED'}),patch.object(agent,'write',side_effect=AssertionError('must not write')),patch.object(agent.subprocess,'Popen',side_effect=AssertionError('must not launch')):
        ok('expired original deadline forbids launch and any new budget',agent.launch()['status']=='ORIGINAL_EVALUATION_DEADLINE_EXPIRED')
    (slot/'started.json').write_text('{}')
    with patch.multiple(agent,LEGACY=legacy,TRAIN=train,PRIOR=prior,SLOT=slot),patch.object(agent,'preflight',side_effect=AssertionError('must not rerun')),patch.object(agent,'snapshot',return_value={'status':'EXISTING'}):
        ok('repeated recovery attaches without rerunning groups',agent.launch()['status']=='EXISTING')

old=ast.parse((HERE.parent/'candidate-evaluation/package/evaluate.py').read_text())
new=ast.parse((HERE/'package/evaluate.py').read_text())
old_observer=next(x for x in old.body if isinstance(x,ast.ClassDef) and x.name=='Observer')
new_observer=next(x for x in new.body if isinstance(x,ast.ClassDef) and x.name=='Observer')
ok('metric observer and all success thresholds are unchanged',ast.dump(old_observer)==ast.dump(new_observer))
calls=[x for x in ast.walk(new) if isinstance(x,ast.Call)]
groups=[x for x in calls if isinstance(x.func,ast.Name) and x.func.id=='evaluate_group']
ok('new simulation only evaluates the 6000 candidate',len(groups)==1 and groups[0].args[0].value=='6000')
ok('no paired-result calculation remains in recovery',not any(isinstance(x.func,ast.Name) and x.func.id=='paired_comparison' for x in calls))
src=(HERE/'package/evaluate.py').read_text()
ok('exact-pairing failure remains explicit in returned results',"paired_comparison_allowed=False" in src and 'NOT PAIRED' in src and 'UNKNOWN' in src)
ok('deterministic inference and zero PPO remain', 'stochastic_output=False' in src and not any(isinstance(x.func,ast.Attribute) and x.func.attr in ('learn','backward','act','compute_returns') for x in calls))
source=(HERE/'package/agent.py').read_text()
ok('start inherits the prior deadline, never starts a fresh 30 minutes',"deadline_utc=previous['deadline_utc']" in source and 'now+MAX_SECONDS' not in source)
ok('old evaluation directory participates in inherited GPU locks', '(LEGACY,TRAIN,PRIOR,RECOVERY_V1,SLOT)' in source)

controller=ROOT/'evaluate_remaining.py';ns={'__name__':'offline'}
exec(compile(controller.read_bytes(),'<recovery-client>','exec'),ns)
payload=ns['decode'](ns['PAYLOAD_B85'],ns['PAYLOAD_SHA'],False)
files=json.loads(zlib.decompress(payload))['files']
ok('embedded recovery files equal reviewed files',all(base64.b64decode(v)==(HERE/'package'/k).read_bytes() for k,v in files.items()))
for path in [*list((HERE/'package').glob('*.py')),HERE/'bootstrap.py',controller]:ast.parse(path.read_text())
ok('all executable sources parse',True)
ok('old evaluation controller remains immutable',hashlib.sha256((ROOT/'evaluate_candidates.py').read_bytes()).hexdigest()=='c472eacde86c8dbb6e7779e3dba81e72c77ba89fea2daa213f5821025977c671')
ok('old training controller remains immutable',hashlib.sha256((ROOT/'train.py').read_bytes()).hexdigest()=='5b57ef3a761757b9d2bd2c39e7faaa6713b1f4e29b54500f5e1ded13f33438cb')
receipt=dict(status='PASS',checks=checks,count=len(checks),scope='Offline standard-library recovery checks only; no Torch/CUDA/physics run.',
    reused_prior_observer_checks=28,additional_PPO_updates=0,additional_simulation_steps=0,cloud_connections=0)
(HERE/'offline-review.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='PASS',checks=len(checks),simulation_steps=0,PPO=0)))
