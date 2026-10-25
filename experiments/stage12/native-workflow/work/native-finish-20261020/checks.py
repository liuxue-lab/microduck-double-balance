"""Focused read-only collector checks and CPU-only metric counter reproduction."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as N
from unittest.mock import patch
import numpy as np
import finish_read as f

ROOT=Path(__file__).resolve().parent
passed=[]
def ok(label,value):
    assert value,label
    passed.append(label)

# Exercise actual geometric_success and upstream MetricsManager compute/reset
# with NumPy-backed arrays. No environment construction or numerical physics.
class A(np.ndarray):
    def float(self):return self.astype(np.float32)
def array(value,dtype=None):return np.asarray(value,dtype=dtype).view(A)
torch=N(where=np.where,zeros_like=np.zeros_like,mean=np.mean,
        clamp=lambda x,min:np.maximum(x,min))
ns={'torch':torch,'geometric_joint_stable':lambda env:env.mask}
native=ast.parse((ROOT.parent/'native-training-v2/package/native_task.py').read_text())
success=next(n for n in native.body if isinstance(n,ast.FunctionDef) and n.name=='geometric_success')
exec(compile(ast.Module(body=[success],type_ignores=[]),'<real-success-counter>','exec'),ns)
upstream=ast.parse((ROOT/'reference/mjlab-v1.3.0-metrics_manager.py').read_text())
cls=next(n for n in upstream.body if isinstance(n,ast.ClassDef) and n.name=='MetricsManager')
methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('compute','reset')]
for node in methods:
    node.returns=None
    for arg in node.args.args:arg.annotation=None
exec(compile(ast.Module(body=methods,type_ignores=[]),'<upstream-metrics>','exec'),ns)
env=N(mask=array([True,True],bool),_s12_stable_ticks=array([0,0],np.int64),_s12_success_ever=array([False,False],bool))
manager=N(_env=env,_step_count=array([0,0],np.int64),_substep_term_indices=[],_step_term_indices=[0],
          _term_names=['success'],_term_cfgs=[N(func=ns['geometric_success'],params={},reduce='last')],
          _step_values=array([[0.],[0.]],np.float32),_episode_sums={'success':array([0.,0.],np.float32)},
          _substep_accum=[],_class_term_cfgs=[])
for _ in range(249):ns['compute'](manager)
ok('249 steps not enough',not env._s12_success_ever.any())
env.mask[1]=False
ns['compute'](manager)
ok('250th step passes only uninterrupted environment',np.array_equal(env._s12_success_ever,[True,False]))
ok('single instability resets consecutive counter',env._s12_stable_ticks[1]==0)
# Match reset event order: clear private state BEFORE metrics_manager.reset.
env._s12_success_ever[0]=False;env._s12_stable_ticks[0]=0
value=ns['reset'](manager,array([0],np.int64))
ok('manager snapshot survives event clearing private success',value['Episode_Metrics/success']==1.)
env.mask[:]=True
for _ in range(250):ns['compute'](manager)
env.mask[:]=False;ns['compute'](manager)
ok('episode success latches after achieving 250 stable steps',env._s12_success_ever.all())

with tempfile.TemporaryDirectory() as tmp:
    slot=Path(tmp)/'campaign';(slot/'run/checkpoints').mkdir(parents=True)
    def put(name,value):
        (slot/name).write_text(json.dumps(value)+'\n')
    dead=dict(pid=999999999,start='0')
    put('started.json',dict(package_identity=f.EXPECTED_PACKAGE))
    put('supervisor.json',dead);put('child.json',dead)
    put('budget.json',dict(completed_updates=300,attempted_updates=300,max_updates=300,pending=False,deadline_utc=123))
    put('run/progress.json',dict(status='TRAINING_BUDGET_COMPLETE',completed_updates=300))
    put('run/smoke-result.json',dict(status='SMOKE_PASSED',completed_updates=5))
    model=slot/'run/checkpoints/update_000300.pt';model.write_bytes(b'fixture model only, never deserialized')
    receipt=dict(path=str(model),bytes=model.stat().st_size,sha256=hashlib.sha256(model.read_bytes()).hexdigest(),completed_updates=300)
    put('run/latest.json',receipt);put('run/best-training-metric.json',dict(checkpoint=receipt,score=[0.,.8,.9]))
    rows=[]
    for i in range(6,301):
        rows.append(dict(campaign_phase='formal',completed_updates=i,session=1,elapsed_seconds=i*2.4,
                         joint_geometric_current_fraction=.8,lower_unassisted_current_fraction=.9,
                         episode_metrics={'Episode_Metrics/native_joint_geometric_success_5s':.2 if i>=275 else 0}))
    (slot/'run/metrics.jsonl').write_text('\n'.join(json.dumps(x) for x in rows)+'\n')
    def hashes():return {str(p.relative_to(slot)):hashlib.sha256(p.read_bytes()).hexdigest() for p in slot.rglob('*') if p.is_file()}
    before=hashes()
    result=f.collect_finished(slot)
    ok('dead workers allow finished read',result['status']=='FINISHED_REPORT_READY')
    ok('all source bytes preserved',hashes()==before)
    ok('both model receipts verified',all(x['verified'] for x in result['checkpoints'].values()))
    ok('history grouped and positive successes retained',len(result['history']['windows_250_updates'])==2 and result['history']['whole_run']['positive_success_logged_updates']==26)
    ok('collector does not claim evaluation',not result['summary']['independent_policy_evaluation_done'] and not result['summary']['training_success_claimed'])
    ok('completed run requires exhausted declared budget',result['summary']['finished_training_run_confirmed'])
    put('budget.json',dict(completed_updates=300,attempted_updates=300,max_updates=6000,pending=False,deadline_utc=123))
    ok('early complete label does not falsely confirm exhausted run',not f.collect_finished(slot)['summary']['finished_training_run_confirmed'])
    # A running or uncertain worker must prevent hashing and history scans.
    live={'status':'RUNNING','files':{},'processes_alive':{'child':True}}
    with patch.object(f,'snapshot',return_value=live),patch.object(f,'verify_checkpoint',side_effect=AssertionError('must not hash live model')):
        ok('active task collection only returns snapshot',f.collect_finished(slot) is live)
    uncertain=dict(live,status='PROCESS_STATE_UNKNOWN')
    with patch.object(f,'snapshot',return_value=uncertain),patch.object(f,'metric_history',side_effect=AssertionError('must not scan uncertain task')):
        ok('unknown task preserved',f.collect_finished(slot) is uncertain)
    bad=dict(receipt,sha256='0'*64)
    ok('corrupted model receipt rejected',f.verify_checkpoint(slot,bad)['verified'] is False)
    outside=Path(tmp)/'outside.pt';outside.write_bytes(b'outside')
    ok('outside checkpoint path rejected',not f.verify_checkpoint(slot,dict(receipt,path=str(outside)))['verified'])
    link=slot/'run/checkpoints/link.pt';link.symlink_to(model)
    ok('symlink model rejected',not f.verify_checkpoint(slot,dict(receipt,path=str(link)))['verified'])
    # This executor does not expose its own /proc PID. Test parser and identity
    # behavior with a Linux-format record, without claiming a local live PID test.
    fields=['R']+['0']*18+['12345']
    with patch.object(f.Path,'read_text',return_value='321 (fixture python) '+' '.join(fields)):
        ok('live process identity parser checked with fixture',f.process_alive(dict(pid=321,start='12345')) is True)
        ok('recycled PID identity rejected with fixture',f.process_alive(dict(pid=321,start='wrong')) is False)
    ok('missing process pin is unknown',f.process_alive(None) is None)

client={'__name__':'offline_test'}
exec(compile((ROOT.parent.parent/'microduck-stage12-finish-readonly-20261020.py').read_bytes(),'<client>','exec'),client)
sequence=iter([dict(status='RUNNING',files={}),dict(status='STOPPED',files={}),result])
calls=[];reports=[]
def request(ssh,operation):calls.append(operation);return next(sequence)
client['request']=request;client['save_report']=lambda value:reports.append(value)
with patch.object(client['time'],'sleep') as sleep,redirect_stdout(io.StringIO()):
    client['run'](True,{})
ok('watch only polls then collects once',calls==['status','status','collect'] and reports==[result] and sleep.call_count==1)
calls.clear();reports.clear()
client['request']=lambda ssh,operation:(calls.append(operation) or dict(status='RUNNING',files={}))
with redirect_stdout(io.StringIO()):client['run'](False,{})
ok('collect during training returns immediate read-only status',calls==['status'] and len(reports)==1)
ok('original running controller unchanged',hashlib.sha256((ROOT.parent.parent/'microduck-stage12-native-train-v2-20261020.py').read_bytes()).hexdigest()=='5b57ef3a761757b9d2bd2c39e7faaa6713b1f4e29b54500f5e1ded13f33438cb')
review=dict(status='PASS',checks=passed,scope='Offline logical checks only; no Torch, GPU, simulator, evaluation or PPO',
            metric_counter_reset_bug_found=False,cloud_changes=0,PPO_updates=0,simulation_steps=0)
(ROOT/'offline-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='PASS',checks=len(passed),metric_counter_reset_bug_found=False,cloud_changes=0,PPO_updates=0,simulation_steps=0)))
