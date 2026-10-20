"""Offline control-flow fixtures; no Torch, CUDA, simulation, SSH or PPO."""
import ast
import base64
from contextlib import ExitStack,redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
from unittest.mock import patch
import zlib

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;PKG=HERE/'package'
sys.path.insert(0,str(PKG))
import common as c
import agent as a
import evaluate as ev
import eval_metrics as m
checks=[]
def ok(label,value):
    assert value,label
    checks.append(label)
def rejects(label,fn,contains=''):
    try:fn()
    except (RuntimeError,FileExistsError) as exc:ok(label,contains in str(exc))
    else:raise AssertionError(label+' did not reject')
def ast_object(path,name):
    tree=ast.parse(Path(path).read_text())
    return ast.dump(next(n for n in tree.body if getattr(n,'name',None)==name),include_attributes=False)

# Pinned physical task, metric decisions, normalization hook and predicates.
ok('native task remains byte-identical', (PKG/'native_task.py').read_bytes()==(HERE.parent/'native-training-v2/package/native_task.py').read_bytes())
for name in ('Observer','initial_fingerprint','fingerprint'):
    ok('unchanged evaluation '+name,ast_object(PKG/'eval_metrics.py',name)==ast_object(HERE.parent/'native-evaluation-recovery-v2-20261020/package/evaluate.py',name))
for name in ('wilson','summarize','summary_audit'):
    ok('unchanged aggregate '+name,ast_object(PKG/'eval_metrics.py',name)==ast_object(HERE.parent/'native-evaluation-recovery-v2-20261020/package/eval_common.py',name))
for name in ('configure_normalizer_updates','equal_state'):
    ok('unchanged recovery '+name,ast_object(PKG/'normalization.py',name)==ast_object(HERE.parent/'native-training-v2/package/transfer.py',name))
for name in ('require_finite_group','require_step_outputs','host_scalar_means'):
    ok('reviewed performance predicate '+name,ast_object(PKG/'runtime_helpers.py',name)==ast_object(HERE.parent/'native-performance-20261020/candidate/train.py',name))
ok('success thresholds unchanged',m.THRESHOLDS==dict(top_center_m=.012,top_height_error_m=.006,top_speed_m_s=.08,lower_offset_m=.06,
    lower_height_min_m=.20,lower_height_max_m=.29,robot_tilt_deg=20.,lower_ball_speed_m_s=.15,hold=1e-6,consecutive_steps=250,control_dt=.02,horizon_seconds=10.))

b=dict(deadline_utc=10,attempted_updates=0,completed_updates=0,pending=False)
c.reserve(b,1);ok('reserve before PPO consumes budget',b['attempted_updates']==1 and b['completed_updates']==0 and b['pending'])
rejects('pending attempt cannot reserve again',lambda:c.reserve(b,1),'pending')
c.commit(b);ok('commit consumes existing reservation',b['attempted_updates']==b['completed_updates']==1 and not b['pending'])
rejects('unreserved commit rejected',lambda:c.commit(b),'without reservation')
rejects('deadline cannot be extended by reserve',lambda:c.reserve(b,10),'deadline')
b.update(attempted_updates=1000);rejects('maximum spent attempts enforced',lambda:c.reserve(b,1),'exhausted')

def cand(segment,n):return dict(segment=segment,model=str(4500+segment*250),summary=dict(episodes=256,successes=n,success_fraction=n/256),checkpoint=dict(completed_updates=segment*250))
s=dict(best=cand(0,212),history=[],no_improvement=0,early_stop=False)
c.choose(s,cand(1,212));ok('tie keeps earlier candidate',s['best']['segment']==0 and s['no_improvement']==1)
c.choose(s,cand(2,213));ok('improvement replaces and clears patience',s['best']['segment']==2 and s['no_improvement']==0)
c.choose(s,cand(3,210));c.choose(s,cand(4,213));ok('two non-improvements stop',s['early_stop'] and s['best']['segment']==2)
rejects('duplicate selection rejected',lambda:c.choose(s,cand(4,220)),'order')
bad=cand(5,220);bad['summary']['episodes']=255;rejects('incomplete evaluation rejected',lambda:c.choose(s,bad),'incomplete')

def rows(seed):
    return [dict(env_id=i,seed=seed,steps=500,stable_steps=500,longest_stable_steps=500,ever_5s=True,terminated=False,
        success=True,terminations=['time_out'],gate_failure_counts=[0]*9,run_break_counts=[0]*9,value_sums=[0.]*8) for i in range(128)]

with tempfile.TemporaryDirectory() as temp:
    folder=Path(temp);checkpoint=dict(sha256='fake')
    ok('unstarted group available',ev.retained_group(folder,c.SEEDS[0],checkpoint) is None)
    group=dict(status='EVALUATED',seed=c.SEEDS[0],checkpoint_sha256='fake',policy_and_normalizer_unchanged=True,episodes=rows(c.SEEDS[0]))
    group['summary']=m.summarize(group['episodes']);c.write(folder/('group-%d.json'%c.SEEDS[0]),group)
    with patch.object(ev,'evaluate_group',wraps=ev.evaluate_group):
        reused=ev.evaluate_group(folder,checkpoint,c.SEEDS[0],0,lambda:(_ for _ in ()).throw(AssertionError('must not initialize')),lambda **kw:None)
    ok('completed group reused without runtime or sampling',reused==group)
    c.write(folder/('group-%d-started.json'%c.SEEDS[1]),{})
    rejects('interrupted group never resampled',lambda:ev.retained_group(folder,c.SEEDS[1],checkpoint),'interrupted')
    group['summary']['successes']-=1;c.write(folder/('group-%d.json'%c.SEEDS[0]),group)
    rejects('modified aggregate rejected',lambda:ev.retained_group(folder,c.SEEDS[0],checkpoint),'aggregate')

def fixture(root):
    for name in ('run/checkpoints','evaluations','logs','cache','package'):(root/name).mkdir(parents=True,exist_ok=True)
    c.write(root/'package/manifest.json',dict(identity='fixture'))
    c.write(root/'started.json',dict(started_utc=100,deadline_utc=time.time()+7200,package_identity='fixture'))
    c.write(root/'budget.json',dict(started_utc=100,deadline_utc=time.time()+7200,attempted_updates=0,completed_updates=0,pending=False,interrupted_attempts=0))
    c.write(root/'session.json',dict(number=1,requested_utc=time.time()))
    c.write(root/'selection.json',dict(best=cand(0,212),history=[],no_improvement=0,early_stop=False))
    c.write(root/'original-files.json',{'original':'unchanged'})
    c.write(root/'progress.json',dict(status='STARTING'))

def run_campaign(scores,train_code=0,pending_attempt=False):
    with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
        root=Path(temp);fixture(root)
        for module in (c,a,ev):stack.enter_context(patch.object(module,'SLOT',root))
        stack.enter_context(patch.object(a,'HERE',root/'package'));stack.enter_context(patch.object(a,'alive',return_value=False))
        stack.enter_context(patch.object(a,'protected_files',return_value={'original':'unchanged'}))
        calls=[]
        def worker(kind,segment,*rest):
            calls.append((kind,segment));budget=c.read(root/'budget.json')
            if kind=='train':
                if train_code:return train_code
                amount=min(250*segment-budget['completed_updates'],1000-budget['attempted_updates'])
                for _ in range(amount):c.reserve(budget,time.time());c.commit(budget)
                c.write(root/'budget.json',budget)
                path=root/'run/checkpoints'/('fake-%d.pt'%budget['completed_updates']);path.write_bytes(b'fixture-model')
                c.write(root/'run/latest.json',dict(path=str(path),sha256=c.sha(path),completed_updates=budget['completed_updates'],policy_updates=4500+budget['completed_updates']))
            else:
                folder=root/'evaluations'/('segment-%02d'%segment);candidate=cand(segment,scores[segment-1])
                candidate.update(status='EVALUATION_COMPLETE',checkpoint=c.read(folder/'candidate.json'))
                c.write(folder/'evaluation-result.json',candidate)
            return 0
        stack.enter_context(patch.object(a,'run_worker',side_effect=worker))
        if pending_attempt:
            b=c.read(root/'budget.json');b.update(attempted_updates=1,interrupted_attempts=1);c.write(root/'budget.json',b)
        result=a.supervise([],1)
        return deepcopy(result),calls

r,calls=run_campaign([210,211]);ok('supervisor early-stops at 500 and retains baseline',r['status']=='REFINEMENT_COMPLETE' and r['budget']['completed_updates']==500 and r['selection']['best']['segment']==0 and len(calls)==4)
r,calls=run_campaign([213,214,215,216]);ok('supervisor maximum four segments / 1000',r['status']=='REFINEMENT_COMPLETE' and r['budget']['completed_updates']==1000 and len(r['selection']['history'])==4 and len(calls)==8)
r,calls=run_campaign([213,214,215,216],pending_attempt=True);ok('interrupted spent attempt never refunded',r['budget']['attempted_updates']==1000 and r['budget']['completed_updates']==999 and r['selection']['best']['checkpoint']['completed_updates']==999)
r,calls=run_campaign([],train_code=2);ok('worker failure has no automatic retry',r['status']=='REFINEMENT_ERROR_PRESERVE' and calls==[('train',1)])
r,calls=run_campaign([],train_code=3);ok('graceful stop preserves run',r['status']=='REFINEMENT_STOPPED_PRESERVE' and calls==[('train',1)])

with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
    root=Path(temp);fixture(root)
    for module in (c,a,ev):stack.enter_context(patch.object(module,'SLOT',root))
    stack.enter_context(patch.object(a,'HERE',root/'package'));stack.enter_context(patch.object(a,'alive',return_value=False))
    stack.enter_context(patch.object(a,'old_slots',return_value=()))
    stack.enter_context(patch.object(a,'preflight',side_effect=AssertionError('must not preflight repeat start')))
    stack.enter_context(patch.object(a.subprocess,'Popen',side_effect=AssertionError('must not launch')))
    def hashes():return {str(p.relative_to(root)):c.sha(p) for p in root.rglob('*') if p.is_file() and p.name!='launch.lock'}
    before=hashes();a.snapshot();a.snapshot();ok('watch snapshot does not mutate files',hashes()==before)
    a.launch(False);ok('repeated start is read-only and cannot launch',hashes()==before)
    b=c.read(root/'budget.json');b['completed_updates']=25;c.write(root/'budget.json',b)
    rejects('resume refuses checkpoint lag',a.check_resume,'lags')
    b['completed_updates']=0;b['deadline_utc']=0;c.write(root/'budget.json',b)
    rejects('resume cannot renew expired deadline',a.check_resume,'expired')
    started=c.read(root/'started.json');started['deadline_utc']=0;c.write(root/'started.json',started)
    stack.enter_context(patch.object(a,'protected_files',return_value={'original':'unchanged'}))
    result=a.supervise([],1)
    ok('expired supervisor does not launch',result['status']=='REFINEMENT_STOPPED_PRESERVE' and result['exit']['reason']=='STOP_REQUEST_OR_ORIGINAL_DEADLINE')

with tempfile.TemporaryDirectory() as temp,ExitStack() as stack:
    root=Path(temp);fixture(root)
    for module in (c,a,ev):stack.enter_context(patch.object(module,'SLOT',root))
    stack.enter_context(patch.object(a,'HERE',root/'package'));stack.enter_context(patch.object(a,'alive',return_value=False))
    stack.enter_context(patch.object(a,'old_slots',return_value=()))
    stack.enter_context(patch.object(a,'preflight',return_value=(None,{'original':'unchanged'})))
    stack.enter_context(patch.object(a.subprocess,'Popen',return_value=SimpleNamespace(pid=999999)))
    stack.enter_context(patch.object(a,'identity',return_value=dict(pid=999999,start='fixture',state='S',group=999999)))
    b=c.read(root/'budget.json');b.update(attempted_updates=1,pending=True);c.write(root/'budget.json',b)
    before_started=(root/'started.json').read_bytes();before_deadline=b['deadline_utc']
    a.launch(True);b=c.read(root/'budget.json')
    ok('explicit resume closes pending without refund',b['attempted_updates']==1 and b['completed_updates']==0 and b['interrupted_attempts']==1 and not b['pending'])
    ok('explicit resume retains started and deadline',b['deadline_utc']==before_deadline and (root/'started.json').read_bytes()==before_started)

controller=ROOT/'microduck-stage12-refine-4500-v1-20261020.py';ns={'__name__':'offline_controller'}
exec(compile(controller.read_bytes(),str(controller),'exec'),ns)
payload=ns['decode'](ns['PAYLOAD_B85'],ns['PAYLOAD_SHA'],False)
files={k:base64.b64decode(v) for k,v in json.loads(zlib.decompress(payload))['files'].items()}
ok('embedded package matches reviewed source',all((PKG/k).read_bytes()==v for k,v in files.items()))
manifest=json.loads(files['manifest.json'])
ok('package identity includes every file',manifest['identity']==hashlib.sha256(json.dumps({k:hashlib.sha256(v).hexdigest() for k,v in sorted(files.items()) if k!='manifest.json'},sort_keys=True).encode()).hexdigest())
for name,value in files.items():
    if name.endswith('.py'):ast.parse(value,filename=name)
ok('all embedded Python parses',True)
bootstrap=ns['decode'](ns['BOOT_B85'],ns['BOOT_SHA']).decode();ast.parse(bootstrap)
ok('bootstrap uses requested start or resume under project Python',"str(target/'agent.py'),sys.argv[2]" in bootstrap and 'workspace/.venv/bin/python' in bootstrap)
ok('SSH helper is unchanged',ns['decode'](ns['SSH_B85'],ns['SSH_SHA'])==(HERE.parent/'native-training-v2/shared_ssh.py').read_bytes())
calls=[]
def rpc(_ssh,args,data=None):calls.append((args,data));return dict(status='NOT_STARTED')
ns['rpc']=rpc
with redirect_stdout(io.StringIO()):
    for operation in ('start','resume','status','stop'):ns['request']({},operation)
ok('only explicit launch operations send payload',all(data is not None for _,data in calls[:2]) and all(data is None for _,data in calls[2:]))
ok('read-only status dispatch uses project Python',calls[2][0][-2:]==[ns['RUNTIME'],'status'])
ok('completed supervisor is polled until process exits',ns['pending'](dict(status='REFINEMENT_COMPLETE',exit={},processes_alive={'supervisor':True})))
ok('network error never auto-restarts',not ns['pending'](dict(status='REMOTE_ERROR_PRESERVE',processes_alive={'supervisor':True})))

report=dict(status='OFFLINE_CHECKS_PASSED',count=len(checks),checks=checks,package_identity=manifest['identity'],
    actual_GPU_training_executed=False,actual_GPU_performance_measured=False,
    scope='Source identities, byte-preserved metric/normalization/performance functions, real budget/selection helpers, supervisor lifecycle with mocked workers and process identities, package/dispatch integrity',
    limitations='No Torch/MuJoCo/CUDA in this executor. Process lifecycle uses fixtures, not real /proc. Cloud native API compatibility checked against pinned runtime source; actual 4096-environment behavior awaits user execution.')
(HERE/'offline-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
