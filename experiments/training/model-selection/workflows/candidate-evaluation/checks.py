"""Offline semantic checks. No Torch/CUDA/physics execution."""
import ast
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace as N
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
sys.path.insert(0,str(HERE/'package'))
import eval_common as c
import evaluate as e
import agent
checks=[]
def ok(name,value):
    assert value,name
    checks.append(name)

# NumPy-backed tensors exercise the actual Observer without importing Torch.
class A(np.ndarray):
    def all(self,axis=None,out=None,keepdims=False,*,dim=None):
        return arr(np.asarray(self).all(axis=dim if dim is not None else axis,out=out,keepdims=keepdims))
    def any(self,axis=None,out=None,keepdims=False,*,dim=None):
        return arr(np.asarray(self).any(axis=dim if dim is not None else axis,out=out,keepdims=keepdims))
    def float(self):return self.astype(np.float32)
    def double(self):return self.astype(np.float64)
    def long(self):return self.astype(np.int64)
    def bool(self):return self.astype(bool)
    def abs(self):return np.abs(self)
    def cpu(self):return self
    def detach(self):return self
    def contiguous(self):return self
    def numpy(self):return np.asarray(self)
def arr(x,dtype=None):return np.asarray(x,dtype=dtype).view(A)
def zeros(shape,dtype=None,device=None):return arr(np.zeros(shape,dtype=dtype))
def ones(shape,dtype=None,device=None):return arr(np.ones(shape,dtype=dtype))
torch=N(zeros=zeros,ones=ones,zeros_like=lambda x,dtype=None:arr(np.zeros_like(x,dtype=dtype)),
    where=lambda cond,a,b:arr(np.where(cond,a,b)),maximum=lambda a,b:arr(np.maximum(a,b)),
    stack=lambda xs,dim=0:arr(np.stack(xs,axis=dim)),isfinite=lambda x:arr(np.isfinite(x)),
    linalg=N(vector_norm=lambda x,dim:arr(np.linalg.norm(x,axis=dim))),bool=bool,long=np.int64,float64=np.float64)
n=3
p=arr([[0.,0.,.02]]*n);v=zeros((n,3));root=arr([[0.,0.,.24]]*n);ball=zeros((n,3));hold=zeros(n)
raw=N(num_envs=n,device='cpu',scene={'robot':N(data=N(root_link_pos_w=root)),
    'ball':N(data=N(root_link_pos_w=ball,root_link_lin_vel_w=zeros((n,3))))},
    command_manager=N(get_command=lambda _:zeros((n,3))),termination_manager=N(get_term=lambda _:zeros(n,bool)))
mdp=N(_stage11_head_kinematics=lambda _: (p,v,None),wrestle_tilt=lambda *args:zeros(n),_bb_state=lambda _:N(hold=hold))
mask=ones(n,bool);ticks=zeros(n,np.int64);ever=zeros(n,bool);cached=zeros((n,2))
def original_compute():
    ticks[:]=np.where(mask,ticks+1,0);ever[:]|=ticks>=250
    cached[:,0]=mask;cached[:,1]=ever
raw.metrics_manager=N(active_terms=['native_joint_geometric_stable_fraction','native_joint_geometric_success_5s'],
    compute=original_compute,_step_values=cached)
observer=e.Observer(raw,torch,mdp)
for _ in range(249):observer.compute()
ok('249 uninterrupted ticks are not success',not observer.ever.any())
mask[1]=False;p[1,0]=.013
observer.compute()
ok('250 ticks pass while one brief violation resets the other run',observer.ever.tolist()==[True,False,True] and observer.current[1]==0)
ok('diagnosis attributes threshold crossing to top-center gate',observer.failures[1,0]==1 and observer.breaks[1,0]==1)
observer.active[2]=False;observer.compute()
ok('finished episodes never contribute a second episode',observer.steps.tolist()==[251,251,250])
mask[:]=True;p[:,0]=0
observer.compute()
rows=observer.rows(20261021,[False,False,True],[['time_out'],['time_out'],['fell']])
ok('physical failure vetoes success even after a five-second run',rows[0]['success'] and rows[2]['ever_5s'] and not rows[2]['success'])
summary=c.summarize(rows)
ok('success denominator counts each first episode exactly once',summary['successes']==1 and summary['episodes']==3 and summary['success_fraction']==1/3)
ok('diagnostic sums use first-episode step denominator',abs(summary['gate_violation_fraction']['top_center']-2/sum(r['steps'] for r in rows))<1e-12)
ok('Wilson limits handle all-failure and all-success samples',abs(c.wilson(0,256)[0])<1e-15 and abs(c.wilson(256,256)[1]-1)<1e-15)
right=deepcopy(rows);right[0]['success']=False;right[1]['success']=True
pair=c.paired_comparison(rows,right)
ok('paired wins and losses computed by seed and environment',pair['only_4500_success']==pair['only_6000_success']==1 and pair['episodes']==3)

state={'a':arr([1.,2.]),'b':arr([3.])}
ok('initial-state fingerprint stable regardless of map order',e.fingerprint(state)==e.fingerprint(dict(reversed(list(state.items())))))
modified=deepcopy(state);modified['a'][0]+=1
ok('fingerprint detects a changed physical or normalizer value',e.fingerprint(state)!=e.fingerprint(modified))
v[0,0]=np.nan
try:observer.compute();raises=False
except RuntimeError:raises=True
ok('nonfinite active diagnostic state stops evaluation',raises)
v[0,0]=0;hold[0]=.1
try:observer.compute();raises=False
except RuntimeError:raises=True
ok('unexpected assistance is rejected without weakening threshold',raises)

for path in [*list((HERE/'package').glob('*.py')),HERE/'bootstrap.py',ROOT/'evaluate_candidates.py']:
    ast.parse(path.read_text())
ok('all executable sources parse',True)
tree=ast.parse((HERE/'package/evaluate.py').read_text())
calls=[x.func.attr for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)]
ok('evaluation contains no learn/backward/optimizer update call',not set(calls)&{'learn','backward','act','process_env_step','compute_returns'})
update_calls=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='update']
ok('update calls only touch hashes and config/status dictionaries',all(isinstance(x.func.value,ast.Name) and x.func.value.id in ('h','values','agent','state') for x in update_calls))
policy_calls=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Name) and x.func.id=='policy']
ok('every policy call requests deterministic output',len(policy_calls)==1 and any(k.arg=='stochastic_output' and k.value.value is False for k in policy_calls[0].keywords))
ok('existing native training task is imported unchanged', 'from native_task import make_cfg,head_scale' in (HERE/'package/evaluate.py').read_text())

client={'__name__':'offline_client'}
exec(compile((ROOT/'evaluate_candidates.py').read_bytes(),'<client>','exec'),client)
payload=client['decode'](client['PAYLOAD_B85'],client['PAYLOAD_SHA'],False)
import zlib,base64
files=json.loads(zlib.decompress(payload))['files']
ok('embedded package matches reviewed source byte-for-byte',all(base64.b64decode(value)==(HERE/'package'/name).read_bytes() for name,value in files.items()))
captured=[]
def fake_rpc(ssh,args,data=None):captured.append((args,data));return {'status':'NOT_STARTED'}
client['rpc']=fake_rpc
client['request']({},'status')
args=captured[-1][0]
with tempfile.TemporaryDirectory() as tmp:
    result=subprocess.run([sys.executable,*args[1:-1],tmp],capture_output=True,text=True,check=True)
    ok('read-only status command executes without creating an evaluation',json.loads(result.stdout.split('=',1)[1])['status']=='NOT_STARTED' and not list(Path(tmp).iterdir()))
with redirect_stdout(io.StringIO()):client['request']({},'start')
ok('only explicit start uploads the immutable package',captured[-1][1]==payload and len(captured)==2)

with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);legacy=root/'old';train=root/'train';slot=root/'eval'
    for p in (legacy,train,slot):p.mkdir()
    (slot/'started.json').write_text('{}')
    with patch.multiple(agent,LEGACY=legacy,TRAIN=train,SLOT=slot),patch.object(agent,'snapshot',return_value={'status':'EXISTING'}),patch.object(agent,'preflight',side_effect=AssertionError('must not rerun')):
        ok('repeated start attaches without re-evaluation or budget reset',agent.launch()=={'status':'EXISTING'})
    (slot/'progress.json').write_text(json.dumps({'status':'EVALUATING'}))
    for name in ('supervisor','child'):(slot/(name+'.json')).write_text(json.dumps({'pid':123,'start':'fixture'}))
    with patch.object(agent,'SLOT',slot),patch.object(agent,'alive',return_value=False):
        ok('stale worker records become interrupted, never automatically rerun',agent.snapshot()['status']=='EVALUATION_INTERRUPTED_PRESERVE')
    (slot/'exit.json').write_text(json.dumps({'status':'EVALUATION_ERROR_PRESERVE'}))
    with patch.object(agent,'SLOT',slot),patch.object(agent,'alive',return_value=False):
        ok('failed exit remains failed even when polling',agent.snapshot()['status']=='EVALUATION_ERROR_PRESERVE')
fields=['R']+['0']*18+['12345']
with patch.object(c.Path,'read_text',return_value='321 (python fixture) '+' '.join(fields)):
    ok('PID reuse does not count as an existing evaluation process',c.alive({'pid':321,'start':'wrong'}) is False and c.alive({'pid':321,'start':'12345'}) is True)

origin=(ROOT/'train.py').read_bytes()
ok('original training controller remains unchanged',hashlib.sha256(origin).hexdigest()=='5b57ef3a761757b9d2bd2c39e7faaa6713b1f4e29b54500f5e1ded13f33438cb')
for key,expected in c.MODELS.items():
    final=json.loads((ROOT/'records/training-results-20261020T083710.722181Z-10474.json').read_text())
    receipt=final['checkpoints']['latest' if key=='6000' else 'best_training_metric']
    ok('checkpoint '+key+' pinned to returned verified receipt',receipt['verified'] and receipt['sha256']==expected)
review=dict(status='PASS',checks=checks,count=len(checks),scope='Offline NumPy semantics and standard-library lifecycle only; no real Torch, CUDA or simulation run.',
            added_PPO_updates=0,added_simulation_steps=0,cloud_connections=0)
(HERE/'offline-review.json').write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(dict(status='PASS',checks=len(checks),PPO=0,simulation_steps=0)))
