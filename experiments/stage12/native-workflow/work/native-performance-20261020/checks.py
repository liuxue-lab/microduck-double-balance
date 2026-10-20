"""Focused CPU/NumPy checks. No CUDA, MuJoCo, SSH connection or PPO run."""
import ast
import base64
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
import zlib
import numpy as np
import perf_read

ROOT = Path(__file__).resolve().parent
checks = []
def ok(label, condition):
    assert condition, label
    checks.append(label)

# Execute the actual candidate predicates with NumPy-backed tensor operations.
# This validates logical decisions and host-read placement, not CUDA behavior.
class Tensor:
    def __init__(self, value, device='cuda:0'):
        self.value = np.asarray(value)
        self.device = device
    def all(self): return Tensor(self.value.all(), self.device)
    def any(self): return Tensor(self.value.any(), self.device)
    def item(self):
        fake.reads += 1
        return self.value.item()
    def __invert__(self): return Tensor(~self.value, self.device)
    def detach(self): return self
    def float(self): return Tensor(self.value.astype(np.float32), self.device)
    def mean(self): return Tensor(self.value.mean(), self.device)
    def cpu(self):
        if self.device != 'cpu': fake.transfers += 1
        return Tensor(self.value, 'cpu')
    def tolist(self): return self.value.tolist()

class FakeTorch:
    reads = 0
    transfers = 0
    def as_tensor(self, value): return value if isinstance(value,Tensor) else Tensor(value, 'cpu')
    def isfinite(self, value): return Tensor(np.isfinite(value.value), value.device)
    def stack(self, values):
        assert len({str(v.device) for v in values}) == 1
        return Tensor(np.stack([v.value for v in values]), values[0].device)

fake = FakeTorch()
tree = ast.parse((ROOT/'candidate/train.py').read_text())
names = {'require_finite_group', 'require_step_outputs', 'host_scalar_means'}
nodes = [n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
def require(condition, message):
    if not condition: raise RuntimeError(message)
ns = {'require':require}
exec(compile(ast.Module(body=nodes,type_ignores=[]),'<candidate-predicates>','exec'),ns)
for location in ('clean','actor','critic','reward','done','nan_state'):
    for bad in ([None] if location in ('clean','nan_state') else [np.nan,np.inf,-np.inf]):
        obs = {'actor':Tensor([1.,2.]),'critic':Tensor([3.,4.])}
        reward,done,nan = Tensor([1.,2.]),Tensor([False,True]),Tensor([False,False])
        if location in obs: obs[location] = Tensor([0.,bad])
        if location == 'reward': reward = Tensor([0.,bad])
        if location == 'done': done = Tensor([0.,bad])
        if location == 'nan_state': nan = Tensor([True,False])
        fake.reads = 0
        try:
            ns['require_step_outputs'](fake,obs,reward,done,nan)
            rejected = False
        except RuntimeError:
            rejected = True
        ok('step '+location+' '+str(bad), rejected == (location != 'clean'))
        if location == 'clean': ok('post-step single host scalar read',fake.reads == 1)
for bad in (None,np.nan,np.inf,-np.inf):
    fake.reads=0
    values=[Tensor(np.ones((2,3))) for _ in range(20)]
    if bad is not None: values[10]=Tensor([bad])
    try:
        ns['require_finite_group'](fake,iter(values),'weights')
        rejected=False
    except RuntimeError:
        rejected=True
    ok('parameter detection '+str(bad),rejected == (bad is not None) and fake.reads == 1)
inputs=[Tensor([1.,3.]),7.,Tensor([2.,4.]),Tensor([6.,10.],'cuda:1'),Tensor([np.nan])]
fake.transfers=0
means=ns['host_scalar_means'](fake,inputs)
ok('metric ordering and float32 means preserved',np.allclose(means[:4],[2.,7.,3.,8.]) and np.isnan(means[4]))
ok('metric transfers grouped by device',fake.transfers == 2)

sample='incomplete leading Collection time: 99.000s\n'
for i in range(40):
    sample += '\x1b[1m Learning iteration %d/6000 \x1b[0m\nCollection time: 1.800s\nLearning time: 0.600s\n' % i
sample+='Learning iteration 40/6000\nCollection time: 1.900s\n'
parsed=perf_read.parse_log(sample)
ok('last 30 complete logs exclude truncated blocks',len(parsed['recent_iterations'])==30 and parsed['recent_iterations'][0]['iteration']==10 and parsed['recent_iterations'][-1]['iteration']==39)
ok('existing timers preserve ratio',abs(parsed['collection_fraction_of_logged_compute']-.75)<1e-12)
metrics='bad partial\n[]\n'
for i in range(35):
    metrics+=json.dumps(dict(campaign_phase='formal',session=1,elapsed_seconds=i*2.5,completed_updates=i))+'\n'
metrics+='{"truncated"'
parsed=perf_read.parse_metrics(metrics)
ok('metrics skip incomplete lines and cap window',parsed['wall_seconds_per_update']['samples']==30 and parsed['wall_seconds_per_update']['median']==2.5)
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);(root/'run').mkdir()
    (root/'run/train-01-formal.log').write_text(sample)
    (root/'run/metrics.jsonl').write_text(metrics)
    (root/'budget.json').write_text('{"completed_updates":34}')
    def hashes():return {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*') if p.is_file()}
    before=hashes()
    result=perf_read.inspect_slot(root)
    ok('collection leaves all input bytes unchanged',hashes()==before)
    ok('collection uses existing files only',result['added_PPO_updates']==0 and result['read_only'])
    bounded=root/'bounded.txt';bounded.write_text('discard_this\nkeep_line\n')
    ok('tail respects byte limit and discards partial first line',perf_read.tail(bounded,12)=='keep_line\n')

client={ '__name__':'offline_test' }
controller=ROOT.parent.parent/'microduck-stage12-perf-readonly-20261020.py'
exec(compile(controller.read_bytes(),str(controller),'exec'),client)
remote=zlib.decompress(base64.b85decode(client['REMOTE_B85']))
ok('remote payload integrity',hashlib.sha256(remote).hexdigest()==client['REMOTE_SHA'])
remote_tree=ast.parse(remote)
for node in ast.walk(remote_tree):
    if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
        ok('remote read-only call '+node.func.attr,node.func.attr not in {'write','write_text','write_bytes','mkdir','unlink','rename','replace','Popen','run','system'})
with tempfile.TemporaryDirectory() as tmp,redirect_stdout(io.StringIO()):
    client['exec']=lambda code,namespace:namespace.update(argv=lambda command:['offline-mock-ssh',command])
    response=SimpleNamespace(returncode=0,stdout=('MD12_PERF_STATE='+json.dumps(result)+'\n').encode(),stderr=b'')
    with patch.object(client['subprocess'],'run',return_value=response) as call,patch.object(client['Path'],'home',return_value=Path(tmp)):
        client['main']()
        ok('client single read-only remote command',call.call_count==1 and 'python3 -B -c' in call.call_args.args[0][1])
    outputs=list(Path(tmp).rglob('*.json'))
    ok('client writes one small local report',len(outputs)==1 and outputs[0].stat().st_size<32768)

# Changes to physical task, budgets, snapshot cadence and checkpoint load are out of scope.
original=ast.parse((ROOT.parent/'native-training-v2/package/train.py').read_text())
candidate_tree=ast.parse((ROOT/'candidate/train.py').read_text())
for function in ('save_model','boundary'):
    def find(t):return next(n for n in ast.walk(t) if isinstance(n,ast.FunctionDef) and n.name==function)
    ok(function+' unchanged AST',ast.dump(find(original))==ast.dump(find(candidate_tree)))
receipt=json.loads((ROOT/'build-receipt.json').read_bytes())
ok('running controller bytes unchanged',receipt['original_controller_sha256']=='5b57ef3a761757b9d2bd2c39e7faaa6713b1f4e29b54500f5e1ded13f33438cb')
summary=dict(status='PASS',checks=len(checks),checks_passed=checks,
             scope='CPU NumPy-backed predicates, parser and mock SSH; no PyTorch/CUDA performance claim',
             speedup_measured=False,cloud_changes=0,PPO_updates=0,simulation_steps=0)
(ROOT/'offline-review.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({k:v for k,v in summary.items() if k!='checks_passed'}))
