"""Focused reward and diagnostics checks with NumPy fixtures, no simulator."""
import ast
from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import sys
import tempfile
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parent;PKG=HERE/'package';sys.path.insert(0,str(PKG))
import common
import evidence
import reward_spec
checks=[]
def ok(name,value):
    assert value,name
    checks.append(name)

class Tensor(np.ndarray):
    def square(self):return tensor(np.square(np.asarray(self)))
def tensor(value):return np.asarray(value,dtype=np.float64).view(Tensor)
fake_torch=ModuleType('torch')
fake_torch.exp=lambda value:tensor(np.exp(value))
fake_torch.sum=lambda value,dim:tensor(np.sum(value,axis=dim))
fake_torch.nan_to_num=lambda value,**kw:tensor(np.nan_to_num(value,**kw))
source=HERE.parent.parent/'handoff-source/unpacked/pinned-project-excerpts/src/mjlab_microduck/tasks/mdp.py'
tree=ast.parse(source.read_text());node=next(n for n in tree.body if getattr(n,'name',None)=='double_balance_top_score_from_values')
for arg in node.args.args:arg.annotation=None
node.returns=None
ns={'torch':fake_torch};exec(compile(ast.Module(body=[node],type_ignores=[]),'<frozen-top-reward>','exec'),ns)
top_score=ns['double_balance_top_score_from_values']
fake_native=ModuleType('native_task')
original_function=lambda env:None
template=SimpleNamespace(scene=dict(num_envs=4096,objects=['robot','ball','top_ball']),
    sim=dict(dt=.002,control=.02,solver='Newton',iterations=10),
    observations=dict(actor=61,critic=85),commands=dict(twist=0),events=dict(reset='original'),terminations=dict(top_ball_lost='original'),
    rewards={'double_balance':SimpleNamespace(func=original_function,weight=4.,params={}),
        'ball_speed':SimpleNamespace(func='original_ball_speed',weight=-.05,params={}),
        'action_rate_l2':SimpleNamespace(func='original_action_rate',weight=-.4,params={})})
fake_native.make_cfg=lambda *_:deepcopy(template)
fake_native.head_scale=lambda env:1.
fake_mdp=SimpleNamespace(double_balance_top_score_from_values=top_score,
    _stage11_head_kinematics=lambda env:(env.position,env.upper_velocity,None),
    double_balance_lower_score=lambda env:env.lower_score)
tasks=ModuleType('mjlab_microduck.tasks');tasks.mdp=fake_mdp
with patch.dict(sys.modules,{'torch':fake_torch,'native_task':fake_native,'mjlab_microduck.tasks':tasks}):
    spec=importlib.util.spec_from_file_location('tested_speed_task',PKG/'speed_task.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cfg=module.make_cfg();ok('exactly one reward function changes',cfg.rewards['double_balance'].func is module.speed_aligned_head_reward)
    cfg.rewards['double_balance'].func=original_function
    ok('physics observations resets rewards weights preserved',vars(cfg)==vars(template))
    ok('metadata accessible without Torch and matches task',module.REWARD_SPEC==reward_spec.REWARD_SPEC)
    speeds=np.linspace(0,.4,101)
    def env(lower,upper,quality=1.,x=0.):
        lower,upper=np.broadcast_arrays(lower,upper);n=lower.size
        return SimpleNamespace(position=tensor(np.column_stack([np.full(n,x),np.zeros(n),np.full(n,.02)])),
            upper_velocity=tensor(np.column_stack([upper.ravel(),np.zeros(n),np.zeros(n)])),
            lower_score=tensor(np.full(n,quality)),scene={'ball':SimpleNamespace(data=SimpleNamespace(root_link_lin_vel_w=tensor(np.column_stack([lower.ravel(),np.zeros(n),np.zeros(n)]))))})
    zeros=module.speed_aligned_head_reward(env(0.,0.));ok('zero-speed nominal reward unchanged',np.allclose(zeros,1.))
    lower=module.speed_aligned_head_reward(env(speeds,0.));upper=module.speed_aligned_head_reward(env(0.,speeds))
    ok('lower speed strictly reduces task reward',np.all(np.diff(lower)<0))
    ok('upper speed strictly reduces task reward',np.all(np.diff(upper)<0))
    grid=np.meshgrid(speeds,speeds);state=env(*grid);actual=module.speed_aligned_head_reward(state)
    expected=np.exp(-(grid[0].ravel()/.15)**2-(grid[1].ravel()/.08)**2)
    ok('actual reward matches Gaussian product on 10201 states',np.allclose(actual,expected,rtol=1e-13,atol=1e-15))
    original=top_score(state.position,state.upper_velocity)
    ok('new reward never exceeds old at fixed geometry',np.all(actual<=original+1e-15) and np.all(actual>=0))
    ok('one threshold gives exp(-1)',np.allclose(module.speed_aligned_head_reward(env(.15,0.)),np.exp(-1)) and np.allclose(module.speed_aligned_head_reward(env(0.,.08)),np.exp(-1)))
    ok('both thresholds give exp(-2)',np.allclose(module.speed_aligned_head_reward(env(.15,.08)),np.exp(-2)))
    ok('quiet ball cannot reward failed lower posture',np.allclose(module.speed_aligned_head_reward(env(speeds,0.,quality=0.)),0.))
    good=module.speed_aligned_head_reward(env(.04,.02));bad=module.speed_aligned_head_reward(env(.04,.02,x=.03))
    ok('original head centering preference remains',np.all(bad<good))
    ok('NaN reward follows original finite fallback',np.allclose(module.speed_aligned_head_reward(env(np.nan,0.)),0.))

old=HERE.parent/'native-refine-4500-20261020/package'
ok('evaluation worker byte-identical to completed campaign',(PKG/'evaluate.py').read_bytes()==(old/'evaluate.py').read_bytes())
ok('metric observer and physical thresholds byte-identical',(PKG/'eval_metrics.py').read_bytes()==(old/'eval_metrics.py').read_bytes())
train=(PKG/'train.py').read_text();ev=(PKG/'evaluate.py').read_text()
ok('training activates new reward but evaluation stays original','from speed_task import make_cfg,REWARD_SPEC' in train and 'from native_task import make_cfg,head_scale' in ev and 'speed_task' not in ev)
ok('new campaign participates in old refinement lock',common.PREVIOUS in common.old_slots() and common.PREVIOUS!=common.SLOT)

log='discard old partial\n'
for i in range(60):
    log+='Learning iteration %d/5000\nEpisode_Reward/ball_speed: -0.0003\nEpisode_Reward/double_balance: 3.5000\nIteration time: 2.00s\n'%i
log+='Learning iteration 60/5000\nEpisode_Reward/double_balance: 9.9000\n'
parsed=evidence.parse_reward_log(log)
ok('diagnostics ignore incomplete final iteration',parsed['samples']==50 and parsed['first_iteration']==10 and parsed['last_iteration']==59)
ok('diagnostics preserve weighted reward sign and magnitude',math.isclose(parsed['contributions']['Episode_Reward/ball_speed']['mean'],-.0003) and parsed['contributions']['Episode_Reward/double_balance']['median']==3.5)

with tempfile.TemporaryDirectory() as temp:
    root=Path(temp);(root/'run').mkdir();(root/'logs').mkdir()
    rows=[]
    for i in range(120):
        rows.append(dict(phase='refine_ppo',segment=1 if i<60 else 2,session=1 if i<60 else 3,completed_updates=i+1,
            elapsed_seconds=i*2+(1000 if i>=60 else 0),collection_seconds=1.5,learning_seconds=.5,
            episode_metrics={'Episode_Reward/ball_speed':-.0002}))
    (root/'run/metrics.jsonl').write_text('\n'.join(map(json.dumps,rows))+'\n{"partial"')
    (root/'logs/001-train-segment-01.log').write_text(log)
    before={p:common.sha(p) for p in root.rglob('*') if p.is_file()}
    data=evidence.inspect_run(root)
    ok('diagnostics leave original log bytes unchanged',before=={p:common.sha(p) for p in root.rglob('*') if p.is_file()})
    ok('timing excludes reinitialization/evaluation gap',len(data['segments'])==2 and all(x['wall_seconds_between_updates']['median']==2 for x in data['segments']))
    ok('4096 by 24 throughput uses existing timers',all(x['environment_transitions_per_logged_compute_second']['median']==4096*24/2 for x in data['segments']))
    ok('malformed metric tail reported without blocking',data['parsed_rows']==120 and data['invalid_lines']==1)
    ok('missing optional diagnostics do not block training',evidence.inspect_safe(root/'missing')['available'] is False)

client={'__name__':'offline'};exec(compile((HERE.parent.parent/'microduck-stage12-speed-refine-v1-20261020.py').read_bytes(),'<controller>','exec'),client)
from contextlib import redirect_stdout
import io
out=io.StringIO()
with redirect_stdout(out):client['show'](dict(status='TRAINING',budget={'completed_updates':250,'attempted_updates':250,'deadline_utc':0},progress={'segment':1,'phase':'refine_ppo'}))
ok('progress renders percent and bounded budget','25.0%' in out.getvalue() and '250/1000' in out.getvalue())

report=dict(status='REWARD_AND_DIAGNOSTICS_CHECKS_PASSED',count=len(checks),checks=checks,
    package_identity=json.loads((PKG/'manifest.json').read_text())['identity'],actual_GPU_training_executed=False,
    tested_reward_grid_states=10201,reward_spec=reward_spec.REWARD_SPEC,
    example_scale_comparison=dict(lower_speed_m_s=.15,old_lower_speed_cost=-.05*.15**2,
        old_upper_speed_std=.12,new_upper_speed_std=.08,old_upper_factor_at_008=math.exp(-(.08/.12)**2),new_upper_factor_at_008=math.exp(-1)),
    limitations='NumPy-backed tensor fixture and configuration fixture; GPU execution and learning benefit remain unmeasured. Analytical reward values are not measured rollout reward contributions.')
(HERE/'reward-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False,indent=2))
