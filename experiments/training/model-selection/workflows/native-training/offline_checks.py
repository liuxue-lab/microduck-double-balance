"""Local code tests only. No simulator, Torch, network, GPU or PPO execution."""
import ast
import base64
from contextlib import ExitStack, nullcontext
from copy import deepcopy
import fcntl
import hashlib
import importlib.util
from importlib.metadata import distribution
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as N
from unittest.mock import patch
import zlib
import numpy as np
ROOT=Path(__file__).resolve().parent;P=ROOT/'package'
sys.path.insert(0,str(P))
import common,agent,probe,transfer,train
checks=[]
def ok(name,value):
    if not value:raise AssertionError(name)
    checks.append(name)
def reject(name,fn):
    try:fn()
    except (RuntimeError,BlockingIOError):checks.append(name)
    else:raise AssertionError(name)

for p in list(P.glob('*.py'))+[ROOT/'bootstrap.py',ROOT.parent.parent/'train.py']:
    compile(p.read_bytes(),str(p),'exec')
ok('Python syntax',True)
manifest=common.read(P/'manifest.json')
for name,pin in manifest['files'].items():
    data=(P/name).read_bytes();ok('manifest '+name,len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['sha256'])
client={'__name__':'offline_client'}
exec(compile((ROOT.parent.parent/'train.py').read_bytes(),'<client>','exec'),client)
blob=base64.b85decode(client['PACKAGE_B85']);payload=json.loads(zlib.decompress(blob))
ok('transport hash',hashlib.sha256(blob).hexdigest()==client['PACKAGE_SHA'])
for name,data in payload['files'].items():ok('embedded '+name,base64.b85decode(data)==(P/name).read_bytes())
probe_src=zlib.decompress(base64.b85decode(client['PROBE_B85']))
ns={'__name__':'probe_offline'};exec(compile(probe_src,'<embedded-probe>','exec'),ns)
ok('embedded probe executable without CUDA',callable(ns['discover']))
cfg=payload['config']
ok('one bounded smoke then formal', (cfg['smoke_envs'],cfg['smoke_updates'],cfg['num_envs'],cfg['max_updates'],cfg['max_hours'])==(64,5,2048,6000,8))
b=dict(deadline_utc=20,max_updates=2,attempted_updates=0,completed_updates=0,pending=False,interrupted_attempts=0)
common.reserve(b,1);common.commit(b);common.reserve(b,2);common.abandon_partial_for_explicit_resume(b)
ok('spent interruption not refunded',b['attempted_updates']==2 and b['completed_updates']==1 and b['interrupted_attempts']==1)
reject('budget exhaustion enforced',lambda:common.reserve(b,3))
reject('deadline enforced',lambda:common.reserve(dict(b,attempted_updates=0),20))

with tempfile.TemporaryDirectory() as tmp,ExitStack() as st:
    root=Path(tmp);slot=root/'v2';legacy=root/'v1';slot.mkdir();legacy.mkdir();(slot/'run').mkdir()
    fakepy=root/'python';fakepy.write_text('not executed')
    for module in (common,agent,probe):st.enter_context(patch.multiple(module,SLOT=slot,LEGACY_SLOT=legacy))
    st.enter_context(patch.object(agent,'PYTHON',fakepy))
    common.write(slot/'config.json',cfg)
    # Only installed v1 is allowed; started/budget/live process must all block.
    common.write(legacy/'budget.json',{'prior':True})
    old=(legacy/'budget.json').read_bytes()
    with patch.object(agent.subprocess,'Popen') as pop:
        result=agent.launch()
        ok('legacy budget-only record blocks v2',result['legacy_attached'] and pop.call_count==0)
    ok('legacy budget byte preserved',(legacy/'budget.json').read_bytes()==old)
    (legacy/'budget.json').unlink()  # only our temporary fixture
    pin=dict(pid=999999999,start='1',state='S',group=999999999)
    with patch.object(agent,'check_sources',return_value=cfg['package_identity']),patch.object(agent,'find_checkpoint',return_value={'sha256':common.PRIMARY_SHA}),patch.object(agent.shutil,'disk_usage',return_value=N(free=16*1024**3)),patch.object(agent,'identity',return_value=pin),patch.object(agent.subprocess,'Popen',return_value=N(pid=pin['pid'])) as pop:
        agent.launch();first=common.read(slot/'budget.json');started=(slot/'started.json').read_bytes()
        agent.launch()
        ok('repeat train never relaunches',pop.call_count==1)
        ok('repeat train keeps deadline and budget',first==common.read(slot/'budget.json') and started==(slot/'started.json').read_bytes())
        reject('failed smoke cannot resume',lambda:agent.launch(resume=True))
        first.update(attempted_updates=6,completed_updates=5,pending=True)
        common.write(slot/'budget.json',first);common.write(slot/'run/latest.json',{'path':'saved.pt'})
        common.write(slot/'run/smoke-result.json',{'status':'SMOKE_PASSED'});common.write(slot/'STOP.json',{})
        agent.launch(resume=True);resumed=common.read(slot/'budget.json')
        ok('explicit formal resume preserves deadline and lost attempt',resumed['deadline_utc']==first['deadline_utc'] and resumed['attempted_updates']==6 and resumed['interrupted_attempts']==1)
        ok('only explicit resume launches again',pop.call_count==2)
        common.write(legacy/'started.json',{'old':True})
        ok('conflicting campaigns detected',probe.discover()['status']=='MULTIPLE_CAMPAIGNS_PRESERVED')
    # Real flock is shared with the unchanged v1 launcher path.
    with (legacy/'launch.lock').open('a') as one,(legacy/'launch.lock').open('a') as two:
        fcntl.flock(one,fcntl.LOCK_EX|fcntl.LOCK_NB)
        reject('real v1 launch lock excludes another opener',lambda:fcntl.flock(two,fcntl.LOCK_EX|fcntl.LOCK_NB))

valid=dict(status='SMOKE_PASSED',completed_updates=5,reload_verified=True)
bb=dict(completed_updates=5,attempted_updates=5,pending=False)
ok('five updates and reload admit formal',agent.smoke_allows_formal(0,valid,bb))
for label,code,r,b in [('nonzero exit',2,valid,bb),('missing reload',0,dict(valid,reload_verified=False),bb),('short run',0,dict(valid,completed_updates=4),bb),('inflight attempt',0,valid,dict(bb,pending=True)),('extra attempt',0,valid,dict(bb,attempted_updates=6))]:
    ok('formal blocked: '+label,not agent.smoke_allows_formal(code,r,b))

with tempfile.TemporaryDirectory() as tmp,ExitStack() as st:
    root=Path(tmp);slot=root/'v2';slot.mkdir();legacy=root/'v1';legacy.mkdir()
    for module in (common,agent,probe):st.enter_context(patch.multiple(module,SLOT=slot,LEGACY_SLOT=legacy))
    data=b'a'*100+b'b'*200;pin=hashlib.sha256(data).hexdigest()
    st.enter_context(patch.object(agent,'PRIMARY_SHA',pin))
    st.enter_context(patch.object(agent,'find_checkpoint',return_value={'sha256':pin}))
    result=agent.receive_checkpoint(len(data));ok('new upload starts at zero',result['offset']==0)
    def chunk(offset,b):
        with patch.object(agent.sys,'stdin',N(buffer=io.BytesIO(b))):
            return agent.receive_checkpoint(len(data),offset,hashlib.sha256(b).hexdigest())
    chunk(0,data[:100]);ok('partial bytes kept',agent.receive_checkpoint(len(data))['offset']==100)
    chunk(0,data[:100]);ok('duplicate chunk idempotent',(slot/'initial-checkpoint.pt.partial').stat().st_size==100)
    reject('noncontiguous upload rejected',lambda:chunk(200,data[200:]))
    reject('differing duplicate preserved',lambda:chunk(0,b'wrong'))
    final=chunk(100,data[100:]);ok('resume transfer verifies whole hash',final['status']=='CHECKPOINT_PRESENT' and (slot/'initial-checkpoint.pt').read_bytes()==data)
    common.write(legacy/'started.json',{})
    reject('model upload refused once legacy starts',lambda:agent.receive_checkpoint(len(data)))

# Array-backed migration check exercises the real migration function. This is
# not a PyTorch/GPU compatibility claim; the actual pinned load API is audited.
class Array(np.ndarray):
    def fill_(self,x):self.fill(x);return self
    def add_(self,x):self+=x;return self
    def mean(self,dim=None,keepdim=False,**kw):return np.asarray(self).mean(axis=dim,keepdims=keepdim).view(Array)
    def var(self,dim=None,unbiased=False,keepdim=False,**kw):return np.asarray(self).var(axis=dim,ddof=int(unbiased),keepdims=keepdim).view(Array)
def arr(shape,value=1.):return np.full(shape,value).view(Array)
fake=N(isfinite=np.isfinite,no_grad=nullcontext,sqrt=np.sqrt)
def sd(width):return {'obs_normalizer._mean':arr((1,width),.1),'obs_normalizer._std':arr((1,width),.7),'obs_normalizer._var':arr((1,width),.49),'obs_normalizer.count':np.array(9999999).view(Array),'kept':arr((2,2),2.)}
a=sd(61);c=sd(85);a['rnn.rnn.weight_ih_l0']=arr((1024,61),.25);c['mlp.0.weight']=arr((512,85),.5)
source={'actor_state_dict':a,'critic_state_dict':c};before=deepcopy(source);result=transfer.migrate_stage08(source,fake)
for role,start,key in [('actor',55,'rnn.rnn.weight_ih_l0'),('critic',76,'mlp.0.weight')]:
    old=source[role+'_state_dict'];new=result[role+'_state_dict']
    ok(role+' unchanged network columns kept',np.array_equal(old[key][:,:start],new[key][:,:start]))
    ok(role+' changed head columns neutral',np.count_nonzero(new[key][:,start:])==0)
    ok(role+' source checkpoint not modified',np.array_equal(old[key],before[role+'_state_dict'][key]))
    norm=N(training=True,_mean=new['obs_normalizer._mean'],_var=new['obs_normalizer._var'],_std=new['obs_normalizer._std'],count=new['obs_normalizer.count'])
    original=norm._mean.copy();transfer.configure_normalizer_updates(norm,start,fake);norm.update(arr((64,new[key].shape[1]),.9))
    ok(role+' old moments frozen, head moments update',np.array_equal(norm._mean[:,:start],original[:,:start]) and not np.array_equal(norm._mean[:,start:],original[:,start:]))

# Exercise the real task-config builder against a lightweight configuration
# object shaped exactly like the pinned factory. No environment is constructed.
code=ast.parse((P/'native_task.py').read_text());make=next(x for x in code.body if isinstance(x,ast.FunctionDef) and x.name=='make_cfg')
noop=lambda *a,**k:None
term=lambda **kw:N(params={},**kw) if 'params' not in kw else N(**kw)
spawn=N(func=noop,params=dict(ball_radius=.12,root_height=.125,xy_noise=.01,yaw_range=(-1.,1.),tilt_noise_deg=3.,joint_noise=.1,ball_vel_noise=.1,tray_site_name='old_tray',top_ball_radius=.02,top_ball_xy_noise=0.,top_ball_clearance=.0005))
cfg=N(scene=N(num_envs=4096,entities={'robot':N(spec_fn=noop),'top_ball':N(spec_fn=noop)}),sim=N(mujoco=N(integrator='implicitfast',iterations=10,ls_iterations=20,tolerance=1e-8)),events={'reset_double_balance':spawn,'push_robot':N(),'expand_bam_friction_fields':N(),'randomize_com':N(params={'ranges':(-.01,.01)}),'randomize_head_com':N(params={'ranges':(-.01,.01)})},actions={'ball_hold':N(levels=())},commands={'twist':N(ranges=N(lin_vel_x=(-1,1),lin_vel_y=(-1,1),ang_vel_z=(-1,1)),rel_standing_envs=.1,rel_heading_envs=.2,rel_turn_in_place_envs=.3)},observations={'actor':N(terms={'body_command':N(func=noop,params={'tray':'x'})}),'critic':N(terms={'top_ball_state':N(func=noop,params={'tray':'x'})})},rewards={'tray_tilt':N(),'action_rate_l2':N(weight=-1)},metrics={k:N(func=noop,params={}) for k in ('double_balance_stable_fraction','double_balance_success','top_ball_center_error_m','top_ball_relative_speed_m_s')},terminations={'top_ball_lost':N()},curriculum={'unwanted_command_ramp':N()},recorders={})
from copy import deepcopy
from functools import partial
import math
ns=dict(deepcopy=deepcopy,partial=partial,math=math,make_microduck_double_balance_env_cfg=lambda **_:cfg,
        make_head_reference_robot_spec=noop,head=N(make_head_contact_top_ball_spec=noop),HOLD_LEVELS=(0.,),
        mdp=N(stage11_head_normal_tilt_l2=noop,stage11_head_center_error=noop,stage11_head_relative_speed=noop),
        EventTermCfg=term,RewardTermCfg=term,TerminationTermCfg=term,CurriculumTermCfg=term,MetricsTermCfg=term)
for name in ('reset_head','startup_geometry','head_actor_observation','head_critic_observation','head_reward','head_speed_cost','lower_hold_curriculum','top_lost','lower_unassisted','unassisted_fraction','geometric_joint_stable','geometric_success'):ns[name]=noop
exec(compile(ast.Module(body=[make],type_ignores=[]),'native_task.make_cfg','exec'),ns)
actual=ns['make_cfg']()
ok('native_fixed_step_solver_config',actual.sim.mujoco.integrator=='implicitfast' and (actual.sim.mujoco.iterations,actual.sim.mujoco.ls_iterations)==(10,20))
ok('all_old_tray_term_parameters_removed',all('tray' not in str(getattr(t,'params',{})) for t in list(actual.events.values())+list(actual.rewards.values())+list(actual.terminations.values())+list(actual.metrics.values())))
ok('stationary_command_and_hold_curriculum',actual.commands['twist'].ranges.lin_vel_x==(0.,0.) and list(actual.curriculum)==[] and actual.actions['ball_hold'].levels[-1]==0)
ok('reset_noise_preserved',actual.events['reset_native_head'].params['joint_noise']==.025)
ok('input dimensions preserved','(num_envs,61)' in (P/'train.py').read_text() and '(num_envs,85)' in (P/'train.py').read_text())

ok('Stage08 E CoM and smoothing retained',actual.rewards['action_rate_l2'].weight==-.4 and actual.events['randomize_com'].params['ranges']==(-.005,.005))

# Client must never install or mutate when a legacy or existing task is found.
with patch.object(sys,'argv',['controller','--train']):
    original=client.copy()
    client['probe']=lambda *_:dict(campaign_exists=True,legacy_attached=True,status='TRAINING')
    client['watch']=lambda ssh,state:state
    client['install']=lambda *_:(_ for _ in ()).throw(AssertionError('unexpected install'))
    result=client['main']();ok('client legacy train is readonly attach',result['legacy_attached'])
    client.update(original)

# Inspect the installed locked distributions as text, without importing their
# numerical or simulator modules. The former handoff excerpts were not included
# in the public source tree. Version checks retain the fixed API provenance.
runtime_sources=[]
for package,version,rel,cls,method,required in [
 ('rsl-rl-lib','5.0.1','rsl_rl/algorithms/ppo.py','PPO','load',{'loaded_dict','load_cfg','strict'}),
 ('rsl-rl-lib','5.0.1','rsl_rl/runners/on_policy_runner.py','OnPolicyRunner','learn',{'num_learning_iterations','init_at_random_ep_len'}),
 ('mjlab','1.3.0','mjlab/rl/vecenv_wrapper.py','RslRlVecEnvWrapper','step',{'actions'})]:
    installed=distribution(package)
    ok('locked dependency '+package,installed.version==version)
    source=Path(installed.locate_file(rel));data=source.read_bytes()
    runtime_sources.append(dict(package=package,version=installed.version,path=rel,sha256=hashlib.sha256(data).hexdigest()))
    tree=ast.parse(data);classnode=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==cls)
    fn=next(n for n in classnode.body if isinstance(n,ast.FunctionDef) and n.name==method)
    ok('pinned API '+cls+'.'+method,required<={n.arg for n in fn.args.args})
ok('no Torch/CUDA/simulator imports',not any(k in sys.modules for k in ('torch','warp','mujoco','mjlab','paramiko')))
report=dict(status='OFFLINE_CODE_CHECKS_COMPLETE',checks_count=len(checks),checks=checks,runtime_sources=runtime_sources,simulations_run=0,PPO_updates_run=0,cloud_commands_run=0,
    planned_cloud_smoke_runs=1,limitations='Control flow, byte integrity, NumPy-backed migration checks and pinned API inspection only; real Stage08 checkpoint/runtime/CUDA smoke not executed here.')
(ROOT/'offline-code-review.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(report,ensure_ascii=False))
