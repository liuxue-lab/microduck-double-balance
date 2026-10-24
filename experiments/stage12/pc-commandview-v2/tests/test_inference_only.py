"""Exercise original mjlab observation methods, RNG and reversible manager hooks."""
import ast
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'package'))
import inference_only as slim


class Noise:
    def apply(self,value):return value+torch.rand_like(value)*.01
class NoiseModel:pass
class Delay:
    def __init__(self):self.values=[]
    def append(self,x):self.values.append(x.clone())
    def compute(self):
        # Draw on every update to reveal any changed critic/actor RNG ordering.
        lag=int(torch.randint(0,2,(1,)).item())
        return self.values[max(0,len(self.values)-1-lag)]
class History:
    def __init__(self):self.values=[]
    @property
    def is_initialized(self):return bool(self.values)
    def append(self,x):self.values.append(x.clone())
    @property
    def buffer(self):return self.values[-1].unsqueeze(1)


tree=ast.parse((ROOT/'tests/fixtures/observation_manager.py').read_text())
original=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ObservationManager')
methods=[n for n in original.body if isinstance(n,ast.FunctionDef) and n.name in ('compute','compute_group','_check_and_handle_nans')]
namespace=dict(torch=torch,noise_cfg=NS(NoiseCfg=Noise,NoiseModelCfg=NoiseModel))
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='OriginalObservation',bases=[],keywords=[],body=methods,decorator_list=[])],type_ignores=[])),'<original-mjlab-observation-methods>','exec'),namespace)
OriginalObservation=namespace['OriginalObservation']


class Reward:
    def compute(self,dt):
        self.calls+=1;self._reward_buf.fill_(.7);self._episode_sums['fixture']+=.7
        return self._reward_buf


def fixture(actor_function=None,critic_function=None):
    raw=NS(num_envs=1,device='cpu',step_dt=.02,physics_dt=.002,
        cfg=NS(auto_reset=False,curriculum={},decimation=10),epoch=0,
        extras={'log':{'Episode_Reward/pose':4.,'Metrics/angular_momentum_mean':3.,'retained_metric':2.}})
    raw.actor_source_calls=0;raw.critic_source_calls=0
    def actor(env):
        env.actor_source_calls+=1
        return actor_function(env) if actor_function else torch.arange(61,dtype=torch.float32).reshape(1,61)/100+env.epoch/1000
    def critic(env):
        env.critic_source_calls+=1
        return critic_function(env) if critic_function else torch.arange(85,dtype=torch.float32).reshape(1,85)/100
    modules={name:NS() for name in slim.SOURCES}
    for module,name in set(slim.CRITIC.values())|set(slim.REWARDS.values()):
        setattr(modules[module],name,lambda *a,**k:None)
    modules['mjlab.managers.observation_manager'].ObservationManager=OriginalObservation
    modules['mjlab.managers.reward_manager'].RewardManager=Reward
    obs=OriginalObservation();raw.observation_manager=obs;obs._env=raw
    obs._group_obs_term_names={'actor':['actor_fixture'],'critic':list(slim.CRITIC)}
    def cfg(fn):return NS(func=fn,params={},noise=None,clip=None,scale=None,delay_max_lag=0,history_length=0,flatten_history_dim=True)
    ac=cfg(actor);ac.noise=Noise();ac.delay_max_lag=1;ac.history_length=1
    cc=[]
    for i,(module,name) in enumerate(slim.CRITIC.values()):
        def term(env,index=i):
            data=critic(env)
            return data[:,index:index+1] if index<13 else data[:,13:]
        setattr(modules[module],name,term);cc.append(cfg(term))
    obs._group_obs_term_cfgs={'actor':[ac],'critic':cc}
    obs.cfg={group:NS(enable_corruption=group=='actor',nan_check_per_term=True,nan_policy='error') for group in ('actor','critic')}
    obs.group_obs_dim={'actor':(61,),'critic':(85,)}
    obs._group_obs_concatenate={'actor':True,'critic':True}
    obs._group_obs_concatenate_dim={'actor':-1,'critic':-1}
    obs._group_obs_class_term_cfgs={'actor':[],'critic':[]}
    obs._group_obs_term_delay_buffer={'actor':{'actor_fixture':Delay()},'critic':{}}
    obs._group_obs_term_history_buffer={'actor':{'actor_fixture':History()},'critic':{}}
    obs._group_obs_class_instances={};obs._obs_buffer=None;obs.reset=lambda:None
    reward=Reward();raw.reward_manager=reward;reward.calls=0
    reward._reward_buf=torch.zeros(1);reward._episode_sums={'fixture':torch.zeros(1)}
    reward._term_names=list(slim.REWARDS)
    reward._term_cfgs=[NS(func=getattr(modules[m],n),weight=1.) for m,n in slim.REWARDS.values()]
    obs.compute(update_history=True)
    return raw,modules


def activate(raw,modules):
    with patch.object(slim,'verify_sources',return_value=(modules,{'CPU_fixture':'source IO mocked, validation real'})):
        return slim.InferenceOnly(raw,torch)


class InferenceTests(unittest.TestCase):
    def test_actor_rng_delay_history_and_retained_reset_dict_identical(self):
        arms=[]
        for reduced in (False,True):
            with torch.inference_mode():
                torch.manual_seed(73);raw,modules=fixture();obs=raw.observation_manager
                original_reset=obs._obs_buffer;full_calls=raw.critic_source_calls
                adapter=activate(raw,modules) if reduced else None
                values=[]
                for i in range(160):
                    raw.epoch=i;raw.reward_manager.compute(dt=.02)
                    output=obs.compute(update_history=True);values.append(output['actor'].clone())
                    self.assertIs(output,obs.compute())
                    self.assertEqual(set(output),{'actor'} if reduced else {'actor','critic'})
                arms.append((values,torch.random.get_rng_state(),list(obs._group_obs_term_delay_buffer['actor']['actor_fixture'].values),
                             list(obs._group_obs_term_history_buffer['actor']['actor_fixture'].values)))
                if reduced:
                    self.assertEqual(raw.critic_source_calls,full_calls)
                    self.assertEqual(set(original_reset),{'actor','critic'})
                    self.assertEqual(adapter.stats()['actor_group_calls'],160)
                    self.assertEqual(adapter.stats()['omitted_reward_calls'],160)
                    self.assertTrue(adapter.stats()['zero_reward_is_wrapper_placeholder'])
                    self.assertEqual(raw.reward_manager.calls,0)
                    self.assertEqual(raw.reward_manager._episode_sums['fixture'].item(),0)
                    self.assertEqual(raw.extras['log'],{'retained_metric':2.})
                    adapter.close();adapter.close()
                    self.assertFalse(adapter.saved);self.assertIsNone(obs._obs_buffer)
                    self.assertEqual(set(obs._group_obs_term_names),{'actor','critic'})
                    self.assertNotIn('compute_group',vars(obs));self.assertNotIn('compute',vars(raw.reward_manager))
                    raw.reward_manager.compute(.02);self.assertEqual(raw.reward_manager.calls,1)
                    self.assertEqual(set(obs.compute(update_history=True)),{'actor','critic'})
        for left,right in zip(*arms):
            if isinstance(left,list):
                self.assertEqual(len(left),len(right))
                for x,y in zip(left,right):self.assertTrue(torch.equal(x,y))
            else:self.assertTrue(torch.equal(left,right))

    def test_critic_random_state_or_unreviewed_term_rejected_without_changes(self):
        mutations=[lambda r:setattr(r.observation_manager._group_obs_term_cfgs['critic'][0],'noise',Noise()),
            lambda r:setattr(r.observation_manager._group_obs_term_cfgs['critic'][0],'delay_max_lag',1),
            lambda r:setattr(r.observation_manager._group_obs_term_cfgs['critic'][0],'history_length',1),
            lambda r:setattr(r.reward_manager._term_cfgs[0],'func',lambda:None),
            lambda r:r.reward_manager._term_names.append('stateful_unknown'),
            lambda r:setattr(r.cfg,'auto_reset',True)]
        for mutate in mutations:
            with self.subTest(mutate=mutate),torch.inference_mode():
                raw,modules=fixture();names=raw.observation_manager._group_obs_term_names;mutate(raw)
                with self.assertRaises(RuntimeError):activate(raw,modules)
                self.assertIs(raw.observation_manager._group_obs_term_names,names)
                self.assertNotIn('compute',vars(raw.reward_manager))

    def test_no_activation_outside_inference_mode(self):
        raw,modules=fixture()
        with self.assertRaisesRegex(RuntimeError,'inference_mode'):activate(raw,modules)

    def test_critic_or_reset_attempt_fails_and_cleanup_restores(self):
        with torch.inference_mode():
            raw,modules=fixture();adapter=activate(raw,modules)
            with self.assertRaises(RuntimeError):raw.observation_manager.compute_group('critic')
            with self.assertRaises(RuntimeError):raw.observation_manager.reset()
            adapter.close();raw.observation_manager.reset()
            self.assertEqual(set(raw.observation_manager.compute(update_history=True)),{'actor','critic'})

    def test_actor_exception_propagates_and_hooks_restore(self):
        with torch.inference_mode():
            raw,modules=fixture();adapter=activate(raw,modules)
            raw.observation_manager._group_obs_term_cfgs['actor'][0].func=lambda _:torch.full((1,61),float('nan'))
            try:
                with self.assertRaisesRegex(ValueError,'NaN/Inf'):raw.observation_manager.compute(update_history=True)
            finally:adapter.close()
            self.assertFalse(adapter.saved)

    def test_partial_hook_installation_rolls_back(self):
        with torch.inference_mode():
            raw,modules=fixture();original_set=slim.Bindings.set
            def fail(self,obj,name,value):
                original_set(self,obj,name,value)
                if name=='compute_group':raise ValueError('injected setter failure')
            with patch.object(slim.InferenceOnly,'set',fail),self.assertRaises(ValueError):activate(raw,modules)
            self.assertNotIn('compute_group',vars(raw.observation_manager))
            self.assertEqual(set(raw.observation_manager._group_obs_term_names),{'actor','critic'})

    def test_reviewed_source_bytes_and_checksum_rejection(self):
        modules={}
        for name,pin in slim.SOURCES.items():
            path=ROOT/('package' if pin['fixture']=='native_task.py' else 'tests/fixtures')/pin['fixture']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),pin['sha256'])
            modules[name]=NS(path=str(path))
        with patch.object(slim.importlib,'import_module',side_effect=modules.__getitem__),patch.object(slim.inspect,'getsourcefile',side_effect=lambda m:m.path):
            checked,receipts=slim.verify_sources();self.assertEqual(set(checked),set(slim.SOURCES))
            with patch.object(slim,'sha',return_value='0'*64),self.assertRaises(RuntimeError):slim.verify_sources()

    def test_original_policy_selection_and_normalization_need_only_actor(self):
        tree=ast.parse((ROOT/'tests/fixtures/rsl_mlp_model.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='MLPModel')
        method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_latent')
        namespace=dict(torch=torch,TensorDict=dict,HiddenState=object)
        exec(compile(ast.Module(body=[method],type_ignores=[]),'<original-rsl-get-latent>','exec'),namespace)
        actor=torch.randn(1,61);calls=[]
        def normalize(x):calls.append(x.clone());return (x-.25)/1.3
        policy=NS(obs_groups=['actor'],obs_normalizer=normalize)
        get_latent=namespace['get_latent']
        a=get_latent(policy,{'actor':actor,'critic':torch.randn(1,85)})
        b=get_latent(policy,{'actor':actor})
        self.assertTrue(torch.equal(a,b));self.assertEqual(len(calls),2)


if __name__=='__main__':unittest.main()
