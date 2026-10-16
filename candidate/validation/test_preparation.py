import contextlib
import copy
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from typing import Generator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage12 import contracts as c
from stage12.vec_adapter import cached_observations, make_wrapper_class, snapshot_observations

class Tensor:
    def __init__(self, shape, value=0):
        self.shape = shape
        self.value = value
    def clone(self):
        return Tensor(self.shape, self.value)

class TD(dict):
    def __init__(self, data, batch_size):
        super().__init__(data)
        self.batch_size = batch_size
    @property
    def batch_dims(self):
        return len(self.batch_size)
    def unbind(self, dim):
        assert dim == 0
        return tuple(TD({k: Tensor(v.shape[1:], v.value) for k, v in self.items()},
                        self.batch_size[1:]) for _ in range(self.batch_size[0]))

# Execute the exact reviewed upstream __iter__ body on the minimal fixture.
# This is a protocol regression test, not an import of the full TensorDict stack.
iteration = json.loads((ROOT/'reviewed-import-sources/tensordict-iteration-review.json').read_bytes())
namespace = {'Generator': Generator}
exec(compile(iteration['iteration_source'], iteration['source_url'], 'exec'), namespace)
TD.__iter__ = namespace['__iter__']

class Cache:
    num_envs = 2
    cfg = type('Cfg', (), {'auto_reset': True})()
    def __init__(self):
        self.obs_buf = {'actor': Tensor((2,61)), 'critic': Tensor((2,85))}
    @property
    def observation_manager(self):
        raise AssertionError('Cached reads must not access the observation manager')

class PreparationTests(unittest.TestCase):
    def test_old_checkpoint_cannot_resume_or_warm_start(self):
        for mode in ['resume','warm_start','actor_only','fresh']:
            with self.assertRaises(RuntimeError):
                c.checkpoint_policy(mode, c.CHECKPOINT)
        with self.assertRaises(RuntimeError):
            c.checkpoint_policy('resume')
    def test_fresh_plan_does_not_load_or_initialize(self):
        p=c.checkpoint_policy()
        self.assertFalse(p['executed'])
        self.assertEqual(p['torch_load_calls'],0)
        self.assertEqual(p['old_checkpoint']['sha256'],c.CHECKPOINT_SHA)
    def test_budget_uses_mandatory_replay_and_all_owners(self):
        p=c.scaling_plan()
        self.assertEqual(p['per_macro_integrations_accept64'],32+64+64)
        self.assertEqual(p['per_macro_integrations_accept128'],32+64+128+128)
        self.assertEqual(p['draft_integrations_total_max'],(1+2)*(1+2)*10*352)
        self.assertTrue(all(x==0 for x in p['authorized_budget'].values()))
        self.assertTrue(all(x['setup_reset_forward_budget'] is None for x in p['draft_local_pilot']))
    def test_preparation_guard_refuses_before_any_callback(self):
        for fn in [c.create_environment,c.PreparedRunner,c.PreparedRunner.learn,c.PreparedRunner.load]:
            with self.assertRaisesRegex(RuntimeError,'STAGE12_NOT_RELEASED'):
                fn(lambda: self.fail('No callback may run'))
    def test_repeated_observation_reads_never_recompute_and_clone(self):
        env=Cache()
        a=cached_observations(env,TD)
        a['actor'].value=100
        self.assertEqual(cached_observations(env,TD)['actor'].value,0)
        self.assertEqual(env.obs_buf['actor'].value,0)
    def test_current_committed_cache_is_read_after_replacement(self):
        env=Cache()
        cached_observations(env,TD)
        env.obs_buf={'actor':Tensor((2,61),9),'critic':Tensor((2,85),10)}
        self.assertEqual(cached_observations(env,TD)['actor'].value,9)
    def test_wrong_actor_shape_and_missing_group_rejected(self):
        env=Cache();env.obs_buf['actor']=Tensor((2,60))
        with self.assertRaisesRegex(RuntimeError,'Actor'):cached_observations(env,TD)
        env=Cache();del env.obs_buf['critic']
        with self.assertRaisesRegex(RuntimeError,'groups'):cached_observations(env,TD)
    def test_manual_reset_task_rejected(self):
        env=Cache();env.cfg=type('Cfg',(),{'auto_reset':False})()
        with self.assertRaisesRegex(RuntimeError,'auto-reset'):cached_observations(env,TD)
    def test_step_reset_delegate_once_and_snapshot_reused_buffers(self):
        class Base:
            num_envs = 2
            steps = resets = 0
            def step(self, actions):
                self.steps += 1
                self.env.obs_buf['actor'].value += 3
                return TD(self.env.obs_buf, [2]), 'reward', 'done', 'extras'
            def reset(self):
                self.resets += 1
                self.env.obs_buf['actor'].value = 0
                return TD(self.env.obs_buf, [2]), 'reset-extras'
            @property
            def unwrapped(self):return self.env
        cls=make_wrapper_class(Base,TD);obj=cls();obj.env=Cache()
        a, reward, done, extras = obj.step(None)
        b, _, _, _ = obj.step(None)
        self.assertEqual(a['actor'].value, 3)
        self.assertEqual(b['actor'].value, 6)
        self.assertEqual((reward, done, extras), ('reward', 'done', 'extras'))
        r, extra = obj.reset()
        self.assertEqual((r['actor'].value, extra, obj.steps, obj.resets), (0, 'reset-extras', 2, 1))
        self.assertEqual(b['actor'].value, 6)
        self.assertEqual(obj.get_observations().batch_size,[2])
    def test_upstream_batch_iteration_reproduces_v6_error_but_key_access_succeeds(self):
        obs=TD(Cache().obs_buf,[2])
        with self.assertRaisesRegex(TypeError,'unhashable'):
            set(obs)  # v6's operation; reproduces the host traceback.
        self.assertEqual(set(obs.keys()),{'actor','critic'})
        result=snapshot_observations(obs,2,TD)
        self.assertEqual(result['actor'].shape,(2,61))
        self.assertIsNot(result['actor'],obs['actor'])
    def test_tensordict_cache_and_shapes_across_small_batch_sizes(self):
        for n in [1,2,4]:
            env=Cache();env.num_envs=n
            env.obs_buf=TD({'actor':Tensor((n,61),5),'critic':Tensor((n,85),7)},[n])
            result=cached_observations(env,TD)
            result['actor'].value=99
            self.assertEqual((env.obs_buf['actor'].value,result['critic'].value,result.batch_size),(5,7,[n]))
            env.obs_buf['critic']=Tensor((n,84))
            with self.assertRaisesRegex(RuntimeError,'Critic'):
                cached_observations(env,TD)
    def test_batch_iterable_missing_extra_and_nested_groups_are_rejected(self):
        cases=[{'actor':Tensor((2,61))},
               dict(Cache().obs_buf,extra=Tensor((2,1))),
               {('nested','actor'):Tensor((2,61)),'critic':Tensor((2,85))}]
        for data in cases:
            with self.assertRaisesRegex(RuntimeError,'groups'):
                snapshot_observations(TD(data,[2]),2,TD)
    def test_new_code_has_no_training_or_deserializer_call(self):
        import ast
        for p in (ROOT/'stage12').glob('*.py'):
            tree=ast.parse(p.read_text())
            for n in ast.walk(tree):
                if isinstance(n,ast.Call):
                    name=ast.unparse(n.func)
                    self.assertNotIn(name,{'torch.load','torch.save','runner.learn','runner.load','PPO'},str(p))
    def test_vendored_g11_files_identical(self):
        actual=ROOT/'baseline/g11'
        manifest=json.loads((actual/'payload-inventory.json').read_bytes())
        for name,row in manifest.items():
            self.assertEqual(hashlib.sha256((actual/name).read_bytes()).hexdigest(),row['sha256'],name)
    def test_mjlab_registry_has_no_environment_class_slot(self):
        import ast
        t=ast.parse((ROOT/'baseline/registry.py').read_text())
        n=next(n for n in t.body if isinstance(n,ast.FunctionDef) and n.name=='register_mjlab_task')
        self.assertEqual([a.arg for a in n.args.args],['task_id','env_cfg','play_env_cfg','rl_cfg','runner_cls'])

if __name__=='__main__':
    unittest.main(verbosity=2)
