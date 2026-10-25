"""CPU-only optimizer lifecycle tests with a fake vector environment.

No MuJoCo robot, GPU work, actual PPO learning or cloud connection is used.
"""
import contextlib
from copy import deepcopy
import io
import json
from pathlib import Path
import random
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch

PACKAGE = Path(__file__).resolve().parents[1]/'package'
sys.path.insert(0, str(PACKAGE))
import train
import budget as budget_module


class SnapshotAndSchedule(unittest.TestCase):
    def test_private_clock_stages_and_learning_rate(self):
        self.assertEqual(train.training_schedule(0)['force_weight_fraction'], .05)
        self.assertEqual(train.training_schedule(.10*train.TRAIN_SECONDS-.01)['force_weight_fraction'], .05)
        self.assertEqual(train.training_schedule(.10*train.TRAIN_SECONDS)['force_weight_fraction'], .075)
        self.assertEqual(train.training_schedule(.25*train.TRAIN_SECONDS)['force_weight_fraction'], .10)
        self.assertEqual(train.training_schedule((2./3.)*train.TRAIN_SECONDS-.01)['learning_rate'], 3e-5)
        self.assertEqual(train.training_schedule((2./3.)*train.TRAIN_SECONDS)['learning_rate'], 1e-5)
        self.assertEqual(train.training_schedule(train.TRAIN_SECONDS)['no_push_probability'], .25)
        for value in (-1., float('nan'), float('inf')):
            with self.assertRaises(RuntimeError): train.training_schedule(value)

    def test_snapshot_has_no_mutable_tensor_numpy_or_metadata_alias(self):
        state = {'actor': {'weight': torch.tensor([1.])},
                 'optimizer': {'moments': [torch.tensor([3.])]},
                 'normalizer': torch.tensor([4.])}
        meta = {'policy_updates': train.retained_policy_updates(7),
                'common_step_counter': 144408, 'campaign_completed_updates': 7,
                'numpy': np.array([8])}
        saved = train.make_committed_snapshot(state, meta, 7, torch)
        state['actor']['weight'].add_(20)
        state['optimizer']['moments'][0].add_(20)
        state['normalizer'].add_(20)
        meta['numpy'][0] = 99
        self.assertEqual(saved['actor']['weight'].item(), 1.)
        self.assertEqual(saved['optimizer']['moments'][0].item(), 3.)
        self.assertEqual(saved['normalizer'].item(), 4.)
        self.assertEqual(saved['infos']['stage12_final6h']['numpy'][0], 8)

    def test_atomic_checkpoint_receipt_is_immutable_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            meta = dict(policy_updates=train.retained_policy_updates(6), common_step_counter=24,
                        campaign_completed_updates=6, campaign_attempted_updates=6,
                        package_identity='fixture')
            saved = train.make_committed_snapshot({'actor': torch.tensor([6.])}, meta, 6, torch)
            receipt = train.persist_snapshot(root, saved, 6, torch)
            self.assertEqual(train.persist_snapshot(root, saved, 6, torch), receipt)
            Path(receipt['path']).write_bytes(b'corrupted')
            with self.assertRaisesRegex(RuntimeError, 'Existing checkpoint differs'):
                train.persist_snapshot(root, saved, 6, torch)

    def test_chunked_json_rows_are_complete_and_bounded(self):
        with tempfile.TemporaryDirectory() as temporary:
            log = train.ChunkedLog(temporary, 'rows', limit=80)
            for i in range(20): log.row({'number': i, 'value': 'abcdef'})
            paths = sorted(Path(temporary).glob('rows-*.jsonl'))
            self.assertGreater(len(paths), 1)
            rows = []
            for path in paths:
                self.assertLessEqual(path.stat().st_size, 80)
                rows.extend(json.loads(line) for line in path.read_text().splitlines())
            self.assertEqual([row['number'] for row in rows], list(range(20)))
            with self.assertRaisesRegex(RuntimeError, 'single log record'):
                log.row({'large': 'x'*200})


class TrainingFlow(unittest.TestCase):
    def run_fixture(self, phase='formal', failure=None, elapsed=10., target=7):
        """A real CPU checkpoint and fake runner exercise the production entrypoint."""
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            out = root/'run'
            (out/'checkpoints').mkdir(parents=True)
            package = root/'package'
            package.mkdir()
            (package/'reward-spec.json').write_text('{}')
            now = [1000.+elapsed]
            ledger = budget_module.new(1000.)
            if phase == 'formal':
                ledger.update(attempted_updates=5, completed_updates=5)
                ledger['phase_attempts']['smoke'] = 5
                ledger['phase_completed']['smoke'] = 5
            (root/'budget.json').write_text(json.dumps(ledger))
            events = []
            observed = {'physical': 0, 'controls': 0, 'updates': 0, 'reset': 0,
                        'normalizer_starts': [], 'stages': [], 'learning_rates': []}
            snapshot_position = ledger['completed_updates']
            original_state = {'actor': {'weight': torch.tensor([1.]), 'normalizer': torch.zeros(1)},
                              'critic': {'weight': torch.tensor([2.]), 'normalizer': torch.zeros(1)},
                              'optimizer': {'moments': torch.tensor([3.])}}
            meta = dict(package_identity=train.ORIGIN_ID if phase == 'smoke' else 'fixture-package',
                        policy_updates=train.retained_policy_updates(snapshot_position),
                        origin_checkpoint_sha256=train.MODEL_SHA,
                        common_step_counter=144240, sim_step_counter=1442400, num_envs=4096,
                        torch_rng=torch.get_rng_state(), cuda_rng=[], numpy_rng=np.random.get_state(),
                        python_rng=random.getstate())
            if failure == 'lineage': meta['package_identity'] = 'wrong-parent'
            parent = root/'parent.pt'
            torch.save(dict(original_state, infos={'stage12_native': meta}), parent)
            parent_receipt = dict(path=str(parent), sha256=train.sha(parent), bytes=parent.stat().st_size,
                                  completed_updates=snapshot_position,
                                  policy_updates=train.retained_policy_updates(snapshot_position),
                                  parent_checkpoint=phase == 'smoke')

            def input_checkpoint():
                latest = out/'latest.json'
                return train.read(latest) if latest.exists() else parent_receipt

            def module(**kwargs): return types.SimpleNamespace(**kwargs)
            def observations(n): return {'actor': torch.zeros(n, 61), 'critic': torch.zeros(n, 85)}
            class Policy:
                def __init__(self, alg, key):
                    self.alg, self.key = alg, key
                    self.obs_normalizer = module()
                def reset(self): events.append(self.key+'_reset')
                def state_dict(self): return self.alg.state[self.key]
                def parameters(self): return list(self.alg.state[self.key].values())
            class Algorithm:
                def __init__(self):
                    self.state = train.clone_to_cpu(original_state, torch)
                    self.actor, self.critic = Policy(self, 'actor'), Policy(self, 'critic')
                    self.optimizer = module(param_groups=[{'lr': 3e-5}])
                    self.learning_rate = 3e-5
                def load(self, saved, options, strict):
                    self_outer.assertIsNone(options)
                    self_outer.assertTrue(strict)
                    self.state = train.clone_to_cpu({key: saved[key] for key in original_state}, torch)
                    events.append('strict_load')
                def save(self): return self.state
                def update(self):
                    observed['updates'] += 1
                    observed['learning_rates'].append(self.learning_rate)
                    self.state['actor']['weight'].add_(1.)
                    self.state['optimizer']['moments'].add_(1.)
                    if failure == 'optimizer' and observed['updates'] == 2:
                        self.state['actor']['weight'].add_(100.)
                        raise RuntimeError('fixture optimizer interruption')
                    return {'value': .1, 'surrogate': .2}
            class Raw:
                def __init__(self, cfg, **kwargs):
                    self.cfg, self.num_envs = cfg, cfg.num_envs
                    self.step_dt, self.physics_dt, self.max_episode_length = .02, .002, 3000
                    self.common_step_counter = self._sim_step_counter = 0
                    self.reset_terminated = torch.zeros(self.num_envs, dtype=torch.bool)
                    self.termination_manager = module(get_term=lambda key: torch.zeros(self.num_envs, dtype=torch.bool))
                    self.sim = module(step=self.integrate)
                    self.bb = module(level=torch.zeros(self.num_envs), hold=torch.zeros(self.num_envs))
                def integrate(self):
                    observed['physical'] += 1
                    self._sim_step_counter += 1
                    if failure in ('deadline_physics', 'deadline_sigterm') and observed['physical'] == 245:
                        now[0] = ledger['train_stop_utc']
                        if failure == 'deadline_sigterm':
                            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
                    if failure == 'runtime' and observed['physical'] == 245:
                        raise RuntimeError('fixture integration interruption')
                    if failure == 'stop' and observed['physical'] == 245:
                        (root/'STOP.json').write_text('{}')
                def reset(self, seed):
                    observed['reset'] += 1
                    self_outer.assertEqual(self.common_step_counter, 144240)
                    self_outer.assertEqual(self._sim_step_counter, 1442400)
                def close(self): events.append('env_close')
            class Wrapper:
                def __init__(self, raw): self.raw = raw
                def get_observations(self): return observations(self.raw.num_envs)
                def step(self, actions):
                    for _ in range(10): self.raw.sim.step()
                    observed['controls'] += 1
                    self.raw.common_step_counter += 1
                    return observations(self.raw.num_envs), torch.zeros(self.raw.num_envs), \
                        torch.zeros(self.raw.num_envs, dtype=torch.bool), {}
                def close(self): self.raw.close()
            class Pushes:
                def __init__(self, raw, torch_module, seed, initial_fraction, no_push_probability):
                    self.raw, self.original = raw, raw.sim.step
                    self.physics_calls = self.control_calls = 0
                    self_outer.assertEqual(no_push_probability, .25)
                    self.seed = seed
                    raw.sim.step = self.step
                    events.append('push_install')
                def step(self):
                    value = self.original()
                    self.physics_calls += 1
                    return value
                def begin_control(self): pass
                def end_control(self, dones): self.control_calls += 1
                def set_stage_fraction(self, fraction): observed['stages'].append(fraction)
                def state_dict(self): return {'force_seed': self.seed, 'generator_state': torch.tensor([1])}
                def snapshot(self):
                    complete = self.physics_calls >= 1050 and failure != 'no_force'
                    return dict(completed_pulses=int(complete), applied_world_physics_steps=50 if complete else 0,
                                owned_force_cleared=True, external_wrench_zero=True,
                                integration_errors=0, cleanup_errors=0)
                def take_rollout(self):
                    return dict(self.snapshot(), completed_episode_receipts=[{'fixture': True}])
                def close(self):
                    events.append('push_close')
                    self.raw.sim.step = self.original
            class Monitor:
                def __init__(self, raw, torch_module, folder):
                    self.raw, self.original = raw, raw.sim.step
                    self.active, self.steps = False, 0
                    self.joint_stats = module(after_control_step=lambda *args: None,
                        take_rollout=lambda: {'any_joint_violation_fraction': 0.})
                    raw.sim.step = self.step
                    events.append('monitor_install')
                def step(self):
                    self_outer.assertTrue(self.active)
                    result = self.original()
                    self.steps += 1
                    return result
                def close(self):
                    events.append('monitor_close')
                    self.raw.sim.step = self.original
            class Runner:
                def __init__(self, env, config, **kwargs):
                    self.env, self.config, self.alg = env, config, Algorithm()
                    self.logger = module(ep_extras=[], git_status_repos=[],
                        init_logging_writer=lambda: None, stop_logging_writer=lambda: None,
                        writer=module(close=lambda: None), log=lambda **kwargs: print('upstream fixture log'))
                    observed['agent'] = deepcopy(config)
                def learn(self, remaining, init_at_random_ep_len):
                    self_outer.assertFalse(init_at_random_ep_len)
                    for _ in range(remaining):
                        for _ in range(24):
                            self.alg.state['actor']['normalizer'].add_(1.)
                            self.env.step(torch.zeros(self.env.raw.num_envs, 14))
                        loss = self.alg.update()
                        self.logger.log(collect_time=.5, learn_time=.1, loss_dict=loss)
                def export_policy_to_onnx(self, folder, name):
                    (Path(folder)/name).write_bytes(b'fixture ONNX')
                    events.append('export')
            def values(raw):
                if failure == 'reward': raise RuntimeError('fixture positive reward')
                return {'head_roll_margin': -torch.ones(raw.num_envs),
                        'other_joint_margin': -torch.ones(raw.num_envs)}
            def shape(size):
                return module(type=module(tensor_type=module(shape=module(dim=[module(dim_value=1), module(dim_value=size)]))))
            def configure(normalizer, start, torch_module): observed['normalizer_starts'].append(start)
            modules = {
                'warp': module(config=module()),
                'mjlab.envs': module(ManagerBasedRlEnv=Raw),
                'mjlab.rl': module(MjlabOnPolicyRunner=Runner, RslRlVecEnvWrapper=Wrapper),
                'mjlab.utils.torch': module(configure_torch_backends=lambda: None),
                'mjlab_microduck.tasks.microduck_basketball_env_cfg': module(MicroduckBasketballRlCfg=object()),
                'mjlab_microduck.tasks': module(mdp=module(_bb_state=lambda raw: raw.bb)),
                'native_task': module(head_scale=lambda raw: 1., lower_unassisted=lambda raw: torch.ones(raw.num_envs),
                                      geometric_joint_stable=lambda raw: torch.ones(raw.num_envs)),
                'robust_task': module(make_cfg=lambda n, seed: module(num_envs=n, decimation=10,
                                      episode_length_s=60., curriculum={}, events={}),
                                      validate_cost_signs=values, verify_runtime_reward_order=lambda raw: {'checked': True}),
                'normalization': module(configure_normalizer_updates=configure,
                                        equal_state=lambda actual, expected, torch_module, top_level: all(
                                            torch.equal(actual[key][name], expected[key][name])
                                            for key in actual for name in actual[key])),
                'train_motor_monitor': module(TrainingMotorMonitor=Monitor),
                'force_runtime': module(VectorPushes=Pushes),
                'statistics_runtime_check': module(check_statistics_runtime=lambda *args: {'passed': True}),
                'onnx': module(load=lambda path: module(graph=module(input=[shape(61)], output=[shape(14)])),
                               checker=module(check_model=lambda model: None)),
                'mjlab_microduck.onnx_policy_contract': module(bake_action_clip=lambda *args: None),
            }
            self_outer = self
            old_path = list(sys.path)
            try:
                with contextlib.ExitStack() as stack:
                    stack.enter_context(patch.dict(sys.modules, modules))
                    for name, value in [('SLOT', root), ('HERE', package), ('SOURCE', root), ('TRAIN', root),
                                        ('verify_package', lambda: 'fixture-package'),
                                        ('verify_runtime', lambda: {'fixture': True}),
                                        ('input_checkpoint', input_checkpoint),
                                        ('TARGETS', {'smoke': 5, 'formal': target}),
                                        ('asdict', lambda item: dict(algorithm={}, num_steps_per_env=24, clip_actions=1.))]:
                        stack.enter_context(patch.object(train, name, value))
                    stack.enter_context(patch.object(train.time, 'time', side_effect=lambda: now[0]))
                    stack.enter_context(patch.object(torch.cuda, 'is_available', return_value=True))
                    stack.enter_context(patch.object(torch.cuda, 'get_device_name', return_value='NVIDIA RTX 5090 fixture'))
                    stack.enter_context(patch.object(torch.cuda, 'get_rng_state_all', return_value=[]))
                    stack.enter_context(patch.object(torch.cuda, 'set_rng_state_all'))
                    stack.enter_context(patch.object(torch, 'set_num_threads'))
                    stack.enter_context(patch.object(sys, 'argv', ['train', '--phase', phase, '--session', '1']))
                    stream = stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    code = train.main()
                    stdout = stream.getvalue()
            finally:
                sys.path[:] = old_path
            result = json.loads((out/'session-001/result.json').read_text())
            final_budget = json.loads((root/'budget.json').read_text())
            latest = input_checkpoint()
            saved = torch.load(latest['path'], weights_only=False)
            candidate_names = sorted(path.name for path in (out/'candidates').glob('*.json'))
            logs = {path.name: path.read_text() for path in (out/'session-001').glob('*.jsonl')}
            return dict(code=code, result=result, budget=final_budget, saved=saved, events=events,
                        observed=observed, candidates=candidate_names, logs=logs, stdout=stdout)

    def test_smoke_five_updates_force_witness_and_original_onnx_export(self):
        value = self.run_fixture(phase='smoke')
        self.assertEqual(value['code'], 0, value['result'])
        self.assertEqual(value['budget']['completed_updates'], 5)
        self.assertEqual(value['observed']['physical'], 1200)
        self.assertTrue(value['result']['smoke_export']['passed'])
        self.assertGreater(value['result']['smoke_disturbance_witness']['completed_pulses'], 0)
        self.assertIn('export', value['events'])
        self.assertEqual(value['observed']['normalizer_starts'], [55, 76])
        self.assertEqual(value['candidates'], [])

    def test_formal_commit_retains_mature_counter_and_normalizer_without_cloud(self):
        value = self.run_fixture()
        self.assertEqual(value['code'], 0, value['result'])
        self.assertEqual(value['budget']['completed_updates'], 7)
        self.assertEqual(value['observed']['physical'], 480)
        meta = value['saved']['infos']['stage12_final6h']
        self.assertEqual(meta['common_step_counter'], 144240+48)
        self.assertEqual(meta['policy_updates'], 6017)
        self.assertTrue(value['result']['formal_training_ready'])
        self.assertEqual(value['observed']['agent']['algorithm']['clip_param'], .15)
        self.assertEqual(value['observed']['agent']['algorithm']['entropy_coef'], .003)
        self.assertEqual(value['candidates'], ['candidate-final.json'])
        self.assertLess(value['events'].index('push_install'), value['events'].index('monitor_install'))
        self.assertLess(value['events'].index('monitor_close'), value['events'].index('push_close'))
        self.assertLess(value['events'].index('push_close'), value['events'].index('env_close'))

    def test_deadline_mid_rollout_saves_previous_complete_state_and_charges_attempt(self):
        value = self.run_fixture(failure='deadline_physics')
        self.assertEqual(value['code'], 0, value['result'])
        self.assertEqual(value['result']['reason'], 'train_window_complete')
        self.assertEqual(value['observed']['physical'], 245)
        self.assertEqual(value['budget']['completed_updates'], 6)
        self.assertEqual(value['budget']['attempted_updates'], 7)
        self.assertIsNotNone(value['budget']['pending'])
        self.assertEqual(value['saved']['actor']['normalizer'].item(), 24.)
        self.assertEqual(value['saved']['actor']['weight'].item(), 2.)
        self.assertEqual(value['saved']['infos']['stage12_final6h']['common_step_counter'], 144264)
        self.assertIn('candidate-final.json', value['candidates'])

    def test_partial_optimizer_is_not_committed_or_serialized(self):
        value = self.run_fixture(failure='optimizer')
        self.assertEqual(value['code'], 2)
        self.assertIn('optimizer interruption', value['result']['error'])
        self.assertEqual(value['budget']['completed_updates'], 6)
        self.assertEqual(value['budget']['attempted_updates'], 7)
        self.assertIsNotNone(value['budget']['pending'])
        self.assertEqual(value['saved']['actor']['weight'].item(), 2.)
        self.assertEqual(value['saved']['actor']['normalizer'].item(), 24.)
        self.assertEqual(value['candidates'], [])

    def test_supervisor_sigterm_at_boundary_keeps_normal_deadline_semantics(self):
        value = self.run_fixture(failure='deadline_sigterm')
        self.assertEqual(value['code'], 0, value['result'])
        self.assertEqual(value['result']['reason'], 'train_window_complete')
        self.assertEqual(value['observed']['physical'], 245)
        self.assertEqual(value['budget']['attempted_updates'], 7)
        self.assertIsNotNone(value['budget']['pending'])

    def test_smoke_without_real_force_witness_cannot_authorize_formal_training(self):
        value = self.run_fixture(phase='smoke', failure='no_force')
        self.assertEqual(value['code'], 2)
        self.assertIn('Smoke did not witness', value['result']['error'])
        self.assertEqual(value['budget']['completed_updates'], 5)
        self.assertNotIn('export', value['events'])

    def test_wrong_parent_lineage_prevents_any_rollout_or_update(self):
        value = self.run_fixture(failure='lineage')
        self.assertEqual(value['code'], 2)
        self.assertIn('lineage differs', value['result']['error'])
        self.assertEqual(value['observed']['physical'], 0)
        self.assertEqual(value['observed']['updates'], 0)

    def test_user_stop_stops_without_marking_training_window_complete(self):
        value = self.run_fixture(failure='stop')
        self.assertEqual(value['code'], 3)
        self.assertEqual(value['result']['status'], 'TRAINING_STOPPED_PRESERVE')
        self.assertEqual(value['result']['reason'], 'user_stop')
        self.assertEqual(value['observed']['physical'], 245)
        self.assertEqual(value['saved']['actor']['normalizer'].item(), 24.)
        self.assertEqual(value['candidates'], [])

    def test_runtime_error_preserves_first_commit_and_closes_hooks(self):
        value = self.run_fixture(failure='runtime')
        self.assertEqual(value['code'], 2)
        self.assertEqual(value['saved']['actor']['weight'].item(), 2.)
        self.assertEqual(value['events'][-3:], ['monitor_close', 'push_close', 'env_close'])

    def test_positive_or_nonfinite_reward_guard_prevents_first_ppo(self):
        value = self.run_fixture(failure='reward')
        self.assertEqual(value['code'], 2)
        self.assertEqual(value['observed']['updates'], 0)
        self.assertEqual(value['observed']['physical'], 0)
        self.assertEqual(value['budget']['attempted_updates'], 5)

    def test_late_stage_lr_candidates_and_bounded_logs(self):
        value = self.run_fixture(elapsed=13000.)
        self.assertEqual(value['code'], 0, value['result'])
        self.assertTrue(all(rate == 1e-5 for rate in value['observed']['learning_rates']))
        self.assertTrue(all(fraction == .10 for fraction in value['observed']['stages']))
        self.assertEqual(value['candidates'], ['candidate-01.json', 'candidate-02.json', 'candidate-final.json'])
        self.assertTrue(any(name.startswith('metrics-') for name in value['logs']))
        self.assertTrue(any(name.startswith('force-episodes-') for name in value['logs']))
        self.assertNotIn('completed_episode_receipts', value['stdout'])
        self.assertNotIn('joint_rollout', value['stdout'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
