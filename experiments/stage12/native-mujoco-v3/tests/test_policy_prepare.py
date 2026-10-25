"""Actual frozen RSL-RL actor, state preservation and startup ordering checks."""
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import RNNModel

from support import loop_sim
from fingerprint import fingerprint
from policy_prepare import prepare_reference, WARMUP_CALLS
from telemetry import Timing

CHECKPOINT = Path(os.environ.get('MICRODUCK_TEST_CHECKPOINT','/nonexistent/checkpoint.pt'))
MODEL_SHA = 'cf7ce9297672943687019afe071d94918d3eb2620c18a1e8fe689512f8c08111'


class PolicyPreparation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT.is_file():
            raise RuntimeError('Set MICRODUCK_TEST_CHECKPOINT to the frozen final6h_005304.pt')
        assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() == MODEL_SHA
        torch.set_num_threads(1)
        cls.saved = torch.load(CHECKPOINT,map_location='cpu',weights_only=False)['actor_state_dict']

    def policy(self):
        result = RNNModel(TensorDict({'actor':torch.zeros(1,61)},batch_size=[1]),
            {'actor':['actor']},'actor',14,hidden_dims=[512,256,128],obs_normalization=True,
            distribution_cfg={'class_name':'rsl_rl.modules:GaussianDistribution',
                              'init_std':1.0,'std_type':'scalar'},
            rnn_hidden_dim=256,rnn_num_layers=1)
        result.load_state_dict(self.saved,strict=True)
        result.eval()
        result.reset()
        return result

    def observations(self, policy, step):
        # Independent deterministic test sequence; no plant/Orin claims.
        value = torch.sin(torch.arange(61,dtype=torch.float32)*.17+step*.031).reshape(1,61)
        actor = policy.obs_normalizer._mean + policy.obs_normalizer._std*value
        return TensorDict({'actor':actor,'critic':torch.zeros(1,85)},batch_size=[1])

    def test_actual_frozen_actor_6000_actions_and_lstm_states_are_bit_identical(self):
        policy = self.policy()
        untouched = deepcopy(policy)
        initial = self.observations(policy,0)
        before = fingerprint(policy.state_dict())
        rng = torch.get_rng_state().clone()
        py_rng = random.getstate()
        np_rng = np.random.get_state()
        with torch.jit.optimized_execution(False):
            receipt = prepare_reference(policy, initial, torch)
            self.assertEqual(receipt['warmup_calls'], WARMUP_CALLS)
            self.assertTrue(receipt['same_actor_instance'])
            self.assertIsNone(policy.get_hidden_state())
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
            self.assertEqual(random.getstate(),py_rng)
            self.assertTrue(np.array_equal(np.random.get_state()[1],np_rng[1]))
            self.assertEqual(fingerprint(policy.state_dict()),before)
            with torch.inference_mode():
                for case in range(2):
                    policy.reset();untouched.reset()
                    for step in range(3000):
                        obs=self.observations(policy,case*3000+step)
                        action=policy(obs,stochastic_output=False)
                        expected=untouched(obs,stochastic_output=False)
                        self.assertTrue(torch.equal(action,expected),(case,step,'action'))
                        self.assertTrue(all(torch.equal(a,b) for a,b in zip(
                            policy.get_hidden_state(),untouched.get_hidden_state())),(case,step,'LSTM'))
        self.assertEqual(fingerprint(policy.state_dict()),before)

    def test_interruption_restores_rng_and_fresh_lstm_state(self):
        policy=self.policy();obs=self.observations(policy,0)
        before=fingerprint(policy.state_dict());rng=torch.get_rng_state().clone();calls=[]
        def boundary():
            calls.append(1)
            torch.rand(3)
            if len(calls)==2:raise InterruptedError('controlled stop')
        with self.assertRaisesRegex(InterruptedError,'controlled stop'):
            prepare_reference(policy,obs,torch,boundary)
        self.assertIsNone(policy.get_hidden_state())
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(fingerprint(policy.state_dict()),before)

    def test_nonfresh_or_training_policy_is_rejected(self):
        policy=self.policy();obs=self.observations(policy,0)
        with torch.inference_mode():policy(obs,stochastic_output=False)
        with self.assertRaisesRegex(RuntimeError,'fresh'):
            prepare_reference(policy,obs,torch)
        policy.reset();policy.train()
        with self.assertRaisesRegex(RuntimeError,'frozen'):
            prepare_reference(policy,obs,torch)

    def test_remote_preparation_consumes_no_requests_or_control_ticks(self):
        policy=self.policy();obs=self.observations(policy,0)
        untouched=deepcopy(policy)
        with tempfile.TemporaryDirectory() as d:
            client=SimpleNamespace(seq=0,ipc_ms=[0.],ros_ms=[0.],inference_ms=[0.])
            timing=Timing(d)
            stats=dict(compared_actions=0,max_action_absolute_error=0.,timing=timing)
            remote=loop_sim.RemotePolicy(policy,client,torch,0,stats)
            remote.reset()
            with self.assertRaisesRegex(RuntimeError,'prepared'):
                remote(obs)
            receipt=remote.prepare(obs,lambda:None)
            self.assertEqual(client.seq,0)
            self.assertEqual(timing.rows,[])
            self.assertIsNone(timing.epoch)
            self.assertEqual(stats['compared_actions'],0)
            self.assertEqual(receipt['physics_steps'],0)
            self.assertIs(remote.reference,policy)
            with self.assertRaises(RuntimeError):remote.prepare(obs,lambda:None)
            def step_with_reference(values,reference):
                reference();client.seq+=1
                return untouched(obs,stochastic_output=False)[0].tolist()
            client.step_with_reference=step_with_reference
            with torch.inference_mode():
                action=remote(obs)
            self.assertEqual(client.seq,1)
            self.assertEqual(stats['compared_actions'],1)
            self.assertEqual(stats['max_action_absolute_error'],0.)
            self.assertEqual(timing.current['step'],1)
            self.assertEqual(tuple(action.shape),(1,14))
