"""Exercise native layout and reference warmup with real frozen checkpoint bytes.

The layout fixture comes from this runtime; the laptop additionally checks the
previously accepted export contract. This test makes no ONNX/Orin parity claim.
"""
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
from support import loop_sim
from fingerprint import fingerprint
import native_evaluate as evaluate
from telemetry import Timing

CHECKPOINT=Path(os.environ.get('MICRODUCK_TEST_CHECKPOINT','/nonexistent/checkpoint.pt'))

@unittest.skipUnless(CHECKPOINT.is_file(),'Pinned checkpoint and original project runtime required')
class FrozenPolicy(unittest.TestCase):
    def test_warmup_preserves_real_recurrent_policy_rng_and_cpu_action(self):
        self.assertEqual(loop_sim.old.sha(CHECKPOINT),loop_sim.MODEL_SHA)
        with patch.object(evaluate,'verify_runtime',return_value={'test_runtime':'CPU'}):
            base=evaluate.load_runtime()
        torch=base.torch
        cfg=base.make_cfg(1,2407);cfg.sim.mujoco.timestep=.005;cfg.decimation=4;cfg.auto_reset=False
        raw=base.ManagerBasedRlEnv(cfg,device='cpu',render_mode=None)
        env=base.RslRlVecEnvWrapper(raw)
        try:
            agent=asdict(deepcopy(base.MicroduckBasketballRlCfg))
            agent.update(seed=2407,logger='tensorboard',upload_model=False)
            saved=torch.load(CHECKPOINT,map_location='cpu',weights_only=False)
            state=saved['actor_state_dict']
            normal={k[len('obs_normalizer.'):]:v for k,v in state.items() if k.startswith('obs_normalizer.')}
            from mjlab.rl.exporter_utils import get_base_metadata
            terms=[];offset=0
            for name,dim in zip(raw.observation_manager.active_terms['actor'],raw.observation_manager.group_obs_term_dim['actor']):
                width=math.prod(dim);term=raw.observation_manager.get_term_cfg('actor',name)
                terms.append(dict(name=name,start=offset,stop=offset+width,
                    function=term.func.__module__+'.'+term.func.__name__,scale=term.scale,clip=term.clip))
                offset+=width
            joint=raw.action_manager.get_term('joint_pos')
            def values(x):return x[0].tolist() if isinstance(x,torch.Tensor) else x
            contract=dict(action_clip=None,action_filter=None,runtime_metadata=get_base_metadata(raw,run_path='fixture'),
                action_offset=values(joint._offset),action_scale=values(joint._scale),
                actor_terms=json.loads(json.dumps(terms)),actor_state_sha256=fingerprint(state),normalizer_sha256=fingerprint(normal))
            with tempfile.TemporaryDirectory() as d:
                client=NS(seq=0,ipc_ms=[0.],ros_ms=[0.],inference_ms=[0.])
                stats=dict(compared_actions=0,max_action_absolute_error=0.,sessions=[],timing=Timing(d))
                context=loop_sim.linked_context(base,client,0,stats,contract)
                runner=context.MjlabOnPolicyRunner(env,agent,log_dir=None,device='cpu')
                runner.alg.load(saved,{'actor':True,'critic':True,'optimizer':False,'iteration':False},strict=True)
                rng=torch.get_rng_state().clone()
                policy=runner.get_inference_policy(device='cpu')
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                self.assertEqual(fingerprint(policy.state_dict()),contract['actor_state_sha256'])
                expected=deepcopy(policy.reference)
                expected.reset()
                def remote_action(obs):
                    client.seq+=1
                    td=base.TensorDict({'actor':torch.tensor([obs])},batch_size=[1])
                    return expected(td,stochastic_output=False)[0].tolist()
                client.step=remote_action
                client.step_with_reference=lambda obs,ref:(ref(),client.step(obs))[1]
                with torch.inference_mode():
                    observations,_=raw.reset(seed=2407)
                    td=base.TensorDict(observations,batch_size=[1])
                    policy.reset()
                    policy.prepare(td,lambda:None)
                    result=policy(td,stochastic_output=False)
                self.assertEqual(tuple(result.shape),(1,14))
                self.assertEqual(result.device.type,'cpu')
                self.assertEqual(stats['max_action_absolute_error'],0.)
                self.assertEqual(stats['compared_actions'],1)
                self.assertEqual(fingerprint(policy.state_dict()),contract['actor_state_sha256'])
        finally:env.close()
