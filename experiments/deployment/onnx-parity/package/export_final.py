"""Export the accepted final LSTM with the original mjlab exporter. Zero PPO."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
from pathlib import Path
import signal
import sys
import time
import traceback

import common as old
from deploy_contract import MODEL_SHA, REPORT_SHA, ATOL, RTOL, OnnxPolicy, validate_report, verify_export
from fingerprint import fingerprint


def dump(path, data):
    old.write(path, data, exclusive=True)


def export(folder):
    old.preflight()
    report_path = old.SLOT / 'report.json'
    cp = validate_report(report_path)
    old.require(Path(cp['path']) == old.INPUTS / 'run/checkpoints/final6h_005304.pt', 'Unexpected final model path')
    old.require(old.sha(cp['path']) == MODEL_SHA, 'Final model bytes changed')
    before_inputs = old.verify_inputs()
    sys.path.insert(0, str(old.SOURCE / 'src'))
    import numpy as np
    import torch
    import onnx
    from tensordict import TensorDict
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
    from mjlab.rl.exporter_utils import attach_metadata_to_onnx, get_base_metadata
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_microduck.tasks.microduck_basketball_env_cfg import MicroduckBasketballRlCfg
    from mjlab_microduck.onnx_policy_contract import bake_action_clip
    from native_task import make_cfg, head_scale
    old.require(torch.cuda.is_available() and '5060' in torch.cuda.get_device_name(0), 'Use the existing local RTX5060')
    configure_torch_backends()
    torch.set_num_threads(1)
    raw = env = None
    begun = time.time()
    try:
        cfg = make_cfg(1, 2026102407)
        cfg.episode_length_s = 60.
        cfg.auto_reset = False
        print('EXPORT: build the pinned one-environment model; no rollout or training', flush=True)
        raw = ManagerBasedRlEnv(cfg, device='cuda:0', render_mode=None)
        env = RslRlVecEnvWrapper(raw)
        agent = asdict(deepcopy(MicroduckBasketballRlCfg))
        agent.update(seed=2026102407, logger='tensorboard', upload_model=False)
        # The final training and successful validation both used the unclipped
        # wrapper. Reject a changed config instead of silently adding a clamp.
        clip = agent['clip_actions']
        old.require(clip is None and env.clip_actions is None, 'Action clipping differs from the validated path')
        agent['algorithm'].update(learning_rate=old.LEARNING_RATE, schedule='fixed', num_mini_batches=4)
        runner = MjlabOnPolicyRunner(env, agent, log_dir=None, device='cuda:0')
        saved = torch.load(cp['path'], map_location='cpu', weights_only=False)
        meta = old.verify_checkpoint_metadata(saved, cp)
        runner.alg.load(saved, {'actor': True, 'critic': True, 'optimizer': False, 'iteration': False}, strict=True)
        policy = runner.get_inference_policy(device='cuda:0')
        old.require(not policy.training and not policy.obs_normalizer.training and policy.is_recurrent,
                    'Expected frozen recurrent actor and normalizer')
        raw.common_step_counter = meta['common_step_counter']
        raw._sim_step_counter = meta['sim_step_counter']
        old.require(head_scale(raw) == 1. and raw.step_dt == .02 and raw.physics_dt == .002,
                    'Mature 50Hz/500Hz contract differs')
        with torch.inference_mode():
            obs, _ = raw.reset(seed=2026102407)
        old.require(tuple(obs['actor'].shape) == (1, 61) and tuple(obs['critic'].shape) == (1, 85), 'Observation schema differs')
        initial = obs['actor'][0].detach().cpu().numpy().astype(np.float32)
        old.require(np.isfinite(initial).all(), 'Nonfinite initial observation')
        before = fingerprint(policy.state_dict())
        normalizer = fingerprint(policy.obs_normalizer.state_dict())
        model = folder / 'final6h_005304.onnx'
        runner.export_policy_to_onnx(str(folder), model.name)
        # Mandatory original export contract, including a no-op for clip=None.
        bake_action_clip(str(model), clip)
        metadata = get_base_metadata(raw, run_path=cp['path'])
        metadata.update(stage12_checkpoint_sha256=MODEL_SHA, actor_observations=61,
                        recurrent_state='Carry h_out/c_out into h_in/c_in; zero only at reset',
                        obs_normalizer_baked='true', action_clip='none (validated wrapper)',
                        hardware_acceptance='false', ros_distribution='humble')
        attach_metadata_to_onnx(str(model), metadata)
        onnx.checker.check_model(str(model))
        runtime = OnnxPolicy(model)
        reference = deepcopy(policy).to('cpu').eval()
        before_reference = fingerprint(reference.state_dict())
        rng = np.random.default_rng(2026102408)
        golden = []
        max_action = max_state = 0.
        print('EXPORT: 128 sequential PyTorch/ONNX comparisons including recurrent state and resets', flush=True)
        for seq in range(128):
            reset = seq in (0, 64)
            x = initial.copy() if reset else initial + rng.normal(0., .05, 61).astype(np.float32)
            with torch.inference_mode():
                if reset:
                    reference.reset()
                expected = reference(TensorDict({'actor': torch.from_numpy(x[None])}, batch_size=[1]),
                                     stochastic_output=False).detach().numpy()[0]
            actual = np.asarray(runtime.step(x.tolist(), reset=reset))
            old.require(np.isfinite(expected).all() and np.allclose(actual, expected, atol=ATOL, rtol=RTOL),
                        'PyTorch/ONNX action parity failed at step ' + str(seq))
            max_action = max(max_action, float(np.max(np.abs(actual - expected))))
            h, c = reference.get_hidden_state()
            expected_state = [h.detach().numpy(), c.detach().numpy()]
            for a, b in zip((runtime.h, runtime.c), expected_state):
                old.require(np.isfinite(b).all() and np.allclose(a, b, atol=ATOL, rtol=RTOL),
                            'PyTorch/ONNX recurrent state parity failed at step ' + str(seq))
                max_state = max(max_state, float(np.max(np.abs(a - b))))
            golden.append(dict(seq=seq, reset=reset, obs=x.tolist(), actions=expected.tolist(),
                               h=expected_state[0].reshape(-1).tolist(), c=expected_state[1].reshape(-1).tolist()))
        old.require(fingerprint(reference.state_dict()) == before_reference
                    and fingerprint(policy.state_dict()) == before, 'Export/parity changed actor or normalizer')
        joint = raw.action_manager.get_term('joint_pos')
        def values(x):
            return x[0].detach().cpu().tolist() if isinstance(x, torch.Tensor) else x
        obs_terms = []
        start = 0
        for name, dim in zip(raw.observation_manager.active_terms['actor'], raw.observation_manager.group_obs_term_dim['actor']):
            width = math.prod(dim)
            term = raw.observation_manager.get_term_cfg('actor', name)
            obs_terms.append(dict(name=name, start=start, stop=start + width,
                                  function=term.func.__module__ + '.' + term.func.__name__,
                                  scale=term.scale, clip=term.clip))
            start += width
        old.require(start == 61 and len(metadata['joint_names']) == 14, 'Runtime observation/joint schema differs')
        body = next(x for x in obs_terms if x['name'] == 'body_command')
        old.require(body['start'] == 55 and body['stop'] == 61, 'Head-ball input slot differs')
        contract = dict(schema='stage12-final-humble-deployment-contract-v1', checkpoint_sha256=MODEL_SHA,
                        onnx_sha256=old.sha(model), actor_state_sha256=before, normalizer_sha256=normalizer,
                        actor_observations=61, critic_observations=85, actions=14,
                        inference_hz=50, physics_hz=500, obs_normalizer_baked=True, external_normalization=False,
                        action_filter=None, action_clip=clip, action_offset=values(joint._offset),
                        action_scale=values(joint._scale), runtime_metadata=metadata, actor_terms=obs_terms,
                        inputs={k: v.shape for k, v in runtime.inputs.items()},
                        outputs={k: v.shape for k, v in runtime.outputs.items()},
                        reset='Zero h/c at activation and episode reset; carry on every subsequent step',
                        top_ball_sensor_requirement=dict(slot=[55, 61], reference_frame='stage11_head_material_frame',
                            position_formula='[20*x,25*y,50*(z-0.020)]', velocity_formula='2*[vx,vy,vz]',
                            simulation_source='Pinned native head_actor_observation', hardware_source='UNCONFIRMED'),
                        hardware_wiring='UNCONFIRMED', hardware_acceptance=False, strict_contact_acceptance=False)
        dump(folder / 'contract.json', contract)
        dump(folder / 'golden.json', dict(schema='sequential-lstm-golden-v1', onnx_sha256=old.sha(model),
             atol=ATOL, rtol=RTOL, source='PyTorch deterministic actor, reset observation plus fixed numerical perturbations',
             simulated_rollout=False, cases=golden))
        old.require(old.verify_inputs() == before_inputs and old.sha(report_path) == REPORT_SHA
                    and old.sha(cp['path']) == MODEL_SHA, 'Original evidence changed')
        env.close(); env = raw = None
        pins = {p.name: dict(bytes=p.stat().st_size, sha256=old.sha(p))
                for p in folder.iterdir() if p.is_file()}
        result = dict(status='EXPORT_PARITY_PASSED', checkpoint_sha256=MODEL_SHA,
                      validation_report_sha256=REPORT_SHA, policy_and_normalizer_unchanged=True,
                      original_training_records_unchanged=True, elapsed_seconds=time.time() - begun,
                      comparisons=128, recurrent_resets=2, max_action_absolute_error=max_action,
                      max_state_absolute_error=max_state, atol=ATOL, rtol=RTOL,
                      files=pins, physics_steps_requested=0, added_PPO_updates=0,
                      hardware_acceptance=False, realtime_acceptance=False, simulation_rollout=False)
        dump(folder / 'export-report.json', result)
        verify_export(folder)
        print('EXPORT_PARITY_PASSED=' + str(folder / 'export-report.json'), flush=True)
    finally:
        if env is not None:
            env.close()
        elif raw is not None:
            raw.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    folder = Path(args.output)
    old.require(folder.is_dir() and not folder.is_symlink() and not list(folder.iterdir()), 'Export output must be new and empty')
    def deadline(signum, frame):
        raise TimeoutError('Export independent 1200-second limit reached')
    signal.signal(signal.SIGALRM, deadline)
    signal.alarm(1200)
    try:
        export(folder)
    except BaseException as exc:
        dump(folder / 'export-error.json', dict(status='EXPORT_FAILED_PRESERVE', error=repr(exc),
             traceback=traceback.format_exc()[-12000:], added_PPO_updates=0, hardware_acceptance=False))
        raise
    finally:
        signal.alarm(0)


if __name__ == '__main__':
    main()
