"""Reproduce the real rsl-rl RNN normalizer failure; no simulation/PPO rollout."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch
from mjlab.rl.runner import MjlabOnPolicyRunner
from rsl_rl.algorithms import PPO
from rsl_rl.models import RNNModel
from tensordict import TensorDict

from mjlab_microduck.double_balance_smoke import require_equal
from mjlab_microduck.double_balance_stage08_state import load_runner_checkpoint


def make_runner():
    torch.manual_seed(808)
    obs = TensorDict({'actor': torch.randn(8, 61), 'critic': torch.randn(8, 85)}, batch_size=[8])
    groups = {'actor': ['actor'], 'critic': ['critic']}
    actor = RNNModel(obs, groups, 'actor', 14, hidden_dims=[8], rnn_hidden_dim=8,
                     obs_normalization=True)
    critic = RNNModel(obs, groups, 'critic', 1, hidden_dims=[8], rnn_hidden_dim=8,
                      obs_normalization=True)
    # The real loader needs models + Adam, not a simulator or rollout storage.
    alg = PPO(actor, critic, storage=None, learning_rate=2.25e-5)
    runner = MjlabOnPolicyRunner.__new__(MjlabOnPolicyRunner)
    runner.alg = alg
    runner.env = NS(unwrapped=NS(common_step_counter=24000))
    runner.current_learning_iteration = 999
    # Materialize real Adam moments using a synthetic differentiable objective.
    loss = sum(p.square().sum() for p in (*actor.parameters(), *critic.parameters()))
    loss.backward(); alg.optimizer.step(); alg.optimizer.zero_grad()
    for state in alg.optimizer.state.values():
        state['step'].fill_(20300)
    with torch.inference_mode():
        actor.update_normalization(obs)
        critic.update_normalization(obs)
    assert torch.is_inference(actor.obs_normalizer._std)
    assert torch.is_inference(critic.obs_normalizer._std)
    checkpoint = deepcopy(alg.save())
    checkpoint.update(iter=1014, infos={'env_state': {'common_step_counter': 24360}})
    return runner, checkpoint


def test_unmodified_registered_loader_reproduces_cloud_error(tmp_path):
    runner, checkpoint = make_runner()
    path = tmp_path / 'saved.pt'; torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match='Inplace update to inference tensor outside InferenceMode'):
        runner.load(str(path), strict=True, map_location='cpu')


@pytest.mark.parametrize('outer_inference', [False, True])
def test_strict_reload_preserves_all_state_and_allows_subsequent_adam(tmp_path, outer_inference):
    runner, checkpoint = make_runner()
    path = tmp_path / 'saved.pt'; torch.save(checkpoint, path)
    params = list(runner.alg.actor.parameters()) + list(runner.alg.critic.parameters())
    parameter_ids = [id(p) for p in params]
    normal_buffer = runner.alg.actor.obs_normalizer._mean
    with torch.inference_mode(outer_inference):
        load_runner_checkpoint(runner, path, map_location='cpu')
        assert torch.is_inference_mode_enabled() == outer_inference
    assert [id(p) for p in (*runner.alg.actor.parameters(), *runner.alg.critic.parameters())] == parameter_ids
    assert [id(p) for p in runner.alg.optimizer.param_groups[0]['params']] == parameter_ids
    assert runner.alg.actor.obs_normalizer._mean is normal_buffer
    for model in (runner.alg.actor, runner.alg.critic):
        assert not any(torch.is_inference(b) for b in model.obs_normalizer.buffers())
    require_equal(runner.alg.save(), {k: checkpoint[k] for k in runner.alg.save()}, 'reloaded')
    assert runner.current_learning_iteration == 1014
    assert runner.env.unwrapped.common_step_counter == 24360
    assert runner.alg.optimizer.param_groups[0]['lr'] == runner.alg.learning_rate == 2.25e-5
    for value in runner.alg.optimizer.state.values():
        assert not any(torch.is_inference(t) for t in value.values() if torch.is_tensor(t))
    sum(p.square().sum() for p in params).backward()
    runner.alg.optimizer.step()
    assert all(v['step'].item() == 20301 for v in runner.alg.optimizer.state.values())


def test_reload_still_rejects_missing_model_keys(tmp_path):
    runner, checkpoint = make_runner()
    checkpoint['actor_state_dict'].pop('obs_normalizer._std')
    path = tmp_path / 'invalid.pt'; torch.save(checkpoint, path)
    with pytest.raises(RuntimeError, match='Missing key'):
        load_runner_checkpoint(runner, path, map_location='cpu')
