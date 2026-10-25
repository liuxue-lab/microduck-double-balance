"""Failure-mode tests for Stage-06 acceptance (CPU, no long training)."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from mjlab_microduck.double_balance_smoke import (
    check_observations,
    check_optimizer,
    expected_progress,
    monitor_smoke,
    require_equal,
    require_finite,
    resume_agent_config,
)


def test_resume_progress_is_last_index_and_vector_steps_not_env_transitions():
    assert expected_progress(6999, 168048) == (7003, 168168)
    with pytest.raises(ValueError, match="positive"):
        expected_progress(6999, 168048, 0)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_nested_loss_or_tensor_is_rejected(bad):
    with pytest.raises(ValueError, match="non-finite"):
        require_finite({"loss": [bad]}, "result")
    with pytest.raises(ValueError, match="non-finite"):
        require_finite({"state": (torch.tensor([1., bad]),)}, "checkpoint")


def test_reload_equality_rejects_corruption_and_wrong_dtype():
    expected = {"params": [torch.tensor([1., 2.])], "lr": 0.001}
    require_equal(deepcopy(expected), expected, "state")
    corrupted = deepcopy(expected)
    corrupted["params"][0][1] += 0.01
    with pytest.raises(ValueError, match="tensor mismatch"):
        require_equal(corrupted, expected, "state")
    corrupted["params"][0] = expected["params"][0].double()
    with pytest.raises(ValueError, match="dtype mismatch"):
        require_equal(corrupted, expected, "state")


def test_reload_config_preserves_saved_adaptive_lr_without_mutating_recipe():
    agent = {"algorithm": {"learning_rate": 0.001, "schedule": "adaptive"}}
    checkpoint = {"optimizer_state_dict": {"param_groups": [{"lr": 0.0002}]}}
    restored = resume_agent_config(agent, checkpoint)
    assert restored["algorithm"] == {"learning_rate": 0.0002, "schedule": "adaptive"}
    assert agent["algorithm"]["learning_rate"] == 0.001
    checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"] = float("nan")
    with pytest.raises(ValueError, match="invalid checkpoint learning rate"):
        resume_agent_config(agent, checkpoint)


def test_observation_contract_rejects_legacy_critic_and_masked_nan():
    obs = {"actor": torch.zeros(64, 61), "critic": torch.zeros(64, 85)}
    check_observations(obs)
    obs["critic"] = torch.zeros(64, 76)
    with pytest.raises(ValueError, match="Nx85"):
        check_observations(obs)
    obs["critic"] = torch.zeros(64, 85)
    obs["actor"][0, 60] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        check_observations(obs)


def test_adam_audit_checks_real_updated_moments_steps_and_shapes():
    parameter = torch.nn.Parameter(torch.ones(3, 2))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    parameter.square().sum().backward()
    optimizer.step()
    check_optimizer(optimizer, expected_steps=1)
    with pytest.raises(ValueError, match="step count mismatch"):
        check_optimizer(optimizer, expected_steps=2)
    optimizer.state[parameter]["exp_avg"] = torch.zeros(3, 1)
    with pytest.raises(ValueError, match="shape mismatch"):
        check_optimizer(optimizer, expected_steps=1)


def test_adam_audit_rejects_untrained_or_nonfinite_moments():
    parameter = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.Adam([parameter], lr=1e-3)
    with pytest.raises(ValueError, match="moments missing"):
        check_optimizer(optimizer, expected_steps=1)
    parameter.sum().backward()
    optimizer.step()
    optimizer.state[parameter]["exp_avg_sq"][0] = float("inf")
    with pytest.raises(ValueError, match="non-finite"):
        check_optimizer(optimizer, expected_steps=1)


def test_nan_autoreset_is_rejected_even_when_returned_observations_are_finite():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.Adam([parameter])
    alg = SimpleNamespace(optimizer=optimizer, compute_returns=lambda obs: None,
                          update=lambda: {}, num_learning_epochs=5, num_mini_batches=4)
    obs = {"actor": torch.zeros(64, 61), "critic": torch.zeros(64, 85)}
    # Simulate a NaN termination followed by autoreset to a finite observation.
    original_step = lambda actions: (obs, torch.zeros(64), torch.zeros(64), {})
    termination = SimpleNamespace(get_term=lambda name: torch.ones(64, dtype=torch.bool))
    env = SimpleNamespace(step=original_step,
                          unwrapped=SimpleNamespace(termination_manager=termination))
    with pytest.raises(ValueError, match="autoreset could mask it"):
        with monitor_smoke(SimpleNamespace(alg=alg), env):
            env.step(torch.zeros(64, 14))
    assert env.step is original_step
