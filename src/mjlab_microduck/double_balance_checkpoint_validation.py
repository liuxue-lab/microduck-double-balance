"""Numerical validation for a migrated double-balance checkpoint."""

from __future__ import annotations

from itertools import chain
from typing import Any

import torch
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from mjlab_microduck.basketball_distillation import load_actor
from mjlab_microduck.double_balance_checkpoint import (
    ACTOR_OBSERVATION_DIM,
    ACTOR_TOP_BALL_START,
    SOURCE_CRITIC_OBSERVATION_DIM,
    TARGET_CRITIC_OBSERVATION_DIM,
    audit_source_schema,
    audit_target_schema,
)


def _make_critic(checkpoint: dict[str, Any], observation_dim: int) -> MLPModel:
    observations = TensorDict(
        {"critic": torch.zeros(1, observation_dim)}, batch_size=[1]
    )
    critic = MLPModel(
        observations,
        {"critic": ["critic"]},
        "critic",
        1,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    )
    critic.load_state_dict(checkpoint["critic_state_dict"], strict=True)
    critic.eval()
    return critic


def _max_hidden_error(left: Any, right: Any) -> float:
    if torch.is_tensor(left) and torch.is_tensor(right):
        return float(torch.max(torch.abs(left - right)))
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        if len(left) != len(right):
            return float("inf")
        return max((_max_hidden_error(a, b) for a, b in zip(left, right)), default=0.0)
    if left is None and right is None:
        return 0.0
    return float("inf")


def validate_migration(
    source: dict[str, Any], migrated: dict[str, Any], source_sha256: str
) -> dict[str, Any]:
    """Return a numerical and structural validation report or raise."""
    source_schema = audit_source_schema(source)
    target_schema = audit_target_schema(migrated)
    provenance = target_schema["migration"]
    if provenance["source_sha256"] != source_sha256:
        raise ValueError("migration provenance does not match the source checkpoint")
    if migrated["iter"] != source["iter"]:
        raise ValueError("source iteration was not preserved")
    if migrated["infos"].get("env_state") != source["infos"].get("env_state"):
        raise ValueError("source environment progress was not preserved")

    source_actor = source["actor_state_dict"]
    target_actor = migrated["actor_state_dict"]
    for key, value in source_actor.items():
        if key == "rnn.rnn.weight_ih_l0":
            if not torch.equal(
                value[..., :ACTOR_TOP_BALL_START],
                target_actor[key][..., :ACTOR_TOP_BALL_START],
            ):
                raise ValueError("actor inherited LSTM columns changed")
        elif key.startswith("obs_normalizer.") and key != "obs_normalizer.count":
            if not torch.equal(
                value[..., :ACTOR_TOP_BALL_START],
                target_actor[key][..., :ACTOR_TOP_BALL_START],
            ):
                raise ValueError(f"actor inherited normalizer values changed: {key}")
        elif not torch.equal(value, target_actor[key]):
            raise ValueError(f"unintended actor tensor change: {key}")

    source_critic = source["critic_state_dict"]
    target_critic = migrated["critic_state_dict"]
    for key, value in source_critic.items():
        candidate = target_critic[key]
        if key == "mlp.0.weight" or (
            key.startswith("obs_normalizer.") and key != "obs_normalizer.count"
        ):
            candidate = candidate[..., :SOURCE_CRITIC_OBSERVATION_DIM]
        if not torch.equal(value, candidate):
            raise ValueError(f"unintended critic tensor change: {key}")

    torch.set_num_threads(1)
    actor_observations = TensorDict(
        {"actor": torch.zeros(1, ACTOR_OBSERVATION_DIM)}, batch_size=[1]
    )
    source_policy = load_actor(source, actor_observations)
    target_policy = load_actor(migrated, actor_observations)
    generator = torch.Generator().manual_seed(20260920)
    action_error = 0.0
    hidden_error = 0.0
    with torch.inference_mode():
        for step in range(16):
            if step == 8:
                source_policy.reset()
                target_policy.reset()
            values = torch.randn(1, ACTOR_OBSERVATION_DIM, generator=generator) * 0.2
            values[..., ACTOR_TOP_BALL_START:] = 0.0
            observations = TensorDict({"actor": values}, batch_size=[1])
            old_action = source_policy(observations)
            new_action = target_policy(observations)
            action_error = max(
                action_error, float(torch.max(torch.abs(old_action - new_action)))
            )
            hidden_error = max(
                hidden_error,
                _max_hidden_error(
                    source_policy.get_hidden_state(), target_policy.get_hidden_state()
                ),
            )
    if action_error != 0.0 or hidden_error != 0.0:
        raise ValueError(
            f"actor parity failed: action_error={action_error}, hidden_error={hidden_error}"
        )

    zero_tail_policy = load_actor(migrated, actor_observations)
    live_tail_policy = load_actor(migrated, actor_observations)
    base = torch.randn(1, ACTOR_OBSERVATION_DIM, generator=generator) * 0.2
    zero_tail = base.clone()
    zero_tail[..., ACTOR_TOP_BALL_START:] = 0.0
    live_tail = base.clone()
    live_tail[..., ACTOR_TOP_BALL_START:] = torch.randn(
        1,
        ACTOR_OBSERVATION_DIM - ACTOR_TOP_BALL_START,
        generator=generator,
    )
    with torch.inference_mode():
        isolated_zero = zero_tail_policy(
            TensorDict({"actor": zero_tail}, batch_size=[1])
        )
        isolated_live = live_tail_policy(
            TensorDict({"actor": live_tail}, batch_size=[1])
        )
    actor_new_input_isolation_error = float(
        torch.max(torch.abs(isolated_zero - isolated_live))
    )
    if actor_new_input_isolation_error != 0.0:
        raise ValueError("migrated actor has non-zero inherited top-ball coupling")

    source_value = _make_critic(source, SOURCE_CRITIC_OBSERVATION_DIM)
    target_value = _make_critic(migrated, TARGET_CRITIC_OBSERVATION_DIM)
    old_critic_obs = torch.randn(
        32, SOURCE_CRITIC_OBSERVATION_DIM, generator=generator
    )
    new_critic_tail = torch.randn(
        32,
        TARGET_CRITIC_OBSERVATION_DIM - SOURCE_CRITIC_OBSERVATION_DIM,
        generator=generator,
    )
    with torch.inference_mode():
        old_values = source_value(
            TensorDict({"critic": old_critic_obs}, batch_size=[32])
        )
        new_values = target_value(
            TensorDict(
                {"critic": torch.cat((old_critic_obs, new_critic_tail), dim=-1)},
                batch_size=[32],
            )
        )
    critic_error = float(torch.max(torch.abs(old_values - new_values)))
    if critic_error > 1.0e-6:
        raise ValueError(f"critic value parity failed: max error {critic_error}")

    optimizer = torch.optim.Adam(
        chain(target_policy.parameters(), target_value.parameters()), lr=1.0e-3
    )
    optimizer.load_state_dict(migrated["optimizer_state_dict"])
    if optimizer.state:
        raise ValueError("rebuilt optimizer unexpectedly contains moments")

    return {
        "status": "PASS",
        "ppo_executed": False,
        "source_schema": source_schema,
        "target_schema": target_schema,
        "actor_zero_top_ball_action_max_abs_error": action_error,
        "actor_lstm_hidden_max_abs_error": hidden_error,
        "actor_new_input_isolation_max_abs_error": actor_new_input_isolation_error,
        "critic_value_max_abs_error": critic_error,
        "strict_actor_load": "PASS",
        "strict_critic_load": "PASS",
        "rebuilt_optimizer_load": "PASS",
    }
