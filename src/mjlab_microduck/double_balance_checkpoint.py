"""Migrate the released basketball checkpoint to the double-balance schema.

The migration is deliberately independent of PPO and the simulator. It keeps
the released recurrent actor's 61-D interface, neutralizes the six formerly
zero-padded inputs, appends nine zero-initialized critic inputs, and rebuilds
the Adam state for the new task.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch


RELEASE_CHECKPOINT_SHA256 = (
    "57a322eff092cb71e7cba831791232f0cc4fd45ede0441cad6d87813b0033e41"
)
ACTOR_OBSERVATION_DIM = 61
SOURCE_CRITIC_OBSERVATION_DIM = 76
TARGET_CRITIC_OBSERVATION_DIM = 85
ACTOR_TOP_BALL_START = 55
ACTOR_TOP_BALL_STOP = 61
TARGET_LEARNING_RATE = 1.0e-3

_NORMALIZER_DEFAULTS = {
    "obs_normalizer._mean": 0.0,
    "obs_normalizer._var": 1.0,
    "obs_normalizer._std": 1.0,
}


def file_sha256(path: str | Path) -> str:
    """Return a streaming SHA-256 digest for *path*."""
    digest = sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_tensor_shape(
    state: dict[str, Any], key: str, shape: tuple[int, ...], owner: str
) -> torch.Tensor:
    value = state.get(key)
    if not torch.is_tensor(value):
        raise ValueError(f"{owner}.{key} must be a tensor")
    if tuple(value.shape) != shape:
        raise ValueError(
            f"{owner}.{key} has shape {tuple(value.shape)}, expected {shape}"
        )
    return value


def audit_source_schema(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Validate and summarize the exact released basketball schema."""
    required = {
        "actor_state_dict",
        "critic_state_dict",
        "optimizer_state_dict",
        "iter",
        "infos",
    }
    missing = sorted(required.difference(checkpoint))
    if missing:
        raise ValueError(f"checkpoint is missing required keys: {missing}")

    actor = checkpoint["actor_state_dict"]
    critic = checkpoint["critic_state_dict"]
    optimizer = checkpoint["optimizer_state_dict"]
    if not isinstance(actor, dict) or not isinstance(critic, dict):
        raise ValueError("actor_state_dict and critic_state_dict must be mappings")
    if not isinstance(optimizer, dict):
        raise ValueError("optimizer_state_dict must be a mapping")

    for key in _NORMALIZER_DEFAULTS:
        _require_tensor_shape(actor, key, (1, ACTOR_OBSERVATION_DIM), "actor")
        _require_tensor_shape(
            critic,
            key,
            (1, SOURCE_CRITIC_OBSERVATION_DIM),
            "critic",
        )
    actor_input = _require_tensor_shape(
        actor,
        "rnn.rnn.weight_ih_l0",
        (1024, ACTOR_OBSERVATION_DIM),
        "actor",
    )
    critic_input = _require_tensor_shape(
        critic,
        "mlp.0.weight",
        (512, SOURCE_CRITIC_OBSERVATION_DIM),
        "critic",
    )
    _require_tensor_shape(actor, "distribution.std_param", (14,), "actor")

    actor_slot = slice(ACTOR_TOP_BALL_START, ACTOR_TOP_BALL_STOP)
    for key in _NORMALIZER_DEFAULTS:
        values = actor[key][..., actor_slot]
        if torch.count_nonzero(values).item() != 0:
            raise ValueError(
                f"actor.{key} columns {ACTOR_TOP_BALL_START}:{ACTOR_TOP_BALL_STOP} "
                "must be the released zero-padding statistics"
            )

    param_groups = optimizer.get("param_groups")
    optimizer_state = optimizer.get("state")
    if not isinstance(param_groups, list) or not param_groups:
        raise ValueError("optimizer_state_dict.param_groups must be a non-empty list")
    if not isinstance(optimizer_state, dict):
        raise ValueError("optimizer_state_dict.state must be a mapping")
    learning_rates = [float(group["lr"]) for group in param_groups]

    return {
        "top_level_keys": list(checkpoint),
        "actor_state_key_count": len(actor),
        "critic_state_key_count": len(critic),
        "actor_input_shape": list(actor_input.shape),
        "critic_input_shape": list(critic_input.shape),
        "actor_normalizer_shape": list(actor["obs_normalizer._mean"].shape),
        "critic_normalizer_shape": list(critic["obs_normalizer._mean"].shape),
        "optimizer_param_group_count": len(param_groups),
        "optimizer_state_entry_count": len(optimizer_state),
        "optimizer_learning_rates": learning_rates,
        "iteration": int(checkpoint["iter"]),
    }


def _append_neutral_columns(tensor: torch.Tensor, count: int, value: float) -> torch.Tensor:
    tail_shape = (*tensor.shape[:-1], count)
    tail = torch.full(tail_shape, value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat((tensor, tail), dim=-1)


def migrate_checkpoint(
    checkpoint: dict[str, Any],
    *,
    source_sha256: str,
    learning_rate: float = TARGET_LEARNING_RATE,
) -> dict[str, Any]:
    """Return a migrated deep copy without modifying *checkpoint*.

    The old optimizer moments are intentionally discarded. Their learning
    rate is the end-of-run adaptive value and their critic tensors have the old
    width; preserving them would not be a neutral initialization for the new
    task.
    """
    if len(source_sha256) != 64:
        raise ValueError("source_sha256 must be a 64-character hexadecimal digest")
    try:
        int(source_sha256, 16)
    except ValueError as exc:
        raise ValueError("source_sha256 must be hexadecimal") from exc
    if not 0.0 < learning_rate:
        raise ValueError("learning_rate must be positive")

    source_schema = audit_source_schema(checkpoint)
    migrated = deepcopy(checkpoint)
    actor = migrated["actor_state_dict"]
    critic = migrated["critic_state_dict"]

    actor_slot = slice(ACTOR_TOP_BALL_START, ACTOR_TOP_BALL_STOP)
    for key, value in _NORMALIZER_DEFAULTS.items():
        actor[key][..., actor_slot] = value
    actor["rnn.rnn.weight_ih_l0"][..., actor_slot] = 0.0

    appended = TARGET_CRITIC_OBSERVATION_DIM - SOURCE_CRITIC_OBSERVATION_DIM
    for key, value in _NORMALIZER_DEFAULTS.items():
        critic[key] = _append_neutral_columns(critic[key], appended, value)
    critic["mlp.0.weight"] = _append_neutral_columns(
        critic["mlp.0.weight"], appended, 0.0
    )

    optimizer = migrated["optimizer_state_dict"]
    old_state_entries = len(optimizer["state"])
    optimizer["state"] = {}
    for group in optimizer["param_groups"]:
        group["lr"] = learning_rate
        if "initial_lr" in group:
            group["initial_lr"] = learning_rate

    infos = migrated.get("infos")
    if infos is None:
        infos = {}
        migrated["infos"] = infos
    if not isinstance(infos, dict):
        raise ValueError("checkpoint infos must be a mapping or None")
    infos["double_balance_checkpoint_migration"] = {
        "stage": 5,
        "source_sha256": source_sha256,
        "source_iteration": int(checkpoint["iter"]),
        "iteration_preserved": True,
        "actor_observation_dim": ACTOR_OBSERVATION_DIM,
        "actor_top_ball_indices": [ACTOR_TOP_BALL_START, ACTOR_TOP_BALL_STOP - 1],
        "actor_normalizer_reset": {"mean": 0.0, "variance": 1.0, "std": 1.0},
        "actor_lstm_input_columns_zeroed": True,
        "critic_observation_dim_before": SOURCE_CRITIC_OBSERVATION_DIM,
        "critic_observation_dim_after": TARGET_CRITIC_OBSERVATION_DIM,
        "critic_appended_columns_zeroed": True,
        "optimizer_moments_reset": True,
        "optimizer_state_entries_removed": old_state_entries,
        "source_optimizer_learning_rates": source_schema[
            "optimizer_learning_rates"
        ],
        "target_optimizer_learning_rate": learning_rate,
        "ppo_executed": False,
    }
    return migrated


def audit_target_schema(checkpoint: dict[str, Any]) -> dict[str, Any]:
    """Validate and summarize the migrated double-balance schema."""
    actor = checkpoint.get("actor_state_dict")
    critic = checkpoint.get("critic_state_dict")
    optimizer = checkpoint.get("optimizer_state_dict")
    if not isinstance(actor, dict) or not isinstance(critic, dict):
        raise ValueError("migrated actor and critic state dictionaries are required")
    if not isinstance(optimizer, dict):
        raise ValueError("migrated optimizer state dictionary is required")

    for key in _NORMALIZER_DEFAULTS:
        _require_tensor_shape(actor, key, (1, ACTOR_OBSERVATION_DIM), "actor")
        _require_tensor_shape(
            critic,
            key,
            (1, TARGET_CRITIC_OBSERVATION_DIM),
            "critic",
        )
    actor_input = _require_tensor_shape(
        actor,
        "rnn.rnn.weight_ih_l0",
        (1024, ACTOR_OBSERVATION_DIM),
        "actor",
    )
    critic_input = _require_tensor_shape(
        critic,
        "mlp.0.weight",
        (512, TARGET_CRITIC_OBSERVATION_DIM),
        "critic",
    )

    actor_slot = slice(ACTOR_TOP_BALL_START, ACTOR_TOP_BALL_STOP)
    if torch.count_nonzero(actor_input[..., actor_slot]).item() != 0:
        raise ValueError("migrated actor top-ball LSTM input columns are not zero")
    for key, value in _NORMALIZER_DEFAULTS.items():
        expected = torch.full_like(actor[key][..., actor_slot], value)
        if not torch.equal(actor[key][..., actor_slot], expected):
            raise ValueError(f"migrated actor {key} top-ball columns are not neutral")

    critic_tail = slice(SOURCE_CRITIC_OBSERVATION_DIM, TARGET_CRITIC_OBSERVATION_DIM)
    if torch.count_nonzero(critic_input[..., critic_tail]).item() != 0:
        raise ValueError("migrated critic appended input columns are not zero")
    for key, value in _NORMALIZER_DEFAULTS.items():
        expected = torch.full_like(critic[key][..., critic_tail], value)
        if not torch.equal(critic[key][..., critic_tail], expected):
            raise ValueError(f"migrated critic {key} appended columns are not neutral")

    if optimizer.get("state") != {}:
        raise ValueError("migrated optimizer moments must be empty")
    param_groups = optimizer.get("param_groups")
    if not isinstance(param_groups, list) or not param_groups:
        raise ValueError("migrated optimizer param_groups must be non-empty")
    learning_rates = [float(group["lr"]) for group in param_groups]

    migration = checkpoint.get("infos", {}).get(
        "double_balance_checkpoint_migration"
    )
    if not isinstance(migration, dict):
        raise ValueError("migrated checkpoint is missing migration provenance")

    return {
        "actor_input_shape": list(actor_input.shape),
        "critic_input_shape": list(critic_input.shape),
        "actor_normalizer_shape": list(actor["obs_normalizer._mean"].shape),
        "critic_normalizer_shape": list(critic["obs_normalizer._mean"].shape),
        "optimizer_state_entry_count": 0,
        "optimizer_learning_rates": learning_rates,
        "iteration": int(checkpoint["iter"]),
        "migration": migration,
    }
