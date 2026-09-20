"""Stage-05 checkpoint migration regression tests (no PPO execution)."""

from __future__ import annotations

from itertools import chain

import pytest
import torch
from rsl_rl.models import MLPModel, RNNModel
from tensordict import TensorDict

from mjlab_microduck.double_balance_checkpoint import (
    ACTOR_OBSERVATION_DIM,
    ACTOR_TOP_BALL_START,
    SOURCE_CRITIC_OBSERVATION_DIM,
    TARGET_CRITIC_OBSERVATION_DIM,
    TARGET_LEARNING_RATE,
    audit_source_schema,
    audit_target_schema,
    migrate_checkpoint,
)
from mjlab_microduck.double_balance_checkpoint_validation import validate_migration
from mjlab_microduck.tasks.microduck_double_balance_env_cfg import (
    ACTOR_OBSERVATION_DIM as ENV_ACTOR_OBSERVATION_DIM,
    CRITIC_OBSERVATION_DIM as ENV_CRITIC_OBSERVATION_DIM,
)


def _clone_state(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in module.state_dict().items()}


@pytest.fixture()
def released_schema_checkpoint() -> dict:
    torch.manual_seed(7)
    observations = TensorDict(
        {
            "actor": torch.zeros(2, ACTOR_OBSERVATION_DIM),
            "critic": torch.zeros(2, SOURCE_CRITIC_OBSERVATION_DIM),
        },
        batch_size=[2],
    )
    actor = RNNModel(
        observations,
        {"actor": ["actor"]},
        "actor",
        14,
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
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
    optimizer = torch.optim.Adam(
        chain(actor.parameters(), critic.parameters()), lr=2.0e-5
    )
    for parameter in chain(actor.parameters(), critic.parameters()):
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    actor_state = _clone_state(actor)
    for key in (
        "obs_normalizer._mean",
        "obs_normalizer._var",
        "obs_normalizer._std",
    ):
        actor_state[key][..., ACTOR_TOP_BALL_START:] = 0.0
    return {
        "actor_state_dict": actor_state,
        "critic_state_dict": _clone_state(critic),
        "optimizer_state_dict": optimizer.state_dict(),
        "iter": 6999,
        "infos": {"env_state": {"common_step_counter": 168048}},
    }


def test_migration_dimensions_match_frozen_environment_contract():
    assert ACTOR_OBSERVATION_DIM == ENV_ACTOR_OBSERVATION_DIM == 61
    assert TARGET_CRITIC_OBSERVATION_DIM == ENV_CRITIC_OBSERVATION_DIM == 85


def test_source_schema_rejects_a_nonzero_zero_padding_stat(
    released_schema_checkpoint,
):
    released_schema_checkpoint["actor_state_dict"]["obs_normalizer._mean"][
        0, ACTOR_TOP_BALL_START
    ] = 0.1
    with pytest.raises(ValueError, match="zero-padding statistics"):
        audit_source_schema(released_schema_checkpoint)


def test_migration_is_non_mutating_and_changes_only_declared_tensors(
    released_schema_checkpoint,
):
    original_actor_input = released_schema_checkpoint["actor_state_dict"][
        "rnn.rnn.weight_ih_l0"
    ].clone()
    original_critic_input = released_schema_checkpoint["critic_state_dict"][
        "mlp.0.weight"
    ].clone()
    original_optimizer_entries = len(
        released_schema_checkpoint["optimizer_state_dict"]["state"]
    )

    migrated = migrate_checkpoint(
        released_schema_checkpoint, source_sha256="a" * 64
    )
    schema = audit_target_schema(migrated)

    assert torch.equal(
        released_schema_checkpoint["actor_state_dict"]["rnn.rnn.weight_ih_l0"],
        original_actor_input,
    )
    assert torch.equal(
        migrated["actor_state_dict"]["rnn.rnn.weight_ih_l0"][
            ..., :ACTOR_TOP_BALL_START
        ],
        original_actor_input[..., :ACTOR_TOP_BALL_START],
    )
    assert torch.count_nonzero(
        migrated["actor_state_dict"]["rnn.rnn.weight_ih_l0"][
            ..., ACTOR_TOP_BALL_START:
        ]
    ).item() == 0
    assert torch.equal(
        migrated["critic_state_dict"]["mlp.0.weight"][
            ..., :SOURCE_CRITIC_OBSERVATION_DIM
        ],
        original_critic_input,
    )
    assert torch.count_nonzero(
        migrated["critic_state_dict"]["mlp.0.weight"][
            ..., SOURCE_CRITIC_OBSERVATION_DIM:
        ]
    ).item() == 0
    assert migrated["optimizer_state_dict"]["state"] == {}
    assert schema["optimizer_learning_rates"] == [TARGET_LEARNING_RATE]
    assert schema["migration"]["optimizer_state_entries_removed"] == (
        original_optimizer_entries
    )
    assert migrated["iter"] == 6999
    assert migrated["infos"]["env_state"] == {
        "common_step_counter": 168048
    }
    assert migrated["infos"]["double_balance_checkpoint_migration"][
        "ppo_executed"
    ] is False


def test_full_structural_and_numerical_validation(released_schema_checkpoint):
    migrated = migrate_checkpoint(
        released_schema_checkpoint, source_sha256="b" * 64
    )
    report = validate_migration(
        released_schema_checkpoint, migrated, source_sha256="b" * 64
    )
    assert report["status"] == "PASS"
    assert report["ppo_executed"] is False
    assert report["actor_zero_top_ball_action_max_abs_error"] == 0.0
    assert report["actor_lstm_hidden_max_abs_error"] == 0.0
    assert report["actor_new_input_isolation_max_abs_error"] == 0.0
    assert report["critic_value_max_abs_error"] <= 1.0e-6
    assert report["strict_actor_load"] == "PASS"
    assert report["strict_critic_load"] == "PASS"
    assert report["rebuilt_optimizer_load"] == "PASS"


def test_invalid_learning_rate_is_rejected(released_schema_checkpoint):
    with pytest.raises(ValueError, match="learning_rate must be positive"):
        migrate_checkpoint(
            released_schema_checkpoint,
            source_sha256="c" * 64,
            learning_rate=0.0,
        )
