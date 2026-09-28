"""Stage 08 experiment profiles and budget calculations; no training on import.

The profiles are training-only overlays. The frozen Stage 04 play factory must
be called directly for official evaluation, never through this module.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
import math

BASELINE_COMMIT = "1fe5f54dc467205b846a807f2bd494c46f3e2c40"
SOURCE_SHA256 = "c667f96607b68383047f23956ba58805920434465245ef8e32d1148b17fc65b7"
SOURCE_UPDATES = 1000
NUM_ENVS = 4096
ROLLOUT_STEPS = 24
INITIAL_LR = 2.2500000000000008e-5
TRAIN_SEEDS = (20260928, 20260929, 20260930)


@dataclass(frozen=True)
class Profile:
    name: str
    zero_command: bool = False
    fixed_action_rate: bool = False
    fixed_com: bool = False
    zero_assistance: bool = False


PROFILES = {
    "A": Profile("A"),
    "B": Profile("B", zero_command=True),
    "C": Profile("C", zero_command=True, fixed_action_rate=True),
    "D": Profile("D", zero_command=True, fixed_action_rate=True, fixed_com=True),
    "E": Profile("E", zero_command=True, fixed_action_rate=True, fixed_com=True,
                 zero_assistance=True),
}


def apply_training_profile(cfg, name: str, *, evaluation: bool = False):
    """Return an independent cfg before manager construction; never mutate cfg.

    E intentionally retains the six-level table, while removing the curriculum.
    Its runtime loader must set every level to 5 and hold to zero AFTER reset.
    Merely constructing this cfg is not sufficient to certify zero assistance.
    """
    if evaluation:
        raise ValueError("Use the frozen Stage 04 play factory for evaluation")
    profile = PROFILES[name]
    if (cfg.scene.num_envs != NUM_ENVS or cfg.decimation != 10
            or not math.isclose(cfg.sim.mujoco.timestep, 0.002, abs_tol=1e-12)
            or cfg.episode_length_s != 10.0):
        raise ValueError("Stage 08 requires the approved 4096-env timing contract")
    required_curricula = {"standing_envs", "action_rate_weight", "com_range",
                          "head_com_range", "basketball_hold"}
    if not required_curricula <= set(cfg.curriculum):
        raise ValueError("Expected a fresh Stage 04 training cfg before overlays")
    if tuple(cfg.actions["ball_hold"].levels) != (1, .5, .25, .1, .03, 0):
        raise ValueError("Assistance table differs from the source checkpoint")
    result = deepcopy(cfg)
    if profile.zero_command:
        twist = result.commands["twist"]
        for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            setattr(twist.ranges, axis, (0.0, 0.0))
        twist.rel_standing_envs = 1.0
        twist.rel_turn_in_place_envs = 0.0
        result.curriculum.pop("standing_envs")
    if profile.fixed_action_rate:
        result.rewards["action_rate_l2"].weight = -0.4
        result.curriculum.pop("action_rate_weight")
    if profile.fixed_com:
        for event, curriculum in (("randomize_com", "com_range"),
                                  ("randomize_head_com", "head_com_range")):
            result.events[event].params["ranges"] = (-0.005, 0.005)
            result.curriculum.pop(curriculum)
    if profile.zero_assistance:
        result.curriculum.pop("basketball_hold")
    return result


def progress(experiment_updates: int) -> dict:
    if type(experiment_updates) is not int or experiment_updates < 0:
        raise ValueError("experiment_updates must be a nonnegative integer")
    lineage = SOURCE_UPDATES + experiment_updates
    return {
        "source_completed_updates": SOURCE_UPDATES,
        "experiment_completed_updates": experiment_updates,
        "lineage_completed_updates": lineage,
        "last_completed_iteration": lineage - 1,
        "next_iteration": lineage,
        "common_step_counter": lineage * ROLLOUT_STEPS,
        "sim_step_counter": lineage * ROLLOUT_STEPS * 10,
        "expected_adam_steps": lineage * 20,
        "experiment_transitions": experiment_updates * NUM_ENVS * ROLLOUT_STEPS,
    }


def budget(gpu: str) -> dict:
    if gpu not in ("A800", "5090"):
        raise ValueError("Choose A800 or 5090; local GPUs cannot be training targets")
    return {
        "gpu": gpu,
        "maximum_cloud_hours": 24 if gpu == "A800" else 60,
        "finalization_reserve_hours": 4 if gpu == "A800" else 8,
        "accounting": "cumulative powered-on cloud time including setup/eval/save/transfer",
        "shutdown": "remind user after required artifacts have transferred and verified",
    }


def budget_decision(gpu: str, *, spent_seconds: float, additional_updates: int,
                    seconds_per_update: float, overhead_seconds: float = 0.0) -> dict:
    """Admission estimate only, not a live process watchdog or billing meter.

    Runtime must persist cumulative elapsed time across all jobs/restarts and
    re-check before update blocks. Include observed save/eval costs as overhead.
    A 20% runtime margin is added without consuming finalization reserves.
    """
    if type(additional_updates) is not int or additional_updates < 0:
        raise ValueError("additional_updates must be a nonnegative integer")
    for name, value in (("spent_seconds", spent_seconds),
                        ("seconds_per_update", seconds_per_update),
                        ("overhead_seconds", overhead_seconds)):
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if seconds_per_update == 0:
        raise ValueError("A positive measured seconds_per_update is required")
    limits = budget(gpu)
    remaining = max(0, (limits["maximum_cloud_hours"] -
                       limits["finalization_reserve_hours"]) * 3600 - spent_seconds)
    estimate = (additional_updates * seconds_per_update + overhead_seconds) * 1.2
    return {"admit": estimate <= remaining,
            "remaining_before_finalization_seconds": remaining,
            "projected_seconds_with_margin": estimate,
            "margin_fraction": .2}


def campaign_manifest(gpu: str) -> dict:
    return {
        "schema_version": 1, "stage": 8,
        "status": "PREPARATION_ONLY_NOT_A_TRAINING_LAUNCHER",
        "baseline_commit": BASELINE_COMMIT,
        "source_checkpoint_sha256": SOURCE_SHA256,
        "num_envs": NUM_ENVS, "steps_per_env": ROLLOUT_STEPS,
        "profiles": [asdict(profile) for profile in PROFILES.values()],
        "screening": {"seeds": [TRAIN_SEEDS[0]], "updates_per_profile": 500,
                      "total_updates": 2500, "save_every": 100,
                      "evaluate_at": [0, 250, 500], "extra_save_at": [250]},
        "replication": {"selected_profiles": 2, "train_seeds": list(TRAIN_SEEDS),
                        "updates_per_lineage": 1500,
                        "incremental_updates_after_screening": 8000,
                        "gated_by_development_evidence": True},
        "conditional_extension": {
            "available": gpu == "5090", "selected_profiles": 1,
            "train_seeds": list(TRAIN_SEEDS), "updates_per_lineage": 4000,
            "incremental_updates": 7500 if gpu == "5090" else 0,
            "gated_by": "multi-seed development improvement and remaining-time estimate",
        },
        "initial_state": {
            **progress(0), "restore_actor_critic_normalizers": True,
            "restore_adam_moments": True, "learning_rate": INITIAL_LR,
            "lr_schedule": "adaptive", "desired_kl": .01,
            "source_seed": 20260921, "reset_episode_rng_rollout_hidden_state": True,
            "assistance_A_to_D": "restore source per-env level/hold after episode reset",
            "assistance_E": "set all level=5 and hold=0 after reset; assert all assistance wrenches zero",
            "exact_trajectory_resume": False,
        },
        "evaluation": {
            "nominal": "unchanged Stage 04, 64 copies, 10 s, continuous final 5 s",
            "development": {"seeds": [8101], "episodes_per_seed": 128},
            "held_out": {"seeds": [8201, 8202, 8203], "episodes_per_seed": 256,
                         "run_after_final_model_selection_only": True},
            "same_persisted_initial_states_across_models": True,
        },
        "budget": budget(gpu),
        "capacity": {"reuse_A800_evidence": gpu == "A800",
                     "new_5090_measurement_required": gpu == "5090",
                     "required_free_fraction": .15,
                     "automatically_reduce_environment_count": False},
        "formal_training_started": False,
    }
