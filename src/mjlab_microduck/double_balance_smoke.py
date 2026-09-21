"""Bounded Stage-06 acceptance using the registered double-balance runner.

The PPO implementation and frozen task are unchanged. Instrumentation fails on
non-finite data, including NaNs hidden by environment autoresets. This is a
training-chain check, never a policy-performance evaluation.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess
import time

import torch

from mjlab_microduck.double_balance_checkpoint import (
    audit_target_schema,
    file_sha256,
)

TASK_ID = "Mjlab-DoubleBalance-MicroDuck"
MIGRATED_SHA256 = "548758afc546e11d8f3f446a4bcc846283a669ff20f4048da7a3a7bb1332b98e"
NUM_ENVS = 64
ITERATIONS = 5
STEPS_PER_ENV = 24
SOURCE_ITERATION = 6999
SOURCE_ENV_STEPS = 168048
SEED = 20260921


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_finite(value, name):
    """Check nested runtime data, including TensorDicts and optimizer state."""
    if torch.is_tensor(value):
        require(bool(torch.isfinite(value).all()), f"non-finite {name}")
    elif hasattr(value, "items"):
        for key, child in value.items():
            require_finite(child, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            require_finite(child, f"{name}[{index}]")
    elif isinstance(value, float):
        require(math.isfinite(value), f"non-finite {name}")


def require_equal(actual, expected, name):
    """Exact state equality, allowing only CPU/CUDA device relocation."""
    if torch.is_tensor(expected):
        require(torch.is_tensor(actual), f"{name}: tensor missing")
        require(actual.dtype == expected.dtype, f"{name}: dtype mismatch")
        require(torch.equal(actual.detach().cpu(), expected.detach().cpu()),
                f"{name}: tensor mismatch")
    elif isinstance(expected, Mapping):
        require(set(actual) == set(expected), f"{name}: key mismatch")
        for key in expected:
            require_equal(actual[key], expected[key], f"{name}.{key}")
    elif isinstance(expected, (tuple, list)):
        require(len(actual) == len(expected), f"{name}: length mismatch")
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            require_equal(left, right, f"{name}[{index}]")
    else:
        require(actual == expected, f"{name}: {actual!r} != {expected!r}")


def expected_progress(start_iteration, start_steps, iterations=ITERATIONS):
    require(iterations > 0, "iterations must be positive")
    # rsl-rl 5.0.1 stores the last loop index, not the next loop index.
    return start_iteration + iterations - 1, start_steps + iterations * STEPS_PER_ENV


def resume_agent_config(agent, checkpoint):
    """Match the scheduler's Python LR to the saved Adam param-group LR.

    rsl-rl restores optimizer state but not PPO.learning_rate. Constructing a
    reload runner with the saved rate avoids silently restarting its adaptive
    schedule at the recipe's 1e-3. The original agent config is not mutated.
    """
    result = deepcopy(agent)
    groups = checkpoint["optimizer_state_dict"]["param_groups"]
    require(len(groups) == 1, "expected one Adam parameter group")
    lr = float(groups[0]["lr"])
    require(math.isfinite(lr) and lr > 0, "invalid checkpoint learning rate")
    result["algorithm"]["learning_rate"] = lr
    return result


def check_observations(obs, num_envs=NUM_ENVS):
    require(tuple(obs["actor"].shape) == (num_envs, 61), "actor observations must be Nx61")
    require(tuple(obs["critic"].shape) == (num_envs, 85), "critic observations must be Nx85")
    require_finite(obs, "observations")


def check_optimizer(optimizer, expected_steps=None):
    require_finite(optimizer.state_dict(), "optimizer")
    if expected_steps is None:
        return
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    require(len(optimizer.state) == len(parameters), "Adam moments missing for parameters")
    for parameter in parameters:
        state = optimizer.state[parameter]
        for key in ("exp_avg", "exp_avg_sq"):
            require(state[key].shape == parameter.shape, f"Adam {key} shape mismatch")
        require(int(state["step"].item()) == expected_steps, "Adam step count mismatch")


def check_loaded_runner(runner, raw, checkpoint):
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
        require_equal(runner.alg.save()[key], checkpoint[key], key)
    require(runner.current_learning_iteration == checkpoint["iter"], "iteration not restored")
    expected_steps = checkpoint["infos"]["env_state"]["common_step_counter"]
    require(raw.common_step_counter == expected_steps, "environment counter not restored")
    lr = checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]
    require(runner.alg.learning_rate == lr, "PPO scheduler learning rate differs from Adam")
    require_finite(runner.alg.save(), "loaded_runner")


@contextmanager
def monitor_smoke(runner, env):
    """Observe actual runner calls; never replace rollout or update algorithms."""
    alg = runner.alg
    original_step, original_returns, original_update = env.step, alg.compute_returns, alg.update
    evidence = {"vector_steps": 0, "return_passes": 0, "optimizer_steps": 0,
                "nan_terminations": 0, "iterations": []}

    def step(actions):
        require_finite(actions, "actions")
        require(tuple(actions.shape) == (NUM_ENVS, 14), "action shape mismatch")
        result = original_step(actions)
        obs, rewards, dones, _ = result
        check_observations(obs)
        require_finite((rewards, dones), "step_output")
        nan_count = int(env.unwrapped.termination_manager.get_term("nan_state").sum().item())
        evidence["nan_terminations"] += nan_count
        require(nan_count == 0, "nan_state termination occurred (autoreset could mask it)")
        evidence["vector_steps"] += 1
        require(evidence["vector_steps"] <= ITERATIONS * STEPS_PER_ENV, "smoke step budget exceeded")
        return result

    def compute_returns(obs):
        require(alg.storage.step == STEPS_PER_ENV, "incomplete PPO rollout")
        result = original_returns(obs)
        for key in ("observations", "actions", "rewards", "values", "returns",
                    "advantages", "actions_log_prob", "distribution_params",
                    "saved_hidden_state_a", "saved_hidden_state_c"):
            require_finite(getattr(alg.storage, key), f"rollout.{key}")
        evidence["return_passes"] += 1
        return result

    def before_optimizer_step(optimizer, args, kwargs):
        require(evidence["optimizer_steps"] < ITERATIONS * alg.num_learning_epochs * alg.num_mini_batches,
                "smoke optimizer budget exceeded")
        gradients = [p.grad for group in optimizer.param_groups for p in group["params"]]
        require(all(g is not None for g in gradients), "a model parameter received no gradient")
        require_finite(gradients, "gradients")
        evidence["optimizer_steps"] += 1

    def update():
        require(len(evidence["iterations"]) < ITERATIONS, "smoke iteration budget exceeded")
        losses = original_update()
        require(bool(losses), "PPO returned no losses")
        require_finite(losses, "losses")
        require_finite(alg.save(), "updated_models_and_optimizer")
        require(bool((alg.actor.output_std > 0).all()), "actor standard deviation is not positive")
        check_optimizer(alg.optimizer, evidence["optimizer_steps"])
        row = {"update": len(evidence["iterations"]) + 1,
               "losses": {key: float(value) for key, value in losses.items()},
               "learning_rate": float(alg.learning_rate),
               "common_step_counter": int(env.unwrapped.common_step_counter)}
        evidence["iterations"].append(row)
        print("Stage06Update=" + json.dumps(row, allow_nan=False), flush=True)
        return losses

    env.step, alg.compute_returns, alg.update = step, compute_returns, update
    hook = alg.optimizer.register_step_pre_hook(before_optimizer_step)
    try:
        yield evidence
    finally:
        hook.remove()
        env.step, alg.compute_returns, alg.update = original_step, original_returns, original_update


def cloud_configuration_preflight(agent):
    """Check the 4096-env config and quantify known buffer storage only.

    Does not allocate a cloud simulator or claim that 4096 envs fit a GPU.
    Simulation contacts, temporary activations and CUDA/Warp allocations are
    excluded; measured hardware capacity remains a separate cloud-side check.
    """
    from mjlab.tasks.registry import load_env_cfg

    cfg = load_env_cfg(TASK_ID)
    cfg.scene.num_envs = 4096
    require(cfg.sim.mujoco.timestep == 0.002 and cfg.decimation == 10, "frozen timing changed")
    require(cfg.scene.num_envs % agent["algorithm"]["num_mini_batches"] == 0,
            "cloud env count must divide into recurrent mini-batches")
    require(agent["num_steps_per_env"] == STEPS_PER_ENV, "unexpected rollout length")
    require(agent["actor"]["rnn_hidden_dim"] == 256, "unexpected LSTM width")
    # FP32: obs, actions, reward/value/logprob/return/advantage, Normal mean/std,
    # plus one byte per done; LSTM h/c are stored for every rollout step.
    scalar_bytes = (61 + 85 + 14 + 5 + 2 * 14) * 4 + 1
    lstm_bytes = 2 * 1 * 256 * 4
    return {
        "configuration_status": "PASS", "target_num_envs": 4096,
        "steps_per_env": STEPS_PER_ENV, "transitions_per_iteration": 4096 * STEPS_PER_ENV,
        "known_rollout_and_lstm_buffer_bytes": 4096 * STEPS_PER_ENV * (scalar_bytes + lstm_bytes),
        "physics_dt": cfg.sim.mujoco.timestep, "control_dt": 0.02,
        "capacity_status": "NOT_MEASURED_ON_CLOUD_GPU",
        "exclusions": ["simulation_and_contacts", "gradient_activations", "CUDA_and_Warp_allocations"],
        "long_training_started": False,
    }


def write_report(path, report):
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _run(args, output):
    # Import simulator/task registration only after checking user overrides.
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    import mjlab_microduck.tasks  # noqa: F401

    checkpoint_path = args.checkpoint.resolve()
    require(file_sha256(checkpoint_path) == MIGRATED_SHA256, "migrated checkpoint SHA-256 mismatch")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    audit_target_schema(checkpoint)
    require(checkpoint["iter"] == SOURCE_ITERATION, "unexpected source iteration")
    require(checkpoint["infos"]["env_state"]["common_step_counter"] == SOURCE_ENV_STEPS,
            "unexpected source environment progress")
    require(torch.cuda.is_available(), "Stage 06 requires a CUDA device for the 64-env acceptance")
    device = "cuda:0"
    configure_torch_backends()
    torch.set_num_threads(4)
    torch.cuda.reset_peak_memory_stats()

    cfg = load_env_cfg(TASK_ID)
    cfg.scene.num_envs, cfg.seed = NUM_ENVS, SEED
    require(cfg.sim.mujoco.timestep == 0.002 and cfg.decimation == 10, "frozen timing changed")
    agent = asdict(load_rl_cfg(TASK_ID))
    agent.update(seed=SEED, logger="tensorboard", upload_model=False, max_iterations=ITERATIONS,
                 save_interval=ITERATIONS, run_name="stage06_smoke", check_for_nan=True)
    require(agent["num_steps_per_env"] == STEPS_PER_ENV, "unexpected rollout length")
    require(agent["algorithm"]["learning_rate"] == 1e-3, "initial learning rate must be 1e-3")
    cloud = cloud_configuration_preflight(agent)
    dump_yaml(output / "params/env.yaml", asdict(cfg))
    dump_yaml(output / "params/agent.yaml", deepcopy(agent))
    report = {
        "status": "RUNNING", "task_id": TASK_ID, "num_envs": NUM_ENVS,
        "seed": SEED, "mode": args.mode, "input_checkpoint": str(checkpoint_path),
        "input_sha256": MIGRATED_SHA256, "input_size_bytes": checkpoint_path.stat().st_size,
        "versions": {key: version(key) for key in ("torch", "mjlab", "rsl-rl-lib", "warp-lang", "mujoco")},
        "gpu": torch.cuda.get_device_name(0), "cloud_preflight": cloud,
        "formal_training_started": False, "formal_success_rate": None,
    }
    raw = ManagerBasedRlEnv(cfg=cfg, device=device)
    env = None
    runner = None
    try:
        env = RslRlVecEnvWrapper(raw, clip_actions=agent["clip_actions"])
        require(env.num_actions == 14, "action dimension must remain 14")
        runner_cls = load_runner_cls(TASK_ID)
        require(runner_cls is not None, "registered runner missing")
        runner = runner_cls(env, deepcopy(agent), str(output / "logs"), device)
        runner.load(str(checkpoint_path), strict=True, map_location=device)
        check_loaded_runner(runner, raw, checkpoint)
        require(not runner.alg.optimizer.state, "migrated Adam must have empty moments")
        env.reset()  # Apply restored curriculum progress before first rollout.
        require(raw.common_step_counter == SOURCE_ENV_STEPS, "reset changed global progress")
        obs = env.get_observations()
        check_observations(obs)
        report["load"] = {
            "status": "PASS", "runner_class": f"{runner_cls.__module__}.{runner_cls.__name__}",
            "actor_dimension": 61, "critic_dimension": 85, "action_dimension": 14,
            "iteration": runner.current_learning_iteration,
            "common_step_counter": raw.common_step_counter,
            "optimizer_state_entries": len(runner.alg.optimizer.state),
            "learning_rate": runner.alg.learning_rate,
            "restored_environment_state_keys": sorted(checkpoint["infos"]["env_state"]),
            "inherited_ball_hold_levels": list(cfg.actions["ball_hold"].levels),
            "per_environment_curriculum_state_restored": False,
        }
        write_report(output / "load-report.json", {**report, "status": "LOAD_ONLY_PASS"})
        print("Stage06RunnerLoad=" + json.dumps(report["load"]), flush=True)
        if args.mode == "load":
            report["status"] = "LOAD_ONLY_PASS"
            return report

        runner.add_git_repo_to_log(__file__)
        started = time.perf_counter()
        with monitor_smoke(runner, env) as evidence:
            runner.learn(num_learning_iterations=ITERATIONS, init_at_random_ep_len=True)
        torch.cuda.synchronize()
        report["smoke_seconds"] = time.perf_counter() - started
        expected_iter, expected_steps = expected_progress(SOURCE_ITERATION, SOURCE_ENV_STEPS)
        require(len(evidence["iterations"]) == ITERATIONS, "not exactly five PPO updates")
        require(evidence["return_passes"] == ITERATIONS, "return pass count mismatch")
        require(evidence["vector_steps"] == ITERATIONS * STEPS_PER_ENV, "rollout step count mismatch")
        require(runner.current_learning_iteration == expected_iter, "final iteration mismatch")
        require(raw.common_step_counter == expected_steps, "final global step count mismatch")
        required_adam_steps = ITERATIONS * runner.alg.num_learning_epochs * runner.alg.num_mini_batches
        require(evidence["optimizer_steps"] == required_adam_steps, "optimizer update count mismatch")
        check_optimizer(runner.alg.optimizer, required_adam_steps)

        smoke_path = output / "logs" / f"model_{expected_iter}.pt"
        saved = torch.load(smoke_path, map_location="cpu", weights_only=False)
        check_loaded_runner(runner, raw, saved)
        require_finite(saved, "saved_checkpoint")
        actor_tail = saved["actor_state_dict"]["rnn.rnn.weight_ih_l0"][:, 55:61]
        critic_tail = saved["critic_state_dict"]["mlp.0.weight"][:, 76:85]
        require(bool(torch.count_nonzero(actor_tail)), "actor new inputs did not learn")
        require(bool(torch.count_nonzero(critic_tail)), "critic new inputs did not learn")

        # Compare four inference steps from reset hidden state. A checkpoint is
        # not a simulator/RNG/episode snapshot; no exact trajectory resume claim.
        probe = env.get_observations().clone()
        reference_outputs = []
        runner.alg.eval_mode()
        runner.alg.actor.reset()
        with torch.inference_mode():
            for _ in range(4):
                action, value = runner.alg.actor(probe).clone(), runner.alg.critic(probe).clone()
                hidden = tuple(t.clone() for t in runner.alg.actor.get_hidden_state())
                reference_outputs.append((action, value, hidden))
        # A fresh registered runner, with fresh networks/storage/optimizer.
        reloaded = runner_cls(env, resume_agent_config(agent, saved), None, device)
        raw.common_step_counter = 0  # Prove load restores progress, not just preserves it.
        reloaded.load(str(smoke_path), strict=True, map_location=device)
        check_loaded_runner(reloaded, raw, saved)
        reloaded.alg.eval_mode()
        reloaded.alg.actor.reset()
        with torch.inference_mode():
            for index, (action, value, hidden) in enumerate(reference_outputs):
                require_equal(reloaded.alg.actor(probe), action, f"reload_action_{index}")
                require_equal(reloaded.alg.critic(probe), value, f"reload_value_{index}")
                require_equal(reloaded.alg.actor.get_hidden_state(), hidden, f"reload_hidden_{index}")
        check_optimizer(reloaded.alg.optimizer, required_adam_steps)
        require(file_sha256(checkpoint_path) == MIGRATED_SHA256, "migration input was modified")

        report.update(
            status="PASS", smoke=evidence, completed_iterations=ITERATIONS,
            final_iteration=expected_iter, final_common_step_counter=expected_steps,
            transitions=NUM_ENVS * ITERATIONS * STEPS_PER_ENV,
            actor_new_columns_max_abs=float(actor_tail.abs().max()),
            critic_new_columns_max_abs=float(critic_tail.abs().max()),
            checkpoint={"path": str(smoke_path), "size_bytes": smoke_path.stat().st_size,
                        "sha256": file_sha256(smoke_path)},
            reload={"status": "PASS", "strict_model_and_optimizer_equality": True,
                    "recurrent_action_hidden_and_value_equality": True,
                    "scheduler_learning_rate": reloaded.alg.learning_rate,
                    "exact_simulator_or_rng_resume": False},
            input_unmodified=True,
            peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(),
            automatic_onnx_files=[str(p) for p in output.rglob("*.onnx")],
            onnx_deployment_validated=False,
        )
        return report
    finally:
        if runner is not None:
            writer = getattr(runner.logger, "writer", None)
            if writer is not None:
                writer.close()
        if env is not None:
            env.close()
        else:
            raw.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", required=True, type=Path,
                        help="New run directory outside the Git checkout (must not exist)")
    parser.add_argument("--mode", choices=("load", "smoke"), default="load")
    args = parser.parse_args()
    # Task imports read these variables at import time: fail rather than silently
    # testing a modified physics/reward recipe.
    overrides = sorted(key for key in os.environ if key.startswith("MICRODUCK_"))
    require(not overrides, f"unset task overrides before acceptance: {overrides}")
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "Stage 06 is single GPU only")
    repo = Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    require(not output.is_relative_to(repo), "place Stage-06 outputs outside the checkout")
    require(not output.exists(), "output directory already exists; use a new run directory")
    require(args.checkpoint.is_file(), "migration checkpoint does not exist")
    require(file_sha256(args.checkpoint) == MIGRATED_SHA256, "migrated checkpoint SHA-256 mismatch")
    output.mkdir(parents=True)
    print(f"Stage06OutputDirectory={output}", flush=True)
    try:
        report = _run(args, output)
        report["created_utc"] = datetime.now(timezone.utc).isoformat()
        report["git_head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        write_report(output / "acceptance.json", report)
        print("Stage06AcceptanceBegin=", flush=True)
        print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False), flush=True)
        print(f"Stage06Acceptance={report['status']}", flush=True)
        print("Stage06AcceptanceEnd=", flush=True)
    except Exception as exc:
        write_report(output / "failure.json", {"status": "FAIL", "error": str(exc), "mode": args.mode})
        raise


if __name__ == "__main__":
    main()
