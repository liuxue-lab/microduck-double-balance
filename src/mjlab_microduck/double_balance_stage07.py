"""Stage 07 initialization and bounded cloud capacity measurement.

Uses the registered task, runner, and PPO unchanged. Capacity policies are
discarded; a formal run must initialize from the Stage 05 checkpoint again.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch

from mjlab_microduck.double_balance_checkpoint import audit_target_schema, file_sha256
from mjlab_microduck.double_balance_smoke import (
    MIGRATED_SHA256, SOURCE_ENV_STEPS, SOURCE_ITERATION, STEPS_PER_ENV, TASK_ID,
    check_loaded_runner, check_observations, check_optimizer,
    require, require_equal, require_finite,
)

BASELINE_COMMIT = "e3f26e3cbf633c70c75d4d073dbeb8c47c4f5bd1"
HOLD_LEVELS = (1.0, 0.5, 0.25, 0.1, 0.03, 0.0)
SEED = 20260921
FROZEN_PATHS = (
    "src/mjlab_microduck/tasks", "src/mjlab_microduck/robot",
    "src/mjlab_microduck/actuator", "pyproject.toml", "uv.lock",
)


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_frozen_source(repo):
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()
    require(git("rev-parse", "stage-06-complete^{}") == BASELINE_COMMIT,
            "Stage 06 annotated baseline differs")
    require(git("cat-file", "-t", "stage-06-complete") == "tag",
            "Stage 06 tag must be annotated")
    subprocess.run(["git", "merge-base", "--is-ancestor", BASELINE_COMMIT, "HEAD"],
                   cwd=repo, check=True)
    changed = git("diff", "--name-only", BASELINE_COMMIT, "--", *FROZEN_PATHS)
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *FROZEN_PATHS)
    require(not changed and not untracked, "frozen physics/task/dependencies changed")
    return git("rev-parse", "HEAD")


def reset_stage07_progress(runner, env):
    """Reset progress before the first new-stage episode, preserving weights.

    Do not randomize episode_length_buf: the inherited curriculum uses episode
    length for promotion, so fabricated initial ages would advance assistance
    before the policy has actually survived that duration.
    """
    raw = env.unwrapped
    require(tuple(raw.cfg.actions["ball_hold"].levels) == HOLD_LEVELS,
            "inherited assistance table changed")
    require(not runner.alg.optimizer.state, "Stage 05 Adam moments must be empty")
    runner.current_learning_iteration = 0
    raw.common_step_counter = 0
    raw._sim_step_counter = 0
    raw.episode_length_buf.zero_()
    raw._basketball_state.level.zero_()
    raw._basketball_state.hold.fill_(HOLD_LEVELS[0])
    runner.alg.learning_rate = 1e-3
    for group in runner.alg.optimizer.param_groups:
        group["lr"] = 1e-3
    env.reset()
    runner.alg.actor.reset()
    require(raw.common_step_counter == 0, "reset changed the global counter")
    require(not bool(raw.episode_length_buf.any()), "initial episode ages must be zero")
    require(not bool(raw._basketball_state.level.any()), "initial assistance level must be zero")
    require(bool((raw._basketball_state.hold == 1).all()), "initial hold must be one")


def progress_after_updates(completed_updates):
    require(type(completed_updates) is int and completed_updates >= 0,
            "completed updates must be a nonnegative integer")
    return {
        "completed_updates": completed_updates,
        "last_completed_iteration": completed_updates - 1,
        "next_iteration": completed_updates,
        "common_step_counter": completed_updates * STEPS_PER_ENV,
    }


def summarize_iterations(rows, warmup, num_envs):
    require(type(warmup) is int and warmup >= 1, "at least one warmup update required")
    measured = rows[warmup:]
    require(len(measured) >= 3, "at least three measured updates required")
    seconds = [float(row["iteration_seconds"]) for row in measured]
    require(all(math.isfinite(x) and x > 0 for x in seconds), "invalid measured duration")
    mean = statistics.mean(seconds)
    return {
        "warmup_updates_excluded": warmup,
        "measured_updates": len(measured),
        "mean_iteration_seconds": mean,
        "median_iteration_seconds": statistics.median(seconds),
        "max_iteration_seconds": max(seconds),
        "transitions_per_second": num_envs * STEPS_PER_ENV / mean,
        "projected_2000_updates_hours_excluding_setup_eval_save": mean * 2000 / 3600,
        "measurement_scope": "rollout + PPO + finite checks + iteration JSON logging; no checkpoint/eval",
    }


@contextmanager
def monitor_updates(runner, env, output, updates, *, start_completed=0, on_update=None,
                    prefix="Stage07CapacityUpdate"):
    """Checks used in capacity and intended for the formal training entry.

    There is one synchronization per environment step for the additional
    action/Inf/NaN-termination checks. Gradient checks stay on the device until
    the end of an update. No optimizer or rollout algorithm is replaced.
    """
    alg, raw = runner.alg, env.unwrapped
    original_step, original_returns = env.step, alg.compute_returns
    original_update, original_log = alg.update, runner.logger.log
    evidence = {"vector_steps": 0, "optimizer_steps": 0, "return_passes": 0,
                "nan_terminations": 0, "iterations": []}
    gradient_ok = torch.ones((), dtype=torch.bool, device=raw.device)
    last = time.perf_counter()
    stream = Path(output, "iterations.jsonl").open("x")

    def step(actions):
        require(tuple(actions.shape) == (env.num_envs, 14), "action shape changed")
        result = original_step(actions)
        obs, rewards, dones, _ = result
        flags = [torch.isfinite(x).all() for x in (actions, rewards, dones, *obs.values())]
        flags.append(~raw.termination_manager.get_term("nan_state").any())
        require(bool(torch.stack(flags).all()), "non-finite rollout or nan_state termination")
        evidence["vector_steps"] += 1
        require(evidence["vector_steps"] <= updates * STEPS_PER_ENV, "session step limit exceeded")
        return result

    def returns(obs):
        require(alg.storage.step == STEPS_PER_ENV, "incomplete rollout")
        result = original_returns(obs)
        for key in ("returns", "advantages", "values", "actions_log_prob"):
            require_finite(getattr(alg.storage, key), f"rollout.{key}")
        evidence["return_passes"] += 1
        return result

    def before_optimizer(optimizer, args, kwargs):
        del args, kwargs
        gradients = [p.grad for group in optimizer.param_groups for p in group["params"]]
        require(all(g is not None for g in gradients), "missing parameter gradient")
        gradient_ok.logical_and_(torch.stack([torch.isfinite(g).all() for g in gradients]).all())
        evidence["optimizer_steps"] += 1

    def update():
        gradient_ok.fill_(True)
        losses = original_update()
        require(bool(gradient_ok), "non-finite gradient")
        require(bool(losses), "PPO returned no losses")
        require_finite(losses, "losses")
        require_finite(alg.save(), "updated model and optimizer")
        return losses

    def log(**kwargs):
        nonlocal last
        torch.cuda.synchronize()
        now = time.perf_counter()
        session_completed = len(evidence["iterations"]) + 1
        completed = start_completed + session_completed
        expected = progress_after_updates(completed)
        require(kwargs["it"] == expected["last_completed_iteration"], "runner iteration mismatch")
        require(raw.common_step_counter == expected["common_step_counter"], "global progress mismatch")
        require(session_completed <= updates, "session update limit exceeded")
        row = {
            "completed_updates": completed, "runner_iteration": kwargs["it"],
            "common_step_counter": raw.common_step_counter,
            "iteration_seconds": now - last,
            "runner_collect_seconds": float(kwargs["collect_time"]),
            "runner_learn_seconds": float(kwargs["learn_time"]),
            "losses": {k: float(v) for k, v in kwargs["loss_dict"].items()},
            "learning_rate": float(alg.learning_rate),
            "hold_level_counts": torch.bincount(raw._basketball_state.level,
                                                minlength=len(HOLD_LEVELS)).cpu().tolist(),
        }
        if on_update is not None:
            # Capture the native logger's episode summaries before it clears them.
            episode = {}
            for item in runner.logger.ep_extras:
                for key, value in item.items():
                    tensor = torch.as_tensor(value, device=raw.device).reshape(-1).float()
                    require_finite(tensor, key)
                    episode.setdefault(key, []).append(tensor)
            row["episode"] = {k: float(torch.cat(v).mean()) for k, v in episode.items()}
            row["mean_episode_return_last_100"] = (
                statistics.mean(runner.logger.rewbuffer) if runner.logger.rewbuffer else None)
            row["mean_episode_seconds_last_100"] = (
                statistics.mean(runner.logger.lenbuffer) * raw.step_dt
                if runner.logger.lenbuffer else None)
        evidence["iterations"].append(row)
        line = json.dumps(row, allow_nan=False)
        stream.write(line + "\n")
        stream.flush()
        print(prefix + "=" + line, flush=True)
        result = original_log(**kwargs)
        if on_update is not None:
            on_update(row, evidence)
        # Checkpoint/evaluation time is tracked by the formal job wall clock.
        last = time.perf_counter()
        return result

    env.step, alg.compute_returns, alg.update, runner.logger.log = step, returns, update, log
    hook = alg.optimizer.register_step_pre_hook(before_optimizer)
    try:
        yield evidence
    finally:
        hook.remove()
        env.step, alg.compute_returns, alg.update, runner.logger.log = (
            original_step, original_returns, original_update, original_log)
        stream.close()


def capacity_worker(args):
    require(not any(k.startswith("MICRODUCK_") for k in os.environ), "unset MICRODUCK task overrides")
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "single GPU only")
    require(Path("/root/autodl-tmp").is_dir(), "capacity runs must execute in the AutoDL cloud container")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one CUDA GPU required")
    require(args.expected_gpu in torch.cuda.get_device_name(0), "unexpected cloud GPU")
    expected_versions = {"mjlab": "1.3.0", "mujoco": "3.10.0", "mujoco-warp": "3.8.1",
                         "warp-lang": "1.12.0", "rsl-rl-lib": "5.0.1"}
    require({name: version(name) for name in expected_versions} == expected_versions,
            "cloud dependency versions differ from the accepted lock")
    require(torch.__version__.split("+")[0] == "2.9.1" and torch.version.cuda == "12.8",
            "expected Torch 2.9.1 with CUDA 12.8")
    require(args.num_envs in (512, 1024, 2048, 4096), "capacity environment count is not allowed")
    require(1 <= args.warmup <= 5 and 3 <= args.measure <= 20, "bounded capacity update count required")
    require(file_sha256(args.checkpoint) == MIGRATED_SHA256, "Stage 05 checkpoint SHA256 mismatch")
    repo = Path(__file__).resolve().parents[2]
    head = verify_frozen_source(repo)
    output = args.output.resolve()
    require(not output.is_relative_to(repo), "place capacity results outside Git")
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    audit_target_schema(checkpoint)
    require(checkpoint["iter"] == SOURCE_ITERATION, "unexpected Stage 05 source iteration")
    require(checkpoint["infos"]["env_state"]["common_step_counter"] == SOURCE_ENV_STEPS,
            "unexpected Stage 05 source counter")
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    import mjlab_microduck.tasks  # noqa: F401

    configure_torch_backends()
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    torch.cuda.reset_peak_memory_stats()
    cfg = load_env_cfg(TASK_ID)
    cfg.scene.num_envs, cfg.seed = args.num_envs, SEED
    require(cfg.sim.mujoco.timestep == 0.002 and cfg.decimation == 10, "frozen timing changed")
    require(tuple(cfg.actions["ball_hold"].levels) == HOLD_LEVELS, "assistance table changed")
    agent = asdict(load_rl_cfg(TASK_ID))
    agent.update(seed=SEED, logger="tensorboard", upload_model=False,
                 max_iterations=args.warmup + args.measure,
                 run_name="stage07_capacity", check_for_nan=True)
    require(agent["num_steps_per_env"] == STEPS_PER_ENV, "rollout length changed")
    require(agent["algorithm"]["learning_rate"] == 1e-3, "initial LR changed")
    require(args.num_envs % agent["algorithm"]["num_mini_batches"] == 0, "minibatch mismatch")
    dump_yaml(output / "params/env.yaml", asdict(cfg))
    dump_yaml(output / "params/agent.yaml", deepcopy(agent))
    report = {
        "status": "RUNNING", "purpose": "Stage 07 cloud capacity only", "task_id": TASK_ID,
        "gpu": torch.cuda.get_device_name(0), "num_envs": args.num_envs, "seed": SEED,
        "cuda_visible_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
        "git_head": head, "stage06_baseline": BASELINE_COMMIT,
        "runtime_source_sha256": file_sha256(__file__),
        "versions": {k: version(k) for k in ("torch", "mjlab", "mujoco", "mujoco-warp", "warp-lang", "rsl-rl-lib")},
        "input_sha256": MIGRATED_SHA256, "source_iteration": SOURCE_ITERATION,
        "source_common_step_counter": SOURCE_ENV_STEPS,
        "initial_stage07_iteration": 0, "initial_common_step_counter": 0,
        "initial_hold_level": 0, "initial_hold": 1.0, "initial_learning_rate": 1e-3,
        "initial_episode_lengths_randomized": False,
        "formal_training_started": False, "checkpoint_saved": False,
    }
    write_json(output / "report.json", report)
    raw = env = None
    started = time.perf_counter()
    try:
        raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        env = RslRlVecEnvWrapper(raw, clip_actions=agent["clip_actions"])
        runner_cls = load_runner_cls(TASK_ID)
        require(runner_cls is not None, "registered runner missing")
        # log_dir=None suppresses TensorBoard and automatic checkpoint/ONNX saves.
        runner = runner_cls(env, deepcopy(agent), None, "cuda:0")
        runner.load(str(args.checkpoint), strict=True, map_location="cuda:0")
        check_loaded_runner(runner, raw, checkpoint)
        reset_stage07_progress(runner, env)
        for key in ("actor_state_dict", "critic_state_dict"):
            require_equal(runner.alg.save()[key], checkpoint[key], key)
        check_observations(env.get_observations(), args.num_envs)
        report["strict_initialization"] = "PASS"
        report["build_and_load_seconds"] = time.perf_counter() - started
        write_json(output / "report.json", report)
        count = args.warmup + args.measure
        with monitor_updates(runner, env, output, count) as evidence:
            runner.learn(num_learning_iterations=count, init_at_random_ep_len=False)
        torch.cuda.synchronize()
        require(evidence["return_passes"] == count, "return count mismatch")
        require(len(evidence["iterations"]) == count, "update count mismatch")
        require(evidence["vector_steps"] == count * STEPS_PER_ENV, "step count mismatch")
        adam_steps = count * runner.alg.num_learning_epochs * runner.alg.num_mini_batches
        require(evidence["optimizer_steps"] == adam_steps, "optimizer count mismatch")
        check_optimizer(runner.alg.optimizer, adam_steps)
        require(file_sha256(args.checkpoint) == MIGRATED_SHA256, "input checkpoint changed")
        report.update(status="PASS", **summarize_iterations(evidence["iterations"], args.warmup, args.num_envs),
                      progress=progress_after_updates(count), optimizer_steps=adam_steps,
                      nan_terminations=0, finite_checks="PASS",
                      torch_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                      torch_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                      torch_memory_note="Torch excludes Warp; use the parent GPU-wide memory samples for capacity.")
    except BaseException as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}",
                      is_out_of_memory=isinstance(exc, torch.cuda.OutOfMemoryError))
        raise
    finally:
        report["elapsed_seconds"] = time.perf_counter() - started
        report["created_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "report.json", report)
        if env is not None:
            env.close()
        elif raw is not None:
            raw.close()
    print("Stage07CapacityWorker=PASS", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, required=True)
    parser.add_argument("--expected-gpu", choices=("A800", "5090"), default="A800")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--measure", type=int, default=12)
    capacity_worker(parser.parse_args())


if __name__ == "__main__":
    main()
