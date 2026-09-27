"""Stage 07 cloud training, explicit curriculum resume, and frozen nominal evaluation.

The registered runner still owns rollout and PPO. This module owns provenance,
completed-update checkpoint names, and evaluation at update boundaries.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import time

import torch

from mjlab_microduck.double_balance_checkpoint import file_sha256
from mjlab_microduck.double_balance_smoke import resume_agent_config
from mjlab_microduck.double_balance_stage07 import (
    BASELINE_COMMIT, HOLD_LEVELS, MIGRATED_SHA256, SEED, STEPS_PER_ENV, TASK_ID,
    check_loaded_runner, check_observations, check_optimizer, monitor_updates,
    progress_after_updates, require, require_equal, require_finite,
    reset_stage07_progress, verify_frozen_source, write_json,
)

ROOT = Path("/root/autodl-tmp/microduck-double-balance")
ARTIFACTS = ROOT / "artifacts/double-balance-stage07"
DEFAULT_CHECKPOINT = ROOT / "artifacts/double-balance-stage05/checkpoint.pt"
EVALUATION_PROTOCOL = "stage04-frozen-play-10s-first-episode-v1"
VERSIONS = {"mjlab": "1.3.0", "mujoco": "3.10.0", "mujoco-warp": "3.8.1",
            "warp-lang": "1.12.0", "rsl-rl-lib": "5.0.1"}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def validate_capacity(summary, num_envs):
    require(summary["status"] == "PASS", "capacity preflight has not passed")
    require(summary["checkpoint_sha256"] == MIGRATED_SHA256, "capacity source mismatch")
    rows = [r for r in summary["matrix"] if r["num_envs"] == num_envs]
    require(len(rows) == 1, "requested scale was not measured")
    row = rows[0]
    require(row["status"] == "PASS" and row["exit_code"] == 0, "capacity case failed")
    require(row["eligible_with_15_percent_headroom"] and
            row["sampled_peak_gpu_used_mib"] <= .85 * row["gpu_total_mib"],
            "insufficient measured GPU headroom")
    require(math.isfinite(row["transitions_per_second"]) and row["transitions_per_second"] > 0,
            "invalid capacity throughput")
    return row


def cloud_preflight():
    require(ROOT.is_dir(), "formal training and evaluation must run on AutoDL")
    require(not any(k.startswith("MICRODUCK_") for k in os.environ), "unset task overrides")
    require(int(os.environ.get("WORLD_SIZE", "1")) == 1, "single GPU required")
    require(torch.cuda.is_available() and torch.cuda.device_count() == 1, "one CUDA GPU required")
    require("A800" in torch.cuda.get_device_name(0), "expected the selected A800")
    require(torch.cuda.get_device_properties(0).total_memory >= 75 * 1024**3,
            "expected full A800 80GB allocation")
    require({k: version(k) for k in VERSIONS} == VERSIONS, "locked dependencies differ")
    require(torch.__version__.split("+")[0] == "2.9.1" and torch.version.cuda == "12.8",
            "expected Torch 2.9.1 CUDA 12.8")
    repo = Path(__file__).resolve().parents[2]
    head = verify_frozen_source(repo)
    require(not subprocess.check_output(["git", "status", "--porcelain"], cwd=repo, text=True).strip(),
            "deploy a clean committed source tree")
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    return head


def checkpoint_metadata(runner, raw, completed, head):
    progress = progress_after_updates(completed)
    require(raw.common_step_counter == progress["common_step_counter"], "checkpoint counter mismatch")
    require(raw._sim_step_counter == raw.common_step_counter * raw.cfg.decimation,
            "checkpoint physics counter mismatch")
    lr = float(runner.alg.learning_rate)
    require(all(float(g["lr"]) == lr for g in runner.alg.optimizer.param_groups),
            "PPO and Adam learning rates disagree")
    state = {
        "schema_version": 1, "task_id": TASK_ID, "stage06_baseline": BASELINE_COMMIT,
        "initialization_sha256": MIGRATED_SHA256, "git_head": head,
        "seed": SEED, "num_envs": raw.num_envs, **progress,
        "sim_step_counter": raw._sim_step_counter, "learning_rate": lr,
        "hold_levels": list(HOLD_LEVELS),
        "level": raw._basketball_state.level.detach().cpu().clone(),
        "hold": raw._basketball_state.hold.detach().cpu().clone(),
        "exact_trajectory_resume": False,
        "resume_rule": "restore curriculum and optimizer; start new episodes and reset recurrent state/RNG",
    }
    return state


def validate_training_checkpoint(checkpoint, num_envs=None):
    state = checkpoint["infos"]["stage07"]
    require(state["schema_version"] == 1 and state["task_id"] == TASK_ID,
            "not a Stage 07 training checkpoint")
    require(state["stage06_baseline"] == BASELINE_COMMIT and
            state["initialization_sha256"] == MIGRATED_SHA256, "checkpoint lineage mismatch")
    require(state["seed"] == SEED, "resume seed differs")
    if num_envs is not None:
        require(state["num_envs"] == num_envs, "resume requires the same environment count")
    progress = progress_after_updates(state["completed_updates"])
    for key, value in progress.items():
        require(state[key] == value, f"invalid resume {key}")
    require(checkpoint["iter"] == progress["last_completed_iteration"], "saved loop index differs")
    require(checkpoint["infos"]["env_state"]["common_step_counter"] == progress["common_step_counter"],
            "saved global counter differs")
    require(state["sim_step_counter"] == progress["common_step_counter"] * 10,
            "saved physics counter differs")
    require(tuple(state["hold_levels"]) == HOLD_LEVELS, "saved assistance table differs")
    level, hold = state["level"], state["hold"]
    require(level.dtype == torch.long and tuple(level.shape) == (state["num_envs"],),
            "invalid saved assistance levels")
    require(bool(((level >= 0) & (level < len(HOLD_LEVELS))).all()), "assistance level out of range")
    require(hold.dtype == torch.float32 and hold.shape == level.shape, "invalid saved hold shape/type")
    require_equal(hold.cpu(), torch.tensor(HOLD_LEVELS)[level.cpu()], "saved hold/level mapping")
    require(math.isfinite(state["learning_rate"]) and state["learning_rate"] > 0, "invalid saved LR")
    groups = checkpoint["optimizer_state_dict"]["param_groups"]
    require(len(groups) == 1 and groups[0]["lr"] == state["learning_rate"], "saved scheduler/Adam LR mismatch")
    require_finite(checkpoint, "checkpoint")
    return state


def restore_training_progress(runner, env, checkpoint):
    """Call after strict runner.load; never demote a saved level on startup reset."""
    raw = env.unwrapped
    state = validate_training_checkpoint(checkpoint, env.num_envs)
    check_loaded_runner(runner, raw, checkpoint)
    if state["completed_updates"]:
        check_optimizer(runner.alg.optimizer, state["completed_updates"] *
                        runner.alg.num_learning_epochs * runner.alg.num_mini_batches)
    raw.common_step_counter = state["common_step_counter"]
    raw._sim_step_counter = state["sim_step_counter"]
    raw.episode_length_buf.zero_()
    # This applies global curricula at the restored counter, then spawns a new episode.
    env.reset()
    # Restore AFTER reset, since the ordinary reset curriculum would demote at age zero.
    raw._basketball_state.level.copy_(state["level"].to(raw.device))
    raw._basketball_state.hold.copy_(state["hold"].to(raw.device))
    runner.alg.actor.reset()
    runner.current_learning_iteration = state["next_iteration"]
    runner.alg.learning_rate = state["learning_rate"]
    require(raw.common_step_counter == state["common_step_counter"], "resume reset changed global progress")
    require(not bool(raw.episode_length_buf.any()), "resume episode ages must start at zero")
    return state["completed_updates"]


def save_checkpoint(runner, env, output, completed, head, registered_save):
    directory = output / "checkpoints"
    directory.mkdir(exist_ok=True)
    path = directory / f"update_{completed:06d}.pt"
    if path.exists():
        receipt = json.loads(path.with_suffix(".json").read_text())
        require(file_sha256(path) == receipt["sha256"], "existing checkpoint checksum differs")
        return path
    state = checkpoint_metadata(runner, env.unwrapped, completed, head)
    temporary = path.with_suffix(".tmp.pt")
    old_iteration = runner.current_learning_iteration
    try:
        runner.current_learning_iteration = completed - 1
        registered_save(str(temporary), infos={"stage07": state})
    finally:
        runner.current_learning_iteration = old_iteration
    saved = torch.load(temporary, map_location="cpu", weights_only=False)
    validate_training_checkpoint(saved, env.num_envs)
    for key, value in runner.alg.save().items():
        require_equal(saved[key], value, "saved." + key)
    temporary.replace(path)
    receipt = {"path": str(path), "sha256": file_sha256(path), "size_bytes": path.stat().st_size,
               "completed_updates": completed, "created_utc": utc_now(), "git_head": head}
    write_json(path.with_suffix(".json"), receipt)
    temporary_link = output / "latest.pt.tmp"
    temporary_link.symlink_to(path.relative_to(output))
    temporary_link.replace(output / "latest.pt")
    print("Stage07Checkpoint=" + json.dumps(receipt), flush=True)
    return path


def evaluation_rank(report):
    require(report["status"] == "PASS" and report["protocol"] == EVALUATION_PROTOCOL,
            "cannot select from a failed or different evaluation protocol")
    rank = (report["success_fraction"], report["mean_survival_seconds"],
            report["mean_stable_fraction"], -report["mean_top_center_error_m"])
    require(all(math.isfinite(x) for x in rank), "non-finite evaluation selection metrics")
    return rank


def terminal_episodes(raw, alive, dones):
    """Read already-computed terminal metrics once, before any reset."""
    names = raw.metrics_manager.active_terms
    success_col = names.index("double_balance_success")
    ended = alive & dones.bool()
    episodes = []
    for i in ended.nonzero(as_tuple=False).flatten().tolist():
        count = int(raw.metrics_manager._step_count[i])
        means = {name: float(raw.metrics_manager._episode_sums[name][i] / max(1, count))
                 for name in names if name != "double_balance_success"}
        terminated = bool(raw.reset_terminated[i])
        episodes.append({"env_id": i, "steps": count, "seconds": count * raw.step_dt,
                         "terminated": terminated, "success": bool(
                             raw.metrics_manager._step_values[i, success_col]) and not terminated,
                         "termination_terms": [name for name in raw.termination_manager.active_terms
                                               if bool(raw.termination_manager.get_term(name)[i])],
                         **means})
    return episodes, ended


def run_evaluation(checkpoint, output, completed):
    directory = output / "evaluations" / f"update_{completed:06d}"
    directory.parent.mkdir(exist_ok=True)
    print(f"Stage07EvaluationStart={completed}", flush=True)
    with directory.with_suffix(".log").open("x") as stream:
        subprocess.run([sys.executable, "-u", "-m", "mjlab_microduck.double_balance_training", "evaluate",
                        "--checkpoint", str(checkpoint), "--output", str(directory)],
                       stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=900)
    report = json.loads((directory / "evaluation.json").read_text())
    require(report["checkpoint_sha256"] == file_sha256(checkpoint), "evaluation used a different checkpoint")
    rank = evaluation_rank(report)
    best_path = output / "best.json"
    best = json.loads(best_path.read_text()) if best_path.exists() else None
    if best is None or rank > tuple(best["rank"]):
        write_json(best_path, {"checkpoint": str(checkpoint), "sha256": file_sha256(checkpoint),
                              "completed_updates": completed, "rank": rank,
                              "evaluation": str(directory / "evaluation.json"),
                              "scope": "best nominal checkpoint within this run segment; not randomized robustness"})
    print("Stage07Evaluation=" + json.dumps({k: report[k] for k in (
        "completed_updates", "success_fraction", "mean_survival_seconds",
        "mean_stable_fraction", "mean_top_center_error_m")}), flush=True)
    return report


def evaluate(args):
    head = cloud_preflight()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = validate_training_checkpoint(checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.os import dump_yaml
    import mjlab_microduck.tasks  # noqa: F401
    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.scene.num_envs, cfg.seed = 64, SEED
    cfg.auto_reset = False  # Observe terminal metrics before resetting; task definitions stay frozen.
    require(cfg.episode_length_s == 10.0, "official play horizon changed")
    require(not cfg.curriculum and tuple(cfg.actions["ball_hold"].levels) == (0.0,),
            "official evaluation must be unassisted")
    agent = resume_agent_config(asdict(load_rl_cfg(TASK_ID)), checkpoint)
    agent.update(logger="tensorboard", upload_model=False)
    dump_yaml(args.output / "params/env.yaml", asdict(cfg))
    raw = env = None
    report = {"status": "RUNNING", "protocol": EVALUATION_PROTOCOL, "git_head": head,
              "completed_updates": state["completed_updates"],
              "checkpoint": str(args.checkpoint), "checkpoint_sha256": file_sha256(args.checkpoint),
              "num_envs": 64, "horizon_seconds": 10.0, "assistance": 0.0,
              "sampling_scope": "64 deterministic nominal initial-state copies; not independent randomized trials",
              "success_definition": "frozen Stage 04 metric at episode end, continuous final 5 seconds; termination is failure",
              "created_utc": utc_now()}
    try:
        raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        env = RslRlVecEnvWrapper(raw, clip_actions=agent["clip_actions"])
        runner = load_runner_cls(TASK_ID)(env, deepcopy(agent), None, "cuda:0")
        runner.load(str(args.checkpoint), strict=True, map_location="cuda:0")
        check_loaded_runner(runner, raw, checkpoint)
        # Training counters and per-env assistance are not restored into play episodes.
        raw.common_step_counter = raw._sim_step_counter = 0
        raw.episode_length_buf.zero_()
        env.reset()
        require(bool((raw._basketball_state.hold == 0).all()), "evaluation assistance is nonzero")
        policy = runner.get_inference_policy()
        policy.reset()
        obs = env.get_observations()
        alive = torch.ones(64, dtype=torch.bool, device=raw.device)
        episodes = []
        with torch.inference_mode():
            for _ in range(raw.max_episode_length + 1):
                actions = policy(obs)
                obs, rewards, dones, _ = env.step(actions)
                require_finite((actions, obs, rewards), "evaluation rollout")
                require(not bool((raw.termination_manager.get_term("nan_state") & alive).any()),
                        "NaN in evaluated episode")
                finished, ended = terminal_episodes(raw, alive, dones)
                episodes.extend(finished)
                alive &= ~ended
                if not bool(alive.any()):
                    break
                done_ids = dones.nonzero(as_tuple=False).flatten()
                if len(done_ids):
                    raw.reset(env_ids=done_ids)
                    policy.reset(dones.bool())
                    obs = env.get_observations()
        require(len(episodes) == 64, "evaluation failed to complete every first episode")
        mean = lambda key: sum(float(e[key]) for e in episodes) / len(episodes)
        report.update(status="PASS", success_fraction=mean("success"),
                      mean_survival_seconds=mean("seconds"),
                      mean_stable_fraction=mean("double_balance_stable_fraction"),
                      mean_top_center_error_m=mean("top_ball_center_error_m"), episodes=episodes)
    except BaseException as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        report["finished_utc"] = utc_now()
        write_json(args.output / "evaluation.json", report)
        if env is not None:
            env.close()
        elif raw is not None:
            raw.close()


class PauseAtUpdateBoundary(Exception):
    pass


def train(args):
    head = cloud_preflight()
    require(args.num_envs == 4096, "this measured Stage 07 plan uses 4096 environments")
    require(1 <= args.target_updates <= 16000, "target must be 1..16000 completed updates")
    capacity = json.loads(args.capacity_summary.read_text())
    measured = validate_capacity(capacity, args.num_envs)
    require(shutil.disk_usage(ROOT).free > 5 * 1024**3, "at least 5 GiB free data disk required")
    checksum = file_sha256(args.checkpoint)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if args.resume:
        start = validate_training_checkpoint(checkpoint, args.num_envs)["completed_updates"]
        receipt = json.loads(args.checkpoint.with_suffix(".json").read_text())
        require(receipt["sha256"] == checksum, "resume checksum mismatch")
    else:
        require(checksum == MIGRATED_SHA256, "fresh training must start from Stage 05")
        start = 0
    require(start < args.target_updates, "target must exceed the completed update count")
    output = args.output.resolve()
    require(output.is_relative_to(ARTIFACTS.resolve()), "training artifacts belong on the cloud data disk")
    output.mkdir(parents=True, exist_ok=False)
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.os import dump_yaml
    import mjlab_microduck.tasks  # noqa: F401
    cfg = load_env_cfg(TASK_ID)
    cfg.scene.num_envs, cfg.seed = args.num_envs, SEED
    require(cfg.sim.mujoco.timestep == .002 and cfg.decimation == 10, "frozen timing differs")
    require(tuple(cfg.actions["ball_hold"].levels) == HOLD_LEVELS, "frozen assistance table differs")
    agent = asdict(load_rl_cfg(TASK_ID))
    agent.update(seed=SEED, logger="tensorboard", upload_model=False,
                 max_iterations=args.target_updates, save_interval=100,
                 run_name="stage07_assisted", check_for_nan=True)
    if args.resume:
        agent = resume_agent_config(agent, checkpoint)
    require(agent["num_steps_per_env"] == STEPS_PER_ENV, "rollout length differs")
    dump_yaml(output / "params/env.yaml", asdict(cfg))
    dump_yaml(output / "params/agent.yaml", deepcopy(agent))
    write_json(output / "capacity-summary.json", capacity)
    report = {"status": "INITIALIZING", "formal_training_started": False,
              "git_head": head, "stage06_baseline": BASELINE_COMMIT, "hostname": socket.gethostname(),
              "gpu": torch.cuda.get_device_name(0), "versions": {**VERSIONS, "torch": torch.__version__},
              "source_sha256": {p.name: file_sha256(p) for p in (
                  Path(__file__), Path(__file__).with_name("double_balance_stage07.py"))},
              "created_utc": utc_now(), "input_checkpoint": str(args.checkpoint), "input_sha256": checksum,
              "initialization_sha256": MIGRATED_SHA256, "resume": args.resume,
              "start_completed_updates": start, "completed_updates": start,
              "target_updates": args.target_updates, "num_envs": args.num_envs,
              "transitions_per_update": args.num_envs * STEPS_PER_ENV,
              "projected_remaining_train_hours_from_capacity":
                  (args.target_updates - start) * measured["mean_iteration_seconds"] / 3600,
              "wall_clock_timeout": None, "save_interval_updates": 100, "eval_interval_updates": 500,
              "latest_checkpoint": None, "exact_trajectory_resume": False}
    write_json(output / "training.json", report)
    raw = env = runner = None
    began = time.monotonic()
    pause = {"requested": False}
    previous_signals = {s: signal.getsignal(s) for s in (signal.SIGTERM, signal.SIGINT)}
    for s in previous_signals:
        signal.signal(s, lambda *_: pause.update(requested=True))
    try:
        raw = ManagerBasedRlEnv(cfg=cfg, device="cuda:0")
        env = RslRlVecEnvWrapper(raw, clip_actions=agent["clip_actions"])
        runner = load_runner_cls(TASK_ID)(env, deepcopy(agent), str(output / "tensorboard"), "cuda:0")
        runner.load(str(args.checkpoint), strict=True, map_location="cuda:0")
        if args.resume:
            restore_training_progress(runner, env, checkpoint)
        else:
            check_loaded_runner(runner, raw, checkpoint)
            reset_stage07_progress(runner, env)
        check_observations(env.get_observations(), args.num_envs)
        registered_save = runner.save
        runner.logger.logger_type = "tensorboard"

        def save(completed):
            path = save_checkpoint(runner, env, output, completed, head, registered_save)
            report["latest_checkpoint"] = str(path)
            return path

        # Own the save schedule by completed updates; the native runner uses zero-based indices.
        def automatic_save(path, infos=None):
            del path, infos
            completed = raw.common_step_counter // STEPS_PER_ENV
            if completed % 100 == 0 or completed == args.target_updates:
                save(completed)
        runner.save = automatic_save
        initial = save(start)
        print("Stage07Initialization=PASS", flush=True)
        # Validate the eval/load path before the first optimizer update (no extra smoke training).
        run_evaluation(initial, output, start)
        report.update(status="RUNNING", formal_training_started=True, strict_initialization="PASS")
        write_json(output / "training.json", report)
        print("Stage07Training=RUNNING", flush=True)

        def updated(row, evidence):
            completed = row["completed_updates"]
            report.update(completed_updates=completed, elapsed_seconds=time.monotonic() - began,
                          latest_update=row, optimizer_steps_this_segment=evidence["optimizer_steps"])
            if completed % 100 == 0 or completed == args.target_updates:
                save(completed)
            should_pause = pause["requested"] or (output / "STOP").exists()
            if should_pause:
                save(completed)
                raise PauseAtUpdateBoundary()
            if completed % 500 == 0 or completed == args.target_updates:
                report["latest_evaluation"] = run_evaluation(save(completed), output, completed)
            if completed % 10 == 0:
                sample = subprocess.check_output([
                    "nvidia-smi", "--id=0", "--query-gpu=memory.used,memory.total,utilization.gpu",
                    "--format=csv,noheader,nounits"], text=True, timeout=10).strip()
                used, total, utilization = map(int, sample.split(","))
                report["gpu_sample"] = {"completed_updates": completed, "created_utc": utc_now(),
                                        "used_mib": used, "total_mib": total,
                                        "utilization_percent": utilization}
                report["sampled_peak_training_gpu_used_mib"] = max(
                    used, report.get("sampled_peak_training_gpu_used_mib", 0))
                with (output / "gpu-samples.jsonl").open("a") as stream:
                    stream.write(json.dumps(report["gpu_sample"]) + "\n")
            write_json(output / "training.json", report)

        count = args.target_updates - start
        with monitor_updates(runner, env, output, count, start_completed=start,
                             on_update=updated, prefix="Stage07TrainingUpdate") as evidence:
            runner.learn(num_learning_iterations=count, init_at_random_ep_len=False)
        require(len(evidence["iterations"]) == count and evidence["return_passes"] == count,
                "completed update/return count mismatch")
        require(evidence["vector_steps"] == count * STEPS_PER_ENV, "vector-step count mismatch")
        steps_per_update = runner.alg.num_learning_epochs * runner.alg.num_mini_batches
        require(evidence["optimizer_steps"] == count * steps_per_update, "optimizer count mismatch")
        check_optimizer(runner.alg.optimizer, args.target_updates * steps_per_update)
        require(file_sha256(args.checkpoint) == checksum, "input checkpoint changed")
        report.update(status="TRAINING_COMPLETE", finite_checks="PASS",
                      stage07_complete=False, note="training segment complete; archive/review/handoff still required")
    except PauseAtUpdateBoundary:
        report.update(status="PAUSED", pause_reason="requested stop at completed PPO update")
    except BaseException as exc:
        report.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        # Never checkpoint partially updated/invalid state; retain the last validated save.
        raise
    finally:
        for s, handler in previous_signals.items():
            signal.signal(s, handler)
        report.update(finished_utc=utc_now(), elapsed_seconds=time.monotonic() - began)
        write_json(output / "training.json", report)
        if runner is not None and getattr(runner.logger, "writer", None) is not None:
            runner.logger.writer.close()
        if env is not None:
            env.close()
        elif raw is not None:
            raw.close()
        print("Stage07Training=" + report["status"], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    training = sub.add_parser("train")
    training.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    training.add_argument("--resume", action="store_true")
    training.add_argument("--num-envs", type=int, default=4096)
    training.add_argument("--target-updates", type=int, default=6000)
    training.add_argument("--capacity-summary", type=Path, default=ARTIFACTS / "latest-capacity/capacity-summary.json")
    training.add_argument("--output", type=Path, required=True)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", type=Path, required=True)
    evaluation.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    (train if args.mode == "train" else evaluate)(args)


if __name__ == "__main__":
    main()
