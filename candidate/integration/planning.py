"""Complete bounded operation worksheets; generating a plan authorizes nothing."""
from copy import deepcopy
from pathlib import Path
import json
import math
from .durable import require, digest, encoded
from .refinement_contract import TIERS, MAX_INTEGRATIONS, TRIALS, MAX_FORWARD

ROOT = Path(__file__).resolve().parents[1]


def owner_plan(public_steps, resets, logical_world=0):
    require(type(public_steps) is int and 1 <= public_steps <= 500, "Steps outside bounded candidate range")
    require(type(resets) is int and public_steps + 1 <= resets <= public_steps + 8,
            "Reserve initial reset plus worst-case every-step reset and explicit tests")
    require(type(logical_world) is int and logical_world in (0, 1), "Only logical owners 0/1 implemented")
    base = json.loads((ROOT / "baseline/g11/plan.json").read_bytes())
    m = 10 * public_steps
    # Conservative declared ceilings for every inherited named operation. These
    # are not measured costs or automatic permissions. Exact macro costs below.
    factor = max(math.ceil(m / 20), resets, 1)
    caps = {k: (v * factor if type(v) is int else v) for k, v in base["caps"].items()}
    static = ("physical_worlds", "CUDA_graphs", "full_RL_environment", "snapshot_bytes_each",
              "Scene_initialize", "Model_expansion", "Sensor_context_set", "context_create",
              "Reduction_module_load", "Reduction_allocation", "Reduction_device_bytes",
              "Rigid_module_load", "SolverOrder_module_load", "Model_material_setup")
    for k in static:
        caps[k] = base["caps"][k]
    external = 80 + 8 * (public_steps + resets)
    f = MAX_FORWARD * m + external
    caps.update(integrations=MAX_INTEGRATIONS * m, RK4_integrator=MAX_INTEGRATIONS * m, GPU_forward=f,
                Action_apply=m, Metrics_substep=m, Env_substep_counter=m,
                state_snapshot=m, state_restore=(TRIALS+1) * m, position_write=MAX_INTEGRATIONS * m,
                public_clock_write=m, private_clock_write=TRIALS * m, model_option_write=2 * TRIALS * m,
                endpoint_sensor_read=(TRIALS+1) * m, Scene_update=m,
                Scene_write=m + resets, BAM_compute=m + resets, BAM_apply_delay=m + resets,
                sense=(TRIALS+1) * m + public_steps + 2 * resets + 8,
                Action_process_action=public_steps, Reward_compute=public_steps,
                Termination_compute=public_steps, Metrics_compute=public_steps,
                Observation_compute=public_steps + resets,
                Observation_noise=4 * (public_steps + resets),
                Command_compute=public_steps + resets,
                Recorder_post_step=public_steps, Recorder_pre_reset=resets,
                Recorder_post_reset=resets, Recorder_close=1, PPO=0, cloud=0)
    for label in ("GPU_reset_data", "Simulation_reset", "Scene_reset", "Action_reset",
                  "Observation_reset", "Event_reset", "Reward_reset", "Termination_reset",
                  "Metrics_reset", "Command_reset", "Curriculum_reset", "Curriculum_compute"):
        caps[label] = resets
    for label in ("GPU_solver", "SolverContext_create", "Solverinit_context", "Solver_update_gradient",
                  "Rigid_metadata_kernel"):
        caps[label] = f
    for label in ("Solver_update_constraint", "Cost_rows_kernel", "Cost_reduce_kernel", "Cost_override_copy"):
        caps[label] = 11 * f
    for label in ("Solver_update_gradient_incremental", "Solver_linesearch", "Solver_solver_iteration", "SolverOrder_kernel"):
        caps[label] = 10 * f
    caps["Norm_kernel"] = 21 * f
    caps["BVH_refit"] = caps["ray_batch"] = caps["sense"]
    graph = deepcopy(base["graph_limits"])
    graph["launch"] = f
    evidence = deepcopy(base["solver_order_evidence"])
    evidence.update(records=10 * f, index_bytes=base["solver_order_evidence"]["index_bytes"] * factor,
                    blob_bytes=base["solver_order_evidence"]["blob_bytes"] * factor,
                    blob_files=base["solver_order_evidence"]["blob_files"] * factor)
    # Raw boundary evidence is intentionally retained; the audit backend is
    # expensive. This bound is a reservation, not a throughput optimization.
    raw_reserve = m * 64 * 1024**2 + 256 * 1024**2 * factor
    return dict(format="stage12-owner-plan-v1", logical_world=logical_world, native_nworld=1,
                public_steps=public_steps, max_macros=m, max_resets=resets,
                max_episode_ticks=5000, caps=caps, graph_limits=graph,
                solver_order_evidence=evidence, contact_layout=base["contact_layout"],
                original_solref_sha256=base["original_solref_sha256"],
                external_forward_reservation=external, evidence_bytes_reservation=raw_reserve,
                estimated_cost=False, tiers=list(TIERS), exact_replay=True,
                numerical_thresholds_changed=False, model_constructions=0, optimizer_constructions=0,
                PPO=0, cloud_training=0)


def proposal():
    # Reuse the completed N1. Only the failed whole N2 point is new work.
    pair = [owner_plan(3, 5, i) for i in (0, 1)]
    points = [dict(name="two_process_short_pilot", owners=pair,
                   warmup_steps=1, timed_steps=2, remaining_steps=0)]
    owners = [p for point in points for p in point["owners"]]
    return dict(format="stage12-measurement-proposal-v1", authorization="PENDING_USER_BUDGET",
                candidate_only=True, device="local RTX 5060", points=points,
                environment_constructions=len(owners), native_integrations_max=sum(p["caps"]["integrations"] for p in owners),
                native_GPU_forward_max=sum(p["caps"]["GPU_forward"] for p in owners),
                native_CPU_forward_max=sum(p["caps"]["CPU_forward"] for p in owners),
                reset_attempts_max=sum(p["max_resets"] for p in owners),
                evidence_disk_reservation_bytes=sum(p["evidence_bytes_reservation"] for p in owners),
                PPO=0, cloud_training=0, torch_load=0, old_batches_rerun=0,
                not_proven=["continuous full episode", "normalization with real rollout",
                            "PPO update", "cloud image/assets", "native batched nworld > 1"],
                interpretation="v7 N2 only, predeclared tiers32/64/128/256/512; not training readiness")


def recurrent_batches(num_envs, mini_batches, steps=24):
    require(type(num_envs) is int and num_envs in (1, 2), "Only independent N=1/2 owner interface prepared")
    require(type(mini_batches) is int and 1 <= mini_batches <= num_envs and num_envs % mini_batches == 0,
            "Recurrent minibatches must divide the environment count; inherited B=4 is invalid")
    require(steps == 24, "Rollout length is fixed until separately reviewed")
    return dict(num_envs=num_envs, num_mini_batches=mini_batches, rollout_steps=steps,
                transitions_per_update=num_envs * steps)


def throughput_row(transitions, elapsed_s, completed, attempted, peak_bytes, device_bytes):
    require(type(transitions) is int and transitions > 0 and math.isfinite(elapsed_s) and elapsed_s > 0,
            "Invalid measured interval")
    require(type(completed) is type(attempted) is int and 0 <= completed <= attempted,
            "Invalid native ledger")
    require(type(peak_bytes) is type(device_bytes) is int and 0 <= peak_bytes <= device_bytes,
            "Memory measurement missing")
    return dict(public_transitions=transitions, end_to_end_seconds=elapsed_s,
                public_transitions_per_s=transitions / elapsed_s,
                simulated_seconds_per_wall_second=transitions * .020 / elapsed_s,
                native_integrations_completed=completed, native_integrations_attempted=attempted,
                native_integrations_per_s=completed / elapsed_s,
                peak_memory_bytes=peak_bytes, device_memory_bytes=device_bytes,
                memory_headroom_bytes=device_bytes - peak_bytes,
                scope="synchronized wall clock; includes IPC and audits, excludes declared warmup",
                stable_rate_claim=False)


def validate_owner(plan):
    expected = owner_plan(plan["public_steps"], plan["max_resets"], plan["logical_world"])
    require(plan == expected, "Plan differs from generated operation ceilings")
    return plan


def fingerprint(value):
    return digest(encoded(value))
