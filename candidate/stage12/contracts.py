"""Pure preparation contracts. No simulator, checkpoint loader or training import."""
from copy import deepcopy
import hashlib
import json

TASK_ID = "Mjlab-DoubleBalance-Head-Nominal-MicroDuck-Stage12-Prepared-v0"
BASE_COMMIT = "6ba0dc2de381321576f32310e1995a8e2e11e566"
TAG_OBJECT = "bd1b219be55c813811d4f7cfa25d245b37166688"
TRACKED_DIFF = "b90daa0274c14d95cec9511ec4ac93568bd604e93ce3c60df1bb1c4155005d7e"
REPO = "/home/lx/microduck-double-balance/workspace"
ARTIFACTS = "/home/lx/microduck-double-balance/artifacts"
CHECKPOINT = ARTIFACTS + "/double-balance-stage08/returned/5d998fc322345102/evidence-20260930T093645-8955/extension/E-20260929/segment-001/checkpoints/update_004000.pt"
CHECKPOINT_SHA = "86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97"
CHECKPOINT_BYTES = 10073500
CHECKPOINT_SCHEMA_SHA = "e740e6ca4ad74c53a0eaaa9c95dcae9b69fd5b0edf122593743cd015dcb388a7"
CANONICAL = {"completed": 693528, "attempted": 693529}
EXECUTION_BUDGET = {"native_integrations": 0, "native_forward": 0,
                    "environment_constructions": 0, "PPO": 0, "cloud_training": 0,
                    "torch_load": 0, "policy_constructions": 0, "optimizer_constructions": 0}
BLOCKERS = [
    "Native runtime/factory/reset/terminal candidates are implemented but not physically accepted; CPU tests are not native evidence.",
    "Process-isolated N=1/2 transport is implemented; native lifecycle, selective reset and measured throughput are pending.",
    "Update-boundary checkpoint/runner controllers are implemented; actual policy/optimizer/update acceptance has zero current budget.",
    "Source/asset input bundle is prepared; dependency wheels, target GPU runtime and cloud environment are not yet reproduced.",
    "No local native measurement or cloud PPO training budget has been authorized for these new candidates.",
]



def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def deny_training(*args, **kwargs):
    raise RuntimeError("STAGE12_NOT_RELEASED: " + " | ".join(BLOCKERS))


class PreparedRunner:
    """Registry placeholder which refuses before constructing any model or PPO."""
    __init__ = deny_training
    load = deny_training
    learn = deny_training


def create_environment(*args, **kwargs):
    """Dedicated factory contract; no fallback to stock ManagerBasedRlEnv."""
    deny_training()


def checkpoint_policy(mode="fresh", checkpoint=None):
    require(mode == "fresh" and checkpoint is None,
            "Direct resume, actor-only transfer and normalizer/optimizer transfer are not released")
    return {
        "status": "FRESH_INITIALIZATION_PLAN_ONLY", "executed": False,
        "old_checkpoint": {"path": CHECKPOINT, "sha256": CHECKPOINT_SHA,
                           "bytes": CHECKPOINT_BYTES, "schema_sha256": CHECKPOINT_SCHEMA_SHA,
                           "disposition": "PRESERVE_READ_ONLY", "direct_resume": "BLOCKED_TASK_AND_TIMING_DIFFER"},
        "actor": "new random LSTM(61->256, one layer) + MLP(512,256,128)->14, not constructed",
        "critic": "new random MLP(85->512->256->128->1), not constructed",
        "normalizers": "fresh actor/critic statistics; do not copy Stage08 statistics",
        "optimizer": "fresh; no Adam moments or LR schedule copied",
        "distribution": "fresh configured action distribution; no old std copied",
        "recurrent_state": "fresh zero state per episode; reset on terminated OR truncated; no mid-rollout resume claim",
        "counters": "new run identity; iteration, common_step_counter and per-env episode counters start at zero",
        "random_episode_length_initialization": False,
        "warm_start": "separate optional study; explicit feature-column/LSTM/normalizer policy and paired fresh baseline required",
        "future_same_task_resume": ["exact task/config/source/dependency identity", "model and optimizer schema",
                                  "normalizers and counters", "RNG and RNN/reset policy",
                                  "consumed/remaining budget ledger", "new explicit training authorization"],
        "key_mapping_is_semantic_compatibility": False, "torch_load_calls": 0,
        "installed_runner_audit": {
            "source_review": "runner-integration-review.json",
            "saved_iter": "last completed zero-based iteration index; completed_updates=iter+1 only with a committed ledger",
            "stock_resume_repeats_index": True,
            "checkpoint_contains": "actor, critic, optimizer, normalizer buffers and iter; mjlab adds common_step_counter in infos",
            "not_saved_by_default": ["native physics state", "environment RNG streams", "live RNN hidden/cell",
                                     "in-flight rollout", "authoritative consumed/remaining budget ledger"],
            "approved_resume_now": False,
        },
        "future_save_protocol_draft": {
            "status": "IMPLEMENTED_CANDIDATE_IN_INTEGRATION_DURABLE_AND_RUNNER_NOT_PHYSICALLY_ACCEPTED",
            "boundary": "completed optimizer update with durable consumed-budget ledger",
            "counter_fields": ["last_completed_index", "completed_updates", "next_iteration",
                               "public_transitions", "native_completed", "native_attempted", "remaining_authorized_budget"],
            "identity": ["task/config/source/dependency hashes", "checkpoint bytes/SHA256", "run id", "seed/reset policy"],
            "publication": "unique temp file; flush/fsync; atomic rename; SHA256 receipt committed last; retain partial files on failure",
            "continuation": "explicit update-boundary reset and RNG/RNN policy; no bitwise or mid-rollout equivalence claim",
            "interrupted_update": "retain attempted counts and started marker; no automatic retry or budget refund",
        },
    }


def scaling_plan():
    # G11's mandatory accepted replay: 32+64+64=160 or 32+64+128+128=352.
    # This is a cost worksheet, never authorization or an implementation of batching.
    points = []
    for owners in (1, 2):
        warmup, timed = 1, 2
        macros = owners * (warmup + timed) * 10
        points.append(dict(independent_owners=owners, native_nworld_per_owner=1,
                           warmup_public_steps=warmup, measured_public_steps=timed,
                           repeats=1, macro_advances=macros,
                           nominal_integrations_if_all_accept64=macros * 160,
                           integrations_worst_case=macros * 352,
                           macro_forward_reservation=macros * 1413,
                           setup_reset_forward_budget=None, full_native_budget_released=False,
                           expected_measured_transitions=owners * timed,
                           status="DRAFT_NOT_AUTHORIZED"))
    return {
        "status": "STATIC_WORKSHEET_ONLY", "executed": False,
        "authorized_budget": deepcopy(EXECUTION_BUDGET),
        "physics_dt_s": .002, "decimation": 10, "public_step_dt_s": .020,
        "adaptive_tiers": [32, 64, 128], "mandatory_exact_replay": True,
        "per_macro_integrations_accept64": 160, "per_macro_integrations_accept128": 352,
        "per_public_step_integrations_max": 3520,
        "draft_local_pilot": points,
        "draft_integrations_total_max": sum(p["integrations_worst_case"] for p in points),
        "later_ladder": {"independent_owners": [4, 8], "native_batch_nworld": "UNRELEASED; separate implementation/acceptance"},
        "scope": "One short pilot at N=1 then N=2 only after backend release and a complete budget; not a stable throughput estimate",
        "admission_checks": ["source/config identity", "single owner long-run and reset acceptance", "selective reset isolation",
                             "per-owner RNG/cache/resource ownership", "immutable old criteria", "peak memory measured with headroom",
                             "bound all setup/reset/forward/sense/graph/ledger calls before execution"],
        "metrics": {
            "env_transitions_per_second": "successful measured public transitions / synchronized measured wall seconds",
            "aggregate_sim_seconds_per_second": "successful measured public transitions * 0.020 / measured wall seconds",
            "per_env_realtime_factor": "measured public steps per owner * 0.020 / measured wall seconds (equal step counts only)",
            "native_integrations_per_second": "measured completed native integrations / measured wall seconds; separate from transitions",
            "latency": "public-step median/p95, include cross-process orchestration, sync before timing boundaries",
            "memory": "CUDA peak allocated/reserved, driver used/free, host RSS and number of contexts",
            "overhead": "startup/JIT, warmup, measured stepping, durable WAL, artifact writing reported separately",
            "numerics": "accepted tier histogram, rejections, replay failures, failed API attempts; no tolerance changes",
        },
        "throughput_measured": None, "cloud_gpu_budget_hours": None, "PPO_update_budget": 0,
        "failure": "stop that new run, retain started and attempted counts; no automatic retry/scale-up/finer tiers",
        "wall_clock": "no invented runtime deadline; operation budget first; 5-second heartbeat is not timeout",
    }
