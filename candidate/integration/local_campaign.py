"""v7 N2-only bounded local RTX5060 campaign. No PPO, policy, optimizer or cloud work."""
from pathlib import Path
import argparse
import json
import os
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from integration.durable import require, write_json, file_digest
from integration.planning import proposal, fingerprint
from stage12.contracts import REPO, CANONICAL


def authorization(plan):
    return dict(authorization="USER_AUTHORIZED_NATIVE_MEASUREMENT",
                owner_plan_sha256=fingerprint(plan), canonical_before=CANONICAL,
                PPO=0, cloud_training=0)


def measure_point(point, out, torch, td, timeout_seconds):
    from integration.process_owners import OwnerProcesses
    started = time.monotonic()
    plans = point["owners"]
    rows = []
    with OwnerProcesses(REPO, out, plans, [authorization(p) for p in plans], torch, td,
                        timeout_seconds=timeout_seconds) as env:
        initialized = time.monotonic()
        a = env.inspect()
        env.get_observations()
        env.get_observations()
        b = env.inspect()
        for before, after in zip(a, b, strict=True):
            for key in ("counts", "rng_sha256", "observation_sha256", "resets", "episode_length"):
                require(before[key] == after[key], "Cached observation changed owner: " + key)
        action = torch.zeros(env.num_envs, 14)
        env.step(action)  # Exactly one declared warmup step per owner.
        before = env.inspect()
        for i in range(2):
            t = time.monotonic()
            obs, rewards, done, extras = env.step(action)
            # Production serialization copies to CPU before replying, and inspect
            # synchronizes both Warp and Torch before returning its witness.
            witness = env.inspect()
            rows.append(dict(index=i, wall_seconds=time.monotonic()-t,
                             reward=rewards.tolist(), done=done.tolist(), witnesses=witness))
        after = rows[-1]["witnesses"]
        elapsed = sum(r["wall_seconds"] for r in rows)
        completed = sum(b["counts"].get("integrations_completed", 0)-a["counts"].get("integrations_completed", 0)
                        for a,b in zip(before, after, strict=True))
        attempted = sum(b["counts"].get("integrations_attempted", 0)-a["counts"].get("integrations_attempted", 0)
                        for a,b in zip(before, after, strict=True))
        require(completed == attempted > 0, "Measured native work was not completed exactly")
        old = env.inspect()
        env.reset([0])
        selected = env.inspect()
        require(selected[0]["resets"] == old[0]["resets"] + 1, "Explicit reset count differs")
        for i in range(1, env.num_envs):
            for key in ("counts", "rng_sha256", "observation_sha256", "resets", "episode_length"):
                require(selected[i][key] == old[i][key], "Selective reset changed sibling: " + key)
        env.reset([])
        empty = env.inspect()
        for a,b in zip(selected, empty, strict=True):
            for key in ("counts", "rng_sha256", "observation_sha256", "resets", "episode_length"):
                require(a[key] == b[key], "Empty reset changed owner: " + key)
        for _ in range(plans[0]["public_steps"] - 3):
            env.step(action)
        final = env.inspect()
        require(all(r["public_steps"] == p["public_steps"] and r["macros"] == p["max_macros"]
                    for r,p in zip(final, plans, strict=True)), "Public/macro completion differs")
    closeout = json.loads((out / "owner-closeout.json").read_bytes())
    require(closeout["all_clean"], "Native owner cleanup failed")
    transitions = 2 * len(plans)
    result = dict(name=point["name"], status="SHORT_NATIVE_PILOT_PASS", owners=len(plans),
                  native_nworld_per_process=1, startup_seconds=initialized-started,
                  total_seconds=time.monotonic()-started, timed_rows=rows,
                  timed_public_transitions=transitions, timed_wall_seconds=elapsed,
                  public_transitions_per_s=transitions/elapsed,
                  native_integrations_completed=completed, native_integrations_attempted=attempted,
                  native_integrations_per_s=completed/elapsed,
                  per_owner_simulated_seconds_per_wall_second=.04/elapsed,
                  measurement_scope="Two public steps after one warmup; includes IPC, audit and synchronized inspection",
                  sustained_throughput_proven=False, full_10s_episode_proven=False,
                  before_timed=before, final=final, closeout=closeout)
    write_json(out / "measurement.json", result)
    return result


def reconcile_native(out, plans):
    """Read durable per-owner WALs once, including incomplete/failed owners."""
    sys.path.insert(0, str(ROOT / "baseline/g11"))
    from runtime_accounting import review_ledger
    rows = []
    for point in plans["points"]:
        for plan in point["owners"]:
            folder = out / point["name"] / ("owner-%d" % plan["logical_world"])
            ledger = folder / "native-ledger.jsonl"
            marker = folder / "started-v1.json"
            parent_started = folder.parent / "started-v1.json"
            if ledger.exists():
                try:
                    state = review_ledger(ledger)
                except BaseException as exc:
                    state = dict(exact=False, error=repr(exc), integration_attempted_lower=0,
                                 integration_attempted_upper=plan["caps"]["integrations"],
                                 integration_completed_lower=0, integration_completed_upper=plan["caps"]["integrations"])
            else:
                # No ledger means no integration could begin in owner_session,
                # but an interrupted process during setup still is not a PASS.
                state = dict(exact=not (marker.exists() or parent_started.exists()),
                             integration_attempted_lower=0, integration_attempted_upper=0,
                             integration_completed_lower=0, integration_completed_upper=0,
                             ledger_not_created=True)
            rows.append(dict(point=point["name"], owner=plan["logical_world"], ledger=state))
    added = {k: sum(r["ledger"]["integration_"+k] for r in rows)
             for k in ("attempted_lower", "attempted_upper", "completed_lower", "completed_upper")}
    exact_integrations = (added["attempted_lower"] == added["attempted_upper"] and
                          added["completed_lower"] == added["completed_upper"])
    return dict(owners=rows, native_added=added, integration_counts_exact=exact_integrations,
                all_operation_ledgers_clean=all(r["ledger"]["exact"] for r in rows),
                cumulative_completed=[CANONICAL["completed"]+added["completed_lower"], CANONICAL["completed"]+added["completed_upper"]],
                cumulative_attempted=[CANONICAL["attempted"]+added["attempted_lower"], CANONICAL["attempted"]+added["attempted_upper"]],
                historical_failure_retained=True, baseline_added_once=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--authorized-native-integrations", type=int, required=True)
    parser.add_argument("--wall-seconds", type=int, required=True)
    args = parser.parse_args()
    out, plans = args.output, proposal()
    require(out.is_dir() and args.authorized_native_integrations == plans["native_integrations_max"],
            "Exact native cap must be explicitly supplied")
    require(60 <= args.wall_seconds <= 86400, "Wall limit must be explicit, 60..86400 seconds")
    require(shutil.disk_usage(out).free >= plans["evidence_disk_reservation_bytes"] + 4*1024**3,
            "Need 15.375 GiB free for retained evidence and scratch; nothing native started")
    report = dict(status="STARTED", points=[], errors=[], native_budget=plans,
                  PPO=0, cloud_training=0, torch_load=0, training_ready=False)
    before = None
    try:
        from stage12.host_sources import verify
        before = verify(ROOT, out, "before")
        from integration.completed_reuse import reuse_completed
        reused = reuse_completed(ROOT, out)
        report['cloud_inputs'] = reused['cloud_inputs']
        report['cpu_regression_reused'] = reused['cpu_regression']
        # The host's completed 13 cases and portable input package are reused.
        # New owner bootstrap and native points execute in this repair only.
        import torch
        from tensordict import TensorDict
        for point in plans["points"]:
            print("Stage12LocalPhase="+point["name"]+"; PPO=0 cloud_training=0", flush=True)
            report["points"].append(measure_point(point, out / point["name"], torch, TensorDict, args.wall_seconds))
        from integration.native_review import verify as verify_new_raw
        report['independent_raw_review'] = verify_new_raw(out)
        report["status"] = "N2_NATIVE_PASS_CLOUD_BUDGET_PENDING"
    except BaseException as exc:
        report["status"] = "LOCAL_COMPLETION_STOPPED_PRESERVE"
        report["errors"].append(dict(error=repr(exc), traceback=traceback.format_exc()))
    finally:
        if before is not None:
            try:
                require(verify(ROOT, out, "after") == before, "Source or retained evidence changed")
                report["preservation"] = "PASS"
            except BaseException as exc:
                report["status"] = "LOCAL_COMPLETION_STOPPED_PRESERVE"
                report["errors"].append(dict(preservation_error=repr(exc)))
        report["accounting"] = reconcile_native(out, plans)
        write_json(out / "host-report.json", report)
    print("Stage12LocalCompletion="+report["status"], flush=True)
    return int(bool(report["errors"]))


if __name__ == "__main__":
    raise SystemExit(main())
