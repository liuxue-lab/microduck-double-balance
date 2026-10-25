#!/usr/bin/env python3
"""Run isolated 512/1024/2048/4096-env capacity jobs on the allocated cloud GPU."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

EXPECTED_CHECKPOINT = "548758afc546e11d8f3f446a4bcc846283a669ff20f4048da7a3a7bb1332b98e"


def gpu_sample():
    result = subprocess.run([
        "nvidia-smi", "--id=0", "--query-gpu=memory.total,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits"], check=True, capture_output=True, text=True, timeout=5)
    total, used, utilization = [int(x.strip()) for x in result.stdout.strip().split(",")]
    return {"total_mib": total, "used_mib": used, "utilization_percent": utilization}


def stop_child(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)


def run_case(args, count, root, deadline):
    output = root / f"envs-{count}"
    log = root / f"envs-{count}.log"
    samples = root / f"envs-{count}-gpu.csv"
    command = [sys.executable, "-u", "-m", "mjlab_microduck.double_balance_stage07",
               str(args.checkpoint), "--output", str(output), "--num-envs", str(count),
               "--expected-gpu", args.expected_gpu]
    started = time.monotonic()
    last_progress = started
    initial = gpu_sample()
    peak, max_utilization, peak_rss_kib = initial["used_mib"], 0, 0
    child = None
    timed_out = False
    with log.open("x") as log_stream, samples.open("x", newline="") as sample_stream:
        writer = csv.writer(sample_stream)
        writer.writerow(["elapsed_seconds", "gpu_used_mib", "gpu_utilization_percent", "process_rss_kib"])
        try:
            child = subprocess.Popen(command, stdout=log_stream, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            while child.poll() is None:
                now = time.monotonic()
                if now >= deadline or now - started >= args.case_timeout:
                    timed_out = True
                    stop_child(child)
                    break
                sample = gpu_sample()
                peak = max(peak, sample["used_mib"])
                max_utilization = max(max_utilization, sample["utilization_percent"])
                rss = 0
                try:
                    for line in Path(f"/proc/{child.pid}/status").read_text().splitlines():
                        if line.startswith("VmRSS:"):
                            rss = int(line.split()[1])
                            peak_rss_kib = max(peak_rss_kib, rss)
                except FileNotFoundError:
                    pass
                writer.writerow([round(now - started, 3), sample["used_mib"],
                                 sample["utilization_percent"], rss])
                sample_stream.flush()
                if now - last_progress >= 30:
                    with log.open("rb") as recent:
                        recent.seek(0, 2)
                        recent.seek(max(0, recent.tell() - 1024))
                        lines = recent.read().decode("utf-8", errors="replace").splitlines()
                    print("Stage07CapacityProgress=" + json.dumps({
                        "num_envs": count, "elapsed_seconds": round(now - started),
                        "gpu_used_mib": sample["used_mib"],
                        "latest_worker_line": lines[-1] if lines else "starting",
                    }, ensure_ascii=False), flush=True)
                    last_progress = now
                time.sleep(0.5)
        finally:
            if child is not None:
                stop_child(child)
    report_path = output / "report.json"
    report = json.loads(report_path.read_text()) if report_path.is_file() else {}
    status = "TIMEOUT" if timed_out else report.get("status", "FAIL")
    if child is None or child.returncode != 0:
        status = "TIMEOUT" if timed_out else "FAIL"
    row = {
        "num_envs": count, "status": status,
        "exit_code": child.returncode if child is not None else None,
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "gpu_total_mib": initial["total_mib"],
        "gpu_used_before_case_mib": initial["used_mib"],
        "sampled_peak_gpu_used_mib": peak,
        "sampled_max_gpu_utilization_percent": max_utilization,
        "sampled_peak_process_rss_kib": peak_rss_kib,
        "gpu_memory_scope": "whole allocated GPU, including Warp and other GPU processes; sampled at >=0.5 s intervals",
        "headroom_fraction": 1 - peak / initial["total_mib"],
        "eligible_with_15_percent_headroom": status == "PASS" and peak <= initial["total_mib"] * 0.85,
        "transitions_per_second": report.get("transitions_per_second"),
        "mean_iteration_seconds": report.get("mean_iteration_seconds"),
        "projected_2000_updates_hours_excluding_setup_eval_save": report.get(
            "projected_2000_updates_hours_excluding_setup_eval_save"),
        "is_out_of_memory": report.get("is_out_of_memory", False),
        "error": report.get("error"), "worker_report": str(report_path), "log": str(log),
    }
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpu", choices=("A800", "5090"), default="A800")
    parser.add_argument("--case-timeout", type=int, default=900)
    parser.add_argument("--max-seconds", type=int, default=2400)
    args = parser.parse_args()
    args.checkpoint = args.checkpoint.resolve()
    if hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() != EXPECTED_CHECKPOINT:
        raise ValueError("Stage 05 checkpoint hash mismatch")
    if not (60 <= args.case_timeout <= 900 and 60 <= args.max_seconds <= 2400):
        raise ValueError("Capacity is limited to 15 minutes per case, 40 minutes total")
    if not Path("/root/autodl-tmp").is_dir():
        raise ValueError("Run this in the AutoDL cloud container")
    repo = Path(__file__).resolve().parents[1]
    root = args.output.resolve()
    if root.is_relative_to(repo):
        raise ValueError("Output must be outside the repository")
    root.mkdir(parents=True, exist_ok=False)
    os.chdir(repo)
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ.setdefault("MUJOCO_GL", "egl")
    cache = Path("/root/autodl-tmp/microduck-double-balance/cache")
    os.environ["XDG_CACHE_HOME"] = str(cache / "xdg")
    os.environ["TMPDIR"] = str(cache / "tmp")
    for name in ("xdg", "tmp"):
        (cache / name).mkdir(parents=True, exist_ok=True)
    summary = {"status": "RUNNING", "formal_training_started": False,
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "checkpoint_sha256": EXPECTED_CHECKPOINT,
               "warmup_updates_per_case": 3, "measured_updates_per_case": 12,
               "max_capacity_seconds": args.max_seconds,
               "matrix": [], "suggested_num_envs": None}
    summary_path = root / "capacity-summary.json"

    def save_summary():
        temp = summary_path.with_suffix(".tmp")
        temp.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        temp.replace(summary_path)

    save_summary()
    print(f"Stage07CapacityOutput={root}", flush=True)
    deadline = time.monotonic() + args.max_seconds
    try:
        for count in (512, 1024, 2048, 4096):
            print(f"Stage07CapacityCaseStart={count}", flush=True)
            row = run_case(args, count, root, deadline)
            summary["matrix"].append(row)
            save_summary()
            print("Stage07CapacityCase=" + json.dumps(row, allow_nan=False), flush=True)
            if row["status"] != "PASS":
                break
        eligible = [r for r in summary["matrix"] if r["eligible_with_15_percent_headroom"]]
        if eligible:
            fastest = max(r["transitions_per_second"] for r in eligible)
            candidates = [r for r in eligible if r["transitions_per_second"] >= 0.95 * fastest]
            summary["suggested_num_envs"] = max(r["num_envs"] for r in candidates)
        complete = len(summary["matrix"]) == 4 and all(r["status"] == "PASS" for r in summary["matrix"])
        summary["status"] = "PASS" if complete and eligible else "REVIEW_REQUIRED"
    except BaseException as exc:
        summary.update(status="FAIL", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        summary["finished_utc"] = datetime.now(timezone.utc).isoformat()
        save_summary()
    print(f"Stage07CapacitySummary={summary_path}", flush=True)
    print(f"Stage07Capacity={summary['status']}", flush=True)
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
