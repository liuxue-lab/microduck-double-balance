#!/usr/bin/env python3
"""Collect cloud metadata without installing packages or launching CUDA/PPO.

Run from any directory with the system Python. JSON goes to stdout; optional
--output writes a new report, never overwrites an existing file.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(argv, *, timeout=15):
    try:
        result = subprocess.run(argv, capture_output=True, text=True,
                                timeout=timeout, check=False)
        return {"returncode": result.returncode, "stdout": result.stdout.strip(),
                "stderr": result.stderr.strip()[:2000]}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"returncode": None, "error": type(exc).__name__}


def parse_gpu_rows(text):
    rows = []
    for row in csv.reader(io.StringIO(text)):
        if not row:
            continue
        if len(row) != 5:
            raise ValueError("Unexpected nvidia-smi columns")
        name, total, used, driver, uuid = (part.strip() for part in row)
        rows.append({"name": name, "total_mib": int(total), "used_mib": int(used),
                     "driver": driver, "uuid": uuid})
    return rows


def file_hash(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", choices=("A800", "5090"), default="5090")
    parser.add_argument("--root", type=Path,
                        default=Path("/root/autodl-tmp/microduck-double-balance"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    repo = root / "workspace"
    report = {"stage": 8, "purpose": "read-only metadata before deployment",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "expected_gpu": args.gpu, "platform": platform.machine(),
              "bootstrap_python": sys.executable,
              "bootstrap_python_version": platform.python_version(),
              "formal_training_started": False, "ppo_updates": 0,
              "capacity_measured": False, "cuda_compute_executed": False,
              "training_ready": False}
    gpu_result = command(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version,uuid",
                          "--format=csv,noheader,nounits"])
    report["gpu_query"] = gpu_result
    try:
        report["gpus"] = parse_gpu_rows(gpu_result.get("stdout", ""))
    except ValueError as exc:
        report["gpus"] = []
        report["gpu_parse_error"] = str(exc)
    devices = report["gpus"]
    if args.gpu == "5090":
        # Reject other variants with 5090 in the name; capacity is still untested.
        expected_name = "NVIDIA GeForce RTX 5090"
        match = len(devices) == 1 and devices[0]["name"] == expected_name
        min_memory = 30000
    else:
        match = len(devices) == 1 and "A800" in devices[0]["name"]
        min_memory = 75000
    report["hardware_identity_matches"] = bool(
        match and devices[0]["total_mib"] >= min_memory)
    report["cpu_affinity_count"] = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    cgroup = Path("/sys/fs/cgroup")
    report["cgroup_limits"] = {}
    for name in ("memory.max", "memory.current", "cpu.max", "cpuset.cpus.effective",
                 "memory/memory.limit_in_bytes", "cpu/cpu.cfs_quota_us",
                 "cpu/cpu.cfs_period_us"):
        path = cgroup / name
        if path.is_file():
            report["cgroup_limits"][name] = path.read_text().strip()
    report["disks"] = {}
    for path in (Path("/"), Path("/root/autodl-tmp")):
        if path.exists():
            usage = shutil.disk_usage(path)
            report["disks"][str(path)] = {"total_bytes": usage.total, "free_bytes": usage.free}
    report["repository_exists"] = (repo / ".git").exists()
    if report["repository_exists"]:
        report["local_head"] = command(["git", "-C", str(repo), "rev-parse", "HEAD"])
        report["local_worktree"] = command(["git", "-C", str(repo), "status", "--porcelain"])
    interpreter = repo / ".venv/bin/python"
    report["project_python_exists"] = interpreter.is_file()
    if interpreter.is_file():
        # Query distributions without importing torch/mjlab or allocating CUDA.
        snippet = (
            "import importlib.metadata as m,json,sys; "
            "names=['torch','mjlab','mujoco','mujoco-warp','warp-lang','rsl-rl-lib']; "
            "installed={d.metadata['Name'].lower():d.version for d in m.distributions()}; "
            "print(json.dumps({'python':sys.version,'versions':{n:installed.get(n) for n in names}}))"
        )
        report["installed_environment"] = command([str(interpreter), "-c", snippet])
    report["checkpoints"] = []
    artifact_root = root / "artifacts/double-balance-stage07"
    if artifact_root.is_dir():
        for name in ("update_001000.pt", "update_006000.pt"):
            for path in sorted(artifact_root.rglob(name)):
                if path.is_file():
                    report["checkpoints"].append({"path": str(path), "bytes": path.stat().st_size,
                                                  "sha256": file_hash(path)})
    report["status"] = "METADATA_COLLECTED" if report["hardware_identity_matches"] else "HARDWARE_REVIEW_REQUIRED"
    report["note"] = "Metadata only: does not certify dependency imports, CUDA kernels, capacity, restore, or training."
    content = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(content)
    print(content, end="")
    return 0 if report["hardware_identity_matches"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
