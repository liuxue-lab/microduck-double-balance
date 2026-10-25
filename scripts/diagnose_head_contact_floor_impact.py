"""Approved five-case CPU diagnostic on copies of the archived material model.

This is not a policy evaluation or a replacement for the blocked v2 validator.
Only the five declared timestep/solref combinations may be integrated, once.
"""
from __future__ import annotations

import argparse
import copy
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import traceback
import zipfile

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_loader = importlib.util.spec_from_file_location("stage11_frozen_v2", ROOT / "scripts/validate_head_contact_physics.py")
v2 = importlib.util.module_from_spec(_loader)
_loader.loader.exec_module(v2)
require, save = v2.require, v2.save

SOURCE = Path("/home/lx/microduck-double-balance/artifacts/double-balance-stage11/physics/20261003T094423Z")
SOURCE_ZIP_SHA256 = "d2c6cb2c788ee3b7020f6557dac1e0646940bebd43b290dc4b3f5c6e8110278d"
SOURCE_COMMIT = "51df1774385fec998b8a8e7f7dcfcde5f5a10a27"
SOURCE_VIDEO_SHA256 = "c55a11f706dc097e3103d6529d8601f010603d98505a2257313e186bef4b9335"
V2_SOURCE_SHA256 = "d5f88ffc2a65c3d98a6714d8aa67b8bffe8144f81be6e1d1fd6d89dcb908067c"
START_INDEX, START_TIME, SOURCE_DT = 195, .390, .002
DURATION, PENETRATION_MAX, WALL_SECONDS, TOTAL_STEPS_MAX = .05, .001, 300, 325
CASES = (
    {"name": "A0", "dt": .002, "solref0": .003, "max_steps": 25},
    {"name": "A1", "dt": .001, "solref0": .003, "max_steps": 50},
    {"name": "A2", "dt": .0005, "solref0": .003, "max_steps": 100},
    {"name": "B1", "dt": .001, "solref0": .004, "max_steps": 50},
    {"name": "B2", "dt": .0005, "solref0": .004, "max_steps": 100},
)
INPUT_NAMES = (
    "run-summary.json", "local-v2/physics-review.json", "local-v2/material-fixture.mjb",
    "local-v2/material-options.json", "local-v2/material-scope.json",
    "local-v2/cpu/release-y-minus12/states.npz",
    "local-v2/cpu/release-y-minus12/contacts.jsonl.gz",
    "local-v2/cpu/release-y-minus12/raw-contacts.jsonl.gz",
    "local-v2/material-cpu-release-y-minus12-BLOCKED.mp4",
)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def verify_source(source):
    """Anchor local files to the already reviewed archive SHA, not a mutable manifest."""
    source = Path(source)
    archive = source.with_suffix(".zip")
    require(archive.is_file() and digest(archive) == SOURCE_ZIP_SHA256,
            "Canonical v2 ZIP missing or SHA256 changed; preserve files and stop")
    hashes = {}
    with zipfile.ZipFile(archive) as z:
        require(len(z.namelist()) == len(set(z.namelist())), "Duplicate paths in source ZIP")
        inventory_bytes = z.read("evidence-inventory.json")
        require((source / "evidence-inventory.json").read_bytes() == inventory_bytes,
                "Local v2 inventory differs from the pinned ZIP")
        inventory = json.loads(inventory_bytes)
        for name in INPUT_NAMES:
            p = source / name
            require(p.is_file() and not p.is_symlink() and p.resolve().is_relative_to(source.resolve()),
                    "Missing/unsafe canonical input: " + name)
            expected = hashlib.sha256(z.read(name)).hexdigest()
            require(digest(p) == expected == inventory[name]["sha256"], "Source input SHA mismatch: " + name)
            hashes[name] = {"sha256": expected, "bytes": p.stat().st_size}
    r = load_json(source / "run-summary.json")
    require(r["status"] == "MATERIAL_BLOCKED_REVIEW_REQUIRED" and r["physics_integration_steps"] == 4697,
            "Unexpected source run status/count")
    require(r["head"] == v2.BASELINE and r["head_after"] == v2.BASELINE,
            "Unexpected source code baseline")
    require(digest(ROOT / "scripts/validate_head_contact_physics.py") == V2_SOURCE_SHA256,
            "Frozen v2 validator changed")
    return {"zip_sha256": SOURCE_ZIP_SHA256, "zip_bytes_verified_on_laptop": True,
            "artifact_commit": SOURCE_COMMIT, "source_directory": str(source), "files": hashes}


def model_signature(model):
    arrays = {}
    for name in dir(model):
        if name.startswith("_"): continue
        value = getattr(model, name)
        if isinstance(value, np.ndarray):
            arrays[name] = {"shape": list(value.shape), "dtype": str(value.dtype),
                            "sha256": hashlib.sha256(value.tobytes()).hexdigest()}
    return {"arrays": arrays, "options": json.loads(json.dumps(v2.options_dict(model), default=v2.serial)),
            "dimensions": [model.nq, model.nv, model.nu, model.na, model.neq, model.ngeom, model.nbody]}


def configured_copy(original, case):
    require(case in CASES, "Case is outside the approved five-case matrix")
    before = model_signature(original)
    model = copy.deepcopy(original)
    model.opt.timestep = case["dt"]
    sphere = model.geom("sphere").id
    model.geom_solref[sphere, 0] = case["solref0"]
    after = model_signature(model)
    require(before["dimensions"] == after["dimensions"], "Model dimensions changed")
    expected_options = copy.deepcopy(before["options"]); expected_options["timestep"] = case["dt"]
    require(after["options"] == expected_options, "Undeclared solver/options change")
    expected_solref = original.geom_solref.copy(); expected_solref[sphere, 0] = case["solref0"]
    require(np.array_equal(model.geom_solref, expected_solref), "Undeclared solref change")
    require({k: val for k, val in after["arrays"].items() if k != "geom_solref"} ==
            {k: val for k, val in before["arrays"].items() if k != "geom_solref"},
            "Material geometry/mass/inertia/friction/masks changed")
    require(model_signature(original) == before, "Original in-memory model was modified")
    changes = {"timestep": [original.opt.timestep, model.opt.timestep],
               "sphere_solref0": [float(original.geom_solref[sphere, 0]), float(model.geom_solref[sphere, 0])],
               "all_other_model_arrays_and_options_unchanged": True}
    return model, changes


def read_source_states(source, model, out):
    prefix = source / "local-v2/cpu/release-y-minus12"
    with np.load(prefix / "states.npz", allow_pickle=False) as data:
        qpos, qvel, dt = data["qpos"].copy(), data["qvel"].copy(), float(data["dt"])
    require(qpos.shape == (198, 7) and qvel.shape == (198, 6) and dt == SOURCE_DT,
            "Canonical state shape or timestep differs from the reviewed run")
    require(np.isfinite(qpos).all() and np.isfinite(qvel).all(), "Nonfinite archived state")
    with gzip.open(prefix / "contacts.jsonl.gz", "rt") as f: contacts = [json.loads(line) for line in f]
    floor = [c for c in contacts if c["other"] == "floor"]
    require(len(floor) == 1 and floor[0]["step"] == 196 and floor[0]["time_s"] == .392,
            "First-floor-contact identity differs")
    expected = floor[0]
    require(abs(expected["dist"] + .0026671883758180547) < 1e-15,
            "Canonical impact overlap differs from reviewed record")
    with gzip.open(prefix / "raw-contacts.jsonl.gz", "rt") as f: raw = [json.loads(line) for line in f]
    require(len(raw) == 197 and all(r["step"] == i for i, r in enumerate(raw)), "Raw contact step sequence differs")
    require(not raw[195]["contacts"], "Restart state already has a collision")
    floor_raw = [c for c in raw[196]["contacts"] if model.geom("floor").id in c["geom"]]
    require(len(floor_raw) == 1 and floor_raw[0]["dist"] == expected["dist"], "Raw/derived contact mismatch")
    require(np.array_equal(qpos[196, :3], np.asarray(expected["ball_center_world_m"])),
            "Contact/state phase mismatch")
    radius = float(model.geom_size[model.geom("sphere").id, 0])
    g = model.opt.gravity.copy(); floor_z = float(model.geom_pos[model.geom("floor").id, 2])
    predicted_velocity = qvel[195, :3] + dt * g
    predicted_position = qpos[195, :3] + dt * predicted_velocity
    position_residual = float(np.max(np.abs(predicted_position - qpos[196, :3])))
    velocity_residual = float(np.max(np.abs(predicted_velocity - qvel[196, :3])))
    require(position_residual <= 1e-12 and velocity_residual <= 1e-12, "Archived free-flight identity failed")
    body = int(model.geom_bodyid[model.geom("sphere").id]); mass = float(model.body_mass[body])
    predicted_vz = qvel[196, 2] + dt * (g[2] + expected["force"][0] / mass)
    result = {"new_integration_steps": 0, "source_is_canonical_laptop": True,
              "initial_index": START_INDEX, "initial_time_s": START_TIME,
              "initial_floor_gap_m": float(qpos[195, 2] - floor_z - radius),
              "initial_vertical_velocity_m_s": float(qvel[195, 2]),
              "predicted_first_overlap_m": float(floor_z + radius - predicted_position[2]),
              "position_residual_m": position_residual, "velocity_residual_m_s": velocity_residual,
              "normal_force_next_velocity_residual_m_s": float(predicted_vz - qvel[197, 2]),
              "head_contact_max_penetration_m": max(-c["dist"] for c in contacts if c["other"] == "head"),
              "first_floor_contact": expected,
              "states_194_through_197": [{"index": i, "time_s": i * dt, "qpos": qpos[i], "qvel": qvel[i]} for i in range(194, 198)]}
    require(abs(result["normal_force_next_velocity_residual_m_s"]) <= 1e-12, "Archived normal impulse identity failed")
    save(out / "source-state-review.json", result)
    return qpos[195].copy(), qvel[195].copy(), expected


def initialize(model, qpos, qvel):
    require(model.nq == 7 and model.nv == 6 and model.nu == model.na == model.neq == 0,
            "Expected one free material sphere, no actuators or equalities")
    data = mujoco.MjData(model)
    data.qpos[:] = qpos; data.qvel[:] = qvel; data.time = START_TIME
    data.qacc_warmstart[:] = 0
    mujoco.mj_forward(model, data)
    require(data.ncon == 0, "Restart state contains contact")
    data.qacc_warmstart[:] = data.qacc
    return data


def baseline_match(first, expected):
    if first is None: return {"pass": False, "reason": "A0 produced no floor contact"}
    dt_error = abs(float(first["time_s"]) - float(expected["time_s"]))
    dist_error = abs(float(first["dist"]) - float(expected["dist"]))
    force_error = np.abs(np.asarray(first["force"]) - np.asarray(expected["force"]))
    ok = dt_error <= 1e-12 and dist_error <= 1e-9 and np.allclose(first["force"], expected["force"], rtol=1e-5, atol=1e-7)
    return {"pass": bool(ok), "time_error_s": dt_error, "distance_error_m": dist_error,
            "force_component_absolute_error": force_error,
            "force_tolerance": {"rtol": 1e-5, "atol": 1e-7}}


def contact_failure(rows, solref0):
    for row in rows:
        require(row["allowed"] and row["other"] == "floor", "Unexpected contact in floor-only event diagnostic")
        require(row["dim"] == 4, "Effective condim changed")
        require(np.allclose(row["friction"], [.6, .6, .005, .0001, .0001], rtol=1e-6, atol=1e-9), "Friction changed")
        require(np.allclose(row["solref"], [solref0, 1.], rtol=1e-6, atol=1e-9), "Contact solref differs from declared case")
        require(np.allclose(row["solimp"], [.95, .99, .001, .5, 2.], rtol=1e-6, atol=1e-9), "Solimp changed")
        require(np.isfinite(np.r_[row["dist"], row["pos"], row["frame"], row["force"]]).all(), "Nonfinite contact")
    return any(-float(row["dist"]) > PENETRATION_MAX for row in rows)


def run_case(original, qpos, qvel, case, destination, deadline, progress):
    destination.mkdir(parents=True, exist_ok=False)
    summary = {"name": case["name"], "case": case, "status": "RUNNING", "steps": 0,
               "attempted_steps": 0, "max_penetration_m": 0., "first_floor_contact": None,
               "horizon_completed": False, "new_ppo_updates": 0, "full_robot_steps": 0}
    poses, velocities, times = [], [], []
    try:
        model, changes = configured_copy(original, case)
        data = initialize(model, qpos, qvel)
        summary["model_changes"] = changes
        summary["effective_normal_timeconst_s"] = max(case["solref0"], 2 * case["dt"])
        save(destination / "initialization.json", {"qpos": data.qpos, "qvel": data.qvel,
             "time_s": data.time, "qacc_warmstart": data.qacc_warmstart,
             "method": "Saved qpos/qvel; zero warm-start, mj_forward, copy free-flight qacc into warm-start",
             "complete_historical_state_restore": False, "initial_contacts": data.ncon})
        poses.append(data.qpos.copy()); velocities.append(data.qvel.copy()); times.append(float(data.time))
        with gzip.open(destination / "contacts.jsonl.gz", "wt") as log:
            for step in range(case["max_steps"]):
                require(time.monotonic() < deadline, "Approved five-minute wall limit reached")
                before_qpos, before_qvel = data.qpos.copy(), data.qvel.copy()
                before_time = START_TIME + step * case["dt"]
                require(abs(float(data.time) - before_time) < 1e-11, "Diagnostic time accounting drift")
                summary["attempted_steps"] += 1
                progress(summary)
                mujoco.mj_step(model, data)
                summary["steps"] += 1
                poses.append(data.qpos.copy()); velocities.append(data.qvel.copy()); times.append(float(data.time))
                require(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all(), "Nonfinite state")
                records = v2.cpu_records(model, data)
                rows = v2.contact_rows(model, before_qpos, before_qvel, records, step, "cpu")
                for row in rows: row["time_s"] = before_time
                log.write(json.dumps({"step": step, "time_s": before_time, "qpos_before": before_qpos,
                    "qvel_before": before_qvel, "qpos_after": data.qpos, "qvel_after": data.qvel,
                    "contacts": rows, "raw_contacts": records}, default=v2.serial, allow_nan=False) + "\n")
                require(len(rows) == len(records), "Unexpected non-sphere contact")
                for row in rows:
                    summary["max_penetration_m"] = max(summary["max_penetration_m"], -float(row["dist"]))
                    if row["other"] == "floor" and summary["first_floor_contact"] is None:
                        summary["first_floor_contact"] = row
                if contact_failure(rows, case["solref0"]):
                    summary["status"] = "PENETRATION_BLOCKED"; break
                progress(summary)
            else:
                require(summary["first_floor_contact"] is not None, "No floor impact within approved horizon")
                summary["horizon_completed"] = True
                summary["status"] = "DIAGNOSTIC_GATES_PASS_FULL_MATERIAL_VALIDATION_STILL_BLOCKED"
    except Exception as exc:
        summary["status"] = "ERROR_STOP_MATRIX"; summary["error"] = str(exc)
        (destination / "error.txt").write_text(traceback.format_exc())
    finally:
        summary["completed_step_count_exact"] = summary["steps"] == summary["attempted_steps"]
        summary["elapsed_simulation_s"] = summary["steps"] * case["dt"]
        np.savez_compressed(destination / "states.npz", qpos=np.asarray(poses), qvel=np.asarray(velocities), time=np.asarray(times), dt=case["dt"])
        save(destination / "summary.json", summary)
        progress(summary)
    return summary


def export_source_video(source, out):
    target = out / "source-v2-blocked.mp4"
    origin = source / "local-v2/material-cpu-release-y-minus12-BLOCKED.mp4"
    require(digest(origin) == SOURCE_VIDEO_SHA256, "Source video differs from reviewed metadata")
    shutil.copyfile(origin, target)
    result = {"original_video_sha256": digest(target), "source_copy_no_resimulation": True,
              "human_visual_review": "PENDING", "frames": []}
    try:
        require(shutil.which("ffmpeg") is not None, "ffmpeg unavailable for source-video frames")
        for index in (0, 6, 10, 11):
            output = out / f"source-v2-frame-{index:02d}.png"
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(target),
                            "-vf", f"select=eq(n\\,{index})", "-frames:v", "1", str(output)],
                           check=True, capture_output=True, timeout=20)
            require(output.is_file() and output.stat().st_size > 0, "Requested video frame missing")
            result["frames"].append({"file": output.name, "video_frame_index": index, "sha256": digest(output)})
    except Exception as exc: result["frame_export_error"] = str(exc)
    save(out / "source-video-export.json", result)


def run_matrix(original, qpos, qvel, expected, out, report, deadline):
    initial_signature = model_signature(original)
    for case in CASES:
        prior_steps = sum(r["steps"] for r in report["cases"])
        def progress(current):
            require(prior_steps + current["attempted_steps"] <= TOTAL_STEPS_MAX, "Approved global step budget exceeded")
            save(out / "progress.json", {"completed_cases": report["cases"], "active_case": current,
                 "completed_steps": prior_steps + current["steps"],
                 "attempted_steps": prior_steps + current["attempted_steps"], "new_ppo_updates": 0})
        result = run_case(original, qpos, qvel, case, out / "cases" / case["name"], deadline, progress)
        report["cases"].append(result)
        if case["name"] == "A0":
            report["baseline_reproduction"] = baseline_match(result["first_floor_contact"], expected)
            save(out / "diagnostic-review.json", report)
            require(report["baseline_reproduction"]["pass"], "A0 does not reproduce archived impact; stop all remaining cases")
        require(result["status"] != "ERROR_STOP_MATRIX", result.get("error", "Case execution error"))
        require(model_signature(original) == initial_signature, "Original model changed between cases")
        save(out / "diagnostic-review.json", report)
    report["original_model_unchanged"] = True
    report["status"] = "FIVE_CASE_DIAGNOSTIC_REVIEW_REQUIRED_MATERIAL_STILL_BLOCKED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--approved-five-case-diagnostic", action="store_true", required=True)
    args = parser.parse_args()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=False)
    report = {"status": "STARTED", "cases": [], "new_ppo_updates": 0, "cloud_hours": 0,
              "stage11_complete": False, "full_robot_steps": 0, "formal_task_migration": False,
              "source_material_steps": 4697, "source_material_status": "MATERIAL_BLOCKED_REVIEW_REQUIRED",
              "sampling_phase": "Same pre-integration contacts and solver-force phase as the frozen v2 validator",
              "approval": {"user_message": "是，继续", "time": "2026-10-03T21:43:11+08:00"},
              "remaining_full_matrix_automatically_run": False}
    started = time.monotonic(); deadline = started + WALL_SECONDS
    source = args.source.resolve(); provenance = None
    try:
        require(source == SOURCE, "Diagnostic input must be the approved canonical v2 evidence directory")
        v2.local_preflight(out)
        provenance = verify_source(source); save(out / "input-provenance.json", provenance)
        model = mujoco.MjModel.from_binary_path(str(source / "local-v2/material-fixture.mjb"))
        require(model.opt.timestep == SOURCE_DT and model.opt.disableflags == 0, "Original timestep/flags changed")
        require(model.nq == 7 and model.nv == 6 and model.nu == model.na == model.neq == 0, "Not a material fixture")
        require(np.array_equal(model.opt.gravity, [0, 0, -9.81]), "Unexpected gravity")
        qpos, qvel, expected = read_source_states(source, model, out)
        export_source_video(source, out)
        run_matrix(model, qpos, qvel, expected, out, report, deadline)
    except Exception as exc:
        report["status"] = "DIAGNOSTIC_BLOCKED_REVIEW_REQUIRED"; report["error"] = str(exc)
        (out / "error.txt").write_text(traceback.format_exc())
    finally:
        report["diagnostic_completed_steps"] = sum(c["steps"] for c in report["cases"])
        report["diagnostic_attempted_steps"] = sum(c["attempted_steps"] for c in report["cases"])
        report["wall_seconds"] = time.monotonic() - started
        if provenance:
            report["source_files_unchanged"] = all(digest(source / n) == m["sha256"] for n, m in provenance["files"].items())
            if not report["source_files_unchanged"]: report["status"] = "SOURCE_CHANGED_BLOCKED"
        report["diagnostic_source_sha256"] = digest(Path(__file__))
        save(out / "diagnostic-review.json", report)
    print("Stage11Diagnostic=" + report["status"], flush=True)
    print("Stage11DiagnosticSteps=" + str(report["diagnostic_completed_steps"]), flush=True)
    return 0 if report["status"] == "FIVE_CASE_DIAGNOSTIC_REVIEW_REQUIRED_MATERIAL_STILL_BLOCKED" else 2


if __name__ == "__main__": raise SystemExit(main())
