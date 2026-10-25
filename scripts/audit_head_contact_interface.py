"""Independent Stage11 contact-array audit. Forward only; never a runtime task.

The adapter reads derived kinematics and contact forces from one completed
forward phase. It does not infer a pre-step timestamp from data.time, use sensor
caches, aggregate success, or move any existing manager/timer boundary.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs/audits/stage-11-head-interface-contract.json"
PHASES = ("static_forward", "reward_metrics_pre_reset", "observation_post_forward")
DERIVED = ("xpos", "xquat", "cvel", "subtree_com", "site_xpos", "site_xmat")


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def serial(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    raise TypeError(type(x).__name__)


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=serial, indent=2, allow_nan=False) + "\n")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_audit(out):
    """Capture installed bytes before importing project/runtime modules."""
    contract = json.loads(CONTRACT.read_text())
    actual = {}
    for module, paths in contract["reviewed_dependency_sources"].items():
        origin = importlib.util.find_spec(module).origin
        require(origin is not None, "No dependency source: " + module)
        root = Path(origin).parent
        for name, sha in paths.items():
            data = (root / name).read_bytes()
            key = module + "/" + name
            actual[key] = {"sha256": hashlib.sha256(data).hexdigest(), "reviewed_sha256": sha,
                           "bytes": len(data), "matches": hashlib.sha256(data).hexdigest() == sha}
            dst = out / "installed-source" / module / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(data)
    save(out / "dependency-sources.json", actual)
    require(all(x["matches"] for x in actual.values()),
            "DEPENDENCY_SOURCE_REVIEW_REQUIRED: installed source differs; no model or forward started")
    return actual


@contextmanager
def zero_integration_guard():
    import mujoco
    import mujoco_warp as mw
    fw = importlib.import_module("mujoco_warp._src.forward")
    attempts = []
    def blocked(name):
        def stop(*args, **kwargs):
            attempts.append(name)
            raise RuntimeError("ZERO_INTEGRATION_GUARD: " + name)
        return stop
    with ExitStack() as stack:
        for module, names in ((mujoco, ("mj_step", "mj_step1", "mj_step2", "mj_Euler", "mj_RungeKutta", "mj_implicit", "mj_integratePos")),
                              (mw, ("step", "step1", "step2", "euler", "rungekutta4", "implicit")),
                              (fw, ("step", "step1", "step2", "euler", "rungekutta4", "implicit"))):
            for name in names:
                if hasattr(module, name):
                    stack.enter_context(patch.object(module, name, blocked(module.__name__ + "." + name)))
        yield attempts


def point_velocity(state, body, root, point):
    cvel = state["cvel"][body]
    return cvel[3:] + np.cross(cvel[:3], point - state["subtree_com"][root])


def pair_contacts(records, sphere, other, world):
    """Force is on geom2; return normal and force oriented other -> sphere."""
    result = []
    for c in records:
        if c["world"] != world or {c["geom1"], c["geom2"]} != {sphere, other}:
            continue
        sign = 1.0 if c["geom2"] == sphere else -1.0
        require(all(np.isfinite(c[k]).all() for k in ("frame", "pos", "force_world", "torque_world"))
                and np.isfinite(c["dist"]), "Nonfinite matched contact")
        frame = np.asarray(c["frame"])
        require(np.max(np.abs(frame @ frame.T-np.eye(3))) <= 2e-5, "Invalid contact basis")
        row = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in c.items()}
        row["normal_to_ball"] = sign * np.asarray(c["frame"])[0]
        row["force_on_ball"] = sign * np.asarray(c["force_world"])
        row["torque_on_ball"] = sign * np.asarray(c["torque_world"])
        row["normal_force"] = float(np.dot(row["normal_to_ball"], row["force_on_ball"]))
        row["active"] = bool(c["dist"] <= 0 and c["efc_address"] >= 0 and row["normal_force"] > 0)
        result.append(row)
    return result


def snapshot(state, contacts, ids, *, phase, revision, world, data_time):
    """Pure frozen snapshot; caller owns completed-forward identity, not a timer."""
    require(phase in PHASES and isinstance(revision, int) and revision >= 0, "Unknown state phase/revision")
    require(all(np.isfinite(state[k]).all() for k in DERIVED), "Nonfinite derived state")
    sid, ball, jaw = ids["site"], ids["ball_body"], ids["head_body"]
    origin = state["site_xpos"][sid].copy()
    rotation = state["site_xmat"][sid].reshape(3, 3).copy()
    pos = state["xpos"][ball].copy()
    omega_ball, omega_head = state["cvel"][ball, :3].copy(), state["cvel"][jaw, :3].copy()
    velocity = point_velocity(state, ball, ids["ball_root"], pos)
    origin_velocity = point_velocity(state, jaw, ids["head_root"], origin)
    head = pair_contacts(contacts, ids["ball_geom"], ids["head_geom"], world)
    floor = pair_contacts(contacts, ids["ball_geom"], ids["floor_geom"], world)
    for c in head:
        point = np.asarray(c["pos"])
        relative = velocity + np.cross(omega_ball, point-pos) - point_velocity(state, jaw, ids["head_root"], point)
        normal = c["normal_to_ball"]
        c["relative_contact_velocity"] = relative
        c["slip"] = relative - np.dot(relative, normal)*normal
    return {"phase": phase, "revision": revision, "world": world,
            "data_time": float(data_time), "time_semantics": "data clock; NOT inferred contact timestamp",
            "origin": origin, "rotation": rotation, "ball_position": pos,
            "ball_velocity": velocity.copy(), "ball_omega": omega_ball, "head_omega": omega_head,
            "position_material": rotation.T@(pos-origin),
            "rate_material": rotation.T@(velocity-origin_velocity-np.cross(omega_head, pos-origin)),
            "head_contacts": head, "floor_contacts": floor,
            "active_head": sum(c["active"] for c in head), "active_floor": sum(c["active"] for c in floor)}


def cpu_contacts(model, data):
    import mujoco
    rows = []
    for i in range(data.ncon):
        c = data.contact[i]
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        frame = c.frame.reshape(3, 3).copy()
        rows.append({"contact_id": i, "world": 0, "geom1": int(c.geom1), "geom2": int(c.geom2),
                     "dist": float(c.dist), "pos": c.pos.copy(), "frame": frame,
                     "efc_address": int(c.efc_address), "force_world": frame.T@force[:3],
                     "torque_world": frame.T@force[3:]})
    return rows


def resolve_ids(model):
    def unique(kind, count, suffix):
        matches = [i for i in range(count) if getattr(model, kind)(i).name.split("/")[-1] == suffix]
        require(len(matches) == 1, "Ambiguous compiled identity: " + suffix)
        return matches[0]
    head = unique("geom", model.ngeom, "stage11_head_contact_surface")
    ball = unique("geom", model.ngeom, "top_ball_sphere")
    floor = unique("geom", model.ngeom, "terrain")
    site = unique("site", model.nsite, "stage11_head_material_frame")
    hb, bb = int(model.geom_bodyid[head]), int(model.geom_bodyid[ball])
    return {"head_geom": head, "ball_geom": ball, "floor_geom": floor, "site": site,
            "head_body": hb, "ball_body": bb, "head_root": int(model.body_rootid[hb]),
            "ball_root": int(model.body_rootid[bb])}


def cpu_cases(model, ids, out):
    import mujoco
    states, records, reports = [], [], []
    j = int(model.body_jntadr[ids["ball_body"]])
    require(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE, "Sphere joint changed")
    qa, va = int(model.jnt_qposadr[j]), int(model.jnt_dofadr[j])
    cases = (("head-gap", .0005, False), ("head-contact-spin", -.00005, False),
             ("floor-contact", -.00005, False), ("turned-head-contact", -.00005, True))
    for revision, (name, gap, turn) in enumerate(cases):
        data = mujoco.MjData(model)
        if model.nkey:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        if turn:
            y = next(i for i in range(model.njnt) if model.joint(i).name.split("/")[-1] == "head_yaw")
            data.qpos[model.jnt_qposadr[y]] += np.pi/2
            require(not model.jnt_limited[y] or model.jnt_range[y, 0] <= data.qpos[model.jnt_qposadr[y]] <= model.jnt_range[y, 1], "Yaw configuration out of range")
        mujoco.mj_forward(model, data)
        origin = data.site_xpos[ids["site"]].copy()
        rotation = data.site_xmat[ids["site"]].reshape(3, 3)
        center = origin + rotation@np.array([0., 0., .02+gap])
        if name == "floor-contact":
            center = np.array([.5, .5, .02+gap])
        data.qpos[qa:qa+7] = np.r_[center, 1., 0., 0., 0.]
        data.qvel[:] = 0
        data.qvel[va:va+6] = [.02, -.015, 0., .3, -.2, .4]
        before = (data.qpos.copy(), data.qvel.copy(), float(data.time))
        mujoco.mj_forward(model, data)
        require(np.array_equal(data.qpos, before[0]) and np.array_equal(data.qvel, before[1]) and data.time == before[2] == 0, "Forward advanced state")
        derived = {k: np.array(getattr(data, k), copy=True) for k in DERIVED}
        contacts = cpu_contacts(model, data)
        snap = snapshot(derived, contacts, ids, phase="static_forward", revision=revision, world=0, data_time=data.time)
        save(out/("cpu-config-"+name+".json"), {"snapshot":snap,"qpos":data.qpos.copy(),"qvel":data.qvel.copy(),"contacts":contacts})
        expect = "floor" if name == "floor-contact" else "head" if gap < 0 else "none"
        require((snap["active_head"] > 0) == (expect == "head") and (snap["active_floor"] > 0) == (expect == "floor"), "Unexpected CPU pair membership: " + name)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, ids["ball_body"], velocity, 0)
        np.testing.assert_allclose(snap["ball_velocity"], velocity[3:], atol=1e-12, rtol=0)
        np.testing.assert_allclose(snap["ball_omega"], velocity[:3], atol=1e-12, rtol=0)
        # These are configured state samples, not a rollout or material stability test.
        state = {"name": name, "expected_pair": expect, "qpos": data.qpos.copy(), "qvel": data.qvel.copy(), "ctrl": data.ctrl.copy()}
        states.append(state); records.append(derived); reports.append(snap)
    save(out/"cpu-forward-snapshots.json", reports)
    save(out/"configured-states.json", states)
    return states, records, reports


def warp_audit(model, ids, states, cpu_snapshots, out):
    """Two actual Warp worlds; no Simulation, Manager or CUDA graph construction."""
    import torch
    require(torch.cuda.is_available() and "5060" in torch.cuda.get_device_name(0), "Expected authorized local RTX5060; no fallback to another GPU")
    gpu = torch.cuda.get_device_name(0)
    return warp_forward_cases(model, ids, states, cpu_snapshots, out, "cuda:0", gpu)


def warp_forward_cases(model, ids, states, cpu_snapshots, out, device, device_name):
    """Device-independent extraction kernel; CLI always requires local RTX5060."""
    import mujoco
    import mujoco_warp as mw
    import warp as wp
    reports = []
    # Every state appears in both world IDs; swapping also probes stale snapshots.
    batches = ((0, 2), (2, 0), (1, 3), (3, 1))
    with wp.ScopedDevice(device):
        wm = mw.put_model(model)
        wd = mw.put_data(model, mujoco.MjData(model), nworld=2, nconmax=512, njmax=4096)
        for revision, batch in enumerate(batches):
            q = np.stack([states[k]["qpos"] for k in batch]).astype(np.float32)
            v = np.stack([states[k]["qvel"] for k in batch]).astype(np.float32)
            ctrl = np.stack([states[k]["ctrl"] for k in batch]).astype(np.float32)
            for dest, source in ((wd.qpos, q), (wd.qvel, v), (wd.ctrl, ctrl)):
                wp.copy(dest, wp.array(source, dtype=dest.dtype, device=device))
            wd.qacc_warmstart.zero_()
            before_time = wd.time.numpy().copy()
            mw.forward(wm, wd)
            wp.synchronize()
            require(np.array_equal(wd.qpos.numpy(), q) and np.array_equal(wd.qvel.numpy(), v)
                    and np.array_equal(wd.time.numpy(), before_time) and np.all(before_time == 0), "Warp forward advanced state")
            ncon = int(wd.nacon.numpy()[0])
            require(ncon < wd.naconmax and np.max(wd.nefc.numpy()) < wd.njmax, "Warp contact/constraint capacity reached")
            contact_ids = wp.array(np.arange(ncon, dtype=np.int32), dtype=int, device=device)
            forces = wp.zeros(ncon, dtype=wp.spatial_vector, device=device)
            mw.contact_force(wm, wd, contact_ids, True, forces)
            wp.synchronize()
            cf = forces.numpy()
            raw = {k: getattr(wd.contact, k).numpy()[:ncon].copy() for k in ("worldid", "geom", "frame", "pos", "dist", "efc_address")}
            require(np.isfinite(cf).all(), "Nonfinite Warp contact force")
            contacts = [{"contact_id": c, "world": int(raw["worldid"][c]),
                         "geom1": int(raw["geom"][c, 0]), "geom2": int(raw["geom"][c, 1]),
                         "frame": raw["frame"][c], "pos": raw["pos"][c], "dist": float(raw["dist"][c]),
                         "efc_address": int(raw["efc_address"][c, 0]), "force_world": cf[c, :3], "torque_world": cf[c, 3:]}
                        for c in range(ncon)]
            derived = {k: getattr(wd, k).numpy().copy() for k in DERIVED}
            save(out/("warp-batch-"+str(revision)+".json"), {"batch":batch,"qpos":q,"qvel":v,"contacts":contacts})
            for world, case in enumerate(batch):
                snap = snapshot({k: x[world] for k, x in derived.items()}, contacts, ids,
                                phase="static_forward", revision=revision, world=world, data_time=0)
                expect = states[case]["expected_pair"]
                require((snap["active_head"] > 0) == (expect == "head") and (snap["active_floor"] > 0) == (expect == "floor"), "Warp pair membership differs: " + states[case]["name"])
                errors = {k: float(np.max(np.abs(np.asarray(snap[k])-np.asarray(cpu_snapshots[case][k]))))
                          for k in ("ball_position", "ball_velocity", "ball_omega", "origin", "rotation", "position_material", "rate_material")}
                require(max(errors.values()) <= 2e-5, "CPU/Warp derived kinematics mismatch")
                snap.update(case=states[case]["name"], cpu_kinematic_absolute_errors=errors)
                reports.append(snap)
            save(out/"warp-forward-snapshots.json", reports)
    return {"device": device_name, "worlds": 2, "forward_calls": 4, "configured_samples": 8,
            "new_integration_steps": 0, "kinematic_abs_tolerance": 2e-5,
            "force_comparison": "within-backend raw solver forces; no CPU/GPU force-convergence claim"}


def cache_probe(out):
    """Native ContactSensor cache with explicit two-state buffer replacement."""
    import mujoco
    import torch
    from types import SimpleNamespace
    from mjlab.sensor.contact_sensor import ContactSensorCfg, ContactMatch
    s = mujoco.MjSpec.from_string('<mujoco><worldbody><geom name="ground" type="plane" size="1 1 .1"/><body name="ball" pos="0 0 .01995"><freejoint/><geom name="sphere" type="sphere" size=".02" mass=".01"/></body></worldbody></mujoco>')
    sensor = ContactSensorCfg(name="cache_probe", primary=ContactMatch(mode="geom", pattern="ground"),
                              secondary=ContactMatch(mode="geom", pattern="sphere"),
                              fields=("found", "force", "normal", "tangent"), global_frame=True).build()
    sensor.edit_spec(s, {})
    m=s.compile();d=mujoco.MjData(m);mujoco.mj_forward(m,d)
    buffer=torch.tensor(d.sensordata[None].copy(), dtype=torch.float64)
    sensor.initialize(m, None, SimpleNamespace(sensordata=buffer, time=torch.zeros(1)), "cpu")
    first=sensor.data
    require(float(first.found.max()) > 0, "Cache probe initial contact missing")
    d.qpos[2]=.05;mujoco.mj_forward(m,d)
    buffer.copy_(torch.tensor(d.sensordata[None].copy(), dtype=torch.float64))
    stale=sensor.data
    require(stale is first and float(stale.found.max()) > 0, "Pinned ContactSensor cache behavior differs")
    sensor.update(0.0)
    require(float(sensor.data.found.max()) == 0, "Cache did not invalidate on update")
    r={"native_contact_sensor_cache_reproduced":True,"replacement_source":"two independently forwarded CPU configurations, no integrated step",
       "old_cache_retained_until_update":True,"new_raw_contact_adapter_uses_sensor_cache":False,"new_steps":0,
       "scope":"Demonstrates cache lifetime; not an executed Manager phase transition"}
    save(out/"sensor-cache-probe.json",r)
    return r


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forward-only", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic();attempts=[]
    report={"status":"INTERFACE_EXECUTION_BLOCKED","stage11_status":"MATERIAL_BLOCKED_REVIEW_REQUIRED",
            "stage11_complete":False,"new_integration_steps":0,"new_ppo_updates":0,"cloud_hours":0,
            "canonical_total_completed_steps":7895,"formal_task_migration":False,
            "full_manager_execution":False,"policy_success":None}
    try:
        report["dependency_source_count"]=len(source_audit(out))
        with zero_integration_guard() as attempts:
            from mjlab.scene import Scene
            from mjlab_microduck.tasks.microduck_head_contact_env_cfg import make_microduck_head_contact_env_cfg
            cfg=make_microduck_head_contact_env_cfg(static_audit_only=True)
            model=Scene(cfg.scene,device="cpu").compile();cfg.sim.mujoco.apply(model)
            # Existing startup guard stays enabled: no Manager or task registration.
            ids=resolve_ids(model);save(out/"compiled-identities.json",ids)
            audit=load_module("head_proto_fields",ROOT/"scripts/audit_head_task_prototype.py")
            signature={k:hashlib.sha256(np.asarray(getattr(model,k)).tobytes()).hexdigest() for k in audit.PHYSICS_FIELDS}
            states,_,snaps=cpu_cases(model,ids,out)
            report["cpu_configurations"]=len(states)
            report["cache_probe"]=cache_probe(out)
            report["warp"]=warp_audit(model,ids,states,snaps,out)
            require(all(hashlib.sha256(np.asarray(getattr(model,k)).tobytes()).hexdigest()==v for k,v in signature.items()),"Physical model arrays mutated")
            save(out/"physics-array-signatures.json",signature)
            require(not attempts and time.monotonic()-start < 600,"Guard or wall budget failed")
            report["status"]="STATIC_INTERFACE_REVIEW_REQUIRED"
    except Exception as exc:
        report["error"]=str(exc)
        (out/"error.txt").write_text(traceback.format_exc())
    report["forbidden_integration_call_attempts"]=attempts
    report["wall_seconds"]=time.monotonic()-start
    save(out/"interface-review.json",report)
    print("Stage11InterfaceResult="+report["status"],flush=True)
    return 0 if report["status"]=="STATIC_INTERFACE_REVIEW_REQUIRED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
