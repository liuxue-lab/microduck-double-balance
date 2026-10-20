"""Zero-integration Scene and tensor audit of the independent head prototype."""
from __future__ import annotations
import argparse
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stage11_frozen_reference", ROOT / "scripts/audit_head_contact_reference.py")
ref = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ref)
v2, require, save = ref.v2, ref.require, ref.save
from mjlab_microduck.robot.head_reference import (
    ORIGIN_IN_JAW, NORMAL_IN_JAW, REFERENCE_SITE, REFERENCE_VERSION,
    reference_rotation_in_jaw,
)

PRIOR = Path("/home/lx/microduck-double-balance/artifacts/double-balance-stage11/head-reference/20261003T195947Z")
PRIOR_ZIP_SHA = "73417e634580a91c6100f8b751a74b7890c58ff3764eb8e47b26ed396f374c0a"
MDP_PATH = "src/mjlab_microduck/tasks/mdp.py"
# Explicit fields unaffected by a massless site; include full inertia and joints.
PHYSICS_FIELDS = (
    "body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos", "body_iquat",
    "body_parentid", "jnt_type", "jnt_bodyid", "jnt_pos", "jnt_axis", "jnt_range",
    "jnt_limited", "jnt_stiffness", "dof_damping", "dof_frictionloss", "dof_armature",
    "geom_type", "geom_bodyid", "geom_dataid", "geom_pos", "geom_quat", "geom_size",
    "geom_contype", "geom_conaffinity", "geom_friction", "geom_solref", "geom_solimp",
    "geom_condim", "geom_priority", "geom_margin", "geom_gap", "mesh_vert", "mesh_face",
    "actuator_trnid", "actuator_gear", "actuator_gainprm", "actuator_biasprm",
    "actuator_dynprm", "actuator_ctrlrange", "actuator_forcerange", "eq_data",
)


def preserved_mdp_prefix(old, new):
    require(new.startswith(old), "Old mdp.py bytes changed instead of append-only extension")
    old_tree, new_tree = ast.parse(old), ast.parse(new)
    require(len(new_tree.body) > len(old_tree.body), "Missing prototype extension")
    require(all(ast.dump(a) == ast.dump(b) for a, b in zip(old_tree.body, new_tree.body)), "Old AST changed")
    appended = new_tree.body[len(old_tree.body):]
    require(all(isinstance(n, ast.FunctionDef) and n.name.startswith(("stage11_head_", "_stage11_head_", "reset_stage11_head_")) for n in appended), "Unexpected top-level execution in extension")
    require(len({n.name for n in new_tree.body if isinstance(n, ast.FunctionDef)}) == sum(isinstance(n, ast.FunctionDef) for n in new_tree.body), "Duplicate function override")
    return {"old_bytes": len(old), "new_bytes": len(new), "added_functions": [n.name for n in appended], "old_prefix_identical": True}


def geometry_audit(old, physical, prototype, out):
    """Three independently compiled models, seven configurations, no dynamics."""
    structure = v2.audit_full_scene(old, prototype)
    save(out / "structure-inertia.json", structure)
    require(structure["status"] == "PASS", "Old tray to head structure check failed")
    checks = {name: np.array_equal(getattr(physical, name), getattr(prototype, name)) for name in PHYSICS_FIELDS}
    checks["options_identical"] = all(
        np.array_equal(value, v2.options_dict(prototype)[name])
        for name, value in v2.options_dict(physical).items()
    )
    checks["one_site_only"] = prototype.nsite == physical.nsite + 1
    checks["dimensions_unchanged"] = all(getattr(physical, name) == getattr(prototype, name) for name in ("nq", "nv", "nu", "na", "neq", "ngeom", "nbody", "njnt"))
    save(out / "reference-site-physics-diff.json", checks)
    require(all(checks.values()), "Reference site altered physical parameters")
    signature = ref.floor.model_signature(prototype)
    home = v2.initial_data(prototype)
    sid = next(i for i in range(prototype.nsite) if prototype.site(i).name.split("/")[-1] == REFERENCE_SITE)
    jaw, head = v2.bid(prototype, "jaw_soft"), v2.gid(prototype, v2.HEAD_GEOM_NAME)
    rows = []
    for pose in ref.POSES:
        data = ref.pose_data(prototype, home.qpos, pose)
        H, origin = data.xmat[jaw].reshape(3, 3), data.xpos[jaw]
        o, R = data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()
        require(np.max(np.abs(o - (origin + H @ ORIGIN_IN_JAW))) < 1e-12, "Wrong material origin")
        require(np.max(np.abs(R - H @ reference_rotation_in_jaw())) < 1e-12, "Wrong reference axes")
        center = o + R @ np.array([0., 0., .0205])
        verts, faces = v2.mesh_triangles(prototype, data, head)
        geometry = ref.place_and_measure(prototype, data, center, verts[faces])
        require(geometry["v2_pass"], "Prototype reset failed existing surface gate")
        require(abs(geometry["original_sphere_gap_m"] - .0005) < 1e-7, "Incorrect true surface clearance")
        rows.append({"pose": pose[0], "origin": o, "rotation": R, "center": center,
                     "geometry": geometry, "qpos": data.qpos.copy(), "time_s": data.time,
                     "tangent_height_error_m": float((R.T @ (center-o))[2] - .02)})
        require(data.time == 0 and not np.any(data.qvel), "Static configuration advanced")
    require(ref.floor.model_signature(prototype) == signature, "Model changed during audit")
    save(out / "prototype-static-configurations.json", rows)
    return {"poses": len(rows), "reset_geometry_pass": all(r["geometry"]["v2_pass"] for r in rows),
            "reference_site_only": all(checks.values()), "time_advanced": False}, rows


def tensor_audit(mdp, rows, out):
    """Actual Torch implementations versus independent NumPy calculations."""
    import torch
    T = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float64)
    results = []
    for row in rows:
        R, o, p = map(np.asarray, (row["rotation"], row["origin"], row["center"]))
        vo, w, spin = np.array([.11, -.07, .03]), np.array([.2, -.3, 1.]), np.array([.5, .1, -.2])
        extra = np.array([.03, -.02, .01])
        vb = vo + np.cross(w, p-o) + R @ extra
        values = mdp.stage11_head_material_values(*[T([v]) for v in (p, vb, w+R@spin, o, vo, w, R)])
        expected = (R.T@(p-o), extra, spin)
        residual = max(float(np.max(np.abs(v.numpy()[0]-e))) for v, e in zip(values, expected))
        require(residual < 1e-12, "Tensor rotating-frame or spin sign error")
        actor = mdp.stage11_head_features_from_values(*values)
        critic = mdp.stage11_head_features_from_values(*values, privileged=True)
        want = np.r_[expected[0] * [20, 25, 50] - [0, 0, 1], extra*2, spin*.2]
        require(actor.shape == (1, 6) and critic.shape == (1, 9), "Feature dimensions changed")
        np.testing.assert_allclose(critic.numpy()[0], want, atol=1e-12, rtol=0)
        np.testing.assert_allclose(actor.numpy(), critic.numpy()[:, :6], atol=0, rtol=0)
        quat=np.empty(4);mujoco.mju_mat2Quat(quat,R.reshape(9))
        robot=SimpleNamespace(find_sites=lambda name: ([0],[REFERENCE_SITE]), data=SimpleNamespace(
            site_pose_w=T([[np.r_[o,quat]]]), site_lin_vel_w=T([[vo]]), site_ang_vel_w=T([[w]])))
        ball=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=T([p]),root_link_lin_vel_w=T([vb]),root_link_ang_vel_w=T([w+R@spin])))
        env=SimpleNamespace(scene={"robot":robot,"top_ball":ball})
        torch.testing.assert_close(mdp.stage11_head_actor_state(env),actor,rtol=0,atol=1e-12)
        torch.testing.assert_close(mdp.stage11_head_critic_state(env),critic,rtol=0,atol=1e-12)
        carried = mdp.stage11_head_material_values(*[T([v]) for v in (p, vo+np.cross(w,p-o), w, o, vo, w, R)])
        require(float(torch.linalg.vector_norm(carried[1])) < 1e-12, "Carried sphere derivative must vanish")
        results.append({"pose": row["pose"], "max_numpy_torch_residual": residual, "actor": actor.tolist(), "critic": critic.tolist()})
    # Distinguish no-slip rolling from a stationary center in the material frame.
    value = mdp.stage11_head_material_values(*[T([v]) for v in ([0,0,.02],[.05,0,0],[0,2.5,0],[0,0,0],[0,0,0],[0,0,0],np.eye(3))])
    require(abs(float(value[1][0,0])-.05)<1e-12, "Center derivative mislabeled as slip")
    # Exact Stage-04 numeric boundaries stay available unchanged; no new success function.
    zeros, ones = torch.zeros(3, dtype=torch.float64), torch.ones(3, dtype=torch.float64)
    stable = mdp.double_balance_stable_from_values(T([[0,0,.02]]*3), T([[0,0,0]]*3), zeros, ones*.245, zeros, T([.149999,.15,.150001]))
    require(stable.tolist() == [True, True, False], "Frozen lower speed boundary changed")
    # The new reset reads after forward, clears 6-D velocity, and only writes selected ids.
    target_ids = torch.tensor([1], dtype=torch.long)
    calls, writes = [], {}
    robot = SimpleNamespace(data=SimpleNamespace(site_pose_w=torch.zeros(2,1,7)))
    robot.find_sites = lambda name: ([0], [REFERENCE_SITE])
    def forward():
        calls.append("forward")
        robot.data.site_pose_w[1,0] = torch.tensor([1.,2.,3.,1.,0.,0.,0.])
    ball = SimpleNamespace(
        write_root_link_pose_to_sim=lambda v, env_ids: writes.update(pose=v.clone(), pose_ids=env_ids.clone()),
        write_root_link_velocity_to_sim=lambda v, env_ids: writes.update(velocity=v.clone(), velocity_ids=env_ids.clone()),
    )
    env = SimpleNamespace(scene={"robot":robot,"top_ball":ball}, sim=SimpleNamespace(forward=forward), device="cpu")
    with patch.object(mdp, "reset_basketball", lambda *a, **kw: calls.append("lower_reset")):
        mdp.reset_stage11_head_reference(env, target_ids)
        require(calls == ["lower_reset", "forward"], "Reset read stale head state")
        torch.testing.assert_close(writes["pose"], torch.tensor([[1.,2.,3.0205,1.,0.,0.,0.]]))
        require(writes["velocity"].shape==(1,6) and not torch.any(writes["velocity"]), "Residual ball velocity")
        require(torch.equal(writes["pose_ids"], target_ids) and torch.equal(writes["velocity_ids"], target_ids), "Wrong reset subset")
        try: mdp.reset_stage11_head_reference(env, target_ids, joint_noise=.01)
        except ValueError: pass
        else: raise RuntimeError("Unreviewed noisy curved reset accepted")
    save(out / "tensor-and-reset-review.json", {"torch_version": torch.__version__, "rows": results,
        "center_derivative_not_contact_slip": True, "legacy_speed_boundary_preserved": True,
        "reset_forward_order_and_subset": True, "reset_scope": "Mock entity write/read order; not full ManagerBasedRlEnv reset execution"})
    return {"rows":len(results), "max_residual": max(r["max_numpy_torch_residual"] for r in results), "reset_wrapper_checked":True}


def config_audit(old, new, mdp):
    require(old.episode_length_s == new.episode_length_s == 10., "Episode length changed")
    require(old.decimation == new.decimation == 10 and new.sim.mujoco.timestep == .002, "Timing changed")
    require(list(old.observations["actor"].terms) == list(new.observations["actor"].terms), "Actor order changed")
    require(list(old.observations["critic"].terms) == list(new.observations["critic"].terms), "Critic order changed")
    require(new.observations["actor"].terms["body_command"].func is mdp.stage11_head_actor_state, "Actor not migrated")
    require(new.observations["critic"].terms["top_ball_state"].func is mdp.stage11_head_critic_state, "Critic not migrated")
    require(new.actions["ball_hold"].levels == (0.,) and not new.curriculum, "Assistance/curriculum enabled")
    require(not any("head_pose" in n or "head_forward" in n or "stage09_head" in n for n in new.rewards), "Head correction reintroduced")
    for name in ("double_balance", "top_ball_speed"):
        require(old.rewards[name].weight == new.rewards[name].weight, "Reward tuning in static prototype")
    require(new.rewards["head_surface_normal_tilt"].weight == old.rewards["tray_tilt"].weight, "Normal weight changed")
    require("double_balance_success" not in new.metrics and "double_balance_stable_fraction" not in new.metrics, "Premature success metric")
    for term in (new.events["stage11_release_block"], new.terminations["top_ball_lost"]):
        require(term.func is mdp.stage11_head_release_block, "Missing runtime release block")
        try: term.func()
        except RuntimeError as exc: require("STAGE11_STATIC_PROTOTYPE_ONLY" in str(exc), "Wrong runtime block")
        else: raise RuntimeError("Prototype inadvertently runnable")
    return {"actor_layout":"unchanged 61D slots, semantics incompatible", "critic_layout":"unchanged 85D slots, semantics incompatible",
            "full_manager_observation_execution":False, "episode_seconds":10, "stable_duration_seconds":5,
            "lower_ball_speed_limit":.15, "registered":False, "runtime_release_block":True,
            "formal_acceptance_implemented":False, "old_timing_functions_byte_identical":True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-prototype-only", action="store_true", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True, exist_ok=False)
    report={"status":"STATIC_PROTOTYPE_REVIEW_REQUIRED", "stage11_status":"MATERIAL_BLOCKED_REVIEW_REQUIRED",
            "new_integration_steps":0,"new_ppo_updates":0,"full_robot_steps":0,"cloud_hours":0,
            "stage11_complete":False,"canonical_total_completed_steps":7895,"reference_version":REFERENCE_VERSION,
            "formal_task_migration":False,"policy_success":None}
    attempts=[];start=time.monotonic()
    try:
        with ref.zero_integration_guard() as attempts:
            required=("run-summary.json","reference/reference-review.json","reference/candidate-references.json")
            provenance=ref.friction.pinned_files(PRIOR, PRIOR_ZIP_SHA, required)
            candidates=json.loads((PRIOR/required[2]).read_text())["inherited_seven_locations"][0]
            np.testing.assert_allclose(candidates["witness_in_jaw"], ORIGIN_IN_JAW, atol=1e-15, rtol=0)
            np.testing.assert_allclose(candidates["normal_in_jaw"], NORMAL_IN_JAW, atol=1e-15, rtol=0)
            prior_run=json.loads((PRIOR/required[0]).read_text())
            require(prior_run["execution_invariants_unchanged"] and prior_run["canonical_total_known_completed_steps"]==7895, "Prior provenance changed")
            old_bytes=subprocess.check_output(["git","show",v2.BASELINE+":"+MDP_PATH],cwd=ROOT)
            prefix=preserved_mdp_prefix(old_bytes, (ROOT/MDP_PATH).read_bytes())
            save(out/"source-and-input-provenance.json", {"archive_sha256":PRIOR_ZIP_SHA,"files":provenance,"mdp_prefix":prefix})
            from mjlab.scene import Scene
            from mjlab_microduck.robot.head_contact import physics_scene_pair
            from mjlab_microduck.tasks import mdp
            from mjlab_microduck.tasks.microduck_head_contact_env_cfg import make_microduck_head_contact_env_cfg
            # Explicit selection excludes the existing manager test that integrates.
            import pytest
            old_tests=("actor_stays_61d_and_critic_privilege_is_separate",
                       "reward_signs_and_functions_follow_repository_convention",
                       "top_score_is_multiplicative_and_not_farmable_from_bad_states",
                       "top_ball_lost_uses_recoverable_geometry_envelope",
                       "success_requires_both_layers_and_low_motion",
                       "success_timer_is_continuous_not_a_one_step_jackpot",
                       "official_play_config_is_deterministic_and_unassisted")
            code=pytest.main(["-q", *[str(ROOT/"tests/test_double_balance_task_definition.py")+"::test_"+n for n in old_tests]])
            require(code==0 and not attempts,"Frozen task regression or zero-step guard failed")
            report["old_task_regression"]={"selected":list(old_tests),"count":7,"manager_step_test_excluded":True}
            old_cfg, old_scene, physical_scene=physics_scene_pair()
            cfg=make_microduck_head_contact_env_cfg(static_audit_only=True)
            report["config"]=config_audit(old_cfg,cfg,mdp)
            models=[Scene(s,device="cpu").compile() for s in (old_scene,physical_scene,cfg.scene)]
            for model in models: cfg.sim.mujoco.apply(model)
            report["geometry"],rows=geometry_audit(*models,out)
            report["tensor"]=tensor_audit(mdp,rows,out)
            require(time.monotonic()-start<300, "Static audit wall budget exceeded")
    except Exception as exc:
        report.update(status="STATIC_PROTOTYPE_EXECUTION_BLOCKED",error=str(exc))
        (out/"error.txt").write_text(traceback.format_exc())
    report["forbidden_integration_call_attempts"]=attempts
    report["wall_seconds"]=time.monotonic()-start
    save(out/"prototype-review.json",report)
    print("Stage11PrototypeResult="+report["status"],flush=True)
    return 0 if report["status"]=="STATIC_PROTOTYPE_REVIEW_REQUIRED" else 2


if __name__=="__main__": raise SystemExit(main())
