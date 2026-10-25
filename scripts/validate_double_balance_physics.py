"""Reproducible Stage-03 double-balance physics acceptance.

The script compares classic CPU MuJoCo at 5/2/1 ms while preserving a 20 ms
control period, runs the selected 2 ms scene through MJWarp, exercises the real
manager reset, and finishes with a fixed-seed original-basketball smoke test.

Examples:

  uv run python scripts/validate_double_balance_physics.py
  uv run python scripts/validate_double_balance_physics.py --warp-device cuda:0
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

import mujoco
import numpy as np
import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.scene import Scene
from mjlab.sim import Simulation
from mjlab_microduck.tasks.microduck_basketball_env_cfg import (
    make_microduck_basketball_env_cfg,
)
from mjlab_microduck.tasks.microduck_double_balance_env_cfg import (
    CONTROL_DT,
    TOP_BALL_GEOM_NAME,
    TOP_BALL_RADIUS,
    TOP_BALL_RESET_CLEARANCE,
    TRAY_GEOM_NAME,
    TRAY_SITE_NAME,
    make_microduck_double_balance_env_cfg,
)

AUDITED_TIMESTEPS = (0.005, 0.002, 0.001)
SELECTED_TIMESTEP = 0.002
ROLLOUT_SECONDS = 1.0
SEED = 20260920


@dataclass(frozen=True)
class PhysicsResult:
    backend: str
    device: str
    physics_dt_ms: float
    control_dt_ms: float
    contact_sequence: tuple[str, ...]
    first_floor_contact_s: float | None
    max_penetration_mm: float
    has_nan: bool


def _build(timestep: float):
    cfg = make_microduck_double_balance_env_cfg(
        play=True,
        blind=True,
        history=1,
        physics_dt=timestep,
    )
    cfg.scene.num_envs = 1
    model = Scene(cfg.scene, device="cpu").compile()
    model.opt.timestep = timestep
    return cfg, model


def _write_freejoint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    position,
    velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
) -> None:
    joint_id = model.joint(joint_name).id
    qpos_adr = model.jnt_qposadr[joint_id]
    dof_adr = model.jnt_dofadr[joint_id]
    data.qpos[qpos_adr : qpos_adr + 7] = (*position, 1.0, 0.0, 0.0, 0.0)
    data.qvel[dof_adr : dof_adr + 6] = velocity


def _initial_data(model: mujoco.MjModel, *, moving: bool) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    _write_freejoint(model, data, "ball/floating_base_joint", (0.0, 0.0, 0.121))
    _write_freejoint(model, data, "robot/trunk_base_freejoint", (0.0, 0.0, 0.368))
    mujoco.mj_forward(model, data)
    tray_site = model.site(f"robot/{TRAY_SITE_NAME}").id
    tray_rotation = data.site_xmat[tray_site].reshape(3, 3)
    offset = 0.020 if moving else 0.0
    top_position = data.site_xpos[tray_site] + tray_rotation @ np.array(
        [offset, 0.0, TOP_BALL_RADIUS + TOP_BALL_RESET_CLEARANCE]
    )
    tangential_speed = 0.18 if moving else 0.0
    _write_freejoint(
        model,
        data,
        "top_ball/top_ball_freejoint",
        top_position,
        velocity=(tangential_speed, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    mujoco.mj_forward(model, data)
    return data


def _update_contacts(
    model: mujoco.MjModel,
    top_geom: int,
    contacts,
    seen: set[str],
    sequence: list[str],
    min_distance: float,
    first_floor_contact: float | None,
    time_s: float,
) -> tuple[float, float | None]:
    floor_geom = model.geom("terrain").id
    for pair, distance in contacts:
        if top_geom not in pair:
            continue
        other = int(pair[1] if pair[0] == top_geom else pair[0])
        name = model.geom(other).name
        if name not in seen:
            seen.add(name)
            sequence.append(name)
        min_distance = min(min_distance, float(distance))
        if other == floor_geom and first_floor_contact is None:
            first_floor_contact = time_s
    return min_distance, first_floor_contact


def run_mujoco(timestep: float) -> PhysicsResult:
    _, model = _build(timestep)
    data = _initial_data(model, moving=True)
    top_geom = model.geom(f"top_ball/{TOP_BALL_GEOM_NAME}").id
    seen: set[str] = set()
    sequence: list[str] = []
    min_distance = 0.0
    first_floor_contact = None
    has_nan = False
    for step in range(round(ROLLOUT_SECONDS / timestep)):
        mujoco.mj_step(model, data)
        contacts = (
            ((contact.geom1, contact.geom2), contact.dist)
            for contact in data.contact
        )
        min_distance, first_floor_contact = _update_contacts(
            model,
            top_geom,
            contacts,
            seen,
            sequence,
            min_distance,
            first_floor_contact,
            (step + 1) * timestep,
        )
        has_nan |= not (
            np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        )
    return PhysicsResult(
        backend="MuJoCo",
        device="cpu",
        physics_dt_ms=1000.0 * timestep,
        control_dt_ms=1000.0 * CONTROL_DT,
        contact_sequence=tuple(sequence),
        first_floor_contact_s=first_floor_contact,
        max_penetration_mm=-1000.0 * min_distance,
        has_nan=has_nan,
    )


def run_mjwarp(device: str) -> PhysicsResult:
    cfg, model = _build(SELECTED_TIMESTEP)
    initial = _initial_data(model, moving=True)
    sim = Simulation(num_envs=1, cfg=cfg.sim, model=model, device=device)
    sim.data.qpos[0] = torch.from_numpy(initial.qpos.copy()).to(device)
    sim.data.qvel[0] = torch.from_numpy(initial.qvel.copy()).to(device)
    sim.data.ctrl[0] = torch.from_numpy(initial.ctrl.copy()).to(device)
    sim.forward()

    top_geom = model.geom(f"top_ball/{TOP_BALL_GEOM_NAME}").id
    seen: set[str] = set()
    sequence: list[str] = []
    min_distance = 0.0
    first_floor_contact = None
    has_nan = False
    for step in range(round(ROLLOUT_SECONDS / SELECTED_TIMESTEP)):
        sim.step()
        num_contacts = int(sim.data.nacon[0].item())
        geom_pairs = sim.data.contact.geom[:num_contacts].cpu().numpy()
        distances = sim.data.contact.dist[:num_contacts].cpu().numpy()
        min_distance, first_floor_contact = _update_contacts(
            model,
            top_geom,
            zip(geom_pairs, distances, strict=False),
            seen,
            sequence,
            min_distance,
            first_floor_contact,
            (step + 1) * SELECTED_TIMESTEP,
        )
        has_nan |= not (
            torch.isfinite(sim.data.qpos).all().item()
            and torch.isfinite(sim.data.qvel).all().item()
        )
    return PhysicsResult(
        backend="MJWarp",
        device=device,
        physics_dt_ms=1000.0 * SELECTED_TIMESTEP,
        control_dt_ms=1000.0 * CONTROL_DT,
        contact_sequence=tuple(sequence),
        first_floor_contact_s=first_floor_contact,
        max_penetration_mm=-1000.0 * min_distance,
        has_nan=has_nan,
    )


def centered_reset_probe() -> dict[str, float | bool]:
    _, model = _build(SELECTED_TIMESTEP)
    data = _initial_data(model, moving=False)
    top_joint = model.joint("top_ball/top_ball_freejoint").id
    top_dof = model.jnt_dofadr[top_joint]
    top_body = model.body("top_ball/top_ball").id
    top_geom = model.geom(f"top_ball/{TOP_BALL_GEOM_NAME}").id
    initial_speed = float(np.linalg.norm(data.qvel[top_dof : top_dof + 6]))
    initial_penetration = 0.0
    for contact in data.contact:
        if top_geom in (contact.geom1, contact.geom2):
            initial_penetration = max(initial_penetration, -float(contact.dist))
    initial_z = float(data.xpos[top_body, 2])
    max_z = initial_z
    finite = True
    for _ in range(5):
        mujoco.mj_step(model, data)
        max_z = max(max_z, float(data.xpos[top_body, 2]))
        finite &= np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
    return {
        "initial_speed_m_s": initial_speed,
        "initial_penetration_mm": 1000.0 * initial_penetration,
        "first_5_steps_upward_excursion_mm": 1000.0 * (max_z - initial_z),
        "finite": bool(finite),
    }


def manager_reset_probe(device: str) -> dict[str, float | int | bool]:
    cfg = make_microduck_double_balance_env_cfg(
        play=True, blind=True, history=1, physics_dt=SELECTED_TIMESTEP
    )
    cfg.scene.num_envs = 1
    cfg.seed = SEED
    # Keep reset/startup terms, but remove interval pushes from this probe.
    cfg.events = {
        name: term
        for name, term in cfg.events.items()
        if term.mode in {"reset", "startup"}
    }
    cfg.events["reset_double_balance"].params.update(
        xy_noise=0.0,
        yaw_range=(0.0, 0.0),
        tilt_noise_deg=0.0,
        joint_noise=0.0,
        top_ball_xy_noise=0.0,
    )
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    first_obs, _ = env.reset(seed=SEED)
    robot = env.scene["robot"]
    top_ball = env.scene["top_ball"]
    tray_site = robot.find_sites(TRAY_SITE_NAME)[0][0]
    first_pose = top_ball.data.root_link_pose_w.clone()
    distance = torch.linalg.norm(
        top_ball.data.root_link_pos_w[0] - robot.data.site_pos_w[0, tray_site]
    )
    residual_speed = torch.linalg.norm(top_ball.data.root_link_vel_w[0])
    second_obs, _ = env.reset(seed=SEED)
    repeat_error = torch.max(
        torch.abs(top_ball.data.root_link_pose_w - first_pose)
    )
    return {
        "actor_dim": int(first_obs["actor"].shape[-1]),
        "second_actor_dim": int(second_obs["actor"].shape[-1]),
        "tray_to_ball_center_m": float(distance.item()),
        "residual_speed": float(residual_speed.item()),
        "repeat_pose_max_abs": float(repeat_error.item()),
        "finite": bool(
            torch.isfinite(top_ball.data.root_link_pose_w).all().item()
            and torch.isfinite(top_ball.data.root_link_vel_w).all().item()
        ),
    }


def original_basketball_smoke(device: str) -> dict[str, int | bool]:
    cfg = make_microduck_basketball_env_cfg(play=True, blind=True, history=1)
    cfg.scene.num_envs = 1
    cfg.seed = SEED
    env = ManagerBasedRlEnv(cfg=cfg, device=device)
    obs, _ = env.reset(seed=SEED)
    finite = torch.isfinite(obs["actor"]).all().item()
    for _ in range(3):
        obs, reward, terminated, truncated, _ = env.step(
            torch.zeros(1, 14, device=device)
        )
        finite &= (
            torch.isfinite(obs["actor"]).all().item()
            and torch.isfinite(reward).all().item()
            and torch.isfinite(terminated).all().item()
            and torch.isfinite(truncated).all().item()
        )
    return {
        "seed": SEED,
        "steps": 3,
        "actor_dim": int(obs["actor"].shape[-1]),
        "finite": bool(finite),
    }


def _assert_acceptance(
    cpu_results: list[PhysicsResult],
    warp_result: PhysicsResult,
    centered: dict,
    reset: dict,
    original: dict,
) -> None:
    expected_sequence = (f"robot/{TRAY_GEOM_NAME}", "terrain")
    for result in [*cpu_results, warp_result]:
        assert result.contact_sequence == expected_sequence, result
        assert result.first_floor_contact_s is not None, result
        assert 0.25 < result.first_floor_contact_s < 0.60, result
        assert result.max_penetration_mm < 10.0, result
        assert not result.has_nan, result
    selected_cpu = next(r for r in cpu_results if r.physics_dt_ms == 2.0)
    assert selected_cpu.max_penetration_mm < 1.0, selected_cpu
    assert warp_result.max_penetration_mm < 2.0, warp_result
    assert abs(
        selected_cpu.first_floor_contact_s - warp_result.first_floor_contact_s
    ) < 0.08
    assert centered["initial_speed_m_s"] == 0.0
    assert centered["initial_penetration_mm"] == 0.0
    assert centered["first_5_steps_upward_excursion_mm"] < 0.1
    assert centered["finite"]
    assert reset["actor_dim"] == reset["second_actor_dim"] == 61
    assert abs(
        reset["tray_to_ball_center_m"]
        - (TOP_BALL_RADIUS + TOP_BALL_RESET_CLEARANCE)
    ) < 1e-6
    assert reset["residual_speed"] == 0.0
    assert reset["repeat_pose_max_abs"] < 1e-7
    assert reset["finite"]
    assert original == {"seed": SEED, "steps": 3, "actor_dim": 61, "finite": True}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warp-device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="MJWarp device (default: cuda:0 when available, otherwise cpu)",
    )
    args = parser.parse_args()

    cpu_results = [run_mujoco(dt) for dt in AUDITED_TIMESTEPS]
    centered = centered_reset_probe()
    warp_result = run_mjwarp(args.warp_device)
    reset = manager_reset_probe(args.warp_device)
    original = original_basketball_smoke(args.warp_device)
    _assert_acceptance(cpu_results, warp_result, centered, reset, original)

    report = {
        "cpu": [asdict(result) for result in cpu_results],
        "warp": asdict(warp_result),
        "centered_reset": centered,
        "manager_reset": reset,
        "original_basketball_smoke": original,
    }
    print("DoubleBalancePhysicsAcceptanceBegin")
    print(json.dumps(report, indent=2, sort_keys=True))
    print("DoubleBalancePhysicsAcceptance=PASS")
    print("DoubleBalancePhysicsAcceptanceEnd")


if __name__ == "__main__":
    main()
