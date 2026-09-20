"""Stage-03 configuration and MuJoCo physics acceptance tests."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from mjlab.scene import Scene
from mjlab.tasks.registry import list_tasks
from mjlab_microduck.robot.microduck_constants import get_standup_spec
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_basketball_env_cfg import (
    _ball_spec,
    make_microduck_basketball_env_cfg,
)
from mjlab_microduck.tasks.microduck_double_balance_env_cfg import (
    CONTROL_DT,
    FLOOR_CONTACT_BIT,
    ROBOT_CONTACT_BIT,
    TOP_BALL_CONTACT_BIT,
    TOP_BALL_GEOM_NAME,
    TOP_BALL_MASS,
    TOP_BALL_RADIUS,
    TOP_BALL_RESET_CLEARANCE,
    TRAY_BODY_NAME,
    TRAY_CONTACT_BIT,
    TRAY_FULL_SIZE,
    TRAY_GEOM_NAME,
    TRAY_INERTIA,
    TRAY_MASS,
    TRAY_SITE_NAME,
    _top_ball_spec,
    make_microduck_double_balance_env_cfg,
)


@pytest.fixture(scope="module")
def cfg():
    return make_microduck_double_balance_env_cfg(
        play=True, blind=True, history=1, physics_dt=0.002
    )


@pytest.fixture(scope="module")
def model(cfg):
    cfg.scene.num_envs = 1
    return Scene(cfg.scene, device="cpu").compile()


def _can_contact(model: mujoco.MjModel, first: int, second: int) -> bool:
    return bool(
        (int(model.geom_contype[first]) & int(model.geom_conaffinity[second]))
        or (int(model.geom_contype[second]) & int(model.geom_conaffinity[first]))
    )


def test_task_registered_and_control_period_fixed(cfg):
    assert "Mjlab-DoubleBalance-MicroDuck" in list_tasks()
    assert cfg.sim.mujoco.timestep == pytest.approx(0.002)
    assert cfg.decimation == 10
    assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(CONTROL_DT)
    for timestep, expected_decimation in ((0.005, 4), (0.002, 10), (0.001, 20)):
        candidate = make_microduck_double_balance_env_cfg(
            play=True, physics_dt=timestep
        )
        assert candidate.decimation == expected_decimation
        assert candidate.sim.mujoco.timestep * candidate.decimation == pytest.approx(
            CONTROL_DT
        )
    with pytest.raises(ValueError, match="must divide"):
        make_microduck_double_balance_env_cfg(play=True, physics_dt=0.003)


def test_original_basketball_cfg_and_ball_are_unchanged():
    before = make_microduck_basketball_env_cfg(play=True, blind=True, history=1)
    _ = make_microduck_double_balance_env_cfg(play=True, blind=True, history=1)
    after = make_microduck_basketball_env_cfg(play=True, blind=True, history=1)
    assert list(before.scene.entities) == list(after.scene.entities) == ["robot", "ball"]
    assert list(before.events) == list(after.events)
    assert before.sim.mujoco.timestep == after.sim.mujoco.timestep == 0.005
    assert before.decimation == after.decimation == 4
    original_ball = _ball_spec().compile()
    geom_id = original_ball.geom("ball_sphere").id
    assert original_ball.geom_contype[geom_id] == 1
    assert original_ball.geom_conaffinity[geom_id] == 1


def test_actor_observation_contract_is_not_extended(cfg):
    basketball = make_microduck_basketball_env_cfg(
        play=True, blind=True, history=1
    )
    original_terms = basketball.observations["actor"].terms
    derived_terms = cfg.observations["actor"].terms
    assert list(derived_terms) == list(original_terms)
    for name in original_terms:
        assert derived_terms[name].func is original_terms[name].func
        assert derived_terms[name].params == original_terms[name].params
        assert derived_terms[name].history_length == original_terms[name].history_length
    # The manager integration smoke separately resolves this unchanged layout
    # to 61; Stage 03 adds no top-ball term here.
    assert all("top_ball" not in name for name in derived_terms)


def test_nan_guard_covers_both_free_ball_entities(cfg):
    nan_guard = cfg.terminations["nan_state"]
    assert nan_guard.func is microduck_mdp.robot_state_is_nan
    assert nan_guard.params["extra_entity_names"] == ("ball", "top_ball")

    def finite_entity():
        return SimpleNamespace(
            data=SimpleNamespace(
                joint_pos=torch.zeros(2, 1),
                joint_vel=torch.zeros(2, 1),
                root_link_pos_w=torch.zeros(2, 3),
                root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2),
                root_link_lin_vel_w=torch.zeros(2, 3),
                root_link_ang_vel_w=torch.zeros(2, 3),
            )
        )

    class FakeScene(dict):
        sensors = {}

    scene = FakeScene(
        robot=finite_entity(), ball=finite_entity(), top_ball=finite_entity()
    )
    scene["top_ball"].data.root_link_lin_vel_w[1, 0] = float("inf")
    env = SimpleNamespace(scene=scene)
    bad = microduck_mdp.robot_state_is_nan(
        env, extra_entity_names=("ball", "top_ball")
    )
    torch.testing.assert_close(bad, torch.tensor([False, True]))


def test_tray_is_explicit_fixed_child_with_full_dimensions(model):
    tray_body = model.body(f"robot/{TRAY_BODY_NAME}").id
    jaw_body = model.body("robot/jaw_soft").id
    tray_geom = model.geom(f"robot/{TRAY_GEOM_NAME}").id
    assert model.body_parentid[tray_body] == jaw_body
    assert not np.any(model.jnt_bodyid == tray_body)
    np.testing.assert_allclose(model.body_mass[tray_body], TRAY_MASS, atol=1e-12)
    np.testing.assert_allclose(model.body_ipos[tray_body], 0.0, atol=1e-12)
    np.testing.assert_allclose(model.body_inertia[tray_body], TRAY_INERTIA, atol=1e-12)
    np.testing.assert_allclose(
        2.0 * model.geom_size[tray_geom], TRAY_FULL_SIZE, atol=1e-12
    )
    base_model = get_standup_spec().compile()
    base_root = base_model.body("trunk_base").id
    derived_root = model.body("robot/trunk_base").id
    assert model.body_subtreemass[derived_root] == pytest.approx(
        base_model.body_subtreemass[base_root] + TRAY_MASS
    )

    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    site_id = model.site(f"robot/{TRAY_SITE_NAME}").id
    # At HOME the reset frame is level, with local +z equal to world +z.
    np.testing.assert_allclose(
        data.site_xmat[site_id].reshape(3, 3), np.eye(3), atol=2e-7
    )


def test_top_ball_has_one_true_freejoint_and_no_joint_impedance(model):
    standalone = _top_ball_spec().compile()
    assert standalone.nq == 7 and standalone.nv == 6
    assert standalone.njnt == 1
    assert standalone.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    np.testing.assert_allclose(standalone.dof_damping, 0.0)
    np.testing.assert_allclose(standalone.dof_frictionloss, 0.0)
    np.testing.assert_allclose(standalone.dof_armature, 0.0)
    np.testing.assert_allclose(standalone.jnt_stiffness, 0.0)
    assert standalone.body_mass[1] == pytest.approx(TOP_BALL_MASS)

    joint_id = model.joint("top_ball/top_ball_freejoint").id
    assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    dof_ids = np.flatnonzero(model.dof_jntid == joint_id)
    assert len(dof_ids) == 6
    np.testing.assert_allclose(model.dof_damping[dof_ids], 0.0)
    np.testing.assert_allclose(model.dof_frictionloss[dof_ids], 0.0)
    np.testing.assert_allclose(model.dof_armature[dof_ids], 0.0)
    assert model.jnt_stiffness[joint_id] == 0.0


def test_contact_matrix_allows_only_tray_and_floor_to_support_top_ball(cfg, model):
    floor = model.geom("terrain").id
    tray = model.geom(f"robot/{TRAY_GEOM_NAME}").id
    lower_ball = model.geom("ball/ball_sphere").id
    top_ball = model.geom(f"top_ball/{TOP_BALL_GEOM_NAME}").id
    left_foot = model.geom("robot/left_foot_collision").id

    assert model.geom_contype[floor] == FLOOR_CONTACT_BIT
    assert model.geom_contype[tray] == TRAY_CONTACT_BIT
    assert model.geom_contype[top_ball] == TOP_BALL_CONTACT_BIT
    assert model.geom_contype[lower_ball] == ROBOT_CONTACT_BIT
    assert _can_contact(model, top_ball, tray)
    assert _can_contact(model, top_ball, floor)
    assert not _can_contact(model, top_ball, lower_ball)
    assert not _can_contact(model, top_ball, left_foot)
    assert _can_contact(model, lower_ball, floor)
    assert _can_contact(model, lower_ball, left_foot)

    for geom_id in range(model.ngeom):
        if geom_id not in {floor, tray, top_ball}:
            assert not _can_contact(model, top_ball, geom_id), model.geom(geom_id).name

    top_body = model.body("top_ball/top_ball").id
    assert model.body_parentid[top_body] == 0
    assert model.neq == 0
    assert all(
        getattr(action, "entity_name", None) != "top_ball"
        for action in cfg.actions.values()
    )
    assert all(
        getattr((term.params or {}).get("asset_cfg"), "name", None) != "top_ball"
        for term in cfg.events.values()
    )


def test_reset_places_top_ball_in_tray_frame_after_forward(monkeypatch):
    order: list[str] = []

    def fake_reset_basketball(*args, **kwargs):
        del args, kwargs
        order.append("lower_stack")

    monkeypatch.setattr(microduck_mdp, "reset_basketball", fake_reset_basketball)
    monkeypatch.setattr(
        torch,
        "rand",
        lambda *shape, **kwargs: torch.tensor([[1.0, 0.5]], **kwargs),
    )

    # 90 degrees about world z: a +x tray offset must become world +y.
    tray_pose = torch.tensor(
        [[[1.0, 2.0, 3.0, 2**-0.5, 0.0, 0.0, 2**-0.5]]]
    )

    class FakeRobot:
        data = SimpleNamespace(site_pose_w=tray_pose)

        @staticmethod
        def find_sites(name):
            assert name == TRAY_SITE_NAME
            return [0], [name]

    class FakeTopBall:
        def write_root_link_pose_to_sim(self, pose, env_ids):
            order.append("top_pose")
            self.pose = pose.clone()
            self.pose_ids = env_ids.clone()

        def write_root_link_velocity_to_sim(self, velocity, env_ids):
            order.append("top_velocity")
            self.velocity = velocity.clone()
            self.velocity_ids = env_ids.clone()

    class FakeSim:
        @staticmethod
        def forward():
            order.append("forward")

    top_ball = FakeTopBall()
    env = SimpleNamespace(
        device="cpu",
        scene={"robot": FakeRobot(), "top_ball": top_ball},
        sim=FakeSim(),
    )
    env_ids = torch.tensor([0])
    microduck_mdp.reset_double_balance(
        env,
        env_ids,
        top_ball_radius=TOP_BALL_RADIUS,
        top_ball_clearance=TOP_BALL_RESET_CLEARANCE,
        top_ball_xy_noise=0.01,
        tray_site_name=TRAY_SITE_NAME,
    )
    assert order == ["lower_stack", "forward", "top_pose", "top_velocity"]
    expected_position = torch.tensor(
        [1.0, 2.01, 3.0 + TOP_BALL_RADIUS + TOP_BALL_RESET_CLEARANCE]
    )
    torch.testing.assert_close(top_ball.pose[0, :3], expected_position)
    torch.testing.assert_close(top_ball.pose[0, 3:], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(top_ball.velocity, torch.zeros(1, 6))


@dataclass(frozen=True)
class RolloutResult:
    contact_sequence: tuple[str, ...]
    first_floor_contact_s: float | None
    max_penetration_m: float
    has_nan: bool


def _write_freejoint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    position: np.ndarray | tuple[float, float, float],
    velocity: tuple[float, float, float, float, float, float] = (0.0,) * 6,
) -> None:
    joint_id = model.joint(joint_name).id
    qpos_adr = model.jnt_qposadr[joint_id]
    dof_adr = model.jnt_dofadr[joint_id]
    data.qpos[qpos_adr : qpos_adr + 7] = (*position, 1.0, 0.0, 0.0, 0.0)
    data.qvel[dof_adr : dof_adr + 6] = velocity


def _run_natural_drop(timestep: float) -> RolloutResult:
    candidate = make_microduck_double_balance_env_cfg(
        play=True, physics_dt=timestep
    )
    candidate.scene.num_envs = 1
    model = Scene(candidate.scene, device="cpu").compile()
    model.opt.timestep = timestep
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    _write_freejoint(model, data, "ball/floating_base_joint", (0.0, 0.0, 0.121))
    _write_freejoint(model, data, "robot/trunk_base_freejoint", (0.0, 0.0, 0.368))
    mujoco.mj_forward(model, data)

    tray_site = model.site(f"robot/{TRAY_SITE_NAME}").id
    tray_rotation = data.site_xmat[tray_site].reshape(3, 3)
    top_position = data.site_xpos[tray_site] + tray_rotation @ np.array(
        [0.020, 0.0, TOP_BALL_RADIUS + TOP_BALL_RESET_CLEARANCE]
    )
    _write_freejoint(
        model,
        data,
        "top_ball/top_ball_freejoint",
        top_position,
        velocity=(0.18, 0.0, 0.0, 0.0, 0.0, 0.0),
    )
    mujoco.mj_forward(model, data)

    top_geom = model.geom(f"top_ball/{TOP_BALL_GEOM_NAME}").id
    floor_geom = model.geom("terrain").id
    seen: set[str] = set()
    sequence: list[str] = []
    first_floor_contact = None
    min_distance = 0.0
    has_nan = False
    for step in range(round(1.0 / timestep)):
        mujoco.mj_step(model, data)
        has_nan |= not (
            np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
        )
        for contact in data.contact:
            if top_geom not in (contact.geom1, contact.geom2):
                continue
            other = contact.geom2 if contact.geom1 == top_geom else contact.geom1
            name = model.geom(other).name
            if name not in seen:
                seen.add(name)
                sequence.append(name)
            min_distance = min(min_distance, float(contact.dist))
            if other == floor_geom and first_floor_contact is None:
                first_floor_contact = (step + 1) * timestep
    return RolloutResult(
        contact_sequence=tuple(sequence),
        first_floor_contact_s=first_floor_contact,
        max_penetration_m=-min_distance,
        has_nan=has_nan,
    )


def test_offset_and_tangential_velocity_naturally_roll_off_at_all_audited_steps():
    results = {dt: _run_natural_drop(dt) for dt in (0.005, 0.002, 0.001)}
    tray_name = f"robot/{TRAY_GEOM_NAME}"
    for dt, result in results.items():
        assert result.contact_sequence == (tray_name, "terrain"), (dt, result)
        assert result.first_floor_contact_s is not None
        assert 0.25 < result.first_floor_contact_s < 0.60
        assert not result.has_nan
        assert result.max_penetration_m < 0.010
    drop_times = [result.first_floor_contact_s for result in results.values()]
    assert max(drop_times) - min(drop_times) < 0.080
    # The selected 2 ms production setting stays below 1 mm in this impact.
    assert results[0.002].max_penetration_m < 0.001
