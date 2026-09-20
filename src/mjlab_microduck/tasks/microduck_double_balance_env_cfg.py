"""Stage-03 physics scene for Microduck double balance.

This task deliberately reuses the released basketball task's 61-D actor
contract and reward stack.  Stage 03 only adds the physical apparatus:

* a rigid, rimless tray fixed to the final head link (``jaw_soft``),
* a second, genuinely free ball, and
* collision masks that let the top ball touch only the tray and the floor.

No top-ball observation, reward, termination, constraint, or external wrench
is introduced here.  Those policy-facing choices belong to Stage 04.
"""
from __future__ import annotations

import math
import os
from copy import deepcopy
from functools import partial
from typing import Callable

import mujoco
from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg

from mjlab_microduck.robot.basketball import basketball_robot_cfg
from mjlab_microduck.robot.microduck_constants import get_standup_spec
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_basketball_env_cfg import (
    MicroduckBasketballRlCfg,
    _ball_spec,
    make_microduck_basketball_env_cfg,
)

# Collision categories are local to the derived double-balance scene.  The
# shared robot XML and the original basketball entity keep their original
# masks.  A pair is enabled when either geom's contype intersects the other's
# conaffinity.
FLOOR_CONTACT_BIT = 1 << 0
ROBOT_CONTACT_BIT = 1 << 2
ROBOT_SELF_CONTACT_BIT = 1 << 3
TRAY_CONTACT_BIT = 1 << 4
TOP_BALL_CONTACT_BIT = 1 << 5

# Full physical dimensions.  MuJoCo's box ``size`` below receives half of each
# value.  The 100 x 80 mm footprint covers the useful central region of the
# measured 122.7 x 91.8 mm head-shell AABB while retaining exposed roll-off
# edges.  The 18 g tray is representative of a thin printed/composite plate.
TRAY_FULL_SIZE = (0.100, 0.080, 0.002)
TRAY_MASS = 0.018
TRAY_INERTIA = (
    TRAY_MASS / 12.0 * (TRAY_FULL_SIZE[1] ** 2 + TRAY_FULL_SIZE[2] ** 2),
    TRAY_MASS / 12.0 * (TRAY_FULL_SIZE[0] ** 2 + TRAY_FULL_SIZE[2] ** 2),
    TRAY_MASS / 12.0 * (TRAY_FULL_SIZE[0] ** 2 + TRAY_FULL_SIZE[1] ** 2),
)
TRAY_FRICTION = (0.6, 0.005, 0.0001)  # sliding, torsional, rolling
TRAY_BODY_NAME = "double_balance_tray"
TRAY_GEOM_NAME = "double_balance_tray_surface"
TRAY_SITE_NAME = "double_balance_tray_frame"

# Audited against robot_allcollisions.xml at HOME.  jaw_soft's HOME transform
# relative to the root is p=(-0.009003, 0, 0.112599),
# q=(-sqrt(1/2), 0, sqrt(1/2), 0).  The child transform below makes the tray
# frame horizontal and centers it at root-relative p=(0.01384, 0, 0.15595),
# 1.5 mm above the top-head collision AABB (z_max=0.154433 m), leaving
# approximately 0.5 mm between the box's lower face and that AABB.
TRAY_POS_IN_JAW = (0.04335135, 0.0, -0.02284261)
TRAY_QUAT_IN_JAW = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)

TOP_BALL_RADIUS = 0.020
TOP_BALL_MASS = 0.010
TOP_BALL_FRICTION = (0.6, 0.005, 0.0001)  # sliding, torsional, rolling
# A 3 ms contact time constant keeps impact penetration below 1 mm at the
# selected 2 ms step in the audited drop case.  MuJoCo safely clamps it for the
# deliberately coarser 5 ms comparison case, which is one reason 5 ms is not
# the production choice.
TOP_BALL_SOLREF = (0.003, 1.0)
TOP_BALL_SOLIMP = (0.95, 0.99, 0.001, 0.5, 2.0)
TOP_BALL_BODY_NAME = "top_ball"
TOP_BALL_GEOM_NAME = "top_ball_sphere"
TOP_BALL_SITE_NAME = "top_ball_center"
TOP_BALL_RESET_CLEARANCE = 0.0005

CONTROL_DT = 0.020
DOUBLE_BALANCE_SIM_DT = float(os.getenv("MICRODUCK_DOUBLE_BALANCE_SIM_DT", "0.002"))


def _remap_robot_collision_masks(spec: mujoco.MjSpec) -> None:
    """Move robot contacts away from the floor's bit in this derived spec.

    Normal robot collisions still see the floor, the lower basketball, and one
    another.  The existing self-collision-only set remains isolated.  Moving
    both sets is what lets the top ball accept the floor bit without also
    accepting any part of the robot.
    """
    for geom in spec.geoms:
        mask = (int(geom.contype), int(geom.conaffinity))
        if mask == (1, 1):
            geom.contype = ROBOT_CONTACT_BIT
            geom.conaffinity = FLOOR_CONTACT_BIT | ROBOT_CONTACT_BIT
        elif mask == (2, 2):
            geom.contype = ROBOT_SELF_CONTACT_BIT
            geom.conaffinity = ROBOT_SELF_CONTACT_BIT
        if geom.name in {"left_foot_collision", "right_foot_collision"}:
            # Preserve the released task's explicit sole-contact settings even
            # though this derived spec does not run the generic collision editor.
            geom.condim = 3
            geom.priority = 1
            geom.friction[0] = 1.0


def _double_balance_robot_spec(
    base_spec_fn: Callable[[], mujoco.MjSpec] = get_standup_spec,
) -> mujoco.MjSpec:
    """Return an independent full-collision robot spec with a rigid tray."""
    spec = base_spec_fn()
    spec.modelname = "microduck_double_balance"
    _remap_robot_collision_masks(spec)

    jaw = next(body for body in spec.bodies if body.name == "jaw_soft")
    tray = jaw.add_body(
        name=TRAY_BODY_NAME,
        pos=TRAY_POS_IN_JAW,
        quat=TRAY_QUAT_IN_JAW,
        mass=TRAY_MASS,
        ipos=(0.0, 0.0, 0.0),
        iquat=(1.0, 0.0, 0.0, 0.0),
        inertia=TRAY_INERTIA,
        explicitinertial=1,
    )
    tray.add_geom(
        name=TRAY_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_BOX,
        size=tuple(value / 2.0 for value in TRAY_FULL_SIZE),
        mass=0.0,
        contype=TRAY_CONTACT_BIT,
        conaffinity=TOP_BALL_CONTACT_BIT,
        condim=4,
        friction=TRAY_FRICTION,
        rgba=(0.10, 0.52, 0.72, 1.0),
        group=0,
    )
    # The reset frame is on the physical top face, not at the box centre.
    tray.add_site(
        name=TRAY_SITE_NAME,
        pos=(0.0, 0.0, TRAY_FULL_SIZE[2] / 2.0),
        size=(0.0025,),
        rgba=(0.95, 0.85, 0.20, 1.0),
        group=3,
    )
    return spec


def _double_balance_lower_ball_spec() -> mujoco.MjSpec:
    """Clone the released basketball and remap contacts only in this task."""
    spec = _ball_spec()
    collision = next(geom for geom in spec.geoms if geom.name == "ball_sphere")
    collision.contype = ROBOT_CONTACT_BIT
    collision.conaffinity = FLOOR_CONTACT_BIT | ROBOT_CONTACT_BIT
    return spec


def _top_ball_spec() -> mujoco.MjSpec:
    """A single-body free sphere with no joint damping or hidden support."""
    spec = mujoco.MjSpec()
    spec.modelname = TOP_BALL_BODY_NAME
    body = spec.worldbody.add_body(
        name=TOP_BALL_BODY_NAME,
        pos=(0.0, 0.0, TOP_BALL_RADIUS),
    )
    body.add_freejoint(name="top_ball_freejoint")
    body.add_geom(
        name=TOP_BALL_GEOM_NAME,
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=(TOP_BALL_RADIUS,),
        mass=TOP_BALL_MASS,
        contype=TOP_BALL_CONTACT_BIT,
        conaffinity=FLOOR_CONTACT_BIT | TRAY_CONTACT_BIT,
        condim=4,
        friction=TOP_BALL_FRICTION,
        solref=TOP_BALL_SOLREF,
        solimp=TOP_BALL_SOLIMP,
        priority=1,
        rgba=(0.93, 0.84, 0.18, 1.0),
        group=0,
    )
    body.add_site(
        name=TOP_BALL_SITE_NAME,
        size=(0.0025,),
        rgba=(0.95, 0.25, 0.15, 1.0),
        group=3,
    )
    return spec


def _decimation_for(physics_dt: float) -> int:
    if not math.isfinite(physics_dt) or physics_dt <= 0.0:
        raise ValueError("Double-balance physics_dt must be finite and positive")
    decimation = round(CONTROL_DT / physics_dt)
    if decimation < 1 or not math.isclose(
        decimation * physics_dt, CONTROL_DT, rel_tol=0.0, abs_tol=1e-12
    ):
        raise ValueError(
            f"physics_dt={physics_dt} must divide the {CONTROL_DT:.3f} s control period"
        )
    return decimation


def make_microduck_double_balance_env_cfg(
    play: bool = False,
    *,
    blind: bool | None = None,
    history: int | None = None,
    physics_dt: float | None = None,
) -> ManagerBasedRlEnvCfg:
    """Build the Stage-03 scene without changing the basketball policy API."""
    cfg = make_microduck_basketball_env_cfg(
        play=play,
        blind=blind,
        history=history,
    )

    robot_cfg = deepcopy(cfg.scene.entities["robot"])
    robot_cfg.spec_fn = get_standup_spec
    robot_cfg = basketball_robot_cfg(robot_cfg)
    cream_spec_fn = robot_cfg.spec_fn
    robot_cfg.spec_fn = partial(_double_balance_robot_spec, cream_spec_fn)
    # The inherited FULL_COLLISION editor only selects ``.*_collision`` and
    # disables every other geom, which would also erase the newly named tray's
    # mask.  The derived spec already carries explicit masks for every visual,
    # robot, self-collision, and tray geom, so no post-build editor is needed.
    robot_cfg.collisions = ()
    cfg.scene.entities = {
        "robot": robot_cfg,
        "ball": EntityCfg(spec_fn=_double_balance_lower_ball_spec),
        "top_ball": EntityCfg(spec_fn=_top_ball_spec),
    }

    dt = DOUBLE_BALANCE_SIM_DT if physics_dt is None else physics_dt
    cfg.sim.mujoco.timestep = dt
    cfg.decimation = _decimation_for(dt)
    cfg.sim.nconmax = max(getattr(cfg.sim, "nconmax", 0) or 0, 256)

    # The inherited guard only watches the robot.  Both balls have free joints,
    # and a contact blow-up in either one must recycle the environment before
    # Stage-04 observations or rewards consume a non-finite state.
    cfg.terminations["nan_state"].params["extra_entity_names"] = (
        "ball",
        "top_ball",
    )

    original_spawn = cfg.events.pop("reset_basketball")
    original_spawn.func = microduck_mdp.reset_double_balance
    original_spawn.params.update(
        {
            "top_ball_radius": TOP_BALL_RADIUS,
            "tray_site_name": TRAY_SITE_NAME,
            "top_ball_xy_noise": 0.0,
            "top_ball_clearance": TOP_BALL_RESET_CLEARANCE,
        }
    )
    cfg.events = {"reset_double_balance": original_spawn, **cfg.events}
    return cfg


MicroduckDoubleBalanceRlCfg = deepcopy(MicroduckBasketballRlCfg)
MicroduckDoubleBalanceRlCfg.experiment_name = "double_balance"
