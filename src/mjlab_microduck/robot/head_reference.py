"""Audited material-point reference. A site has no mass or collision surface."""
from __future__ import annotations

import mujoco
import numpy as np

from mjlab_microduck.robot.head_contact import make_head_contact_robot_spec

REFERENCE_SITE = "stage11_head_material_frame"
REFERENCE_VERSION = "head-material-tangent-v1-prototype"
# Pinned 2ce9434 return: candidate-references.json, inherited center, jaw axes.
ORIGIN_IN_JAW = (0.037626392020943145, -1.0378391528327583e-13, -0.02441209959360968)
NORMAL_IN_JAW = (0.9970649400829094, 5.0626300138182914e-12, 0.0765604679809603)
RADIUS = 0.020
CLEARANCE = 0.0005


def reference_rotation_in_jaw():
    """Z = surface normal; X = projected HOME forward; Y = Z cross X."""
    z = np.asarray(NORMAL_IN_JAW, dtype=float)
    z = z / np.linalg.norm(z)
    seed = np.array([0., 0., -1.])
    x = seed - z * (seed @ z)
    x /= np.linalg.norm(x)
    return np.column_stack((x, np.cross(z, x), z))


def make_head_reference_robot_spec(base_spec_fn=None):
    spec = make_head_contact_robot_spec(base_spec_fn)
    if spec.site(REFERENCE_SITE) is not None:
        raise ValueError("Duplicate head material reference")
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, reference_rotation_in_jaw().reshape(9))
    spec.body("jaw_soft").add_site(
        name=REFERENCE_SITE, pos=ORIGIN_IN_JAW, quat=quat,
        size=(0.001,), rgba=(0., 0., 0., 0.), group=5,
    )
    return spec
