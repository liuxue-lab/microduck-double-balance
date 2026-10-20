"""Stage 11 isolated physical apparatus; no RL task, reward or policy adapter.

Factories always obtain a fresh Stage-03 spec. The only apparatus edits are
removing the tray subtree and allowing the upper sphere to touch the real upper
head shell. Fixed-head fixtures belong to the diagnostic script, never here.
"""
from __future__ import annotations

from functools import partial
import hashlib
from pathlib import Path

import numpy as np

HEAD_CONTACT_BIT = 1 << 6
HEAD_GEOM_NAME = "stage11_head_contact_surface"
TRAY_BODY = "double_balance_tray"
TRAY_GEOM = "double_balance_tray_surface"
TOP_GEOM = "top_ball_sphere"
BASE_HEAD_MESH = "top_head_shell"


def make_head_contact_robot_spec(base_spec_fn=None):
    """Clone the old tray robot through its factory; do not edit a shared spec."""
    if base_spec_fn is None:
        from mjlab_microduck.tasks.microduck_double_balance_env_cfg import (
            _double_balance_robot_spec,
        )
        base_spec_fn = _double_balance_robot_spec
    spec = base_spec_fn()
    tray = spec.body(TRAY_BODY)
    if tray is None or not np.isclose(tray.mass, 0.018, rtol=0, atol=1e-12):
        raise ValueError("Expected the frozen explicit 18 g tray subtree")
    jaw = spec.body("jaw_soft")
    if jaw is None:
        raise ValueError("Missing jaw_soft")
    shells = [g for g in jaw.geoms if g.meshname == BASE_HEAD_MESH
              and (int(g.contype) or int(g.conaffinity))]
    if len(shells) != 1:
        raise ValueError(f"Expected exactly one collision head shell, got {len(shells)}")
    if any((int(g.contype) | int(g.conaffinity)) & HEAD_CONTACT_BIT for g in spec.geoms):
        raise ValueError("Head contact bit is already used")
    shell = shells[0]
    shell.name = HEAD_GEOM_NAME
    shell.contype = int(shell.contype) | HEAD_CONTACT_BIT
    # Retain the old shell's affinities, friction, priority, mesh and pose.
    spec.delete(tray)
    spec.modelname = "microduck_stage11_head_contact_physics_only"
    return spec


def make_head_contact_top_ball_spec(base_spec_fn=None):
    if base_spec_fn is None:
        from mjlab_microduck.tasks.microduck_double_balance_env_cfg import _top_ball_spec
        base_spec_fn = _top_ball_spec
    spec = base_spec_fn()
    geom = spec.geom(TOP_GEOM)
    if geom is None or int(geom.conaffinity) != ((1 << 0) | (1 << 4)):
        raise ValueError("Unexpected frozen upper-ball collision mask")
    geom.conaffinity = (1 << 0) | HEAD_CONTACT_BIT
    return spec


def physics_scene_pair():
    """Return original cfg and two SceneCfgs. Never construct an RL environment.

    Callers must not pass these scene edits back to the old ManagerBasedRlEnv:
    its reset/observation/reward/acceptance references still require a tray.
    """
    from copy import deepcopy
    from mjlab_microduck.tasks.microduck_double_balance_env_cfg import (
        make_microduck_double_balance_env_cfg,
    )
    cfg = make_microduck_double_balance_env_cfg(play=True, physics_dt=0.002)
    cfg.scene.num_envs = 1
    original = deepcopy(cfg.scene)
    candidate = deepcopy(cfg.scene)
    robot = candidate.entities["robot"]
    robot.spec_fn = partial(make_head_contact_robot_spec, robot.spec_fn)
    ball = candidate.entities["top_ball"]
    ball.spec_fn = partial(make_head_contact_top_ball_spec, ball.spec_fn)
    # No registry change, manager construction, checkpoint load or optimizer.
    return cfg, original, candidate


def sha256(path: Path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rotation(wxyz):
    q = np.asarray(wxyz, dtype=float)
    q = q / np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def axis_rotation(axis, angle):
    q = np.zeros(4); q[0] = np.cos(angle / 2); q[axis+1] = np.sin(angle / 2)
    return rotation(q)


def parallel_axis(mass, offset):
    r = np.asarray(offset, dtype=float)
    return mass * ((r @ r) * np.eye(3) - np.outer(r, r))


def combine_inertias(masses, centers, tensors):
    masses, centers, tensors = map(np.asarray, (masses, centers, tensors))
    mass = float(masses.sum())
    if mass <= 0:
        raise ValueError("Positive mass required")
    center = np.einsum("i,ij->j", masses, centers) / mass
    inertia = sum((i + parallel_axis(m, p-center)
                   for m, p, i in zip(masses, centers, tensors, strict=True)), np.zeros((3, 3)))
    return {"mass_kg": mass, "com_m": center.tolist(), "inertia_com_kg_m2": inertia.tolist()}


def model_cluster(model, data, ids):
    """World axes, about the cluster's own COM; includes off-diagonal terms."""
    ids = list(ids)
    rotations = np.asarray(data.ximat)[ids].reshape(-1, 3, 3)
    tensors = [r @ np.diag(model.body_inertia[i]) @ r.T for r, i in zip(rotations, ids)]
    return combine_inertias(model.body_mass[ids], data.xipos[ids], tensors)


def can_contact(model, a, b):
    return bool((int(model.geom_contype[a]) & int(model.geom_conaffinity[b]))
                or (int(model.geom_contype[b]) & int(model.geom_conaffinity[a])))


def mesh_triangles(model, data, geom_id):
    mesh_id = int(model.geom_dataid[geom_id])
    va, vn = int(model.mesh_vertadr[mesh_id]), int(model.mesh_vertnum[mesh_id])
    fa, fn = int(model.mesh_faceadr[mesh_id]), int(model.mesh_facenum[mesh_id])
    vertices = model.mesh_vert[va:va+vn].astype(float)
    vertices = vertices @ data.geom_xmat[geom_id].reshape(3, 3).T + data.geom_xpos[geom_id]
    faces = model.mesh_face[fa:fa+fn].copy()
    return vertices, faces


def closest_on_triangles(point, triangles):
    """Vectorized exact point/triangle closest points (faces, edges, vertices).

    No hull approximation and no optional spatial-index dependency. Degenerate
    faces fall back to their edges. Returns one witness per input triangle.
    """
    p = np.asarray(point, dtype=float)
    t = np.asarray(triangles, dtype=float)
    a, b, c = t[:, 0], t[:, 1], t[:, 2]
    ab, ac, ap = b-a, c-a, p-a
    aa = np.einsum("ij,ij->i", ab, ab)
    bb = np.einsum("ij,ij->i", ac, ac)
    cc = np.einsum("ij,ij->i", ab, ac)
    d = np.einsum("ij,ij->i", ap, ab)
    e = np.einsum("ij,ij->i", ap, ac)
    det = aa*bb-cc*cc
    valid = det > 1e-28
    u = np.divide(d*bb-e*cc, det, out=np.zeros_like(det), where=valid)
    v = np.divide(e*aa-d*cc, det, out=np.zeros_like(det), where=valid)
    projected = a + u[:, None]*ab + v[:, None]*ac
    interior = valid & (u >= 0) & (v >= 0) & (u+v <= 1)
    candidates = [projected]
    distances = [np.where(interior, np.sum((projected-p)**2, axis=1), np.inf)]
    for start, end in ((a, b), (b, c), (c, a)):
        delta = end-start
        den = np.sum(delta*delta, axis=1)
        s = np.divide(np.sum((p-start)*delta, axis=1), den,
                      out=np.zeros_like(den), where=den > 0).clip(0, 1)
        q = start + s[:, None]*delta
        candidates.append(q); distances.append(np.sum((q-p)**2, axis=1))
    distances = np.stack(distances, axis=1)
    witnesses = np.stack(candidates, axis=1)
    pick = distances.argmin(axis=1)
    return witnesses[np.arange(len(t)), pick], distances[np.arange(len(t)), pick]


def nearest_surface(point, triangles):
    points, d2 = closest_on_triangles(point, triangles)
    k = int(d2.argmin())
    return points[k], float(np.sqrt(d2[k])), k


def vertical_surface_height(xy, triangles):
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac, p = b[:, :2]-a[:, :2], c[:, :2]-a[:, :2], np.asarray(xy)-a[:, :2]
    det = ab[:, 0]*ac[:, 1]-ab[:, 1]*ac[:, 0]
    good = np.abs(det) > 1e-20
    u = np.divide(p[:, 0]*ac[:, 1]-p[:, 1]*ac[:, 0], det, out=np.zeros_like(det), where=good)
    v = np.divide(ab[:, 0]*p[:, 1]-ab[:, 1]*p[:, 0], det, out=np.zeros_like(det), where=good)
    good &= (u >= -1e-9) & (v >= -1e-9) & (u+v <= 1+1e-9)
    if not good.any():
        raise ValueError("Requested center projection is outside the original shell")
    z = a[:, 2]+u*(b[:, 2]-a[:, 2])+v*(c[:, 2]-a[:, 2])
    return float(z[good].max())


def sphere_center_above(xy, triangles, radius=0.02, clearance=0.0005):
    """First nonpenetrating finite-radius support from above, not ray_z + R."""
    if radius <= 0 or clearance < 0:
        raise ValueError("Invalid sphere dimensions")
    lo = vertical_surface_height(xy, triangles)
    hi = float(triangles[:, :, 2].max()) + radius + clearance + 0.001
    target = radius + clearance
    for _ in range(42):
        z = (lo+hi)/2
        _, distance, _ = nearest_surface([*xy, z], triangles)
        if distance < target: lo = z
        else: hi = z
    center = np.array([*xy, hi])
    witness, distance, k = nearest_surface(center, triangles)
    return center, witness, (center-witness)/distance, k


def contact_slip(v_ball, w_ball, p_ball, v_head, w_head, p_head, point, normal):
    """Tangential material-point velocity, in world axes, at one common point."""
    n = np.asarray(normal, dtype=float); n = n/np.linalg.norm(n)
    relative = (np.asarray(v_ball) + np.cross(w_ball, np.asarray(point)-p_ball)
                - np.asarray(v_head) - np.cross(w_head, np.asarray(point)-p_head))
    return relative - n * (relative @ n)


def point_matrix():
    return [("center", 0., 0.), ("x-minus12", -.012, 0.), ("x-plus12", .012, 0.),
            ("y-minus12", 0., -.012), ("y-plus12", 0., .012),
            ("y-minus20", 0., -.020), ("y-plus20", 0., .020)]


def pose_matrix():
    return [("home", np.eye(3)), ("yaw-minus90", axis_rotation(2, -np.pi/2)),
            ("yaw-plus90", axis_rotation(2, np.pi/2)),
            ("pitch-minus10", axis_rotation(1, -np.pi/18)),
            ("pitch-plus10", axis_rotation(1, np.pi/18)),
            ("roll-minus10", axis_rotation(0, -np.pi/18)),
            ("roll-plus10", axis_rotation(0, np.pi/18))]


def dynamic_matrix(triangles, center_xy):
    result = []
    for name, x, y in point_matrix():
        p, witness, normal, _ = sphere_center_above(np.asarray(center_xy)+[x, y], triangles)
        result.append(dict(name="release-"+name, center=p, witness=witness,
                           normal=normal, velocity=np.zeros(3), omega=np.zeros(3)))
    base = result[0]
    n = base["normal"]
    tx = np.array([1., 0., 0.]); tx -= n*(tx@n); tx /= np.linalg.norm(tx)
    ty = np.cross(n, tx)
    for name, v in (("x-plus", tx), ("x-minus", -tx), ("y-plus", ty), ("y-minus", -ty)):
        result.append({**base, "name":"slide-"+name, "velocity":.05*v})
    result.append({**base, "name":"initial-no-slip-roll", "velocity":.05*tx,
                   "omega":np.cross(n, .05*tx)/.02})
    result.append({**base, "name":"normal-spin", "omega":n.copy()})
    return result
