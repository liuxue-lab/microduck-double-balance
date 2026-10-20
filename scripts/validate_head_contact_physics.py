"""Stage 11 zero-PPO physical validation. Never evaluates policy success.

Only material fixtures are stepped. The complete robot scene is compiled for
structural/inertial audits, with its head/neck free and unmodified. Fixed-head
fixtures are separate models with an explicit fixed-material label.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np

from mjlab_microduck.robot.head_contact import (
    HEAD_CONTACT_BIT, HEAD_GEOM_NAME, TOP_GEOM, TRAY_BODY, TRAY_GEOM,
    can_contact, combine_inertias, contact_slip, dynamic_matrix,
    mesh_triangles, model_cluster, nearest_surface, physics_scene_pair,
    point_matrix, pose_matrix, sha256, sphere_center_above,
)

ROOT = Path(__file__).resolve().parents[1]
DT = 0.002
SECONDS = 3.0
RADIUS = 0.02
GEOMETRY_TOL = 0.0002
PENETRATION_TOL = 0.001
NORMAL_TOL_DEG = 5.0
GEOMETRY_GATE_VERSIONS = ('v1', 'v2')
BASELINE = "056eaf2b0c78261f34e3fa581858c8573cab981e"


def serial(x):
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, Path): return str(x)
    raise TypeError(type(x).__name__)


def save(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=serial,
                              allow_nan=False)+"\n")
    tmp.replace(path)


def emit(message):
    print(datetime.now(timezone.utc).isoformat(), message, flush=True)


def require(condition, message):
    if not bool(condition): raise RuntimeError(message)


def gid(model, suffix):
    ids = [i for i in range(model.ngeom) if (model.geom(i).name or "").split("/")[-1] == suffix]
    require(len(ids) == 1, f"Expected one geom {suffix}, got {ids}")
    return ids[0]


def bid(model, suffix):
    ids = [i for i in range(model.nbody) if (model.body(i).name or "").split("/")[-1] == suffix]
    require(len(ids) == 1, f"Expected one body {suffix}, got {ids}")
    return ids[0]


def initial_data(model):
    data = mujoco.MjData(model)
    if model.nkey: mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return data


def descendants(model, root):
    result = []
    for i in range(1, model.nbody):
        j = i
        while j and j != root: j = int(model.body_parentid[j])
        if j == root: result.append(i)
    return result


def options_dict(model):
    result = {}
    for name in dir(model.opt):
        if name.startswith("_"): continue
        v = getattr(model.opt, name)
        if isinstance(v, np.ndarray): result[name] = v.copy()
        elif isinstance(v, (float, int, bool)): result[name] = v
        elif hasattr(v, "value"): result[name] = int(v)
    return result


def copy_options(source, destination):
    for name, value in options_dict(source).items():
        target = getattr(destination.opt, name)
        if isinstance(target, np.ndarray): target[:] = value
        else: setattr(destination.opt, name, value)


def audit_full_scene(old, new):
    od, nd = initial_data(old), initial_data(new)
    checks = {}
    checks["same_generalized_dimensions"] = (old.nq, old.nv, old.nu, old.njnt) == (new.nq, new.nv, new.nu, new.njnt)
    checks["no_added_equalities"] = new.neq == old.neq == 0
    checks["exactly_one_body_and_geom_removed"] = new.nbody == old.nbody-1 and new.ngeom == old.ngeom-1
    checks["no_tray_names"] = not any("double_balance_tray" in (new.body(i).name or "") for i in range(new.nbody))
    checks["no_tray_sites"] = not any("double_balance_tray" in (new.site(i).name or "") for i in range(new.nsite))
    for field in ("jnt_type", "jnt_axis", "jnt_range", "jnt_limited", "jnt_stiffness",
                  "dof_damping", "dof_armature", "dof_frictionloss", "actuator_trntype",
                  "actuator_gear", "actuator_gainprm", "actuator_biasprm", "actuator_ctrlrange",
                  "actuator_forcerange"):
        checks["unchanged_"+field] = np.array_equal(getattr(old, field), getattr(new, field))
    checks["unchanged_joint_names"] = [old.joint(i).name for i in range(old.njnt)] == [new.joint(i).name for i in range(new.njnt)]
    tray = gid(old, TRAY_GEOM); head = gid(new, HEAD_GEOM_NAME)
    top = gid(new, TOP_GEOM); floor = gid(new, "terrain")
    allowed = [i for i in range(new.ngeom) if i != top and can_contact(new, top, i)]
    checks["upper_ball_only_head_and_floor"] = set(allowed) == {head, floor}
    checks["no_explicit_contact_pairs"] = old.npair == new.npair == 0
    old_ids = [i for i in range(old.ngeom) if i != tray]
    # Validate correspondence using physical identities before comparing masks.
    structural_ok = True
    fields = ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction",
              "geom_priority", "geom_condim", "geom_solref", "geom_solimp", "geom_margin", "geom_gap")
    for oi, ni in zip(old_ids, range(new.ngeom), strict=True):
        structural_ok &= old.body(int(old.geom_bodyid[oi])).name == new.body(int(new.geom_bodyid[ni])).name
        for field in fields:
            structural_ok &= np.array_equal(getattr(old, field)[oi], getattr(new, field)[ni])
    checks["all_retained_geom_physics_unchanged"] = bool(structural_ok)
    checks["other_contact_matrix_unchanged"] = all(
        can_contact(old, old_ids[a], old_ids[b]) == can_contact(new, a, b)
        for a in range(new.ngeom) for b in range(a+1, new.ngeom) if top not in (a,b))
    checks["upper_ball_free_no_joint_impedance"] = True
    tb = int(new.geom_bodyid[top]); joints = np.flatnonzero(new.jnt_bodyid == tb)
    checks["upper_ball_free_no_joint_impedance"] &= len(joints) == 1 and new.jnt_type[joints[0]] == mujoco.mjtJoint.mjJNT_FREE
    dofs = np.flatnonzero(np.isin(new.dof_jntid, joints))
    checks["upper_ball_free_no_joint_impedance"] &= all(np.all(getattr(new,f)[dofs] == 0) for f in ("dof_damping","dof_armature","dof_frictionloss"))
    reports = {}
    for label, root in (("robot", "trunk_base"), ("jaw_cluster", "jaw_soft")):
        old_cluster = model_cluster(old, od, descendants(old, bid(old, root)))
        new_cluster = model_cluster(new, nd, descendants(new, bid(new, root)))
        tray_cluster = model_cluster(old, od, [bid(old, TRAY_BODY)])
        recombined = combine_inertias(
            [new_cluster["mass_kg"], tray_cluster["mass_kg"]],
            [new_cluster["com_m"], tray_cluster["com_m"]],
            [new_cluster["inertia_com_kg_m2"], tray_cluster["inertia_com_kg_m2"]])
        checks[label+"_mass_removes_18g"] = abs(old_cluster["mass_kg"]-new_cluster["mass_kg"]-.018) <= 1e-9
        checks[label+"_com_parallel_axis"] = bool(np.allclose(old_cluster["com_m"], recombined["com_m"], rtol=0, atol=1e-7))
        checks[label+"_full_inertia_parallel_axis"] = bool(np.allclose(old_cluster["inertia_com_kg_m2"], recombined["inertia_com_kg_m2"], rtol=1e-6, atol=1e-10))
        checks[label+"_positive_inertia"] = bool(np.linalg.eigvalsh(new_cluster["inertia_com_kg_m2"]).min() > 0)
        reports[label] = {"before":old_cluster,"after":new_cluster,"after_plus_removed_tray":recombined,
                          "coordinates":"HOME world axes; each tensor about its own COM"}
    checks["robot_mass_matches_stage10"] = abs(reports["robot"]["after"]["mass_kg"]-.73724318) <= 1e-9
    return {"status":"PASS" if all(checks.values()) else "BLOCKED", "checks":checks,
            "clusters":reports,"scene_mass_before_kg":float(old.body_mass.sum()),
            "scene_mass_after_kg":float(new.body_mass.sum()),
            "allowed_upper_contact_names":[new.geom(i).name for i in allowed],
            "head_neck_locked":False,"full_robot_dynamics_steps":0}


def fixture_from_model(source, source_data, orientation=None):
    """Copy the actual shell triangles and contact params into a fixed material rig."""
    head = gid(source, HEAD_GEOM_NAME); top = gid(source, TOP_GEOM); floor = gid(source, "terrain")
    vertices, faces = mesh_triangles(source, source_data, head)
    pivot = source_data.xpos[bid(source, "jaw_soft")].copy()
    r = np.eye(3) if orientation is None else orientation
    vertices = (vertices-pivot) @ r.T + pivot
    spec = mujoco.MjSpec(); spec.modelname = "stage11_FIXED_HEAD_MATERIAL_ONLY"
    spec.add_mesh(name="original_head_triangles", uservert=vertices.ravel(), userface=faces.ravel())
    body = spec.worldbody.add_body(name="fixed_material_head")
    shell = body.add_geom(name="head", type=mujoco.mjtGeom.mjGEOM_MESH,
                          meshname="original_head_triangles", mass=0, rgba=[.88,.83,.78,1])
    ground = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                                    size=source.geom_size[floor], pos=source_data.geom_xpos[floor], rgba=[.2,.25,.3,1])
    ballbody = spec.worldbody.add_body(name="ball", pos=[0,0,1]); ballbody.add_freejoint(name="ball_free")
    sphere = ballbody.add_geom(name="sphere",type=mujoco.mjtGeom.mjGEOM_SPHERE,
                              size=[RADIUS], mass=float(source.body_mass[int(source.geom_bodyid[top])]), rgba=[.95,.7,.15,1])
    for target, source_id in ((shell,head),(sphere,top),(ground,floor)):
        for name in ("contype","conaffinity","condim","priority","friction","solref","solimp","margin","gap"):
            value = getattr(source,"geom_"+name)[source_id]
            setattr(target,name, value.copy() if isinstance(value,np.ndarray) else value)
    spec.worldbody.add_light(pos=[.3,-.4,1.2],dir=[-.2,.3,-1])
    model = spec.compile(); copy_options(source,model)
    require(model.nq == 7 and model.nv == 6 and model.nu == model.neq == 0, "Material fixture must only have one free sphere")
    require(model.opt.timestep == DT, "Unapproved material timestep")
    return model, vertices, faces, pivot


def set_ball(model, center, velocity=None, omega=None):
    data=mujoco.MjData(model); data.qpos[:3]=center; data.qpos[3]=1
    data.qvel[:3]=np.zeros(3) if velocity is None else velocity
    # Identity initial freejoint quaternion: world and local angular axes coincide.
    data.qvel[3:]=np.zeros(3) if omega is None else omega
    mujoco.mj_forward(model,data)
    return data


def geometry_metrics(center, witness, normal, gap, triangles):
    """Separate surface error from tangential nearest-witness displacement.

    The sphere centre stays outside the shell (penetration is separately
    capped at 1 mm, versus a 20 mm radius). Distances use all original faces.
    """
    center=np.asarray(center,float);witness=np.asarray(witness,float)
    normal=np.asarray(normal,float)
    require(np.isfinite(np.r_[center,witness,normal,gap]).all(),'Non-finite geometry input')
    normal_norm=np.linalg.norm(normal)
    require(abs(normal_norm-1)<1e-4,'Contact normal is not a unit vector')
    normal=normal/normal_norm
    original_witness,distance,face=nearest_surface(center,triangles)
    require(distance>1e-9,'Sphere centre lies on the original shell')
    expected_normal=(center-original_witness)/distance
    _,surface_distance,_=nearest_surface(witness,triangles)
    gap_error=abs(float(gap)-(distance-RADIUS))
    point_error=float(np.linalg.norm(witness-original_witness))
    angle=float(np.degrees(np.arccos(np.clip(normal@expected_normal,-1,1))))
    common=gap_error<=GEOMETRY_TOL and angle<=NORMAL_TOL_DEG
    return {'mesh_witness':original_witness,'expected_normal':expected_normal,
            'original_face':face,'original_sphere_gap_m':distance-RADIUS,
            'engine_witness':witness,'engine_normal':normal,'engine_distance_m':float(gap),
            'witness_separation_m':point_error,'surface_distance_m':surface_distance,
            'sphere_gap_error_m':gap_error,'normal_error_deg':angle,
            'legacy_v1_pass':bool(common and point_error<=GEOMETRY_TOL),
            'v2_pass':bool(common and surface_distance<=GEOMETRY_TOL)}


def selected_geometry_pass(metrics, gate_version):
    require(gate_version in GEOMETRY_GATE_VERSIONS,'Unknown geometry gate version')
    return metrics['legacy_v1_pass'] if gate_version=='v1' else metrics['v2_pass']


def static_geometry(source, source_data, center_xy, out, gate_version='v1'):
    model, vertices, faces, pivot = fixture_from_model(source,source_data)
    triangles=vertices[faces]
    points=[]
    for name, dx, dy in point_matrix():
        center,witness,normal,face = sphere_center_above(np.asarray(center_xy)+[dx,dy],triangles)
        points.append((name,center,witness,normal,face))
    rows=[]
    for pose, r in pose_matrix():
        m, vv, ff, _ = fixture_from_model(source,source_data,r)
        for name,c,w,n,face in points:
            center=(c-pivot)@r.T+pivot; expected_w=(w-pivot)@r.T+pivot; expected_n=r@n
            d=set_ball(m,center); ft=np.full(6,np.nan)
            distance=mujoco.mj_geomDistance(m,d,m.geom("head").id,m.geom("sphere").id,1.,ft)
            vector=ft[3:]-ft[:3]; length=np.linalg.norm(vector)
            normal=vector/length if length > 1e-12 else np.zeros(3)
            metrics=geometry_metrics(center,ft[:3],normal,distance,vv[ff])
            require(np.linalg.norm(metrics['mesh_witness']-expected_w)<1e-7,
                    'Original-mesh rigid transform witness check failed')
            rows.append(dict(pose=pose,point=name,center=center,**metrics,
                             pass_=selected_geometry_pass(metrics,gate_version)))
    result={"status":"PASS" if all(x['pass_'] for x in rows) else "BLOCKED",
            "geometry_gate_version":gate_version,
            "legacy_v1_status":"PASS" if all(x['legacy_v1_pass'] for x in rows) else "BLOCKED",
            "v2_status":"PASS" if all(x['v2_pass'] for x in rows) else "BLOCKED",
            "count":len(rows),"rows":rows,"simulation_steps":0,
            "mesh_vertices":len(vertices),"mesh_faces":len(faces),
            "criterion":"Versioned original-triangle surface/gap/normal gates; v1 point-to-point gate retained",
            "historical_v1_evidence_overwritten":False}
    save(out/'static-geometry.json',result)
    np.savez_compressed(out/'material-geometry.npz',vertices=vertices,faces=faces,pivot=pivot,center_xy=center_xy)
    return result,model,triangles


def contact_rows(model, before_qpos, before_qvel, records, step, backend, triangles=None):
    """All values refer to the pre-integration state of this same physics step."""
    ball_id=model.geom("sphere").id; head_id=model.geom("head").id; floor_id=model.geom("floor").id
    p=before_qpos[:3]
    # MuJoCo freejoint angular qvel is local; convert to world using this qpos.
    from mjlab_microduck.robot.head_contact import rotation
    w=rotation(before_qpos[3:7])@before_qvel[3:6]
    rows=[]
    for rec in records:
        a,b=map(int,rec['geom'])
        if ball_id not in (a,b): continue
        other=b if a==ball_id else a
        frame=np.asarray(rec['frame']).reshape(3,3)
        normal=frame[0] if b==ball_id else -frame[0]
        slip=contact_slip(before_qvel[:3],w,p,np.zeros(3),np.zeros(3),np.zeros(3),rec['pos'],normal)
        force=np.asarray(rec['force']); force_world=frame.T@force[:3]
        force_on_ball=force_world if b==ball_id else -force_world
        ft=force_on_ball-normal*(force_on_ball@normal)
        power=float(ft@slip)
        item={**rec,"geom": [a,b],"backend":backend,"step":step,"time_s":step*DT,
                     "phase":"pre_integration_contacts_and_solver_force",
                     "other":model.geom(other).name,"allowed":other in {head_id,floor_id},
                     "normal_toward_ball":normal,"slip_world_m_s":slip,
                     "slip_speed_m_s":float(np.linalg.norm(slip)),
                     "tangential_power_W":power,"ball_omega_world_rad_s":w,
                     "ball_center_world_m":p.copy()}
        if other==head_id and triangles is not None:
            # mjContact.pos is the midpoint of the two surface witnesses.
            # Do not treat the midpoint inside a soft contact as the head surface.
            head_witness=np.asarray(rec['pos'])-.5*float(rec['dist'])*normal
            item['geometry']=geometry_metrics(p,head_witness,normal,rec['dist'],triangles)
        fn=float(force[0]);item['normal_force_N']=fn
        item['tangent_force_N']=float(np.linalg.norm(force[1:3]))
        item['torsional_torque_Nm']=float(force[3])
        item['active_torsional_relative_rate_rad_s']=float(w@normal)
        item['torsional_power_W']=float(force[3])*(1 if b==ball_id else -1)*float(w@frame[0])
        item['pyramidal_cone_utilization']=(float(np.sum(np.abs(force[1:4])/np.asarray(rec['friction'])[:3]))/fn
                                          if fn>1e-8 else None)
        rows.append(item)
    return rows


def cpu_records(model,data):
    result=[]
    for i,c in enumerate(data.contact):
        force=np.zeros(6); mujoco.mj_contactForce(model,data,i,force)
        result.append(dict(geom=c.geom.copy(),dist=float(c.dist),pos=c.pos.copy(),
                           frame=c.frame.copy(),dim=int(c.dim),friction=c.friction.copy(),
                           solref=c.solref.copy(),solimp=c.solimp.copy(),force=force,
                           efc_address=int(c.efc_address)))
    return result


class WarpStepper:
    def __init__(self,model,data,device,ls_parallel=True):
        import warp as wp
        import mujoco_warp as mw
        self.wp,self.mw=wp,mw; self.device=wp.get_device(device)
        with wp.ScopedDevice(self.device):
            self.model=mw.put_model(model); self.model.opt.ls_parallel=ls_parallel
            self.data=mw.put_data(model,data,nworld=1,nconmax=256,njmax=1024)
            self.ids=wp.array(np.arange(256,dtype=np.int32),dtype=wp.int32)
            self.forces=wp.zeros(256,dtype=wp.spatial_vector)
            self.graph=None
            if self.device.is_cuda:
                # Capture only; no discarded warmup dynamics or changed initial state.
                with wp.ScopedCapture(device=self.device) as capture:
                    mw.step(self.model,self.data)
                    mw.contact_force(self.model,self.data,self.ids,False,self.forces)
                self.graph=capture.graph

    def step(self):
        wp,mw=self.wp,self.mw
        with wp.ScopedDevice(self.device):
            if self.graph is not None: wp.capture_launch(self.graph)
            else:
                mw.step(self.model,self.data)
                mw.contact_force(self.model,self.data,self.ids,False,self.forces)
            qpos=self.data.qpos.numpy()[0].copy(); qvel=self.data.qvel.numpy()[0].copy()
            count=int(self.data.nacon.numpy()[0]); constraints=int(self.data.nefc.numpy()[0])
            require(count<=256 and constraints<=1024,"Warp contact/constraint capacity exceeded")
            arrays={k:getattr(self.data.contact,k).numpy()[:count].copy()
                    for k in ('geom','dist','pos','frame','dim','friction','solref','solimp','efc_address')}
            forces=self.forces.numpy()[:count].copy()
        records=[]
        for i in range(count):
            records.append({**{k:v[i] for k,v in arrays.items()},"force":forces[i],
                            "efc_address":int(arrays['efc_address'][i,0])})
        return qpos,qvel,records


def run_case(model,case,backend,out,*,warp_device='cuda:0',deadline=None,ls_parallel=True,
             triangles=None,gate_version='v1'):
    destination=out/backend/case['name']; destination.mkdir(parents=True,exist_ok=False)
    save(destination/'initial.json',case)
    poses=[];velocities=[];started=time.monotonic();heartbeat=started
    summary={"backend":backend,"case":case['name'],"status":"RUNNING","steps":0,
             "geometry_gate_version":gate_version,"max_penetration_m":0.,
             "head_contacts":0,"floor_contacts":0,"max_slip_m_s":0.,
             "max_positive_tangential_power_W":0.,"max_cone_utilization":0.,
             "max_surface_distance_m":0.,"max_sphere_gap_error_m":0.,
             "max_normal_error_deg":0.,"legacy_v1_failed_contact_count":0,
             "new_ppo_updates":0,"material_fixture":True,"policy_success":None}
    try:
        d=set_ball(model,case['center'],case['velocity'],case['omega'])
        require(all(c.dist >= -1e-8 for c in d.contact),"Initial state intersects a collision surface")
        poses=[d.qpos.copy()];velocities=[d.qvel.copy()]
        stepper=WarpStepper(model,d,warp_device,ls_parallel) if backend=='warp' else None
        qpos=d.qpos.copy();qvel=d.qvel.copy()
        if stepper is not None:
            # Actual float32 device upload, not the float64 source state.
            qpos=stepper.data.qpos.numpy()[0].copy();qvel=stepper.data.qvel.numpy()[0].copy()
            poses[0]=qpos.copy();velocities[0]=qvel.copy()
            save(destination/'warp-device.json',{'device':str(stepper.device),
                 'is_cuda':stepper.device.is_cuda,'ls_parallel':ls_parallel,
                 'capture_performs_no_warmup_steps':True})
        with gzip.open(destination/'contacts.jsonl.gz','wt',encoding='utf-8') as log, \
             gzip.open(destination/'raw-contacts.jsonl.gz','wt',encoding='utf-8') as raw:
            for step in range(round(SECONDS/DT)):
                require(deadline is None or time.monotonic()<deadline,"Diagnostic wall-clock limit reached")
                before_qpos=qpos.copy();before_qvel=qvel.copy()
                if stepper is None:
                    mujoco.mj_step(model,d);qpos=d.qpos.copy();qvel=d.qvel.copy();records=cpu_records(model,d)
                else:qpos,qvel,records=stepper.step()
                poses.append(qpos.copy());velocities.append(qvel.copy());summary['steps']=step+1
                raw.write(json.dumps({'step':step,'phase':'pre_integration',
                    'contacts':records},default=serial,allow_nan=False)+'\n')
                require(np.isfinite(qpos).all() and np.isfinite(qvel).all(),"Non-finite physical state")
                rows=contact_rows(model,before_qpos,before_qvel,records,step,backend,triangles)
                for row in rows:
                    log.write(json.dumps(row,default=serial,allow_nan=False)+'\n')
                    require(row['allowed'],"Unexpected upper-ball contact")
                    summary[row['other']+'_contacts']+=1
                    summary['max_penetration_m']=max(summary['max_penetration_m'],-float(row['dist']))
                    summary['max_slip_m_s']=max(summary['max_slip_m_s'],row['slip_speed_m_s'])
                    summary['max_positive_tangential_power_W']=max(summary['max_positive_tangential_power_W'],row['tangential_power_W'])
                    if row['pyramidal_cone_utilization'] is not None:
                        summary['max_cone_utilization']=max(summary['max_cone_utilization'],row['pyramidal_cone_utilization'])
                    require(row['dim']==4,"Effective contact dimension changed")
                    require(np.allclose(row['friction'],[.6,.6,.005,.0001,.0001],rtol=1e-6,atol=1e-9),"Effective contact friction changed")
                    require(np.allclose(row['solref'],[.003,1.],rtol=1e-6,atol=1e-9),"Effective solref changed")
                    require(np.allclose(row['solimp'],[.95,.99,.001,.5,2.],rtol=1e-6,atol=1e-9),"Effective solimp changed")
                    require(-float(row['dist'])<=PENETRATION_TOL,"Penetration exceeds approved 1 mm gate")
                    if 'geometry' in row:
                        geo=row['geometry']
                        for name in ('surface_distance_m','sphere_gap_error_m','normal_error_deg'):
                            summary['max_'+name]=max(summary['max_'+name],geo[name])
                        summary['legacy_v1_failed_contact_count']+=not geo['legacy_v1_pass']
                        require(selected_geometry_pass(geo,gate_version),
                                f"Actual head contact {gate_version} geometry gate blocked at step {step}")
                if time.monotonic()-heartbeat>15:
                    emit(f"{backend}/{case['name']} step={step+1}/1500");heartbeat=time.monotonic()
        require(summary['head_contacts']>0,"No real head contact recorded")
        summary['status']='PASS'
    except Exception as exc:
        summary['status']='BLOCKED';summary['error']=str(exc)
        (destination/'error.txt').write_text(traceback.format_exc())
    finally:
        summary['wall_seconds']=time.monotonic()-started
        np.savez_compressed(destination/'states.npz',qpos=np.asarray(poses),qvel=np.asarray(velocities),dt=DT)
        save(destination/'summary.json',summary)
    emit(f"{backend}/{case['name']} {summary['status']} max_penetration_mm={summary['max_penetration_m']*1000:.4f}")
    return summary


def render_case(model,case_dir,target):
    """Render saved states; no new physical steps and no model/fixture edits."""
    require(shutil.which('ffmpeg') is not None,"ffmpeg is required for saved-state video")
    poses=np.load(case_dir/'states.npz',allow_pickle=False)['qpos']
    require(len(poses)>1,'No integrated states available for video')
    from PIL import Image, ImageDraw, ImageFont
    by_step={}
    with gzip.open(case_dir/'contacts.jsonl.gz','rt') as f:
        for line in f:
            row=json.loads(line);by_step.setdefault(row['step'],[]).append(row)
    width,height=960,720
    model.vis.global_.offwidth=max(model.vis.global_.offwidth,width)
    model.vis.global_.offheight=max(model.vis.global_.offheight,height)
    renderer=mujoco.Renderer(model,height=height,width=width)
    d=mujoco.MjData(model); camera=mujoco.MjvCamera(); camera.type=mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:]=poses[0,:3]-[0,0,.045];camera.distance=.42;camera.azimuth=130;camera.elevation=-22
    process=subprocess.Popen(['ffmpeg','-nostdin','-hide_banner','-loglevel','error','-f','rawvideo',
        '-pix_fmt','rgb24','-s',f'{width}x{height}','-r','25','-i','-','-an','-c:v','libx264',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(target)],stdin=subprocess.PIPE,stderr=subprocess.PIPE)
    frames=0
    frame_steps=sorted(set(range(0,len(poses)-1,20)) | {len(poses)-2,len(poses)-1})
    font=ImageFont.load_default(size=19)
    try:
        for step in frame_steps:
            d.qpos[:]=poses[step];mujoco.mj_forward(model,d);renderer.update_scene(d,camera=camera)
            picture=Image.fromarray(renderer.render());draw=ImageDraw.Draw(picture)
            phase='POST FINAL STEP; no stale contact overlay' if step==len(poses)-1 else 'PRE STEP'
            draw.rectangle((0,0,width,30),fill=(0,0,0))
            draw.text((8,5),f'FIXED MATERIAL | {case_dir.name} | t={step*DT:.3f}s | {phase}',
                      font=font,fill=(255,255,255))
            current=by_step.get(step,[])
            if current:
                text=' | '.join(f"{r['other']}: pen={max(0.,-r['dist'])*1000:.3f}mm slip={r['slip_speed_m_s']:.4f}m/s n="+
                              ','.join(f'{x:+.3f}' for x in r['normal_toward_ball']) for r in current)
                draw.rectangle((0,height-30,width,height),fill=(0,0,0))
                draw.text((8,height-25),text,font=font,fill=(255,255,255))
            process.stdin.write(np.asarray(picture).tobytes());frames+=1
        process.stdin.close();errors=process.stderr.read().decode(errors='replace');code=process.wait(timeout=30)
        require(code==0 and frames>0,'Video encoding failed: '+errors)
    finally:
        renderer.close()
        if process.poll() is None: process.kill();process.wait()
    require(subprocess.run(['ffmpeg','-nostdin','-v','error','-i',str(target),'-f','null','-'],capture_output=True).returncode==0,'Video decode failed')
    return {"file":target.name,"sha256":sha256(target),"frames":frames,"fps":25,
            "fixed_material_fixture":True,"human_visual_review":"PENDING","normal_overlay":"World-space numeric normal caption",
            "no_new_simulation_steps":True,"frame_steps":frame_steps,
            "frame_times_s":[step*DT for step in frame_steps],
            "last_two_frames":"Final pre-step and post-step states explicitly included; use timestamp captions"}


def run_physics(old,new,out,*,warp_device='cuda:0',cpu_only=False,static_only=False,ls_parallel=True,
                wall_seconds=2700,render=True,geometry_gate='v1'):
    require(old.opt.timestep==new.opt.timestep==DT,'Frozen 2 ms timestep required')
    report={"status":"RUNNING","new_ppo_updates":0,"cloud_gpu_hours":0,
            "policy_objective":"NOT_EVALUATED_PHYSICS_ONLY","formal_task_migration":False,
            "fixed_head_only_in_material_fixture":True,"full_robot_head_neck_locked":False,
            "old_10s_terminal_5s_unassisted_015mps_contract_unchanged":True,
            "B1_C1_gaps_preserved":True,"old_stage10_tag_unchanged":True,
            "cloud_status":"OFF_CONFIRMED_BY_USER","stage09_cloud_data_delete_allowed":False,
            "cases":[],"videos":[],"gpu_physics":"NOT_RUN","geometry_gate_version":geometry_gate}
    started=time.monotonic();deadline=started+wall_seconds
    try:
        structure=audit_full_scene(old,new);save(out/'structure-inertia.json',structure)
        require(structure['status']=='PASS','Structure/inertia audit blocked; inspect structure-inertia.json')
        source_data=initial_data(new);old_data=initial_data(old)
        sites=[i for i in range(old.nsite) if (old.site(i).name or '').endswith('double_balance_tray_frame')]
        require(len(sites)==1,'Old reference tray site is ambiguous')
        center_xy=old_data.site_xpos[sites[0],:2].copy()
        geometry,material,triangles=static_geometry(new,source_data,center_xy,out,geometry_gate)
        report['static_geometry_status']=geometry['status']
        report['legacy_v1_static_status']=geometry.get('legacy_v1_status',geometry['status'])
        save(out/'material-options.json',options_dict(material))
        save(out/'material-scope.json',{"fixed_head":True,"gravity":material.opt.gravity,
             "dt":DT,"sphere_radius_m":RADIUS,"normal_compliance_note":"solref[0]=0.003; refsafe may clamp internal normal time constant to 2*dt; flags are recorded, not changed",
             "friction_is_existing_simulation_assumption_not_measured_material":True})
        # MjSpec.compile does not populate mj_saveLastXML's process-global XML.
        # Save the exact compiled model, including copied source solver options.
        mujoco.mj_saveModel(material,str(out/'material-fixture.mjb'))
        require(geometry['status']=='PASS','Original mesh / compiled collision geometry gate blocked')
        if static_only:
            report['status']='STATIC_ONLY_PENDING_DYNAMIC_REVIEW'
            return report
        cases=dynamic_matrix(triangles,center_xy);require(len(cases)==13,'Unexpected matrix size')
        for backend in ('cpu',) if cpu_only else ('cpu','warp'):
            if backend=='warp':
                import warp as wp
                require(wp.get_device(warp_device).is_cuda,'Laptop validation requires a CUDA device; CPU Warp is not RTX evidence')
                report['gpu_physics']='RUNNING'
            for case in cases:
                summary=run_case(material,case,backend,out,warp_device=warp_device,deadline=deadline,
                                 ls_parallel=ls_parallel,triangles=triangles,gate_version=geometry_gate)
                report['cases'].append(summary);save(out/'physics-review.json',report)
                if summary['status']!='PASS' and render and summary['steps']>0:
                    try:
                        target=out/f"material-{backend}-{case['name']}-BLOCKED.mp4"
                        report['videos'].append(render_case(material,out/backend/case['name'],target))
                    except Exception as exc:
                        report['videos'].append({'render_error':str(exc),'case':case['name'],
                                                 'physical_failure_preserved':True})
                require(summary['status']=='PASS',f"{backend}/{case['name']} blocked; no automatic retry or parameter change")
            if backend=='warp': report['gpu_physics']='PASS'
        report['status']='CPU_ONLY_PENDING_GPU' if cpu_only else 'PHYSICS_GATES_PASS_PENDING_VIDEO_REVIEW'
        if render:
            chosen=[('cpu','release-center')]
            if not cpu_only:
                chosen += [('warp','release-center'),('warp','slide-x-plus')]
                worst=max((c for c in report['cases'] if c['backend']=='warp'),key=lambda c:c['max_penetration_m'])
                chosen += [('warp',worst['case'])]
            for backend,name in dict.fromkeys(chosen):
                target=out/f'material-{backend}-{name}.mp4'
                report['videos'].append(render_case(material,out/backend/name,target))
    except Exception as exc:
        report['status']='BLOCKED';report['error']=str(exc)
        (out/'error.txt').write_text(traceback.format_exc())
    finally:
        report['wall_seconds']=time.monotonic()-started
        report['stage11_complete']=False
        save(out/'physics-review.json',report)
    return report


def local_preflight(out):
    actual={name:importlib.metadata.version(name) for name in ('mujoco','mujoco-warp','mjlab','warp-lang')}
    expected={'mujoco':'3.10.0','mujoco-warp':'3.8.1','mjlab':'1.3.0','warp-lang':'1.12.0'}
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    status=subprocess.check_output(['git','status','--short','--branch'],cwd=ROOT,text=True)
    result={'head':head,'versions':actual,'expected_versions':expected,'working_tree':status,
            'python':sys.version,'new_ppo_updates':0,'network_used_by_validator':False}
    save(out/'preflight.json',result)
    require(actual==expected,'Runtime versions changed; preserve this report and stop')
    # A locally imported Stage 11 commit may follow the reviewed baseline, but
    # all baseline-tracked files must remain byte-identical. Only new files are
    # permitted; this includes uncommitted changes in the worktree/index.
    require(subprocess.run(['git','merge-base','--is-ancestor',BASELINE,'HEAD'],cwd=ROOT).returncode==0,
            'HEAD does not descend from the reviewed baseline')
    changed=subprocess.check_output(['git','diff','--name-status',BASELINE,'--'],cwd=ROOT,text=True)
    invalid=[line for line in changed.splitlines() if not line.startswith('A\t')]
    require(not invalid,'A baseline file was changed; stop: '+repr(invalid))
    flags={k:v for k,v in os.environ.items() if k.startswith('MICRODUCK_')}
    save(out/'environment-overrides.json',flags)
    require(not flags,'MICRODUCK environment overrides are present; inspect before using frozen parameters')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--warp-device',default='cuda:0')
    parser.add_argument('--cpu-only',action='store_true')
    parser.add_argument('--static-only',action='store_true',
                        help='Compile and inspect only; no CPU/GPU dynamics')
    parser.add_argument('--no-video',action='store_true')
    parser.add_argument('--geometry-gate',choices=GEOMETRY_GATE_VERSIONS,default='v1',
                        help='v2 requires the explicitly approved surface/gap/normal revision')
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    try:
        local_preflight(out)
        from mjlab.scene import Scene
        cfg,old_cfg,new_cfg=physics_scene_pair()
        old=Scene(old_cfg,device='cpu').compile();new=Scene(new_cfg,device='cpu').compile()
        cfg.sim.mujoco.apply(old);cfg.sim.mujoco.apply(new)
        report=run_physics(old,new,out,warp_device=args.warp_device,cpu_only=args.cpu_only,
                           static_only=args.static_only,ls_parallel=cfg.sim.ls_parallel,
                           render=not args.no_video,geometry_gate=args.geometry_gate)
    except Exception as exc:
        report={'status':'BLOCKED','error':str(exc),'stage11_complete':False,'new_ppo_updates':0,
                'policy_objective':'NOT_EVALUATED_PHYSICS_ONLY'}
        save(out/'physics-review.json',report);(out/'error.txt').write_text(traceback.format_exc())
    emit('Stage11Physics='+report['status']);emit('Stage11Evidence='+str(out))
    return 0 if report['status'] in ('STATIC_ONLY_PENDING_DYNAMIC_REVIEW','CPU_ONLY_PENDING_GPU','PHYSICS_GATES_PASS_PENDING_VIDEO_REVIEW') else 2


if __name__=='__main__': raise SystemExit(main())
