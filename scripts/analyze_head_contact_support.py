"""Offline interpretation of pinned saved material traces; never runs physics.

This is a diagnostic, not a new task, reward, crown boundary, success predicate,
or timing implementation. Pre-solver contacts and post-state geometry stay apart.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path
import time
import zipfile

import numpy as np

FRICTION_SHA = "f692ff89cfdd0b1036570a2eb5cc4a7f679eb49bc407af37e5e3410478dd343d"
PROTOTYPE_SHA = "c22ab08e67979c33f397ee099e68d816eb2877663d4d3cae2419176dfa017e4d"
ARTIFACTS = Path("/home/lx/microduck-double-balance/artifacts/double-balance-stage11")
CASES = ("slide-x-plus", "slide-x-minus", "slide-y-plus", "slide-y-minus", "initial-no-slip-roll", "normal-spin")
PHASE = "pre_integration_contacts_and_solver_force"


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


class PinnedZip:
    def __init__(self, path, expected):
        blob=Path(path).read_bytes()
        require(hashlib.sha256(blob).hexdigest()==expected, "Archive SHA256 mismatch: " + str(path))
        self.zip=zipfile.ZipFile(io.BytesIO(blob)); self.sha=expected; self.used={}
        require(len(self.zip.namelist())==len(set(self.zip.namelist())), "Duplicate archive member")
        self.inventory=json.loads(self.zip.read("evidence-inventory.json"))

    def read(self, name):
        blob=self.zip.read(name); expected=self.inventory[name]
        sha=hashlib.sha256(blob).hexdigest()
        require(sha==expected["sha256"] and len(blob)==expected["bytes"], "Archived member mismatch: " + name)
        self.used[name]={"sha256":sha,"bytes":len(blob)}
        return blob

    def json(self, name):
        return json.loads(self.read(name))

    def lines(self, name):
        return [json.loads(s) for s in gzip.decompress(self.read(name)).decode().splitlines()]


def rotation(q):
    q=np.asarray(q,float)
    require(q.shape==(4,) and np.isfinite(q).all() and np.linalg.norm(q)>0, "Invalid quaternion")
    w,x,y,z=q/np.linalg.norm(q)
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def nearest_distance(point, triangles):
    """Independent exact triangle/edge distance; not a signed solid distance."""
    a,b,c=triangles[:,0],triangles[:,1],triangles[:,2]
    ab,ac=b-a,c-a; normal=np.cross(ab,ac); nn=np.sum(normal*normal,axis=1)
    projected=point-normal*np.divide(np.sum((point-a)*normal,axis=1),nn,out=np.zeros(len(nn)),where=nn>1e-30)[:,None]
    aa=np.sum(ab*ab,axis=1);bb=np.sum(ac*ac,axis=1);cc=np.sum(ab*ac,axis=1)
    x=np.sum((projected-a)*ab,axis=1);y=np.sum((projected-a)*ac,axis=1);den=aa*bb-cc*cc
    u=np.divide(bb*x-cc*y,den,out=np.zeros_like(den),where=np.abs(den)>1e-30)
    v=np.divide(aa*y-cc*x,den,out=np.zeros_like(den),where=np.abs(den)>1e-30)
    inside=(nn>1e-30)&(u>=0)&(v>=0)&(u+v<=1)
    d2=np.where(inside,np.sum((point-projected)**2,axis=1),np.inf)
    for start,end in ((a,b),(b,c),(c,a)):
        edge=end-start; ll=np.sum(edge*edge,axis=1)
        t=np.clip(np.divide(np.sum((point-start)*edge,axis=1),ll,out=np.zeros(len(ll)),where=ll>0),0,1)
        cp=start+t[:,None]*edge;d2=np.minimum(d2,np.sum((point-cp)**2,axis=1))
    return float(np.sqrt(d2.min()))


def upper_kinematic_sample(position, rate):
    """Descriptive reuse of old three upper thresholds; no lower stack tested."""
    p,v=np.asarray(position),np.asarray(rate)
    require(np.isfinite(p).all() and np.isfinite(v).all(), "Nonfinite kinematic sample")
    return bool(np.linalg.norm(p[:2])<=.012 and abs(p[2]-.02)<=.006 and np.linalg.norm(v)<=.08)


def contact_facts(row, velocity, omega, center):
    """A force-bearing contact is a fact, not equilibrium or policy success."""
    require(row["phase"]==PHASE, "Pre/post force phase mismatch")
    frame=np.asarray(row["frame"],float).reshape(3,3)
    n=np.asarray(row["normal_toward_ball"],float)
    force=np.asarray(row["force"],float);pos=np.asarray(row["pos"],float)
    require(np.isfinite(np.r_[frame.ravel(),n,force,pos,row["dist"]]).all(), "Nonfinite contact")
    require(abs(np.linalg.norm(n)-1.)<1e-6, "Nonunit contact normal")
    sign=1. if np.dot(frame[0],n)>0 else -1.
    require(np.linalg.norm(sign*frame[0]-n)<1e-8, "Inconsistent oriented normal")
    on_ball=sign*frame.T@force[:3]
    point_velocity=np.asarray(velocity)+np.cross(omega,pos-center)
    slip=point_velocity-n*(point_velocity@n)
    require(abs(force[0]-row["normal_force_N"])<1e-12, "Saved normal force mismatch")
    require(np.max(np.abs(slip-row["slip_world_m_s"]))<1e-10, "Independent slip differs")
    active=bool(row["dist"]<=0 and force[0]>0)
    return {"active":active,"normal_force_N":float(force[0]),"vertical_force_on_ball_N":float(on_ball[2]),
            "world_up_normal":bool(n[2]>0),"slip_speed_m_s":float(np.linalg.norm(slip)),
            "slip_residual":float(np.max(np.abs(slip-row["slip_world_m_s"]))) }


def analyze(friction, prototype, out, deadline):
    prior=friction.json("run-summary.json");proto=prototype.json("run-summary.json")
    require(prior["physics_integration_steps"]==1684 and proto["physics_integration_steps"]==0, "Unexpected source runs")
    require(prior["canonical_total_known_completed_steps"]==proto["canonical_total_known_completed_steps"]==7895, "Unexpected cumulative count")
    require(proto["execution_invariants_unchanged"] and not proto["formal_task_migration"], "Wrong prototype provenance")
    poses=prototype.json("prototype/prototype-static-configurations.json")
    home=next(p for p in poses if p["pose"]=="home");R=np.asarray(home["rotation"]);origin=np.asarray(home["origin"])
    with np.load(io.BytesIO(friction.read("diagnostic/material-geometry.npz")),allow_pickle=False) as z:
        vertices,faces=z["vertices"].copy(),z["faces"].copy()
    triangles=vertices[faces]
    max_mesh_normal_coordinate=float(((vertices-origin)@R)[:,2].max())
    # Geometry proof: accepted height slab lies strictly above all mesh vertices.
    require(max_mesh_normal_coordinate<.014, "Cannot infer outside-solid clearance in the upper height slab")
    summaries=[];total=0;all_metrics=[]
    for name in CASES:
        start="diagnostic/cases/"+name+"/"
        rows=friction.lines(start+"step-log.jsonl.gz"); raw=friction.lines(start+"raw-contacts.jsonl.gz")
        previous=friction.json(start+"summary.json")
        with np.load(io.BytesIO(friction.read(start+"states.npz")),allow_pickle=False) as z:
            qpos,qvel,times=z["qpos"].copy(),z["qvel"].copy(),z["time"].copy()
            dt=float(z["dt"])
        require(dt==.002 and len(rows)==len(raw)==len(qpos)-1==previous["steps"], "Step/state coverage differs")
        require(not previous["horizon_completed"], "Historical endpoint unexpectedly reclassified")
        counts={k:0 for k in ("pre_head_geometry","pre_active_head_contact","pre_active_upward_head_force","upper_kinematic_only","upper_kinematic_without_active_head","upper_kinematic_with_active_head","pre_post_head_geometry_disagreement","post_floor_overlap","head_contact_without_positive_force")}
        gaps=[];mismatch=[];maxslip=0.;upper_force_eps={str(e):0 for e in (0.,1e-8,1e-6)}
        for i,(row,raw_row) in enumerate(zip(rows,raw,strict=True)):
            require(time.monotonic()<deadline, "Saved-data analysis wall budget exceeded")
            require(row["step"]==raw_row["step"]==i and raw_row["phase"]==PHASE, "Phase/index mismatch")
            np.testing.assert_allclose(row["qpos_before"],qpos[i],rtol=0,atol=0)
            np.testing.assert_allclose(row["qvel_before"],qvel[i],rtol=0,atol=0)
            np.testing.assert_allclose(row["qpos_after"],qpos[i+1],rtol=0,atol=0)
            require(abs(row["time_s"]-times[i])<1e-12, "Pre-state time mismatch")
            center=qpos[i,:3];velocity=qvel[i,:3];omega=rotation(qpos[i,3:7])@qvel[i,3:]
            position=R.T@(center-origin);rate=R.T@velocity
            kinematic=upper_kinematic_sample(position,rate)
            contacts=row["contacts"];require(len(contacts)==len(raw_row["contacts"]),"Raw contact coverage differs")
            facts=[]
            for c,raw_contact in zip(contacts,raw_row["contacts"],strict=True):
                require(c["other"] in ("head","floor"),"Unexpected contact category")
                for key in ("geom","dist","pos","frame","force"):
                    require(c[key]==raw_contact[key],"Derived contact differs from raw record: "+key)
                f=contact_facts(c,velocity,omega,center)
                maxslip=max(maxslip,f["slip_residual"])
                if c["other"]=="head": facts.append(f)
            active=any(f["active"] for f in facts)
            upward=any(f["active"] and f["vertical_force_on_ball_N"]>0 for f in facts)
            post=row["post_state_geometry"]
            require(post["phase"]=="post_integration_geometry_only_no_force_claim" and abs(post["time_s"]-times[i+1])<1e-12,"Post geometry phase mismatch")
            post_head=any(c["other"]=="head" for c in post["contacts"])
            floor_overlap=post["floor_overlap_m"]>0
            flags={"pre_head_geometry":bool(facts),"pre_active_head_contact":active,"pre_active_upward_head_force":upward,
                   "upper_kinematic_only":kinematic,"upper_kinematic_without_active_head":kinematic and not active,
                   "upper_kinematic_with_active_head":kinematic and active,
                   "pre_post_head_geometry_disagreement":bool(facts)!=post_head,"post_floor_overlap":floor_overlap,
                   "head_contact_without_positive_force":bool(facts) and not active}
            for key,value in flags.items():counts[key]+=int(value)
            for eps in (0.,1e-8,1e-6):
                upper_force_eps[str(eps)]+=int(kinematic and any(f["active"] and f["normal_force_N"]>eps for f in facts))
            gap=None;gap_bias=None
            if kinematic:
                # Entire shell is below this sample in material Z; outside solid.
                gap=nearest_distance(center,triangles)-.02
                gap_bias=gap-(position[2]-.02);gaps.append(gap_bias)
            if kinematic and not active:mismatch.append(i)
            all_metrics.append({"case":name,"pre_step":i,"pre_time_s":float(times[i]),"post_time_s":float(times[i+1]),
                                **flags,"position_material_m":position.tolist(),"rate_material_m_s":rate.tolist(),
                                "outside_surface_gap_if_upper_kinematic_m":gap,"gap_minus_tangent_height_if_upper_kinematic_m":gap_bias,
                                "post_force_status":"NOT_RECORDED; do not reuse pre force"})
        require(counts["pre_head_geometry"]==previous["head_contact_steps"],"Saved head count differs")
        total+=len(rows)
        summaries.append({"case":name,"source_steps":len(rows),"source_end_status":previous["status"],"counts":counts,
                          "kinematic_without_active_head_steps":mismatch,"normal_force_threshold_sensitivity":upper_force_eps,
                          "maximum_slip_recompute_residual_m_s":maxslip,"max_abs_curved_gap_minus_tangent_height_when_kinematic_m":max(map(abs,gaps),default=None),
                          "fixed_head_material_not_policy":True,"horizon_completed":False})
    require(total==1684,"Missing saved steps")
    with gzip.open(out/"sample-metrics.jsonl.gz","wt") as f:
        for row in all_metrics:f.write(json.dumps(row,allow_nan=False)+"\n")
    return {"source_steps_read":total,"new_integration_steps":0,"new_ppo_updates":0,
            "mesh_max_material_z_m":max_mesh_normal_coordinate,"upper_height_slab_min_material_z_m":.014,
            "cases":summaries,"phase_contract":"All forces, position, slip and descriptive kinematic gates use the SAME saved pre state; post geometry is reported separately.",
            "formal_success_predicate":None,"physical_crown_boundary":None,
            "interpretation":"Kinematic-only samples are not evidence of head contact. Positive contact force is not equilibrium, load retention, lower-ball balance or 5-second terminal success."}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--saved-data-only",action="store_true",required=True)
    parser.add_argument("--friction-zip",type=Path,default=ARTIFACTS/"head-friction/20261003T183812Z.zip")
    parser.add_argument("--prototype-zip",type=Path,default=ARTIFACTS/"head-prototype/20261003T205659Z.zip")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    start=time.monotonic()
    report={"status":"SAVED_SUPPORT_REVIEW_REQUIRED","stage11_status":"MATERIAL_BLOCKED_REVIEW_REQUIRED",
            "stage11_complete":False,"new_integration_steps":0,"new_ppo_updates":0,"cloud_hours":0,
            "canonical_total_completed_steps":7895,"formal_task_migration":False,"policy_success":None}
    try:
        friction=PinnedZip(args.friction_zip,FRICTION_SHA);prototype=PinnedZip(args.prototype_zip,PROTOTYPE_SHA)
        report["analysis"]=analyze(friction,prototype,out,start+300)
        save(out/"input-provenance.json",{"friction":{"sha256":friction.sha,"files":friction.used},"prototype":{"sha256":prototype.sha,"files":prototype.used}})
    except Exception as exc:
        import traceback
        report.update(status="SAVED_SUPPORT_EXECUTION_BLOCKED",error=str(exc))
        (out/"error.txt").write_text(traceback.format_exc())
    report["wall_seconds"]=time.monotonic()-start
    save(out/"support-review.json",report)
    print("Stage11SupportResult="+report["status"],flush=True)
    return 0 if report["status"]=="SAVED_SUPPORT_REVIEW_REQUIRED" else 2


if __name__=="__main__":raise SystemExit(main())
