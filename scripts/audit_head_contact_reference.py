"""Stage 11 reference design audit: compile/forward/distance only; no dynamics.

Seven articulated head configurations and seven inherited material locations.
No task registration, selected reset, reward change, control action or training.
"""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import importlib.util
import json
from pathlib import Path
import time
import traceback
from unittest.mock import patch

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location('stage11_reference_frozen_friction', ROOT/'scripts/diagnose_head_contact_friction.py')
friction = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(friction)
floor, v2 = friction.floor, friction.v2
require, save, digest = v2.require, v2.save, floor.digest
from mjlab_microduck.robot.head_contact import rotation

PRIOR = Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage11/head-friction/20261003T183812Z')
PRIOR_COMMIT = '7c3157818500b357e733a22ae6df54ce3fbcf7b4'
PRIOR_ZIP_SHA = 'f692ff89cfdd0b1036570a2eb5cc4a7f679eb49bc407af37e5e3410478dd343d'
FROZEN = {**friction.FROZEN, 'scripts/diagnose_head_contact_friction.py':'e787f4e68ae20880b2f619dd80b01d80363308bc356b6f434e29b0f4739a9794'}
POSES = (('home',None,0.), ('yaw-minus90','head_yaw',-90.), ('yaw-plus90','head_yaw',90.),
         ('pitch-minus10','head_pitch',-10.), ('pitch-plus10','head_pitch',10.),
         ('roll-minus10','head_roll',-10.), ('roll-plus10','head_roll',10.))
FORBIDDEN = ('mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta','mj_implicit','mj_integratePos')
PRIOR_FILES = ('run-summary.json','diagnostic/preflight.json','diagnostic/diagnostic-review.json',
               'diagnostic/input-provenance.json','diagnostic/material-fixture.mjb',
               'diagnostic/material-geometry.npz','diagnostic/cases/normal-spin/initial.json')


@contextmanager
def zero_integration_guard():
    """Guard runs before model construction and restores APIs even on failure."""
    attempts = []
    stack = []
    def forbidden(name):
        def stop(*args, **kwargs):
            attempts.append(name)
            raise RuntimeError('ZERO_INTEGRATION_GUARD: '+name+' is forbidden')
        return stop
    try:
        for name in FORBIDDEN:
            if hasattr(mujoco,name):
                p = patch.object(mujoco,name,forbidden(name)); p.start(); stack.append(p)
        yield attempts
    finally:
        for p in reversed(stack): p.stop()


def verify_prior(current):
    files = friction.pinned_files(PRIOR, PRIOR_ZIP_SHA, PRIOR_FILES)
    for name,sha in FROZEN.items(): require(digest(ROOT/name)==sha,'Frozen source changed: '+name)
    run = floor.load_json(PRIOR/'run-summary.json')
    old = floor.load_json(PRIOR/'diagnostic/preflight.json')
    require(all(current[k]==old[k] for k in ('head','versions','python')),'Runtime differs from returned friction evidence')
    require(run['head']==run['head_after']==v2.BASELINE and run['execution_invariants_unchanged'], 'Prior baseline/invariants differ')
    require(run['physics_integration_steps']==run['physics_attempted_steps']==1684 and
            run['canonical_total_known_completed_steps']==7895 and not run['stage11_complete'], 'Unexpected prior count/status')
    require(run['local_diagnostic_status']=='HEAD_FRICTION_FLOOR_ENDPOINTS_REVIEW_REQUIRED', 'Prior floor blockers must remain')
    require(files['diagnostic/material-fixture.mjb']['sha256']=='e6cd330bddc3a29abf9ce4eca4f308a2680e4910405725e5ce054cb0a19d2da0', 'Wrong original MJB')
    return {'artifact_commit':PRIOR_COMMIT,'zip_sha256':PRIOR_ZIP_SHA,'files':files,'frozen_sources':FROZEN,
            'prior_canonical_steps':7895,'old_diagnostics_replayed':False}


def joint_id(model, suffix):
    ids = [i for i in range(model.njnt) if (model.joint(i).name or '').split('/')[-1]==suffix]
    require(len(ids)==1,'Ambiguous joint: '+suffix)
    return ids[0]


def pose_data(model, home_qpos, spec):
    """Only data qpos changes: joint increments relative to HOME, no ctrl drive."""
    name,joint,degrees = spec
    require(spec in POSES,'Undeclared pose')
    data = mujoco.MjData(model); data.qpos[:] = home_qpos
    if joint:
        j = joint_id(model,joint); a = int(model.jnt_qposadr[j])
        require(model.jnt_type[j]==mujoco.mjtJoint.mjJNT_HINGE,'Expected head hinge')
        data.qpos[a] += np.radians(degrees)
        require(not model.jnt_limited[j] or model.jnt_range[j,0]<=data.qpos[a]<=model.jnt_range[j,1], 'Requested pose exceeds physical joint range')
    before = data.qpos.copy(); mujoco.mj_forward(model,data)
    require(data.time==0 and np.array_equal(data.qpos,before) and not np.any(data.qvel),'Forward mutated configuration/time')
    return data


def material_coordinates(position,velocity,origin,origin_velocity,R,omega):
    r = np.asarray(position)-origin
    return R.T@r, R.T@(np.asarray(velocity)-origin_velocity-np.cross(omega,r))


def boundary_inventory(vertices,faces):
    """Weld exactly coincident vertices; never label AABB as a physical rim."""
    unique,inverse = np.unique(vertices,axis=0,return_inverse=True)
    f = inverse[faces]
    edges = np.sort(np.concatenate((f[:,[0,1]],f[:,[1,2]],f[:,[2,0]])),axis=1)
    edges = edges[edges[:,0]!=edges[:,1]]
    e,counts = np.unique(edges,axis=0,return_counts=True)
    boundary = unique[e[counts==1]]
    return boundary, {'exact_welded_vertices':len(unique),'open_edge_count':len(boundary),
                       'nonmanifold_edge_count':int(np.sum(counts>2)),
                       'definition':'Topological open edges after exact-coordinate welding; closed shell rims are NOT such boundaries.'}


def boundary_distances(point,vertices,edges):
    point = np.asarray(point)
    margin = np.minimum(point[:2]-vertices[:,:2].min(0),vertices[:,:2].max(0)-point[:2])
    distance = None
    if len(edges):
        a,b = edges[:,0],edges[:,1]; ab=b-a
        u=np.clip(np.einsum('ij,ij->i',point-a,ab)/np.einsum('ij,ij->i',ab,ab),0,1)
        distance=float(np.min(np.linalg.norm(point-(a+u[:,None]*ab),axis=1)))
    return {'topological_open_edge_distance_m':distance,'home_xy_aabb_margin_m':margin,
            'minimum_home_xy_aabb_margin_m':float(margin.min()),
            'physical_rim_distance_m':None,'physical_rim_status':'NOT_SEMANTICALLY_DEFINED; AABB margin is only an extent diagnostic'}


def place_and_measure(model,pose,center,triangles):
    data=mujoco.MjData(model);data.qpos[:]=pose.qpos
    top=v2.gid(model,v2.TOP_GEOM); head=v2.gid(model,v2.HEAD_GEOM_NAME)
    body=int(model.geom_bodyid[top]); js=np.flatnonzero(model.jnt_bodyid==body)
    require(len(js)==1 and model.jnt_type[js[0]]==mujoco.mjtJoint.mjJNT_FREE,'Top sphere must remain free')
    a=int(model.jnt_qposadr[js[0]]);data.qpos[a:a+3]=center;data.qpos[a+3:a+7]=[1,0,0,0]
    original=data.qpos.copy();mujoco.mj_forward(model,data)
    require(data.time==0 and np.array_equal(original,data.qpos) and not np.any(data.qvel),'Static placement integrated')
    witness=np.full(6,np.nan)
    gap=float(mujoco.mj_geomDistance(model,data,head,top,1.,witness))
    vec=witness[3:]-witness[:3];length=float(np.linalg.norm(vec))
    require(length>1e-12 and np.isfinite(witness).all(),'Invalid engine distance witness')
    metrics=v2.geometry_metrics(center,witness[:3],vec/length,gap,triangles)
    require(metrics['original_sphere_gap_m']>=-1e-8,'Proposed static center intersects original surface')
    return metrics


def derivative_audit(model,data,head_body,local_origin,local_point):
    """Jacobian plus central configuration samples; no time stepping of any kind."""
    R=data.xmat[head_body].reshape(3,3);p=data.xpos[head_body]
    o=p+R@local_origin;ball=p+R@local_point;r=ball-o
    jp=np.zeros((3,model.nv));jr=jp.copy()
    mujoco.mj_jac(model,data,jp,jr,o,head_body)
    j=joint_id(model,'head_yaw'); velocity=np.zeros(model.nv);velocity[int(model.jnt_dofadr[j])]=1.
    vo=jp@velocity;omega=jr@velocity; vb=vo+np.cross(omega,r)
    pos,relative=material_coordinates(ball,vb,o,vo,R,omega)
    legacy=R.T@(vb-vo)
    eps=1e-6; positions=[];origins=[];frames=[]
    for sign in (-1,1):
        d=mujoco.MjData(model);d.qpos[:]=data.qpos;d.qpos[int(model.jnt_qposadr[j])]+=sign*eps
        mujoco.mj_forward(model,d)
        rr=d.xmat[head_body].reshape(3,3);oo=d.xpos[head_body]+rr@local_origin
        positions.append(d.xpos[head_body]+rr@local_point);origins.append(oo);frames.append(rr)
        require(d.time==0,'Configuration sample advanced time')
    finite_v=(positions[1]-positions[0])/(2*eps)
    finite_vo=(origins[1]-origins[0])/(2*eps)
    fixed_world_fd=(frames[1].T@(ball-origins[1])-frames[0].T@(ball-origins[0]))/(2*eps)
    _,fixed_world_rate=material_coordinates(ball,np.zeros(3),o,vo,R,omega)
    errors=[np.linalg.norm(finite_v-vb),np.linalg.norm(finite_vo-vo),np.linalg.norm(fixed_world_fd-fixed_world_rate)]
    require(max(errors)<1e-8 and np.linalg.norm(relative)<1e-12,'Rotating-frame derivative identity failed')
    return {'virtual_head_yaw_rate_rad_s':1.,'legacy_velocity_for_head_carried_ball_m_s':legacy,
            'rotating_coordinates_derivative_m_s':relative,'head_carried_ball_position_in_reference_m':pos,
            'finite_configuration_difference_errors_m_s':errors,'sample_delta_rad':eps,
            'interpretation':'Kinematic test vectors only; not a new reset, joint controller or physical trajectory.'}


def audit_models(old,new,geometry,out,deadline):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    oldsig,newsig=floor.model_signature(old),floor.model_signature(new)
    structure=v2.audit_full_scene(old,new);save(out/'structure-inertia.json',structure)
    require(structure['status']=='PASS','Original independent factory structure/inertia check failed')
    od,home=v2.initial_data(old),v2.initial_data(new)
    head=v2.gid(new,v2.HEAD_GEOM_NAME);jaw=v2.bid(new,'jaw_soft')
    verts,faces=v2.mesh_triangles(new,home,head)
    require(np.array_equal(faces,geometry['faces']) and np.max(np.abs(verts-geometry['vertices']))<1e-10, 'Compiled HOME mesh differs from canonical returned geometry')
    site=next(i for i in range(old.nsite) if (old.site(i).name or '').endswith('double_balance_tray_frame'))
    center_xy=od.site_xpos[site,:2].copy()
    require(np.max(np.abs(center_xy-geometry['center_xy']))<1e-10,'Old tray XY differs from archived material center')
    H=home.xmat[jaw].reshape(3,3);p=home.xpos[jaw].copy();triangles=verts[faces]
    edges,boundaries=boundary_inventory(verts,faces)
    save(out/'mesh-boundary-definition.json',boundaries)
    points=[]
    for name,dx,dy in v2.point_matrix():
        c,w,n,f=v2.sphere_center_above(center_xy+[dx,dy],triangles)
        points.append({'name':name,'center':c,'witness':w,'normal':n,'face':f,
                       'center_in_jaw':H.T@(c-p),'witness_in_jaw':H.T@(w-p),
                       'normal_in_jaw':H.T@n,'boundary':boundary_distances(w,verts,edges)})
    highest=verts[np.argmax(verts[:,2])]
    c,w,n,f=v2.sphere_center_above(highest[:2],triangles)
    candidates={'inherited_seven_locations':points,'highest_vertex_only':{'center':c,'witness':w,'normal':n,
                  'shift_xy_from_legacy_center_m':c[:2]-center_xy,'boundary':boundary_distances(w,verts,edges)},
                'selected_reference':None,'selection_status':'DESIGN_REVIEW_REQUIRED',
                'highest_point_warning':'Near positive-X mesh extent; no proof of useful crown region or passive stability.'}
    save(out/'candidate-references.json',candidates)
    rows=[];poses=[];qstates=[]
    for pose_spec in POSES:
        require(time.monotonic()<deadline,'Static audit wall limit reached')
        data=pose_data(new,home.qpos,pose_spec); R=data.xmat[jaw].reshape(3,3);origin=data.xpos[jaw].copy()
        vv,ff=v2.mesh_triangles(new,data,head);tt=vv[ff]
        expected=(verts-p)@H@R.T+origin
        require(np.max(np.abs(vv-expected))<1e-10,'Articulated head mesh is not the same rigid shell')
        poselog={'name':pose_spec[0],'joint':pose_spec[1],'delta_from_HOME_deg':pose_spec[2],
                 'head_origin_world_m':origin,'head_rotation_world':R,'actual_rotation_from_HOME':R@H.T,
                 'qpos':data.qpos.copy(),'qvel':data.qvel.copy(),'time_s':data.time,
                 'derivative':derivative_audit(new,data,jaw,points[0]['witness_in_jaw'],points[-1]['center_in_jaw'])}
        poses.append(poselog);qstates.append(data.qpos.copy())
        for point in points:
            require(time.monotonic()<deadline,'Static audit wall limit reached')
            carried=origin+R@point['center_in_jaw']; expected_w=origin+R@point['witness_in_jaw']
            expected_n=R@point['normal_in_jaw']
            metrics=place_and_measure(new,data,carried,tt)
            require(np.linalg.norm(metrics['mesh_witness']-expected_w)<1e-7,'Carried witness covariance failed')
            row={'pose':pose_spec[0],'point':point['name'],'carried_center_world_m':carried,
                 'carried_witness_world_m':expected_w,'carried_normal_world':expected_n,
                 'normal_tilt_from_world_up_deg':float(np.degrees(np.arccos(np.clip(expected_n[2],-1,1)))),
                 'carried_geometry':metrics,'boundary_in_HOME':point['boundary']}
            # Alternative geometric placement is diagnostic only, never chosen as reset.
            try:
                c2,w2,n2,f2=v2.sphere_center_above(expected_w[:2],tt)
            except ValueError as exc:
                row['vertical_replacement']={'status':'OUTSIDE_PROJECTED_SHELL','reason':str(exc)}
            else:
                g2=place_and_measure(new,data,c2,tt)
                row['vertical_replacement']={'status':'MEASURED','xy_definition':'World XY of the carried SURFACE witness, not the carried sphere center',
                    'requested_xy_m':expected_w[:2],'center_world_m':c2,'witness_world_m':w2,
                    'normal_world':n2,'face':f2,'delta_from_carried_center_m':c2-carried,
                    'witness_shift_from_carried_m':float(np.linalg.norm(w2-expected_w)),'geometry':g2}
            rows.append(row)
        save(out/'reference-geometry.json',{'complete':False,'rows':rows})
        v2.emit('Stage11StaticPose='+pose_spec[0]+'; reference placements='+str(len(rows))+'; integration=0')
    geom=[r['carried_geometry'] for r in rows]+[r['vertical_replacement']['geometry'] for r in rows if r['vertical_replacement']['status']=='MEASURED']
    outcome={'complete':True,'pose_count':len(poses),'location_count':len(points),'primary_comparisons':len(rows),
             'distance_measurements':len(geom),'geometry_v2_pass_count':sum(g['v2_pass'] for g in geom),
             'all_measured_geometry_v2_pass':all(g['v2_pass'] for g in geom),
             'outside_projection_count':sum(r['vertical_replacement']['status']!='MEASURED' for r in rows),
             'rows':rows,'no_reference_selected':True,'new_integration_steps':0}
    require(floor.model_signature(old)==oldsig and floor.model_signature(new)==newsig, 'Model parameters changed during static audit')
    save(out/'reference-geometry.json',outcome);save(out/'pose-and-derivative-review.json',poses)
    save(out/'model-signatures.json',{'old':oldsig,'independent_head_variant':newsig,'unchanged_after_audit':True})
    np.savez_compressed(out/'static-configurations.npz',qpos=np.asarray(qstates),time=np.zeros(len(qstates)))
    return {k:v for k,v in outcome.items() if k!='rows'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--static-reference-only',action='store_true',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    report={'status':'STATIC_REFERENCE_REVIEW_REQUIRED','stage11_status':'MATERIAL_BLOCKED_REVIEW_REQUIRED',
            'stage11_complete':False,'new_ppo_updates':0,'new_integration_steps':0,'full_robot_steps':0,
            'cloud_hours':0,'canonical_total_completed_steps':7895,'formal_task_migration':False,
            'selected_reference':None,'head_neck_locked':False,'policy_success':None,'old_failures_retained':True}
    attempts=[];started=time.monotonic()
    try:
        with zero_integration_guard() as attempts:
            v2.local_preflight(out);current=floor.load_json(out/'preflight.json')
            save(out/'input-provenance.json',verify_prior(current))
            from mjlab.scene import Scene
            cfg,old_cfg,new_cfg=v2.physics_scene_pair()
            old=Scene(old_cfg,device='cpu').compile();new=Scene(new_cfg,device='cpu').compile()
            cfg.sim.mujoco.apply(old);cfg.sim.mujoco.apply(new)
            with np.load(PRIOR/'diagnostic/material-geometry.npz',allow_pickle=False) as z:
                geometry={k:z[k].copy() for k in z.files}
            report['geometry']=audit_models(old,new,geometry,out,started+300)
            if not report['geometry']['all_measured_geometry_v2_pass'] or report['geometry']['outside_projection_count']:
                report['status']='STATIC_REFERENCE_GEOMETRY_BLOCKED_REVIEW_REQUIRED'
    except Exception as exc:
        report.update(status='STATIC_REFERENCE_EXECUTION_BLOCKED',error=str(exc))
        (out/'error.txt').write_text(traceback.format_exc())
    report['forbidden_integration_call_attempts']=attempts
    report['wall_seconds']=time.monotonic()-started
    report['audit_source_sha256']=digest(Path(__file__))
    report['scope']='Full free-joint Scene; only MjData poses and virtual kinematics. Not a controller, reset choice, equilibrium or policy test.'
    save(out/'reference-review.json',report)
    v2.emit('Stage11ReferenceResult='+report['status']+'; new integration=0; PPO=0')
    return 0 if report['status']=='STATIC_REFERENCE_REVIEW_REQUIRED' else 2


if __name__=='__main__': raise SystemExit(main())
