"""Nominal reset-v2 geometry for a private diagnostic Data, without Managers.

No material/config edits, site insertion, RNG draws, BAM calls or integration.
The reference constants and default joints are the existing audited prototype.
"""
import numpy as np

RESET_ATOL = 2e-5  # Existing reset IO tolerance, not physics acceptance.
FLOOR_GUARD = -.001

def require(ok, message):
    if not bool(ok): raise RuntimeError(message)

def unique(items, message):
    require(len(items)==1, message)
    return items[0]

def resolve_ids(model, api):
    geoms={key:unique([i for i in range(model.ngeom)
        if model.geom(i).name.split('/')[-1]==suffix], 'Ambiguous geom: '+suffix)
        for key,suffix in [('top','top_ball_sphere'),('lower','ball_sphere'),('head','stage11_head_contact_surface')]}
    geoms['floor']=unique([i for i in range(model.ngeom)
        if int(model.geom_type[i])==int(api.mjtGeom.mjGEOM_PLANE)],'Ambiguous floor')
    jaw=unique([i for i in range(model.nbody) if model.body(i).name.split('/')[-1]=='jaw_soft'],'Ambiguous jaw')
    free=[i for i in range(model.njnt) if int(model.jnt_type[i])==int(api.mjtJoint.mjJNT_FREE)]
    require(len(free)==3,'Expected three free roots')
    roots={key:unique([i for i in free if model.joint(i).name.startswith(key+'/')],
        'Ambiguous free root: '+key) for key in ('robot','ball','top_ball')}
    qadr={key:int(model.jnt_qposadr[i]) for key,i in roots.items()}
    for key,entity in [('top','top_ball'),('lower','ball')]:
        require(int(model.geom_bodyid[geoms[key]])==int(model.jnt_bodyid[roots[entity]]),'Sphere/root identity differs')
        require(int(model.geom_type[geoms[key]])==int(api.mjtGeom.mjGEOM_SPHERE),'Expected sphere')
        require(not np.any(model.geom_pos[geoms[key]]),'Sphere is offset from free root')
    require(abs(float(model.geom_size[geoms['top'],0])-.020)<1e-12 and
            abs(float(model.geom_size[geoms['lower'],0])-.120)<1e-12,'Ball radii differ')
    return dict(geoms=geoms,jaw=jaw,free_joints=roots,qadr=qadr)

def reference_center(jaw_position,jaw_matrix,origin,normal,radius,clearance):
    p,R,o,n=map(lambda x:np.asarray(x,dtype=float),(jaw_position,jaw_matrix,origin,normal))
    require(p.shape==o.shape==n.shape==(3,) and R.shape==(3,3),'Reference shape differs')
    require(all(np.isfinite(x).all() for x in (p,R,o,n)),'Nonfinite material reference')
    require(abs(np.linalg.norm(n)-1)<1e-10 and np.max(abs(R.T@R-np.eye(3)))<1e-6
            and abs(np.linalg.det(R)-1)<1e-6,'Invalid reference rotation/normal')
    require(radius==.020 and clearance==.0005,'Nominal reference dimensions differ')
    return p+R@(o+(radius+clearance)*n/np.linalg.norm(n))

def geometry_row(geom_position,geom_matrix,ids,model):
    g=ids['geoms'];p=np.asarray(geom_position);R=np.asarray(geom_matrix).reshape(model.ngeom,3,3)
    require(np.isfinite(p).all() and np.isfinite(R).all(),'Nonfinite geometry')
    normal=R[g['floor'],:,2];floor=p[g['floor']]
    return dict(geom_ids=g,geom_names={k:model.geom(v).name for k,v in g.items()},
        geom_positions={k:p[v].tolist() for k,v in g.items()},floor_normal=normal.tolist(),
        top_floor_separation_m=float((p[g['top']]-floor)@normal-model.geom_size[g['top'],0]),
        lower_floor_separation_m=float((p[g['lower']]-floor)@normal-model.geom_size[g['lower'],0]))

def validate_geometry(row, *, nominal=False, expected_top=None):
    require(np.max(abs(np.asarray(row['floor_normal'])-[0,0,1]))<1e-6,'Unexpected floor orientation')
    require(row['top_floor_separation_m']>=FLOOR_GUARD,'Upper-ball floor penetration exceeded retained 1mm guard')
    if nominal:
        require(row['top_floor_separation_m']>0,'Nominal upper ball is not above floor')
        require(abs(row['lower_floor_separation_m']-.001)<=RESET_ATOL,'Nominal lower-ball clearance differs')
        require(expected_top is not None and np.max(abs(np.asarray(row['geom_positions']['top'])-expected_top))<=RESET_ATOL,
                'Nominal upper-ball placement differs')

def nominal_state(model, api, joint_qadr, default_joints, reference, counters, persist, record):
    """Two kinematic refreshes and one static distance query; zero forwards."""
    ids=resolve_ids(model,api);qadr=ids['qadr'];d=api.MjData(model)
    q=np.asarray(joint_qadr,dtype=int).reshape(-1);home=np.asarray(default_joints,dtype=float).reshape(-1)
    require(len(q)==len(home)==14 and len(set(q))==14 and np.isfinite(home).all(),'Nominal HOME layout differs')
    expected_hinges={int(model.jnt_qposadr[i]) for i in range(model.njnt)
        if model.joint(i).name.startswith('robot/') and int(model.jnt_type[i])==int(api.mjtJoint.mjJNT_HINGE)}
    require(set(q)==expected_hinges,'Named robot hinge addresses differ from Entity defaults')
    require(model.nkey>=1,'Missing HOME keyframe')
    d.qpos[:]=model.key_qpos[0];d.qpos[q]=home
    for name,z in [('robot',.368),('ball',.121)]:
        a=qadr[name];d.qpos[a:a+7]=[0,0,z,1,0,0,0]
    a=qadr['top_ball'];d.qpos[a:a+7]=[0,0,0,1,0,0,0]
    d.qvel[:]=0;d.ctrl[:]=0;d.time=0
    def kinematics():
        require(counters['CPU_kinematics_attempted']<2,'Kinematics budget exhausted')
        before=d.qpos.copy();counters['CPU_kinematics_attempted']+=1;persist()
        api.mj_kinematics(model,d)
        counters['CPU_kinematics_completed']+=1;persist()
        require(np.array_equal(d.qpos,before) and not d.qvel.any() and d.time==0,'Kinematics changed primary state')
    kinematics()
    top=reference_center(d.xpos[ids['jaw']],d.xmat[ids['jaw']].reshape(3,3),
        reference.ORIGIN_IN_JAW,reference.NORMAL_IN_JAW,reference.RADIUS,reference.CLEARANCE)
    d.qpos[a:a+3]=top;kinematics()
    row=geometry_row(d.geom_xpos,d.geom_xmat,ids,model)
    row.update(qpos=d.qpos.tolist(),qvel=d.qvel.tolist(),time_s=float(d.time),
        root_qadr=qadr,default_joint_qadr=q.tolist(),default_joint_pos=home.tolist(),
        material_reference_version=reference.REFERENCE_VERSION,
        expected_top_position=top.tolist(),reference_origin_in_jaw=list(reference.ORIGIN_IN_JAW),
        reference_normal_in_jaw=list(reference.NORMAL_IN_JAW),radius_m=.020,clearance_m=.0005,
        source='Existing reset_stage11_head_reference nominal geometry; private CPU Data, no reset function or Manager invoked')
    record(row)  # Persist the actual geometry before any threshold can fail.
    validate_geometry(row,nominal=True,expected_top=top)
    require(counters['CPU_geom_distance_attempted']<1,'Distance query budget exhausted')
    witness=np.zeros(6);counters['CPU_geom_distance_attempted']+=1;persist()
    gap=float(api.mj_geomDistance(model,d,ids['geoms']['head'],ids['geoms']['top'],1.,witness))
    counters['CPU_geom_distance_completed']+=1;persist()
    row.update(top_head_gap_m=gap,head_top_witness=witness.tolist());record(row)
    require(np.isfinite(gap) and np.isfinite(witness).all() and abs(gap-.0005)<=RESET_ATOL,
        'Nominal head/upper-ball gap differs from reviewed reset geometry')
    require(np.array_equal(d.qpos[q],home) and not d.qvel.any() and d.time==0,'Nominal primary state differs')
    return d.qpos.copy(),ids,row
