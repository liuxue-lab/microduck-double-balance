"""Read-only geometric/inertial audit; no scene edits or simulation steps."""
from pathlib import Path
import ast
import hashlib
import json
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'stage10-batch-a-review'
OUT=Path(__file__).resolve().parent

def main():
    inventory=json.loads((DATA/'contact-inventory.json').read_text())
    assert hashlib.sha256((DATA/'contact-inventory.npz').read_bytes()).hexdigest()==inventory['mesh_npz_sha256']
    with np.load(DATA/'contact-inventory.npz',allow_pickle=False) as z:
        vertices=z['geom_49_world_vertices']; faces=z['geom_49_mesh_faces']
        assert np.array_equal(vertices,z['geom_48_world_vertices'])
    hull=ConvexHull(vertices)
    a,b,c=vertices[faces[:,0]],vertices[faces[:,1]],vertices[faces[:,2]]
    ba=b[:,:2]-a[:,:2];ca=c[:,:2]-a[:,:2]
    determinant=ba[:,0]*ca[:,1]-ba[:,1]*ca[:,0]
    valid_plane=np.abs(determinant)>1e-16
    upper=hull.equations[hull.equations[:,2]>1e-10]
    def heights(x,y):
        offset=np.array([x,y])-a[:,:2]
        u=np.divide(offset[:,0]*ca[:,1]-offset[:,1]*ca[:,0],determinant,out=np.zeros_like(determinant),where=valid_plane)
        v=np.divide(ba[:,0]*offset[:,1]-ba[:,1]*offset[:,0],determinant,out=np.zeros_like(determinant),where=valid_plane)
        keep=valid_plane&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)
        heights=a[:,2]+u*(b[:,2]-a[:,2])+v*(c[:,2]-a[:,2])
        index=np.argmax(np.where(keep,heights,-np.inf))
        upper_z=-(upper[:,:2]@np.array([x,y])+upper[:,3])/upper[:,2]
        h_index=np.argmin(upper_z)
        return {'xy_m':[x,y],'mesh_vertical_top_z_m':float(heights[index]),
                'computed_hull_vertical_top_z_m':float(upper_z[h_index]),
                'computed_hull_facet_slope_deg':float(np.degrees(np.arccos(upper[h_index,2])))}
    samples=[heights(.01384+dx,dy) for dx,dy in [(0,0),(-.012,0),(.012,0),(0,-.012),(0,.012),(0,-.020),(0,.020)]]
    bodies={x['name']:x for x in inventory['bodies']}
    jaw=bodies['robot/jaw_soft']; tray=bodies['robot/double_balance_tray']
    cfg=ast.parse((ROOT/'stage10-current/src/mjlab_microduck/tasks/microduck_double_balance_env_cfg.py').read_text())
    tray_position=None
    for node in cfg.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='TRAY_POS_IN_JAW' for t in node.targets):
            tray_position=np.array(ast.literal_eval(node.value))
    assert tray_position is not None
    jaw_com=np.array(jaw['ipos_m']); mt=tray['mass_kg']; mj=jaw['mass_kg']
    combined_com=(mj*jaw_com+mt*tray_position)/(mj+mt)
    q=jaw['iquat_wxyz'];rot_j=Rotation.from_quat(q[1:]+q[:1]).as_matrix()
    rot_t=Rotation.from_euler('y',90,degrees=True).as_matrix()
    inertia_j=rot_j@np.diag(jaw['principal_inertia_kg_m2'])@rot_j.T
    inertia_t=rot_t@np.diag(tray['principal_inertia_kg_m2'])@rot_t.T
    parallel=lambda mass,offset:mass*((offset@offset)*np.eye(3)-np.outer(offset,offset))
    cluster_inertia=inertia_j+parallel(mj,jaw_com-combined_com)+inertia_t+parallel(mt,tray_position-combined_com)
    robot_mass=sum(x['mass_kg'] for x in inventory['bodies'] if x['name'].startswith('robot/'))
    output={'status':'OFFLINE_GEOMETRY_AUDIT_COMPLETE_DYNAMIC_CONTACT_UNVERIFIED',
        'source_contact_inventory_sha256':hashlib.sha256((DATA/'contact-inventory.json').read_bytes()).hexdigest(),
        'simulation_steps':0,'new_ppo_updates':0,'physics_modified':False,
        'top_ball_mask_compatible_geom_ids':inventory['mask_compatible_geom_ids'],
        'collision_shell_geom_id_in_this_compilation':49,'visual_shell_geom_id_in_this_compilation':48,
        'mesh_vertices':len(vertices),'mesh_triangles':len(faces),
        'offline_convex_hull_vertices':len(hull.vertices),'offline_convex_hull_faces':len(hull.simplices),
        'home_aabb_m':[vertices.min(axis=0).tolist(),vertices.max(axis=0).tolist()],
        'vertical_ray_samples':samples,
        'approximate_old_tray_top_face_z_from_source_home_comment_m':.15695,
        'approximate_old_tray_top_to_shell_at_center_m':.15695-samples[0]['mesh_vertical_top_z_m'],
        'mass_audit':{'scene_with_tray_kg':inventory['scene_total_mass_kg'],
                      'scene_after_only_tray_body_removal_kg':inventory['scene_total_mass_kg']-mt,
                      'robot_with_tray_kg':robot_mass,'robot_after_only_tray_body_removal_kg':robot_mass-mt,
                      'robot_mass_reduction_percent':100*mt/robot_mass,
                      'jaw_with_tray_kg':mj+mt,'jaw_without_tray_kg':mj,
                      'jaw_cluster_com_before_in_jaw_m':combined_com.tolist(),
                      'jaw_cluster_com_after_in_jaw_m':jaw_com.tolist(),
                      'jaw_cluster_com_shift_on_removal_m':(jaw_com-combined_com).tolist(),
                      'jaw_cluster_inertia_before_about_own_com_in_jaw_axes_kg_m2':cluster_inertia.tolist(),
                      'jaw_inertia_after_about_own_com_in_jaw_axes_kg_m2':inertia_j.tolist(),
                      'removed_tray_inertia_about_jaw_origin_kg_m2':(inertia_t+parallel(mt,tray_position)).tolist()},
        'limits':['HOME only; all dimensions refer to the exported HOME world frame.',
                  'SciPy hull is a geometric check, not the captured MuJoCo-Warp contact hull.',
                  'Vertical ray points are not finite-radius ball contact solutions, support areas, or friction validation.',
                  'AABB maximum is not the height at the old tray center.',
                  'Inertia subtraction is analytical bookkeeping for jaw plus fixed tray; no compiled model was changed.']}
    (OUT/'geometry-review.json').write_text(json.dumps(output,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'samples':samples,'mass_audit':output['mass_audit']},indent=2))

if __name__=='__main__':main()
