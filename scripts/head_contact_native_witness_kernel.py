"""Isolated native-witness normal candidate; exact original CCD geometry retained.

One native CCD call, no repeated GJK/inflate. For contacting shallow sphere2/
zero-margin headmesh1 only, construct the frame from native head witness minus
sphere center. Original distance, both witnesses, midpoint and presence are
returned untouched. Deep/no-contact branches preserve native frame. No engine patch.
"""
import numpy as np
import warp as wp
from mujoco_warp._src.collision_core import geom_collision_pair,Geom
from mujoco_warp._src.collision_gjk import ccd,FLOAT_MAX
from mujoco_warp._src.collision_convex import MJ_MAX_EPAFACES,MJ_MAX_EPAHORIZON
from mujoco_warp._src.math import make_frame
from typing import Tuple
MODEL_FIELDS=('geom_type', 'geom_dataid', 'geom_size', 'mesh_vertadr', 'mesh_vertnum', 'mesh_graphadr', 'mesh_vert', 'mesh_graph', 'mesh_polynum', 'mesh_polyadr', 'mesh_polynormal', 'mesh_polyvertadr', 'mesh_polyvertnum', 'mesh_polyvert', 'mesh_polymapadr', 'mesh_polymapnum', 'mesh_polymap')


def make_kernel(iterations):
 @wp.kernel(module="unique",enable_backward=False)
 def witness(
  geom_type: wp.array[int],
  geom_dataid: wp.array2d[int],
  geom_size: wp.array2d[wp.vec3],
  mesh_vertadr: wp.array[int],
  mesh_vertnum: wp.array[int],
  mesh_graphadr: wp.array[int],
  mesh_vert: wp.array[wp.vec3],
  mesh_graph: wp.array[int],
  mesh_polynum: wp.array[int],
  mesh_polyadr: wp.array[int],
  mesh_polynormal: wp.array[wp.vec3],
  mesh_polyvertadr: wp.array[int],
  mesh_polyvertnum: wp.array[int],
  mesh_polyvert: wp.array[int],
  mesh_polymapadr: wp.array[int],
  mesh_polymapnum: wp.array[int],
  mesh_polymap: wp.array[int],
  geom_xpos_in: wp.array2d[wp.vec3],
  geom_xmat_in: wp.array2d[wp.mat33],
  tolerance: wp.array[float],
  epa_vert: wp.array2d[wp.vec3],
  epa_index: wp.array2d[int],
  epa_face: wp.array2d[int],
  epa_pr: wp.array2d[wp.vec3],
  epa_norm2: wp.array2d[float],
  epa_horizon: wp.array2d[int],
  scalars: wp.array[float],
  vectors: wp.array[wp.vec3],
  frames: wp.array[wp.mat33],
 ):
  g1,g2=geom_collision_pair(
    geom_type,
    geom_dataid,
    geom_size,
    mesh_vertadr,
    mesh_vertnum,
    mesh_graphadr,
    mesh_vert,
    mesh_graph,
    mesh_polynum,
    mesh_polyadr,
    mesh_polynormal,
    mesh_polyvertadr,
    mesh_polyvertnum,
    mesh_polyvert,
    mesh_polymapadr,
    mesh_polymapnum,
    mesh_polymap,
    geom_xpos_in,
    geom_xmat_in,
    wp.vec2i(2,1),0)
  dist,ncon,w1,w2,multi=ccd(tolerance[0],0.0,iterations,iterations,g1,g2,2,7,g1.pos,g2.pos,
    epa_vert[0],epa_index[0],epa_face[0],epa_pr[0],epa_norm2[0],epa_horizon[0])
  scalars[0]=dist;scalars[1]=float(ncon);scalars[2]=float(multi)
  vectors[0]=w1;vectors[1]=w2;vectors[9]=0.5*(w1+w2)
  frames[0]=make_frame(w1-w2)
  # The zero-margin head witness w2 is native CCD output. Do not recompute
  # GJK or inflation. The center-to-witness vector is radius scale, avoiding
  # cancellation between the two almost-coincident surface witnesses.
  radius=g1.size[0]
  direction=w1-w2
  applied=int(0)
  if ncon==1 and multi==-1 and dist<0.0 and dist+radius>tolerance[0]:
   direction=w2-g1.pos
   applied=1
  scalars[3]=wp.length(w2-g1.pos);scalars[4]=float(multi);scalars[5]=float(applied)
  scalars[6]=dist;scalars[7]=tolerance[0];scalars[8]=radius
  scalars[9]=dist;scalars[10]=float(ncon);scalars[11]=float(applied)
  vectors[2]=g1.pos;vectors[3]=w2;vectors[4]=w1;vectors[5]=w2
  vectors[6]=wp.normalize(direction);vectors[7]=wp.normalize(w1-w2)
  vectors[8]=wp.vec3(frames[0][0,0],frames[0][0,1],frames[0][0,2])
  vectors[10]=w1-w2;vectors[11]=w2-g1.pos
  vectors[12]=w1;vectors[13]=w2;vectors[15]=0.5*(w1+w2);vectors[16]=direction
  frames[1]=make_frame(direction)
  vectors[14]=wp.vec3(frames[1][0,0],frames[1][0,1],frames[1][0,2])

 return witness


def query(model,saved_geom_xpos,saved_geom_xmat,kernel):
 """One thread, no Model/Data mutation; accepts archived device kinematics."""
 iterations=model.opt.ccd_iterations
 vectors=wp.zeros(17,dtype=wp.vec3);scalars=wp.zeros(12,dtype=float);frames=wp.zeros(2,dtype=wp.mat33)
 xpos=wp.array(np.asarray(saved_geom_xpos,np.float32),dtype=wp.vec3)
 xmat=wp.array(np.asarray(saved_geom_xmat,np.float32),dtype=wp.mat33)
 scratch=[wp.empty((1,10+2*iterations),dtype=wp.vec3),wp.empty((1,10+2*iterations),dtype=int),
  wp.empty((1,6+MJ_MAX_EPAFACES*iterations),dtype=int),wp.empty((1,6+MJ_MAX_EPAFACES*iterations),dtype=wp.vec3),
  wp.empty((1,6+MJ_MAX_EPAFACES*iterations),dtype=float),wp.empty((1,MJ_MAX_EPAHORIZON),dtype=int)]
 wp.launch(kernel,dim=1,inputs=[*[getattr(model,n) for n in MODEL_FIELDS],xpos,xmat,model.opt.ccd_tolerance,*scratch],outputs=[scalars,vectors,frames])
 wp.synchronize()
 s,v,f=scalars.numpy(),vectors.numpy(),frames.numpy()
 if not (np.isfinite(s).all() and np.isfinite(v).all() and np.isfinite(f).all()):raise RuntimeError('Nonfinite witness output')
 return {'scalars':s,'vectors':v,'frame':f[0],'candidate_frame':f[1],'scalar_names':['ccd_dist','ccd_ncon','ccd_multiccd_index','center_to_native_head_witness_distance','native_multiccd_index_copy','native_witness_normal_used','native_distance_copy','ccd_tolerance','sphere_radius','candidate_dist','candidate_ncon','candidate_native_witness_normal_used'],
  'vector_names':['ccd_w1','ccd_w2','native_sphere_center','native_head_witness','native_witness1_copy','native_witness2_copy','selected_direction_normalized','native_surface_witness_direction','ccd_frame_normal','ccd_midpoint','native_surface_witness_difference','native_head_witness_minus_center','candidate_w1','candidate_w2','candidate_frame_normal','candidate_midpoint','candidate_unscaled_direction']}
