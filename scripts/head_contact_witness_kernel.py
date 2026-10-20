"""Read-only witness recorder using original MuJoCo Warp3.8.1 functions.

No installed-engine changes. This separately invokes native CCD plus the
sphere-point GJK/inflate branch to expose intermediate arithmetic. A match to
saved native contact fields is required before calling it representative.
"""
import numpy as np
import warp as wp
from mujoco_warp._src.collision_core import geom_collision_pair
from mujoco_warp._src.collision_gjk import ccd,gjk,_inflate,FLOAT_MAX
from mujoco_warp._src.collision_convex import MJ_MAX_EPAFACES,MJ_MAX_EPAHORIZON
from mujoco_warp._src.math import make_frame
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
  # Fresh structs: do not assume CCD did or did not mutate local copies.
  a,b=geom_collision_pair(
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
  radius=a.size[0]
  a.size=wp.vec3(0.0,a.size[1],a.size[2])
  result=gjk(tolerance[0],iterations,a,b,a.pos,b.pos,2,7,radius,False)
  scalars[3]=result.dist;scalars[4]=float(result.dim);scalars[7]=tolerance[0];scalars[8]=radius
  vectors[2]=result.x1;vectors[3]=result.x2
  vectors[11]=result.x2-result.x1
  if result.dist>tolerance[0] and result.dist!=FLOAT_MAX:
   scalars[5]=1.0
   d,x1,x2=_inflate(result,a,b,2,7,radius,0.0)
   scalars[6]=d;vectors[4]=x1;vectors[5]=x2
   vectors[6]=wp.normalize(result.x2-result.x1)
   vectors[7]=wp.normalize(x1-x2)
   vectors[10]=x1-x2
  vectors[8]=wp.vec3(frames[0][0,0],frames[0][0,1],frames[0][0,2])
 return witness


def query(model,saved_geom_xpos,saved_geom_xmat,kernel):
 """One thread, no Model/Data mutation; accepts archived device kinematics."""
 iterations=model.opt.ccd_iterations
 vectors=wp.zeros(12,dtype=wp.vec3);scalars=wp.zeros(9,dtype=float);frames=wp.zeros(1,dtype=wp.mat33)
 xpos=wp.array(np.asarray(saved_geom_xpos,np.float32),dtype=wp.vec3)
 xmat=wp.array(np.asarray(saved_geom_xmat,np.float32),dtype=wp.mat33)
 scratch=[wp.empty((1,10+2*iterations),dtype=wp.vec3),wp.empty((1,10+2*iterations),dtype=int),
  wp.empty((1,6+MJ_MAX_EPAFACES*iterations),dtype=int),wp.empty((1,6+MJ_MAX_EPAFACES*iterations),dtype=wp.vec3),
  wp.empty((1,6+MJ_MAX_EPAFACES*iterations),dtype=float),wp.empty((1,MJ_MAX_EPAHORIZON),dtype=int)]
 wp.launch(kernel,dim=1,inputs=[*[getattr(model,n) for n in MODEL_FIELDS],xpos,xmat,model.opt.ccd_tolerance,*scratch],outputs=[scalars,vectors,frames])
 wp.synchronize()
 s,v,f=scalars.numpy(),vectors.numpy(),frames.numpy()
 if not (np.isfinite(s).all() and np.isfinite(v).all() and np.isfinite(f).all()):raise RuntimeError('Nonfinite witness output')
 return {'scalars':s,'vectors':v,'frame':f[0],'scalar_names':['ccd_dist','ccd_ncon','ccd_multiccd_index','shrunk_gjk_distance','shrunk_gjk_dimension','shallow_finite_branch','inflated_dist','ccd_tolerance','sphere_radius'],
  'vector_names':['ccd_w1','ccd_w2','shrunk_w1','shrunk_w2','inflated_w1','inflated_w2','preinflate_direction','inflated_witness_direction','ccd_frame_normal','ccd_midpoint','inflated_witness_difference','preinflate_difference']}
