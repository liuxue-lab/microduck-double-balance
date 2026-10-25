"""Pinned ray sense lifecycle: mutable outputs, fixed native owners, exact rollback.

Only the eleven rebindings and five buffer writes in pinned RayCastSensor
prepare_rays/postprocess_rays/raycast_kernel are permitted at a sense boundary.
The whole owner snapshot still contains every field for exact restoration.
"""
import numpy as np
from lifecycle_support import require
from sensor_state_snapshot import tensor_meta,warp_meta
from scene_evidence import owner_state,array_meta
from transaction_contract import topology_signature

REBOUND=('_cached_world_origins','_cached_world_rays','_cached_frame_pos','_cached_frame_mat',
         '_hit_pos_w','_distances','_normals_w','_frame_pos_w','_frame_quat_w','_pos_w','_quat_w')
BUFFERS=('_ray_pnt','_ray_vec','_ray_dist','_ray_geomid','_ray_normal')
RAY_TYPE=('mjlab.sensor.terrain_height_sensor','TerrainHeightSensor')

class SensorLifecycleContract:
    def __init__(self,scene,roots,torch,wp):
        self.scene,self.torch,self.wp=scene,torch,wp
        self.rays=[(name,s) for name,s in scene.sensors.items() if (type(s).__module__,type(s).__qualname__)==RAY_TYPE]
        require(len(self.rays)==1,'Exactly one pinned TerrainHeightSensor required')
        self.roots=[];self.layouts={};self.buffers={};self.schemas={}
        for name,s in self.rays:
            require(s._num_frames==2 and s._num_rays==6 and s._num_rays_per_frame==3,'Pinned ray shape differs')
            self.roots.append(('sensor:'+name,s,REBOUND+BUFFERS))
            for n in REBOUND:
                t=getattr(s,n);require(isinstance(t,torch.Tensor),'Ray output Tensor missing: '+n)
                meta=tensor_meta(t);self.layouts[id(s),n]=(meta[:5],meta[8])
            for n in BUFFERS:
                a=getattr(s,n);require(isinstance(a,wp.array),'Ray native buffer missing: '+n)
                self.buffers[id(s),n]=(a,warp_meta(a))
            self.schemas[id(s)]=frozenset(vars(s))
        selected={(id(obj),n) for _,obj,names in roots for n in names}
        require(all((id(s),n) in selected for _,s,names in self.roots for n in names),'Ray rollback roots incomplete')
        self.check_bindings()
    def held_roots(self,roots):
        sensors={id(s) for s in self.scene.sensors.values()};rays={id(s) for _,s in self.rays}
        return [(label,obj,tuple(n for n in names if not (id(obj) in sensors and n in ('_cached_data','_cache_valid'))
            and not (id(obj) in rays and n in REBOUND+BUFFERS))) for label,obj,names in roots]
    def check_bindings(self):
        for _,s in self.rays:
            require(frozenset(vars(s))==self.schemas[id(s)],'Ray attribute schema changed')
            for n in REBOUND:
                t=getattr(s,n);require(isinstance(t,self.torch.Tensor),'Ray result type changed: '+n)
                meta=tensor_meta(t);require((meta[:5],meta[8])==self.layouts[id(s),n],'Ray result shape/stride/device changed: '+n)
                require(np.isfinite(t.detach().cpu().numpy()).all(),'Nonfinite ray result: '+n)
            for n in BUFFERS:
                a,meta=self.buffers[id(s),n]
                require(getattr(s,n) is a and warp_meta(a)==meta,'Native ray buffer binding/layout changed: '+n)
            require(s._frame_pos_w is s._cached_frame_pos,'Ray frame position alias changed')
            for output,native in (('_distances','_ray_dist'),('_normals_w','_ray_normal')):
                t,a=getattr(s,output),getattr(s,native)
                require(t.data_ptr()==a.ptr and t.untyped_storage().data_ptr()==a.ptr and t.storage_offset()==0,'Ray output/native buffer alias changed: '+output)
            for view,parent in (('_pos_w','_frame_pos_w'),('_quat_w','_frame_quat_w')):
                t,p=getattr(s,view),getattr(s,parent)
                require(t.data_ptr()==p.data_ptr() and t.untyped_storage().data_ptr()==p.untyped_storage().data_ptr() and t.storage_offset()==p.storage_offset(),
                    'First-frame view/storage alias changed: '+view)
                require(t.detach().cpu().numpy().tobytes()==p.detach().cpu().numpy()[:,0].tobytes(),'First-frame view values differ: '+view)
    def current(self):
        self.check_bindings();raw,scalars,topology=owner_state(self.roots,self.torch,self.wp)
        require(not scalars,'Ray numeric state unexpectedly contains scalar leaves')
        return dict(arrays=array_meta(raw),topology=topology)
    def semantic(self):
        value=self.current()
        return dict(arrays=value['arrays'],topology=topology_signature(value['topology']),
            allowed_rebound_fields=list(REBOUND),fixed_mutable_buffers=list(BUFFERS))
