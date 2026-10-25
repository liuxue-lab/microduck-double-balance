"""Trace release of every instantiated render BVH, mesh and texture owner."""
import gc,hashlib,os,weakref
from lifecycle_support import require
from report_io import save
from segment_runtime import ptr

class Resources:
    def __init__(self,wp,ctx,out,identity):
        from warp._src import context
        self.wp,self.out,self.identity,self.core=wp,out,identity,context.runtime.core
        self.originals={};self.releases=[];self.targets=[];self.refs=[];self.closed=False
        names=('wp_bvh_destroy_device','wp_mesh_destroy_device','wp_texture1d_destroy_device','wp_texture2d_destroy_device','wp_texture3d_destroy_device','wp_texture_array_destroy_surface_device')
        for name in names:
            original=getattr(self.core,name);self.originals[name]=original
            def observed(*args,_name=name,_original=original):
                row=dict(api=_name,args=[ptr(x) for x in args],capturing=bool(wp.get_device('cuda:0').is_capturing),returned_normally=False)
                self.releases.append(row);value=_original(*args);row['returned_normally']=True;self.persist();return value
            setattr(self.core,name,observed)
        rc=ctx.render_context;self.render_ref=weakref.ref(rc)
        def register(label,obj,api,handle,**extra):
            require(obj is not None and handle,'Missing native resource')
            self.targets.append(dict(label=label,type=type(obj).__name__,api=api,handle=int(handle),owner=id(obj),**extra));self.refs.append(weakref.ref(obj))
        register('scene_bvh',rc.bvh,'wp_bvh_destroy_device',rc.bvh.id)
        self.logical={}
        for field,id_field in (('mesh_registry','mesh_bvh_id'),('hfield_registry','hfield_bvh_id'),('flex_mesh_registry','flex_bvh_id')):
            registry=getattr(rc,field);ids=getattr(rc,id_field).numpy().reshape(-1).tolist()
            require(set(int(x) for x in ids if x)=={int(v.id) for v in registry.values()},'Resource table/owner mismatch: '+field)
            self.logical[field]=[]
            for index,handle in enumerate(ids):
                if not handle:continue
                matches=[v for v in registry.values() if int(v.id)==int(handle)];require(len(matches)==1,'Ambiguous mesh owner')
                obj=matches[0];parts={}
                for n in ('points','indices','velocities'):
                    a=getattr(obj,n)
                    if a is not None:
                        raw=a.numpy();parts[n]=dict(shape=list(raw.shape),dtype=str(raw.dtype),sha256=hashlib.sha256(raw.tobytes()).hexdigest())
                register(field+'.'+str(index),obj,'wp_mesh_destroy_device',obj.id,geometry=parts)
                self.logical[field].append(dict(index=index,geometry=parts))
        self.logical['textures']=[]
        for index,obj in enumerate(rc.textures_registry):
            require(obj._ndim in (1,2,3),'Unknown texture dimension')
            extra=dict(array_handle=int(obj._array_handle),surface_handle=int(obj._surface_handle),ndim=obj._ndim)
            register('textures.'+str(index),obj,'wp_texture%dd_destroy_device'%obj._ndim,obj._tex_handle,**extra)
            self.logical['textures'].append(dict(index=index,ndim=obj._ndim))
        self.logical.update(scene_bvh_present=True,cameras_active=False,native_registry_lengths={n:len(getattr(rc,n)) for n in ('mesh_registry','hfield_registry','flex_mesh_registry','textures_registry')})
        require(len({(r['api'],r['handle']) for r in self.targets})==len(self.targets),'Aliased resource target repeated')
        self.persist()
    def persist(self):
        save(self.out/'resource-lifetime.json',dict(identity=self.identity,targets=self.targets,releases=self.releases,logical=getattr(self,'logical',{}),closed=self.closed))
    def close(self,ctx,sim):
        require(not self.wp.get_device('cuda:0').is_capturing,'Release inside capture')
        self.wp.synchronize();ctx._ctx=None;sim._sensor_context=None;gc.collect()
        require(self.render_ref() is None,'RenderContext remains strongly owned')
        for row,ref in zip(self.targets,self.refs):
            matches=[r for r in self.releases if r['api']==row['api'] and row['handle'] in r['args']]
            row['weakref_dead']=ref() is None
            row['released']=row['weakref_dead'] and len(matches)==1 and matches[0]['returned_normally'] and not matches[0]['capturing']
            if row.get('surface_handle'):
                surface=[r for r in self.releases if r['api']=='wp_texture_array_destroy_surface_device' and r['args'][-1]==row['surface_handle']]
                row['released'] &=len(surface)==1 and surface[0]['returned_normally'] and not surface[0]['capturing']
            require(row['released'],'Resource release not witnessed: '+row['label'])
        self.closed=True;self.persist();self.unpatch()
        return dict(targets=len(self.targets),all_released=True,by_type={t:sum(r['type']==t for r in self.targets) for t in sorted({r['type'] for r in self.targets})},scope='all instantiated RenderContext mesh/hfield/flex/BVH/texture owners; allocator pools and driver context remain process-managed')
    def unpatch(self):
        for name,fn in self.originals.items():setattr(self.core,name,fn)
        self.originals.clear()
