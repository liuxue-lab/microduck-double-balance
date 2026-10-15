"""Archive selected numeric owners, aliases, Model and RNG for offline review."""
import dataclasses
import math
import numpy as np
from report_io import save
from sensor_state_snapshot import digest, tensor_meta, warp_meta, OwnerSnapshot
from lifecycle_support import require


def owner_state(roots, torch, wp):
    raw, scalars, topology = {}, {}, {}
    def walk(value, path, field='', ancestors=()):
        require(len(raw)+len(scalars)+len(topology)<12000,'Selected owner record cap')
        if field in OwnerSnapshot.REFERENCES:
            topology[path]=dict(kind='reference',owner=id(value),type=str(type(value)));return
        if value is None or type(value) in (str,int,float,bool):
            # Preserve native empty-mask NaN logs without modifying live values.
            scalars[path]={'nonfinite_float':repr(value)} if type(value) is float and not math.isfinite(value) else value
            return
        if isinstance(value, torch.Tensor):
            raw[path]=value.detach().cpu().numpy().copy()
            topology[path]=dict(kind='torch',owner=id(value),metadata=tensor_meta(value));return
        if isinstance(value,wp.array):
            require('Texture' not in str(value.dtype),'Opaque owner outside selected state')
            raw[path]=value.numpy().copy();topology[path]=dict(kind='warp',owner=id(value),metadata=warp_meta(value));return
        require(id(value) not in ancestors,'Cycle in selected owners')
        ancestors=ancestors+(id(value),)
        topology[path]=dict(kind=type(value).__name__,owner=id(value))
        if isinstance(value,dict):items=value.items()
        elif isinstance(value,(tuple,list)):items=enumerate(value)
        else:
            require(type(value).__module__.startswith(('mjlab.sensor.','mjlab.utils.buffers.')),'Unknown selected owner '+path)
            items=vars(value).items()
        for k,v in items:walk(v,path+'.'+str(k),str(k),ancestors)
    for label,obj,names in roots:
        for name in names:walk(getattr(obj,name),label+'.'+name,name)
    require(sum(a.nbytes for a in raw.values())<=64*1024*1024,'Owner raw byte cap')
    return raw,scalars,topology


def model_state(model,wp):
    raw, scalars, opaque, owners = {}, {}, {}, {}
    seen=set()
    def walk(v,p):
        if isinstance(v,wp.array):
            owners[p]=dict(owner=id(v),metadata=warp_meta(v))
            if 'Texture' in str(v.dtype):opaque[p]='opaque texture; not validated'
            else:raw[p]=v.numpy().copy()
        elif v is None or type(v) in (int,float,bool,str):scalars[p]=v
        elif dataclasses.is_dataclass(v) and id(v) not in seen:
            seen.add(id(v))
            for f in dataclasses.fields(v):walk(getattr(v,f.name),p+'.'+f.name)
        else:opaque[p]=dict(owner=id(v),type=str(type(v)),contents_validated=False)
    walk(model,'Model')
    require(raw and sum(a.nbytes for a in raw.values())<=64*1024*1024,'Model raw byte cap')
    return raw,dict(scalars=scalars,opaque=opaque,owners=owners)


def array_meta(raw):
    return {n:dict(shape=list(a.shape),dtype=str(a.dtype),bytes=a.nbytes,sha256=digest(a)) for n,a in raw.items()}


def save_arrays(out,label,raw,**metadata):
    np.savez_compressed(out/(label+'.npz'),**raw)
    save(out/(label+'.json'),dict(arrays=array_meta(raw),**metadata))


def rng_raw(torch):
    return dict(cpu=torch.get_rng_state().numpy().copy(),cuda=torch.cuda.get_rng_state('cuda:0').cpu().numpy().copy())


def outputs(scene,torch,counts):
    result={}
    def walk(v,p):
        if isinstance(v,torch.Tensor):
            result[p]=v.detach().cpu().numpy().copy()
            require(np.isfinite(result[p]).all(),'Nonfinite sensor output '+p)
        else:
            require(dataclasses.is_dataclass(v),'Unknown sensor result '+p)
            for f in dataclasses.fields(v):
                child=getattr(v,f.name)
                if child is not None:walk(child,p+'.'+f.name)
    for name,sensor in scene.sensors.items():
        a,b=sensor.data,sensor.data
        counts['sensor_data_reads']=counts.get('sensor_data_reads',0)+2
        require(a is b and sensor._cache_valid,'Sensor cache reuse differs')
        walk(a,name)
    require(len(result)==26,'Sensor output count differs')
    return result
