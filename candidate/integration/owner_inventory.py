"""Bounded read-only metadata for diagnosing all owner interfaces in one return.

This is not a state snapshot or acceptance test. It never copies array data,
calls a manager method, samples RNG, or alters an owner.
"""
from integration.durable import write_json


def record(path,env,bank,torch,wp):
    budget=[10000]
    def describe(value,depth=0):
        budget[0]-=1
        kind=type(value).__module__+'.'+type(value).__qualname__
        if budget[0]<0:return dict(type=kind,metadata_truncated=True)
        if isinstance(value,torch.Tensor):
            return dict(type=kind,shape=list(value.shape),stride=list(value.stride()),
                        dtype=str(value.dtype),device=str(value.device),requires_grad=value.requires_grad)
        if isinstance(value,wp.array):
            return dict(type=kind,shape=list(value.shape),dtype=str(value.dtype),device=str(value.device))
        if value is None or type(value) in (str,int,bool,float):return dict(type=kind)
        if isinstance(value,dict):
            return dict(type=kind,length=len(value),entries=[dict(key_type=type(k).__name__,value=describe(v,depth+1))
                for k,v in value.items()] if depth<2 else [])
        if isinstance(value,(tuple,list)):
            return dict(type=kind,length=len(value),entries=[describe(v,depth+1) for v in value] if depth<2 else [])
        fields=getattr(value,'__dict__',None)
        return dict(type=kind,fields=sorted(fields) if isinstance(fields,dict) else None,callable=callable(value))
    owners=[('env',env),('RNGService',bank)]
    for name in ('action','observation','event','reward','termination','metrics','command','curriculum','recorder'):
        manager=vars(env).get(name+'_manager')
        if manager is not None:
            owners.append((name+'Manager',manager))
            terms=vars(manager).get('_terms')
            if isinstance(terms,dict):owners.extend((name+':'+str(k),v) for k,v in terms.items())
            elif isinstance(terms,list):owners.extend((name+':'+str(i),v) for i,v in enumerate(terms))
    scene=vars(env).get('scene')
    if scene is not None:
        for name,entity in scene.entities.items():
            owners.append(('entity:'+name,entity.data))
            for i,act in enumerate(entity.actuators):owners.append(('actuator:'+name+'/'+str(i),act))
        owners.extend(('sensor:'+name,obj) for name,obj in scene.sensors.items())
    rows=[dict(label=name,type=type(obj).__module__+'.'+type(obj).__qualname__,
               fields={k:describe(v) for k,v in vars(obj).items()}) for name,obj in owners]
    report=dict(owners=rows,diagnostic_metadata_only=True,metadata_truncated=budget[0]<0,
                array_data_copies=0,manager_calls=0,native_integrations=0,acceptance=False)
    write_json(path,report)
    return dict(file=str(path),owners=len(rows),metadata_truncated=report['metadata_truncated'])
