"""Rebuild sensing, then its Python cache, before comparing full owner topology.

The intermediate post-sense state is diagnostic evidence, never an accepted
checkpoint. No field, alias, numeric byte, or scalar is removed from acceptance.
"""
import json
from portable_state import read, write, exact_arrays
from lifecycle_support import require
from report_io import save

FIELDS=('distances','normals_w','hit_pos_w','pos_w','quat_w','frame_pos_w','frame_quat_w')

def cache_links(sensor):
    cached=sensor._cached_data
    return dict(cache_valid=bool(sensor._cache_valid),
                fields={n:getattr(cached,n) is getattr(sensor,'_'+n) for n in FIELDS})

def differences(a,b,path='$',out=None):
    if out is None:out=[]
    if type(a)!=type(b):out.append(dict(path=path,kind='type',before=type(a).__name__,after=type(b).__name__))
    elif isinstance(a,dict):
        for k in sorted(set(a)|set(b)):
            if k not in a or k not in b:out.append(dict(path=path+'.'+k,kind='missing'))
            else:differences(a[k],b[k],path+'.'+k,out)
    elif isinstance(a,list):
        if len(a)!=len(b):out.append(dict(path=path,kind='length',before=len(a),after=len(b)))
        for i,(x,y) in enumerate(zip(a,b)):differences(x,y,path+'[%d]'%i,out)
    elif a!=b:out.append(dict(path=path,kind='value',before=a,after=b))
    return out

def restore_boundary(portable,source_path,sim,sensors,read_outputs,out):
    source,schema=read(source_path)
    def snapshot(label):
        raw,meta=portable.snapshot();meta=json.loads(json.dumps(meta))
        write(out/label,raw,meta)
        proof=dict(arrays=exact_arrays(source,raw),schema_exact=meta==schema,
                   schema_differences=differences(schema,meta),
                   cache=cache_links(sensors['foot_height_scan']))
        save(out/(label+'-proof.json'),proof)
        return proof
    snapshot('receiver-before-load')
    restored=portable.restore(source_path)
    before=snapshot('receiver-loaded')
    require(before['arrays']['passed'] and before['schema_exact'],'Loaded state differs before sensing')
    require(before['cache']['cache_valid'] and all(before['cache']['fields'].values()),'Source sensor cache is not coherent')
    sim.sense()
    # This source method intentionally creates new Torch tensors while the old
    # cached TerrainHeightData still points to the prior tensors. Record this
    # boundary before invoking the existing normal cache invalidation contract.
    intermediate=snapshot('receiver-sense-only')
    for sensor in sensors.values():sensor._invalidate_cache()
    sensor_outputs=read_outputs()
    after=snapshot('receiver-refreshed')
    result=dict(portable_restore=restored,before=before,intermediate=intermediate,after=after,
                sense_calls=1,observation_history_pushes=0,physics_integrations=0,
                sensor_output_fields=len(sensor_outputs),fields_exempted=[],
                passed=after['arrays']['passed'] and after['schema_exact'] and after['cache']['cache_valid'] and all(after['cache']['fields'].values()))
    save(out/'rebuild-boundary.json',result)
    require(result['passed'],'Post-rebuild complete cache refresh differs; raw snapshots retained')
    return result
