"""Whole-batch selected-state rollback and replay contract. No physics claims."""
import numpy as np
from lifecycle_support import require


def exact(a,b):return a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()


def topology_signature(topology):
    """Retain schemas and object/storage alias equivalence across fresh outputs.
    Exact rollback separately requires every original binding/address. Replay
    permits fresh Torch outputs and containers, but reference/native Warp
    metadata and all recorded layout/offset/alias relations remain constrained.
    """
    owners={};storage={};result={}
    for path,node in sorted(topology.items()):
        kind=node['kind'];owner=node['owner'];owner_key=owners.setdefault(owner,path)
        if kind=='reference':result[path]=dict(kind=kind,owner=owner,type=node['type']);continue
        row=dict(kind=kind,alias=owner_key)
        if kind=='torch':
            meta=node['metadata'];ptr,base,offset=meta[5:8]
            # Zero-sized tensors have no dereferenceable storage relation.
            key=('torch',base) if base else ('empty',owner)
            row.update(metadata=[*meta[:5],offset,meta[8]],storage_alias=storage.setdefault(key,path),byte_offset=ptr-base)
        elif kind=='warp':row['metadata']=node['metadata']
        result[path]=row
    return result


def compare_records(reference,actual,exact_topology,outputs):
    sizes={}
    for group in ('data','owners','rng'):
        a,b=reference[group],actual[group]
        require(a.keys()==b.keys(),'Transaction '+group+' roster differs')
        for name,value in a.items():require(exact(value,b[name]),'Transaction '+group+' bytes differ: '+name)
        sizes[group]=len(a)
    require(reference['scalars']==actual['scalars'],'Transaction owner scalars/trace differ')
    if exact_topology:
        require(reference['topology']==actual['topology'],'Rollback original bindings/layouts differ')
    else:
        require(topology_signature(reference['topology'])==topology_signature(actual['topology']),
                'Replay schema/layout/object/storage alias relations differ')
    if outputs:
        a,b=reference['outputs'],actual['outputs'];require(a.keys()==b.keys(),'Replay sensor roster differs')
        for n,v in a.items():require(exact(v,b[n]),'Replay sensor bytes differ: '+n)
        require(exact(reference['delayed'],actual['delayed']),'Replay BAM delay differs')
        sizes['sensor_outputs']=len(a)
    return dict(all_recorded_bytes_exact=True,arrays=sizes,scalars=len(reference['scalars']),
        topology_entries=len(reference['topology']),original_bindings_exact=exact_topology,
        replay_alias_signature_exact=not exact_topology,opaque_native_contents_validated=False,
        per_environment_RNG_isolation_claim=False)
