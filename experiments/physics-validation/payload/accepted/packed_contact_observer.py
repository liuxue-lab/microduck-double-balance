"""Strict offline observer for the pinned two-world pyramidal Contact layout.

Reads active packed rows via worldid, never assumes contiguous world slices.
The caller must prove that forward has refreshed the cache for this state.
Raw solver rows, their sum, and decoded contact-frame force stay separate.
"""
import math
import numpy as np

def require(ok,message):
    if not ok:raise RuntimeError(message)

def array(data,name,shape,dtype):
    a=data[name]
    require(isinstance(a,np.ndarray) and a.shape==shape and a.dtype==np.dtype(dtype),name+' layout differs')
    return a

def observe(data,*,state_generation,solved_generation):
    require(type(state_generation) is int and state_generation>=0
            and state_generation==solved_generation,'Fresh forward for this state required')
    count=array(data,'Data.nacon',(1,),np.int32)
    nefc=array(data,'Data.nefc',(1,),np.int32)
    force=data['Data.efc.force'];require(force.ndim==2 and force.shape[0]==1 and force.dtype==np.float32,'Single-world efc.force required')
    capacity=data['Data.contact.worldid'].shape[0];active=int(count[0])
    require(0<=active<=capacity,'Packed contact overflow; no truncation permitted')
    require(((0<=nefc)&(nefc<=force.shape[1])).all(),'Constraint overflow; no truncation permitted')
    fields={n:array(data,'Data.contact.'+n,(capacity,)+tail,dtype) for n,tail,dtype in
        [('worldid',(),np.int32),('geom',(2,),np.int32),('dist',(),np.float32),('dim',(),np.int32),
         ('type',(),np.int32),('friction',(5,),np.float32),('frame',(3,3),np.float32)]}
    address=data['Data.contact.efc_address']
    require(address.ndim==2 and address.shape[0]==capacity and address.dtype==np.int32,'Contact address layout differs')
    output=[[]];used=[set()]
    for cid in range(active):
        w=int(fields['worldid'][cid]);require(w == 0,'Invalid active contact world')
        geom=tuple(map(int,fields['geom'][cid]));dim=int(fields['dim'][cid]);kind=int(fields['type'][cid])
        require(all(0<=g<85 for g in geom) and geom[0]!=geom[1],'Scoped rigid geom IDs differ')
        require(dim in (1,3,4,6) and kind in (1,2,3),'Contact kind/dimension differs')
        require(all(np.isfinite(fields[n][cid]).all() for n in ('dist','friction','frame')),'Nonfinite active contact')
        require((fields['friction'][cid]>=0).all(),'Negative friction')
        needed=1 if dim==1 else 2*(dim-1)
        adr=address[cid];valid=adr[adr>=0]
        values=np.empty(0,np.float32);decoded=np.zeros(6,np.float32)
        if kind&1:
            # No stale, out-of-world, duplicate or silently clipped row access.
            require(len(valid)==needed and needed<=address.shape[1], 'Incomplete active pyramidal rows')
            require(np.array_equal(adr[:needed],np.arange(int(adr[0]),int(adr[0])+needed))
                    and (adr[needed:]==-1).all(),'Pyramid row addresses must be contiguous')
            require((valid<nefc[w]).all(),'Contact cache references inactive constraint rows')
            require(not (set(map(int,valid))&used[w]),'Duplicate constraint ownership in a world')
            used[w].update(map(int,valid));values=force[w,valid]
            require(np.isfinite(values).all(),'Nonfinite active solver force')
            if dim==1:decoded[0]=values[0]
            else:
                for axis in range(dim-1):
                    p,n=values[2*axis:2*axis+2]
                    decoded[0]+=p+n
                    decoded[axis+1]=(p-n)*fields['friction'][cid,axis]
        else:require((adr==-1).all(),'Sensor-only contact owns constraint rows')
        output[w].append(dict(packed_id=cid,world=w,geom=list(geom),type=kind,dim=dim,
            distance_m=float(fields['dist'][cid]),constraint=bool(kind&1),
            raw_solver_rows=values.tolist(),raw_solver_sum=math.fsum(map(float,values)),
            contact_frame_force=decoded.tolist(),
            world_frame_force=(decoded[:3]@fields['frame'][cid]).tolist(),
            world_frame_torque=(decoded[3:]@fields['frame'][cid]).tolist()))
    return output

def occupancy(rows,pairs):
    active={tuple(sorted(r['geom'])) for r in rows if r['constraint']}
    return {name:tuple(sorted(pair)) in active for name,pair in pairs.items()}

def semantic_world(rows):
    """Order-independent observations; this is not a substitute for exact replay."""
    import json
    return sorted(json.dumps({k:v for k,v in r.items() if k!='packed_id'},sort_keys=True) for r in rows)
