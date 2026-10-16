"""Prove which external views alias numeric Data before allowing trial mutation."""
from lifecycle_support import require

def data_view_aliases(owner_snapshot,data_snapshot):
    allowed=set();proof=[]
    for i,node in enumerate(owner_snapshot.nodes):
        v=node['original'];kind=node['kind']
        if kind=='torch' and v.numel():
            require(all(s>=0 for s in v.stride()), 'Negative Data view stride')
            lo=int(v.data_ptr());hi=lo+(1+sum((n-1)*s for n,s in zip(v.shape,v.stride())))*v.element_size()
        elif kind=='warp' and v.size:
            lo=int(v.ptr);hi=lo+int(v.capacity)
        else:continue
        for name,a in data_snapshot.arrays.items():
            if not a.size or str(a.device)!=str(v.device):continue
            base=int(a.ptr);end=base+int(a.capacity)
            if hi<=base or lo>=end:continue
            require(base<=lo<hi<=end, 'External view partially overlaps numeric Data allocation')
            allowed.add(id(v));proof.append(dict(node=i,owner=id(v),kind=kind,view_begin=lo,view_end=hi,
                Data_array=name,Data_begin=base,Data_end=end,device=str(a.device)))
    return frozenset(allowed),proof
