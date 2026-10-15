"""Portable numeric checkpoint bound to freshly constructed reviewed owners.

No pickle, native pointer import, arbitrary object construction or native work.
The receiver must have the same explicit owner schema and alias relations.
"""
import hashlib,json,struct
from pathlib import Path
import numpy as np
from lifecycle_support import require
from native_state_bridge import StrictOwnerSnapshot,model_tree
from head_contact_data_commit_r3 import describe
from scene_evidence import array_meta
from report_io import save

def scalar(value):
    if type(value) is float:return ['float64',struct.pack('!d',value).hex()]
    require(value is None or type(value) in (str,int,bool),'Unsupported scalar')
    return [type(value).__name__,value]

def unscalar(row):
    if row[0]=='float64':return struct.unpack('!d',bytes.fromhex(row[1]))[0]
    require(row[0] in ('NoneType','str','int','bool'),'Unknown scalar encoding')
    return row[1]

def exact_arrays(a,b):
    missing=sorted(set(a)^set(b));different=[]
    for n in sorted(set(a)&set(b)):
        if a[n].dtype!=b[n].dtype or a[n].shape!=b[n].shape or a[n].tobytes()!=b[n].tobytes():different.append(n)
    return dict(passed=not missing and not different,fields=len(a),missing=missing,different=different,
                criterion='dtype_shape_and_every_byte; no numeric exemptions')

def write(path,raw,metadata):
    path=Path(path);require(not path.with_suffix('.npz').exists() and not path.with_suffix('.json').exists(),'Checkpoint already exists')
    np.savez_compressed(path.with_suffix('.npz'),**raw)
    save(path.with_suffix('.json'),dict(metadata=metadata,arrays=array_meta(raw),npz_sha256=hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()))

def read(path):
    path=Path(path);meta=json.loads(path.with_suffix('.json').read_bytes())
    require(hashlib.sha256(path.with_suffix('.npz').read_bytes()).hexdigest()==meta['npz_sha256'],'Checkpoint container changed')
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as data:raw={n:data[n].copy() for n in data.files}
    require(array_meta(raw)==meta['arrays'],'Checkpoint numeric metadata differs')
    return raw,meta['metadata']

class PortableState:
    def __init__(self,sim,roots,torch,wp):
        self.sim,self.roots,self.torch,self.wp=sim,roots,torch,wp
        self.refresh()

    def refresh(self):
        # Sensing may legitimately rebind host-side temporary tensors. Never
        # audit stale snapshot nodes in place of the current live root graph.
        self.data=describe(self.sim.wp_data,self.wp.array)[1]
        self.owners=StrictOwnerSnapshot(self.roots,self.torch,self.wp)
        self.model,self.model_meta=model_tree(self.sim.wp_model,self.wp)

    def schema(self):
        refs={};storage={};nodes=[]
        # Tie selected Torch/Warp storage to full numeric Data where it aliases.
        ranges=[(n,int(a.ptr),int(a.ptr)+int(a.capacity)) for n,a in self.data.items() if a.capacity]
        def alias(pointer,size,key):
            hits=[(n,pointer-start) for n,start,end in ranges if start<=pointer and pointer+size<=end]
            if hits:return ['Data',sorted(hits)[0]]
            return ['owned_storage',storage.setdefault(key,len(storage))]
        def ref(r):
            if r[0]=='scalar':return ['scalar',scalar(r[1])]
            if r[0]=='node':return ['node',r[1]]
            value=r[1]
            return ['fresh_reference',refs.setdefault(id(value),len(refs)),type(value).__module__+'.'+type(value).__qualname__]
        for node in self.owners.nodes:
            value,kind=node['original'],node['kind'];row=dict(kind=kind,type=type(value).__module__+'.'+type(value).__qualname__)
            if kind=='torch':
                base=value.untyped_storage().data_ptr();size=value.numel()*value.element_size()
                row.update(shape=list(value.shape),strides=list(value.stride()),dtype=str(value.dtype),device=str(value.device),offset=value.storage_offset(),alias=alias(value.data_ptr(),size,('storage',str(value.device),base) if size else ('empty',id(value))))
            elif kind=='warp':
                row.update(shape=list(value.shape),strides=list(value.strides),dtype=str(value.dtype),device=str(value.device),capacity=int(value.capacity),alias=alias(int(value.ptr),int(value.capacity),('storage',str(value.device),int(value.ptr)) if value.size else ('empty',id(value))))
            elif kind in ('dict','object'):row['children']=[[scalar(k),ref(v)] for k,v in node['children'].items()]
            else:row['children']=[ref(v) for v in node['children']]
            nodes.append(row)
        return dict(nodes=nodes,roots=[[label,ref(r)] for label,_,_,r in self.owners.roots],
                    model_scalars=self.model_meta['scalars'],model_opaque_types={n:r[1] for n,r in self.model_meta['opaque'].items()})

    def snapshot(self):
        self.wp.synchronize();self.torch.cuda.synchronize()
        self.refresh()
        raw={n:a.numpy().copy() for n,a in self.data.items()}
        raw.update({n:a.numpy().copy() for n,a in self.model.items()})
        for i,node in enumerate(self.owners.nodes):
            v=node['original']
            if node['kind']=='torch':raw['Owner.%04d'%i]=v.detach().cpu().numpy().copy()
            elif node['kind']=='warp':raw['Owner.%04d'%i]=v.numpy().copy()
        raw['RNG.cpu']=self.torch.get_rng_state().numpy().copy()
        raw['RNG.cuda']=self.torch.cuda.get_rng_state('cuda:0').cpu().numpy().copy()
        return raw,self.schema()

    def restore(self,path):
        raw,schema=read(path);current,current_schema=self.snapshot()
        # This bounded recovery targets the fully initialized pre-macro state.
        # Matching schema includes the actual scalar state, not only array shapes.
        require(json.loads(json.dumps(current_schema))==schema,'Fresh initialized schema/scalars/aliases differ; no improvised object reconstruction')
        require(set(raw)==set(current),'Portable field roster differs')
        require(all(raw[n].shape==current[n].shape and raw[n].dtype==current[n].dtype for n in raw),'Portable shape/dtype differs')
        # Prove actual loading rather than simply accepting a deterministic seed.
        q=self.data['Data.qvel'].numpy().copy();q[0,0]+=np.float32(.125)
        self.data['Data.qvel'].assign(q);self.wp.synchronize()
        require(self.data['Data.qvel'].numpy().tobytes()!=raw['Data.qvel'].tobytes(),'Restore challenge did not change qvel')
        for n,a in self.model.items():
            # All Model fields are numeric in the pinned model. Native render
            # resource IDs are outside this object and are never imported.
            if a.size:a.assign(raw[n])
        for n,a in self.data.items():
            if a.size:a.assign(raw[n])
        with self.torch.no_grad():
            for i,node in enumerate(self.owners.nodes):
                value=node['original'];name='Owner.%04d'%i
                if node['kind']=='torch':value.copy_(self.torch.as_tensor(raw[name],device=value.device,dtype=value.dtype))
                elif node['kind']=='warp' and value.size:value.assign(raw[name])
        self.torch.set_rng_state(self.torch.from_numpy(raw['RNG.cpu']))
        self.torch.cuda.set_rng_state(self.torch.from_numpy(raw['RNG.cuda']),'cuda:0')
        after,after_schema=self.snapshot();proof=exact_arrays(raw,after)
        require(proof['passed'] and json.loads(json.dumps(after_schema))==schema,'Portable restore is not exact')
        return dict(**proof,challenge_restored=True,native_pointers_imported=0,process_local_owners_recreated=True,
                    scope='complete initialized pre-macro numeric state and selected owners; arbitrary mid-rollout checkpoint remains open')
