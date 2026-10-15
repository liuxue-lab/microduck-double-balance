"""Deterministic active Contact row permutation before native make_constraint.

No comparator exemptions or numerical tolerances. The original full-Data and
stage byte comparison remains unchanged. All16 Contact arrays move together;
no contact is dropped, no world row is split, inactive tails remain untouched.
This policy must run after collision and before any constraint address exists.
"""
import hashlib,json
import numpy as np
from lifecycle_support import require
FIELDS=('dim','dist','efc_address','flex','frame','friction','geom','geomcollisionid',
        'includemargin','pos','solimp','solref','solreffriction','type','vert','worldid')
sha=lambda a:hashlib.sha256(a.tobytes()).hexdigest()
canon=lambda x:json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def permutation(raw,n,layout):
    require(type(n) is int and 0<=n<=256,'Packed Contact overflow; no clipping')
    require(set(raw)==set(layout)==set(FIELDS),'Complete16-field Contact schema required')
    for name,a in raw.items():
        require(a.shape==tuple(layout[name]['shape']) and str(a.dtype)==layout[name]['dtype'] and a.shape[0]==256,'Contact layout differs: '+name)
        require(np.isfinite(a[:n]).all(),'Nonfinite active Contact field: '+name)
    require(((raw['worldid'][:n]>=0)&(raw['worldid'][:n]<1)).all(),'Invalid Contact world')
    require(((raw['geom'][:n]>=0)&(raw['geom'][:n]<85)).all(),'Scoped rigid geom ID differs')
    require((raw['geom'][:n,0]!=raw['geom'][:n,1]).all(),'Self-geom Contact invalid')
    require(np.isin(raw['dim'][:n],[1,3,4,6]).all() and np.isin(raw['type'][:n],[1,2,3]).all(),'Contact type/dimension differs')
    require((raw['efc_address'][:n]==-1).all(),'Must order before constraint construction; live addresses refused')
    keys=[(int(raw['worldid'][i]),*map(int,raw['geom'][i]),int(raw['geomcollisionid'][i])) for i in range(n)]
    require(len(set(keys))==n,'Ambiguous Contact identity; no geometry-only merge or tie breaking')
    order=np.asarray(sorted(range(n),key=lambda i:keys[i]),np.int64)
    out={name:a.copy() for name,a in raw.items()}
    for name,a in raw.items():out[name][:n]=a[order]
    return out,order

def evidence(before,after,order,n,serial):
    rows={}
    for name,a in before.items():
        b=after[name]
        require(b.dtype==a.dtype and b.shape==a.shape and b[:n].tobytes()==a[order].tobytes() and b[n:].tobytes()==a[n:].tobytes(),'Row permutation or untouched tail proof failed: '+name)
        rows[name]=dict(shape=list(a.shape),dtype=str(a.dtype),active_before=a[:n].tolist(),active_before_sha256=sha(a[:n]),
            active_after_sha256=sha(b[:n]),inactive_sha256=sha(a[n:]),full_before_sha256=sha(a),full_after_sha256=sha(b))
    return dict(protocol='live-preconstraint-contact-order-v1',serial=serial,nacon=n,order=order.tolist(),
        identity_key=['worldid','geom0','geom1','geomcollisionid'],arrays=rows,
        exact_replay_exemptions=[],inactive_tails_unchanged=True,constraint_addresses_unallocated=True,
        native_forward_added=0,integrations_added=0)

def verify_evidence(proof):
    require(proof['protocol']=='live-preconstraint-contact-order-v1' and proof['exact_replay_exemptions']==[],'Ordering protocol/exemptions differ')
    n=proof['nacon'];order=np.asarray(proof['order'],np.int64)
    require(type(n) is int and 0<=n<=256 and sorted(order.tolist())==list(range(n)),'Nonbijective active permutation')
    arrays=proof['arrays'];require(set(arrays)==set(FIELDS),'Ordering evidence schema differs')
    active={}
    for name,row in arrays.items():
        a=np.asarray(row['active_before'],dtype=np.dtype(row['dtype'])).reshape((n,*row['shape'][1:]))
        require(sha(a)==row['active_before_sha256'] and sha(a[order])==row['active_after_sha256'],'Active row bytes or permutation changed')
        active[name]=a
    require((active['efc_address']==-1).all(),'Ordering applied after EFC allocation')
    keys=[(int(active['worldid'][i]),*map(int,active['geom'][i]),int(active['geomcollisionid'][i])) for i in range(n)]
    require(len(set(keys))==n and order.tolist()==sorted(range(n),key=lambda i:keys[i]),'Incorrect world/geom/ordinal order')
    require(proof['inactive_tails_unchanged'] and proof['constraint_addresses_unallocated'] and proof['native_forward_added']==proof['integrations_added']==0,'Ordering proof scope differs')
    return True

class ContactOrderHook:
    """One ordering pass inside each already counted native forward; eager only."""
    CALL_CAP=1424
    @classmethod
    def validate_capacity(cls,nworld,naconmax,cap):
        require(nworld==1 and naconmax==256 and type(cap) is int and cap==cls.CALL_CAP,'Pinned single-world Contact capacity/call cap required')
    def __init__(self,model,data,wp,gate,counts,layout,cap):
        self.model,self.data,self.wp,self.gate,self.counts=model,data,wp,gate,counts
        self.validate_capacity(data.nworld,data.naconmax,cap)
        self.layout=layout;self.cap=cap;self.last=None;self.serial=0;self.digest=hashlib.sha256();self.assigns=0
        self.buffers={n:getattr(data.contact,n) for n in FIELDS}
        require({n for n,v in vars(data.contact).items() if isinstance(v,wp.array)}==set(FIELDS),'Contact native array schema changed')
        self.bindings={n:self.token(a) for n,a in self.buffers.items()}
        self.apply_counted=gate.limited('Contact_order_apply',cap,self.apply)
    @staticmethod
    def token(a):return (id(a),a.ptr,int(a.capacity),tuple(a.shape),tuple(a.strides),str(a.dtype),str(a.device))
    def wrap(self,native):
        def ordered(model,data):
            require(model is self.model and data is self.data,'Foreign contact Model/Data')
            require(self.counts.get('GPU_forward_attempted',0)==self.counts.get('GPU_forward_completed',0)+1==self.serial+1,'Contact ordering must be inside exactly one counted forward')
            primary={n:getattr(data,n).numpy().tobytes() for n in ('qpos','qvel','time','ctrl')}
            result=native(model,data)
            self.apply_counted()
            require(all(getattr(data,n).numpy().tobytes()==b for n,b in primary.items()),'Collision/order changed primary state or controls')
            return result
        return ordered
    def apply(self):
        self.wp.synchronize()
        for n,a in self.buffers.items():require(getattr(self.data.contact,n) is a and self.token(a)==self.bindings[n],'Contact allocation/binding changed: '+n)
        raw={n:a.numpy().copy() for n,a in self.buffers.items()};n=int(self.data.nacon.numpy()[0])
        proposal,order=permutation(raw,n,self.layout)
        # In-place uploads preserve every original Warp object and allocation.
        for name,a in self.buffers.items():
            if proposal[name].tobytes()!=raw[name].tobytes():
                require(self.assigns<16*self.cap,'Contact assignment cap exhausted')
                self.assigns+=1;self.counts['Contact_array_assign_attempted']=self.assigns;self.gate.persist()
                a.assign(proposal[name]);self.wp.synchronize()
                self.counts['Contact_array_assign_completed']=self.counts.get('Contact_array_assign_completed',0)+1;self.gate.persist()
        after={n:a.numpy().copy() for n,a in self.buffers.items()}
        for name,a in self.buffers.items():require(self.token(a)==self.bindings[name],'Contact assignment reallocated buffer')
        self.serial+=1;proof=evidence(raw,after,order,n,self.serial);verify_evidence(proof)
        self.last=proof;self.digest.update(canon(proof))
    def summary(self):return dict(protocol='live-preconstraint-contact-order-v1',calls=self.serial,array_assignments=self.assigns,
        call_cap=self.cap,array_assignment_cap=16*self.cap,proof_stream_sha256=self.digest.hexdigest(),
        comparator_unchanged=True,installed_sources_modified=False,opaque_native_contents_cloned=False)
