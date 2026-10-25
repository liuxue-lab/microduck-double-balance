"""Eager two-world state bridge using real Warp/Torch object operations.

No simulator imports or environment construction. A future bounded worker must
supply its initialized, pinned objects and explicit owner roots. Native opaque
contents remain outside the byte snapshot; restore leaves sense stale.
Default execution budgets are zero. NumPy fixtures do not certify GPU behavior.
"""
from dataclasses import dataclass, fields, is_dataclass
import hashlib, json, math, struct, time
import numpy as np
from lifecycle_support import require
from head_contact_data_commit_r3 import describe
from sensor_state_snapshot import DataSnapshot, OwnerSnapshot, digest, warp_meta
from scene_evidence import owner_state, array_meta
from trial_invariants import data_view_aliases

CALLBACKS=('passive','control','act_dyn','act_gain','act_bias','sensor','contactfilter')
GRAPH_NAMES=('step_graph','forward_graph','reset_graph','sense_graph')
APPROVED_LIMITS={k:0 for k in ('state_snapshot','state_restore','integrations','RK4_integrator','GPU_forward','position_write','sense','public_clock_write')}

def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()

def bits(value):
    if type(value) is float:return ('float64_bits',struct.pack('!d',value).hex())
    return (type(value).__name__,value)

class CallBudget:
    """Attempt counts are durable before calling native code; never roll back."""
    def __init__(self,limits=None,counts=None,persist=lambda:None,sync=lambda:None,deadline=None):
        self.limits=dict(APPROVED_LIMITS if limits is None else limits)
        require(set(self.limits)==set(APPROVED_LIMITS) and all(type(n) is int and n>=0 for n in self.limits.values()),'Explicit complete nonnegative call limits required')
        self.counts={} if counts is None else counts
        self.persist,self.sync=persist,sync
        self.deadline=time.monotonic()+300 if deadline is None else deadline
    def reserve(self,**cost):
        require(time.monotonic()<self.deadline,'Wall budget exhausted')
        for name,n in cost.items():
            require(name in self.limits and type(n) is int and n>=0,'Unknown budget or invalid cost')
            require(self.counts.get(name+'_attempted',0)+n<=self.limits[name],name+' budget exhausted before mutation')
    def run(self,name,fn):
        self.reserve(**{name:1})
        a,c=name+'_attempted',name+'_completed'
        self.counts[a]=self.counts.get(a,0)+1;self.persist()
        result=fn();self.sync()
        self.counts[c]=self.counts.get(c,0)+1;self.persist()
        require(time.monotonic()<self.deadline,'Wall budget expired after call; no retry credit')
        return result

class StrictOwnerSnapshot(OwnerSnapshot):
    """Same native copy operations; exact scalar bits also preserve NaN and -0."""
    def assert_restored(self,mutable_data_views=frozenset()):
        self.preflight()
        def binding(actual,ref,label):
            expected=self.value(ref)
            if ref[0]=='scalar':require(bits(actual)==bits(expected),'Scalar bits differ: '+label)
            else:require(actual is expected,'Original owner binding not restored: '+label)
        for label,obj,name,ref in self.roots:binding(getattr(obj,name),ref,label)
        for index,node in enumerate(self.nodes):
            v,k=node['original'],node['kind']
            if k in ('torch','warp'):
                raw=v.detach().cpu().numpy() if k=='torch' else v.numpy()
                require(id(v) in mutable_data_views or digest(raw)==node['hash'],'Selected owner bytes differ')
            elif k in ('dict','object'):
                actual=v if k=='dict' else vars(v)
                require(set(actual)==set(node['children']),'Container fields differ')
                for key,ref in node['children'].items():binding(actual[key],ref,'node'+str(index)+'.'+str(key))
            elif k in ('tuple','list'):
                require(len(v)==len(node['children']),'Sequence length differs')
                for actual,ref in zip(v,node['children']):binding(actual,ref,'node'+str(index))

def model_tree(model,wp):
    """Automatic numeric arrays, static scalars and identity-only opaque leaves."""
    arrays={};layout={};scalars={};opaque={};objects={};seen={}
    def walk(v,p):
        if isinstance(v,wp.array):
            layout[p]=(id(v),warp_meta(v))
            if 'Texture' in str(v.dtype):opaque[p]=(id(v),str(type(v)))
            else:arrays[p]=v
        elif v is None or type(v) in (int,float,str,bool):scalars[p]=bits(v)
        elif is_dataclass(v):
            if id(v) in seen:objects[p]=('alias',seen[id(v)]);return
            seen[id(v)]=p;names=tuple(f.name for f in fields(v))
            require(set(vars(v))==set(names),'Dynamic Model schema: '+p)
            objects[p]=(id(v),type(v).__module__,type(v).__qualname__,names)
            for n in names:walk(getattr(v,n),p+'.'+n)
        else:opaque[p]=(id(v),str(type(v)))
    walk(model,'Model')
    require(arrays and sum(int(v.capacity) for v in arrays.values())<=64*1024*1024,'Model numeric capacity differs')
    return arrays,dict(layout=layout,scalars=scalars,opaque=opaque,objects=objects)

def native_handles(sim,scene,wp):
    """Observed native identities; deliberately not a claim to clone BVH internals."""
    ctx=scene.sensor_context
    require(ctx is not None and sim._sensor_context is ctx and not ctx.has_cameras,'Pinned noncamera sensor context required')
    require(not sim.use_cuda_graph and all(getattr(sim,n) is None for n in GRAPH_NAMES),'Eager bridge refuses native CUDA graphs')
    mask=sim._reset_mask
    require(mask._wp_array is sim._reset_mask_wp,'Reset mask bridge differs')
    rc=ctx.render_context
    arrays,meta=model_tree(rc,wp)
    return dict(sim=id(sim),scene=id(scene),data=id(sim.wp_data),model=id(sim.wp_model),context=id(ctx),
        mask=(id(mask),id(sim._reset_mask_wp),warp_meta(sim._reset_mask_wp),id(mask._torch_stream)),
        graph_ids={n:id(getattr(sim,n)) for n in GRAPH_NAMES},render_meta=meta,
        bvh_identity=id(rc.bvh),bvh_handle=int(rc.bvh_id),opaque_contents_cloned=False)

@dataclass
class Checkpoint:
    data:object
    owners:object
    model_arrays:dict
    model_meta:dict
    model_hashes:dict
    dt:np.ndarray
    integrator:int
    handles:dict
    rng:dict
    mutable_views:frozenset
    alias_proof:list

class NativeStateBridge:
    """Restore complete numeric Data + explicitly selected owners + private RNG.

Global RNG is asserted unchanged, never rewound to hide accidental consumers.
Only the declared timestep/integrator Model override may be restored. Any other
Model mutation, binding change, copy failure or native-handle change poisons the
bridge. Numeric restoration does not mark contacts/sense fresh.
"""
    def __init__(self,sim,scene,roots,bank,torch,wp,budget):
        self.sim,self.scene,self.bank,self.torch,self.wp,self.budget=sim,scene,bank,torch,wp,budget
        self.roots=list(roots)
        slots={(id(o),n) for _,o,names in self.roots for n in names}
        for n in ('states','seeds','draw_counts'):
            if (id(bank),n) not in slots:self.roots.append(('private_rng',bank,(n,)))
        self.saved=None;self.phase='idle';self.generation=0;self.solved=None;self.sensed=None
    def sync(self):self.wp.synchronize();self.torch.cuda.synchronize()
    def global_rng(self):
        return dict(cpu=digest(self.torch.get_rng_state().numpy()),cuda=digest(self.torch.cuda.get_rng_state('cuda:0').cpu().numpy()))
    def preflight(self):
        require(self.phase!='poisoned','State bridge poisoned')
        m,d=self.sim.wp_model,self.sim.wp_data
        require(d.nworld==1 and tuple(d.qpos.shape)==(1,35) and tuple(d.qvel.shape)==(1,32),'Pinned single-world layout differs')
        require(all(getattr(m.callback,k) is None for k in CALLBACKS),'Native callback present')
        require(tuple(m.opt.timestep.shape)==(1,),'Shared timestep required; no silent per-world expansion')
        require(len(self.bank.states)==9 and set(self.bank.states)==set(self.bank.seeds)==set(self.bank.draw_counts),'Private stream roster differs')
        return native_handles(self.sim,self.scene,self.wp)
    def checkpoint(self):
        require(self.phase=='idle','Nested or poisoned transaction')
        def capture():
            self.sync();handles=self.preflight()
            data=DataSnapshot(self.sim.wp_data,self.wp)
            owners=StrictOwnerSnapshot(self.roots,self.torch,self.wp)
            arrays,meta=model_tree(self.sim.wp_model,self.wp)
            allow,proof=data_view_aliases(owners,data)
            saved=Checkpoint(data,owners,arrays,meta,{n:digest(v.numpy()) for n,v in arrays.items()},
                self.sim.wp_model.opt.timestep.numpy().copy(),self.sim.wp_model.opt.integrator,handles,self.global_rng(),allow,proof)
            self.saved=saved;self.phase='trial';self.invalidate();return saved
        try:return self.budget.run('state_snapshot',capture)
        except Exception:
            # A denied reservation made no mutation; a partial capture cannot run.
            if self.budget.counts.get('state_snapshot_attempted',0):self.phase='poisoned'
            raise
    def invalidate(self):
        self.generation+=1;self.solved=None;self.sensed=None
    def check_protected(self):
        require(self.saved is not None and self.phase=='trial','Active checkpoint required')
        s=self.saved
        require(self.preflight()==s.handles,'Native handle/reset binding changed')
        s.data.preflight();s.owners.assert_restored(mutable_data_views=s.mutable_views)
        require(self.global_rng()==s.rng,'Default RNG changed; do not rewind it to hide a draw')
        self._model_preflight()
    def _model_preflight(self):
        s=self.saved;arrays,meta=model_tree(self.sim.wp_model,self.wp)
        for k in ('layout','opaque','objects'):require(meta[k]==s.model_meta[k],'Model binding/opaque/schema changed')
        def stable(v):return {k:x for k,x in v.items() if k!='Model.opt.integrator'}
        require(stable(meta['scalars'])==stable(s.model_meta['scalars']),'Unapproved Model scalar changed')
        require(set(arrays)==set(s.model_arrays),'Model array roster differs')
        require(all(n=='Model.opt.timestep' or digest(v.numpy())==s.model_hashes[n] for n,v in arrays.items()),'Unapproved Model numeric change')
    def restore(self):
        require(self.saved is not None and self.phase=='trial','No active rollback')
        def perform():
            s=self.saved;self.sync()
            require(self.preflight()==s.handles,'Native handle/reset binding changed before rollback')
            require(self.global_rng()==s.rng,'Default RNG changed before rollback')
            s.data.preflight();s.owners.preflight();self._model_preflight()
            self.sim.wp_model.opt.timestep.assign(s.dt)
            self.sim.wp_model.opt.integrator=s.integrator
            s.data.restore();s.owners.restore();self.sync()
            s.data.assert_restored();s.owners.assert_restored()
            arrays,meta=model_tree(self.sim.wp_model,self.wp)
            require(meta==s.model_meta and {n:digest(v.numpy()) for n,v in arrays.items()}==s.model_hashes,'Model not exactly restored')
            require(self.preflight()==s.handles and self.global_rng()==s.rng,'Bindings or RNG not restored')
            self.invalidate()
            return dict(data_exact=True,owners_exact=True,rng_exact=True,model_exact=True,bindings_exact=True,
                proof_origin='Measured after actual copies and synchronization',native_opaque_contents_cloned=False,
                contacts_fresh=False,sense_fresh=False,data_arrays=len(s.data.arrays),private_streams=len(self.bank.states))
        try:return self.budget.run('state_restore',perform)
        except Exception:self.phase='poisoned';self.invalidate();raise
    def restore_into(self,coordinator):
        """Connect measured copy/byte checks to the prior transaction guard."""
        require(coordinator.phase in ('restore_required','restore_for_replay','abort_required'),'Coordinator not at rollback boundary')
        try:evidence=self.restore()
        except Exception:
            coordinator.restored(**{k:False for k in ('data_exact','owners_exact','rng_exact','model_exact','bindings_exact')})
            raise
        coordinator.restored(**{k:evidence[k] for k in ('data_exact','owners_exact','rng_exact','model_exact','bindings_exact')})
        return evidence
    def microstep(self,engine,accumulator,coordinates=(None,None)):
        """Link native stepping to automatic protection/freshness bookkeeping."""
        require(engine.model is self.sim.wp_model and engine.data is self.sim.wp_data and engine.wp is self.wp
                and engine.budget is self.budget,'Native engine/bridge identities or budgets differ')
        self.check_protected();self.invalidate()
        try:
            row=engine.microstep(accumulator,coordinates)
            self.sync();self.check_protected();return row
        except Exception:self.phase='poisoned';self.invalidate();raise
    def forward(self,native_forward):
        require(self.phase=='trial','Private trial required')
        self.budget.reserve(GPU_forward=1)
        self.check_protected();self.invalidate()
        try:
            self.budget.run('GPU_forward',lambda:native_forward(self.sim.wp_model,self.sim.wp_data))
            self.sync();self.check_protected();self.solved=self.generation
        except Exception:self.phase='poisoned';self.invalidate();raise
    def contacts_ready(self):
        require(self.phase=='trial' and self.solved==self.generation,'Fresh completed forward required')
    def sense(self):
        self.contacts_ready();self.budget.reserve(sense=1)
        try:
            self.budget.run('sense',self.sim.sense)
            for sensor in self.scene.sensors.values():sensor._invalidate_cache()
            self.sync();require(self.preflight()==self.saved.handles and self.global_rng()==self.saved.rng,'Sense changed native bindings or RNG')
            self.sensed=self.generation
        except Exception:self.phase='poisoned';self.invalidate();raise
    def endpoint_bytes(self):
        self.contacts_ready();require(self.sensed==self.generation,'Fresh explicit sense required')
        self.sync();raw,scalars,topology=owner_state(self.roots,self.torch,self.wp)
        arrays,meta=model_tree(self.sim.wp_model,self.wp)
        return canonical(dict(data=self.saved.data.fingerprint(),owners=array_meta(raw),scalars=scalars,topology=topology,
            model={n:digest(v.numpy()) for n,v in arrays.items()},model_meta=meta,handles=self.preflight(),rng=self.global_rng()))
