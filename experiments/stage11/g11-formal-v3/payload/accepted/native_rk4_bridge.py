"""Instrument the pinned eager native RK4 entry; no CUDA graph or split-step.

Takes real fw/mw namespaces supplied by the bounded worker. Intermediate
position corrections use the previously reviewed two-world arithmetic. This
is a single-microstep bridge, not a completed adaptive RL/environment driver.
"""
from contextlib import contextmanager
import hashlib, threading
import numpy as np
from lifecycle_support import require
from head_contact_data_commit_r3 import describe
from sensor_state_snapshot import digest
from native_state_bridge import CALLBACKS, canonical
from batch_position_candidate import batch_stage_position, OTHER
from packed_contact_observer import observe

class NativeRK4Bridge:
    def __init__(self,fw,mw,model,data,wp,budget):
        self.fw,self.mw,self.model,self.data,self.wp,self.budget=fw,mw,model,data,wp,budget
        self.originals={(id(module),name):getattr(module,name) for module in (fw,mw) for name in ('forward','rungekutta4','step')}
        self.native_step=fw.step;self.native_forward=fw.forward;self.native_rk=fw.rungekutta4
        self.active=False;self.phase=None;self.stages=[];self.rks=0;self.prepared=None;self.thread=None;self.poisoned=False
    def _check(self,m,d):
        require(m is self.model and d is self.data,'Foreign Model/Data owner')
        require(self.active and not self.poisoned and threading.get_ident()==self.thread,'Native call outside owned hook scope or poisoned')
        require(d.nworld==1 and tuple(d.qpos.shape)==(1,35) and tuple(d.qvel.shape)==(1,32),'Single-world layout differs')
        require(m.opt.integrator==1 and tuple(m.opt.timestep.shape)==(1,),'Native RK4/shared dt required')
        require(all(getattr(m.callback,k) is None for k in CALLBACKS),'Per-stage callback forbidden')
        if self.phase=='micro':
            desc,_,owners,_=describe(d,self.wp.array)
            require((desc,owners)==self.binding,'Data allocation/schema changed during RK4')
    @contextmanager
    def hooks(self):
        require(not self.active,'Reentrant native hooks refused')
        for module in (self.fw,self.mw):
            for n in ('forward','rungekutta4','step'):
                require(getattr(module,n) is self.originals[id(module),n],'Native entry changed before instrumentation')
        self.active=True;self.thread=threading.get_ident();patched=[]
        try:
            for module in (self.fw,self.mw):
                for n,fn in (('forward',self.forward),('rungekutta4',self.rk),('step',self.blocked_step)):
                    patched.append((module,n));setattr(module,n,fn)
            yield self
        finally:
            for module,n in reversed(patched):setattr(module,n,self.originals[id(module),n])
            self.active=False;self.thread=None;self.phase=None
    def blocked_step(self,*args,**kwargs):raise RuntimeError('Only bounded microstep entry may call native step')
    def _write_position(self,value):
        d=self.data;before=d.qpos.numpy();owner=(id(d.qpos),d.qpos.ptr,d.qpos.capacity)
        qvel=d.qvel.numpy().tobytes();clock=d.time.numpy().tobytes()
        require(value.dtype==np.float32 and value.shape==(1,35) and np.isfinite(value).all(),'Position publication layout differs')
        require(before[:,OTHER].tobytes()==value[:,OTHER].tobytes(),'Quaternion publication forbidden')
        self.budget.run('position_write',lambda:d.qpos.assign(value))
        require(owner==(id(d.qpos),d.qpos.ptr,d.qpos.capacity) and d.qpos.numpy().tobytes()==value.tobytes(), 'Position owner or bytes differ')
        require(qvel==d.qvel.numpy().tobytes() and clock==d.time.numpy().tobytes(),'Position write changed velocity/time')
    def forward(self,m,d):
        self._check(m,d);require(self.phase=='micro' and len(self.stages)<4,'Forward outside RK stage')
        k=len(self.stages);record=None
        if k and any(c is not None for c in self.coordinates):
            value,record=batch_stage_position(self.acc.value,self.start,self.stages[-1]['qvel'],d.qpos.numpy(),self.dt,k,self.coordinates)
            self._write_position(value)
        result=self.budget.run('GPU_forward',lambda:self.native_forward(m,d))
        self.wp.synchronize()
        _,arrays,owners,_=describe(d,self.wp.array)
        raw={n:a.numpy().copy() for n,a in arrays.items()}
        # Freshness is established by the completed native_forward immediately above.
        contacts=observe(raw,state_generation=k,solved_generation=k)
        row=dict(stage=k,qpos=raw['Data.qpos'],qvel=raw['Data.qvel'],qacc=raw['Data.qacc'],contacts=contacts,
            full_Data_hashes={n:digest(a) for n,a in raw.items()},position_correction=record)
        require(all(np.isfinite(row[n]).all() for n in ('qpos','qvel','qacc')),'Nonfinite RK primary state')
        if hasattr(self,'contact_order'):
            proof=self.contact_order.last
            require(proof['serial']==self.contact_order.serial,'Fresh ordering proof required')
            self.ordering.append(proof)
        self.stages.append(row);return result
    def rk(self,m,d):
        self._check(m,d);require(self.phase=='micro' and self.rks==0 and len(self.stages)==1,'Unexpected RK4 order')
        self.rks+=1;return self.budget.run('RK4_integrator',lambda:self.native_rk(m,d))
    def microstep(self,accumulator,coordinates=(None,)):
        self._check(self.model,self.data);require(self.phase is None,'Nested microstep')
        require(isinstance(coordinates,tuple) and len(coordinates)==1 and all(x in (None,'upper','all') for x in coordinates),'Explicit per-world stage scope required')
        writes=int(any(accumulator.enabled))+3*int(any(c is not None for c in coordinates))
        self.budget.reserve(integrations=1,RK4_integrator=1,GPU_forward=4,position_write=writes)
        self.dt=float(self.model.opt.timestep.numpy()[0]);require(np.isfinite(self.dt) and self.dt>0,'Positive shared dt required')
        self.start=self.data.qpos.numpy().copy();before_vel=self.data.qvel.numpy().copy();before_time=self.data.time.numpy().copy()
        self.acc,self.coordinates=accumulator,coordinates
        desc,_,owners,_=describe(self.data,self.wp.array);self.binding=(desc,owners)
        accumulator_start=accumulator.snapshot()
        self.phase='micro';self.stages=[];self.rks=0;self.ordering=[]
        try:
            self.budget.run('integrations',lambda:self.native_step(self.model,self.data))
            require(len(self.stages)==4 and self.rks==1,'Four native solved stages and one RK4 call required')
            expected_time=before_time+np.float32(self.dt)
            require(expected_time.shape==(1,) and self.data.time.numpy().tobytes()==expected_time.tobytes(),'The native world clock must advance once by shared dt')
            require(self.stages[0]['qpos'].tobytes()==self.start.tobytes() and self.stages[0]['qvel'].tobytes()==before_vel.tobytes(),'Native initial stage changed')
            native=self.data.qpos.numpy().copy()
            stages=np.asarray([r['qvel'] for r in self.stages],dtype=np.float32)
            proposal,correction=accumulator.advance(self.start,stages,native,self.dt,np.ones(1,dtype=np.bool_))
            if accumulator.enabled.any():self._write_position(proposal)
            velocity=self.data.qvel.numpy().copy();a=np.asarray([r['qacc'] for r in self.stages],dtype=np.float64)
            residual=velocity.astype(float)-before_vel.astype(float)-self.dt*np.einsum('k,kwn->wn',np.array([1,2,2,1])/6,a)
            bounds=[]
            for w in range(1):
                bound=8*np.finfo(np.float32).eps*(1+max(float(np.max(abs(before_vel[w]))),float(np.max(abs(velocity[w]))),float(np.max(abs(self.dt*a[:,w])))))
                require(float(np.max(abs(residual[w])))<=bound,'RK weighted velocity residual exceeded original bound')
                bounds.append(dict(world=w,residual_max=float(np.max(abs(residual[w]))),bound=bound))
            compact=[{k:v for k,v in row.items() if k not in ('qpos','qvel','qacc','position_correction')} for row in self.stages]
            return dict(contact_ordering=list(self.ordering),post_integrator=dict(qpos=self.data.qpos.numpy(),qvel=velocity,time=self.data.time.numpy()),physical_stages=4,stage_sha256=hashlib.sha256(canonical(compact)).hexdigest(),stages=self.stages,
                correction=correction,velocity_bounds=bounds,contacts_fresh_at_endpoint=False,
                scope='Numeric eager microstep candidate; endpoint forward/sense and adaptive publication remain required')
        except Exception:
            self.poisoned=True;accumulator.restore(accumulator_start)
            raise
        finally:self.phase=None
