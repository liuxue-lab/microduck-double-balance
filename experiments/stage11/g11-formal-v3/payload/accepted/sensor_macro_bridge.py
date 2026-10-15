"""Separate sense-aware macro adapter; original bridge is preserved unchanged.

Pinned ray sense may rebind11 results and write5 fixed native buffers. Their
numeric state, schema and aliases enter exact replay, and outside sense these
fields remain frozen. All other selected owners stay protected. Whole rollback
still restores every original binding and byte, including sensor results.
Default native/manager budgets remain zero; no native environment is imported.
"""
import dataclasses
import numpy as np
from native_state_bridge import (NativeStateBridge, CallBudget, APPROVED_LIMITS,
    StrictOwnerSnapshot, Checkpoint, model_tree, canonical, digest, require)
from sensor_state_snapshot import DataSnapshot
from trial_invariants import data_view_aliases
from scene_evidence import owner_state, array_meta
from sensor_lifecycle_contract import SensorLifecycleContract

MACRO_LIMITS={**APPROVED_LIMITS, **{k:0 for k in (
    'model_option_write','private_clock_write','endpoint_sensor_read',
    'Env_substep_counter','Action_apply','Scene_write','Scene_update','Metrics_substep')}}

class MacroBudget(CallBudget):
    def __init__(self,limits=None,**kwargs):
        limits=dict(MACRO_LIMITS if limits is None else limits)
        require(set(limits)==set(MACRO_LIMITS) and all(type(v) is int and v>=0 for v in limits.values()),'Complete explicit macro budget required')
        super().__init__({k:limits[k] for k in APPROVED_LIMITS},**kwargs)
        self.limits=limits

def sensor_values(scene,torch):
    """Read every initialized sensor twice; require native cache reuse and finiteness."""
    result={}
    def walk(value,path):
        if isinstance(value,torch.Tensor):
            a=value.detach().cpu().numpy().copy()
            require(a.ndim>=1 and a.shape[0]==1 and np.isfinite(a).all(),'Invalid single-world sensor output: '+path)
            result[path]=dict(shape=list(a.shape),dtype=str(a.dtype),sha256=digest(a))
        else:
            require(dataclasses.is_dataclass(value),'Unrecognized sensor result: '+path)
            for field in dataclasses.fields(value):
                v=getattr(value,field.name)
                if v is not None:walk(v,path+'.'+field.name)
    require(scene.sensors,'Explicit sensor roster required')
    for name,sensor in scene.sensors.items():
        a,b=sensor.data,sensor.data
        require(a is b and sensor._cache_valid,'Sensor cache not reused: '+name)
        walk(a,name)
    require(result,'Empty sensor output roster')
    return result

class SensorAwareMacroStateBridge(NativeStateBridge):
    def checkpoint(self):
        require(self.phase=='idle','Nested/poisoned macro checkpoint')
        self.budget.reserve(state_snapshot=1)
        def capture():
            self.sync();handles=self.preflight()
            require(self.sim.wp_model.opt.integrator==3 and self.sim.wp_model.opt.timestep.numpy().tobytes()==np.array([.002],np.float32).tobytes(),'Original Model options required at public boundary')
            sensors={id(v) for v in self.scene.sensors.values()}
            self.sensor_policy=SensorLifecycleContract(self.scene,self.roots,self.torch,self.wp)
            self.held_roots=self.sensor_policy.held_roots(self.roots)
            slots={(id(obj),n) for _,obj,names in self.roots for n in names}
            require(all((id(s),n) in slots for s in self.scene.sensors.values() for n in ('_cached_data','_cache_valid')),'All sensor cache roots must be selected for rollback')
            data=DataSnapshot(self.sim.wp_data,self.wp)
            owners=StrictOwnerSnapshot(self.roots,self.torch,self.wp)
            self.held=StrictOwnerSnapshot(self.held_roots,self.torch,self.wp)
            self.held_views,_=data_view_aliases(self.held,data)
            allow,proof=data_view_aliases(owners,data)
            arrays,meta=model_tree(self.sim.wp_model,self.wp)
            self.saved=Checkpoint(data,owners,arrays,meta,{n:digest(v.numpy()) for n,v in arrays.items()},self.sim.wp_model.opt.timestep.numpy().copy(),self.sim.wp_model.opt.integrator,handles,self.global_rng(),allow,proof)
            self.input_hashes={n:data.hashes[n] for n in ('Data.ctrl','Data.qfrc_applied','Data.xfrc_applied','Data.mocap_pos','Data.mocap_quat') if n in data.arrays}
            require('Data.ctrl' in self.input_hashes,'Held native controls missing')
            self.sensor_roster=tuple((n,id(s)) for n,s in self.scene.sensors.items())
            self.cache_phase=False;self.sensor_fingerprint=None;self.semantic=None;self.semantic_schema=None
            self.phase='trial';self.invalidate();return self.saved
        try:return self.budget.run('state_snapshot',capture)
        except Exception:self.phase='poisoned';raise

    def check_protected(self):
        require(self.saved is not None and self.phase=='trial','Active macro checkpoint required')
        s=self.saved
        require(self.preflight()==s.handles and tuple((n,id(v)) for n,v in self.scene.sensors.items())==self.sensor_roster,'Native/sensor binding changed')
        s.data.preflight();s.owners.preflight()
        self.held.assert_restored(mutable_data_views=self.held_views)
        if not self.cache_phase:s.owners.assert_restored(mutable_data_views=s.mutable_views)
        self.sensor_policy.check_bindings()
        if self.cache_phase and self.sensor_fingerprint is not None:
            require(self.sensor_policy.current()==self.sensor_fingerprint,'Ray state changed outside measured sense boundary')
        require(self.global_rng()==s.rng,'Default RNG changed inside macro; no rewind')
        require(all(digest(s.data.arrays[n].numpy())==h for n,h in self.input_hashes.items()),'Native controls/external forces changed inside trial')
        self._model_preflight()

    def restore(self):
        evidence=super().restore()
        self.cache_phase=False;self.sensor_fingerprint=None;self.semantic=None
        self.check_protected();return evidence

    def options(self,resolution=None):
        self.check_protected()
        if resolution is not None:require(resolution in (64,128,256),'Only declared late-window tiers supported')
        def assign():
            self.sim.wp_model.opt.timestep.assign(self.saved.dt if resolution is None else np.array([.002/resolution],np.float32))
            self.sim.wp_model.opt.integrator=self.saved.integrator if resolution is None else 1
        self.budget.run('model_option_write',assign);self.check_protected()

    def private_origin(self):
        self.check_protected()
        self.budget.run('private_clock_write',lambda:self.sim.wp_data.time.assign(np.zeros(1,np.float32)))
        require(self.sim.wp_data.time.numpy().tobytes()==np.zeros(1,np.float32).tobytes(),'Private clock write differs')
        self.invalidate()

    def sense(self):
        self.check_protected();self.cache_phase=True;self.sensor_fingerprint=None
        super().sense()
        try:
            self.semantic=self.budget.run('endpoint_sensor_read',lambda:sensor_values(self.scene,self.torch))
            self.sensor_fingerprint=self.sensor_policy.current()
            schema={n:(r['shape'],r['dtype']) for n,r in self.semantic.items()}
            if self.semantic_schema is None:self.semantic_schema=schema
            require(schema==self.semantic_schema,'Sensor semantic schema changed between trials')
            self.check_protected()
        except Exception:self.phase='poisoned';self.invalidate();raise

    def endpoint_bytes(self):
        self.contacts_ready();require(self.sensed==self.generation and self.semantic is not None,'Fresh measured sensor outputs required')
        self.check_protected();self.sync()
        require(all(s._cache_valid for s in self.scene.sensors.values()),'Endpoint sensor cache became stale')
        require(sensor_values(self.scene,self.torch)==self.semantic,'Endpoint sensor values changed after measured sense')
        raw,scalars,topology=owner_state(self.held_roots,self.torch,self.wp)
        arrays,meta=model_tree(self.sim.wp_model,self.wp)
        return canonical(dict(data=self.saved.data.fingerprint(),held_owners=array_meta(raw),held_scalars=scalars,held_topology=topology,
            sensors=self.semantic,ray_sense_state=self.sensor_policy.semantic(),model={n:digest(v.numpy()) for n,v in arrays.items()},model_meta=meta,handles=self.preflight(),rng=self.global_rng()))

    def close(self):
        require(self.phase=='trial','Cannot close poisoned macro')
        self.saved=None;self.held=None;self.phase='idle'
