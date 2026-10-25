"""Bounded diagnostic RK4 entry point; no simulator import or production hook.

Only this object's microstep/refresh methods can enter the native GPU solver.
One macro is 64 fixed microsteps; 10 approach macros + one reject/retry pair.
This is a bounded rollback diagnostic, not material/contact acceptance.
"""
from contextlib import contextmanager
import struct
import time
import math

MICROSTEPS = 64
MACRO_DT = 0.002
MICRO_DT = struct.unpack('f', struct.pack('f', MACRO_DT / MICROSTEPS))[0]
MAX_APPROACH_MACROS = 10
MAX_MACRO_ATTEMPTS = MAX_APPROACH_MACROS + 2
MAX_STEPS = MAX_MACRO_ATTEMPTS * MICROSTEPS

def contact_evidence(geoms, distances, addresses, forces, worldids, nefc, lower, floor):
    """Identify actual pair-linked force; noncontact friction rows do not count.

    efc.force values are raw solver rows, not Cartesian or summed normal force.
    No contact/material threshold is changed here.
    """
    require(len(geoms)==len(distances)==len(addresses)==len(worldids),'Contact array lengths differ')
    require(0<=nefc<=len(forces),'Invalid active constraint length')
    require(all(math.isfinite(float(x)) for x in forces[:nefc]),'Nonfinite solver force')
    rows=[]
    for i,(pair,dist,addr,world) in enumerate(zip(geoms,distances,addresses,worldids)):
        require(int(world)==0 and len(pair)==2,'Unexpected contact world/geom shape')
        require(math.isfinite(float(dist)),'Nonfinite contact distance')
        valid=[int(a) for a in addr if int(a)>=0]
        require(all(a<nefc for a in valid),'Contact constraint address outside active rows')
        values=[float(forces[a]) for a in valid]
        target=set(map(int,pair))=={int(lower),int(floor)}
        active=target and float(dist)<=0 and any(x!=0 for x in values)
        rows.append(dict(contact_index=i,geoms=list(map(int,pair)),distance_m=float(dist),
            constraint_addresses=valid,raw_solver_forces=values,lower_floor_pair=target,active_lower_floor=active))
    return dict(lower_floor_active=any(r['active_lower_floor'] for r in rows),contact_force_rows=rows)

def find_contact_endpoint(advance):
    """One new natural approach, stopping at its first loaded floor endpoint."""
    for logical_index in range(MAX_APPROACH_MACROS):
        endpoint=advance(logical_index)
        if endpoint['lower_floor_active']:
            return logical_index+1,endpoint
    raise RuntimeError('No loaded lower-ball/floor endpoint within 10-macro approach budget')

def require(ok, message):
    if not ok: raise RuntimeError(message)

@contextmanager
def diagnostic_rk_options(model, api):
    """Change only this diagnostic device Model's integration options, then undo."""
    old_dt = float(model.opt.timestep.numpy()[0])
    old_integrator = model.opt.integrator
    require(old_integrator == 3, 'Expected original implicitfast option')
    try:
        model.opt.integrator = 1  # pinned MuJoCo RK4 enum
        model.opt.timestep.fill_(MICRO_DT)
        api.synchronize()
        require(float(model.opt.timestep.numpy()[0]) == MICRO_DT, 'Microstep assignment mismatch')
        yield
    finally:
        model.opt.timestep.fill_(old_dt)
        model.opt.integrator = old_integrator
        api.synchronize()
        require(float(model.opt.timestep.numpy()[0]) == old_dt and model.opt.integrator == old_integrator,
                'Diagnostic integration options not restored')

class BoundedRK4:
    def __init__(self, cpu, mw, fw, wp, counts, persist, deadline):
        self.cpu,self.mw,self.fw,self.wp = cpu,mw,fw,wp
        self.counts,self.persist,self.deadline = counts,persist,deadline
        self.originals=[];self.phase=None;self.stage=0;self.rk_calls=0
        self.observer=None;self.target=None

    def _time(self): require(time.monotonic() < self.deadline, 'Diagnostic wall budget exceeded')
    def _patch(self, obj, name, value):
        if hasattr(obj,name):
            self.originals.append((obj,name,getattr(obj,name)));setattr(obj,name,value)
    def _blocked(self,*args,**kwargs):
        self.counts['blocked_step_or_capture_requests']+=1;self.persist()
        raise RuntimeError('Only bounded diagnostic RK4 entry point is authorized')
    def _render_blocked(self,*args,**kwargs):
        self.counts['blocked_render_context_requests']+=1;self.persist()
        raise RuntimeError('No render/sensor context in dynamic macro diagnostic')

    def __enter__(self):
        self.native_step=self.fw.step;self.native_forward=self.fw.forward
        self.native_rk=self.fw.rungekutta4;native_cpu=self.cpu.mj_forward
        def cpu_forward(*args,**kwargs):
            self._time();require(self.counts['CPU_forward_attempted'] < 1,'CPU forward budget exhausted')
            self.counts['CPU_forward_attempted']+=1;self.persist()
            result=native_cpu(*args,**kwargs)
            self.counts['CPU_forward_completed']+=1;self.persist();return result
        def forward(model,data):
            self._time()
            if self.phase=='step':
                require(self.target==(id(model),id(data)) and self.stage < 4,'Unexpected RK4 stage/owner')
            elif self.phase=='refresh':
                require(self.target==(id(model),id(data)) and self.stage == 0,'Unexpected endpoint forward')
            else:
                require(self.phase is None and self.counts['setup_GPU_forward_completed']==0 and
                        self.counts['GPU_forward_attempted']==0,'Unbudgeted GPU forward')
            self.counts['GPU_forward_attempted']+=1;self.persist()
            result=self.native_forward(model,data);self.wp.synchronize()
            self.counts['GPU_forward_completed']+=1
            if self.phase=='step':
                self.counts['rk_stages_observed']+=1
                index=self.stage;self.stage+=1;self.persist()
                if self.observer is not None:self.observer(model,data,index)
            elif self.phase=='refresh':
                self.stage+=1;self.counts['endpoint_GPU_forward_completed']+=1
            else:self.counts['setup_GPU_forward_completed']+=1
            self.persist();return result
        def rk(model,data):
            require(self.phase=='step' and self.target==(id(model),id(data)) and self.rk_calls==0,
                    'RK4 call outside native bounded step')
            self.rk_calls+=1;self.counts['RK4_integrator_calls']+=1
            return self.native_rk(model,data)
        for name in ('mj_step','mj_step1','mj_step2','mj_Euler','mj_RungeKutta','mj_implicit'):
            self._patch(self.cpu,name,self._blocked)
        for module in (self.fw,self.mw):
            for name in ('step','step1','step2','euler','implicit'):
                self._patch(module,name,self._blocked)
            self._patch(module,'rungekutta4',rk)
            self._patch(module,'forward',forward)
        for name in ('ScopedCapture','capture_launch'):self._patch(self.wp,name,self._blocked)
        self._patch(self.mw,'create_render_context',self._render_blocked)
        self._patch(self.cpu,'mj_forward',cpu_forward)
        return self

    def microstep(self,model,data,observer):
        self._time()
        require(self.phase is None and self.counts['attempted'] < MAX_STEPS,'Microstep budget/reentrancy')
        require(model.opt.integrator==1 and float(model.opt.timestep.numpy()[0])==MICRO_DT,
                'Unexpected diagnostic integrator/timestep')
        self.phase='step';self.target=(id(model),id(data));self.observer=observer;self.stage=0;self.rk_calls=0
        self.counts['attempted']+=1;self.persist()
        try:
            self.native_step(model,data);self.wp.synchronize()
            self.counts['completed']+=1;self.persist()
            require(self.stage==4 and self.rk_calls==1,'Native RK4 stage count mismatch')
        finally:
            self.phase=None;self.target=None;self.observer=None

    def refresh(self,model,data):
        self._time()
        require(self.phase is None and self.counts['endpoint_GPU_forward_attempted'] < MAX_MACRO_ATTEMPTS,
                'Endpoint forward budget/reentrancy')
        self.counts['endpoint_GPU_forward_attempted']+=1
        self.phase='refresh';self.target=(id(model),id(data));self.stage=0
        try:self.fw.forward(model,data)
        finally:self.phase=None;self.target=None

    def __exit__(self,*exc):
        for obj,name,value in reversed(self.originals):setattr(obj,name,value)
        self.originals=[]
