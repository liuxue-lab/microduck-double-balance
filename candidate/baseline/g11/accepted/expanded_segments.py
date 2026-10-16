"""Captured native GPU blocks with unchanged host adaptive/ordering decisions.

Pure pinned functions capture as blocks; remaining launch/copy/zero/fill APIs
inside forward/RK regions capture as primitives. CPU transfers and observations
stay on the host. No unmeasured claim of monolithic whole-RK4 graph execution.
"""
import dataclasses,enum,functools,hashlib,json
import numpy as np
from contextlib import contextmanager
from segment_runtime import Segments,LABELS
from lifecycle_support import Gate,require
from native_state_bridge import model_tree
from head_contact_data_commit_r3 import describe
from report_io import save

BLOCKS={
    'forward':('fwd_velocity','fwd_actuation','fwd_acceleration','_rk_perturb_state','_rk_accumulate','_advance'),
    'smooth':('kinematics','com_pos','camlight','crb','transmission'),
    'constraint':('make_constraint',),
    'collision':('collision',),
    'sensor':('sensor_pos','sensor_vel','sensor_acc'),
    'solver':('_update_constraint','_update_gradient','_update_gradient_incremental','_linesearch'),
}
PERSISTENT=frozenset(f'{module}.{name}' for module,names in BLOCKS.items() for name in names
                    if module!='solver' and not name.startswith('_'))

class ExpandedGate(Gate):
    def __init__(self,*a,**k):super().__init__(*a,**k);self.segments=None;self.caps={}
    def limited(self,label,limit,native,check=None):
        limit=min(limit,self.caps.get(label,limit))
        def execute(*a,**k):
            s=self.segments
            if s is not None and s.enabled:
                if label=='GPU_forward':
                    with s.region('forward'):return native(*a,**k)
                if label in LABELS:
                    with s.scope(label):return native(*a,**k)
            return native(*a,**k)
        counted=super().limited(label,limit,execute,check)
        def call(*a,**k):
            try:return counted(*a,**k)
            finally:
                if label=='GPU_forward' and self.segments is not None:self.segments.retire()
        return call

class ExpandedSegments(Segments):
    def __init__(self,wp,out,gate,apis,contact_order,model,data):
        self.model,self.data=model,data;self.regions=[];self.capturing_ours=False
        self.keep=set();self.closing=False;self.binding_files={};self.census={};self.current_ops=None
        self.block_operations={};self.host_transfers={};self.operation_specs={}
        super().__init__(wp,out,gate,apis,contact_order)
        self.stream=self.device.stream.cuda_stream
        from warp._src import context
        require(context.runtime.driver_version>=(12,3),'Capture-time compilation requires the pinned modern CUDA driver')
        self.raw_tiled=wp.launch_tiled
        gate.patch(wp,'launch_tiled',self.launch_tiled)
        for name in ('zero_','fill_'):
            raw=getattr(wp.array,name)
            def dispatch(array,*args,_name=name,_raw=raw,**kwargs):
                if self.capturing_ours:self.note(_name);return _raw(array,*args,**kwargs)
                if self.enabled and self.regions and array.device==self.device:
                    return self.primitive(_name,_raw,(array,*args),kwargs)
                return _raw(array,*args,**kwargs)
            gate.patch(wp.array,name,dispatch)
        save(out/'native-binding-preflight.json',dict(model=self.token(model,[]),data=self.token(data,[]),
             cuda_stream=str(self.stream),native_integrations=0,handles_serialized_as_restorable=False))

    def install(self,fw,solver):
        from mujoco_warp._src import warp_util
        require(warp_util._STACK is None,'Native event profiler must be inactive for the declared resource inventory')
        modules=dict(forward=fw,smooth=fw.smooth,constraint=fw.constraint,
                     collision=fw.collision_driver,sensor=fw.sensor,solver=solver)
        for module,names in BLOCKS.items():
            for name in names:
                label=module+'.'+name;native=getattr(modules[module],name)
                @functools.wraps(native)
                def wrapped(*a,_label=label,_native=native,**k):
                    if not self.enabled or not self.regions:return _native(*a,**k)
                    require(not self.capturing_ours,'Unexpected nested native block: '+_label)
                    with self.scope(_label):return self.execute('block',_native,a,k)
                self.gate.patch(modules[module],name,wrapped)

    @contextmanager
    def scope(self,label):
        require(self.label is None,'Nested numerical label')
        self.label=label
        try:yield
        finally:self.label=None

    @contextmanager
    def region(self,label):
        require(label in ('forward','rk4') and not self.capturing_ours,'Invalid native region')
        self.regions.append(label)
        try:yield
        finally:
            require(self.regions.pop()==label,'Unbalanced native region')

    def rk(self,native):
        @functools.wraps(native)
        def wrapped(*a,**k):
            if not self.enabled:return native(*a,**k)
            try:
                with self.region('rk4'):return native(*a,**k)
            finally:self.retire()
        return wrapped

    def binding(self,label,rows):
        raw=json.dumps(rows,sort_keys=True,separators=(',',':')).encode()
        digest=hashlib.sha256(raw).hexdigest()
        if digest not in self.binding_files:
            require(len(self.binding_files)<64,'Unexpected Model/Data binding churn')
            path=self.out/'native-bindings';path.mkdir(exist_ok=True)
            save(path/(digest+'.json'),dict(label=label,bindings=rows))
            self.binding_files[digest]=label
        return label,digest

    def token(self,value,refs):
        if value is self.model:
            # Tuples such as qM_tiles/body_tree contain further arrays. An
            # identity-only opaque tuple token would miss those rebindings.
            return self.binding('Model',self.model_bindings(value,refs))
        if value is self.data:
            desc,arrays,owners,_=describe(value,self.wp.array);refs.extend(arrays.values())
            return self.binding('Data',dict(identity=id(value),descriptors=desc,owners=owners))
        if dataclasses.is_dataclass(value):
            fields=dataclasses.fields(value);require(set(vars(value))=={f.name for f in fields},'Dynamic native dataclass')
            return ('dataclass',id(value),type(value).__module__+'.'+type(value).__qualname__,
                    tuple((f.name,self.token(getattr(value,f.name),refs)) for f in fields))
        return super().token(value,refs)

    def model_bindings(self,value,refs):
        if isinstance(value,self.wp.array):return super().token(value,refs)
        if value is None or type(value) in (str,int,float,bool):return (type(value).__name__,value)
        if isinstance(value,enum.Enum):return (type(value).__module__+'.'+type(value).__qualname__,int(value.value))
        if isinstance(value,np.generic):return (str(value.dtype),value.item())
        if isinstance(value,np.ndarray):return ('host-array',id(value),str(value.dtype),value.shape,hashlib.sha256(value.tobytes()).hexdigest())
        if isinstance(value,(tuple,list)):return (type(value).__name__,id(value),[self.model_bindings(x,refs) for x in value])
        if isinstance(value,dict):return ('dict',id(value),[(k,self.model_bindings(v,refs)) for k,v in sorted(value.items())])
        require(dataclasses.is_dataclass(value),'Unsupported native binding: '+type(value).__qualname__)
        fields=dataclasses.fields(value);require(set(vars(value))=={f.name for f in fields},'Dynamic Model binding fields')
        return (type(value).__module__+'.'+type(value).__qualname__,id(value),[(f.name,self.model_bindings(getattr(value,f.name),refs)) for f in fields])

    def note(self,name):
        if self.current_ops is not None:self.current_ops[name]=self.current_ops.get(name,0)+1

    def primitive(self,kind,fn,args,kwargs):
        if self.label is not None:return self.execute(kind,fn,args,kwargs)
        with self.scope('native.'+kind):return self.execute(kind,fn,args,kwargs)

    def launch(self,*a,**k):
        if self.capturing_ours:self.note('launch');return self.raw_launch(*a,**k)
        if not self.enabled or not (self.regions or self.label):return self.raw_launch(*a,**k)
        require(not k.get('record_cmd') and not k.get('adjoint'),'Deferred/adjoint native launch refused')
        return self.primitive('launch',self.raw_launch,a,k)

    def launch_tiled(self,*a,**k):
        if self.capturing_ours:self.note('launch_tiled');return self.raw_tiled(*a,**k)
        if not self.enabled or not (self.regions or self.label):return self.raw_tiled(*a,**k)
        require(not k.get('record_cmd') and not k.get('adjoint'),'Deferred/adjoint tiled launch refused')
        return self.primitive('launch_tiled',self.raw_tiled,a,k)

    def copy(self,*a,**k):
        if self.capturing_ours:self.note('copy');return self.raw_copy(*a,**k)
        if not self.enabled or not (self.regions or self.label):return self.raw_copy(*a,**k)
        dest=a[0] if a else k.get('dest');src=a[1] if len(a)>1 else k.get('src')
        if isinstance(dest,self.wp.array) and isinstance(src,self.wp.array) and dest.device==src.device==self.device:
            return self.primitive('copy',self.raw_copy,a,k)
        name=str(getattr(src,'device',None))+'->'+str(getattr(dest,'device',None))
        self.host_transfers[name]=self.host_transfers.get(name,0)+1
        return self.raw_copy(*a,**k)

    def execute(self,kind,fn,args,kwargs):
        require(not self.capturing_ours,'Nested capture dispatch')
        require(self.device.stream.cuda_stream==self.stream,'CUDA stream changed before graph dispatch')
        # The inherited stale-binding challenge clones outside all regions, so
        # its copy cannot accidentally become a second captured dispatch.
        refs=[];key=(self.label,kind,self.token(args,refs),self.token(kwargs,refs))
        new=key not in self.entries;self.current_ops={} if new else None
        def captured(*a,**k):
            self.capturing_ours=True
            try:
                if kind!='block':self.note(kind)
                return fn(*a,**k)
            finally:self.capturing_ours=False
        saved_regions=list(self.regions)
        # Numeric callbacks are captured inside fn; region classification is
        # unnecessary during inherited setup/challenge/native dispatch.
        self.regions=[]
        try:result=super().execute(kind,captured,args,kwargs)
        finally:self.regions=saved_regions
        if new:
            self.block_operations[str(self.captures)]=dict(self.current_ops or {})
            self.operation_specs[str(self.captures)]=dict(label=self.label,kind=kind)
        if self.label in PERSISTENT:self.keep.add(key)
        entry=self.entries[key];ops=self.block_operations[str(entry['serial'])]
        for name,count in ops.items():self.census[name]=self.census.get(name,0)+count
        self.current_ops=None
        return result

    def challenge(self,entry,args,kwargs,refs):
        require(any(value is self.data for value in args),'First native graph must bind the real Data owner')
        original=self.data.qpos;label=self.label;self.label=None
        try:clone=self.wp.clone(original)
        finally:self.label=label
        changed=[];before=self.launches
        try:
            self.data.qpos=clone
            fake=(entry['token'][0],entry['token'][1],self.token(args,changed),self.token(kwargs,changed))
            require(fake!=entry['token'],'Actual qpos rebinding did not change the graph token')
            try:self.check(entry,fake)
            except RuntimeError:pass
            else:raise RuntimeError('Rebound Data graph was dispatched')
        finally:self.data.qpos=original
        require(before==self.launches,'Stale challenge dispatched CUDA graph')
        self.log('stale_rejected',serial=self.captures,new_array=list(super().token(clone,[])),before_dispatch=True,
                 actual_Data_qpos_temporarily_rebound=True,original_binding_restored=self.data.qpos is original)
        self.checked_stale=True

    def retire(self):
        held={} if self.closing else {k:v for k,v in self.entries.items() if k in self.keep}
        self.entries={k:v for k,v in self.entries.items() if k not in held}
        try:super().retire()
        finally:self.entries.update(held)

    def close(self):
        self.closing=True;result=super().close()
        census=dict(captured_operations=self.block_operations,operation_specs=self.operation_specs,
                    logical_device_operations=self.census,host_transfers=self.host_transfers,
                    persistent_labels=sorted(PERSISTENT),declared_blocks=BLOCKS,
                    numeric_fields_exempted=[],CPU_compensation_and_adaptive_decisions='unchanged outside capture',
                    full_native_RK4_graph=False,scope='native pure blocks and remaining four device API families within forward/RK regions')
        save(self.out/'native-graph-coverage.json',census)
        return result
