"""Use live basic slices for proven contiguous BAM command mappings.

Only get_command is replaced. Original Entity call order, delay buffer, BAM
graph staging, model writes and all per-substep audits remain unchanged.
The returned views are consumed immediately by that reviewed call chain.
"""
import importlib
from pathlib import Path

from deploy_contract import require,sha
from phase_cache import Bindings
from bam_graph import storage_signature

CHECK_CALLS=40
FIELDS=('position_target','velocity_target','effort_target','pos','vel')
SOURCE_PINS={
    'mjlab.actuator.actuator':'ed57149b527c529b37b6322f20cac3eeed89b7355847ad712c5d44b13549f1ab',
    'mjlab.entity.data':'fd1116692f82278bda721d44c015d27e253d2e866608c8e12824df039dfb4657',
    'mjlab.entity.entity':'817896ec3ce69c8ef18cb3d8e2519b85d1db25f70126af5e643c34a0b054e572',
    'mjlab.sim.sim_data':'bc73da4811fc44336f9016d826ef1b2044b8cb0830f2ad01a06b9b84ffa8b4c6',
    'mjlab.utils.buffers.delay_buffer':'845e68da9e681adee1e93f90b4b1937c282a27f82a147f1fe3c19d83204049b5',
    'mjlab.utils.buffers.circular_buffer':'6a96cb79afbfb2d1234c3ee8e7531db371262673ddc509da7f3e66410ef2ae13',
}


def verify_sources():
    modules={};sources={}
    for name,pin in SOURCE_PINS.items():
        module=importlib.import_module(name);path=Path(module.__file__)
        require(sha(path)==pin,'Unreviewed command-view source: '+name)
        modules[name]=module;sources[name]=dict(path=str(path),sha256=pin)
    return modules,sources


def basic_slice(indices):
    if indices and indices==list(range(indices[0],indices[0]+len(indices))):
        return slice(indices[0],indices[-1]+1)
    return None


class CommandView:
    def __init__(self,group,data,torch,modules):
        self.group,self.data,self.torch=group,data,torch
        self.original=group.get_command;self.compute=group.compute
        act=modules['mjlab.actuator.actuator'];self.command_type=act.ActuatorCmd
        require(type(group).__name__=='FrictionDRBamActuator','Expected original non-backlash BAM')
        require(getattr(self.original,'__func__',None) is act.Actuator.get_command,
                'Unreviewed actuator command extraction')
        require(getattr(group.apply_delay,'__func__',None) is act.Actuator.apply_delay,
                'Unreviewed actuator delay method')
        require(type(data) is modules['mjlab.entity.data'].EntityData,'Unreviewed entity data')
        self.delay=group.apply_delay.__func__;self.transmission=group.transmission_type
        require(self.transmission==act.TransmissionType.JOINT,'Expected joint transmission')
        self.wrapper_type=modules['mjlab.sim.sim_data'].TorchArray
        self.calls=self.fast_calls=self.fallback_calls=self.checked_calls=self.scalar_checks=0
        self.final_mapping_verified=False
        self.indices=self.index_fields()
        self.index_signatures={k:storage_signature(v) for k,v in self.indices.items()}
        self.index_values=self.mapping_values()
        ids,qadr,vadr=(self.index_values[k] for k in ('target','qadr','vadr'))
        require(ids and len(ids)==len(set(ids)) and min(ids)>=0 and max(ids)<min(len(qadr),len(vadr)),
                'Invalid original command mapping')
        q=[qadr[i] for i in ids];v=[vadr[i] for i in ids]
        self.slices=dict(target=basic_slice(ids),qpos=basic_slice(q),qvel=basic_slice(v))
        self.eligible=all(x is not None for x in self.slices.values())
        self.width=len(ids)
        self.schema={k:(tuple(x.shape),str(x.dtype),str(x.device)) for k,x in self.sources().items()}
        require(all(len(shape)==2 and shape[0]==1 for shape,_,_ in self.schema.values()),
                'Expected one environment of joint commands')

    def index_fields(self):
        return dict(target=self.group.target_ids,qadr=self.data.indexing.joint_q_adr,
                    vadr=self.data.indexing.joint_v_adr)

    def mapping_values(self):
        fields=self.index_fields()
        require(all(isinstance(x,self.torch.Tensor) and x.ndim==1 and x.dtype==self.torch.long
                    for x in fields.values()),'Unexpected joint address type')
        return {k:x.detach().cpu().tolist() for k,x in fields.items()}

    def sources(self):
        d=self.data
        fields=dict(position_target=d.joint_pos_target,velocity_target=d.joint_vel_target,
                    effort_target=d.joint_effort_target,pos=d.data.qpos,vel=d.data.qvel)
        require(all(isinstance(x,self.torch.Tensor) or type(x) is self.wrapper_type for x in fields.values()),
                'Unexpected live command array type')
        return fields

    def __call__(self,data):
        require(data is self.data and self.group.compute is self.compute
                and getattr(self.group.apply_delay,'__func__',None) is self.delay
                and self.group.transmission_type==self.transmission,'Command consumer changed')
        require({k:storage_signature(x) for k,x in self.index_fields().items()}==self.index_signatures,
                'Joint address storage changed')
        # Sources are read afresh: no command snapshot survives a physics step.
        if not self.eligible:
            result=self.original(data);self.fallback_calls+=1
        else:
            sources=self.sources()
            require({k:(tuple(x.shape),str(x.dtype),str(x.device)) for k,x in sources.items()}==self.schema,
                    'Command source schema changed')
            values={k:x[:,self.slices['qpos' if k=='pos' else 'qvel' if k=='vel' else 'target']]
                    for k,x in sources.items()}
            require(all(x.shape==(1,self.width) and x.dtype==self.torch.float32 and not x.requires_grad
                        for x in values.values()),'Unexpected command view schema')
            result=self.command_type(**values)
            if self.checked_calls<CHECK_CALLS:
                expected=self.original(data)
                for name in FIELDS:
                    a,b=getattr(expected,name),getattr(result,name)
                    require(a.shape==b.shape and a.dtype==b.dtype and
                            bool(self.torch.equal(a.view(self.torch.int32),b.view(self.torch.int32))),
                            'Command view differs from original: '+name)
                    self.scalar_checks+=a.numel()
                self.checked_calls+=1
            self.fast_calls+=1
        self.calls+=1
        return result

    def verify_final_mapping(self):
        require(self.mapping_values()==self.index_values,'Joint address values changed during candidate')
        self.final_mapping_verified=True

    def stats(self):
        return dict(calls=self.calls,fast_calls=self.fast_calls,fallback_calls=self.fallback_calls,
            same_input_checked_calls=self.checked_calls,same_input_scalar_comparisons=self.scalar_checks,
            mapping=self.index_values,contiguous_mapping=self.eligible,
            final_mapping_verified=self.final_mapping_verified,comparison='exact FP32 bit pattern',
            command_fields=list(FIELDS),changes_to_delay=False,changes_to_BAM=False,
            live_sources_read_on_every_call=True,additional_physics_steps=0)


class CommandViews(Bindings):
    def __init__(self,guard,torch,graphs):
        super().__init__();self.runners=[];self.closed=False
        modules,self.sources=verify_sources()
        require([id(x[0]) for x in graphs.saved if x[1]=='compute']==[id(g) for g in guard.groups],
                'Command views require the accepted BAM graph staging consumer')
        try:
            for group in guard.groups:
                runner=CommandView(group,guard.robot.data,torch,modules)
                self.runners.append(runner);self.set(group,'get_command',runner)
        except BaseException:
            super().close();self.closed=True;raise

    def close(self):
        if self.closed:return
        try:
            for runner in self.runners:runner.verify_final_mapping()
        finally:
            super().close();self.closed=True

    def stats(self):
        return dict(groups=[x.stats() for x in self.runners],sources=self.sources,closed=self.closed,
            original_delay_and_entity_call_order=True,original_BAM_staging_copies_retained=True,
            index_guard='storage checked every call; values checked at setup and close; no resets allowed',
            view_lifetime='immediate original delay + accepted BAM staging; not an owned historical snapshot')
