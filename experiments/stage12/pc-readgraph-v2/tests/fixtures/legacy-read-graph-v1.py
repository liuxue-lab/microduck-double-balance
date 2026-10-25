"""Capture read-only audit gathers; run the original CPU Audit every substep.

No graph advances physics or writes live data. The inherited force lifecycle
retains its zero-before-write check and exact pre/post full-wrench comparisons.
"""
import importlib
from pathlib import Path

from deploy_contract import require,sha
from bam_graph import storage_signature
from packed_checks import PackedChecks,KEYS

READ_CHECK_CALLS=40
ENTITY_SHA='fd1116692f82278bda721d44c015d27e253d2e866608c8e12824df039dfb4657'


def verify_entity(guard):
    module=importlib.import_module('mjlab.entity.data');path=Path(module.__file__)
    require(sha(path)==ENTITY_SHA,'Unreviewed EntityData source')
    data=guard.robot.data
    require(type(data) is module.EntityData,'Unreviewed joint-position property type')
    require(storage_signature(data.data.qpos)==storage_signature(guard.raw.sim.data.qpos),
            'Entity joint positions do not use current simulator qpos')
    return dict(path=str(path),sha256=ENTITY_SHA,
                joint_position='Original EntityData.joint_pos, current qpos with original indexing')


class ReadGraph:
    """A pure FP32 gather with fixed input storage and owned output storage."""
    def __init__(self,torch,inputs,gather):
        self.torch=torch;self.inputs=inputs;self.gather=gather
        self.fields=inputs();self.signatures=self.signature(self.fields)
        tensors=[v for v in self.fields.values() if isinstance(v,torch.Tensor)]
        require(tensors and all(v.is_cuda and not v.requires_grad for v in tensors),
                'Read graph requires CUDA inference tensors')
        self.calls=self.checked_calls=self.scalar_comparisons=0
        self.output=None;self.output_signature=None
        device=tensors[0].device;current=torch.cuda.current_stream(device)
        stream=torch.cuda.Stream(device=device);stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):gather()
        current.wait_stream(stream);torch.cuda.synchronize(device)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.output=gather()
        current.wait_stream(stream)
        self.output_signature=storage_signature(self.output)
        require(self.output.ndim==1 and self.output.dtype==torch.float32,'Unexpected packed read output')
        require(all(self.output.untyped_storage().data_ptr()!=v.untyped_storage().data_ptr() for v in tensors),
                'Read output aliases a live input')

    def signature(self,fields):
        out={}
        for name,value in fields.items():
            if isinstance(value,self.torch.Tensor):out[name]=('tensor',storage_signature(value))
            elif isinstance(value,slice):out[name]=('slice',value.start,value.stop,value.step)
            elif isinstance(value,(list,tuple)):out[name]=('indices',tuple(value))
            else:raise RuntimeError('Unexpected captured input type: '+name)
        return out

    def read(self):
        t=self.torch
        require(self.signature(self.inputs())==self.signatures,'Captured read backing storage changed')
        require(storage_signature(self.output)==self.output_signature,'Captured read output storage changed')
        self.graph.replay()
        if self.checked_calls<READ_CHECK_CALLS:
            reference=self.gather()
            require(reference.shape==self.output.shape and reference.dtype==self.output.dtype,
                    'Read graph comparison schema changed')
            # Compare bit patterns, so an identical NaN reaches the original
            # CPU Audit and retains its original failure event and stop phase.
            require(bool(t.equal(reference.view(t.int32),self.output.view(t.int32))),
                    'Read graph differs from original same-input gather')
            self.checked_calls+=1;self.scalar_comparisons+=self.output.numel()
        # This creates a separate host list on every call. Later replays cannot
        # overwrite an earlier Audit sample or first-violation context.
        values=self.output.detach().cpu().tolist();self.calls+=1
        return values

    def stats(self):
        return dict(replays=self.calls,same_input_checked_calls=self.checked_calls,
            same_input_scalar_comparisons=self.scalar_comparisons,comparison='exact FP32 bit pattern',
            output_scalars=self.output.numel(),warmup_physics_steps=0,live_writes=False,
            live_storage_fields=len(self.fields),host_snapshot_per_read=True)


class GraphPackedChecks(PackedChecks):
    def __init__(self,session,torch):
        super().__init__(session,torch)
        self.source=verify_entity(self.guard)
        self.pre_graph=self.post_graph=None;self.closed=False

    def base_inputs(self):
        d=self.session.raw.sim.data
        return {k:getattr(d,k) for k in ('qpos','qvel','ctrl','actuator_force','xfrc_applied')}

    def pre_inputs(self):
        fields=self.base_inputs();g=self.guard
        fields['entity_qpos']=g.robot.data.data.qpos
        fields['entity_qadr']=g.robot.data.indexing.joint_q_adr
        for i,group in enumerate(g.groups):
            d=group.last_diagnostics
            for name in ('effective_vin','joint_velocity','duty_applied','motor_torque','position_error'):
                fields[str(i)+'.'+name]=d[name]
            for name in ('kd_scale','global_ctrl_ids','target_ids'):
                fields[str(i)+'.'+name]=getattr(group,name)
        return fields

    def post_inputs(self):
        return dict(self.base_inputs(),qids=self.qids,vids=self.vids,cids=self.cids)

    def cat(self,parts):
        t=self.torch
        require(all(p.dtype==t.float32 and p.device==self.device and not p.requires_grad for p in parts),
                'Packed audit input schema changed')
        return t.cat([p.reshape(-1) for p in parts])

    def pre_values(self):
        s,g,t=self.session,self.guard,self.torch;parts=[s.raw.sim.data.xfrc_applied]
        for group in g.groups:
            d=group.last_diagnostics;velocity=d['joint_velocity']
            fields=(d['effective_vin'].expand_as(velocity),velocity,group.kd_scale.expand_as(velocity),
                d['duty_applied'],d['motor_torque'],s.raw.sim.data.ctrl[:,group.global_ctrl_ids],
                g.robot.data.joint_pos[:,group.target_ids],d['position_error'])
            parts.append(t.stack(fields,dim=-1)[0])
        return self.cat(parts)

    def post_values(self):
        data=self.session.raw.sim.data
        return self.cat((data.xfrc_applied,data.qpos[0,self.qids],data.qvel[0,self.vids],
                         data.actuator_force[0,self.cids]))

    def before(self,expected):
        require(not self.closed,'Read adapter is closed');s,g=self.session,self.guard
        diagnostics=[group.last_diagnostics for group in g.groups]
        if any(d is None or d is g.last_seen[i] for i,d in enumerate(diagnostics)):
            return super().before(expected)  # Preserve original force/freshness error ordering.
        if self.pre_graph is None:self.pre_graph=ReadGraph(self.torch,self.pre_inputs,self.pre_values)
        values=self.pre_graph.read();self.pre_reads+=1
        require(values[:self.width]==expected,'Applied wrench differs from planned force')
        s.checked_physics_steps+=1;rows=values[self.width:]
        require(len(rows)==self.nf*len(KEYS),'Motor transfer mapping changed')
        points=[dict(zip(KEYS,rows[i:i+len(KEYS)])) for i in range(0,len(rows),len(KEYS))]
        g.last_seen[:]=diagnostics
        g.capture(lambda:g.audit.before_integration(points))

    def after(self,expected):
        require(not self.closed,'Read adapter is closed');g=self.guard
        if self.post_graph is None:self.post_graph=ReadGraph(self.torch,self.post_inputs,self.post_values)
        values=self.post_graph.read();self.post_reads+=1
        require(values[:self.width]==expected,'Integrator modified external wrench unexpectedly')
        values=values[self.width:]
        require(len(values)==self.nq+self.nv+self.nf,'State transfer mapping changed')
        g.capture(lambda:g.audit.after_integration(values[:self.nq],values[self.nq:self.nq+self.nv],
                                                  values[self.nq+self.nv:]))

    def stats(self):
        return dict(super().stats(),pre_read_graph=None if self.pre_graph is None else self.pre_graph.stats(),
            post_read_graph=None if self.post_graph is None else self.post_graph.stats(),
            entity_source=self.source,closed=self.closed,original_physics_and_force_lifecycle=True)

    def close(self):self.closed=True
