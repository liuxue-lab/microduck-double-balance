"""Replay the pinned BAM compute kernels; keep live writes and audits explicit.

Capture runs on a shallow, isolated actuator copy. Its only simulator write,
_write_frictions, is redirected to output tensors. No warmup/capture advances
physics or writes to the live model, command delays, RNG or previous torque.
The original arithmetic is called unchanged, with no compiler or precision change.
"""
import copy
import dataclasses
import math

from deploy_contract import require, ATOL, RTOL
from phase_cache import Bindings

COMMAND_FIELDS=('pos','vel','position_target')
PARAMETERS=('kt','R','friction_base','friction_viscous','friction_stribeck',
    'load_friction_external','load_friction_motor','load_friction_external_stribeck',
    'load_friction_motor_stribeck','load_friction_external_quad','load_friction_motor_quad',
    'load_friction_base','load_friction_stribeck','dtheta_stribeck','alpha')
FLAGS=('stribeck','load_dependent','directional','quadratic')
SHADOW_CHECK_CALLS=40


def configuration(group):
    bam=group._bam_model;act=bam.actuator
    values=tuple((n,float(getattr(bam,n).value)) for n in PARAMETERS if hasattr(bam,n))
    require(all(math.isfinite(v) for _,v in values),'Nonfinite BAM parameter')
    return (type(group),type(act),values,tuple((n,getattr(bam,n,None)) for n in FLAGS),
        group._base_kp,group._dt,group.cfg.vin_min,act.error_gain,act.max_pwm,act.max_current,
        group.diagnostics_enabled)


class Shadow:
    """Use original methods and read-only live inputs, with isolated Python state."""
    def __init__(self,group):
        self.live=group;self.group=copy.copy(group)
        self.group._bam_model=copy.copy(group._bam_model)
        self.group._bam_model.actuator=copy.copy(group._bam_model.actuator)
        self.group._bam_model.actuator.model=self.group._bam_model
        self.frictions=None
        self.group._write_frictions=self.collect
        self.original=type(group).compute
    def collect(self,frictionloss,damping):self.frictions=(frictionloss,damping)
    def run(self,cmd,previous):
        self.group._prev_motor_torque=previous
        self.frictions=None;self.group.last_diagnostics=None
        torque=self.original(self.group,cmd)
        require(self.frictions is not None and self.group.last_diagnostics is not None,
                'Original BAM output/diagnostics missing')
        result=dict(self.group.last_diagnostics)
        result.update(output_torque=torque,frictionloss=self.frictions[0],
            actuator_vin=self.group._bam_model.actuator.vin,
            actuator_kp=self.group._bam_model.actuator.kp)
        return result


def live_fields(group):
    data=group._data;efc=data.efc
    fields={name:group._as_tensor(getattr(data,name))
            for name in ('qfrc_bias','qfrc_constraint','qfrc_actuator','nefc')}
    fields.update({'efc.'+name:group._as_tensor(getattr(efc,name)) for name in ('type','id','force')})
    fields.update({name:getattr(group,name) for name in
                   ('vin_tensor','vin_drop_gain','kp_scale','kd_scale','friction_scale','_dof_ids')
                   if getattr(group,name) is not None})
    return fields


def storage_signature(t):
    return (t.data_ptr(),tuple(t.shape),tuple(t.stride()),str(t.dtype),str(t.device))


class ComparisonFailure(RuntimeError):
    def __init__(self,details):
        self.details=details
        super().__init__('BAM same-input comparison failed: '+details['field']+
            '; max_error='+str(details.get('maximum_absolute_error'))+
            '; outside='+str(details.get('values_outside_tolerance')))


def check_outputs(torch,expected,actual):
    require(expected.keys()==actual.keys(),'BAM shadow output schema changed')
    worst=0.;checked=0
    for key,value in expected.items():
        candidate=actual[key]
        require(value.shape==candidate.shape and value.dtype==candidate.dtype,'BAM output shape/dtype changed')
        require(bool(torch.isfinite(value).all()) and bool(torch.isfinite(candidate).all()),'Nonfinite BAM output')
        error=(value-candidate).abs()
        worst=max(worst,float(error.max().item()));checked+=value.numel()
        threshold=ATOL+RTOL*value.abs()
        if not bool((error<=threshold).all()):
            raise ComparisonFailure(dict(field=key,maximum_absolute_error=float(error.max().item()),
                values_outside_tolerance=int((error>threshold).sum().item()),atol=ATOL,rtol=RTOL,
                expected=value.detach().reshape(-1).cpu().tolist(),
                actual=candidate.detach().reshape(-1).cpu().tolist()))
    return worst,checked


class GraphedBam:
    def __init__(self,group,torch,cmd):
        self.group=group;self.torch=torch;self.config=configuration(group)
        require(type(group).__name__=='FrictionDRBamActuator' and group.diagnostics_enabled,
                'Expected original audited non-backlash BAM actuator')
        require(group._prev_motor_torque is not None,'BAM recurrent state missing')
        self.fields=live_fields(group)
        require(all(isinstance(t,torch.Tensor) and t.is_cuda for t in self.fields.values()),
                'Graph requires existing CUDA tensor/bridge fields')
        self.signatures={n:storage_signature(t) for n,t in self.fields.items()}
        tensors={n:getattr(cmd,n).detach().clone() for n in COMMAND_FIELDS}
        require(all(t.is_cuda and t.dtype==torch.float32 and not t.requires_grad for t in tensors.values()),
                'Expected FP32 inference commands')
        self.command=dataclasses.replace(cmd,**tensors)
        self.previous=group._prev_motor_torque.detach().clone()
        # Three distinct stores: snapshot read by the graph, reusable graph
        # output, and committed recurrent state. A detach() alone does not copy.
        self.committed_previous=group._prev_motor_torque.detach().clone()
        self.shadow=Shadow(group);self.reference=Shadow(group)
        self.calls=0;self.checked_calls=0;self.scalar_comparisons=0;self.maximum_error=0.
        self.last_failure=None
        self.outputs=None
        device=tensors['pos'].device;current=torch.cuda.current_stream(device)
        stream=torch.cuda.Stream(device=device);stream.wait_stream(current)
        with torch.cuda.stream(stream):
            for _ in range(3):self.shadow.run(self.command,self.previous)
        current.wait_stream(stream);torch.cuda.synchronize(device)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            self.outputs=self.shadow.run(self.command,self.previous)
        current.wait_stream(stream)
        self.validate_recurrent_storage()
        self.damping=self.shadow.frictions[1]
        require(isinstance(self.damping,(int,float)) and math.isfinite(self.damping),'Unexpected damping output')
        # Holding fields and all graph outputs keeps captured storage alive.

    def validate_recurrent_storage(self):
        buffers=(self.previous,self.committed_previous,self.outputs['output_torque'])
        require(len({t.untyped_storage().data_ptr() for t in buffers})==3,
                'BAM snapshot, committed state and graph output must not share storage')

    def validate(self):
        require(configuration(self.group)==self.config,'BAM scalar configuration changed during graph run')
        current=live_fields(self.group)
        require(current.keys()==self.signatures.keys() and all(
            storage_signature(t)==self.signatures[n] for n,t in current.items()),
            'BAM backing storage changed; recapture is not automatic')

    def __call__(self,cmd):
        self.validate();self.validate_recurrent_storage();t=self.torch;g=self.group
        for name in COMMAND_FIELDS:
            original=getattr(cmd,name);target=getattr(self.command,name)
            require(original.shape==target.shape and original.dtype==target.dtype
                and original.device==target.device,'BAM command schema changed')
            target.copy_(original)
        require(g._prev_motor_torque.shape==self.previous.shape and
                g._prev_motor_torque.dtype==self.previous.dtype,'BAM previous torque schema changed')
        self.previous.copy_(g._prev_motor_torque)
        self.graph.replay()
        # Compare before friction writes and the recurrent-state commit.
        # These first 40 substep checks lie inside excluded timing warmup steps.
        if self.checked_calls<SHADOW_CHECK_CALLS:
            # The graph has already overwritten its reusable output. Compare
            # against the exact staged inputs, never a live output alias.
            expected=self.reference.run(self.command,self.previous)
            try:worst,count=check_outputs(t,expected,self.outputs)
            except ComparisonFailure as exc:
                self.last_failure=dict(**exc.details,attempted_replay=self.calls+1,
                    previous_source='pre_replay_snapshot',recurrent_stores_distinct=True)
                raise
            self.maximum_error=max(self.maximum_error,worst)
            self.scalar_comparisons+=count;self.checked_calls+=1
        require(self.reference.frictions is None or self.reference.frictions[1]==self.damping,
                'BAM damping changed')
        out=self.outputs
        g._write_frictions(out['frictionloss'],self.damping)
        self.committed_previous.copy_(out['output_torque'])
        g._prev_motor_torque=self.committed_previous
        g._bam_model.actuator.vin=out['actuator_vin']
        g._bam_model.actuator.kp=out['actuator_kp']
        # A fresh dictionary preserves Guard's per-substep freshness check.
        g.last_diagnostics={name:out[name] for name in self.shadow.group.last_diagnostics}
        self.calls+=1
        return out['output_torque']

    def stats(self):
        return dict(replays=self.calls,same_input_checked_calls=self.checked_calls,
            scalar_comparisons=self.scalar_comparisons,maximum_absolute_error=self.maximum_error,
            captured_live_fields=len(self.fields),warmup_physics_steps=0,precision='original FP32',
            previous_reference_source='pre_replay_snapshot',recurrent_stores_distinct=True,
            last_comparison_failure=self.last_failure)


class BamGraphs(Bindings):
    def __init__(self,groups,torch):
        super().__init__();self.runners=[];self.torch=torch
        try:
            for group in groups:
                holder=[None];self.runners.append(holder)
                def compute(cmd,group=group,holder=holder):
                    if holder[0] is None:holder[0]=GraphedBam(group,torch,cmd)
                    return holder[0](cmd)
                self.set(group,'compute',compute)
        except BaseException:self.close();raise
    def stats(self):return [None if h[0] is None else h[0].stats() for h in self.runners]


class BatchedAuditReads(Bindings):
    """Same 500-Hz CPU Audit, with one packed post-step transfer instead of three."""
    def __init__(self,guard,torch):
        super().__init__();self.calls=0
        device=guard.raw.sim.data.qpos.device
        qids=torch.tensor(guard.qids,device=device,dtype=torch.long)
        vids=torch.tensor(guard.vids,device=device,dtype=torch.long)
        cids=torch.tensor(guard.ctrl_indices,device=device,dtype=torch.long)
        nq,nv,nf=len(guard.qids),len(guard.vids),len(guard.ctrl_indices)
        def after():
            data=guard.raw.sim.data
            packed=torch.cat((data.qpos[0,qids],data.qvel[0,vids],data.actuator_force[0,cids]))
            values=packed.detach().cpu().tolist()
            require(len(values)==nq+nv+nf,'Audit transfer shape differs')
            guard.capture(lambda:guard.audit.after_integration(values[:nq],values[nq:nq+nv],values[nq+nv:]))
            self.calls+=1
        self.set(guard,'after_integration',after)
