"""Diagnostic-only per-world RNG and reset boundary policy; never install/merge.

All random tensors are sampled on cuda:0 with explicit torch.Generator objects.
Only generator byte states are retained as ordinary snapshotted tensor owners.
Defaults are never rewound to hide draws. Constructors precede this policy;
public RL reset/step, general user callbacks and CUDA graphs remain out of scope.
APIs: https://docs.pytorch.org/docs/2.9/generated/torch.Generator.html
https://docs.pytorch.org/docs/2.9/generated/torch.rand.html
https://docs.pytorch.org/docs/2.9/generated/torch.randint.html
"""
from numbers import Integral
from types import MethodType
from pathlib import Path
import hashlib
import inspect
import math
from lifecycle_support import require
from report_io import save
from scene_evidence import save_arrays

CHANNELS=('bam_lag','obs_lag','obs_phase','obs_noise','command_time','command_value','event_local_0','event_local_1')


def canonical_ids(mask,num_envs):
    """Strict pre-mutation selector; empty is a true reset no-op boundary."""
    require(type(num_envs) is int and num_envs>0,'Invalid world count')
    if mask is None:return tuple(range(num_envs))
    if isinstance(mask,slice):return tuple(range(*mask.indices(num_envs)))
    if hasattr(mask,'tolist'):mask=mask.tolist()
    require(isinstance(mask,(list,tuple)),'Selector must be 1D indices, bool mask, slice or None')
    if not mask:return ()
    if all(type(v) is bool for v in mask):
        require(len(mask)==num_envs,'Boolean mask length mismatch')
        return tuple(i for i,v in enumerate(mask) if v)
    require(all(isinstance(v,Integral) and type(v) is not bool for v in mask),'Indices must be integral and not mixed booleans')
    ids=tuple(int(v) for v in mask)
    require(len(set(ids))==len(ids) and all(0<=v<num_envs for v in ids),'Duplicate or out-of-range world index')
    return ids


class StreamBank:
    def __init__(self,torch,gate,counts,out,caps,logical_world):
        require(type(logical_world) is int and logical_world in (0,1),'Explicit logical world required')
        self.logical_world=logical_world
        self.torch,self.counts,self.out=torch,counts,out
        self.states={};self.seeds={};self.draw_counts={}
        keys=[(c,logical_world) for c in CHANNELS]+[('event_global_2',0)]
        for channel,world in keys:
            key=f'{channel}/{world}'
            seed=int.from_bytes(hashlib.sha256(('stage11-isolation-v1/'+key).encode()).digest()[:8],'little')&((1<<63)-1)
            g=torch.Generator(device='cuda:0');g.manual_seed(seed)
            state=g.get_state();require(state.device.type=='cpu' and state.dtype==torch.uint8,'Generator state format differs')
            self.states[key]=state.clone();self.seeds[key]=seed;self.draw_counts[key]=0
        require(len(set(self.seeds.values()))==9,'Stream seed collision')
        self.draw=gate.limited('RNG_sample',caps['RNG_sample'],self._draw)

    def _draw(self,channel,world,shape=(),integer_range=None):
        require(world==0,'Singleton local world index must be zero')
        local_world=world
        world=0 if channel=='event_global_2' else self.logical_world
        key=f'{channel}/{world}';require(key in self.states,'Unknown random stream')
        shape=tuple(shape);require(all(type(n) is int and n>=0 for n in shape) and 0<math.prod(shape)<=14,'Draw shape outside bounded scope')
        t=self.torch;before=self.states[key].clone();g=t.Generator(device='cuda:0');g.set_state(before)
        if integer_range is None:value=t.rand(shape,device='cuda:0',dtype=t.float32,generator=g)
        else:value=t.randint(integer_range[0],integer_range[1],shape,device='cuda:0',dtype=t.int64,generator=g)
        after=g.get_state();self.states[key].copy_(after);self.draw_counts[key]+=1
        index=self.counts['RNG_sample_attempted']
        save_arrays(self.out,f'stream-draw-{index:03d}',dict(before=before.numpy().copy(),after=after.numpy().copy(),sample=value.detach().cpu().numpy().copy()),
            channel=channel,world=world,local_world=local_world,seed=self.seeds[key],shape=list(shape),integer_range=integer_range,stream_draw_count=self.draw_counts[key])
        return value


def reset_delay(buffer,mask,bank,channel,torch):
    ids=canonical_ids(mask,buffer.batch_size)
    if not ids:return
    idx=torch.tensor(ids,dtype=torch.long,device=buffer.device)
    buffer._buffer.reset(batch_ids=idx);buffer._current_lags[idx]=0;buffer._step_count[idx]=0
    if buffer.update_period>0 and buffer.per_env_phase:
        require(channel=='obs_lag','Unexpected phase-enabled delay')
        for w in ids:buffer._phase_offsets[w]=bank.draw('obs_phase',w,integer_range=(0,buffer.update_period))


def sample_lags(buffer,mask,bank,channel):
    require(buffer.per_env and buffer.hold_prob==0.,'Only fixed per-world no-hold delay configuration supported')
    result=buffer._current_lags.clone()
    for w,active in enumerate(mask.tolist()):
        if active:result[w]=bank.draw(channel,w,integer_range=(buffer.min_lag,buffer.max_lag+1))
    return result


def reset_observation(manager,mask,native,torch):
    ids=canonical_ids(mask,manager._env.num_envs)
    if not ids:return {}
    return native(manager,torch.tensor(ids,dtype=torch.long,device=manager._env.device))


def reset_event(manager,mask,bank,torch):
    ids=canonical_ids(mask,manager._env.num_envs)
    if not ids:return {}
    idx=torch.tensor(ids,dtype=torch.long,device=manager.device)
    for cfgs in manager._mode_class_term_cfgs.values():
        for cfg in cfgs:cfg.func.reset(env_ids=idx)
    for index,cfg in enumerate(manager._mode_term_cfgs.get('interval',[])):
        if cfg.is_global_time:continue
        low,high=cfg.interval_range_s
        for w in ids:manager._interval_term_time_left[index][w]=bank.draw(f'event_local_{index}',w)*(high-low)+low
    return {}


def apply_interval(manager,dt,bank,torch):
    require(dt is not None and math.isfinite(float(dt)) and dt>=0,'Invalid interval dt')
    strongest=0
    for index,cfg in enumerate(manager._mode_term_cfgs['interval']):
        timer=manager._interval_term_time_left[index];timer-=dt;fired=False
        low,high=cfg.interval_range_s
        if cfg.is_global_time:
            if timer.item()<1e-6:
                timer[:]=(bank.draw(f'event_global_{index}',0)*(high-low)+low).to(timer.device)
                cfg.func(manager._env,None,**cfg.params);fired=True
        else:
            ids=tuple(int(w) for w in (timer<1e-6).nonzero().flatten().tolist())
            if ids:
                for w in ids:timer[w]=bank.draw(f'event_local_{index}',w)*(high-low)+low
                cfg.func(manager._env,torch.tensor(ids,dtype=torch.long,device=manager.device),**cfg.params);fired=True
        if fired:strongest=max(strongest,getattr(cfg.func,'recompute',0))
    if strongest:manager._env.sim.recompute_constants(strongest)


def resample_command(command,mask,bank,torch):
    ids=canonical_ids(mask,command._env.num_envs)
    low,high=command.cfg.resampling_time_range
    for w in ids:
        command.time_left[w]=bank.draw('command_time',w)*(high-low)+low
        command._command[w]=bank.draw('command_value',w,(2,));command.command_counter[w]+=1


def apply_noise(noise,data,bank,torch):
    require(tuple(data.shape)==(1,14) and data.dtype==torch.float32 and noise.operation=='add','Noise scope differs')
    lo=noise._get_cached_tensor('n_min',noise.n_min,data.device);hi=noise._get_cached_tensor('n_max',noise.n_max,data.device)
    require(float(noise.n_min)==-2**-12 and float(noise.n_max)==2**-12,'Noise bounds changed')
    random=torch.stack([bank.draw('obs_noise',w,(14,)) for w in range(1)],dim=0)
    noise_value=random*(hi-lo)+lo
    return data+noise_value


def install(env,act,noise,command,bank,gate,torch,api,out):
    """Attach only to private diagnostic instances; Gate removes class dispatch."""
    bam=act._delay_buffer;obs=env.observation_manager._group_obs_term_delay_buffer['diagnostic']['action_history']
    cls=type(bam);require(type(obs) is cls,'Delay class differs')
    native_sample,native_reset=cls._sample_lags,cls.reset
    def sample(self,mask):
        if self is bam:return sample_lags(self,mask,bank,'bam_lag')
        if self is obs:return sample_lags(self,mask,bank,'obs_lag')
        return native_sample(self,mask)
    def delay_reset(self,batch_ids=None):
        if self is bam:return reset_delay(self,batch_ids,bank,'bam_lag',torch)
        if self is obs:return reset_delay(self,batch_ids,bank,'obs_lag',torch)
        return native_reset(self,batch_ids)
    gate.patch(cls,'_sample_lags',sample);gate.patch(cls,'reset',delay_reset)
    gate.patch(noise,'apply',lambda data:apply_noise(noise,data,bank,torch))
    gate.patch(command,'_resample',lambda env_ids:resample_command(command,env_ids,bank,torch))
    om=env.observation_manager;em=env.event_manager
    native_obs=api.observation.ObservationManager.reset;native_event=api.event.EventManager.apply
    gate.patch(om,'reset',lambda env_ids=None:reset_observation(om,env_ids,native_obs,torch))
    gate.patch(em,'reset',lambda env_ids=None:reset_event(em,env_ids,bank,torch))
    def event_apply(mode,env_ids=None,dt=None,global_env_step_count=None):
        if mode=='interval':
            require(env_ids is None,'Interval selection is determined by timer expiry')
            return apply_interval(em,dt,bank,torch)
        return native_event(em,mode,env_ids=env_ids,dt=dt,global_env_step_count=global_env_step_count)
    gate.patch(em,'apply',event_apply)
    functions=(canonical_ids,reset_delay,sample_lags,reset_observation,reset_event,apply_interval,resample_command,apply_noise,StreamBank._draw)
    source=Path(__file__);save(out/'isolation-callable-review.json',dict(source=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        functions={f.__qualname__:dict(first_line=inspect.getsourcelines(f)[1],module=f.__module__) for f in functions},
        installed_sources_modified=False,distribution_bounds_retained=True,random_sequence_policy_changed=True,
        channels=list(bank.seeds),shared_global_timer_policy='Preserve phase across episode reset; independent global event stream',
        constructor_RNG_scope='Native constructor draws precede recorded initial state; default RNG must remain unchanged afterwards'))
