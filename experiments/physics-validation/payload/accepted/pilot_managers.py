"""Nine installed managers on two private worlds, diagnostic terms only.

Native _reset_idx order is exercised. Public reset/step and physical writes are
not called. CandidateEvent retains the previously reviewed local reset method.
"""
import copy
import hashlib
import importlib
import inspect
from pathlib import Path
from types import SimpleNamespace, MethodType

from event_reset_candidate import reset_consistent
from lifecycle_support import require
from report_io import save
from isolation_policy import install as install_isolation


def build(scene, sim, action_cfg, gate, counts, out, torch, pins, caps, bank, act):
    names=('action','observation','event','reward','termination','metrics','command','curriculum','recorder')
    api=SimpleNamespace(**{n:importlib.import_module('mjlab.managers.'+n+'_manager') for n in names})
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.envs.mdp.actions.actions import JointPositionAction
    from mjlab.utils.noise.noise_cfg import UniformNoiseCfg
    from mjlab_microduck.tasks import mdp
    require(api.reward.RewardManager.compute is mdp._nan_safe_reward_compute,'Project reward wrapper differs')
    native_reset=ManagerBasedRlEnv._reset_idx
    callables={'env._reset_idx':native_reset,'reward.active':api.reward.RewardManager.compute,
        'reward.original':mdp._orig_reward_compute,'event.native_reset':api.event.EventManager.reset,
        'noise.apply':UniformNoiseCfg.apply,'action.process':JointPositionAction.process_actions,
        'action.term_reset':JointPositionAction.reset}
    for name in names:
        cls=getattr(getattr(api,name),name.title()+'Manager')
        callables[name+'.constructor']=cls.__init__
        if hasattr(cls,'reset'):callables[name+'.reset']=cls.reset
    pinmap={p.resolve():h for p,h in pins.items()};provenance={}
    for name,fn in callables.items():
        p=Path(inspect.getsourcefile(fn)).resolve()
        require(p in pinmap and hashlib.sha256(p.read_bytes()).hexdigest()==pinmap[p],'Unpinned callable '+name)
        provenance[name]=dict(module=fn.__module__,qualname=fn.__qualname__,source_path=str(p),sha256=pinmap[p])
    save(out/'active-callable-review.json',provenance)
    device='cuda:0';zeros=lambda *shape,**kw:torch.zeros(*shape,device=device,**kw)
    env=SimpleNamespace(num_envs=1,device=device,physics_dt=.002,step_dt=.02,max_episode_length_s=2.,
        cfg=SimpleNamespace(decimation=10,auto_reset=False),scene=scene,sim=sim,extras={},obs_buf={},
        reward_buf=zeros(1),reset_buf=zeros(1,dtype=torch.bool),reset_terminated=zeros(1,dtype=torch.bool),
        reset_time_outs=zeros(1,dtype=torch.bool),episode_length_buf=zeros(1,dtype=torch.int64),
        _manual_reset_pending=zeros(1,dtype=torch.bool),common_step_counter=0,_sim_step_counter=0,
        trace=[],events=zeros(4,1,dtype=torch.int64),level=zeros(1),observation_calls=0,
        fail=zeros(1,dtype=torch.bool),timeout=zeros(1,dtype=torch.bool),features=zeros(1))
    env._reset_idx=MethodType(native_reset,env)
    A,O,E,R,T,M,C,U,D=(getattr(api,n) for n in names)
    class DiagnosticCommand(C.CommandTerm):
        def __init__(self,cfg,env):
            super().__init__(cfg,env);self._command=zeros(1,2);self.metrics={'calls':zeros(1)};self.updates=0
        @property
        def command(self):return self._command
        def _resample_command(self,env_ids):self._command[env_ids]=torch.rand(len(env_ids),2,device=device)
        def _update_metrics(self):self.metrics['calls']+=1
        def _update_command(self):self.updates+=1
    class CommandCfg(C.CommandTermCfg):
        def build(self,env):return DiagnosticCommand(self,env)
    class MemoryRecorder(D.RecorderTerm):
        def __init__(self,cfg,env):super().__init__(cfg,env);self.entries=[];self.closed=False
        def entry(self,hook,env_ids=None):
            self.entries.append(dict(hook=hook,ids=None if env_ids is None else env_ids.clone(),
                qpos=sim.data.qpos.clone(),time=sim.data.time.clone(),action=env.action_manager.action.clone(),
                obs=env.obs_buf['diagnostic'].clone(),reward=env.reward_buf.clone()))
        def record_pre_reset(self,env_ids):self.entry('pre',env_ids)
        def record_post_reset(self,env_ids):self.entry('reset',env_ids)
        def record_post_step(self):self.entry('step')
        def close(self):self.closed=True
    def reset_flags(env,env_ids):
        env.events[0,env_ids]+=1;env.fail[env_ids]=False;env.timeout[env_ids]=False
    def event_step(env,env_ids):env.events[1]+=1
    def event_interval(env,env_ids):env.events[2,env_ids]+=1
    def event_global(env,env_ids):env.events[3]+=1
    class LocalIntervalClass:
        def __init__(self,cfg,env):
            self.resets=zeros(1,dtype=torch.int64);self.fires=zeros(1,dtype=torch.int64)
        def reset(self,env_ids=None):self.resets[env_ids]+=1
        def __call__(self,env,env_ids):self.fires[env_ids]+=1
    def curriculum(env,env_ids):env.level[env_ids]+=1;return {'mean':torch.mean(env.level)}
    def observed(name,fn):
        limited=gate.limited(name,caps[name],fn)
        def invoke(env):env.observation_calls+=1;return limited(env)
        return invoke
    action_obs=observed('observation_action_reads',lambda env:env.action_manager.action)
    def height(env):
        value=scene['foot_height_scan'].data.heights.reshape(1,-1)
        require(tuple(value.shape)==(1,2) and bool(torch.isfinite(value).all()),'Height observation differs')
        return value
    height_obs=observed('observation_height_reads',height)
    class CandidateEvent(E.EventManager):reset=reset_consistent
    def construct(name,cls,*args):return gate.limited(name,caps[name],cls)(*args)
    env.event_manager=construct('EventManager',CandidateEvent,{
        'reset':E.EventTermCfg(func=reset_flags,mode='reset'),
        'step':E.EventTermCfg(func=event_step,mode='step'),
        'interval':E.EventTermCfg(func=event_interval,mode='interval',interval_range_s=(.25,.35)),
        'interval_class':E.EventTermCfg(func=LocalIntervalClass,mode='interval',interval_range_s=(.35,.45)),
        'interval_global':E.EventTermCfg(func=event_global,mode='interval',interval_range_s=(.5,.6),is_global_time=True)},env)
    env.command_manager=construct('CommandManager',C.CommandManager,{'diagnostic':CommandCfg(resampling_time_range=(.25,.35))},env)
    env.action_manager=construct('ActionManager',A.ActionManager,{'joint_pos':copy.deepcopy(action_cfg)},env)
    am=env.action_manager;action=am.get_term('joint_pos')
    require(type(action) is JointPositionAction and am.total_action_dim==14 and action._scale==1. and
        action.cfg.clip is None and action.cfg.use_default_offset,'JointPositionAction differs')
    env.observation_manager=construct('ObservationManager',O.ObservationManager,{
        'diagnostic':O.ObservationGroupCfg(terms={
            'action_history':O.ObservationTermCfg(func=action_obs,noise=UniformNoiseCfg(n_min=-2**-12,n_max=2**-12),
                delay_min_lag=1,delay_max_lag=2,delay_per_env=True,delay_update_period=3,
                delay_hold_prob=0.,delay_per_env_phase=True,history_length=3,flatten_history_dim=True),
            'foot_height':O.ObservationTermCfg(func=height_obs)},concatenate_terms=True,enable_corruption=True,nan_policy='error')},env)
    env.termination_manager=construct('TerminationManager',T.TerminationManager,{
        'failure':T.TerminationTermCfg(func=lambda env:env.fail.clone()),
        'timeout':T.TerminationTermCfg(func=lambda env:env.timeout.clone(),time_out=True)},env)
    env.reward_manager=construct('RewardManager',R.RewardManager,{
        'feet_on_ball':R.RewardTermCfg(func=lambda env:torch.full((1,),1000.,device=device),weight=1.)},env)
    env.curriculum_manager=construct('CurriculumManager',U.CurriculumManager,{'level':U.CurriculumTermCfg(func=curriculum)},env)
    env.metrics_manager=construct('MetricsManager',M.MetricsManager,{
        'sub':M.MetricsTermCfg(func=lambda env:env.features.clone(),per_substep=True),
        'last':M.MetricsTermCfg(func=lambda env:env.features.clone(),reduce='last')},env)
    env.recorder_manager=construct('RecorderManager',D.RecorderManager,{'memory':D.RecorderTermCfg(func=MemoryRecorder)},env)
    am,om,em,rm,tm,mm,cm,um,dm=(getattr(env,n+'_manager') for n in names)
    require(om.group_obs_dim=={'diagnostic':(44,)},'Observation dimensions differ')
    noise=om.get_term_cfg('diagnostic','action_history').noise;command=cm.get_term('diagnostic');rec=dm._terms[0]
    event_class=em._mode_term_cfgs['interval'][1].func
    require(isinstance(event_class,LocalIntervalClass),'Mixed Event class construction differs')
    install_isolation(env,act,noise,command,bank,gate,torch,api,out)
    roots=[('RNGService',bank,('states','seeds','draw_counts')),('EventClass',event_class,('resets','fires')),('env',env,('extras','obs_buf','reward_buf','reset_buf','reset_terminated','reset_time_outs',
        'episode_length_buf','common_step_counter','_sim_step_counter','_manual_reset_pending','trace',
        'events','level','observation_calls','fail','timeout','features')),
        ('ActionManager',am,('_action','_prev_action','_prev_prev_action')),
        ('JointPositionAction',action,('_raw_actions','_processed_actions','_scale','_offset','_target_ids')),
        ('ObservationManager',om,('_obs_buffer','_group_obs_term_delay_buffer','_group_obs_term_history_buffer')),
        ('UniformNoiseCfg',noise,('_tensor_cache',)),
        ('EventManager',em,('_interval_term_time_left','_reset_term_last_triggered_step_id','_reset_term_last_triggered_once')),
        ('RewardManager',rm,('_reward_buf','_step_reward','_episode_sums')),
        ('TerminationManager',tm,('_term_dones','_truncated_buf','_terminated_buf')),
        ('MetricsManager',mm,('_episode_sums','_substep_accum','_substep_episode_sums','_substep_count','_step_count','_step_values')),
        ('Command',command,('metrics','time_left','command_counter','_command','updates')),
        ('CommandManager',cm,('_commands',)),('CurriculumManager',um,('_curriculum_state',)),('Recorder',rec,('entries','closed'))]
    specs=[('Action',am,('process_action','reset')),('Observation',om,('compute','reset')),
        ('Event',em,('apply','reset')),('Reward',rm,('compute','reset')),('Termination',tm,('compute','reset')),
        ('Metrics',mm,('compute','reset')),('Command',cm,('compute','reset')),
        ('Curriculum',um,('compute','reset')),('Recorder',dm,('record_pre_reset','record_post_reset','record_post_step')),
        ('Env',env,('_reset_idx',))]
    for prefix,obj,methods in specs:
        for name in methods:
            label=prefix+'_'+name;limited=gate.limited(label,caps[label],getattr(obj,name))
            def logged(*args,_fn=limited,_name=prefix+'.'+name,**kwargs):
                env.trace.append(dict(call=_name,**{k:v for k,v in kwargs.items() if type(v) in (str,int,float,bool)}))
                return _fn(*args,**kwargs)
            gate.patch(obj,name,logged)
    for obj,name,label in ((am,'apply_action','Action_apply'),(mm,'compute_substep','Metrics_substep')):
        gate.patch(obj,name,gate.limited(label,caps[label],getattr(obj,name)))
    gate.patch(dm,'close',gate.blocked('Recorder.close'))
    return SimpleNamespace(env=env,roots=roots,action=action,noise=noise,command=command,recorder=rec,api=api)


