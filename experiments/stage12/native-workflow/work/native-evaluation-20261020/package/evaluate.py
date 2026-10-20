"""Deterministic first-episode evaluation; never calls learn/update/backward."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from eval_common import *

def fingerprint(tensors):
    h=hashlib.sha256()
    for name,tensor in sorted(tensors.items()):
        value=tensor.detach().cpu().contiguous()
        h.update(name.encode());h.update(str((str(value.dtype),tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()

def initial_fingerprint(raw,obs):
    values={}
    for name in ('robot','ball','top_ball'):
        e=raw.scene[name]
        values.update({name+'/'+field:getattr(e.data,field) for field in
            ('root_link_pose_w','root_link_vel_w','joint_pos','joint_vel')})
    values.update({'obs/'+name:value for name,value in obs.items()})
    for field in sorted(raw.event_manager.domain_randomization_fields):
        values['randomized_model/'+field]=getattr(raw.sim.model,field)
    return dict(sha256=fingerprint(values),fields=sorted(values),
                scope='initial physical states, initial observations, declared randomized model fields')

class Observer:
    """Sample after the original metric computation, before any reset/forward."""
    def __init__(self,raw,torch,mdp):
        self.raw,self.torch,self.mdp=raw,torch,mdp
        self.original=raw.metrics_manager.compute
        names=raw.metrics_manager.active_terms
        self.stable_col=names.index('native_joint_geometric_stable_fraction')
        self.success_col=names.index('native_joint_geometric_success_5s')
        n,device=raw.num_envs,raw.device
        self.active=torch.ones(n,dtype=torch.bool,device=device)
        self.steps=torch.zeros(n,dtype=torch.long,device=device)
        self.stable_steps=torch.zeros_like(self.steps)
        self.current=torch.zeros_like(self.steps);self.longest=torch.zeros_like(self.steps)
        self.ever=torch.zeros(n,dtype=torch.bool,device=device)
        self.failures=torch.zeros((n,len(GATES)),dtype=torch.long,device=device)
        self.breaks=torch.zeros_like(self.failures)
        self.values=torch.zeros((n,len(VALUES)),dtype=torch.float64,device=device)

    def compute(self):
        self.original()
        raw,t,mdp=self.raw,self.torch,self.mdp
        p,v,_=mdp._stage11_head_kinematics(raw)
        r,b=raw.scene['robot'],raw.scene['ball']
        center=t.linalg.vector_norm(p[:,:2],dim=-1);height_error=(p[:,2]-.02).abs()
        speed=t.linalg.vector_norm(v,dim=-1)
        offset=t.linalg.vector_norm(r.data.root_link_pos_w[:,:2]-b.data.root_link_pos_w[:,:2],dim=-1)
        height=r.data.root_link_pos_w[:,2]-b.data.root_link_pos_w[:,2]
        tilt=mdp.wrestle_tilt(raw,'robot');ball_speed=t.linalg.vector_norm(b.data.root_link_lin_vel_w[:,:2],dim=-1)
        hold=mdp._bb_state(raw).hold
        values=t.stack((center,height_error,speed,offset,height,tilt,ball_speed,hold),dim=1)
        gates=t.stack((center<=.012,height_error<=.006,speed<=.08,offset<=.06,height>=.20,
                      height<=.29,tilt<=math.radians(20),ball_speed<=.15,hold<=1e-6),dim=1)
        cached=raw.metrics_manager._step_values
        stable=cached[:,self.stable_col].bool();ever=cached[:,self.success_col].bool()
        a=self.active
        valid=t.isfinite(values[a]).all() & (gates.all(dim=1)[a]==stable[a]).all()
        valid &= (hold[a]<=1e-6).all() & (raw.command_manager.get_command('twist')[a]==0).all()
        valid &= ~raw.termination_manager.get_term('nan_state')[a].any()
        need(bool(valid),'Nonfinite state, changed metric gates, nonzero command/assistance, or nan_state')
        self.steps+=a.long();self.stable_steps+=(stable&a).long()
        self.failures+=(~gates&a[:,None]).long()
        self.breaks+=(~gates&(a&(self.current>0))[:,None]).long()
        self.current=t.where(a,t.where(stable,self.current+1,t.zeros_like(self.current)),self.current)
        self.longest=t.maximum(self.longest,self.current)
        self.ever|=a&ever
        self.values+=t.where(a[:,None],values.double(),t.zeros_like(values,dtype=t.float64))
        need(bool(((self.longest[a]>=250)==self.ever[a]).all()),'Cached five-second latch differs from independent consecutive-step count')

    def rows(self,seed,terminated,terms):
        arrays={name:getattr(self,name).cpu().tolist() for name in
                ('steps','stable_steps','longest','ever','failures','breaks','values')}
        return [dict(seed=seed,env_id=i,steps=arrays['steps'][i],stable_steps=arrays['stable_steps'][i],
            longest_stable_steps=arrays['longest'][i],ever_5s=arrays['ever'][i],
            terminated=terminated[i],success=arrays['ever'][i] and not terminated[i],
            terminations=terms[i],gate_failure_counts=arrays['failures'][i],run_break_counts=arrays['breaks'][i],
            value_sums=arrays['values'][i]) for i in range(self.raw.num_envs)]

def evaluate_group(key,seed,index,boundary,update_progress):
    import torch
    from tensordict import TensorDict
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
    from mjlab_microduck.tasks.microduck_basketball_env_cfg import MicroduckBasketballRlCfg
    from mjlab_microduck.tasks import mdp
    from native_task import make_cfg,head_scale
    cfg=make_cfg(NUM_ENVS,seed);cfg.auto_reset=False
    need(abs(cfg.episode_length_s-10.)<1e-12 and not cfg.curriculum,'Evaluation horizon or curriculum changed')
    agent=asdict(deepcopy(MicroduckBasketballRlCfg))
    agent.update(seed=seed,logger='tensorboard',upload_model=False)
    agent['algorithm']['learning_rate']=5e-5;agent['algorithm']['schedule']='fixed'
    agent['algorithm']['num_mini_batches']=4
    raw=env=observer=None
    try:
        update_progress(status='EVALUATING',phase='build_environment',model=key,seed=seed,group=index+1,control_steps_in_group=0)
        raw=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode=None)
        # Training used the wrapper default (no clipping); preserve it exactly.
        env=RslRlVecEnvWrapper(raw)
        runner=MjlabOnPolicyRunner(env,agent,log_dir=None,device='cuda:0')
        path=model_path(key)
        need(sha(path)==MODELS[key],'Checkpoint changed before loading')
        saved=torch.load(path,map_location='cpu',weights_only=False)
        meta=saved['infos']['stage12_native']
        need(meta['package_identity']==ORIGIN_ID and meta['policy_updates']==int(key),'Checkpoint metadata differs')
        runner.alg.load(saved,{'actor':True,'critic':True,'optimizer':False,'iteration':False},strict=True)
        policy=runner.get_inference_policy(device='cuda:0');policy.reset()
        need(not policy.training and not policy.obs_normalizer.training,'Policy/normalizer must be frozen in evaluation')
        before=fingerprint(policy.state_dict())
        # Both trained candidates have already completed the ramp; use one identical clock.
        raw.common_step_counter=1000*24;raw._sim_step_counter=1000*24*10
        raw_obs,_=raw.reset(seed=seed)
        obs=TensorDict(raw_obs,batch_size=[NUM_ENVS]);policy.reset()
        need(raw.max_episode_length==HORIZON_STEPS and abs(raw.step_dt-.02)<1e-12 and head_scale(raw)==1.,'Evaluation time contract differs')
        need(tuple(obs['actor'].shape)==(NUM_ENVS,61) and tuple(obs['critic'].shape)==(NUM_ENVS,85),'Observation contract differs')
        initial=initial_fingerprint(raw,obs)
        init_path=SLOT/('initial-seed-%d.json'%seed)
        if init_path.exists():
            need(read(init_path)==initial,'The two policies do not share identical initial state/observation/declared DR fields')
        else:
            need(key=='4500','Initial reference missing for the first candidate')
            write(init_path,initial,exclusive=True)
        observer=Observer(raw,torch,mdp);raw.metrics_manager.compute=observer.compute
        terminated=[False]*NUM_ENVS;terms=[[] for _ in range(NUM_ENVS)]
        with torch.inference_mode():
            for step in range(1,HORIZON_STEPS+1):
                boundary()
                actions=policy(obs,stochastic_output=False)
                need(bool(torch.isfinite(actions).all()),'Nonfinite deterministic action')
                obs,rewards,dones,_=env.step(actions)
                finite=[torch.isfinite(v).all() for v in obs.values()]
                finite.append(torch.isfinite(rewards).all())
                need(bool(torch.stack(finite).all()),'Nonfinite evaluation observation/reward')
                ended=observer.active&dones.bool()
                ids=ended.nonzero(as_tuple=False).flatten().cpu().tolist()
                if ids:
                    physical=raw.reset_terminated.cpu().tolist()
                    term_values={name:raw.termination_manager.get_term(name).cpu().tolist()
                                 for name in raw.termination_manager.active_terms}
                    for i in ids:
                        terminated[i]=bool(physical[i]);terms[i]=[name for name,val in term_values.items() if val[i]]
                observer.active &= ~ended
                done_count=NUM_ENVS-int(observer.active.sum().item())
                if step%25==0 or done_count==NUM_ENVS:
                    update_progress(phase='deterministic_evaluation',model=key,seed=seed,group=index+1,
                        control_steps_in_group=step,completed_first_episodes=index*NUM_ENVS+done_count,
                        progress_percent=100*(index*NUM_ENVS+done_count)/(len(MODELS)*len(SEEDS)*NUM_ENVS))
                if done_count==NUM_ENVS:
                    break
                reset_ids=dones.nonzero(as_tuple=False).flatten()
                if len(reset_ids):
                    raw.reset(env_ids=reset_ids);policy.reset(dones.bool())
                    obs=env.get_observations()
        need(not bool(observer.active.any()),'Some first episodes did not terminate at the 10-second horizon')
        rows=observer.rows(seed,terminated,terms)
        need(fingerprint(policy.state_dict())==before,'Policy weights or normalizer changed during evaluation')
        need(sha(path)==MODELS[key],'Original checkpoint changed during evaluation')
        result=dict(status='EVALUATED',model=key,seed=seed,checkpoint_sha256=MODELS[key],
            initial=initial,summary=summarize(rows),episodes=rows,policy_and_normalizer_unchanged=True,
            vector_steps_executed=step,all_environment_control_transitions=step*NUM_ENVS,
            counted_first_episode_steps=sum(r['steps'] for r in rows))
        write(SLOT/('group-%s-%d.json'%(key,seed)),result,exclusive=True)
        return result
    finally:
        if observer is not None:raw.metrics_manager.compute=observer.original
        if env is not None:env.close()
        elif raw is not None:raw.close()

def main():
    started=read(SLOT/'started.json');state=dict(status='INITIALIZING',phase='load_runtime',
        started_utc=started['started_utc'],deadline_utc=started['deadline_utc'],added_PPO_updates=0,
        total_first_episodes=len(MODELS)*len(SEEDS)*NUM_ENVS,completed_first_episodes=0,progress_percent=0.)
    stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
    def boundary():
        need(not stopped[0] and time.time()<started['deadline_utc'],'Evaluation stopped or original evaluation time limit reached')
    def progress(**kw):
        state.update(kw,elapsed_seconds=time.time()-started['started_utc']);write(SLOT/'progress.json',state)
    groups=[]
    try:
        progress()
        expected={'mjlab':'1.3.0','mujoco':'3.10.0','mujoco-warp':'3.8.1','warp-lang':'1.12.0','rsl-rl-lib':'5.0.1'}
        need({k:importlib.metadata.version(k) for k in expected}==expected,'Existing runtime versions differ; no installation is performed')
        sys.path.insert(0,str(SOURCE/'src'));sys.path.insert(0,str(TRAIN/'package'))
        import torch
        import warp as wp
        from mjlab.utils.torch import configure_torch_backends
        need(torch.cuda.is_available() and '5090' in torch.cuda.get_device_name(0),'Run evaluation only on the existing cloud RTX5090')
        configure_torch_backends();torch.set_num_threads(min(4,len(os.sched_getaffinity(0))))
        wp.config.kernel_cache_dir=str(TRAIN/'cache/warp')
        for key in MODELS:
            for seed in SEEDS:
                boundary();groups.append(evaluate_group(key,seed,len(groups),boundary,progress))
        rows={key:[r for g in groups if g['model']==key for r in g['episodes']] for key in MODELS}
        summaries={key:summarize(value) for key,value in rows.items()}
        ranked=sorted(MODELS,key=lambda key:(summaries[key]['successes'],summaries[key]['mean_survival_seconds'],
            summaries[key]['longest_stable_seconds']['median'],-int(key)),reverse=True)
        report=dict(status='EVALUATION_COMPLETE',added_PPO_updates=0,thresholds=THRESHOLDS,
            checkpoints=MODELS,seeds=list(SEEDS),num_envs_per_group=NUM_ENVS,groups=len(groups),
            summaries=summaries,paired=paired_comparison(rows['4500'],rows['6000']),
            candidate_for_review=ranked[0],selection_rule='success count, survival mean, median longest run, earlier checkpoint',
            independent_of_training_rollouts=True,selection_set_not_final_heldout_test=True,
            task_acceptance_claimed=False,old_strict_physics_acceptance=False,video_review_done=False,
            success_definition='First episode: >=250 consecutive original metric ticks, and no physical termination before the episode ends; timeout allowed.',
            policy_mode='deterministic; frozen model and normalization; existing observation noise and DR retained',
            pairing='Identical initial physical state, initial observations and declared randomized model fields; later noise sequences need not be identical after different resets.',
            all_environment_control_transitions=sum(g['all_environment_control_transitions'] for g in groups),
            counted_first_episode_steps=sum(g['counted_first_episode_steps'] for g in groups),
            physics_substeps_per_control_step=10,
            group_summaries=[{k:g[k] for k in ('model','seed','summary','initial','vector_steps_executed')} for g in groups],
            started_utc=started['started_utc'],finished_utc=time.time())
        write(SLOT/'evaluation-result.json',report,exclusive=True)
        progress(status='EVALUATION_COMPLETE',phase='finished',completed_first_episodes=512,progress_percent=100.)
    except BaseException as exc:
        state.update(status='EVALUATION_ERROR_PRESERVE',error=repr(exc),traceback=traceback.format_exc()[-12000:],
            completed_groups=len(groups),finished_utc=time.time())
        write(SLOT/'progress.json',state)
        return 2
    return 0

if __name__=='__main__':raise SystemExit(main())
