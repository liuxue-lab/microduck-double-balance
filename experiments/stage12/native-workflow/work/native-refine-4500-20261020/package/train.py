"""Native PPO, 4096 environments, one bounded segment; no repeated smoke."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib.metadata
import math
import os
import random
import signal
import sys
import time
import traceback
from common import *
from runtime_helpers import JsonScalarWriter,require_finite_group,require_step_outputs,host_scalar_means

class TrainingStop(Exception):pass

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--segment',type=int,required=True);ap.add_argument('--session',type=int,required=True);a=ap.parse_args()
    budget=read(SLOT/'budget.json');manifest=read(HERE/'manifest.json');out=SLOT/'run'
    target=min(a.segment*SEGMENT_UPDATES,MAX_UPDATES)
    state=dict(status='INITIALIZING',phase='load_runtime',segment=a.segment,session=a.session,num_envs=NUM_ENVS,
        completed_updates=budget['completed_updates'],attempted_updates=budget['attempted_updates'],max_updates=MAX_UPDATES,
        remaining_validation_runs=0,progress_percent=100*budget['attempted_updates']/MAX_UPDATES)
    raw=env=runner=None;in_rollout=[0];update_in_flight=[False];stop=[False];last_saved=[None];last_save_time=[0.]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def progress(**kw):state.update(kw,elapsed_seconds=time.time()-budget['started_utc']);write(SLOT/'progress.json',state)
    def boundary():
        if stop[0] or (SLOT/'STOP.json').exists() or time.time()>=budget['deadline_utc']:raise TrainingStop('Stop requested or original two-hour deadline reached')
    def save_model(_path=None,infos=None):
        if _path is not None:return
        completed=budget['completed_updates'];path=out/'checkpoints'/('refine_%06d.pt'%completed)
        if path.exists():
            receipt=read(path.with_suffix('.json'));require(sha(path)==receipt['sha256'],'Existing refinement checkpoint differs')
            require(receipt['completed_updates']==completed,'Existing checkpoint counter differs')
            write(out/'latest.json',receipt);last_saved[0]=completed;return
        meta=dict(package_identity=manifest['identity'],origin_package_identity=ORIGIN_ID,origin_checkpoint_sha256=MODEL_SHA,
            policy_updates=4500+completed,refine_updates=completed,campaign_attempted_updates=budget['attempted_updates'],
            num_envs=NUM_ENVS,common_step_counter=raw.common_step_counter,sim_step_counter=raw._sim_step_counter,
            lower_assistance=0.,learning_rate=LEARNING_RATE,physical_trajectory_resume=False,
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
        saved=runner.alg.save();saved.update(iter=4500+completed-1,infos={'stage12_native':meta,'stage12_refine':meta,
            'env_state':{'common_step_counter':raw.common_step_counter}})
        tmp=path.with_suffix('.tmp.pt');require(not tmp.exists(),'An earlier partial checkpoint exists; preserve it')
        torch.save(saved,tmp)
        with tmp.open('rb') as f:os.fsync(f.fileno())
        os.replace(tmp,path)
        receipt=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,completed_updates=completed,
                     policy_updates=4500+completed,attempted_updates=budget['attempted_updates'])
        write(path.with_suffix('.json'),receipt,exclusive=True);write(out/'latest.json',receipt)
        last_saved[0]=completed;last_save_time[0]=time.monotonic()
    try:
        progress();boundary()
        expected={'mjlab':'1.3.0','mujoco':'3.10.0','mujoco-warp':'3.8.1','warp-lang':'1.12.0','rsl-rl-lib':'5.0.1'}
        require({k:importlib.metadata.version(k) for k in expected}==expected,'Existing runtime versions differ; no installation')
        sys.path.insert(0,str(SOURCE/'src'))
        import numpy as np
        import torch
        import warp as wp
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
        from mjlab.utils.torch import configure_torch_backends
        from mjlab_microduck.tasks.microduck_basketball_env_cfg import MicroduckBasketballRlCfg
        from mjlab_microduck.tasks import mdp
        from native_task import make_cfg,head_scale,lower_unassisted,geometric_joint_stable
        from normalization import configure_normalizer_updates,equal_state
        require(torch.cuda.is_available() and '5090' in torch.cuda.get_device_name(0),'Refinement runs only on cloud RTX5090')
        configure_torch_backends();torch.set_num_threads(min(4,len(os.sched_getaffinity(0))))
        wp.config.kernel_cache_dir=str(TRAIN/'cache/warp')
        cfg=make_cfg(NUM_ENVS,TRAIN_SEED)
        progress(phase='build_native_environment')
        raw=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode=None);env=RslRlVecEnvWrapper(raw)
        agent=asdict(deepcopy(MicroduckBasketballRlCfg))
        agent.update(seed=TRAIN_SEED,logger='tensorboard',upload_model=False,experiment_name='stage12_refine_4500',
                     run_name='segment-%02d'%a.segment,save_interval=25,max_iterations=MAX_UPDATES,check_for_nan=False)
        # The wrapper below checks outputs before PPO consumes them. Avoid duplicate upstream synchronization.
        agent['algorithm'].update(learning_rate=LEARNING_RATE,schedule='fixed',num_mini_batches=4)
        require(agent['num_steps_per_env']==24,'Original rollout length changed')
        declared_agent=deepcopy(agent)
        runner=MjlabOnPolicyRunner(env,agent,log_dir=str(out/('tensorboard-%02d'%a.session)),device='cuda:0')
        runner.logger.git_status_repos=[]
        original_init=runner.logger.init_logging_writer
        def init_writer():
            try:original_init();state['logging_backend']='tensorboard'
            except ImportError:
                runner.logger.writer=JsonScalarWriter(out/('scalar-metrics-%02d.jsonl'%a.session));runner.logger.logger_type='json'
                state['logging_backend']='scalar_json'
        runner.logger.init_logging_writer=init_writer
        source=input_checkpoint();require(source['completed_updates']==budget['completed_updates'],'Durable checkpoint lags the committed budget; preserve without silently replaying updates')
        require(sha(source['path'])==source['sha256'],'Input checkpoint changed')
        saved=torch.load(source['path'],map_location='cpu',weights_only=False);meta=saved['infos']['stage12_native']
        expected_id=ORIGIN_ID if source['completed_updates']==0 and source['sha256']==MODEL_SHA else manifest['identity']
        require(meta['package_identity']==expected_id and meta['policy_updates']==4500+source['completed_updates'],'Checkpoint lineage differs')
        runner.alg.load(saved,None,strict=True)
        require(equal_state(runner.alg.save(),saved,torch,top_level=True),'Restored model, optimizer or normalization differs')
        old_lr=[float(g['lr']) for g in runner.alg.optimizer.param_groups]
        for group in runner.alg.optimizer.param_groups:group['lr']=LEARNING_RATE
        runner.alg.learning_rate=LEARNING_RATE
        raw.common_step_counter=meta['common_step_counter'];raw._sim_step_counter=meta['sim_step_counter'];env.reset()
        bb=mdp._bb_state(raw);bb.level.zero_();bb.hold.zero_()
        torch.set_rng_state(meta['torch_rng']);torch.cuda.set_rng_state_all(meta['cuda_rng'])
        np.random.set_state(meta['numpy_rng']);random.setstate(meta['python_rng'])
        runner.alg.actor.reset();runner.alg.critic.reset()
        configure_normalizer_updates(runner.alg.actor.obs_normalizer,55,torch)
        configure_normalizer_updates(runner.alg.critic.obs_normalizer,76,torch)
        obs=env.get_observations()
        require(tuple(obs['actor'].shape)==(NUM_ENVS,61) and tuple(obs['critic'].shape)==(NUM_ENVS,85),'Observation schema differs')
        require(abs(raw.step_dt-.02)<1e-12 and head_scale(raw)==1.,'Completed head schedule must remain fully enabled')
        require_finite_group(torch,[*obs.values(),*runner.alg.actor.state_dict().values(),*runner.alg.critic.state_dict().values()],'Nonfinite restored policy or observation')
        write(out/('session-%02d-load.json'%a.session),dict(source=source,optimizer_restored=True,normalization_restored=True,
            state_verified_before_learning_rate_change=True,previous_learning_rates=old_lr,new_learning_rate=LEARNING_RATE,
            old_num_envs=meta['num_envs'],new_num_envs=NUM_ENVS,new_episodes=True,recurrent_state_reset=True,
            physical_trajectory_resume=False,agent=declared_agent,source_task_sha256=sha(HERE/'native_task.py'),
            python=sys.version,device=torch.cuda.get_device_name(0),torch=torch.__version__,original_deadline_utc=budget['deadline_utc']),exclusive=True)
        original_step=env.step
        def step(actions):
            boundary()
            if in_rollout[0]==0:reserve(budget,time.time());write(SLOT/'budget.json',budget)
            require_finite_group(torch,[actions],'Nonfinite policy action')
            result=original_step(actions);observations,rewards,dones,_=result
            require_step_outputs(torch,observations,rewards,dones,raw.termination_manager.get_term('nan_state'))
            in_rollout[0]+=1
            if budget['completed_updates']==0 and in_rollout[0]%6==0:progress(phase='first_native_rollout',rollout_steps=in_rollout[0])
            return result
        env.step=step
        original_update=runner.alg.update
        def update():
            require(in_rollout[0]==24,'Incomplete native PPO rollout');update_in_flight[0]=True
            loss=original_update();require(all(math.isfinite(float(v)) for v in loss.values()),'Nonfinite PPO loss')
            require_finite_group(torch,(p for model in (runner.alg.actor,runner.alg.critic) for p in model.parameters()),'Nonfinite updated parameters')
            commit(budget);in_rollout[0]=0;update_in_flight[0]=False;write(SLOT/'budget.json',budget)
            return loss
        runner.alg.update=update;runner.save=save_model
        original_log=runner.logger.log
        def log(**kw):
            keys=[];values=[];episodes={}
            for item in runner.logger.ep_extras:
                for key,value in item.items():
                    if any(s in key for s in ('native_','top_ball_center','Episode_Termination')):keys.append(key);values.append(value)
            for key,value in zip(keys,host_scalar_means(torch,values)):
                if math.isfinite(value):episodes.setdefault(key,[]).append(value)
            original_log(**kw)
            current=host_scalar_means(torch,[(mdp._bb_state(raw).hold<=1e-6).float(),lower_unassisted(raw).float(),geometric_joint_stable(raw).float()])
            progress(status='TRAINING',phase='refine_ppo',rollout_steps=24,completed_updates=budget['completed_updates'],
                attempted_updates=budget['attempted_updates'],policy_updates=4500+budget['completed_updates'],
                progress_percent=100*budget['attempted_updates']/MAX_UPDATES,segment_target=target,learning_rate=runner.alg.learning_rate,
                lower_unassisted_current_fraction=current[1],joint_geometric_current_fraction=current[2],unassisted_env_fraction=current[0],
                collection_seconds=kw['collect_time'],learning_seconds=kw['learn_time'],head_reward_scale=head_scale(raw),
                episode_metrics={k:sum(v)/len(v) for k,v in episodes.items()},last_losses={k:float(v) for k,v in kw['loss_dict'].items()})
            if budget['completed_updates']%25==0 or time.monotonic()-last_save_time[0]>=120:save_model()
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(state,ensure_ascii=False,allow_nan=False)+'\n')
            print('REFINE_TRAINING_PROGRESS='+json.dumps(state,ensure_ascii=False),flush=True)
        runner.logger.log=log;runner.current_learning_iteration=4500+budget['completed_updates']
        remaining=min(target-budget['completed_updates'],MAX_UPDATES-budget['attempted_updates'])
        require(remaining>0,'No unspent updates in this segment');boundary()
        runner.learn(remaining,init_at_random_ep_len=False);save_model()
        progress(status='TRAINING_SEGMENT_COMPLETE',phase='checkpoint_saved',latest_checkpoint=input_checkpoint())
    except TrainingStop as exc:
        if runner is not None and not update_in_flight[0]:save_model()
        progress(status='TRAINING_STOPPED_PRESERVE',reason=str(exc))
    except BaseException as exc:
        progress(status='TRAINING_ERROR_PRESERVE',error=repr(exc),traceback=traceback.format_exc()[-12000:])
    finally:
        if env is not None:
            try:env.close()
            except BaseException as exc:state.update(status='TRAINING_ERROR_PRESERVE',close_error=repr(exc))
        elif raw is not None:
            try:raw.close()
            except BaseException:pass
        if runner is not None:
            try:runner.logger.stop_logging_writer()
            except BaseException:pass
            try:runner.logger.writer.close()
            except (AttributeError,ValueError):pass
        progress(finished_utc=time.time(),completed_updates=budget['completed_updates'],attempted_updates=budget['attempted_updates'])
        write(out/('session-%02d-train-result.json'%a.session),state,exclusive=True)
    return 0 if state['status']=='TRAINING_SEGMENT_COMPLETE' else 3 if state['status']=='TRAINING_STOPPED_PRESERVE' else 2

if __name__=='__main__':raise SystemExit(main())
