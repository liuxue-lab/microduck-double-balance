"""One reserved PPO phase, warm-starting the selected Stage12 speed500 model."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import math
import os
import random
import signal
import sys
import time
import traceback

from common import *
from budget import reserve_update, commit_update
from runtime_helpers import JsonScalarWriter, require_finite_group, require_step_outputs, host_scalar_means
from fingerprint import fingerprint


class TrainingStop(Exception):
    pass


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=tuple(TARGETS),required=True)
    parser.add_argument('--session',type=int,required=True)
    args=parser.parse_args()
    budget=read(SLOT/'budget.json');package_id=verify_package();out=SLOT/'run'
    count=64 if args.phase=='smoke' else NUM_ENVS
    seed=SMOKE_SEED if args.phase=='smoke' else TRAIN_SEED
    target=TARGETS[args.phase]
    state=dict(status='INITIALIZING',phase=args.phase,detail='load_runtime',num_envs=count,
        completed_updates=budget['completed_updates'],attempted_updates=budget['attempted_updates'],
        max_updates=MAX_UPDATES,phase_target=target,policy_updates=5000+budget['completed_updates'])
    raw=env=runner=monitor=None
    restored=False;rollout_steps=0;update_in_flight=False;stop=False;last_save_time=time.monotonic()
    session_dir=out/('session-%03d'%args.session)
    session_dir.mkdir(exist_ok=False)

    def progress(**kw):
        state.update(kw,elapsed_seconds=time.time()-budget['started_utc'],
            original_deadline_utc=budget['deadline_utc'],
            completed_updates=budget['completed_updates'],attempted_updates=budget['attempted_updates'],
            policy_updates=5000+budget['completed_updates'],
            progress_percent=100*budget['attempted_updates']/MAX_UPDATES)
        write(SLOT/'progress.json',state)

    def handle_signal(*_):
        nonlocal stop
        stop=True

    def boundary():
        if stop or (SLOT/'STOP.json').exists() or time.time()>=budget['deadline_utc']:
            raise TrainingStop('Stop request or approved compute deadline')

    def save_model(_path=None,infos=None):
        nonlocal last_save_time
        if _path is not None:return  # Ignore upstream periodic naming; own atomic checkpoints only.
        require(restored and runner is not None and budget['pending'] is None and not update_in_flight,
                'Only fully committed optimizer state may be checkpointed')
        completed=budget['completed_updates'];path=out/'checkpoints'/('headroom_%06d.pt'%completed)
        if path.exists():
            receipt=read(path.with_suffix('.json'))
            require(receipt['completed_updates']==completed and sha(path)==receipt['sha256'],
                    'Existing checkpoint differs; preserve it')
            write(out/'latest.json',receipt);last_save_time=time.monotonic();return
        meta=dict(package_identity=package_id,origin_package_identity=ORIGIN_ID,
            origin_checkpoint_sha256=MODEL_SHA,policy_updates=5000+completed,
            headroom_updates=completed,campaign_attempted_updates=budget['attempted_updates'],
            num_envs=count,common_step_counter=raw.common_step_counter,sim_step_counter=raw._sim_step_counter,
            lower_assistance=0.,learning_rate=LEARNING_RATE,physical_trajectory_resume=False,
            reward_spec=read(HERE/'reward-spec.json'),
            torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
            numpy_rng=np.random.get_state(),python_rng=random.getstate())
        saved=runner.alg.save()
        saved.update(iter=5000+completed-1,infos={'stage12_native':meta,'stage12_headroom':meta,
            'env_state':{'common_step_counter':raw.common_step_counter}})
        tmp=path.with_suffix('.tmp.pt');require(not tmp.exists(),'Earlier partial checkpoint exists; preserve')
        torch.save(saved,tmp)
        with tmp.open('rb') as f:os.fsync(f.fileno())
        os.replace(tmp,path)
        receipt=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size,
            completed_updates=completed,policy_updates=5000+completed,attempted_updates=budget['attempted_updates'])
        write(path.with_suffix('.json'),receipt,exclusive=True);write(out/'latest.json',receipt)
        last_save_time=time.monotonic()

    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,handle_signal)
    try:
        progress();boundary();versions=verify_runtime()
        require(budget['pending'] is None,'Earlier attempt remains pending')
        sys.path.insert(0,str(SOURCE/'src'))
        import numpy as np
        import torch
        import warp as wp
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
        from mjlab.utils.torch import configure_torch_backends
        from mjlab_microduck.tasks.microduck_basketball_env_cfg import MicroduckBasketballRlCfg
        from mjlab_microduck.tasks import mdp
        from native_task import head_scale,lower_unassisted,geometric_joint_stable
        from headroom_task import make_cfg,head_roll_position_cost,head_roll_outward_cost
        from headroom_costs import POSITION_WEIGHT,OUTWARD_WEIGHT
        from normalization import configure_normalizer_updates,equal_state
        from train_motor_monitor import TrainingMotorMonitor
        require(torch.cuda.is_available() and '5090' in torch.cuda.get_device_name(0),
                'Formal PPO uses the current cloud RTX5090 only')
        configure_torch_backends();torch.set_num_threads(min(4,len(os.sched_getaffinity(0))))
        # Reuse installed kernels; do not run previous experiments or rebuild dependencies.
        wp.config.kernel_cache_dir=str(TRAIN/'cache/warp')
        cfg=make_cfg(count,seed)
        require(cfg.episode_length_s==60. and not cfg.curriculum and 'push_robot' not in cfg.events,
                'Static 60-second headroom configuration differs')
        progress(detail='build_native_environment')
        raw=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode=None)
        env=RslRlVecEnvWrapper(raw)
        agent=asdict(deepcopy(MicroduckBasketballRlCfg))
        agent.update(seed=seed,logger='tensorboard',upload_model=False,experiment_name='stage12_headroom',
            run_name=args.phase,save_interval=25,max_iterations=MAX_UPDATES,check_for_nan=False)
        agent['algorithm'].update(learning_rate=LEARNING_RATE,schedule='fixed',num_mini_batches=4)
        require(agent['num_steps_per_env']==24,'Original rollout length changed')
        declared=deepcopy(agent)
        runner=MjlabOnPolicyRunner(env,agent,log_dir=str(session_dir/'tensorboard'),device='cuda:0')
        runner.logger.git_status_repos=[]
        original_writer=runner.logger.init_logging_writer
        def init_writer():
            try:original_writer()
            except ImportError:
                runner.logger.writer=JsonScalarWriter(session_dir/'scalars.jsonl')
                runner.logger.logger_type='json'
        runner.logger.init_logging_writer=init_writer
        source=input_checkpoint()
        require(source['completed_updates']==budget['completed_updates'],'Checkpoint lags committed ledger; no replay')
        require(sha(source['path'])==source['sha256'],'Input checkpoint changed')
        saved=torch.load(source['path'],map_location='cpu',weights_only=False)
        meta=saved['infos']['stage12_native']
        expected_id=ORIGIN_ID if source['completed_updates']==0 else package_id
        require(meta['package_identity']==expected_id and meta['policy_updates']==5000+source['completed_updates'],
                'Warm-start lineage differs')
        runner.alg.load(saved,None,strict=True)
        require(equal_state(runner.alg.save(),saved,torch,top_level=True),
                'Restored actor/critic/optimizer/normalizer differs')
        previous_lr=[float(group['lr']) for group in runner.alg.optimizer.param_groups]
        for group in runner.alg.optimizer.param_groups:group['lr']=LEARNING_RATE
        runner.alg.learning_rate=LEARNING_RATE
        # New episodes with a recorded seed; keep original maturity, then restore PPO RNG.
        raw.common_step_counter=meta['common_step_counter'];raw._sim_step_counter=meta['sim_step_counter']
        raw.reset(seed=seed)
        bb=mdp._bb_state(raw);bb.level.zero_();bb.hold.zero_()
        torch.set_rng_state(meta['torch_rng']);torch.cuda.set_rng_state_all(meta['cuda_rng'])
        np.random.set_state(meta['numpy_rng']);random.setstate(meta['python_rng'])
        runner.alg.actor.reset();runner.alg.critic.reset()
        configure_normalizer_updates(runner.alg.actor.obs_normalizer,55,torch)
        configure_normalizer_updates(runner.alg.critic.obs_normalizer,76,torch)
        obs=env.get_observations()
        require(tuple(obs['actor'].shape)==(count,61) and tuple(obs['critic'].shape)==(count,85),
                'Observation schema differs')
        require(raw.max_episode_length==3000 and abs(raw.step_dt-.02)<1e-12
                and abs(raw.physics_dt-.002)<1e-12 and raw.cfg.decimation==10 and head_scale(raw)==1.,
                'Time contract or mature head schedule changed')
        require_finite_group(torch,[*obs.values(),*runner.alg.actor.state_dict().values(),
                            *runner.alg.critic.state_dict().values()],'Nonfinite restored state')
        # Resolves the actual model's named joint and verifies nominal and soft bounds.
        require_finite_group(torch,[head_roll_position_cost(raw),head_roll_outward_cost(raw)],
                             'Headroom reward construction failed')
        monitor=TrainingMotorMonitor(raw,torch,session_dir)
        restored=True
        write(session_dir/'load.json',dict(source=source,optimizer_restored=True,normalization_restored=True,
            state_verified_before_learning_rate_change=True,previous_learning_rates=previous_lr,
            new_learning_rate=LEARNING_RATE,previous_num_envs=meta['num_envs'],num_envs=count,
            new_episode_seed=seed,rng_order='seeded environment reset then checkpoint Torch/CUDA/NumPy/Python RNG',
            recurrent_state_reset=True,physical_trajectory_resume=False,agent=declared,
            runtime=versions,device=torch.cuda.get_device_name(0),original_deadline_utc=budget['deadline_utc']),exclusive=True)
        original_step=env.step
        def step(actions):
            nonlocal rollout_steps
            boundary()
            if rollout_steps==0:
                reserve_update(budget,args.phase,time.time());write(SLOT/'budget.json',budget)
            require_finite_group(torch,[actions],'Nonfinite action')
            before=monitor.steps;monitor.active=True
            try:result=original_step(actions)
            finally:monitor.active=False
            require(monitor.steps==before+10,'Training physics substep count differs')
            observations,rewards,dones,_=result
            require_step_outputs(torch,observations,rewards,dones,raw.termination_manager.get_term('nan_state'))
            pc=POSITION_WEIGHT*head_roll_position_cost(raw)
            oc=OUTWARD_WEIGHT*head_roll_outward_cost(raw)
            require(bool(torch.stack([torch.isfinite(pc).all(),torch.isfinite(oc).all(),
                                      (pc<=0).all(),(oc<=0).all()]).all()),
                    'Headroom reward sign/nonfinite check failed')
            rollout_steps+=1
            if args.phase=='smoke' and rollout_steps%6==0:
                progress(detail='new_config_smoke_rollout',rollout_steps=rollout_steps)
            return result
        env.step=step
        original_update=runner.alg.update
        def update():
            nonlocal rollout_steps,update_in_flight
            boundary();require(rollout_steps==24,'Incomplete PPO rollout')
            update_in_flight=True
            loss=original_update()
            require(all(math.isfinite(float(v)) for v in loss.values()),'Nonfinite PPO loss')
            require_finite_group(torch,(p for model in (runner.alg.actor,runner.alg.critic)
                                       for p in model.parameters()),'Nonfinite updated policy')
            commit_update(budget,args.phase);write(SLOT/'budget.json',budget)
            rollout_steps=0;update_in_flight=False
            return loss
        runner.alg.update=update;runner.save=save_model
        original_log=runner.logger.log
        def log(**kw):
            keys=[];values=[];episodes={}
            for item in runner.logger.ep_extras:
                for key,value in item.items():
                    if any(s in key for s in ('native_','top_ball_center','Episode_Termination','Episode_Reward/')):
                        keys.append(key);values.append(value)
            for key,value in zip(keys,host_scalar_means(torch,values)):
                require(math.isfinite(value),'Nonfinite episode metric: '+key)
                if 'head_roll_margin' in key or 'head_roll_outward' in key:
                    require(value<=1e-7,'Headroom episode cost became a positive reward')
                episodes.setdefault(key,[]).append(value)
            original_log(**kw)
            current=host_scalar_means(torch,[lower_unassisted(raw).float(),geometric_joint_stable(raw).float(),
                head_roll_position_cost(raw),head_roll_outward_cost(raw)])
            progress(status='TRAINING',detail='PPO',rollout_steps=24,
                collection_seconds=kw['collect_time'],learning_seconds=kw['learn_time'],
                lower_unassisted_current_fraction=current[0],geometric_current_fraction=current[1],
                head_roll_margin_mean=current[2],head_roll_outward_mean=current[3],
                episode_metrics={k:sum(v)/len(v) for k,v in episodes.items()},
                last_losses={k:float(v) for k,v in kw['loss_dict'].items()},
                checked_training_physics_steps=monitor.steps)
            if budget['completed_updates']%25==0 or time.monotonic()-last_save_time>=120:
                save_model()
            with (out/'metrics.jsonl').open('a') as f:
                f.write(json.dumps(state,ensure_ascii=False,allow_nan=False)+'\n')
            print('HEADROOM_TRAINING_PROGRESS='+json.dumps(state,ensure_ascii=False),flush=True)
        runner.logger.log=log
        runner.current_learning_iteration=5000+budget['completed_updates']
        remaining=target-budget['completed_updates'];require(remaining>0,'No remaining updates in this phase')
        boundary();runner.learn(remaining,init_at_random_ep_len=False);save_model()
        require(budget['completed_updates']==target and budget['pending'] is None,'Training phase not complete')
        if args.phase=='smoke':
            boundary();progress(detail='smoke_onnx_export')
            import onnx
            from mjlab_microduck.onnx_policy_contract import bake_action_clip
            before=fingerprint(runner.alg.actor.state_dict())
            runner.export_policy_to_onnx(str(session_dir),'smoke.onnx')
            onnx_path=session_dir/'smoke.onnx'
            bake_action_clip(str(onnx_path),declared['clip_actions'])
            model=onnx.load(str(onnx_path));onnx.checker.check_model(model)
            inputs=[[d.dim_value for d in v.type.tensor_type.shape.dim] for v in model.graph.input]
            outputs=[[d.dim_value for d in v.type.tensor_type.shape.dim] for v in model.graph.output]
            require(any(x and x[-1]==61 for x in inputs) and any(x and x[-1]==14 for x in outputs),
                    'Smoke ONNX observation/action schema differs')
            require(fingerprint(runner.alg.actor.state_dict())==before,'Export changed policy or normalization')
            state['smoke_export']=dict(passed=True,path=str(onnx_path),sha256=sha(onnx_path),
                inputs=inputs,outputs=outputs,exporter='original runner.export_policy_to_onnx with baked normalizer',
                deployment_acceptance=False)
        progress(status='TRAINING_PHASE_COMPLETE',detail='checkpoint_saved',latest_checkpoint=input_checkpoint())
    except TrainingStop as exc:
        if restored and runner is not None and budget['pending'] is None and not update_in_flight:
            save_model()
        progress(status='TRAINING_STOPPED_PRESERVE',reason=str(exc))
    except BaseException as exc:
        progress(status='TRAINING_ERROR_PRESERVE',error=repr(exc),traceback=traceback.format_exc()[-12000:])
    finally:
        if monitor is not None:
            try:monitor.close()
            except BaseException as exc:state.update(status='TRAINING_ERROR_PRESERVE',monitor_close_error=repr(exc))
        if env is not None:
            try:env.close()
            except BaseException as exc:state.update(status='TRAINING_ERROR_PRESERVE',environment_close_error=repr(exc))
        elif raw is not None:
            try:raw.close()
            except BaseException:pass
        if runner is not None:
            try:runner.logger.stop_logging_writer()
            except BaseException:pass
            try:runner.logger.writer.close()
            except (AttributeError,ValueError):pass
        progress(finished_utc=time.time())
        write(session_dir/'result.json',state,exclusive=True)
    return 0 if state['status']=='TRAINING_PHASE_COMPLETE' else 3 if state['status']=='TRAINING_STOPPED_PRESERVE' else 2


if __name__=='__main__':
    raise SystemExit(main())
