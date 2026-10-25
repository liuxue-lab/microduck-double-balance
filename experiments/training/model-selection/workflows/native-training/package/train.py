"""Normal mjlab/rsl-rl training with small monitoring and durable checkpoints.

One native 64-environment/5-update smoke, then 2048-environment continuation.
No fixed-state GPU probe, adaptive RK4, exact replay or standalone evaluation.
The upstream OnPolicyRunner.learn loop owns every rollout and PPO update.
"""
import argparse
from collections import deque
from copy import deepcopy
from dataclasses import asdict
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import signal
import sys
import time
import traceback
from common import SLOT, SOURCE, read, write, sha, require, reserve, commit

class TrainingStop(Exception):
    pass

class JsonScalarWriter:
    """Dependency-free fallback; logging must not require another download."""
    def __init__(self,path):
        self.file=Path(path).open('a',buffering=1)
    def add_scalar(self,tag,value,global_step=None,**kwargs):
        value=float(value)
        self.file.write(json.dumps(dict(tag=tag,value=value if math.isfinite(value) else None,
            step=global_step),ensure_ascii=False,allow_nan=False)+'\n')
    def flush(self):self.file.flush()
    def close(self):self.file.close()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--session', type=int, required=True); p.add_argument('--phase', choices=['smoke','formal'], required=True); a=p.parse_args()
    package=Path(__file__).resolve().parent
    cfg_doc=read(SLOT/'config.json'); budget=read(SLOT/'budget.json')
    num_envs=cfg_doc['smoke_envs'] if a.phase=='smoke' else cfg_doc['num_envs']
    out=SLOT/'run'; out.mkdir(exist_ok=True); checkpoints=out/'checkpoints'; checkpoints.mkdir(exist_ok=True)
    sys.path.insert(0, str(SOURCE/'src'))
    state=dict(status='INITIALIZING', phase='load_runtime', session=a.session, completed_updates=budget['completed_updates'],
               max_updates=budget['max_updates'], remaining_validation_runs=0, historical_reserved_PPO=4, campaign_phase=a.phase, num_envs=num_envs)
    write(out/'progress.json', state)
    raw=env=runner=None; model_updates=0; update_in_flight=False; stop_signal=[False]
    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, lambda *_:stop_signal.__setitem__(0, True))
    try:
        expected={'mjlab':'1.3.0','mujoco':'3.10.0','mujoco-warp':'3.8.1','warp-lang':'1.12.0','rsl-rl-lib':'5.0.1'}
        require({k:importlib.metadata.version(k) for k in expected}==expected, 'Existing runtime versions differ; no installer is run')
        import numpy as np
        import torch
        import warp as wp
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
        from mjlab.utils.torch import configure_torch_backends
        from mjlab_microduck.tasks.microduck_basketball_env_cfg import MicroduckBasketballRlCfg
        from mjlab_microduck.tasks import mdp
        from native_task import make_cfg, head_scale, lower_unassisted, geometric_joint_stable
        require(torch.cuda.is_available() and '5090' in torch.cuda.get_device_name(0), 'Formal training requires the cloud RTX5090')
        configure_torch_backends(); torch.set_num_threads(min(4,len(os.sched_getaffinity(0))))
        wp.config.kernel_cache_dir=str(SLOT/'cache/warp')
        cfg=make_cfg(num_envs, cfg_doc['seed'])
        state['phase']='build_native_environment';write(out/'progress.json',state)
        raw=ManagerBasedRlEnv(cfg, device='cuda:0', render_mode=None)
        env=RslRlVecEnvWrapper(raw)
        agent=asdict(deepcopy(MicroduckBasketballRlCfg))
        agent.update(seed=cfg_doc['seed'], logger='tensorboard', upload_model=False,
                     experiment_name='stage12_native_head_balance', run_name='native_training_v2',
                     save_interval=50, max_iterations=cfg_doc['max_updates'], check_for_nan=True)
        agent['algorithm']['learning_rate']=5e-5
        agent['algorithm']['schedule']='fixed'  # Keep transfer LR small; inherited adaptive ceiling is 1e-2.
        require(agent['num_steps_per_env']==24,'Unexpected rollout length')
        agent['algorithm']['num_mini_batches']=4
        initial_cfg=deepcopy(agent)
        runner=MjlabOnPolicyRunner(env, agent, log_dir=str(out/'tensorboard'), device='cuda:0')
        runner.logger.git_status_repos=[]  # No Git network or checkout mutation.
        original_init_writer=runner.logger.init_logging_writer
        def init_writer():
            try:
                original_init_writer()
                state['logging_backend']='tensorboard'
            except ImportError:
                runner.logger.writer=JsonScalarWriter(out/'scalar-metrics.jsonl')
                runner.logger.logger_type='json'
                state['logging_backend']='scalar_json; TensorBoard dependency unavailable'
                print('TensorBoard unavailable; recording scalar JSON without installing dependencies.',flush=True)
        runner.logger.init_logging_writer=init_writer
        obs=env.get_observations()
        require(tuple(obs['actor'].shape)==(num_envs,61) and tuple(obs['critic'].shape)==(num_envs,85), 'Training observation schema mismatch')
        state['phase']='load_training_checkpoint';write(out/'progress.json',state)
        if a.phase=='smoke':
            init=read(SLOT/'initialization.json'); checkpoint=Path(init['path'])
            require(sha(checkpoint)==init['sha256'], 'Initialization checkpoint changed')
            source=torch.load(checkpoint, map_location='cpu', weights_only=False)
            require(init['kind']=='stage08_selected_E','Only the retained Stage08 primary policy may initialize v2')
            from transfer import migrate_stage08
            migrated=migrate_stage08(source, torch)
            actor,critic=migrated['actor_state_dict'],migrated['critic_state_dict']
            runner.alg.load(migrated, {'actor':True,'critic':True,'optimizer':False,'iteration':False}, strict=True)
            runner.alg.actor.reset(); runner.alg.critic.reset()
            write(out/'transfer.json',dict(source=init, actor_columns_neutralized=[55,61],critic_columns_neutralized=[76,85],
                optimizer='new Adam at fixed 5e-5; historical optimizer files untouched',normalizer_prior_count=10000,
                normalization='unchanged feature statistics frozen; new head features adapt',
                recurrent_state='fresh episode', previous_learning_counters_restored=False,
                direct_trajectory_resume=False, inherited_performance_claimed=False,
                source_policy='Stage08 E selected primary; earlier task performance is not claimed on the new head task',
                dedicated_lower_only_updates=0, initial_lower_assistance=0.), exclusive=True)
        else:
            last=read(out/'latest.json'); checkpoint=Path(last['path'])
            require(sha(checkpoint)==last['sha256'], 'Saved training checkpoint differs')
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            meta=saved['infos']['stage12_native']
            require(meta['package_identity']==cfg_doc['package_identity'], 'Resume task identity differs')
            if a.session==1:
                require(read(out/'smoke-result.json')['status']=='SMOKE_PASSED' and meta['policy_updates']==5 and meta['num_envs']==64, 'Missing five-update smoke handoff')
            runner.alg.load(saved,None,strict=True)
            runner.alg.learning_rate=float(runner.alg.optimizer.param_groups[0]['lr'])
            model_updates=meta['policy_updates']
            raw.common_step_counter=meta['common_step_counter']; raw._sim_step_counter=meta['sim_step_counter']
            env.reset()
            # No per-environment physical state is restored across 64 -> 2048.
            bb=mdp._bb_state(raw); bb.level.zero_();bb.hold.zero_()
            torch.set_rng_state(meta['torch_rng']); torch.cuda.set_rng_state_all(meta['cuda_rng'])
            np.random.set_state(meta['numpy_rng']); random.setstate(meta['python_rng'])
            runner.alg.actor.reset(); runner.alg.critic.reset()
            write(out/('formal-load.json' if a.session==1 else 'formal-resume-%02d.json'%a.session),dict(
                source=last,optimizer_restored=True,normalization_restored=True,policy_updates=model_updates,
                old_num_envs=meta['num_envs'],new_num_envs=num_envs,new_episodes=True,
                original_deadline=budget['deadline_utc']),exclusive=True)
        from transfer import configure_normalizer_updates
        configure_normalizer_updates(runner.alg.actor.obs_normalizer,55,torch)
        configure_normalizer_updates(runner.alg.critic.obs_normalizer,76,torch)
        write(out/('session-%02d-%s-configuration.json'%(a.session,a.phase)),dict(agent=initial_cfg,
            physics=dict(integrator=cfg.sim.mujoco.integrator, timestep=raw.physics_dt,control_dt=raw.step_dt,
                         iterations=cfg.sim.mujoco.iterations,ls_iterations=cfg.sim.mujoco.ls_iterations,
                         tolerance=cfg.sim.mujoco.tolerance,native_mesh_collision=True),
            standalone_validation_runs=1, smoke_settings=dict(num_envs=64,updates=5), old_strict_audit_passed=False,
            task='lower basketball plus real head shell and upper free sphere, no tray',
            curriculum='No lower-only phase; head task enabled at update zero and ramps through update 1000; lower assistance zero',
            num_envs=num_envs,device=torch.cuda.get_device_name(0)),exclusive=True)
        last_save=[0.]; batches=deque(maxlen=100); best_score=None
        if (out/'best-training-metric.json').exists():
            best_score=read(out/'best-training-metric.json')['score']
        def save_model(_path=None, infos=None):
            if _path is not None:
                return  # Our timed/50-update schedule owns checkpoint cadence.
            target=checkpoints/('update_%06d.pt'%budget['completed_updates'])
            if target.exists():
                return
            bb=mdp._bb_state(raw)
            meta=dict(policy_updates=model_updates,campaign_completed_updates=budget['completed_updates'],
                campaign_attempted_updates=budget['attempted_updates'],num_envs=raw.num_envs,
                package_identity=cfg_doc['package_identity'],common_step_counter=raw.common_step_counter,
                sim_step_counter=raw._sim_step_counter,lower_assistance=0.,phase=a.phase,
                torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),
                python_rng=random.getstate(), physical_trajectory_resume=False)
            value=runner.alg.save();value.update(iter=model_updates-1,infos={'env_state':{'common_step_counter':raw.common_step_counter},'stage12_native':meta})
            tmp=target.with_suffix('.tmp.pt');torch.save(value,tmp)
            with tmp.open('rb') as f:os.fsync(f.fileno())
            os.replace(tmp,target)
            receipt=dict(path=str(target),sha256=sha(target),bytes=target.stat().st_size,
                         policy_updates=model_updates,completed_updates=budget['completed_updates'])
            write(target.with_suffix('.json'),receipt,exclusive=True);write(out/'latest.json',receipt);last_save[0]=time.monotonic()
        runner.save=save_model
        if a.phase=='smoke':save_model()
        def boundary():
            if stop_signal[0] or (SLOT/'STOP.json').exists() or time.time()>=budget['deadline_utc']:
                raise TrainingStop('Requested stop or campaign time limit')
        original_step=env.step; in_rollout=[0]; nan_since=[0]
        def step(actions):
            boundary()
            if in_rollout[0]==0:
                reserve(budget,time.time());write(SLOT/'budget.json',budget)
            require(bool(torch.isfinite(actions).all()), 'Nonfinite policy action')
            result=original_step(actions)
            nan_count=int(raw.termination_manager.get_term('nan_state').sum().item())
            nan_since[0]+=nan_count
            require(nan_count==0,'Nonfinite simulator state detected; preserve, no automatic retry')
            observations,rewards,dones,_=result
            require(all(bool(torch.isfinite(v).all()) for v in observations.values()) and bool(torch.isfinite(rewards).all()), 'Nonfinite observations or rewards')
            in_rollout[0]+=1
            if (a.phase=='smoke' or model_updates==5) and in_rollout[0]%6==0:
                state.update(phase=a.phase+'_rollout',rollout_steps=in_rollout[0])
                write(out/'progress.json',state)
            return result
        env.step=step
        original_update=runner.alg.update
        def update():
            nonlocal model_updates,update_in_flight
            require(in_rollout[0]==24, 'Incomplete PPO batch')
            update_in_flight=True
            loss=original_update()
            require(all(math.isfinite(float(v)) for v in loss.values()), 'Nonfinite PPO loss')
            require(all(bool(torch.isfinite(p).all()) for m in (runner.alg.actor,runner.alg.critic) for p in m.parameters()), 'Nonfinite learned weights')
            commit(budget);model_updates+=1;in_rollout[0]=0;update_in_flight=False
            write(SLOT/'budget.json',budget)
            return loss
        runner.alg.update=update
        original_log=runner.logger.log
        def log(**kw):
            nonlocal best_score
            episodes={}
            for item in runner.logger.ep_extras:
                for key,v in item.items():
                    if any(s in key for s in ('native_','top_ball_center','double_balance','Episode_Termination')):
                        f=float(torch.as_tensor(v).float().mean().item())
                        if math.isfinite(f):episodes.setdefault(key,[]).append(f)
            original_log(**kw)
            summary={k:sum(v)/len(v) for k,v in episodes.items()}
            batches.append(summary)
            state.update(status='TRAINING',phase=a.phase+'_ppo',rollout_steps=24,completed_updates=budget['completed_updates'],attempted_updates=budget['attempted_updates'],
                policy_updates=model_updates,progress_percent=100*budget['attempted_updates']/budget['max_updates'],
                elapsed_seconds=time.time()-budget['started_utc'],head_reward_scale=head_scale(raw),
                unassisted_env_fraction=float((mdp._bb_state(raw).hold<=1e-6).float().mean()),
                lower_unassisted_current_fraction=float(lower_unassisted(raw).float().mean()),
                joint_geometric_current_fraction=float(geometric_joint_stable(raw).float().mean()),
                mean_episode_length_steps=sum(runner.logger.lenbuffer)/len(runner.logger.lenbuffer) if runner.logger.lenbuffer else None,
                mean_reward=sum(runner.logger.rewbuffer)/len(runner.logger.rewbuffer) if runner.logger.rewbuffer else None,
                nan_resets_in_last_rollout=nan_since[0],episode_metrics=summary,learning_rate=runner.alg.learning_rate,
                last_losses={k:float(v) for k,v in kw['loss_dict'].items()},
                prior_validation_reservations_preserved=4,validation_runs_added=1)
            nan_since[0]=0
            if model_updates==1 or model_updates%50==0 or time.monotonic()-last_save[0]>=120:
                save_model()
            if model_updates%50==0 and head_scale(raw)>=1:
                names=('native_joint_geometric_success_5s','native_joint_geometric_stable_fraction','native_lower_unassisted_fraction')
                keys=[next((k for k in summary if k.endswith(name)),None) for name in names]
                values=[[x[k] for x in batches if k and k in x] for k in keys]
                score=[sum(v)/len(v) for v in values] if all(values) else None
                if score is not None and (best_score is None or score>best_score):
                    save_model();best_score=score
                    write(out/'best-training-metric.json',dict(score=score,metrics=list(names),comparison='lexicographic, rolling 100 logged batches',checkpoint=read(out/'latest.json'),
                        held_out_evaluation=False,strict_physics_acceptance=False))
            if (out/'latest.json').exists():state['latest_checkpoint']=read(out/'latest.json')
            write(out/'progress.json',state)
            with (out/'metrics.jsonl').open('a') as f:f.write(json.dumps(state,ensure_ascii=False,allow_nan=False)+'\n')
            print('NATIVE_TRAINING_PROGRESS='+json.dumps(state,ensure_ascii=False,allow_nan=False),flush=True)
        runner.logger.log=log
        runner.current_learning_iteration=model_updates
        remaining=cfg_doc['smoke_updates'] if a.phase=='smoke' else budget['max_updates']-budget['attempted_updates']
        boundary();runner.learn(remaining,init_at_random_ep_len=False)
        save_model()
        if a.phase=='smoke':
            require(model_updates==5 and budget['completed_updates']==5,'Smoke update count differs')
            latest=read(out/'latest.json');require(sha(latest['path'])==latest['sha256'],'Smoke checkpoint hash differs')
            saved=torch.load(latest['path'],map_location='cpu',weights_only=False)
            runner.alg.load(saved,None,strict=True)
            from transfer import equal_state
            require(equal_state(runner.alg.save(),saved,torch,top_level=True),'Checkpoint reloaded state differs')
            write(out/'smoke-result.json',dict(status='SMOKE_PASSED',completed_updates=5,
                num_envs=64,steps_per_env=24,transitions=7680,reload_verified=True,
                checkpoint=latest,physical_success_required=False,finished_utc=time.time()),exclusive=True)
            state['status']='SMOKE_PASSED'
        else:state['status']='TRAINING_BUDGET_COMPLETE'
    except TrainingStop as exc:
        if runner is not None and not update_in_flight:
            save_model()
        state.update(status='TRAINING_STOPPED_CHECKPOINT_PRESERVED',reason=str(exc))
    except BaseException as exc:
        state.update(status='TRAINING_ERROR_PRESERVED',error=repr(exc),traceback=traceback.format_exc()[-14000:])
    finally:
        if env is not None:
            try:env.close()
            except BaseException as exc:
                state['close_error']=repr(exc)
                state['status']='TRAINING_ERROR_PRESERVED'
        if runner is not None:
            try:runner.logger.stop_logging_writer()
            except BaseException:pass
            try:runner.logger.writer.close()
            except (AttributeError,ValueError):pass
        state.update(finished_utc=time.time(),validation_runs_added=1,old_strict_audit_passed=False,
                     trained_successfully_not_yet_claimed=True,prior_validation_reservations_preserved=4,
                     completed_updates=budget['completed_updates'],attempted_updates=budget['attempted_updates'])
        if (out/'latest.json').exists():state['latest_checkpoint']=read(out/'latest.json')
        write(out/'progress.json',state);write(out/('session-%02d-%s-result.json'%(a.session,a.phase)),state,exclusive=True)
        print('NATIVE_TRAINING_RESULT='+json.dumps(state,ensure_ascii=False,allow_nan=False),flush=True)
    return 0 if state['status'] in ('SMOKE_PASSED','TRAINING_BUDGET_COMPLETE','TRAINING_STOPPED_CHECKPOINT_PRESERVED') else 2

if __name__=='__main__':
    raise SystemExit(main())
