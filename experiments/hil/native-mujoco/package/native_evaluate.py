"""One frozen-policy episode using the existing task and native CPU backend."""
from copy import deepcopy
from dataclasses import asdict
import signal
import time
import traceback
from types import SimpleNamespace
from common import *
from fingerprint import fingerprint
from native_criteria import accept_case
from evaluate import EvaluationInterrupted, validate_checkpoint
from native_backend import native_environment_factory


def load_runtime():
    verify_runtime()
    sys.path.insert(0,str(SOURCE/'src'))
    import torch
    from tensordict import TensorDict
    from mjlab.envs import ManagerBasedRlEnv as OriginalEnvironment
    from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
    from mjlab_microduck.tasks.microduck_basketball_env_cfg import MicroduckBasketballRlCfg
    from native_task import make_cfg,head_scale,mdp
    from native_session import EvaluationSession
    torch.set_num_threads(1)
    def ManagerBasedRlEnv(*args, **kwargs):
        require(kwargs.get('device')=='cpu','Native environment must stay on CPU')
        with native_environment_factory():
            return OriginalEnvironment(*args,**kwargs)
    return SimpleNamespace(**locals())


def evaluate_one(context,run,case,checkpoint,package_id,boundary,progress):
    # Shape profiling/recompilation must not occur during a 20ms release. This
    # scoped public context restores the prior setting even on interruption.
    with context.torch.jit.optimized_execution(False):
        row=_evaluate_one(context,run,case,checkpoint,package_id,boundary,progress)
    row['runtime_jit_optimization']=dict(enabled=False,scope='complete evaluation case',
                                       original_setting_restored=True)
    return row


def _evaluate_one(context,run,case,checkpoint,package_id,boundary,progress):
    torch=context.torch
    raw=env=policy=session=None
    before_policy=before_inputs=None
    attempted=completed=0
    begun=time.time()
    row=dict(status='EPISODE_ERROR_PRESERVE',case_id=case['case_id'],seed=case['seed'],
        force_seed=case['force_seed'],checkpoint=checkpoint,package_identity=package_id,
        execution_completed=False,maximum_episodes=1,maximum_control_steps=3000,
        policy_and_normalizer_unchanged=False,checkpoint_unchanged=False,frozen_inputs_unchanged=False,
        external_body_wrench_zero_at_close=False,added_PPO_updates=0,training_started=False,
        hardware_acceptance=False,strict_contact_acceptance=False,population_success_rate_claimed=False)
    def finite(values,message):
        require(bool(torch.stack([torch.isfinite(v).all() for v in values]).all().item()),message)
    try:
        boundary()
        before_inputs=verify_inputs()
        validate_checkpoint(checkpoint)
        cfg=context.make_cfg(1,case['seed'])
        require(cfg.episode_length_s==10. and not cfg.curriculum and 'push_robot' not in cfg.events,
                'Original evaluation configuration differs')
        cfg.episode_length_s=60.
        cfg.auto_reset=False
        cfg.sim.mujoco.timestep=.005
        cfg.decimation=4
        progress(detail='build_candidate_environment',case_id=case['case_id'])
        raw=context.ManagerBasedRlEnv(cfg,device='cpu',render_mode=None)
        env=context.RslRlVecEnvWrapper(raw)
        agent=asdict(deepcopy(context.MicroduckBasketballRlCfg))
        agent.update(seed=case['seed'],logger='tensorboard',upload_model=False)
        agent['algorithm'].update(learning_rate=LEARNING_RATE,schedule='fixed',num_mini_batches=4)
        runner=context.MjlabOnPolicyRunner(env,agent,log_dir=None,device='cpu')
        saved=torch.load(checkpoint['path'],map_location='cpu',weights_only=False)
        meta=verify_checkpoint_metadata(saved,checkpoint)
        runner.alg.load(saved,{'actor':True,'critic':True,'optimizer':False,'iteration':False},strict=True)
        policy=runner.get_inference_policy(device='cpu')
        require(list(policy.obs_groups)==['actor'], 'Frozen inference policy must consume only actor observations')
        require(not policy.training and not policy.obs_normalizer.training,
                'Evaluation must freeze actor and observation normalizer')
        before_policy=fingerprint(policy.state_dict())
        raw.common_step_counter=meta['common_step_counter']
        raw._sim_step_counter=meta['common_step_counter']*4
        require(raw.max_episode_length==3000 and abs(raw.step_dt-.02)<1e-12
                and abs(raw.physics_dt-.005)<1e-12 and raw.cfg.decimation==4
                and context.head_scale(raw)==1. and not raw.cfg.auto_reset,
                'Original mature policy timing/reset contract differs')
        session=context.EvaluationSession(raw,torch,context.mdp,run,case['seed'],case['force_seed'])
        session.boundary=boundary
        with torch.inference_mode():
            observations,_=raw.reset(seed=case['seed'])
            obs=context.TensorDict(observations,batch_size=[1])
            policy.reset()
            session.after_reset(1)
            require(tuple(obs['actor'].shape)==(1,61) and tuple(obs['critic'].shape)==(1,85),
                    'Original 61D actor / 85D critic schema differs')
            finite(list(obs.values()),'Nonfinite initial evaluation observations')
            row['reference_preparation'] = policy.prepare(obs, boundary)
            session.arm_control()
            ended=False
            for step in range(1,3001):
                boundary()
                actions=policy(obs,stochastic_output=False)
                finite([actions],'Nonfinite deterministic action')
                attempted+=1
                obs,rewards,dones,_=session.step(env,actions)
                completed+=1
                after_environment=context.timing.clock()
                finite([*obs.values(),rewards,dones],'Nonfinite evaluation step output')
                require(not bool(raw.termination_manager.get_term('nan_state').any().item()),
                        'Nonfinite simulator state')
                if bool(dones[0]):
                    physical=bool(raw.reset_terminated[0])
                    terms=[name for name in raw.termination_manager.active_terms
                           if bool(raw.termination_manager.get_term(name)[0])]
                    session.finish('physical_termination' if physical else 'time_out',physical,terms)
                    ended=True
                context.timing.add(post_environment_checks_ms=(context.timing.clock()-after_environment)/1e6)
                progress_started=context.timing.clock()
                if step%25==0 or ended:
                    progress(detail='candidate_first_episode',case_id=case['case_id'],
                        control_steps=completed,checked_physics_steps=session.physics_calls,
                        completed_pulses=session.completed_pulses,
                        model_bounds_status='MODEL_BOUND_VIOLATION_OBSERVED' if session.guard.audit.first_event is not None else 'OBSERVED_WITHIN_CHECKED_MODEL_BOUNDS')
                context.timing.add(progress_emit_ms=(context.timing.clock()-progress_started)/1e6)
                context.timing.end()
                if ended:break
            require(ended,'Evaluation exceeded its single 60-second horizon')
        row.update(status='EPISODE_COMPLETE',execution_completed=True)
    except EvaluationInterrupted as exc:
        row.update(status='EPISODE_INTERRUPTED',stop_reason=str(exc))
    except BaseException as exc:
        if type(exc).__name__=='LimitStop' and hasattr(exc,'event'):
            row.update(status='EPISODE_MODEL_FAULT_PRESERVE',model_fault=exc.event,error=str(exc))
        else:
            row.update(status='EPISODE_ERROR_PRESERVE',error=repr(exc),
                       traceback=traceback.format_exc()[-10000:])
    finally:
        # Disable the per-case alarm during bounded cleanup; the supervisor owns
        # the final campaign deadline and exact child process group.
        signal.alarm(0)
        try:
            if session is not None:
                try:
                    session.finish(row['status'])
                    row.update(session.snapshot())
                finally:session.close()
                row['external_body_wrench_zero_at_close']=bool((raw.sim.data.xfrc_applied==0).all())
                require(row['external_body_wrench_zero_at_close'],'External force was not cleared')
            if policy is not None and before_policy is not None:
                row['policy_and_normalizer_unchanged']=fingerprint(policy.state_dict())==before_policy
                require(row['policy_and_normalizer_unchanged'],'Evaluation changed actor or normalizer')
            if before_inputs is not None:
                row['frozen_inputs_unchanged']=verify_inputs()==before_inputs
                row['checkpoint_unchanged']=sha(checkpoint['path'])==checkpoint['sha256']
                require(row['frozen_inputs_unchanged'] and row['checkpoint_unchanged'],'Frozen input changed')
        except BaseException as exc:
            row.update(status='EPISODE_ERROR_PRESERVE',cleanup_error=repr(exc))
        finally:
            try:
                if env is not None:env.close()
                elif raw is not None:raw.close()
            except BaseException as exc:
                row.update(status='EPISODE_ERROR_PRESERVE',environment_close_error=repr(exc))
        row.update(attempted_control_steps=attempted,completed_control_steps=completed,
            finished_utc=time.time(),elapsed_seconds=time.time()-begun)
        row['acceptance']=accept_case(row,case,checkpoint['sha256'])
    return row

