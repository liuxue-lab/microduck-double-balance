"""Fixed first-episode candidate selection and heldout verification; never PPO."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import math
import os
from pathlib import Path
import signal
import sys
import time
import traceback
from types import SimpleNamespace

from common import *
from budget import reserve_evaluation, finish_evaluation, validate
from eval_plan import case_plan, choose_candidate, MAX_CASE_SECONDS
from eval_criteria import accept_case, summarize_phase
from fingerprint import fingerprint


class EvaluationInterrupted(Exception):
    pass


def valid_evaluated_case(row):
    if row['status']=='EPISODE_COMPLETE':return True
    # Finite physical boundary violations are measured failed trials. They
    # disqualify this candidate, without hiding the other prescribed candidates.
    # Numerical/runtime faults stop the whole task instead of being resampled.
    event=row.get('model_fault') or {}
    return row['status']=='EPISODE_MODEL_FAULT_PRESERVE' and event.get('kind') in (
        'JOINT_POSITION_LIMIT','PWM_LIMIT','MODELED_CURRENT_LIMIT','APPLIED_TORQUE_LIMIT') \
        and all(row.get(k) is True for k in ('policy_and_normalizer_unchanged',
            'checkpoint_unchanged','frozen_inputs_unchanged','external_body_wrench_zero_at_close')) \
        and not any(row.get(k) for k in ('cleanup_error','environment_close_error'))


def validate_run_dir(folder,phase):
    require(folder.parent.resolve()==(SLOT/'evaluations').resolve() and not folder.is_symlink(),
            'Unexpected evaluation directory')
    expected=('selection-001','selection-002','selection-003') if phase=='selection' else ('heldout',)
    require(folder.name in expected,'Unapproved candidate evaluation slot')


def validate_checkpoint(checkpoint):
    path=Path(checkpoint['path'])
    require(path.is_file() and not path.is_symlink()
            and path.resolve().is_relative_to((INPUTS/'run/checkpoints').resolve()),
            'Evaluate a returned immutable training checkpoint only')
    require(path.stat().st_size==checkpoint['bytes'] and sha(path)==checkpoint['sha256'],
            'Candidate checkpoint changed')
    require(type(checkpoint['completed_updates']) is int and checkpoint['completed_updates']>0
            and checkpoint['policy_updates']==START_POLICY_UPDATES+checkpoint['completed_updates'],
            'Candidate policy accounting differs')


def validate_heldout_selection(checkpoint):
    locked=read(SLOT/'selection.json')
    selected=read(SLOT/'selected-candidate.json')
    require(locked.get('qualified') is True and locked.get('checkpoint')==checkpoint and selected==checkpoint,
            'Heldout candidate differs from the immutable selection lock')
    selections=[]
    for number in range(1,4):
        path=SLOT/'evaluations'/('selection-%03d'%number)/'evaluation-result.json'
        if not path.is_file():continue
        item=read(path)
        for evidence in item['evidence']:
            require(sha(evidence['path'])==evidence['sha256'],'Selection case evidence changed')
        selections.append(item)
    chosen=choose_candidate(selections)
    require(chosen['qualified'] is True and chosen['checkpoint']==checkpoint,
            'Heldout must use the single qualified selection winner')
    return chosen


def load_runtime():
    verify_runtime()
    sys.path.insert(0,str(SOURCE/'src'))
    import torch
    import warp as wp
    from tensordict import TensorDict
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from mjlab_microduck.tasks.microduck_basketball_env_cfg import MicroduckBasketballRlCfg
    from native_task import make_cfg,head_scale,mdp
    from eval_runtime import EvaluationSession
    require(torch.cuda.is_available() and '5060' in torch.cuda.get_device_name(0),
            'Use the existing local RTX5060 for frozen-policy validation')
    configure_torch_backends()
    torch.set_num_threads(min(4,len(os.sched_getaffinity(0))))
    # Reuse the existing local Warp cache, as in the successful local pilot.
    return SimpleNamespace(**locals())


def evaluate_one(context,run,case,checkpoint,package_id,boundary,progress):
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
        progress(detail='build_candidate_environment',case_id=case['case_id'])
        raw=context.ManagerBasedRlEnv(cfg,device='cuda:0',render_mode=None)
        env=context.RslRlVecEnvWrapper(raw)
        agent=asdict(deepcopy(context.MicroduckBasketballRlCfg))
        agent.update(seed=case['seed'],logger='tensorboard',upload_model=False)
        agent['algorithm'].update(learning_rate=LEARNING_RATE,schedule='fixed',num_mini_batches=4)
        runner=context.MjlabOnPolicyRunner(env,agent,log_dir=None,device='cuda:0')
        saved=torch.load(checkpoint['path'],map_location='cpu',weights_only=False)
        meta=verify_checkpoint_metadata(saved,checkpoint)
        runner.alg.load(saved,{'actor':True,'critic':True,'optimizer':False,'iteration':False},strict=True)
        policy=runner.get_inference_policy(device='cuda:0')
        require(list(policy.obs_groups)==['actor'], 'Frozen inference policy must consume only actor observations')
        require(not policy.training and not policy.obs_normalizer.training,
                'Evaluation must freeze actor and observation normalizer')
        before_policy=fingerprint(policy.state_dict())
        raw.common_step_counter=meta['common_step_counter']
        raw._sim_step_counter=meta['sim_step_counter']
        require(raw.max_episode_length==3000 and abs(raw.step_dt-.02)<1e-12
                and abs(raw.physics_dt-.002)<1e-12 and raw.cfg.decimation==10
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
            ended=False
            for step in range(1,3001):
                boundary()
                actions=policy(obs,stochastic_output=False)
                finite([actions],'Nonfinite deterministic action')
                attempted+=1
                obs,rewards,dones,_=session.step(env,actions)
                completed+=1
                finite([*obs.values(),rewards,dones],'Nonfinite evaluation step output')
                require(not bool(raw.termination_manager.get_term('nan_state').any().item()),
                        'Nonfinite simulator state')
                if bool(dones[0]):
                    physical=bool(raw.reset_terminated[0])
                    terms=[name for name in raw.termination_manager.active_terms
                           if bool(raw.termination_manager.get_term(name)[0])]
                    session.finish('physical_termination' if physical else 'time_out',physical,terms)
                    ended=True
                if step%25==0 or ended:
                    snapshot=session.snapshot()
                    progress(detail='candidate_first_episode',case_id=case['case_id'],
                        control_steps=completed,checked_physics_steps=session.physics_calls,
                        completed_pulses=snapshot['disturbance']['completed_pulses'],
                        model_bounds_status=snapshot['model_limits']['status'])
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


def main(keep_shutdown_handlers=False):
    parser=argparse.ArgumentParser()
    parser.add_argument('--phase',choices=('selection','heldout'),required=True)
    parser.add_argument('--checkpoint-json',required=True)
    parser.add_argument('--run-dir',required=True)
    args=parser.parse_args()
    phase=args.phase;folder=Path(args.run_dir)
    validate_run_dir(folder,phase)
    checkpoint=read(args.checkpoint_json)
    package_id=verify_package()
    validate_checkpoint(checkpoint)
    ledger=read(SLOT/'budget.json')
    validate(ledger)
    plan=case_plan(phase)
    output=folder/'evaluation-result.json'
    if output.is_file():
        existing=read(output)
        require(existing['phase']==phase and existing['checkpoint']==checkpoint,'Retained evaluation identity changed')
        require(existing['summary']==summarize_phase(existing['cases'],phase,checkpoint['sha256']),
                'Retained evaluation summary changed')
        for item in existing['evidence']:
            require(sha(item['path'])==item['sha256'],'Retained case evidence changed')
        return 0 if existing['status']=='EVALUATION_COMPLETE' else 3 if existing['status']=='EVALUATION_INCOMPLETE_PRESERVE' else 2
    folder.mkdir(parents=True,exist_ok=True)
    if phase=='heldout':validate_heldout_selection(checkpoint)
    write(folder/'phase-started.json',dict(phase=phase,checkpoint=checkpoint,package_identity=package_id,
        plan=plan,deadline_utc=ledger['deadline_utc'],maximum_episodes=len(plan),added_PPO_updates=0),exclusive=True)
    rows=[];evidence=[];status='EVALUATION_ERROR_PRESERVE';failure=None;stopped=[False]
    original_handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGALRM)}
    def interrupt(signum,frame):
        if signum==signal.SIGALRM:raise EvaluationInterrupted('per_case_or_campaign_deadline')
        stopped[0]=True
    for sig in original_handlers:signal.signal(sig,interrupt)
    case_deadline=ledger['deadline_utc']
    def boundary():
        if stopped[0] or (SLOT/'STOP.json').exists():
            raise EvaluationInterrupted('stop_requested')
        if time.time()>=min(case_deadline,ledger['deadline_utc']):
            raise EvaluationInterrupted('per_case_or_campaign_deadline')
    def progress(**values):
        write(SLOT/'progress.json',dict(status='EVALUATING',phase=phase,
            completed_updates=ledger['completed_updates'],attempted_updates=ledger['attempted_updates'],
            elapsed_seconds=time.time()-ledger['started_utc'],deadline_utc=ledger['deadline_utc'],
            completed_evaluations=len(evidence),planned_evaluations=len(plan),
            checkpoint=checkpoint,added_PPO_updates=0,**values))
    try:
        boundary()
        context=load_runtime()
        for index,case in enumerate(plan):
            boundary()
            key=folder.name+':'+case['case_id']
            reserve_evaluation(ledger,phase,key,checkpoint['sha256'],time.time())
            write(SLOT/'budget.json',ledger)
            run=folder/('case-%03d'%index)
            run.mkdir(exist_ok=False)
            now=time.time();case_deadline=min(now+MAX_CASE_SECONDS,ledger['deadline_utc'])
            write(run/'started.json',dict(case,phase=phase,checkpoint=checkpoint,
                package_identity=package_id,started_utc=now,deadline_utc=case_deadline,
                maximum_episodes=1,maximum_control_steps=3000,maximum_physics_steps=30000,
                added_PPO_updates=0,automatic_retry=False),exclusive=True)
            signal.alarm(max(1,math.ceil(case_deadline-time.time())))
            row=evaluate_one(context,run,case,checkpoint,package_id,boundary,progress)
            result_path=run/'result.json'
            write(result_path,row,exclusive=True)
            finish_evaluation(ledger,key,sha(result_path))
            write(SLOT/'budget.json',ledger)
            rows.append(row)
            evidence.append(dict(path=str(result_path),sha256=sha(result_path),key=key))
            if row['status']=='EPISODE_INTERRUPTED':
                raise EvaluationInterrupted(row.get('stop_reason','case_interrupted'))
            require(valid_evaluated_case(row),'Evaluation runtime or numerical fault; preserve')
            if phase=='heldout' and not row['acceptance']['passed']:
                break
        status='EVALUATION_COMPLETE'
    except EvaluationInterrupted as exc:
        status='EVALUATION_INCOMPLETE_PRESERVE';failure=dict(stop_reason=str(exc))
    except BaseException as exc:
        status='EVALUATION_ERROR_PRESERVE';failure=dict(error=repr(exc),traceback=traceback.format_exc()[-10000:])
    finally:
        signal.alarm(0)
        summary=summarize_phase(rows,phase,checkpoint['sha256'])
        result=dict(status=status,phase=phase,checkpoint=checkpoint,package_identity=package_id,
            planned_cases=len(plan),evaluated_cases=len(rows),summary=summary,cases=rows,evidence=evidence,
            stopped_on_first_failure=phase=='heldout' and bool(rows) and not rows[-1]['acceptance']['passed'],
            added_PPO_updates=0,automatic_training=False,automatic_retry=False,
            finished_utc=time.time(),deadline_utc=ledger['deadline_utc'],
            parent_model_preserved=True,hardware_acceptance=False,strict_contact_acceptance=False,
            all_possible_sequences_proven=False,old_64_case_suite_replaced_or_claimed_passed=False)
        if failure:result['failure']=failure
        write(output,result,exclusive=True)
        # Keep the no-new-work stop handler during interpreter/CUDA teardown.
        # Restoring SIGTERM's default after publishing a result races with the
        # supervisor deadline and can turn a completed worker into exit -15.
        for sig,handler in original_handlers.items():
            if keep_shutdown_handlers and sig in (signal.SIGINT,signal.SIGTERM):
                signal.signal(sig,lambda *_:stopped.__setitem__(0,True))
            else:signal.signal(sig,handler)
        progress(detail='evaluation_saved',evaluation_status=status,summary=summary)
    return 0 if status=='EVALUATION_COMPLETE' else 3 if status=='EVALUATION_INCOMPLETE_PRESERVE' else 2


if __name__=='__main__':
    raise SystemExit(main(keep_shutdown_handlers=True))

