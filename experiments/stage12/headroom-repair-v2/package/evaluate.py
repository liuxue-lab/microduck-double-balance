"""Bounded first episodes of NEW candidates only; frozen policy and normalizer."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import os
import signal
import sys
import time
import traceback

from common import *
from budget import reserve_evaluation,finish_evaluation
from criteria import accept_episode,summarize
from fingerprint import fingerprint
from runtime_helpers import require_finite_group,require_step_outputs


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=tuple(EVAL_SEEDS),required=True)
    args=parser.parse_args();phase=args.phase
    folder=SLOT/'evaluations'/phase
    checkpoint=read(folder/'candidate.json');ledger=read(SLOT/'budget.json')
    package_id=verify_package();stopped=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stopped.__setitem__(0,True))

    def boundary():
        require(not stopped[0] and not (SLOT/'STOP.json').exists()
                and time.time()<ledger['deadline_utc'],'Evaluation stopped or original deadline reached')

    def progress(**values):
        write(SLOT/'progress.json',dict(status='EVALUATING',phase=phase,
            completed_updates=ledger['completed_updates'],attempted_updates=ledger['attempted_updates'],
            elapsed_seconds=time.time()-ledger['started_utc'],original_deadline_utc=ledger['deadline_utc'],**values))

    try:
        boundary();verify_runtime();sys.path.insert(0,str(SOURCE/'src'))
        import torch
        import warp as wp
        from tensordict import TensorDict
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
        from mjlab.utils.torch import configure_torch_backends
        from mjlab_microduck.tasks.microduck_basketball_env_cfg import MicroduckBasketballRlCfg
        from native_task import make_cfg,head_scale,mdp
        from baseline_runtime import Session
        from model_limits import LimitStop
        require(torch.cuda.is_available() and '5090' in torch.cuda.get_device_name(0),'Candidate evaluation requires cloud RTX5090')
        configure_torch_backends();torch.set_num_threads(min(4,len(os.sched_getaffinity(0))))
        wp.config.kernel_cache_dir=str(TRAIN/'cache/warp')
        rows=[]
        for seed in EVAL_SEEDS[phase]:
            key=phase+':'+str(seed);run=folder/str(seed);result_path=run/'result.json'
            previous=ledger['evaluations'].get(key)
            if previous is not None:
                require(previous['status']=='COMPLETE' and result_path.is_file()
                        and sha(result_path)==previous['result_sha256'],
                        'Interrupted or changed evaluation; preserve, never resample')
                row=read(result_path)
                require(row['seed']==seed and row['checkpoint']==checkpoint,'Retained candidate identity differs')
                rows.append(row);continue
            boundary()
            reserve_evaluation(ledger,phase,seed,checkpoint['sha256'],time.time())
            write(SLOT/'budget.json',ledger)
            run.mkdir(exist_ok=False)
            write(run/'started.json',dict(seed=seed,phase=phase,checkpoint=checkpoint,
                deadline_utc=ledger['deadline_utc'],maximum_control_steps=3000,
                maximum_episodes=1,disturbance_enabled=False),exclusive=True)
            raw=env=policy=session=None;before=None
            row=dict(status='EPISODE_ERROR_PRESERVE',seed=seed,checkpoint=checkpoint,
                package_identity=package_id,selection_set=phase!='heldout',
                policy_and_normalizer_unchanged=False,checkpoint_unchanged=False,
                full_hardware_limits_verified=False,extra_PPO_updates=0)
            try:
                progress(detail='build_candidate_environment',seed=seed,completed_episodes=len(rows))
                cfg=make_cfg(1,seed);cfg.auto_reset=False
                require(cfg.episode_length_s==10 and not cfg.curriculum and 'push_robot' not in cfg.events,
                        'Original static evaluation configuration differs')
                cfg.episode_length_s=60.
                raw=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode=None);env=RslRlVecEnvWrapper(raw)
                agent=asdict(deepcopy(MicroduckBasketballRlCfg))
                agent.update(seed=seed,logger='tensorboard',upload_model=False)
                agent['algorithm'].update(learning_rate=LEARNING_RATE,schedule='fixed',num_mini_batches=4)
                runner=MjlabOnPolicyRunner(env,agent,log_dir=None,device='cuda:0')
                require(sha(checkpoint['path'])==checkpoint['sha256'],'Candidate bytes changed')
                saved=torch.load(checkpoint['path'],map_location='cpu',weights_only=False)
                meta=saved['infos']['stage12_headroom']
                require(meta['package_identity']==package_id and meta['policy_updates']==checkpoint['policy_updates']
                        and meta['origin_checkpoint_sha256']==MODEL_SHA,'Candidate lineage differs')
                runner.alg.load(saved,{'actor':True,'critic':True,'optimizer':False,'iteration':False},strict=True)
                policy=runner.get_inference_policy(device='cuda:0');policy.reset()
                require(not policy.training and not policy.obs_normalizer.training,'Evaluation must freeze policy and normalization')
                before=fingerprint(policy.state_dict())
                raw.common_step_counter=1000*24;raw._sim_step_counter=1000*24*10
                require(raw.max_episode_length==3000 and abs(raw.step_dt-.02)<1e-12 and head_scale(raw)==1.,
                        'Evaluation time/schedule contract differs')
                session=Session(raw,torch,mdp,run,seed)
                with torch.inference_mode():
                    observations,_=raw.reset(seed=seed)
                    obs=TensorDict(observations,batch_size=[1]);policy.reset()
                    session.after_reset(1)
                    require(tuple(obs['actor'].shape)==(1,61) and tuple(obs['critic'].shape)==(1,85),
                            'Evaluation observations differ')
                    ended=False
                    for step in range(1,3001):
                        boundary();actions=policy(obs,stochastic_output=False)
                        require_finite_group(torch,[actions],'Nonfinite deterministic action')
                        obs,rewards,dones,_=session.step(env,actions)
                        require_step_outputs(torch,obs,rewards,dones,raw.termination_manager.get_term('nan_state'))
                        if bool(dones[0]):
                            physical=bool(raw.reset_terminated[0])
                            terms=[name for name in raw.termination_manager.active_terms
                                   if bool(raw.termination_manager.get_term(name)[0])]
                            session.finish('physical_termination' if physical else 'time_out',physical,terms)
                            ended=True
                        if step%25==0 or ended:
                            progress(detail='candidate_first_episode',seed=seed,control_steps=step,
                                completed_episodes=len(rows),checked_physics_steps=session.physics_calls)
                        if ended:break
                    require(ended,'Candidate exceeded the approved single 60-second episode')
                row['status']='EPISODE_COMPLETE'
            except LimitStop as exc:
                row.update(status='EPISODE_MODEL_FAULT_PRESERVE',error=str(exc),model_fault=exc.event)
            except BaseException as exc:
                row.update(error=repr(exc),traceback=traceback.format_exc()[-10000:])
            finally:
                try:
                    if session is not None:
                        session.finish(row['status'])
                        row.update(session.snapshot());session.close()
                        row['external_body_wrench_zero_at_close']=True
                    if policy is not None and before is not None:
                        row['policy_and_normalizer_unchanged']=fingerprint(policy.state_dict())==before
                    row['checkpoint_unchanged']=sha(checkpoint['path'])==checkpoint['sha256']
                    if env is not None:env.close()
                    elif raw is not None:raw.close()
                except BaseException as exc:
                    row.update(status='EPISODE_ERROR_PRESERVE',cleanup_error=repr(exc))
                row['acceptance']=accept_episode(row)
                write(result_path,row,exclusive=True)
                finish_evaluation(ledger,key,sha(result_path));write(SLOT/'budget.json',ledger)
            rows.append(row)
            require(row['status']=='EPISODE_COMPLETE','Candidate evaluation encountered an electrical/numerical/runtime fault; preserve')
        result=dict(status='EVALUATION_COMPLETE',phase=phase,checkpoint=checkpoint,
            seeds=list(EVAL_SEEDS[phase]),summary=summarize(rows,EVAL_SEEDS[phase],checkpoint['sha256']),
            evidence=[dict(path=str(folder/str(r['seed'])/'result.json'),
                           sha256=sha(folder/str(r['seed'])/'result.json')) for r in rows],
            extra_PPO_updates=0,finished_utc=time.time(),paired_baseline_comparison=False)
        write(folder/'evaluation-result.json',result,exclusive=True)
        progress(detail='candidate_evaluation_saved',completed_episodes=len(rows),summary=result['summary'])
        return 0
    except BaseException as exc:
        write(folder/'error.json',dict(status='EVALUATION_ERROR_PRESERVE',error=repr(exc),
            traceback=traceback.format_exc()[-10000:],finished_utc=time.time()),exclusive=True)
        return 2


if __name__=='__main__':
    raise SystemExit(main())
