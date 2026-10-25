"""Evaluate each candidate once; reuse the original 4500 reference episodes."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import importlib.metadata
import os
import signal
import sys
import time
import traceback
from common import *
from eval_metrics import Observer,fingerprint,initial_fingerprint,summarize,THRESHOLDS
from runtime_helpers import require_finite_group,require_step_outputs

def retained_group(folder,seed,checkpoint):
    path=folder/('group-%d.json'%seed)
    if path.exists():
        value=read(path)
        require(value['status']=='EVALUATED' and value['seed']==seed and value['checkpoint_sha256']==checkpoint['sha256'], 'Completed candidate group identity differs')
        require(value['policy_and_normalizer_unchanged'] and len(value['episodes'])==EVAL_ENVS,'Completed candidate group is incomplete')
        require(sorted(r['env_id'] for r in value['episodes'])==list(range(EVAL_ENVS)),'Candidate episode identities differ')
        require(value['summary']==summarize(value['episodes']),'Candidate aggregate differs under the project Python')
        return value
    require(not (folder/('group-%d-started.json'%seed)).exists(),'An evaluation group was interrupted; preserve it instead of resampling first episodes')
    return None

def evaluate_group(folder,checkpoint,seed,index,boundary,progress):
    value=retained_group(folder,seed,checkpoint)
    if value is not None:return value
    boundary()
    # Reservation is durable before constructing any environment or sampling.
    write(folder/('group-%d-started.json'%seed),dict(seed=seed,checkpoint=checkpoint,started_utc=time.time()),exclusive=True)
    import torch
    from tensordict import TensorDict
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
    from mjlab_microduck.tasks.microduck_basketball_env_cfg import MicroduckBasketballRlCfg
    from mjlab_microduck.tasks import mdp
    from native_task import make_cfg,head_scale
    cfg=make_cfg(EVAL_ENVS,seed);cfg.auto_reset=False
    require(abs(cfg.episode_length_s-10.)<1e-12 and not cfg.curriculum,'Evaluation horizon or curriculum changed')
    agent=asdict(deepcopy(MicroduckBasketballRlCfg))
    agent.update(seed=seed,logger='tensorboard',upload_model=False)
    agent['algorithm'].update(learning_rate=LEARNING_RATE,schedule='fixed',num_mini_batches=4)
    raw=env=observer=None
    try:
        progress(status='EVALUATING',phase='build_evaluation_environment',seed=seed,group=index+1,
            completed_first_episodes=index*EVAL_ENVS,control_steps_in_group=0)
        raw=ManagerBasedRlEnv(cfg,device='cuda:0',render_mode=None);env=RslRlVecEnvWrapper(raw)
        runner=MjlabOnPolicyRunner(env,agent,log_dir=None,device='cuda:0')
        path=Path(checkpoint['path']);require(sha(path)==checkpoint['sha256'],'Candidate checkpoint changed')
        saved=torch.load(path,map_location='cpu',weights_only=False);meta=saved['infos']['stage12_native']
        require(meta['package_identity']==read(HERE/'manifest.json')['identity'] and meta['policy_updates']==checkpoint['policy_updates'], 'Candidate lineage differs')
        runner.alg.load(saved,{'actor':True,'critic':True,'optimizer':False,'iteration':False},strict=True)
        policy=runner.get_inference_policy(device='cuda:0');policy.reset()
        require(not policy.training and not policy.obs_normalizer.training,'Evaluation policy and normalization must be frozen')
        before=fingerprint(policy.state_dict())
        raw.common_step_counter=1000*24;raw._sim_step_counter=1000*24*10
        raw_obs,_=raw.reset(seed=seed);obs=TensorDict(raw_obs,batch_size=[EVAL_ENVS]);policy.reset()
        require(raw.max_episode_length==500 and abs(raw.step_dt-.02)<1e-12 and head_scale(raw)==1.,'Evaluation time contract differs')
        require(tuple(obs['actor'].shape)==(EVAL_ENVS,61) and tuple(obs['critic'].shape)==(EVAL_ENVS,85),'Evaluation observation schema differs')
        initial=initial_fingerprint(raw,obs);reference=read(PRIOR/('initial-seed-%d.json'%seed))
        require(reference['fields']==initial['fields'],'Initial physical/observation/randomization schema differs')
        require(all(v['finite'] for v in initial['field_diagnostics'].values()),'Nonfinite evaluation initial state')
        initial.update(prior_4500_sha256=reference['sha256'],matches_prior_4500_exactly=reference['sha256']==initial['sha256'],
            paired_comparison_allowed=False,difference_cause='UNKNOWN; exact paired initial states are not claimed')
        write(folder/('initial-%d.json'%seed),initial,exclusive=True)
        observer=Observer(raw,torch,mdp);raw.metrics_manager.compute=observer.compute
        terminated=[False]*EVAL_ENVS;terms=[[] for _ in range(EVAL_ENVS)]
        with torch.inference_mode():
            for step in range(1,501):
                boundary();actions=policy(obs,stochastic_output=False)
                require_finite_group(torch,[actions],'Nonfinite deterministic action')
                obs,rewards,dones,_=env.step(actions)
                require_step_outputs(torch,obs,rewards,dones,raw.termination_manager.get_term('nan_state'))
                ended=observer.active&dones.bool();ids=ended.nonzero(as_tuple=False).flatten().cpu().tolist()
                if ids:
                    physical=raw.reset_terminated.cpu().tolist()
                    term_values={name:raw.termination_manager.get_term(name).cpu().tolist() for name in raw.termination_manager.active_terms}
                    for i in ids:
                        terminated[i]=bool(physical[i]);terms[i]=[name for name,val in term_values.items() if val[i]]
                observer.active &= ~ended
                done_count=EVAL_ENVS-int(observer.active.sum().item())
                if step%25==0 or done_count==EVAL_ENVS:
                    progress(phase='deterministic_evaluation',seed=seed,group=index+1,control_steps_in_group=step,
                        completed_first_episodes=index*EVAL_ENVS+done_count)
                if done_count==EVAL_ENVS:break
                reset_ids=dones.nonzero(as_tuple=False).flatten()
                if len(reset_ids):
                    raw.reset(env_ids=reset_ids);policy.reset(dones.bool());obs=env.get_observations()
        require(not bool(observer.active.any()),'First episode exceeded the original ten-second horizon')
        rows=observer.rows(seed,terminated,terms)
        require(fingerprint(policy.state_dict())==before,'Policy or normalization changed during evaluation')
        require(sha(path)==checkpoint['sha256'],'Candidate checkpoint changed during evaluation')
        value=dict(status='EVALUATED',model=str(checkpoint['policy_updates']),seed=seed,checkpoint_sha256=checkpoint['sha256'],
            initial=initial,summary=summarize(rows),episodes=rows,policy_and_normalizer_unchanged=True,
            vector_steps_executed=step,all_environment_control_transitions=step*EVAL_ENVS,
            counted_first_episode_steps=sum(r['steps'] for r in rows))
        write(folder/('group-%d.json'%seed),value,exclusive=True)
        return value
    finally:
        if observer is not None:raw.metrics_manager.compute=observer.original
        if env is not None:env.close()
        elif raw is not None:raw.close()

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--segment',type=int,required=True);a=ap.parse_args()
    require(1<=a.segment<=MAX_SEGMENTS,'Invalid segment')
    folder=SLOT/'evaluations'/('segment-%02d'%a.segment)
    checkpoint=read(folder/'candidate.json');budget=read(SLOT/'budget.json')
    state=dict(status='INITIALIZING',phase='load_evaluation_runtime',segment=a.segment,total_first_episodes=256,
        completed_first_episodes=0,completed_updates=budget['completed_updates'],attempted_updates=budget['attempted_updates'])
    stop=[False]
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,lambda *_:stop.__setitem__(0,True))
    def boundary():
        require(not stop[0] and not (SLOT/'STOP.json').exists() and time.time()<budget['deadline_utc'],'Evaluation stopped or original two-hour deadline reached')
    def progress(**kw):state.update(kw,elapsed_seconds=time.time()-budget['started_utc']);write(SLOT/'progress.json',state)
    try:
        require(not (folder/'evaluation-result.json').exists(),'Completed evaluation must be reused by the supervisor')
        # Detect interrupted groups before loading Torch or creating any new group.
        for seed in SEEDS:retained_group(folder,seed,checkpoint)
        progress();boundary()
        expected={'mjlab':'1.3.0','mujoco':'3.10.0','mujoco-warp':'3.8.1','warp-lang':'1.12.0','rsl-rl-lib':'5.0.1'}
        require({k:importlib.metadata.version(k) for k in expected}==expected,'Existing runtime differs; no installation')
        sys.path.insert(0,str(SOURCE/'src'))
        import torch
        import warp as wp
        from mjlab.utils.torch import configure_torch_backends
        require(torch.cuda.is_available() and '5090' in torch.cuda.get_device_name(0),'Evaluation requires the existing cloud RTX5090')
        configure_torch_backends();torch.set_num_threads(min(4,len(os.sched_getaffinity(0))))
        wp.config.kernel_cache_dir=str(TRAIN/'cache/warp')
        groups=[evaluate_group(folder,checkpoint,seed,i,boundary,progress) for i,seed in enumerate(SEEDS)]
        rows=[r for g in groups for r in g['episodes']]
        result=dict(status='EVALUATION_COMPLETE',segment=a.segment,model=str(checkpoint['policy_updates']),checkpoint=checkpoint,
            summary=summarize(rows),thresholds=THRESHOLDS,seeds=list(SEEDS),num_envs_per_group=EVAL_ENVS,
            paired_comparison_allowed=False,selection_set_not_final_heldout_test=True,task_acceptance_claimed=False,
            success_definition='First episode: >=250 consecutive original metric ticks, and no physical termination before episode ends; timeout allowed.',
            policy_mode='deterministic; frozen model and normalization; original noise and DR retained',
            group_evidence=[dict(path=str(folder/('group-%d.json'%g['seed'])),sha256=sha(folder/('group-%d.json'%g['seed'])),
                seed=g['seed'],summary=g['summary'],initial_sha256=g['initial']['sha256'],matches_prior_4500_exactly=g['initial']['matches_prior_4500_exactly']) for g in groups],
            new_all_environment_control_transitions=sum(g['all_environment_control_transitions'] for g in groups),
            python=sys.version,executable=sys.executable,finished_utc=time.time())
        write(folder/'evaluation-result.json',result,exclusive=True)
        progress(status='EVALUATION_SEGMENT_COMPLETE',phase='evaluation_saved',completed_first_episodes=256)
        return 0
    except BaseException as exc:
        progress(status='EVALUATION_ERROR_PRESERVE',error=repr(exc),traceback=traceback.format_exc()[-10000:])
        return 2

if __name__=='__main__':raise SystemExit(main())
