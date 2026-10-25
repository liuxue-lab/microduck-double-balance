"""Stage 09 checkpoint provenance and explicitly authorized restoration rules."""
from __future__ import annotations
from dataclasses import asdict
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess

import torch

from mjlab_microduck.double_balance_stage08_state import (
    TASK_ID, HOLD_LEVELS, VERSIONS, check_loaded_runner, check_optimizer, check_observations,
    require, require_equal, require_finite, file_sha256, utc_now, resume_agent_config,
    load_runner_checkpoint, validate_stage08_checkpoint, validate_assistance,
    runtime_profile as runtime_e_profile, assert_zero_assistance, check_training_step,
)
from mjlab_microduck.double_balance_stage09_plan import (
    BASELINE_COMMIT, PRIMARY_SHA, NUM_ENVS, SEED, PROFILES, TERMS, progress, profile_record, apply_profile,
)
from mjlab_microduck.double_balance_stage09_budget import atomic_json

ROOT = Path('/root/autodl-tmp/microduck-double-balance')
ARTIFACTS = ROOT/'artifacts/double-balance-stage09'


def verify_source(repo, *, clean=False):
    repo = Path(repo)
    manifest = json.loads((repo/'docs/audits/stage-09-runtime-hashes.json').read_text())
    for relative, checksum in manifest.items():
        require(file_sha256(repo/relative) == checksum, 'Stage 09 source differs: '+relative)
    head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    subprocess.run(['git','merge-base','--is-ancestor',BASELINE_COMMIT,head],cwd=repo,check=True)
    if clean:
        require(not subprocess.check_output(['git','status','--porcelain'],cwd=repo,text=True).strip(),
                'Commit the validated Stage 09 code before cloud execution')
    return head


def preflight(*, cloud=False):
    require(not any(k.startswith('MICRODUCK_') for k in os.environ), 'Unset task overrides')
    require(int(os.environ.get('WORLD_SIZE','1')) == 1, 'Single GPU only')
    require({k:version(k) for k in VERSIONS} == VERSIONS, 'Simulation lock differs')
    require(torch.__version__.split('+')[0]=='2.9.1' and torch.version.cuda=='12.8', 'Torch/CUDA lock differs')
    require(torch.cuda.is_available() and torch.cuda.device_count()==1, 'One CUDA GPU required')
    name = torch.cuda.get_device_name(0)
    if cloud:
        require(ROOT.is_dir() and name=='NVIDIA GeForce RTX 5090' and
                torch.cuda.get_device_properties(0).total_memory>=30*1024**3,
                'Formal PPO requires the approved cloud RTX 5090')
    else:
        require('5060' in name, 'This local zero-PPO entry is for the RTX 5060')
    torch.set_num_threads(min(4,len(os.sched_getaffinity(0))))
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    return verify_source(Path(__file__).resolve().parents[2],clean=cloud)


def validate_checkpoint(checkpoint, *, profile=None, campaign_id=None):
    state = checkpoint['infos']['stage09']
    require(state['schema_version']==1 and state['task_id']==TASK_ID and
            state['source_sha256']==PRIMARY_SHA and state['baseline_commit']==BASELINE_COMMIT,
            'Invalid Stage 09 checkpoint provenance')
    name = state['profile']
    require(name in PROFILES and state['profile_record']==profile_record(name), 'Treatment changed')
    require(state['seed']==SEED and state['purpose']=='formal', 'Invalid Stage 09 seed/purpose')
    if profile is not None:
        require(name==profile,'Cannot resume another treatment')
    if campaign_id is not None:
        require(state['campaign_id']==campaign_id,'Cannot reset or switch campaign budget')
    expected = progress(state['experiment_completed_updates'])
    for key,value in expected.items():
        require(state[key]==value,'Progress differs: '+key)
    require(checkpoint['iter']==expected['last_completed_iteration'],'Saved iteration mismatch')
    require(checkpoint['infos']['env_state']['common_step_counter']==expected['common_step_counter'],
            'Saved global counter mismatch')
    validate_assistance(state)
    require(bool((state['level']==5).all() and (state['hold']==0).all()),'Saved assistance is nonzero')
    groups = checkpoint['optimizer_state_dict']['param_groups']
    lr = state['learning_rate']
    require(math.isfinite(lr) and lr>0 and len(groups)==1 and groups[0]['lr']==lr,'Saved LR mismatch')
    moments = checkpoint['optimizer_state_dict']['state']
    require(len(moments)==len(groups[0]['params']) and bool(moments),'Missing Adam state')
    for value in moments.values():
        require(int(value['step'].item())==expected['expected_adam_steps'] and
                'exp_avg' in value and 'exp_avg_sq' in value,'Adam progress/moments differ')
    require_finite(checkpoint,'stage09 checkpoint')
    return state


def load_checked(path, *, evaluation=False, profile=None, campaign_id=None):
    path = Path(path).resolve()
    checksum = file_sha256(path)
    if checksum != PRIMARY_SHA:
        receipt = json.loads(path.with_suffix('.json').read_text())
        require(receipt['sha256']==checksum,'Checkpoint receipt SHA mismatch')
    checkpoint = torch.load(path,map_location='cpu',weights_only=False)
    if checksum == PRIMARY_SHA:
        state = validate_stage08_checkpoint(checkpoint,profile='E',seed=20260929)
        require(state['purpose']=='formal' and state['experiment_completed_updates']==4000 and
                state['lineage_completed_updates']==5000,'Wrong Stage 09 source progress')
    elif evaluation and checksum in (
        '42e759a80626859aa6756bec81e3fd07e53845a4ec7ef1a56f5054fd3dfacd1f',
        '89c7cedb9567f144228deb4d98465a9b0aad08be9f7c3ad391aa6a12ade8431c'):
        state=validate_stage08_checkpoint(checkpoint,profile='E')
    else:
        require('stage09' in checkpoint['infos'],'Only the primary source or Stage 09 checkpoints are admitted')
        state = validate_checkpoint(checkpoint,profile=profile,campaign_id=campaign_id)
    return checkpoint,state,checksum


def build_config(profile, checkpoint):
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    import mjlab_microduck.tasks  # noqa: F401
    cfg = load_env_cfg(TASK_ID)
    cfg.scene.num_envs,cfg.seed = NUM_ENVS,SEED
    cfg = apply_profile(cfg,profile)
    agent = resume_agent_config(asdict(load_rl_cfg(TASK_ID)),checkpoint)
    agent.update(seed=SEED,logger='tensorboard',upload_model=False,check_for_nan=True,
                 save_interval=100,max_iterations=5500,run_name='stage09_'+profile)
    require(agent['num_steps_per_env']==24 and agent['algorithm']['num_learning_epochs']*
            agent['algorithm']['num_mini_batches']==20,'PPO batch contract changed')
    require(agent['algorithm']['schedule']=='adaptive' and agent['algorithm']['desired_kl']==.01,
            'PPO LR schedule changed')
    return cfg,agent


def runtime_profile(raw, profile):
    result = runtime_e_profile(raw,'E')
    names = set(raw.reward_manager.active_terms)
    expected = {term:w for term,w in zip(TERMS,PROFILES[profile]) if w}
    require(names.intersection(TERMS)==set(expected),'Unexpected Stage 09 reward terms')
    for term,weight in expected.items():
        require(raw.reward_manager.get_term_cfg(term).weight==weight,'Live reward weight differs')
    result['stage09_rewards']=expected
    return result


def restore(runner, env, checkpoint, *, profile, campaign_id):
    raw = env.unwrapped
    check_loaded_runner(runner,raw,checkpoint)
    is_new = 'stage09' not in checkpoint['infos']
    state = (validate_stage08_checkpoint(checkpoint,profile='E',seed=20260929) if is_new else
             validate_checkpoint(checkpoint,profile=profile,campaign_id=campaign_id))
    completed = 0 if is_new else state['experiment_completed_updates']
    counters = progress(completed)
    check_optimizer(runner.alg.optimizer,counters['expected_adam_steps'])
    raw.common_step_counter = counters['common_step_counter']
    raw._sim_step_counter = counters['sim_step_counter']
    raw.episode_length_buf.zero_()
    raw.reset(seed=SEED)
    raw._basketball_state.level.fill_(5)
    raw._basketball_state.hold.zero_()
    term = raw.action_manager.get_term('ball_hold')
    term.process_actions(torch.empty(raw.num_envs,0,device=raw.device))
    term.apply_actions()
    runner.alg.actor.reset()
    if hasattr(runner.alg.critic,'reset'):
        runner.alg.critic.reset()
    runner.current_learning_iteration = counters['next_iteration']
    runner.alg.learning_rate = float(state['learning_rate'])
    for group in runner.alg.optimizer.param_groups:
        group['lr'] = runner.alg.learning_rate
    require(raw.common_step_counter==counters['common_step_counter'] and
            raw._sim_step_counter==counters['sim_step_counter'] and not raw.episode_length_buf.any(),
            'Reset changed progress')
    check_observations(env.get_observations(),NUM_ENVS)
    assert_zero_assistance(raw)
    runtime_profile(raw,profile)
    return completed


def save(runner, env, output, completed, registered_save, *, profile, campaign_id, head):
    raw = env.unwrapped
    counters = progress(completed)
    require(raw.common_step_counter==counters['common_step_counter'] and
            raw._sim_step_counter==counters['sim_step_counter'],'Save progress mismatch')
    check_optimizer(runner.alg.optimizer,counters['expected_adam_steps'])
    assert_zero_assistance(raw)
    lr = float(runner.alg.learning_rate)
    require(all(float(g['lr'])==lr for g in runner.alg.optimizer.param_groups),'Runtime LR mismatch')
    state = dict(schema_version=1,task_id=TASK_ID,baseline_commit=BASELINE_COMMIT,source_sha256=PRIMARY_SHA,
                 git_head=head,purpose='formal',campaign_id=campaign_id,profile=profile,
                 profile_record=profile_record(profile),seed=SEED,num_envs=NUM_ENVS,**counters,
                 learning_rate=lr,hold_levels=list(HOLD_LEVELS),
                 level=raw._basketball_state.level.detach().cpu().clone(),
                 hold=raw._basketball_state.hold.detach().cpu().clone(),exact_trajectory_resume=False,
                 resume_rule='reset episode/RNG/RNN/delay; restore models/normalizers/Adam/LR/progress; zero assistance',
                 live_profile=runtime_profile(raw,profile))
    folder = Path(output)/'checkpoints';folder.mkdir(exist_ok=True)
    path = folder/f'update_{completed:06d}.pt'
    if path.exists():
        old,_,_ = load_checked(path,profile=profile,campaign_id=campaign_id)
        for key,value in runner.alg.save().items():
            require_equal(old[key],value,'existing saved '+key)
        return path
    temporary = path.with_suffix('.tmp.pt')
    index = runner.current_learning_iteration
    try:
        runner.current_learning_iteration = counters['last_completed_iteration']
        registered_save(str(temporary),infos={'stage09':state})
    finally:
        runner.current_learning_iteration = index
    saved = torch.load(temporary,map_location='cpu',weights_only=False)
    validate_checkpoint(saved,profile=profile,campaign_id=campaign_id)
    for key,value in runner.alg.save().items():
        require_equal(saved[key],value,'saved '+key)
    temporary.replace(path)
    atomic_json(path.with_suffix('.json'),dict(sha256=file_sha256(path),size_bytes=path.stat().st_size,
                experiment_completed_updates=completed,profile=profile,campaign_id=campaign_id,git_head=head))
    return path
