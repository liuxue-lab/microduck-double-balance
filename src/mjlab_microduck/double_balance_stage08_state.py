"""Stage 08 checkpoint identity, strict restoration and runtime invariants."""
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import subprocess

import torch

from mjlab_microduck.double_balance_checkpoint import file_sha256
from mjlab_microduck.double_balance_smoke import (
    TASK_ID, check_loaded_runner, check_observations, check_optimizer,
    require, require_equal, require_finite, resume_agent_config,
)
from mjlab_microduck.double_balance_stage07 import HOLD_LEVELS, verify_frozen_source
from mjlab_microduck.double_balance_training import VERSIONS, validate_training_checkpoint
from mjlab_microduck.double_balance_stage08_budget import atomic_json
from mjlab_microduck.double_balance_stage08_plan import (
    BASELINE_COMMIT, INITIAL_LR, NUM_ENVS, PROFILES, SOURCE_SHA256,
    TRAIN_SEEDS, apply_training_profile, progress,
)

ROOT = Path('/root/autodl-tmp/microduck-double-balance')
ARTIFACTS = ROOT / 'artifacts/double-balance-stage08'
FINAL_STAGE07_SHA256 = 'a5ad0aedb500555b649c5d4b11aded833cbc0f932c1fff1a747e1492534c495c'


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def verify_source(repo, *, clean=True):
    head = verify_frozen_source(repo)
    subprocess.run(['git', 'merge-base', '--is-ancestor', BASELINE_COMMIT, head], cwd=repo, check=True)
    frozen = ('src/mjlab_microduck/tasks', 'src/mjlab_microduck/robot',
              'src/mjlab_microduck/actuator', 'pyproject.toml', 'uv.lock',
              'src/mjlab_microduck/double_balance_training.py',
              'src/mjlab_microduck/double_balance_stage07.py',
              'src/mjlab_microduck/double_balance_smoke.py',
              'docs/audits/stage-07-cloud-capacity-summary.json')
    changes = subprocess.check_output(['git', 'diff', '--name-only', BASELINE_COMMIT, '--', *frozen], cwd=repo, text=True)
    require(not changes.strip(), 'Stage 03/04 or reused Stage 07 runtime changed')
    if clean:
        require(not subprocess.check_output(['git', 'status', '--porcelain'], cwd=repo, text=True).strip(),
                'Use a clean committed source tree')
    return head


def environment_preflight(gpu, *, cloud=True):
    require(not any(k.startswith('MICRODUCK_') for k in os.environ), 'Unset task environment overrides')
    require(int(os.environ.get('WORLD_SIZE', '1')) == 1, 'Single GPU only')
    require({k: version(k) for k in VERSIONS} == VERSIONS, 'Locked simulation dependencies differ')
    require(torch.__version__.split('+')[0] == '2.9.1', 'Expected Torch 2.9.1')
    if cloud:
        require(ROOT.is_dir(), 'Formal PPO is restricted to the cloud data directory')
        require(torch.cuda.is_available() and torch.cuda.device_count() == 1, 'One CUDA GPU required')
        name = torch.cuda.get_device_name(0)
        total = torch.cuda.get_device_properties(0).total_memory
        require((gpu == 'A800' and 'A800' in name and total >= 75 * 1024**3) or
                (gpu == '5090' and name == 'NVIDIA GeForce RTX 5090' and total >= 30 * 1024**3),
                'GPU does not match the selected cloud budget')
        require(torch.version.cuda == '12.8', 'Expected locked CUDA 12.8 Torch build')
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    from mjlab.utils.torch import configure_torch_backends
    configure_torch_backends()
    return verify_source(Path(__file__).resolve().parents[2])


def profile_record(name):
    return {'schema_version': 1, **asdict(PROFILES[name])}


def validate_assistance(state):
    require(state['num_envs'] == NUM_ENVS, 'Cannot remap source assistance to another batch size')
    require(tuple(state['hold_levels']) == HOLD_LEVELS, 'Assistance table mismatch')
    level, hold = state['level'], state['hold']
    require(level.dtype == torch.long and tuple(level.shape) == (NUM_ENVS,), 'Invalid assistance levels')
    require(bool(((level >= 0) & (level < len(HOLD_LEVELS))).all()), 'Assistance index out of range')
    require(hold.dtype == torch.float32 and hold.shape == level.shape, 'Invalid hold shape/type')
    require_equal(hold.cpu(), torch.tensor(HOLD_LEVELS)[level.cpu()], 'assistance mapping')


def validate_stage08_checkpoint(checkpoint, *, profile=None, seed=None, campaign_id=None):
    state = checkpoint['infos']['stage08']
    require(state['schema_version'] == 1 and state['task_id'] == TASK_ID, 'Unknown Stage 08 checkpoint')
    require(state['baseline_commit'] == BASELINE_COMMIT and state['source_sha256'] == SOURCE_SHA256,
            'Checkpoint source lineage differs')
    require(state['purpose'] in ('formal', 'capacity'), 'Unknown checkpoint purpose')
    require(state['profile'] in PROFILES and state['profile_record'] == profile_record(state['profile']),
            'Checkpoint profile differs')
    require(state['seed'] in TRAIN_SEEDS, 'Unexpected experiment seed')
    if profile is not None:
        require(state['profile'] == profile, 'Cannot resume into another experimental treatment')
    if seed is not None:
        require(state['seed'] == seed, 'Cannot resume into another seed')
    if campaign_id is not None:
        require(state['campaign_id'] == campaign_id, 'Resume cannot reset the campaign budget')
    expected = progress(state['experiment_completed_updates'])
    for key, value in expected.items():
        require(state[key] == value, f'Invalid progress: {key}')
    require(checkpoint['iter'] == expected['last_completed_iteration'], 'Saved iteration differs')
    require(checkpoint['infos']['env_state']['common_step_counter'] == expected['common_step_counter'],
            'Saved curriculum counter differs')
    validate_assistance(state)
    if state['profile'] == 'E':
        require(bool((state['level'] == 5).all() and (state['hold'] == 0).all()), 'E contains assistance')
    lr = state['learning_rate']
    require(math.isfinite(lr) and lr > 0, 'Invalid saved LR')
    groups = checkpoint['optimizer_state_dict']['param_groups']
    require(len(groups) == 1 and groups[0]['lr'] == lr, 'Adam and scheduler LR mismatch')
    adam = checkpoint['optimizer_state_dict']['state']
    require(len(adam) == len(groups[0]['params']) and bool(adam), 'Adam moments are missing')
    for entry in adam.values():
        require(int(entry['step'].item()) == expected['expected_adam_steps'], 'Adam step count differs')
        require('exp_avg' in entry and 'exp_avg_sq' in entry, 'Incomplete Adam state')
    require_finite(checkpoint, 'checkpoint')
    return state


def load_checked(path, *, resume=False, evaluation=False, profile=None, seed=None, campaign_id=None):
    path = Path(path).resolve()
    checksum = file_sha256(path)
    if not resume:
        allowed = (SOURCE_SHA256, FINAL_STAGE07_SHA256) if evaluation else (SOURCE_SHA256,)
        require(checksum in allowed, 'Fresh training must use the archived update 1000 checkpoint')
    else:
        receipt = json.loads(path.with_suffix('.json').read_text())
        require(receipt['sha256'] == checksum, 'Checkpoint receipt SHA mismatch')
    # Only project-generated, hash-checked checkpoints reach pickle loading.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if resume:
        state = validate_stage08_checkpoint(checkpoint, profile=profile, seed=seed, campaign_id=campaign_id)
        require(evaluation or state['purpose'] == 'formal', 'Capacity optimization is never a formal initializer')
    else:
        state = validate_training_checkpoint(checkpoint, NUM_ENVS)
        if not evaluation:
            require(state['completed_updates'] == 1000 and state['learning_rate'] == INITIAL_LR,
                    'Unexpected source update/LR')
    return checkpoint, state, checksum


def runtime_profile(raw, name):
    """Inspect live manager cfgs, not the pre-construction env.cfg copy."""
    profile = PROFILES[name]
    curricula = set(raw.curriculum_manager.active_terms)
    twist = raw.command_manager.get_term('twist')
    command = raw.command_manager.get_command('twist')
    report = {'curricula': sorted(curricula),
              'zero_command_fraction': float((command == 0).all(dim=1).float().mean()),
              'action_rate_weight': raw.reward_manager.get_term_cfg('action_rate_l2').weight,
              'com_range': list(raw.event_manager.get_term_cfg('randomize_com').params['ranges']),
              'head_com_range': list(raw.event_manager.get_term_cfg('randomize_head_com').params['ranges']),
              'standing_fraction_setting': twist.cfg.rel_standing_envs}
    if profile.zero_command:
        require(not bool(command.any()), 'Nonzero twist in a zero-command group')
        require(twist.cfg.rel_standing_envs == 1 and twist.cfg.rel_turn_in_place_envs == 0,
                'Command bucket override lost')
        require(all(getattr(twist.cfg.ranges, k) == (0., 0.) for k in ('lin_vel_x','lin_vel_y','ang_vel_z')),
                'Command ranges were overwritten')
        require('standing_envs' not in curricula, 'Standing curriculum is still active')
    if profile.fixed_action_rate:
        require(report['action_rate_weight'] == -.4 and 'action_rate_weight' not in curricula,
                'Fixed action penalty was overwritten')
    if profile.fixed_com:
        require(report['com_range'] == report['head_com_range'] == [-.005, .005], 'Fixed COM range changed')
        require(not {'com_range', 'head_com_range'} & curricula, 'COM curriculum still active')
    if profile.zero_assistance:
        require('basketball_hold' not in curricula, 'Assistance curriculum still active')
        assert_zero_assistance(raw)
    return report


def assert_zero_assistance(raw):
    require(not bool(raw._basketball_state.hold.any()), 'Assistance hold is nonzero')
    term = raw.action_manager.get_term('ball_hold')
    for key in ('_force', '_torque', '_duck_force', '_duck_torque'):
        value = getattr(term, key, None)
        require(value is not None and not bool(value.any()), f'Residual assistance cache: {key}')
    for name in ('ball', 'robot'):
        body_id = int(raw.scene[name].indexing.body_ids[0])
        require(not bool(raw.sim.data.xfrc_applied[:, body_id].any()), f'Residual applied wrench: {name}')


def check_training_step(raw, name):
    """One device synchronization per control step for command/wrench guards."""
    profile=PROFILES[name]
    flags=[]
    if profile.zero_command:
        flags.append(~raw.command_manager.get_command('twist').any())
    if profile.zero_assistance:
        flags.append(~raw._basketball_state.hold.any())
        term=raw.action_manager.get_term('ball_hold')
        for key in ('_force','_torque','_duck_force','_duck_torque'):
            flags.append(~getattr(term,key).any())
        for entity in (term._entity,term._duck):
            flags.append(~raw.sim.data.xfrc_applied[:,entity.indexing.body_ids[0]].any())
    if flags:
        require(bool(torch.stack(flags).all()), 'Nonzero command or assistance in a constrained group')


def restore(runner, env, checkpoint, *, profile, seed, campaign_id, resume):
    raw = env.unwrapped
    check_loaded_runner(runner, raw, checkpoint)
    if resume:
        state = validate_stage08_checkpoint(checkpoint, profile=profile, seed=seed, campaign_id=campaign_id)
        completed = state['experiment_completed_updates']
    else:
        state = validate_training_checkpoint(checkpoint, NUM_ENVS)
        require(state['completed_updates'] == 1000, 'Stage 08 must branch from update 1000')
        completed = 0
    counters = progress(completed)
    check_optimizer(runner.alg.optimizer, counters['expected_adam_steps'])
    raw.common_step_counter = counters['common_step_counter']
    raw._sim_step_counter = counters['sim_step_counter']
    raw.episode_length_buf.zero_()
    env.reset()
    if profile == 'E':
        raw._basketball_state.level.fill_(5)
        raw._basketball_state.hold.zero_()
    else:
        raw._basketball_state.level.copy_(state['level'].to(raw.device))
        raw._basketball_state.hold.copy_(state['hold'].to(raw.device))
    # BallHold.reset is a no-op. Recompute and overwrite BOTH cached/applied wrenches.
    term = raw.action_manager.get_term('ball_hold')
    term.process_actions(torch.empty(raw.num_envs, 0, device=raw.device))
    term.apply_actions()
    runner.alg.actor.reset()
    if hasattr(runner.alg.critic, 'reset'):
        runner.alg.critic.reset()
    runner.current_learning_iteration = counters['next_iteration']
    runner.alg.learning_rate = float(state['learning_rate'])
    for group in runner.alg.optimizer.param_groups:
        group['lr'] = runner.alg.learning_rate
    require(raw.common_step_counter == counters['common_step_counter'] and not raw.episode_length_buf.any(),
            'Reset changed global progress or fabricated episode age')
    check_observations(env.get_observations(), NUM_ENVS)
    runtime_profile(raw, profile)
    return completed


def metadata(runner, raw, completed, *, profile, seed, campaign_id, head, purpose):
    counters = progress(completed)
    require(raw.common_step_counter == counters['common_step_counter'] and
            raw._sim_step_counter == counters['sim_step_counter'], 'Checkpoint counters differ')
    lr = float(runner.alg.learning_rate)
    require(all(float(g['lr']) == lr for g in runner.alg.optimizer.param_groups), 'Runtime LR mismatch')
    check_optimizer(runner.alg.optimizer, counters['expected_adam_steps'])
    return {'schema_version': 1, 'task_id': TASK_ID, 'baseline_commit': BASELINE_COMMIT,
            'source_sha256': SOURCE_SHA256, 'git_head': head, 'purpose': purpose,
            'campaign_id': campaign_id, 'profile': profile, 'profile_record': profile_record(profile),
            'seed': seed, 'num_envs': NUM_ENVS, **counters, 'learning_rate': lr,
            'hold_levels': list(HOLD_LEVELS), 'level': raw._basketball_state.level.detach().cpu().clone(),
            'hold': raw._basketball_state.hold.detach().cpu().clone(),
            'exact_trajectory_resume': False, 'resume_rule': 'new episodes/RNG/hidden state; retain optimizer/LR/progress/assistance',
            'live_profile': runtime_profile(raw, profile)}


def save(runner, env, output, completed, registered_save, **identity):
    folder = Path(output) / 'checkpoints'
    folder.mkdir(exist_ok=True)
    path = folder / f'update_{completed:06d}.pt'
    if path.exists():
        saved, state, _ = load_checked(path, resume=True, evaluation=True)
        require(state['experiment_completed_updates'] == completed, 'Existing checkpoint progress differs')
        for key in ('profile', 'seed', 'campaign_id', 'purpose'):
            require(state[key] == identity[key], f'Existing checkpoint {key} differs')
        return path
    state = metadata(runner, env.unwrapped, completed, **identity)
    temporary = path.with_suffix('.tmp.pt')
    old_index = runner.current_learning_iteration
    try:
        runner.current_learning_iteration = state['last_completed_iteration']
        registered_save(str(temporary), infos={'stage08': state})
    finally:
        runner.current_learning_iteration = old_index
    saved = torch.load(temporary, map_location='cpu', weights_only=False)
    validate_stage08_checkpoint(saved)
    for key, value in runner.alg.save().items():
        require_equal(saved[key], value, 'saved.' + key)
    temporary.replace(path)
    atomic_json(path.with_suffix('.json'), {'sha256': file_sha256(path), 'size_bytes': path.stat().st_size,
                'experiment_completed_updates': completed, 'git_head': identity['head'], 'created_utc': utc_now()})
    link = Path(output) / 'latest.pt.tmp'
    link.symlink_to(path.relative_to(output))
    link.replace(Path(output) / 'latest.pt')
    return path


def build_training_config(profile, seed, checkpoint):
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
    import mjlab_microduck.tasks  # noqa: F401
    require(seed in TRAIN_SEEDS, 'Use the preregistered experiment seeds')
    cfg = load_env_cfg(TASK_ID)
    cfg.scene.num_envs, cfg.seed = NUM_ENVS, seed
    cfg = apply_training_profile(cfg, profile)
    agent = resume_agent_config(asdict(load_rl_cfg(TASK_ID)), checkpoint)
    agent.update(seed=seed, logger='tensorboard', upload_model=False, check_for_nan=True, save_interval=100)
    require(agent['num_steps_per_env'] == 24 and agent['algorithm']['num_learning_epochs'] *
            agent['algorithm']['num_mini_batches'] == 20, 'PPO batch/epoch contract changed')
    require(agent['algorithm']['schedule'] == 'adaptive' and agent['algorithm']['desired_kl'] == .01,
            'LR schedule changed')
    return cfg, agent
