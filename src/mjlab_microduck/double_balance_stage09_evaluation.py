"""Frozen nominal evaluation and separately identified randomized initial states."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import inspect
import json
import math
from pathlib import Path
import statistics

import torch

from mjlab_microduck.double_balance_stage08_budget import atomic_json
from mjlab_microduck.double_balance_stage08_state import (
    BASELINE_COMMIT, TASK_ID, check_loaded_runner, file_sha256,
    require, require_equal, require_finite, resume_agent_config, utc_now,
)
from mjlab_microduck.double_balance_training import terminal_episodes
from mjlab_microduck.double_balance_stage09_state import load_checked

PROTOCOL_VERSION = 'stage08-first-episode-metric-phase-v1'
PROTOCOLS = {'nominal': (20260921, 64), 'dev': (8101, 128),
             'test-8201': (8201, 256), 'test-8202': (8202, 256), 'test-8203': (8203, 256),
             'test-930901':(930901,256),'test-930902':(930902,256),'test-930903':(930903,256)}
GATES = ('top_center', 'top_height', 'top_speed', 'lower_offset', 'lower_height_min',
         'lower_height_max', 'robot_tilt', 'lower_ball_speed', 'unassisted')
VALUES = ('top_center_m', 'top_height_error_m', 'top_speed_m_s', 'lower_offset_m',
          'lower_height_m', 'robot_tilt_deg', 'lower_ball_speed_m_s', 'hold', 'action_delta_l2')


def evaluation_config(protocol):
    from mjlab.tasks.registry import load_env_cfg
    import mjlab_microduck.tasks  # noqa: F401
    seed, count = PROTOCOLS[protocol]
    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = count, seed, False
    require(cfg.episode_length_s == 10 and not cfg.curriculum, 'Frozen play definition changed')
    require(tuple(cfg.actions['ball_hold'].levels) == (0.,), 'Evaluation assistance differs')
    if protocol != 'nominal':
        # Only this auxiliary protocol changes initial-state sampling. No DR/pushes.
        cfg.events['reset_double_balance'].params.update(
            xy_noise=.01, yaw_range=(-math.pi, math.pi), tilt_noise_deg=2., joint_noise=.05,
            ball_vel_noise=0., top_ball_xy_noise=0.)
    return cfg


def seal_initial_states(raw, protocol, directory):
    """Persist actual states once; subsequent policies must reproduce them bitwise.

    Reset uses an explicit model-independent seed immediately beforehand. We
    fail instead of silently accepting a new set after an RNG/backend change.
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{protocol}.pt'
    states = {}
    for name in ('robot', 'ball', 'top_ball'):
        entity = raw.scene[name]
        pose = entity.data.root_link_pose_w.detach().cpu().clone()
        pose[:, :3] -= raw.scene.env_origins.detach().cpu()
        states[name] = {'pose_relative_to_origin': pose,
                        'velocity': entity.data.root_link_vel_w.detach().cpu().clone(),
                        'joint_pos': entity.data.joint_pos.detach().cpu().clone(),
                        'joint_vel': entity.data.joint_vel.detach().cpu().clone()}
    actual = {'schema_version': 1, 'protocol_version': PROTOCOL_VERSION,
              'baseline_commit': BASELINE_COMMIT, 'protocol': protocol,
              'seed': PROTOCOLS[protocol][0], 'count': raw.num_envs, 'states': states}
    ids = []
    for i in range(raw.num_envs):
        digest = hashlib.sha256()
        for fields in states.values():
            for tensor in fields.values():
                digest.update(tensor[i].contiguous().numpy().tobytes())
        ids.append(digest.hexdigest())
    actual['state_ids'] = ids
    require_finite(actual, 'initial states')
    if path.exists():
        receipt = json.loads(path.with_suffix('.json').read_text())
        require(receipt['sha256'] == file_sha256(path), 'Initial-state set checksum mismatch')
        expected = torch.load(path, map_location='cpu', weights_only=True)
        require_equal(actual, expected, 'persisted initial states')
    else:
        if protocol != 'nominal':
            require(len(set(ids)) == len(ids), 'Random initial-state set contains duplicates')
            for other in directory.glob('*.json'):
                data = json.loads(other.read_text())
                if data.get('protocol') != 'nominal':
                    require(not set(ids) & set(data['state_ids']), 'Development/test initial states overlap')
        temporary = path.with_suffix('.tmp.pt')
        torch.save(actual, temporary)
        temporary.replace(path)
        atomic_json(path.with_suffix('.json'), {'sha256': file_sha256(path), 'protocol': protocol,
                    'protocol_version': PROTOCOL_VERSION, 'seed': actual['seed'],
                    'count': raw.num_envs, 'state_ids': ids})
    return {'path': str(path), 'sha256': file_sha256(path), 'state_ids': ids}


def wilson_interval(successes, count):
    require(0 <= successes <= count and count > 0, 'Invalid binomial count')
    p, z = successes / count, 1.959963984540054
    denominator = 1 + z*z/count
    center = (p + z*z/(2*count))/denominator
    radius = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count))/denominator
    return [max(0., center-radius), min(1., center+radius)]


def validate_selection(path, checkpoint_sha):
    require(path is not None, 'Held-out evaluation requires a frozen selection file')
    selection = json.loads(Path(path).read_text())
    require(selection.get('schema_version') == 1 and selection.get('selection_locked') is True,
            'Selection is not frozen')
    require(selection['protocol_version'] == PROTOCOL_VERSION, 'Selection protocol differs')
    require(checkpoint_sha in [x['sha256'] for x in selection['checkpoints']], 'Model is absent from final selection')
    require(bool(selection['development_reports']), 'Final selection has no development evidence')
    for item in selection['development_reports']:
        require(file_sha256(item['path']) == item['sha256'], 'Selection evidence changed')
        report = json.loads(Path(item['path']).read_text())
        require(report['status'] == 'PASS' and report['protocol'] in ('nominal', 'dev'),
                'Held-out reports cannot select models')
    return file_sha256(path)


class MetricObserver:
    """Observe exactly after the original compute; never call success again."""
    def __init__(self, raw, alive):
        from mjlab_microduck.tasks import mdp
        self.raw, self.alive, self.mdp = raw, alive, mdp
        self.original = raw.metrics_manager.compute
        self.defaults = {k: p.default for k, p in inspect.signature(mdp.double_balance_stable_from_values).parameters.items()
                         if p.default is not inspect.Parameter.empty}
        names = raw.metrics_manager.active_terms
        self.stable_col, self.success_col = names.index('double_balance_stable_fraction'), names.index('double_balance_success')
        self.rows, self.active = [], []

    def compute(self):
        self.original()
        raw, mdp, d = self.raw, self.mdp, self.defaults
        position, velocity, _ = mdp._double_balance_top_ball_kinematics(raw)
        root = raw.scene['robot'].data.root_link_pos_w
        lower_pos = mdp._bb_ball_pos(raw)
        center = position[:, :2].norm(dim=-1)
        height_error = (position[:, 2] - d['ball_radius']).abs()
        top_speed = velocity.norm(dim=-1)
        offset = (root[:, :2] - lower_pos[:, :2]).norm(dim=-1)
        height = root[:, 2] - lower_pos[:, 2]
        tilt = mdp.wrestle_tilt(raw, 'robot')
        lower_speed = raw.scene['ball'].data.root_link_lin_vel_w[:, :2].norm(dim=-1)
        hold = raw._basketball_state.hold
        delta = (raw.action_manager.action - raw.action_manager.prev_action).square().sum(dim=-1)
        gates = torch.stack((center <= d['top_center_radius'], height_error <= d['top_height_tolerance'],
                             top_speed <= d['top_speed_limit'], offset <= d['lower_offset_limit'],
                             height >= d['lower_height_min'], height <= d['lower_height_max'],
                             tilt <= math.radians(d['robot_tilt_limit_deg']),
                             lower_speed <= d['lower_ball_speed_limit'], hold <= 1e-6), dim=1)
        cached = raw.metrics_manager._step_values
        stable, success = cached[:, self.stable_col].bool(), cached[:, self.success_col].bool()
        require(torch.equal(gates.all(dim=1)[self.alive], stable[self.alive]), 'Diagnostic gates differ from frozen metric')
        values = torch.stack((center, height_error, top_speed, offset, height, torch.rad2deg(tilt), lower_speed, hold, delta), dim=1)
        require_finite(values[self.alive], 'metric-phase values')
        row = torch.cat((values, gates.float(), stable[:,None].float(), success[:,None].float(),
                         raw._double_balance_stable_time[:,None]), dim=1)
        self.rows.append(row.detach().cpu())
        self.active.append(self.alive.detach().cpu().clone())

    def finish(self, episodes, state_ids, dt):
        trace, active = torch.stack(self.rows), torch.stack(self.active)
        start = len(VALUES)
        for episode in episodes:
            i = episode['env_id']
            rows = trace[active[:,i], i]
            flags = rows[:, start + len(GATES)].bool().tolist()
            current = longest = 0
            for flag in flags:
                current = current + 1 if flag else 0
                longest = max(longest, current)
            failures = ~rows[:, start:start+len(GATES)].bool()
            speed = rows[:, VALUES.index('lower_ball_speed_m_s')]
            episode.update(initial_state_id=state_ids[i], terminal_stable_seconds=current*dt,
                           longest_stable_seconds=longest*dt, official_timer_seconds=float(rows[-1,-1]),
                           lower_ball_speed_quantiles_m_s=dict(zip(('p50','p95','p99'),
                               torch.quantile(speed, torch.tensor([.5,.95,.99])).tolist())),
                           lower_ball_speed_max_m_s=float(speed.max()),
                           lower_speed_violation_fraction=float(failures[:,GATES.index('lower_ball_speed')].float().mean()),
                           violation_fractions=dict(zip(GATES, failures.float().mean(dim=0).tolist())),
                           co_violation_counts=(failures.long().T @ failures.long()).tolist(),
                           mean_action_delta_l2=float(rows[:,VALUES.index('action_delta_l2')].mean()))
        return {'columns': [*VALUES, *['pass_'+g for g in GATES], 'cached_stable', 'cached_success', 'official_timer_s'],
                'values': trace, 'active_first_episode': active, 'dt': dt}


def summarize(episodes, protocol):
    require(bool(episodes), 'No episodes to summarize')
    count, successes = len(episodes), sum(e['success'] for e in episodes)
    mean = lambda k: statistics.mean(e[k] for e in episodes)
    return {'episodes_count': count, 'successes': successes, 'success_fraction': successes/count,
            'success_wilson95': None if protocol == 'nominal' else wilson_interval(successes,count),
            'mean_survival_seconds': mean('seconds'), 'mean_stable_fraction': mean('double_balance_stable_fraction'),
            'median_terminal_stable_seconds': statistics.median(e['terminal_stable_seconds'] for e in episodes),
            'median_longest_stable_seconds': statistics.median(e['longest_stable_seconds'] for e in episodes),
            'mean_lower_speed_violation_fraction': mean('lower_speed_violation_fraction'),
            'mean_action_delta_l2': mean('mean_action_delta_l2'),
            'termination_fraction': mean('terminated')}


def selection_rank(nominal, development, completed):
    for report, protocol in ((nominal,'nominal'), (development,'dev')):
        require(report['status'] == 'PASS' and report['protocol'] == protocol and
                report['protocol_version'] == PROTOCOL_VERSION, 'Invalid selection protocol')
    require(nominal['checkpoint_sha256'] == development['checkpoint_sha256'], 'Selection reports use different models')
    return (development['success_fraction'], nominal['success_fraction'], development['mean_survival_seconds'],
            development['median_terminal_stable_seconds'], development['median_longest_stable_seconds'],
            -development['mean_lower_speed_violation_fraction'], -completed)


def evaluate_one(checkpoint_path, output, protocol, datasets, *, device, head, selection=None, video_env_id=None, reward_probe=False):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
    from mjlab.utils.os import dump_yaml
    import mjlab_microduck.tasks  # noqa: F401
    checkpoint_path, output = Path(checkpoint_path), Path(output)
    checksum = file_sha256(checkpoint_path)
    checkpoint, state, checksum = load_checked(checkpoint_path, evaluation=True)
    selection_sha = validate_selection(selection, checksum) if protocol.startswith('test-') else None
    output.mkdir(parents=True, exist_ok=False)
    cfg = evaluation_config(protocol)
    if reward_probe:
        require(protocol=='nominal','Reward probe is not an official evaluation')
        from mjlab.managers import RewardTermCfg
        from mjlab_microduck.tasks import mdp
        cfg.rewards['stage09_head_yaw']=RewardTermCfg(func=mdp.stage09_head_yaw_cost,weight=-.25)
        cfg.rewards['stage09_head_margin']=RewardTermCfg(func=mdp.stage09_head_margin_cost,weight=-.25)
    if video_env_id is not None:
        require(0 <= video_env_id < cfg.scene.num_envs, 'Video environment index out of range')
        from mjlab_microduck.video_effects import configure_video_cfg
        configure_video_cfg(cfg,width=1280,height=720)
        cfg.viewer.distance,cfg.viewer.azimuth,cfg.viewer.elevation=.85,135,-20
        cfg.viewer.env_idx=video_env_id
    agent = resume_agent_config(asdict(load_rl_cfg(TASK_ID)), checkpoint)
    agent.update(logger='tensorboard', upload_model=False)
    dump_yaml(output / 'params/env.yaml', asdict(cfg))
    report = {'status':'RUNNING', 'protocol':protocol, 'protocol_version':PROTOCOL_VERSION,
              'checkpoint':str(checkpoint_path), 'checkpoint_sha256':checksum, 'git_head':head,
              'profile':state.get('profile','stage07-reference'), 'seed':state['seed'],
              'campaign_id':state.get('campaign_id'),
              'experiment_completed_updates':state.get('experiment_completed_updates'),
              'lineage_completed_updates':state.get('lineage_completed_updates',state.get('completed_updates')),
              'selection_sha256':selection_sha, 'created_utc':utc_now(), 'device':device,
              'num_envs':cfg.scene.num_envs,'horizon_seconds':10.,'assistance':0.,
              'video_env_id':video_env_id,'video_review':'NOT_PERFORMED',
              'success_definition':'unchanged Stage 04 cached terminal success; physical termination is failure',
              'sampling_scope':'deterministic nominal copies' if protocol == 'nominal' else 'random reset states; no physics DR or pushes',
              'policy_objective_status':'NOT_ASSESSED','reward_probe_only':reward_probe}
    env = raw = observer = writer = None
    try:
        raw = ManagerBasedRlEnv(cfg=cfg, device=device,render_mode='rgb_array' if video_env_id is not None else None)
        env = RslRlVecEnvWrapper(raw, clip_actions=agent['clip_actions'])
        runner = load_runner_cls(TASK_ID)(env, deepcopy(agent), None, device)
        runner.load(str(checkpoint_path), strict=True, map_location=device)
        check_loaded_runner(runner, raw, checkpoint)
        policy = runner.get_inference_policy()
        raw.common_step_counter = raw._sim_step_counter = 0
        raw.episode_length_buf.zero_()
        raw.reset(seed=PROTOCOLS[protocol][0])
        policy.reset()
        initial = seal_initial_states(raw, protocol, datasets)
        report['initial_states'] = {k:v for k,v in initial.items() if k != 'state_ids'}
        obs = env.get_observations()
        alive = torch.ones(raw.num_envs, dtype=torch.bool, device=device)
        from mjlab_microduck.double_balance_stage09_diagnostics import observer_class,kinematic_calibration
        provenance=dict(protocol=protocol,checkpoint_sha256=checksum,source_head=head,new_ppo_updates=0)
        observer_type=observer_class(MetricObserver,output,provenance,
                                    kinematic_calibration(Path(__file__).resolve().parents[2]))
        observer = observer_type(raw, alive)
        report['thresholds']=observer.defaults
        raw.metrics_manager.compute = observer.compute
        if video_env_id is not None:
            import imageio.v2 as imageio
            writer=imageio.get_writer(output/f'env-{video_env_id}.mp4',fps=25,codec='libx264',quality=7,macro_block_size=16)
        episodes = []
        with torch.inference_mode():
            for step in range(1,raw.max_episode_length + 2):
                film=video_env_id is not None and bool(alive[video_env_id])
                require(not bool(raw.command_manager.get_command('twist').any()), 'Evaluation command is nonzero')
                require(not bool(raw._basketball_state.hold.any()), 'Evaluation assistance is nonzero')
                actions = policy(obs)
                obs, rewards, dones, _ = env.step(actions)
                require_finite((actions,obs,rewards), 'evaluation rollout')
                require(not bool((raw.termination_manager.get_term('nan_state') & alive).any()), 'NaN in evaluation')
                if film and (step%2==0 or bool(dones[video_env_id])):
                    from PIL import Image,ImageDraw,ImageFont
                    import numpy as np
                    frame=Image.fromarray(raw.render())
                    draw=ImageDraw.Draw(frame)
                    values=observer.rows[-1][video_env_id]
                    text=(f'env={video_env_id}  t={step*.02:.2f}s\n'
                          f'lower speed={values[6]:.4f} m/s  limit=0.1500\n'
                          f'top offset={values[0]*1000:.2f} mm  limit=12.00\n'
                          f'continuous stable={values[-1]:.2f}s  hold={values[7]:.1f}')
                    draw.rectangle((8,8,570,125),fill=(0,0,0))
                    draw.multiline_text((18,15),text,fill='white',font=ImageFont.load_default(size=22),spacing=4)
                    writer.append_data(np.asarray(frame))
                finished, ended = terminal_episodes(raw, alive, dones)
                episodes.extend(finished)
                alive &= ~ended
                if not bool(alive.any()):
                    break
                ids = dones.nonzero(as_tuple=False).flatten()
                if len(ids):
                    raw.reset(env_ids=ids)
                    policy.reset(dones.bool())
                    obs = env.get_observations()
        require(len(episodes) == raw.num_envs, 'Some first episodes did not finish')
        episodes.sort(key=lambda e:e['env_id'])
        trace = observer.finish(episodes, initial['state_ids'], raw.step_dt)
        torch.save(trace, output / 'trace.pt')
        report['posture_summary']=json.loads((output/'posture-summary.json').read_text())
        report.update(status='PASS', episodes=episodes, **summarize(episodes, protocol),
                      trace_sha256=file_sha256(output/'trace.pt'),
                      policy_objective_status='NO_STRICT_SUCCESSES' if not any(e['success'] for e in episodes)
                                              else 'STRICT_SUCCESSES_OBSERVED_REVIEW_REQUIRED')
    except BaseException as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report['finished_utc'] = utc_now()
        atomic_json(output/'evaluation.json', report)
        if observer is not None:
            raw.metrics_manager.compute = observer.original
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        elif raw is not None:
            raw.close()
    return report
