#!/usr/bin/env python3
"""Short local MuJoCo inference and video for the two archived Stage 07 policies.

Run using the existing project .venv. No training, optimizer update, cloud access,
task edits, assistance, or success-threshold changes are performed.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
from importlib.metadata import version
import inspect
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import tempfile

REPO = Path('/home/lx/microduck-double-balance/workspace')
ARTIFACTS = REPO.parent / 'artifacts/double-balance-stage07'
EXPECTED_MODELS = {
    'best_nominal': 'c667f96607b68383047f23956ba58805920434465245ef8e32d1148b17fc65b7',
    'final_state': 'a5ad0aedb500555b649c5d4b11aded833cbc0f932c1fff1a747e1492534c495c',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def streaks(flags):
    current = longest = 0
    for stable in flags:
        current = current + 1 if stable else 0
        longest = max(longest, current)
    return longest, current


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def worker(args):
    sys.path.insert(0, str(REPO / 'src'))
    import numpy as np
    import torch
    import imageio.v2 as imageio
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    import mjlab_microduck.tasks
    from mjlab_microduck.tasks import mdp
    from mjlab_microduck.double_balance_training import (
        VERSIONS, validate_training_checkpoint, terminal_episodes,
    )
    from mjlab_microduck.double_balance_smoke import resume_agent_config
    from mjlab_microduck.double_balance_stage07 import (
        TASK_ID, SEED, verify_frozen_source, check_loaded_runner, require_finite,
    )
    from mjlab_microduck.video_effects import configure_video_cfg

    require(not any(k.startswith('MICRODUCK_') for k in os.environ), 'Unset task overrides')
    require({k: version(k) for k in VERSIONS} == VERSIONS, 'Project dependencies differ from the training lock')
    require(torch.__version__.split('+')[0] == '2.9.1', 'Expected locked Torch 2.9.1')
    if args.device.startswith('cuda'):
        require(torch.cuda.is_available(), 'CUDA unavailable; use the existing working project environment')
    head = verify_frozen_source(REPO)
    require(checksum(args.checkpoint) == EXPECTED_MODELS[args.role], 'Archived checkpoint checksum mismatch')
    training = json.loads((args.checkpoint.parents[1] / 'training.json').read_text())
    for name, expected in training['source_sha256'].items():
        require(checksum(REPO / 'src/mjlab_microduck' / name) == expected, 'Training implementation differs')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    state = validate_training_checkpoint(checkpoint)
    args.output.mkdir(parents=True, exist_ok=False)
    configure_torch_backends()
    torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
    cfg = load_env_cfg(TASK_ID, play=True)
    cfg.scene.num_envs, cfg.seed, cfg.auto_reset = 64, SEED, False
    require(cfg.episode_length_s == 10.0 and not cfg.curriculum, 'Frozen play protocol differs')
    require(tuple(cfg.actions['ball_hold'].levels) == (0.0,), 'Play assistance must be zero')
    configure_video_cfg(cfg, width=1280, height=720)
    cfg.viewer.distance, cfg.viewer.azimuth, cfg.viewer.elevation = .85, 135, -20
    cfg.viewer.env_idx = 0
    agent = resume_agent_config(asdict(load_rl_cfg(TASK_ID)), checkpoint)
    agent.update(logger='tensorboard', upload_model=False)
    dump_yaml(args.output / 'params/env.yaml', asdict(cfg))
    defaults = {name: p.default for name, p in inspect.signature(mdp.double_balance_stable_from_values).parameters.items()
                if p.default is not inspect.Parameter.empty}
    require(cfg.metrics['double_balance_stable_fraction'].params['require_unassisted'], 'Unexpected metric assistance rule')
    report = dict(status='RUNNING', stage07_complete=False, role=args.role,
                  protocol='stage04-frozen-play-10s-metric-phase-diagnostic-v1',
                  completed_updates=state['completed_updates'], checkpoint_sha256=checksum(args.checkpoint),
                  git_head=head, device=args.device, torch_version=torch.__version__, versions=VERSIONS,
                  seed=SEED, num_envs=64, horizon_seconds=10.0, thresholds=defaults,
                  observation_phase='Immediately after original MetricsManager.compute, before forward/reset',
                  official_success='Read cached metric once; never call stateful success function again',
                  comparison_scope='Local diagnostic replay; hardware/renderer differ from cloud, original cloud evaluation retained',
                  created_utc=datetime.now(timezone.utc).isoformat())
    raw = env = writer = None
    original_compute = None
    histories = [[] for _ in range(64)]
    ever_success = [False] * 64
    last_timers = [0.] * 64
    traces = []
    gate_names = ['top_center', 'top_height', 'top_speed', 'lower_offset', 'lower_height_min',
                  'lower_height_max', 'robot_tilt', 'lower_ball_speed', 'unassisted']
    value_names = ['top_center_m', 'top_height_error_m', 'top_speed_m_s', 'lower_offset_m',
                   'lower_height_m', 'robot_tilt_deg', 'lower_ball_speed_m_s', 'hold']
    try:
        raw = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode='rgb_array')
        env = RslRlVecEnvWrapper(raw, clip_actions=agent['clip_actions'])
        runner = load_runner_cls(TASK_ID)(env, agent, None, args.device)
        runner.load(str(args.checkpoint), strict=True, map_location=args.device)
        check_loaded_runner(runner, raw, checkpoint)
        raw.common_step_counter = raw._sim_step_counter = 0
        raw.episode_length_buf.zero_()
        env.reset()
        require(bool((raw._basketball_state.hold == 0).all()), 'Nonzero evaluation assistance')
        policy = runner.get_inference_policy()
        policy.reset()
        obs = env.get_observations()
        alive = torch.ones(64, dtype=torch.bool, device=raw.device)
        names = list(raw.metrics_manager.active_terms)
        stable_col, success_col = names.index('double_balance_stable_fraction'), names.index('double_balance_success')
        original_compute = raw.metrics_manager.compute

        def observed_compute():
            original_compute()
            position, velocity, _ = mdp._double_balance_top_ball_kinematics(raw)
            robot, lower = raw.scene['robot'], raw.scene['ball']
            root = torch.nan_to_num(robot.data.root_link_pos_w, nan=0.)
            lower_pos = mdp._bb_ball_pos(raw)
            center = position[:, :2].norm(dim=-1)
            height_error = (position[:, 2] - defaults['ball_radius']).abs()
            top_speed = velocity.norm(dim=-1)
            offset = (root[:, :2] - lower_pos[:, :2]).norm(dim=-1)
            height = root[:, 2] - lower_pos[:, 2]
            tilt = mdp.wrestle_tilt(raw, 'robot')
            lower_speed = torch.nan_to_num(lower.data.root_link_lin_vel_w[:, :2], nan=0.).norm(dim=-1)
            hold = raw._basketball_state.hold
            gates = torch.stack((center <= defaults['top_center_radius'], height_error <= defaults['top_height_tolerance'],
                                top_speed <= defaults['top_speed_limit'], offset <= defaults['lower_offset_limit'],
                                height >= defaults['lower_height_min'], height <= defaults['lower_height_max'],
                                tilt <= math.radians(defaults['robot_tilt_limit_deg']),
                                lower_speed <= defaults['lower_ball_speed_limit'], hold <= 1e-6), dim=1)
            cached = raw.metrics_manager._step_values
            stable, success = cached[:, stable_col].bool(), cached[:, success_col].bool()
            require(torch.equal(gates.all(dim=1)[alive], stable[alive]), 'Observer disagrees with frozen instantaneous metric')
            values = torch.stack((center, height_error, top_speed, offset, height, torch.rad2deg(tilt), lower_speed, hold), dim=1)
            batch = torch.cat((values, gates.float(), stable[:, None].float(), success[:, None].float(),
                               raw._double_balance_stable_time[:, None]), dim=1).detach().cpu().tolist()
            active = alive.cpu().tolist()
            for env_id, row in enumerate(batch):
                if active[env_id]:
                    stable_flag, success_flag, timer = bool(row[-3]), bool(row[-2]), row[-1]
                    histories[env_id].append(stable_flag)
                    ever_success[env_id] |= success_flag
                    last_timers[env_id] = timer
                    traces.append([raw.common_step_counter, env_id, *row])

        raw.metrics_manager.compute = observed_compute
        writer = imageio.get_writer(args.output / 'env0.mp4', fps=25, codec='libx264', quality=7, macro_block_size=16)
        episodes, frames = [], 0
        with torch.inference_mode():
            for step in range(1, raw.max_episode_length + 2):
                film = bool(alive[0])
                actions = policy(obs)
                obs, rewards, dones, _ = env.step(actions)
                require_finite((actions, obs, rewards), 'diagnostic inference')
                require(not bool((raw.termination_manager.get_term('nan_state') & alive).any()), 'NaN in first episode')
                if film and (step % 2 == 0 or bool(dones[0])):
                    writer.append_data(raw.render())
                    frames += 1
                finished, ended = terminal_episodes(raw, alive, dones)
                episodes.extend(finished)
                alive &= ~ended
                if step % 100 == 0:
                    print(f'Stage07DiagnosticProgress={args.role} {step}/500', flush=True)
                if not bool(alive.any()):
                    break
                done_ids = dones.nonzero(as_tuple=False).flatten()
                if len(done_ids):
                    raw.reset(env_ids=done_ids)
                    policy.reset(dones.bool())
                    obs = env.get_observations()
        require(len(episodes) == 64, 'Not all first episodes completed')
        header = ['step', 'env_id', *value_names, *[f'pass_{name}' for name in gate_names],
                  'cached_stable', 'cached_success', 'cached_stable_time_s']
        with gzip.open(args.output / 'trace.csv.gz', 'wt', newline='') as stream:
            csv_writer = csv.writer(stream)
            csv_writer.writerow(header)
            csv_writer.writerows(traces)
        array = np.asarray(traces)
        masks = array[:, 2 + len(value_names):2 + len(value_names) + len(gate_names)].astype(bool)
        report['failure_fraction_per_condition'] = dict(zip(gate_names, (~masks).mean(axis=0).tolist()))
        one_failure = (~masks).sum(axis=1) == 1
        report['sole_failure_fraction_per_condition'] = dict(zip(gate_names, ((~masks) & one_failure[:, None]).mean(axis=0).tolist()))
        report['value_statistics'] = {name: dict(mean=float(array[:, col + 2].mean()),
                                                min=float(array[:, col + 2].min()), max=float(array[:, col + 2].max()),
                                                p95=float(np.quantile(array[:, col + 2], .95)))
                                      for col, name in enumerate(value_names)}
        for episode in episodes:
            env_id = episode['env_id']
            longest, final = streaks(histories[env_id])
            require(abs(statistics.mean(histories[env_id]) - episode['double_balance_stable_fraction']) < 1e-5,
                    'Trace and frozen episode average differ')
            require(abs(final * raw.step_dt - last_timers[env_id]) < 1e-3, 'Trace and official timer differ')
            episode.update(longest_stable_streak_s=longest * raw.step_dt, final_stable_streak_s=final * raw.step_dt,
                           official_final_timer_s=last_timers[env_id], success_reached_at_any_step=ever_success[env_id])
        report.update(status='PASS', finite_checks='PASS', episodes=episodes, video_frames=frames,
                      trace_rows=len(traces), cached_metric_parity='PASS',
                      success_fraction=statistics.mean(e['success'] for e in episodes),
                      success_at_any_step_fraction=statistics.mean(ever_success),
                      mean_stable_fraction=statistics.mean(e['double_balance_stable_fraction'] for e in episodes),
                      mean_survival_seconds=statistics.mean(e['seconds'] for e in episodes),
                      longest_stable_streak_s=max(e['longest_stable_streak_s'] for e in episodes),
                      checkpoint_unchanged=checksum(args.checkpoint) == EXPECTED_MODELS[args.role])
        require(report['checkpoint_unchanged'], 'Input checkpoint changed')
    except BaseException as exc:
        report.update(status='FAIL', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if original_compute is not None:
            raw.metrics_manager.compute = original_compute
        if writer is not None:
            writer.close()
        if env is not None:
            env.close()
        elif raw is not None:
            raw.close()
        report['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(args.output / 'diagnostic.json', report)
    print('Stage07Diagnostic=' + json.dumps({k: report[k] for k in (
        'role', 'status', 'success_fraction', 'mean_stable_fraction', 'failure_fraction_per_condition')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cuda:0', choices=['cuda:0', 'cpu'])
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--role', choices=list(EXPECTED_MODELS), help=argparse.SUPPRESS)
    parser.add_argument('--checkpoint', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require(REPO.is_dir(), 'Run on the lx laptop with its existing project environment')
    if args.worker:
        worker(args)
        return
    candidates = sorted(ARTIFACTS.glob('microduck-stage07-results-*/collection-manifest.json'))
    require(bool(candidates), 'No collected Stage 07 archive found on laptop')
    source = candidates[-1].parent
    manifest = json.loads(candidates[-1].read_text())
    for role, expected in EXPECTED_MODELS.items():
        require(manifest['selected'][role]['sha256'] == expected, 'Unexpected selected archive')
    output = Path(tempfile.mkdtemp(prefix='diagnostic-', dir=ARTIFACTS))
    environment = os.environ.copy()
    environment['MUJOCO_GL'] = 'egl'
    reports = {}
    for role in EXPECTED_MODELS:
        command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--role', role,
                   '--checkpoint', str(source / manifest['selected'][role]['path']),
                   '--output', str(output / role), '--device', args.device]
        print(f'Stage07DiagnosticStart={role}\nLog={output / (role + ".log")}', flush=True)
        with (output / f'{role}.log').open('x') as log:
            try:
                result = subprocess.run(command, cwd=REPO, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=900)
            except subprocess.TimeoutExpired:
                raise ValueError(f'Diagnostic timed out, no training was started. Inspect {output / (role + ".log")}')
        if result.returncode:
            print('\n'.join((output / f'{role}.log').read_text().splitlines()[-35:]), file=sys.stderr)
            raise ValueError(f'Diagnostic failed; original files retained: {output}')
        report = json.loads((output / role / 'diagnostic.json').read_text())
        require(report['status'] == 'PASS', 'Diagnostic report failed')
        reports[role] = report
        print('Stage07DiagnosticCase=PASS ' + role, flush=True)
    write_json(output / 'summary.json', reports)
    destination = Path('/home/lx/下载') / f'microduck-stage07-{output.name}.tar.gz'
    with tarfile.open(destination, 'x:gz') as archive:
        archive.add(output, arcname=output.name)
    destination.with_suffix('.gz.sha256').write_text(f'{checksum(destination)}  {destination.name}\n')
    print(f'Stage07LocalDiagnostic=PASS\nArchive={destination}\nCloudGPU=NOT_USED\nPPOUpdates=0')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, KeyError) as exc:
        print(f'Stage07LocalDiagnostic=FAIL: {exc}', file=sys.stderr)
        sys.exit(1)
