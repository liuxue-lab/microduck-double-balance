"""One bounded PPO phase of the final six-hour disturbance campaign.

Every optimizer commit retains a CPU snapshot.  A later interrupted rollout
therefore cannot contaminate the checkpoint's normalizer, RNG or optimizer.
"""
import argparse
import contextlib
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
from budget import reserve_update, commit_update, validate as validate_budget
from runtime_helpers import JsonScalarWriter, require_finite_group, require_step_outputs, host_scalar_means
from fingerprint import fingerprint


class TrainingStop(Exception):
    pass


class ChunkedLog:
    """Bound each evidence file while keeping every complete write/JSON row."""
    def __init__(self, folder, stem, suffix='jsonl', limit=16*1024*1024):
        self.folder, self.stem, self.suffix = Path(folder), stem, suffix
        self.limit, self.index, self.size = limit, 0, 0
        self.path = None

    def write(self, value):
        data = value.encode('utf-8')
        if not data:
            return 0
        require(len(data) <= self.limit, 'A single log record exceeds its bounded file size')
        if self.path is None or self.size+len(data) > self.limit:
            self.index += 1
            self.path = self.folder/('%s-%06d.%s' % (self.stem, self.index, self.suffix))
            with self.path.open('xb'):
                pass
            self.size = 0
        with self.path.open('ab') as stream:
            stream.write(data)
        self.size += len(data)
        return len(value)

    def row(self, value):
        return self.write(json.dumps(value, ensure_ascii=False, allow_nan=False)+'\n')

    def flush(self):
        pass  # Each write has already closed and flushed its descriptor.


def training_schedule(elapsed_seconds):
    """Independent new-campaign wall clock; never resets task maturity."""
    require(math.isfinite(elapsed_seconds) and elapsed_seconds >= 0., 'Invalid elapsed training time')
    fraction = .05 if elapsed_seconds < .10*TRAIN_SECONDS else .075 if elapsed_seconds < .25*TRAIN_SECONDS else .10
    learning_rate = LEARNING_RATE if elapsed_seconds < (2./3.)*TRAIN_SECONDS else 1e-5
    return dict(force_weight_fraction=fraction, learning_rate=learning_rate,
                no_push_probability=.25)


def clone_to_cpu(value, torch):
    """Detach all tensor storage, including nested Adam and normalizer state."""
    if torch.is_tensor(value):
        return value.detach().to(device='cpu').clone()
    if isinstance(value, dict):
        return {key: clone_to_cpu(item, torch) for key, item in value.items()}
    if isinstance(value, list):
        return [clone_to_cpu(item, torch) for item in value]
    if isinstance(value, tuple):
        return tuple(clone_to_cpu(item, torch) for item in value)
    return deepcopy(value)


def make_committed_snapshot(algorithm_state, metadata, completed, torch):
    """Called only after a complete, finite optimizer update (or verified load)."""
    require(metadata['policy_updates'] == retained_policy_updates(completed),
            'Snapshot policy/update position differs')
    saved = clone_to_cpu(algorithm_state, torch)
    meta = clone_to_cpu(metadata, torch)
    saved.update(iter=retained_policy_updates(completed)-1,
                 infos={'stage12_native': meta, 'stage12_final6h': meta,
                        'env_state': {'common_step_counter': meta['common_step_counter']}})
    return saved


def persist_snapshot(out, saved, completed, torch):
    """An immutable model/receipt pair, followed by an atomic latest pointer."""
    out = Path(out)
    meta = saved['infos']['stage12_final6h']
    require(meta['campaign_completed_updates'] == completed
            and meta['policy_updates'] == retained_policy_updates(completed),
            'Checkpoint metadata differs from its committed update position')
    path = out/'checkpoints'/('final6h_%06d.pt' % completed)
    path.parent.mkdir(exist_ok=True)
    if path.exists():
        receipt = read(path.with_suffix('.json'))
        require(receipt['completed_updates'] == completed and sha(path) == receipt['sha256']
                and path.stat().st_size == receipt['bytes'],
                'Existing checkpoint differs; preserve it')
    else:
        tmp = path.with_suffix('.tmp.pt')
        require(not tmp.exists(), 'Earlier partial checkpoint exists; preserve it')
        torch.save(saved, tmp)
        with tmp.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(tmp, path)
        receipt = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size,
                       completed_updates=completed, policy_updates=retained_policy_updates(completed),
                       attempted_updates=meta['campaign_attempted_updates'],
                       package_identity=meta['package_identity'])
        write(path.with_suffix('.json'), receipt, exclusive=True)
    write(out/'latest.json', receipt)
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=tuple(TARGETS), required=True)
    parser.add_argument('--session', type=int, required=True)
    args = parser.parse_args()
    budget = read(SLOT/'budget.json')
    validate_budget(budget)
    package_id = verify_package()
    out = SLOT/'run'
    count = 64 if args.phase == 'smoke' else NUM_ENVS
    seed = SMOKE_SEED if args.phase == 'smoke' else TRAIN_SEED
    force_seed = FORCE_SMOKE_SEED if args.phase == 'smoke' else FORCE_TRAIN_SEED
    target = TARGETS[args.phase]
    state = dict(status='INITIALIZING', phase=args.phase, detail='load_runtime', num_envs=count,
                 completed_updates=budget['completed_updates'], attempted_updates=budget['attempted_updates'],
                 max_updates=MAX_UPDATES, phase_target=target,
                 policy_updates=retained_policy_updates(budget['completed_updates']),
                 physical_trajectory_resume=False, automatic_retry=False)
    raw = env = runner = monitor = pushes = None
    restored = False
    rollout_steps = 0
    update_in_flight = False
    stop = False
    stop_at_deadline = False
    committed_snapshot = None
    committed_position = None
    last_save_time = time.monotonic()
    unwrapped_sim_step = None
    session_dir = out/('session-%03d' % args.session)
    session_dir.mkdir(exist_ok=False)
    metrics_log = ChunkedLog(session_dir, 'metrics')
    episode_log = ChunkedLog(session_dir, 'force-episodes')
    logger_console = ChunkedLog(session_dir, 'logger-console', suffix='log')
    candidates = out/'candidates'
    candidates.mkdir(exist_ok=True)
    original_handlers = {}

    def progress(**kw):
        elapsed = max(0., time.time()-budget['started_utc'])
        state.update(kw, elapsed_seconds=elapsed,
                     original_deadline_utc=budget['deadline_utc'],
                     train_stop_utc=budget['train_stop_utc'],
                     completed_updates=budget['completed_updates'],
                     attempted_updates=budget['attempted_updates'],
                     pending_update=budget['pending'],
                     policy_updates=retained_policy_updates(budget['completed_updates']),
                     formal_completed_updates=budget['phase_completed']['formal'],
                     progress_percent=min(100., 100.*elapsed/MAX_SECONDS))
        write(SLOT/'progress.json', state)

    def handle_signal(signum, _frame):
        nonlocal stop, stop_at_deadline
        if signum == signal.SIGTERM and time.time() >= budget['train_stop_utc']:
            stop_at_deadline = True
        else:
            stop = True

    def boundary():
        if stop or (SLOT/'STOP.json').exists():
            raise TrainingStop('user_stop')
        if stop_at_deadline or time.time() >= budget['train_stop_utc']:
            raise TrainingStop('train_window_complete')

    def capture(completed):
        schedule = training_schedule(max(0., time.time()-budget['started_utc']))
        meta = dict(package_identity=package_id, origin_package_identity=ORIGIN_ID,
                    origin_checkpoint_sha256=MODEL_SHA, original_selected_sha256=SELECTED_BASE_SHA,
                    parent_policy_updates=START_POLICY_UPDATES,
                    policy_updates=retained_policy_updates(completed),
                    campaign_completed_updates=completed,
                    campaign_attempted_updates=budget['attempted_updates'],
                    num_envs=count, common_step_counter=raw.common_step_counter,
                    sim_step_counter=raw._sim_step_counter, lower_assistance=0.,
                    learning_rate=float(runner.alg.learning_rate), physical_trajectory_resume=False,
                    reward_spec=read(HERE/'reward-spec.json'),
                    disturbance=pushes.snapshot() if pushes is not None else None,
                    force_rng=pushes.state_dict() if pushes is not None and hasattr(pushes, 'state_dict') else None,
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                    numpy_rng=np.random.get_state(), python_rng=random.getstate(),
                    committed_snapshot=True)
        return make_committed_snapshot(runner.alg.save(), meta, completed, torch)

    def save_model(_path=None, infos=None):
        nonlocal last_save_time
        if _path is not None:
            return  # Upstream periodic filenames never bypass this checkpoint writer.
        require(restored and committed_snapshot is not None and committed_position is not None,
                'No complete optimizer snapshot exists')
        # A pending new rollout or interrupted optimizer is never serialized.
        # The CPU snapshot always represents the preceding committed update.
        receipt = persist_snapshot(out, committed_snapshot, committed_position, torch)
        last_save_time = time.monotonic()
        return receipt

    def candidate(label):
        if committed_position is None or committed_position <= 5:
            return
        path = candidates/(label+'.json')
        if path.exists():
            return
        receipt = save_model()
        write(path, dict(receipt, candidate_label=label,
                         captured_elapsed_seconds=max(0., time.time()-budget['started_utc']),
                         training_pending_attempt=budget['pending']), exclusive=True)

    def apply_schedule():
        schedule = training_schedule(max(0., time.time()-budget['started_utc']))
        for group in runner.alg.optimizer.param_groups:
            group['lr'] = schedule['learning_rate']
        runner.alg.learning_rate = schedule['learning_rate']
        pushes.set_stage_fraction(schedule['force_weight_fraction'])
        return schedule

    for sig in (signal.SIGTERM, signal.SIGINT):
        original_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, handle_signal)
    try:
        progress()
        boundary()
        versions = verify_runtime()
        require(budget['pending'] is None, 'Earlier attempt remains pending; no automatic continuation')
        sys.path.insert(0, str(SOURCE/'src'))
        import numpy as np
        import torch
        import warp as wp
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
        from mjlab.utils.torch import configure_torch_backends
        from mjlab_microduck.tasks.microduck_basketball_env_cfg import MicroduckBasketballRlCfg
        from mjlab_microduck.tasks import mdp
        from native_task import head_scale, lower_unassisted, geometric_joint_stable
        from robust_task import make_cfg, validate_cost_signs, verify_runtime_reward_order
        from normalization import configure_normalizer_updates, equal_state
        from train_motor_monitor import TrainingMotorMonitor
        from force_runtime import VectorPushes
        from statistics_runtime_check import check_statistics_runtime
        require(torch.cuda.is_available() and '5090' in torch.cuda.get_device_name(0),
                'Formal PPO uses the current cloud RTX5090 only')
        write(session_dir/'statistics-runtime-check.json', check_statistics_runtime(torch, 'cuda:0'), exclusive=True)
        configure_torch_backends()
        torch.set_num_threads(min(4, len(os.sched_getaffinity(0))))
        wp.config.kernel_cache_dir = str(TRAIN/'cache/warp')
        cfg = make_cfg(count, seed)
        require(cfg.episode_length_s == 60. and not cfg.curriculum and 'push_robot' not in cfg.events,
                'Original time contract or independent disturbance curriculum differs')
        progress(detail='build_native_environment')
        raw = ManagerBasedRlEnv(cfg, device='cuda:0', render_mode=None)
        env = RslRlVecEnvWrapper(raw)
        agent = asdict(deepcopy(MicroduckBasketballRlCfg))
        agent.update(seed=seed, logger='tensorboard', upload_model=False,
                     experiment_name='stage12_push10_final6h_v1', run_name=args.phase,
                     save_interval=25, max_iterations=MAX_UPDATES, check_for_nan=False)
        agent['algorithm'].update(learning_rate=LEARNING_RATE, schedule='fixed', num_mini_batches=4,
                                  clip_param=.15, entropy_coef=.003)
        require(agent['num_steps_per_env'] == 24, 'Original rollout length changed')
        declared = deepcopy(agent)
        runner = MjlabOnPolicyRunner(env, agent, log_dir=str(session_dir/'tensorboard'), device='cuda:0')
        runner.logger.git_status_repos = []
        original_writer = runner.logger.init_logging_writer

        def init_writer():
            try:
                original_writer()
            except ImportError:
                runner.logger.writer = JsonScalarWriter(session_dir/'scalars.jsonl')
                runner.logger.logger_type = 'json'
        runner.logger.init_logging_writer = init_writer
        source = input_checkpoint()
        require(checkpoint_accounting_position(source) == budget['completed_updates'],
                'Checkpoint lags committed ledger; no replay')
        require(sha(source['path']) == source['sha256'], 'Input checkpoint changed')
        saved = torch.load(source['path'], map_location='cpu', weights_only=False)
        meta = saved['infos']['stage12_native']
        expected_id = ORIGIN_ID if source.get('parent_checkpoint') else package_id
        require(meta['package_identity'] == expected_id
                and meta['policy_updates'] == retained_policy_updates(source['completed_updates']),
                'Warm-start lineage differs')
        if not source.get('parent_checkpoint'):
            require(meta['origin_checkpoint_sha256'] == MODEL_SHA, 'New checkpoint parent differs')
        runner.alg.load(saved, None, strict=True)
        require(equal_state(runner.alg.save(), saved, torch, top_level=True),
                'Restored actor/critic/optimizer/normalizer differs')
        previous_lr = [float(group['lr']) for group in runner.alg.optimizer.param_groups]
        raw.common_step_counter = meta['common_step_counter']
        raw._sim_step_counter = meta['sim_step_counter']
        raw.reset(seed=seed)
        bb = mdp._bb_state(raw)
        bb.level.zero_()
        bb.hold.zero_()
        torch.set_rng_state(meta['torch_rng'])
        torch.cuda.set_rng_state_all(meta['cuda_rng'])
        np.random.set_state(meta['numpy_rng'])
        random.setstate(meta['python_rng'])
        runner.alg.actor.reset()
        runner.alg.critic.reset()
        configure_normalizer_updates(runner.alg.actor.obs_normalizer, 55, torch)
        configure_normalizer_updates(runner.alg.critic.obs_normalizer, 76, torch)
        obs = env.get_observations()
        require(tuple(obs['actor'].shape) == (count, 61) and tuple(obs['critic'].shape) == (count, 85),
                'Observation schema differs')
        require(raw.max_episode_length == 3000 and abs(raw.step_dt-.02) < 1e-12
                and abs(raw.physics_dt-.002) < 1e-12 and raw.cfg.decimation == 10 and head_scale(raw) == 1.,
                'Time contract or mature head schedule changed')
        require_finite_group(torch, [*obs.values(), *runner.alg.actor.state_dict().values(),
                                     *runner.alg.critic.state_dict().values()], 'Nonfinite restored state')
        write(session_dir/'reward-runtime-check.json', verify_runtime_reward_order(raw), exclusive=True)
        validate_cost_signs(raw)
        initial_schedule = training_schedule(max(0., time.time()-budget['started_utc']))
        # Hook order matters: scene/action -> motor zero-wrench check ->
        # actual force write -> integration -> force cleanup -> qpos audit.
        pushes = VectorPushes(raw, torch, force_seed,
                              initial_fraction=initial_schedule['force_weight_fraction'], no_push_probability=.25)
        monitor = TrainingMotorMonitor(raw, torch, session_dir)
        unwrapped_sim_step = raw.sim.step

        def timed_sim_step():
            if monitor.active:
                boundary()  # No new integration after the training boundary.
            return unwrapped_sim_step()
        raw.sim.step = timed_sim_step
        schedule = apply_schedule()
        restored = True
        committed_position = budget['completed_updates']
        committed_snapshot = capture(committed_position)
        write(session_dir/'load.json', dict(source=source, optimizer_restored=True, normalization_restored=True,
              state_verified_before_learning_rate_change=True, previous_learning_rates=previous_lr,
              new_learning_rate=schedule['learning_rate'], previous_num_envs=meta['num_envs'], num_envs=count,
              new_episode_seed=seed, private_force_seed=force_seed, recurrent_state_reset=True,
              rng_order='seeded environment reset then checkpoint Torch/CUDA/NumPy/Python RNG; private force generator',
              physical_trajectory_resume=False, agent=declared, runtime=versions,
              device=torch.cuda.get_device_name(0), original_deadline_utc=budget['deadline_utc'],
              train_stop_utc=budget['train_stop_utc']), exclusive=True)
        original_step = env.step

        def step(actions):
            nonlocal rollout_steps
            boundary()
            apply_schedule()
            if rollout_steps == 0:
                reserve_update(budget, args.phase, time.time())
                write(SLOT/'budget.json', budget)
            require_finite_group(torch, [actions], 'Nonfinite action')
            before = monitor.steps
            pushes.begin_control()
            monitor.active = True
            try:
                result = original_step(actions)
            finally:
                monitor.active = False
            require(monitor.steps == before+10, 'Training physics substep count differs')
            observations, rewards, dones, _ = result
            pushes.end_control(dones)
            monitor.joint_stats.after_control_step(dones, raw.reset_terminated)
            require_step_outputs(torch, observations, rewards, dones, raw.termination_manager.get_term('nan_state'))
            validate_cost_signs(raw)
            rollout_steps += 1
            if args.phase == 'smoke' and rollout_steps % 6 == 0:
                progress(detail='new_config_smoke_rollout', rollout_steps=rollout_steps,
                         disturbance=pushes.snapshot())
            return result
        env.step = step
        original_update = runner.alg.update

        def update():
            nonlocal rollout_steps, update_in_flight, committed_snapshot, committed_position
            boundary()
            require(rollout_steps == 24, 'Incomplete PPO rollout')
            update_in_flight = True
            loss = original_update()
            require(all(math.isfinite(float(value)) for value in loss.values()), 'Nonfinite PPO loss')
            require_finite_group(torch, (p for model in (runner.alg.actor, runner.alg.critic)
                                         for p in model.parameters()), 'Nonfinite updated policy')
            # Snapshot before committing the ledger.  It owns its storage and
            # cannot be mutated by the next rollout/normalizer/optimizer.
            next_position = budget['completed_updates']+1
            next_snapshot = capture(next_position)
            commit_update(budget, args.phase)
            write(SLOT/'budget.json', budget)
            committed_snapshot = next_snapshot
            committed_position = next_position
            rollout_steps = 0
            update_in_flight = False
            if args.phase == 'smoke' or budget['phase_completed']['formal'] == 1:
                save_model()
            return loss
        runner.alg.update = update
        runner.save = save_model
        original_log = runner.logger.log

        def log(**kw):
            keys, values, episodes = [], [], {}
            for item in runner.logger.ep_extras:
                for key, value in item.items():
                    if any(term in key for term in ('native_', 'top_ball_center', 'Episode_Termination', 'Episode_Reward/')):
                        keys.append(key)
                        values.append(value)
            for key, value in zip(keys, host_scalar_means(torch, values)):
                require(math.isfinite(value), 'Nonfinite episode metric: '+key)
                if any(term in key for term in ('head_roll_margin', 'head_roll_outward',
                                                'other_joint_margin', 'other_joint_outward', 'physical_failure')):
                    require(value <= 1e-7, 'A constraint/failure cost became a positive reward: '+key)
                episodes.setdefault(key, []).append(value)
            # Retain upstream human-readable output in bounded files rather
            # than growing the supervisor's single worker.log indefinitely.
            with contextlib.redirect_stdout(logger_console):
                original_log(**kw)
            weighted_costs = validate_cost_signs(raw)
            names = list(weighted_costs)
            current = host_scalar_means(torch, [lower_unassisted(raw).float(), geometric_joint_stable(raw).float(),
                                                *weighted_costs.values()])
            schedule = training_schedule(max(0., time.time()-budget['started_utc']))
            disturbance_rollout = pushes.take_rollout()
            receipts = disturbance_rollout.pop('completed_episode_receipts', [])
            for receipt in receipts:
                episode_log.row(dict(completed_update=budget['completed_updates'], episode=receipt))
            disturbance_rollout['completed_episode_receipt_rows'] = len(receipts)
            disturbance_rollout['episode_receipt_files_prefix'] = str(session_dir/'force-episodes-')
            progress(status='TRAINING', detail='PPO', rollout_steps=24,
                     collection_seconds=kw['collect_time'], learning_seconds=kw['learn_time'],
                     lower_unassisted_current_fraction=current[0], geometric_current_fraction=current[1],
                     weighted_cost_current_means=dict(zip(names, current[2:])),
                     learning_rate=schedule['learning_rate'],
                     force_stage_fraction=schedule['force_weight_fraction'],
                     joint_rollout=monitor.joint_stats.take_rollout(), disturbance_rollout=disturbance_rollout,
                     episode_metrics={key: sum(items)/len(items) for key, items in episodes.items()},
                     last_losses={key: float(value) for key, value in kw['loss_dict'].items()},
                     checked_training_physics_steps=monitor.steps,
                     formal_training_ready=args.phase == 'formal' and budget['phase_completed']['formal'] > 0)
            elapsed = time.time()-budget['started_utc']
            if args.phase == 'formal':
                if elapsed >= TRAIN_SECONDS/3.:
                    candidate('candidate-01')
                if elapsed >= 2.*TRAIN_SECONDS/3.:
                    candidate('candidate-02')
            if budget['completed_updates'] % 25 == 0 or time.monotonic()-last_save_time >= 120:
                save_model()
            metrics_log.row(state)
            brief = {key: state[key] for key in ('phase', 'completed_updates', 'attempted_updates',
                'formal_completed_updates', 'elapsed_seconds', 'train_stop_utc', 'original_deadline_utc',
                'collection_seconds', 'learning_seconds', 'learning_rate', 'force_stage_fraction',
                'formal_training_ready')}
            brief.update(joint_violation_fraction=state['joint_rollout'].get('any_joint_violation_fraction'),
                         completed_force_pulses=disturbance_rollout.get('completed_pulses'))
            print('FINAL6H_TRAINING_PROGRESS='+json.dumps(brief, ensure_ascii=False), flush=True)
        runner.logger.log = log
        runner.current_learning_iteration = retained_policy_updates(budget['completed_updates'])
        remaining = target-budget['completed_updates']
        require(remaining > 0, 'No remaining updates in this phase')
        boundary()
        runner.learn(remaining, init_at_random_ep_len=False)
        save_model()
        require(budget['completed_updates'] == target and budget['pending'] is None,
                'Training phase ended without its target or time boundary')
        if args.phase == 'smoke':
            boundary()
            witness = pushes.snapshot()
            require(witness['completed_pulses'] > 0 and witness['applied_world_physics_steps'] >= 50
                    and witness['owned_force_cleared'] is True and witness['external_wrench_zero'] is True
                    and witness['integration_errors'] == 0 and witness['cleanup_errors'] == 0,
                    'Smoke did not witness a complete physical-force pulse with clean integration')
            state['smoke_disturbance_witness'] = witness
            progress(detail='smoke_onnx_export')
            import onnx
            from mjlab_microduck.onnx_policy_contract import bake_action_clip
            before = fingerprint(runner.alg.actor.state_dict())
            runner.export_policy_to_onnx(str(session_dir), 'smoke.onnx')
            onnx_path = session_dir/'smoke.onnx'
            bake_action_clip(str(onnx_path), declared['clip_actions'])
            model = onnx.load(str(onnx_path))
            onnx.checker.check_model(model)
            inputs = [[d.dim_value for d in value.type.tensor_type.shape.dim] for value in model.graph.input]
            outputs = [[d.dim_value for d in value.type.tensor_type.shape.dim] for value in model.graph.output]
            require(any(shape and shape[-1] == 61 for shape in inputs)
                    and any(shape and shape[-1] == 14 for shape in outputs), 'Smoke ONNX schema differs')
            require(fingerprint(runner.alg.actor.state_dict()) == before,
                    'Export changed policy or normalization')
            state['smoke_export'] = dict(passed=True, path=str(onnx_path), sha256=sha(onnx_path),
                 inputs=inputs, outputs=outputs,
                 exporter='original runner.export_policy_to_onnx with baked normalizer', deployment_acceptance=False)
        else:
            candidate('candidate-final')
        progress(status='TRAINING_PHASE_COMPLETE', detail='checkpoint_saved',
                 reason='smoke_complete' if args.phase == 'smoke' else 'update_ceiling',
                 latest_checkpoint=input_checkpoint())
    except TrainingStop as exc:
        if restored and committed_snapshot is not None:
            save_model()
        normal_end = str(exc) == 'train_window_complete' and args.phase == 'formal'
        if normal_end:
            candidate('candidate-final')
        progress(status='TRAINING_PHASE_COMPLETE' if normal_end else 'TRAINING_STOPPED_PRESERVE',
                 detail='checkpoint_saved' if restored else 'initialization_stopped', reason=str(exc),
                 latest_checkpoint=input_checkpoint(),
                 abandoned_attempt_remains_charged=budget['pending'] is not None,
                 committed_snapshot_saved=restored and committed_snapshot is not None)
    except BaseException as exc:
        original_error, original_traceback = repr(exc), traceback.format_exc()[-12000:]
        if restored and committed_snapshot is not None:
            try:
                save_model()
            except BaseException as save_exc:
                state['recovery_checkpoint_error'] = repr(save_exc)
        progress(status='TRAINING_ERROR_PRESERVE', error=original_error, traceback=original_traceback,
                 update_in_flight=update_in_flight,
                 abandoned_attempt_remains_charged=budget['pending'] is not None)
    finally:
        if raw is not None and unwrapped_sim_step is not None:
            raw.sim.step = unwrapped_sim_step
        if monitor is not None:
            try:
                monitor.close()
            except BaseException as exc:
                state.update(status='TRAINING_ERROR_PRESERVE', monitor_close_error=repr(exc))
        if pushes is not None:
            try:
                pushes.close()
                state['final_disturbance'] = pushes.snapshot()
            except BaseException as exc:
                state.update(status='TRAINING_ERROR_PRESERVE', disturbance_close_error=repr(exc))
        if env is not None:
            try:
                env.close()
            except BaseException as exc:
                state.update(status='TRAINING_ERROR_PRESERVE', environment_close_error=repr(exc))
        elif raw is not None:
            try:
                raw.close()
            except BaseException as exc:
                state.update(status='TRAINING_ERROR_PRESERVE', environment_close_error=repr(exc))
        if runner is not None:
            try:
                runner.logger.stop_logging_writer()
            except BaseException:
                pass
            try:
                runner.logger.writer.close()
            except (AttributeError, ValueError):
                pass
        progress(finished_utc=time.time())
        write(session_dir/'result.json', state, exclusive=True)
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)
    return 0 if state['status'] == 'TRAINING_PHASE_COMPLETE' else 3 if state['status'] == 'TRAINING_STOPPED_PRESERVE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
