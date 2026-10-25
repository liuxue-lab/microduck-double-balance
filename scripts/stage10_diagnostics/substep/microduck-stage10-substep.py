#!/usr/bin/env python3
"""Stage 10 batch B: three fresh processes, 128 environments, 0.2 s each, zero PPO."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import os
from pathlib import Path
import shlex
import shutil
import sys
import traceback
import uuid
import zipfile

import stage10_core as core
from stage10_core import require, read, write, sha, checked, utc

HERE = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
SIDECAR = '6878f43233fb5c064f4500251ba0b3353e5e0d8179e08bfd3d56aefe08edc4cc'
JOBS = ['01-substep', '02-substep', '03-substep']
TOOLS = ('microduck-stage10-substep.py', 'stage10_core.py', 'stage10_trace.py', 'stage10_capture.py', 'reference-runtime.json')
SCOPE = {'runs': 3, 'environments': 128, 'control_steps_each': 10,
         'physics_steps_each': 100, 'seconds_each': 0.2, 'new_ppo_updates': 0,
         'old_episode_length_seconds': 10, 'assistance': 0, 'video': False}


def tool_hashes():
    return {name: sha(HERE / name) for name in TOOLS}


def validate_plan(folder):
    plan = read(folder / 'plan.json')
    require(plan['schema_version'] == 1 and plan['batch'] == 'stage10-substep-B', 'Wrong batch plan')
    require(plan['source_head'] == core.BASE and plan['tools'] == tool_hashes(), 'Source/tool identity changed')
    require(plan['scope'] == SCOPE and plan['jobs'] == JOBS, 'Diagnostic budget or scope changed')
    checked(plan['checkpoint'], core.PRIMARY)
    checked(folder / 'initial-states/dev.pt', core.INITIAL)
    checked(folder / 'initial-states/dev.json', SIDECAR)
    return plan


def runtime_differences(actual, expected, prefix=''):
    if isinstance(actual, dict) and isinstance(expected, dict):
        result = []
        for key in sorted(set(actual) | set(expected)):
            path = prefix + '.' + key if prefix else key
            if key not in actual or key not in expected:
                result.append({'field': path, 'actual': actual.get(key), 'expected': expected.get(key),
                               'missing_from': 'actual' if key not in actual else 'expected'})
            else:
                result.extend(runtime_differences(actual[key], expected[key], path))
        return result
    return [] if actual == expected else [{'field': prefix, 'actual': actual, 'expected': expected}]


def runtime_identity(torch, audit_path):
    # Same version source as Batch A: the loaded torch build, including +cu128.
    # Distribution metadata (e.g. 2.9.1) is separate evidence, not that build ID.
    names = ('mjlab', 'mujoco', 'mujoco-warp', 'warp-lang', 'rsl-rl-lib')
    identity = {'versions': {**{name: version(name) for name in names}, 'torch': str(torch.__version__)},
                'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0), 'num_threads': torch.get_num_threads(),
                'backend': core.backend_snapshot(torch),
                'environment': {name: os.environ.get(name) for name in
                                ('MUJOCO_GL', 'PYTHONHASHSEED', 'CUBLAS_WORKSPACE_CONFIG', 'CUDA_LAUNCH_BLOCKING')}}
    reference = read(HERE / 'reference-runtime.json')
    expected = {name: reference[name] for name in identity}
    differences = runtime_differences(identity, expected)
    write(audit_path, {'status': 'MISMATCH' if differences else 'MATCH',
                      'actual': identity, 'expected': expected, 'differences': differences,
                      'torch_distribution_version': version('torch'),
                      'torch_build_version': str(torch.__version__),
                      'comparison_version_source': 'torch.__version__, identical to Batch A',
                      'backend_settings_changed_by_check': False, 'new_ppo_updates': 0})
    summary = '; '.join(str(row['field']) + ': expected=' + repr(row['expected']) +
                        ', actual=' + repr(row['actual']) for row in differences)
    require(not differences, 'Runtime/backend differs from Batch A: ' + summary +
            '; full values saved in ' + str(audit_path))
    return identity


def worker(plan_path, output):
    # Validate paths, source and hashes before deserializing the approved model.
    folder = plan_path.resolve().parent
    require(folder.is_relative_to((core.ART / 'substep').resolve()), 'Worker requires a Stage 10 substep plan')
    require(output.parent.parent == folder and output.parent.name in JOBS and not output.exists(), 'Unsafe worker destination')
    core.verify_source()
    plan = validate_plan(folder)
    output.mkdir(parents=True, exist_ok=False)
    report = {'status': 'RUNNING', 'created_utc': utc(), 'source_head': core.BASE,
              'scope': SCOPE, 'new_ppo_updates': 0, 'physics_steps': 0, 'stage10_complete': False,
              'checkpoint_sha256': core.PRIMARY, 'dataset_sha256': core.INITIAL,
              'task_success_not_evaluated': True, 'video': 'NOT_REQUESTED_FOR_0_2S_DIAGNOSTIC',
              'observer_effect': 'Readback synchronizes GPU work; this trace is an instrumented condition.'}
    write(output / 'result.json', report)
    raw = env = observer = capture = trace = None
    try:
        import torch
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
        from mjlab.utils.os import dump_yaml
        from mjlab_microduck.double_balance_stage09 import no_optimizer_steps
        from mjlab_microduck.double_balance_stage09_state import preflight, load_checked
        from mjlab_microduck.double_balance_stage09_evaluation import (
            TASK_ID, evaluation_config, resume_agent_config, check_loaded_runner,
            seal_initial_states, MetricObserver, require_finite, require_equal)
        from mjlab_microduck.double_balance_stage09_diagnostics import observer_class, kinematic_calibration
        from mjlab_microduck.double_balance_stage08_state import assert_zero_assistance
        from stage10_capture import Capture, module_digest
        from stage10_trace import TraceWriter
        import mjlab_microduck.tasks  # noqa: F401

        require(preflight(cloud=False) == core.BASE, 'Frozen local preflight HEAD differs')
        import inspect
        from mjlab.sim import Simulation
        expected_sources = read(HERE / 'reference-runtime.json')['sources']
        for cls in (ManagerBasedRlEnv, Simulation, RslRlVecEnvWrapper):
            for method in ('step', 'reset', 'forward', 'get_observations'):
                key = cls.__name__ + '.' + method
                if key in expected_sources:
                    require(inspect.getsource(getattr(cls, method)) == expected_sources[key],
                            'Installed dependency method changed since Batch A: ' + key)
        report['runtime_identity'] = runtime_identity(torch, output / 'runtime-check.json')
        report['backend_before'] = core.backend_snapshot(torch)
        report['backend_reader_before'] = core.precision_reader_audit(torch)
        write(output / 'result.json', report)
        # Includes construction/load: any accidental optimizer step is forbidden.
        with no_optimizer_steps():
            checkpoint, state, checksum = load_checked(plan['checkpoint'], evaluation=True)
            require(checksum == core.PRIMARY, 'Primary checkpoint changed')
            cfg = evaluation_config('dev')
            agent = resume_agent_config(asdict(load_rl_cfg(TASK_ID)), checkpoint)
            agent.update(logger='tensorboard', upload_model=False)
            dump_yaml(output / 'params/env.yaml', asdict(cfg))
            raw = ManagerBasedRlEnv(cfg=cfg, device='cuda:0', render_mode=None)
            env = RslRlVecEnvWrapper(raw, clip_actions=agent['clip_actions'])
            runner = load_runner_cls(TASK_ID)(env, deepcopy(agent), None, 'cuda:0')
            runner.load(plan['checkpoint'], strict=True, map_location='cuda:0')
            check_loaded_runner(runner, raw, checkpoint)
            report['model_loaded'] = {key: module_digest(getattr(runner.alg, key), torch) for key in ('actor', 'critic')}
            report['restoration'] = {
                'checkpoint': plan['checkpoint'], 'checkpoint_sha256': checksum,
                'adam': 'Loaded and compared exactly by frozen check_loaded_runner; step prohibited',
                'learning_rate': runner.alg.learning_rate,
                'saved_iteration_loaded': runner.current_learning_iteration,
                'saved_common_step_counter_loaded': raw.common_step_counter,
                'diagnostic_counter_rule': 'Reset only evaluation counters to zero; run 10 controls',
                'recurrent_state_rule': 'Original policy.reset after raw.reset(seed=8101)',
                'assistance_rule': 'Frozen dev config, hold 0; checked every control step',
                'training_progress_not_advanced': True,
            }
            policy = runner.get_inference_policy()
            raw.common_step_counter = raw._sim_step_counter = 0
            raw.episode_length_buf.zero_()
            raw.reset(seed=8101)
            policy.reset()
            initial = seal_initial_states(raw, 'dev', folder / 'initial-states')
            write(output / 'initial-receipt.json', initial)
            obs = env.get_observations()
            require(raw.num_envs == 128 and raw.max_episode_length == 500 and abs(raw.step_dt - .02) < 1e-12
                    and raw.cfg.decimation == 10 and abs(raw.physics_dt - .002) < 1e-12,
                    'Frozen environment count, episode length or control timestep differs')
            alive = torch.ones(raw.num_envs, dtype=torch.bool, device='cuda:0')
            observer_type = observer_class(MetricObserver, output,
                                          dict(protocol='dev', checkpoint_sha256=checksum, source_head=core.BASE, new_ppo_updates=0),
                                          kinematic_calibration(core.REPO))
            observer = observer_type(raw, alive)
            raw.metrics_manager.compute = observer.compute
            trace = TraceWriter(output / 'trace')
            capture = Capture(raw, runner, policy, observer.bam, trace, torch, output)
            report['dependency_sources_sha256'] = sha(output / 'dependency-sources.json')
            parameter_digest = {key: module_digest_parameters(getattr(runner.alg, key)) for key in ('actor', 'critic')}
            write(output / 'result.json', report)
            capture.emit('reset_ready', substep=0, physical=True, bam=True, contact=True, action=True, obs=obs)
            capture.install()
            with torch.inference_mode():
                for control in range(1, 11):
                    capture.control = control
                    require(not bool(raw.command_manager.get_command('twist').any()), 'Nonzero evaluation command')
                    require(not bool(raw._basketball_state.hold.any()), 'Nonzero evaluation assistance')
                    capture.emit('policy_before', substep=0, obs=obs)
                    actions = policy(obs)
                    require(tuple(actions.shape) == (128, 14), 'Policy action dimensions changed')
                    capture.emit('policy_after', substep=0, obs=obs, extras={'policy_action': actions})
                    obs, rewards, dones, _ = env.step(actions)
                    require_finite((actions, obs, rewards), 'short diagnostic rollout')
                    require(not bool(dones.any()), 'Episode ended inside 0.2 s; retain partial trace for review')
                    require(not bool(raw.termination_manager.get_term('nan_state').any()), 'NaN termination')
                    assert_zero_assistance(raw)
                    capture.emit('control_return', substep=10, physical=True, obs=obs,
                                 extras={'rewards': rewards, 'dones': dones, 'old_metric_row': observer.rows[-1]})
                    report['physics_steps'] = capture.steps
                    report['control_steps'] = control
                    write(output / 'result.json', report)
            capture.verify_counts()
            require(capture.steps == raw._sim_step_counter == 100 and raw.common_step_counter == 10,
                    'Short diagnostic counters differ')
            require(len(observer.rows) == 10, 'Metric observer did not execute exactly 10 times')
            require(trace.events == 561, 'Unexpected trace event count')
            for key in ('actor', 'critic'):
                require(module_digest_parameters(getattr(runner.alg, key)) == parameter_digest[key], 'Model weights changed')
            require_equal(runner.alg.save()['optimizer_state_dict'], checkpoint['optimizer_state_dict'], 'Adam after short replay')
            require(runner.current_learning_iteration == checkpoint['iter'], 'Training iteration changed')
            require(runner.alg.learning_rate == agent['algorithm']['learning_rate'], 'Learning rate changed')
            report['backend_after'] = core.backend_snapshot(torch)
            report['backend_reader_after'] = core.precision_reader_audit(torch)
            require(report['backend_after'] == report['backend_before'], 'Backend settings changed during diagnostic')
            capture.hooks.close()
            trace.close()
            report.update(status='SHORT_TRACE_COMPLETE_REVIEW_REQUIRED',
                          physics_steps=capture.steps, control_steps=10, sim_forward_calls=capture.forwards,
                          bam_calls=capture.bam_calls, action_process_calls=capture.actions,
                          action_apply_calls=capture.applies,
                          recorded_events=trace.events, unique_trace_arrays=len(trace.blobs),
                          raw_unique_array_bytes=trace.total, schema=capture.schema,
                          model_parameters_unchanged=True, adam_state_unchanged=True)
            core.verify_source()
    except BaseException as exc:
        report.update(status='FAILED_REVIEW_REQUIRED', error=type(exc).__name__ + ': ' + str(exc))
        if capture is not None:
            report['physics_steps'] = capture.steps
        raise
    finally:
        if capture is not None:
            capture.hooks.close()
        if trace is not None:
            trace.close()
        report['finished_utc'] = utc()
        write(output / 'result.json', report)
        if observer is not None:
            raw.metrics_manager.compute = observer.original
        if env is not None:
            env.close()
        elif raw is not None:
            raw.close()


def module_digest_parameters(model):
    import hashlib
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        a = value.detach().cpu().numpy()
        digest.update(name.encode() + b'\0' + a.tobytes())
    return digest.hexdigest()


def validate_result(output):
    from stage10_trace import report_for_comparison
    report = report_for_comparison(output)
    require((report['status'] == 'SHORT_TRACE_COMPLETE_REVIEW_REQUIRED' or report['reviewed_trace_reuse'])
            and report['scope'] == SCOPE,
            'Short diagnostic is not complete')
    require(report['new_ppo_updates'] == 0 and report['physics_steps'] == 100 and report['control_steps'] == 10,
            'Short diagnostic counts differ')
    require(report['checkpoint_sha256'] == core.PRIMARY and report['dataset_sha256'] == core.INITIAL,
            'Result input identity differs')
    return report


def package_review(folder):
    # Include every completed or partial short trace; no model copies or cloud files.
    paths = [p for p in sorted(folder.rglob('*')) if p.is_file() and
             not p.name.startswith('review-') and p != folder / 'initial-states/dev.pt']
    for p in paths:
        require(not p.is_symlink(), 'Refuse linked review input: ' + str(p))
    manifest = folder / 'review-files.json'
    write(manifest, [{'path': str(p.relative_to(folder)), 'size_bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths])
    paths.append(manifest)
    groups, group, size = [], [], 0
    for p in paths:
        n = p.stat().st_size
        require(n < 48 * 1024**2, 'Review item exceeds part limit: ' + str(p))
        if group and size + n > 48 * 1024**2:
            groups.append(group)
            group, size = [], 0
        group.append(p)
        size += n
    if group:
        groups.append(group)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
    outputs = []
    for number, group in enumerate(groups, 1):
        path = folder / f'review-{stamp}-part{number:02d}.zip'
        with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for p in group:
                archive.write(p, p.relative_to(folder))
        require(path.stat().st_size < 49 * 1024**2, 'Review ZIP exceeds 49 MiB')
        outputs.append({'path': str(path), 'sha256': sha(path), 'size_bytes': path.stat().st_size})
    for row in outputs:
        print('Stage10ReviewZIP=' + row['path'], flush=True)
        print('Stage10ReviewSHA256=' + row['sha256'], flush=True)
    print('Stage10ReviewParts=' + str(len(outputs)), flush=True)
    return outputs


def consumed_physics(parent):
    # A partial physics attempt is not silently repeated on --resume.
    for path in parent.glob('attempt-*/trace/events.jsonl'):
        if '"physics_before"' in path.read_text():
            return True
    return any(read(p).get('physics_steps', 0) > 0 for p in parent.glob('attempt-*/result.json'))


def controller(resume=None, package_only=None):
    import fcntl
    core.verify_source()
    core.ART.mkdir(parents=True, exist_ok=True)
    with (core.ART / 'local-diagnostic.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another Stage 10 local controller holds the lock')
        if package_only:
            folder = package_only.resolve()
            require(folder.is_relative_to((core.ART / 'substep').resolve()), 'Package only Stage 10 substep evidence')
            validate_plan(folder)
            package_review(folder)
            return
        checkpoint, dataset, sidecar = core.resolve_inputs()
        checked(sidecar, SIDECAR)
        require(shutil.disk_usage(core.REPO.parent).free >= 8 * 1024**3, 'Need 8 GiB free to retain and return traces')
        if resume:
            folder = resume.resolve()
            require(folder.is_relative_to((core.ART / 'substep').resolve()) and folder.is_dir(), 'Invalid resume directory')
            plan = validate_plan(folder)
            require(plan['checkpoint'] == str(checkpoint), 'Checkpoint path changed')
        else:
            folder = core.ART / 'substep' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:8])
            (folder / 'initial-states').mkdir(parents=True, exist_ok=False)
            shutil.copy2(dataset, folder / 'initial-states/dev.pt')
            shutil.copy2(sidecar, folder / 'initial-states/dev.json')
            (folder / 'tool').mkdir()
            for name in TOOLS:
                shutil.copy2(HERE / name, folder / 'tool' / name)
            plan = {'schema_version': 1, 'batch': 'stage10-substep-B', 'created_utc': utc(),
                    'source_head': core.BASE, 'tools': tool_hashes(), 'jobs': JOBS, 'scope': SCOPE,
                    'checkpoint': str(checkpoint), 'checkpoint_sha256': core.PRIMARY,
                    'dataset_sha256': core.INITIAL, 'dataset_receipt_sha256': SIDECAR,
                    'new_ppo_updates': 0, 'cloud_gpu_hours': 0,
                    'manual_cloud_shutdown_status': 'OFF_CONFIRMED_BY_USER',
                    'cloud_data_must_be_preserved': True, 'stage10_complete': False,
                    'prior_batch_review_sha256': '8c5ae3d437f22b685f8b4f528bcaf9939e85c94253ff2b16b54dc629336524f4'}
            write(folder / 'plan.json', plan)
        print('Stage10Output=' + str(folder), flush=True)
        print('Stage10Resume=' + shlex.join(['python3', str(SELF), '--resume', str(folder)]), flush=True)
        run = {'status': 'RUNNING', 'created_utc': utc(), 'scope': SCOPE, 'completed_jobs': [],
               'new_ppo_updates': 0, 'stage10_complete': False, 'cloud_contacted': False,
               'source_edited': False, 'stage06_smoke_run': False, 'job_review_status': {}}
        write(folder / 'run.json', run)
        try:
            outputs = []
            for job in JOBS:
                parent = folder / job
                parent.mkdir(exist_ok=True)
                receipt_path = parent / 'completed.json'
                if receipt_path.exists():
                    receipt = read(receipt_path)
                    require(Path(receipt['attempt']).name == receipt['attempt'], 'Unsafe receipt path')
                    output = parent / receipt['attempt']
                    core.verify_inventory(output, receipt['files'])
                    validate_result(output)
                else:
                    # Recover an atomically finished worker if the controller was interrupted.
                    finished = [p.parent for p in parent.glob('attempt-*/result.json')
                                if read(p)['status'] == 'SHORT_TRACE_COMPLETE_REVIEW_REQUIRED']
                    require(len(finished) <= 1, 'More than one completed attempt; return evidence')
                    if finished:
                        output = finished[0]
                        validate_result(output)
                    else:
                        require(not consumed_physics(parent),
                                'Partial physics trace exists; return review ZIPs before authorizing another replay')
                        require(len(list(parent.glob('attempt-*'))) < 3, 'Preflight attempt limit reached; return evidence')
                        attempt = 'attempt-' + uuid.uuid4().hex[:8]
                        output = parent / attempt
                        core.supervised([sys.executable, '-u', str(SELF), '--worker', str(folder / 'plan.json'),
                                         '--output', str(output)], parent / (attempt + '.log'), timeout_s=900)
                        validate_result(output)
                    write(receipt_path, {'attempt': output.name, 'completed_utc': utc(), 'files': core.inventory(output)})
                outputs.append(output)
                run['completed_jobs'].append(job)
                reviewed = validate_result(output)
                run['job_review_status'][job] = reviewed.get('review_status', reviewed['status'])
                write(folder / 'run.json', run)
                if reviewed['reviewed_trace_reuse']:
                    print(f'Stage10Substep={job}; existing 100-step trace reused; backend_after UNAVAILABLE; no replay', flush=True)
                else:
                    print(f'Stage10Substep={job}; 10 controls; 100 physics steps; zero PPO', flush=True)
            from stage10_trace import compare_all
            comparison = compare_all(outputs, folder / 'comparison.json')
            core.verify_source()
            validate_plan(folder)
            run['status'] = comparison['status']
        except BaseException as exc:
            run.update(status='STOPPED_REVIEW_REQUIRED', error=type(exc).__name__ + ': ' + str(exc))
            raise
        finally:
            run['finished_utc'] = utc()
            write(folder / 'run.json', run)
            package_review(folder)
            print('Stage10Batch=' + run['status'] + '; stage not complete; upload all printed review ZIP parts', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--resume', type=Path)
    group.add_argument('--package-only', type=Path)
    group.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    parser.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    require((args.worker is not None) == (args.output is not None), '--output is only for a supervised worker')
    python = core.REPO / '.venv/bin/python'
    require(python.is_file(), 'Existing project .venv missing; no packages will be installed')
    if Path(sys.prefix).resolve() != (core.REPO / '.venv').resolve():
        os.execv(str(python), [str(python), '-u', str(SELF), *sys.argv[1:]])
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.chdir(core.REPO)
    if args.worker:
        worker(args.worker, args.output.resolve())
    else:
        controller(args.resume, args.package_only)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print('Stage10Stopped=' + type(exc).__name__ + ': ' + str(exc), flush=True)
        traceback.print_exc()
        raise SystemExit(1)
