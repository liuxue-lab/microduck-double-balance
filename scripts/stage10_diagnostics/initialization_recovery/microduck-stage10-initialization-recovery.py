#!/usr/bin/env python3
"""Reuse reviewed C1 and capture C2/C3 once, without rollout or PPO."""
from pathlib import Path
import argparse
import fcntl
import os
import shlex
import shutil
import sys
import traceback
import uuid

import stage10_core as core
from stage10_core import read, write, require, sha, utc
from stage10_recovery_common import (
    HERE, REFERENCE, GAP_STATUS, expected_folder, verify_package,
    validate_saved_plan, validate_approved_first,
)
import stage10_initialization_worker as worker
from stage10_runtime import package_review

SELF = Path(__file__).resolve()
RECOVERY_DIR = 'initialization-recovery-v1'


def retain_recovery_tools(folder):
    dest = folder / RECOVERY_DIR / 'tool'
    require(not dest.is_symlink() and not dest.parent.is_symlink(), 'Linked recovery output')
    dest.mkdir(parents=True, exist_ok=True)
    files = [HERE / row['path'] for row in read(HERE / 'manifest.json')] + [HERE / 'manifest.json']
    for source in files:
        target = dest / source.name
        if target.exists():
            core.checked(target, sha(source))
    for source in files:
        target = dest / source.name
        if not target.exists():
            with target.open('xb') as stream:
                stream.write(source.read_bytes())
        core.checked(target, sha(source))
    approval = folder / RECOVERY_DIR / 'reviewed-first-trace.json'
    source = HERE / 'reviewed-first-trace.json'
    if not approval.exists():
        with approval.open('xb') as stream:
            stream.write(source.read_bytes())
    core.checked(approval, sha(source))
    return dest.parent


def controller(resume=None, package_only=False):
    core.verify_source()
    verify_package()
    folder = expected_folder() if resume is None else resume.resolve()
    require(folder == expected_folder(), 'Recovery is restricted to the existing reviewed batch')
    with (core.ART / 'local-diagnostic.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another Stage 10 local controller holds the lock')
        plan = validate_saved_plan(folder)
        worker.prior_review()
        first = folder / REFERENCE['approved_attempt']
        validate_approved_first(first)
        recovery = retain_recovery_tools(folder)
        print('Stage10Output=' + str(folder), flush=True)
        print('Stage10Resume=' + shlex.join(['python3', str(SELF), '--resume', str(folder)]), flush=True)
        if package_only:
            package_review(folder)
            return
        require(shutil.disk_usage(core.REPO.parent).free >= 8 * 1024**3, 'Need 8 GiB free for evidence')
        run = {'status': 'RUNNING', 'created_utc': utc(), 'scope': plan['scope'],
               'original_plan_unchanged': True, 'original_run_unchanged': True,
               'original_first_result': 'FAILED_REVIEW_REQUIRED', 'first_run_postcheck_gap_preserved': True,
               'first_trace_reused': str(first), 'completed_jobs': [], 'reviewed_jobs': [worker.JOBS[0]],
               'additional_processes_allowed': 2, 'new_ppo_updates': 0, 'rollout_physics_steps': 0,
               'stage10_complete': False, 'cloud_contacted': False, 'cloud_status': 'OFF_CONFIRMED_BY_USER',
               'source_edited': False, 'stage06_smoke_run': False}
        write(recovery / 'run.json', run)
        try:
            print('Stage10Initialization=01-initialization; reviewed trace reused; original postcheck gap retained', flush=True)
            outputs = [first]
            for job in worker.JOBS[1:]:
                parent = folder / job
                require(not parent.is_symlink(), 'Linked job folder refused')
                parent.mkdir(exist_ok=True)
                output = worker.choose_existing(parent)
                reused = output is not None
                if output is None:
                    attempt = 'attempt-' + uuid.uuid4().hex[:8]
                    output = parent / attempt
                    core.supervised([sys.executable, '-u', str(SELF), '--worker', str(folder / 'plan.json'),
                                     '--output', str(output)], parent / (attempt + '.log'), timeout_s=900)
                report = worker.validate_result(output)
                require(report.get('initialization_assistance_check', {}).get('status') ==
                        'PASS_UNPROCESSED_INITIALIZATION_ONLY', 'Missing initialization assistance check')
                if not (parent / 'completed.json').exists():
                    write(parent / 'completed.json', {'attempt': output.name, 'completed_utc': utc(),
                                                     'files': core.inventory(output)})
                outputs.append(output)
                run['completed_jobs'].append(job)
                write(recovery / 'run.json', run)
                print(f'Stage10Initialization={job}; {"reused" if reused else "captured"}; '
                      f'{report["recorded_events"]} events; zero rollout; zero PPO', flush=True)
            from stage10_init_compare import compare_all
            comparison = compare_all(outputs, recovery / 'comparison.json')
            require(comparison['status'] == GAP_STATUS, 'Recovery comparison lost the original evidence gap')
            core.verify_source()
            validate_saved_plan(folder)
            validate_approved_first(first)
            run['source_recheck_after'] = 'PASS_101_FROZEN_FILES_AT_RECOVERY_TIME'
            run['status'] = GAP_STATUS
        except BaseException as exc:
            run.update(status='STOPPED_REVIEW_REQUIRED', error=type(exc).__name__ + ': ' + str(exc))
            raise
        finally:
            run['finished_utc'] = utc()
            write(recovery / 'run.json', run)
            package_review(folder)
            print('Stage10Batch=' + run['status'] + '; stage not complete; upload all printed review ZIP parts', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument('--resume', type=Path)
    g.add_argument('--worker', type=Path, help=argparse.SUPPRESS)
    p.add_argument('--package-only', action='store_true')
    p.add_argument('--output', type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    require((args.worker is not None) == (args.output is not None) and not (args.worker and args.package_only),
            'Worker arguments must be paired and cannot package')
    python = core.REPO / '.venv/bin/python'
    require(python.is_file(), 'Existing project venv missing; no packages will be installed')
    if Path(sys.prefix).resolve() != (core.REPO / '.venv').resolve():
        os.execv(str(python), [str(python), '-u', str(SELF), *sys.argv[1:]])
    os.environ.setdefault('MUJOCO_GL', 'egl')
    os.environ.setdefault('MPLBACKEND', 'Agg')
    os.chdir(core.REPO)
    if args.worker:
        require(args.worker == expected_folder() / 'plan.json', 'Unexpected worker plan')
        worker.worker(args.worker, args.output.resolve())
    else:
        controller(args.resume, args.package_only)


if __name__ == '__main__':
    try:
        main()
    except (Exception, KeyboardInterrupt) as exc:
        print('Stage10Stopped=' + type(exc).__name__ + ': ' + str(exc), flush=True)
        traceback.print_exc()
        raise SystemExit(1)
