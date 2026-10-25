"""Stage 08 cloud runner: controlled fine-tuning, evaluation and bounded capacity.

PPO and the registered task are reused. Public jobs run under an independent
process watchdog; local 5060 training is rejected by the cloud preflight.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time

from mjlab_microduck.double_balance_stage08_budget import BudgetLedger, atomic_json, create_ledger, supervise
from mjlab_microduck.double_balance_stage08_plan import NUM_ENVS, SOURCE_SHA256, TRAIN_SEEDS, PROFILES

MODULE = 'mjlab_microduck.double_balance_stage08'

def gpu_sample():
    text = subprocess.check_output(['nvidia-smi','--id=0','--query-gpu=memory.total,memory.used,uuid',
                                    '--format=csv,noheader,nounits'], text=True, timeout=5)
    total, used, uuid = text.strip().split(',')
    return {'total_mib':int(total), 'used_mib':int(used), 'uuid':uuid.strip()}


class GPUSampler:
    interval_seconds = 2.
    max_consecutive_timeouts = 3

    def __init__(self, output):
        self.path = Path(output)/'gpu-samples.jsonl'
        self.stop = threading.Event()
        self.peak, self.samples, self.error = 0, 0, None
        self.initial = gpu_sample()
        self.timeouts, self.consecutive_timeouts = 0, 0
        self.max_consecutive_timeouts_seen = 0
        self.thread = threading.Thread(target=self.run, daemon=True)

    def record_sample(self, sample, stream, *, final=False):
        if sample['uuid'] != self.initial['uuid'] or sample['total_mib'] != self.initial['total_mib']:
            raise ValueError('GPU allocation changed')
        self.peak = max(self.peak, sample['used_mib'])
        self.samples += 1
        self.consecutive_timeouts = 0
        stream.write(json.dumps({'epoch':time.time(), 'event':'sample', 'final':final, **sample})+'\n')
        stream.flush()

    def run(self):
        try:
            with self.path.open('x') as stream:
                while not self.stop.is_set():
                    try:
                        sample = gpu_sample()
                    except subprocess.TimeoutExpired as exc:
                        self.timeouts += 1
                        self.consecutive_timeouts += 1
                        self.max_consecutive_timeouts_seen = max(self.max_consecutive_timeouts_seen,
                                                                 self.consecutive_timeouts)
                        stream.write(json.dumps({'epoch':time.time(), 'event':'query_timeout',
                            'consecutive_timeouts':self.consecutive_timeouts, 'error':str(exc)})+'\n')
                        stream.flush()
                        if self.consecutive_timeouts >= self.max_consecutive_timeouts:
                            raise RuntimeError('GPU sampling timed out three consecutive times') from exc
                    else:
                        self.record_sample(sample, stream)
                    self.stop.wait(self.interval_seconds)
        except BaseException as exc:
            self.error = str(exc)

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive() or self.error or not self.samples:
            raise ValueError(f'GPU sampling failed: {self.error}')
        # A final live query must succeed. Do not certify a stale last sample.
        with self.path.open('a') as stream:
            self.record_sample(gpu_sample(), stream, final=True)
        return {**self.initial, 'sampled_peak_gpu_used_mib':self.peak, 'samples':self.samples,
                'query_timeouts':self.timeouts,
                'max_consecutive_timeouts':self.max_consecutive_timeouts_seen,
                'sampling_interval_seconds':self.interval_seconds,
                'sampling_status':'RECOVERED_WITH_GAPS' if self.timeouts else 'PASS',
                # Gaps may be tolerated in an already admitted formal run, but
                # must never be used to certify a new capacity measurement.
                'eligible_with_15_percent_headroom':not self.timeouts and self.peak <= .85*self.initial['total_mib']}


def measured_capacity(path, gpu, head):
    from mjlab_microduck.double_balance_training import validate_capacity
    report = json.loads(Path(path).read_text())
    if gpu == 'A800':
        row = validate_capacity(report, NUM_ENVS)
        return {'seconds_per_update':max(4.745, row['mean_iteration_seconds']),
                'source':'reused Stage 07 A800 measurement', 'summary':str(path)}
    if report.get('status') != 'PASS' or report.get('source_sha256') != SOURCE_SHA256:
        raise ValueError('5090 capacity has not passed for the update 1000 source')
    if report.get('git_head') != head or report.get('gpu') != '5090' or report.get('num_envs') != NUM_ENVS:
        raise ValueError('5090 capacity source/GPU/batch differs')
    if set(report['cases']) != {'D','E'}:
        raise ValueError('Capacity must cover assisted and fully unassisted paths')
    current = gpu_sample()
    for case in report['cases'].values():
        if (case['status'] != 'CAPACITY_COMPLETE' or case['gpu_memory']['uuid'] != current['uuid'] or
                not case['gpu_memory']['eligible_with_15_percent_headroom'] or
                not case['resident_development_evaluation_passed'] or not case['save_reload_passed'] or
                not case['trained_save_reload_passed']):
            raise ValueError('Incomplete 5090 capacity evidence')
    seconds = max(c['max_measured_iteration_seconds'] for c in report['cases'].values())
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('Invalid measured update duration')
    return {'seconds_per_update':seconds, 'source':'Stage 08 5090 D/E including resident evaluation', 'summary':str(path)}


def run_development(checkpoint, output, args, completed, ledger):
    from mjlab_microduck.double_balance_stage08_evaluation import PROTOCOL_VERSION, selection_rank
    from mjlab_microduck.double_balance_stage08_state import file_sha256, require
    results = {}
    for protocol in ('nominal','dev'):
        directory = output/'evaluations'/f'update_{completed:06d}'/protocol
        directory.parent.mkdir(parents=True, exist_ok=True)
        timeout = min(900., ledger.remaining(training=True) - 120)
        require(timeout >= 30, 'Not enough time for development evaluation and checkpoint exit')
        argv = [sys.executable,'-u','-m',MODULE,'evaluate','--gpu',args.gpu,
                '--ledger',str(args.ledger),'--checkpoint',str(checkpoint),'--output',str(directory),
                '--datasets',str(args.datasets),'--protocol',protocol,'--worker']
        with directory.with_suffix('.log').open('x') as stream:
            subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=True, timeout=timeout)
        report = json.loads((directory/'evaluation.json').read_text())
        require(report['status'] == 'PASS' and report['checkpoint_sha256'] == file_sha256(checkpoint),
                'Development evaluation failed or evaluated another model')
        results[protocol] = report
    rank = selection_rank(results['nominal'], results['dev'], completed)
    best_path = output/'best-development.json'
    previous = json.loads(best_path.read_text()) if best_path.exists() else None
    if previous is None or tuple(previous['rank']) < rank:
        atomic_json(best_path, {'protocol_version':PROTOCOL_VERSION, 'checkpoint':str(checkpoint),
                    'sha256':file_sha256(checkpoint), 'experiment_completed_updates':completed,
                    'rank':rank, 'scope':'this run segment only; held-out data unused'})
    return results


class PauseAtBoundary(Exception):
    pass


def train_worker(args, *, capacity=False):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.os import dump_yaml
    from mjlab_microduck.double_balance_stage07 import monitor_updates
    from mjlab_microduck.double_balance_stage08_state import (
        ARTIFACTS, TASK_ID, build_training_config, check_loaded_runner, check_optimizer,
        environment_preflight, file_sha256, load_checked, require, restore, runtime_profile, save, utc_now,
        check_training_step, load_runner_checkpoint,
    )
    from mjlab_microduck.double_balance_stage08_review import validate_decision, deteriorated
    head = environment_preflight(args.gpu)
    ledger = BudgetLedger(args.ledger)
    require(ledger.snapshot()['gpu'] == args.gpu, 'Budget GPU differs')
    require(ledger.path.is_relative_to(ARTIFACTS), 'Place campaign ledger on the cloud data disk')
    checkpoint, state, checksum = load_checked(args.checkpoint, resume=args.resume, profile=args.profile,
                                              seed=args.seed, campaign_id=ledger.identity if args.resume else None)
    start = state['experiment_completed_updates'] if args.resume else 0
    require((capacity and not args.resume and args.profile in ('D','E') and args.target_updates == 15) or
            (not capacity and 1 <= args.target_updates <= 4000 and 0 <= start <= args.target_updates),
            'Invalid target or capacity request')
    if not capacity:
        require(args.gpu == '5090' or args.target_updates <= 1500, 'A800 plan ends after core replication')
        require(args.capacity_summary is not None, 'A capacity summary is required')
        measured = measured_capacity(args.capacity_summary, args.gpu, head)
        decision = validate_decision(args.decision, profile=args.profile, target=args.target_updates,
                                     campaign_id=ledger.identity)
    else:
        measured, decision = {'seconds_per_update':30., 'source':'conservative bounded probe'}, None
    require(ledger.admit(1, measured['seconds_per_update'], 180), 'Insufficient training time remaining')
    output = args.output.resolve()
    require(output.is_relative_to(ARTIFACTS), 'Cloud outputs must use the Stage 08 artifact directory')
    require(args.datasets.resolve().is_relative_to(ARTIFACTS), 'Datasets belong in the Stage 08 artifact directory')
    output.mkdir(parents=True, exist_ok=False)
    cfg, agent = build_training_config(args.profile,args.seed,checkpoint)
    agent.update(max_iterations=1000+args.target_updates, run_name=f'stage08_{args.profile}_{args.seed}')
    dump_yaml(output/'params/env.yaml', asdict(cfg))
    dump_yaml(output/'params/agent.yaml', deepcopy(agent))
    report = {'status':'INITIALIZING','git_head':head, 'gpu':args.gpu, 'campaign_id':ledger.identity,
              'profile':args.profile, 'seed':args.seed, 'purpose':'capacity' if capacity else 'formal',
              'source_sha256':SOURCE_SHA256, 'input_checkpoint':str(args.checkpoint), 'input_sha256':checksum,
              'start_completed_updates':start, 'experiment_completed_updates':start,
              'target_updates':args.target_updates,'formal_training_started':False,
              'exact_trajectory_resume':False, 'created_utc':utc_now(), 'measured_capacity':measured,
              'decision':decision, 'stage08_complete':False}
    atomic_json(output/'training.json',report)
    raw = env = runner = sampler = None
    pause = {'requested':False}
    handlers = {sig:signal.signal(sig, lambda *_:pause.update(requested=True)) for sig in (signal.SIGINT,signal.SIGTERM)}
    began = time.monotonic()
    save_times, update_times, evaluation_times, results_history = [],[],[],[]
    try:
        sampler = GPUSampler(output)
        sampler.thread.start()
        raw = ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
        env = RslRlVecEnvWrapper(raw,clip_actions=agent['clip_actions'])
        runner = load_runner_cls(TASK_ID)(env,deepcopy(agent),str(output/'tensorboard'),'cuda:0')
        load_runner_checkpoint(runner,args.checkpoint,map_location='cuda:0')
        restore(runner,env,checkpoint,profile=args.profile,seed=args.seed,campaign_id=ledger.identity,resume=args.resume)
        registered_save = runner.save
        runner.save = lambda *a,**k:None  # completed-update schedule below owns every save
        runner.logger.logger_type = 'tensorboard'

        def persist(completed):
            began_save = time.monotonic()
            path = save(runner,env,output,completed,registered_save,profile=args.profile,seed=args.seed,
                        campaign_id=ledger.identity,head=head,purpose=report['purpose'])
            save_times.append(time.monotonic()-began_save)
            report['latest_checkpoint'] = str(path)
            return path

        initial = persist(start)
        # Explicit reload into the registered runner, with tensor/Adam/LR comparisons.
        reloaded = torch.load(initial,map_location='cpu',weights_only=False)
        load_runner_checkpoint(runner,initial,map_location='cuda:0')
        check_loaded_runner(runner,raw,reloaded)
        restore(runner,env,reloaded,profile=args.profile,seed=args.seed,campaign_id=ledger.identity,resume=True)
        report['save_reload_passed'] = True
        t = time.monotonic()
        initial_eval = run_development(initial,output,args,start,ledger)
        evaluation_times.append(time.monotonic()-t)
        results_history.append(initial_eval['dev'])
        report['resident_development_evaluation_passed'] = True
        if not capacity and start == args.target_updates:
            report.update(status='TRAINING_COMPLETE',completion_recovery_only=True,
                          optimizer_steps_this_segment=0)
            return report
        baseline = initial_eval['dev']
        if decision is not None:
            baseline = json.loads(Path(decision['baseline_report']['path']).read_text())
        # Capture both current commands and live curriculum cfgs throughout the run.
        original_step = env.step
        def step(actions):
            result = original_step(actions)
            check_training_step(raw,args.profile)
            return result
        env.step = step
        report.update(status='RUNNING',formal_training_started=not capacity)
        atomic_json(output/'training.json',report)

        def updated(row,evidence):
            completed = row['completed_updates']-1000
            update_times.append(row['iteration_seconds'])
            report.update(experiment_completed_updates=completed,lineage_completed_updates=1000+completed,
                          latest_update=row,live_profile=runtime_profile(raw,args.profile),
                          optimizer_steps_this_segment=evidence['optimizer_steps'],budget=ledger.snapshot())
            if sampler.error:
                persist(completed)
                raise RuntimeError('GPU monitor failed: '+sampler.error)
            if completed % 100 == 0 or completed % 250 == 0 or completed == args.target_updates:
                persist(completed)
            remaining_updates = min(250-completed%250,args.target_updates-completed)
            # Startup compilation is already counted by the wall-clock ledger;
            # do not project that one-time cost onto every subsequent update.
            steady = update_times[3:][-20:]
            seconds = max(measured['seconds_per_update'],max(steady,default=0.))
            overhead = max(30.,max(save_times,default=0)*2)
            if remaining_updates:
                overhead += max(evaluation_times,default=0)
            stop_reason = None
            if pause['requested'] or (output/'STOP').exists():
                stop_reason = 'requested_stop'
            elif remaining_updates and not ledger.admit(remaining_updates,seconds,overhead):
                stop_reason = 'insufficient_time_for_next_evaluation_block'
            if stop_reason:
                persist(completed)
                report['pause_reason'] = stop_reason
                raise PauseAtBoundary()
            if not capacity and (completed%250 == 0 or completed == args.target_updates):
                t = time.monotonic()
                evaluations = run_development(persist(completed),output,args,completed,ledger)
                evaluation_times.append(time.monotonic()-t)
                results_history.append(evaluations['dev'])
                report['latest_evaluation'] = {p:{k:v for k,v in r.items() if k != 'episodes'} for p,r in evaluations.items()}
                if args.target_updates>500 and len(results_history)>=3 and all(deteriorated(r,baseline) for r in results_history[-2:]):
                    report['pause_reason'] = 'two_consecutive_development_regressions'
                    raise PauseAtBoundary()
            atomic_json(output/'training.json',report)

        count = args.target_updates-start
        # Admission is based on measured initial evaluation/save costs before the first rollout.
        if not ledger.admit(min(count,250),measured['seconds_per_update'],max(evaluation_times)+max(save_times)+120):
            report['pause_reason'] = 'insufficient_time_after_initial_evaluation'
            raise PauseAtBoundary()
        with monitor_updates(runner,env,output,count,start_completed=1000+start,
                             on_update=updated,prefix='Stage08Update') as evidence:
            runner.learn(num_learning_iterations=count,init_at_random_ep_len=False)
        require(len(evidence['iterations']) == count and evidence['return_passes'] == count and
                evidence['vector_steps'] == count*24 and evidence['optimizer_steps'] == count*20,
                'PPO rollout/update counts differ')
        check_optimizer(runner.alg.optimizer,(1000+args.target_updates)*20)
        final_checkpoint=persist(args.target_updates)
        require(file_sha256(args.checkpoint) == checksum,'Input checkpoint changed')
        report['status'] = 'CAPACITY_COMPLETE' if capacity else 'TRAINING_COMPLETE'
        if capacity:
            updated_checkpoint=torch.load(final_checkpoint,map_location='cpu',weights_only=False)
            load_runner_checkpoint(runner,final_checkpoint,map_location='cuda:0')
            check_loaded_runner(runner,raw,updated_checkpoint)
            check_optimizer(runner.alg.optimizer,(1000+args.target_updates)*20)
            measured_times = update_times[3:]
            require(len(measured_times)==12,'Capacity must have 3 warmup + 12 measured updates')
            report.update(mean_measured_iteration_seconds=statistics.mean(measured_times),
                          max_measured_iteration_seconds=max(measured_times),
                          warmup_updates=3,measured_updates=12,optimization_discarded=True,
                          trained_save_reload_passed=True)
    except PauseAtBoundary:
        report['status'] = 'PAUSED'
    except BaseException as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for sig,handler in handlers.items():
            signal.signal(sig,handler)
        if sampler is not None:
            try:
                report['gpu_memory'] = sampler.close()
            except BaseException as exc:
                report.update(status='FAIL',sampler_error=str(exc))
        report.update(finished_utc=utc_now(),elapsed_seconds=time.monotonic()-began,
                      save_seconds=save_times,evaluation_seconds=evaluation_times)
        atomic_json(output/'training.json',report)
        if runner is not None and getattr(runner.logger,'writer',None) is not None:
            runner.logger.writer.close()
        if env is not None:
            env.close()
        elif raw is not None:
            raw.close()
        print('Stage08Training='+report['status'],flush=True)
    return report


def capacity(args):
    from mjlab_microduck.double_balance_stage08_state import environment_preflight, require, utc_now
    require(args.gpu == '5090','Reuse A800 Stage 07 capacity; do not remeasure it')
    head = environment_preflight(args.gpu)
    ledger = BudgetLedger(args.ledger)
    args.output.mkdir(parents=True,exist_ok=False)
    report = {'status':'RUNNING','gpu':'5090','num_envs':4096,'source_sha256':SOURCE_SHA256,
              'git_head':head,'formal_training_started':False,'created_utc':utc_now(),'cases':{}}
    try:
        for profile in ('D','E'):
            directory = args.output/profile
            argv = [sys.executable,'-u','-m',MODULE,'capacity-worker','--gpu',args.gpu,
                    '--ledger',str(args.ledger),'--checkpoint',str(args.checkpoint),'--output',str(directory),
                    '--datasets',str(args.datasets),'--profile',profile,'--seed',str(TRAIN_SEEDS[0]),
                    '--target-updates','15','--worker']
            result = supervise(argv,ledger,training=True,timeout_seconds=900,grace_seconds=30)
            require(result['exit_code']==0 and result['watchdog_reason'] is None,'Capacity worker failed/timed out')
            case = json.loads((directory/'training.json').read_text())
            require(case['status']=='CAPACITY_COMPLETE' and case['gpu_memory']['eligible_with_15_percent_headroom'],
                    '4096-env case failed or lacks 15% headroom')
            report['cases'][profile] = case
            atomic_json(args.output/'capacity-summary.json',report)
        report['status'] = 'PASS'
    except BaseException as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        atomic_json(args.output/'capacity-summary.json',report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='mode',required=True)
    init = sub.add_parser('init-budget')
    init.add_argument('--gpu',choices=('A800','5090'),required=True)
    init.add_argument('--started-at',required=True,help='Retained deployment budget start, ISO timestamp including timezone')
    init.add_argument('--ledger',type=Path,required=True)
    for mode in ('train','evaluate','capacity','capacity-worker'):
        p = sub.add_parser(mode)
        p.add_argument('--gpu',choices=('A800','5090'),required=True)
        p.add_argument('--ledger',type=Path,required=True)
        p.add_argument('--checkpoint',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        p.add_argument('--datasets',type=Path,required=True)
        p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
        if mode in ('train','capacity-worker'):
            p.add_argument('--profile',choices=tuple(PROFILES),required=True)
            p.add_argument('--seed',type=int,choices=TRAIN_SEEDS,required=True)
            p.add_argument('--target-updates',type=int,required=True)
            p.add_argument('--resume',action='store_true')
            p.add_argument('--capacity-summary',type=Path)
            p.add_argument('--decision',type=Path)
        if mode == 'evaluate':
            p.add_argument('--protocol',choices=('nominal','dev','test-8201','test-8202','test-8203'),required=True)
            p.add_argument('--selection',type=Path)
    args = parser.parse_args()
    if args.mode == 'init-budget':
        print(json.dumps(create_ledger(args.ledger,args.gpu,args.started_at),indent=2))
        return
    args.checkpoint,args.output,args.datasets,args.ledger = (p.resolve() for p in
        (args.checkpoint,args.output,args.datasets,args.ledger))
    if args.mode == 'capacity':
        capacity(args)
        return
    if not args.worker:
        result = supervise([sys.executable,'-u','-m',MODULE,*sys.argv[1:],'--worker'],
                           BudgetLedger(args.ledger),training=args.mode!='evaluate',
                           timeout_seconds=900 if args.mode=='evaluate' else None)
        if args.output.is_dir() and not (args.output/'watchdog.json').exists():
            atomic_json(args.output/'watchdog.json',result)
        print('Stage08Watchdog='+json.dumps(result),flush=True)
        raise SystemExit(result['exit_code'] if result['exit_code']>=0 else 128-result['exit_code'])
    if 'STAGE08_SUPERVISOR_PID' not in os.environ:
        raise ValueError('Workers must be launched through the public watchdog entry')
    if args.mode == 'evaluate':
        from mjlab_microduck.double_balance_stage08_state import ARTIFACTS,environment_preflight,require
        from mjlab_microduck.double_balance_stage08_evaluation import evaluate_one
        head = environment_preflight(args.gpu)
        ledger = BudgetLedger(args.ledger)
        require(ledger.snapshot()['gpu']==args.gpu and ledger.remaining(training=False)>30,'No evaluation budget')
        require(args.output.is_relative_to(ARTIFACTS) and args.datasets.is_relative_to(ARTIFACTS),'Use cloud artifact paths')
        evaluate_one(args.checkpoint,args.output,args.protocol,args.datasets,device='cuda:0',head=head,selection=args.selection)
    else:
        report = train_worker(args,capacity=args.mode=='capacity-worker')
        if report['status']=='FAIL':
            raise SystemExit(1)


if __name__ == '__main__':
    main()
