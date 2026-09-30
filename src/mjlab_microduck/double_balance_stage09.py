"""Approved Stage 09 cloud pilot and laptop zero-PPO checks. No power operations."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid
import zipfile

from mjlab_microduck.double_balance_stage09_plan import PROFILES, PRIMARY_SHA, SEED, TERMS
from mjlab_microduck.double_balance_stage09_budget import BudgetLedger, create_ledger, atomic_json, supervise

MODULE='mjlab_microduck.double_balance_stage09'


@contextmanager
def no_optimizer_steps():
    import torch
    originals={cls:cls.step for cls in (torch.optim.Adam,torch.optim.AdamW)}
    def reject(*args,**kwargs):
        raise RuntimeError('Optimizer.step is forbidden in zero-PPO inference')
    try:
        for cls in originals:cls.step=reject
        yield
    finally:
        for cls,method in originals.items():cls.step=method


def evaluate(args):
    from mjlab_microduck.double_balance_stage09_state import preflight
    from mjlab_microduck.double_balance_stage09_evaluation import evaluate_one
    head=preflight(cloud=not args.local)
    if not args.local:
        ledger=BudgetLedger(args.ledger)
        if ledger.remaining(training=False)<60:raise ValueError('No evaluation budget')
    with no_optimizer_steps():
        return evaluate_one(args.checkpoint,args.output,args.protocol,args.datasets,device='cuda:0',
                            head=head,selection=args.selection,video_env_id=args.video_env_id,
                            reward_probe=args.reward_probe)


class Pause(Exception):pass


def train(args):
    import torch
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_runner_cls
    from mjlab.utils.os import dump_yaml
    from mjlab_microduck.double_balance_stage07 import monitor_updates
    from mjlab_microduck.double_balance_stage08 import GPUSampler
    from mjlab_microduck.double_balance_stage09_state import (
        ARTIFACTS,TASK_ID,preflight,load_checked,build_config,load_runner_checkpoint,
        restore,save,require,check_optimizer,check_training_step,runtime_profile,file_sha256,utc_now,
    )
    head=preflight(cloud=True)
    ledger=BudgetLedger(args.ledger)
    require(args.output.resolve().is_relative_to(ARTIFACTS),'Use Stage 09 cloud artifact output')
    checkpoint,state,checksum=load_checked(args.checkpoint,profile=args.profile,campaign_id=ledger.identity)
    start=state['experiment_completed_updates'] if 'stage09' in checkpoint['infos'] else 0
    require(args.target in (250,500) and start<args.target,'Invalid milestone or already complete')
    require(ledger.admit(args.target-start,3.1,180),'Insufficient time for this training segment')
    charged=ledger.snapshot()['charged_updates'][args.profile]
    require(charged+(args.target-start)<=500,'Lost/repeated updates would exceed the approved branch cap')
    args.output.mkdir(parents=True,exist_ok=False)
    cfg,agent=build_config(args.profile,checkpoint)
    dump_yaml(args.output/'params/env.yaml',asdict(cfg));dump_yaml(args.output/'params/agent.yaml',agent)
    report=dict(status='INITIALIZING',profile=args.profile,campaign_id=ledger.identity,git_head=head,
                source_sha256=PRIMARY_SHA,input_sha256=checksum,start_completed_updates=start,
                target_updates=args.target,experiment_completed_updates=start,created_utc=utc_now(),
                exact_trajectory_resume=False,stage09_complete=False)
    raw=env=runner=sampler=None;pause={'requested':False};completed=start
    handlers={sig:signal.signal(sig,lambda *_:pause.update(requested=True)) for sig in (signal.SIGTERM,signal.SIGINT)}
    times=[];began=time.monotonic()
    atomic_json(args.output/'training.json',report)
    try:
        sampler=GPUSampler(args.output);sampler.thread.start()
        raw=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
        env=RslRlVecEnvWrapper(raw,clip_actions=agent['clip_actions'])
        runner=load_runner_cls(TASK_ID)(env,deepcopy(agent),str(args.output/'tensorboard'),'cuda:0')
        load_runner_checkpoint(runner,args.checkpoint,map_location='cuda:0')
        restore(runner,env,checkpoint,profile=args.profile,campaign_id=ledger.identity)
        native_save=runner.save;runner.save=lambda *a,**kw:None
        runner.logger.logger_type='tensorboard'
        def persist(n):
            path=save(runner,env,args.output,n,native_save,profile=args.profile,campaign_id=ledger.identity,head=head)
            report['latest_checkpoint']=str(path);atomic_json(args.output/'training.json',report)
            return path
        initial=persist(start)
        # This tests the exact source of the next rollout without taking a PPO step.
        copied=torch.load(initial,map_location='cpu',weights_only=False)
        load_runner_checkpoint(runner,initial,map_location='cuda:0')
        restore(runner,env,copied,profile=args.profile,campaign_id=ledger.identity)
        report.update(save_reload_passed=True,restored_learning_rate=runner.alg.learning_rate,
                      restored_adam_steps=(5000+start)*20,live_profile=runtime_profile(raw,args.profile))
        native_step=env.step
        def step(actions):
            result=native_step(actions);check_training_step(raw,'E');return result
        env.step=step
        native_update=runner.alg.update
        def update():
            if ledger.remaining(training=True)<=120:raise Pause('Training reserve reached')
            ledger.charge_update(args.profile)
            return native_update()
        runner.alg.update=update
        def updated(row,evidence):
            nonlocal completed
            completed=row['completed_updates']-5000
            times.append(row['iteration_seconds'])
            report.update(experiment_completed_updates=completed,latest_update=row,
                          charged_updates=ledger.snapshot()['charged_updates'])
            for name in TERMS:
                value=row.get('episode',{}).get('Episode_Reward/'+name)
                require(value is None or value<=1e-8,'Penalty became a positive reward')
            if completed%50==0 or completed==args.target:persist(completed)
            if sampler.error:raise RuntimeError('GPU sampling failed: '+sampler.error)
            steady=max(3.1,max(times[3:][-20:],default=0.))
            if pause['requested'] or (args.output/'STOP').exists() or (completed<args.target and
                    not ledger.admit(min(50,args.target-completed),steady,120)):
                persist(completed);raise Pause('Requested stop or next save block exceeds budget')
            atomic_json(args.output/'training.json',report)
        count=args.target-start
        report['status']='RUNNING';atomic_json(args.output/'training.json',report)
        with monitor_updates(runner,env,args.output,count,start_completed=5000+start,
                             on_update=updated,prefix='Stage09Update') as evidence:
            runner.learn(num_learning_iterations=count,init_at_random_ep_len=False)
        require(evidence['vector_steps']==count*24 and evidence['optimizer_steps']==count*20 and
                evidence['return_passes']==count and len(evidence['iterations'])==count,'PPO counts differ')
        check_optimizer(runner.alg.optimizer,(5000+args.target)*20)
        persist(args.target)
        require(file_sha256(args.checkpoint)==checksum,'Input checkpoint changed')
        report['status']='TRAINING_COMPLETE'
    except Pause as exc:
        report.update(status='PAUSED',reason=str(exc))
    except BaseException as exc:
        report.update(status='FAIL',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        for sig,handler in handlers.items():signal.signal(sig,handler)
        if sampler is not None:
            try:report['gpu_memory']=sampler.close()
            except BaseException as exc:report.update(status='FAIL',sampler_error=str(exc))
        report.update(finished_utc=utc_now(),elapsed_seconds=time.monotonic()-began,
                      experiment_completed_updates=completed)
        atomic_json(args.output/'training.json',report)
        if runner is not None and getattr(runner.logger,'writer',None) is not None:runner.logger.writer.close()
        if env is not None:env.close()
        elif raw is not None:raw.close()
    return report


def package_review(folder):
    from mjlab_microduck.double_balance_checkpoint import file_sha256
    import numpy as np
    folder=Path(folder);members=[];raw_files=[]
    for trace in sorted(folder.rglob('head-trace.npz')):
        raw_files.append(dict(path=str(trace),sha256=file_sha256(trace),size_bytes=trace.stat().st_size))
        selection=json.loads((trace.parent/'plot-selection.json').read_text())['env_ids']
        with np.load(trace,allow_pickle=False) as z:
            shape=z['active_first_episode'].shape
            sample={k:(z[k][:,selection] if z[k].ndim>=2 and z[k].shape[:2]==shape else z[k]) for k in z.files}
        sample['review_env_ids']=np.array(selection)
        sample['full_trace_sha256']=np.array(raw_files[-1]['sha256'])
        np.savez_compressed(trace.parent/'head-review-samples.npz',**sample)
    atomic_json(folder/'raw-trace-archive.json',dict(files=raw_files,retained_locally=True,full_traces_in_review_zip=False))
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.name not in ('head-trace.npz','review-files.json') and path.suffix in ('.json','.log','.txt','.yaml','.png','.mp4','.npz','.jsonl'):
            members.append(dict(path=str(path.relative_to(folder)),sha256=file_sha256(path),size_bytes=path.stat().st_size))
    atomic_json(folder/'review-files.json',members)
    archive=folder.with_suffix('.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for row in members:z.write(folder/row['path'],row['path'])
        z.write(folder/'review-files.json','review-files.json')
    return dict(path=str(archive),sha256=file_sha256(archive),checkpoint_files_included=False)


def local_check(args):
    from mjlab_microduck.double_balance_stage09_state import preflight,load_checked,build_config,verify_source,utc_now,file_sha256
    head=preflight(cloud=False)
    checkpoint,state,checksum=load_checked(args.checkpoint)
    if checksum!=PRIMARY_SHA:raise ValueError('Local preparation must use the approved primary source')
    args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='RUNNING',git_head=head,source_sha256=checksum,new_ppo_updates=0,
                learning_rate=state['learning_rate'],source_lineage_updates=5000,
                runtime_manifest_sha256=file_sha256(Path(__file__).resolve().parents[2]/'docs/audits/stage-09-runtime-hashes.json'),
                expected_adam_steps=100000,cloud_contacted=False,jobs=[],stage09_complete=False)
    for name in PROFILES:
        cfg,agent=build_config(name,checkpoint)
        if cfg.scene.num_envs!=4096:raise ValueError('Formal batch changed')
    atomic_json(args.output/'local-preflight.json',report)
    print('Stage09RestoredLearningRate='+str(state['learning_rate']),flush=True)
    # Separate processes keep allocator and renderer state isolated between probes.
    try:
        jobs=[dict(protocol='nominal',name='nominal',checkpoint=str(args.checkpoint),video=0),
              dict(protocol='dev',name='dev',checkpoint=str(args.checkpoint)),
              dict(protocol='nominal',name='reward-probe',checkpoint=str(args.checkpoint),probe=True)]
        if args.extra_jobs:
            extra=json.loads(args.extra_jobs.read_text())
            jobs+=extra['jobs']
            for item in jobs[:2]:item['cloud_report']=extra['primary_reports'][item['protocol']]
        for item in jobs:
            protocol=item['protocol'];probe=item.get('probe',False);name=item['name']
            argv=[sys.executable,'-u','-m',MODULE,'evaluate','--local','--checkpoint',item['checkpoint'],
                  '--datasets',str(args.datasets),'--output',str(args.output/name),'--protocol',protocol]
            if item.get('video') is not None:argv+=['--video-env-id',str(item['video'])]
            if item.get('selection'):argv+=['--selection',item['selection']]
            if probe:argv+=['--reward-probe']
            with (args.output/(name+'.log')).open('x') as log:
                proc=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=900)
            report['jobs'].append(dict(name=name,returncode=proc.returncode))
            if proc.returncode:raise RuntimeError('Local check failed: '+name)
            if item.get('cloud_report'):
                ref=item['cloud_report']
                if file_sha256(ref['path'])!=ref['sha256']:raise ValueError('Frozen cloud report changed')
                cloud=json.loads(Path(ref['path']).read_text())
                replay=json.loads((args.output/name/'evaluation.json').read_text())
                if cloud['checkpoint_sha256']!=replay['checkpoint_sha256']:raise ValueError('Replay model differs')
                indexed={e['env_id']:e for e in cloud['episodes']};comparisons=[]
                for e in replay['episodes']:
                    c=indexed[e['env_id']]
                    if c['initial_state_id']!=e['initial_state_id']:raise ValueError('Replay initial-state ID differs')
                    comparisons.append(dict(env_id=e['env_id'],initial_state_id=e['initial_state_id'],
                        cloud_success=c['success'],local_success=e['success'],cloud_terminated=c['terminated'],
                        local_terminated=e['terminated'],cloud_terminal_stable_s=c['terminal_stable_seconds'],
                        local_terminal_stable_s=e['terminal_stable_seconds']))
                atomic_json(args.output/name/'cloud-local-comparison.json',dict(episodes=comparisons,
                    original_cloud_report=ref,cloud_results_replaced=False))
        report['status']='ZERO_PPO_RUNTIME_CHECKS_COMPLETE_REVIEW_PENDING'
    except BaseException as exc:
        report.update(status='FAIL',error=str(exc));raise
    finally:
        report['finished_utc']=utc_now();atomic_json(args.output/'local-preflight.json',report)
        bundle=package_review(args.output)
        print('Stage09LocalCheck='+report['status'],flush=True)
        print('Stage09Review='+bundle['path'],flush=True)
        print('Stage09ReviewSHA256='+bundle['sha256'],flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('init-budget');p.add_argument('--ledger',type=Path,required=True);p.add_argument('--started-at',required=True)
    for mode in ('local-check','evaluate','train','campaign'):
        p=sub.add_parser(mode)
        p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
        p.add_argument('--datasets',type=Path,required=True);p.add_argument('--ledger',type=Path)
        p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
        if mode=='local-check':p.add_argument('--extra-jobs',type=Path)
        if mode=='campaign':p.add_argument('--local-report',type=Path,required=True)
        if mode=='train':p.add_argument('--profile',choices=PROFILES,required=True);p.add_argument('--target',type=int,choices=(250,500),required=True)
        if mode=='evaluate':
            p.add_argument('--local',action='store_true');p.add_argument('--protocol',required=True)
            p.add_argument('--selection',type=Path);p.add_argument('--video-env-id',type=int)
            p.add_argument('--reward-probe',action='store_true')
    args=parser.parse_args()
    if args.mode=='init-budget':
        from mjlab_microduck.double_balance_stage09_state import ARTIFACTS,preflight
        preflight(cloud=True)
        if args.ledger.resolve()!=ARTIFACTS/'ledger.json':raise ValueError('Use the single retained Stage 09 ledger')
        print(json.dumps(create_ledger(args.ledger,'5090',args.started_at)));return
    args.checkpoint,args.output,args.datasets=(p.resolve() for p in (args.checkpoint,args.output,args.datasets))
    os.environ.setdefault('MUJOCO_GL','egl')
    if args.mode=='local-check':local_check(args);return
    if args.mode=='evaluate' and args.local:evaluate(args);return
    from mjlab_microduck.double_balance_stage09_state import ARTIFACTS,preflight
    preflight(cloud=True)
    if args.ledger is None or args.ledger.resolve()!=ARTIFACTS/'ledger.json':raise ValueError('Use the retained Stage 09 ledger')
    if not args.output.is_relative_to(ARTIFACTS):raise ValueError('Use Stage 09 artifact outputs')
    if not args.worker:
        result=supervise([sys.executable,'-u','-m',MODULE,*sys.argv[1:],'--worker'],BudgetLedger(args.ledger),
                         training=args.mode=='train',timeout_seconds=900 if args.mode=='evaluate' else None)
        print('Stage09ProcessSupervisor='+json.dumps(result),flush=True)
        raise SystemExit(result['exit_code'] if result['exit_code']>=0 else 128-result['exit_code'])
    if 'STAGE09_SUPERVISOR_PID' not in os.environ:raise ValueError('Use the public process supervisor entry')
    if args.mode=='evaluate':evaluate(args)
    elif args.mode=='train':
        report=train(args)
        if report['status']!='TRAINING_COMPLETE':raise SystemExit(2)
    else:
        from mjlab_microduck.double_balance_stage09_campaign import run_campaign
        run_campaign(args)


if __name__=='__main__':main()
