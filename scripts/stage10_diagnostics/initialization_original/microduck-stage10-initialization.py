#!/usr/bin/env python3
"""Stage 10 C: three fresh initialization traces, no rollout, zero PPO."""
from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import inspect
import os
from pathlib import Path
import shlex
import shutil
import sys
import traceback
import uuid

import stage10_core as core
from stage10_core import checked, read, require, sha, utc, write
from stage10_runtime import runtime_identity, module_digest_parameters, package_review

HERE, SELF = Path(__file__).resolve().parent, Path(__file__).resolve()
SIDECAR = '6878f43233fb5c064f4500251ba0b3353e5e0d8179e08bfd3d56aefe08edc4cc'
JOBS = ['01-initialization', '02-initialization', '03-initialization']
TOOLS = ('microduck-stage10-initialization.py','stage10_core.py','stage10_capture.py',
         'stage10_trace.py','stage10_runtime.py','stage10_init_capture.py','stage10_init_compare.py',
         'reference-runtime.json','reference-b.json')
SCOPE = {'fresh_processes':3,'environments':128,'seed':8101,'explicit_final_resets_per_process':1,
         'original_forward_calls_in_final_reset':2,'rollout_physics_steps':0,
         'policy_inference_calls_after_load':0,'new_ppo_updates':0,'assistance':0,
         'old_episode_seconds':10,'video':False,'cloud_gpu_hours':0,
         'initialization_internal_calls_counted_separately':True}
COMPLETE = 'INITIALIZATION_TRACE_COMPLETE_REVIEW_REQUIRED'


def tool_hashes():
    return {n:sha(HERE/n) for n in TOOLS}


def prior_review():
    ref=read(HERE/'reference-b.json')
    folder=core.ART/'substep'/ref['batch_directory']
    core.verify_inventory(folder,ref['files'])
    return {'batch':str(folder),'verified_metadata_files':len(ref['files']),
            'first_run_gap_preserved':True,'batch_b_replayed':False,
            'reference_sha256':sha(HERE/'reference-b.json')}


def validate_folder(folder):
    require(folder.is_dir() and not folder.is_symlink() and
            folder.resolve().parent==(core.ART/'initialization').resolve(), 'Unexpected initialization output directory')


def validate_plan(folder):
    validate_folder(folder)
    plan=read(folder/'plan.json')
    require(plan['schema_version']==1 and plan['batch']=='stage10-initialization-C', 'Wrong plan')
    require(plan['source_head']==core.BASE and plan['tools']==tool_hashes(), 'Source/tool identity changed')
    require(plan['scope']==SCOPE and plan['jobs']==JOBS, 'Diagnostic scope changed')
    checked(plan['checkpoint'],core.PRIMARY)
    checked(folder/'initial-states/dev.pt',core.INITIAL)
    checked(folder/'initial-states/dev.json',SIDECAR)
    for name,digest in plan['tools'].items():
        checked(folder/'tool'/name,digest)
    return plan


def pin_installed_sources(env_cls, sim_cls, wrapper_cls):
    expected=read(HERE/'reference-runtime.json')['sources']
    for cls in (env_cls,sim_cls,wrapper_cls):
        for method in ('step','reset','forward','get_observations'):
            key=cls.__name__+'.'+method
            if key in expected:
                require(inspect.getsource(getattr(cls,method))==expected[key], 'Installed method changed: '+key)


def worker(plan_path,output):
    folder=plan_path.resolve().parent
    core.verify_source()
    plan=validate_plan(folder)
    require(output.parent.parent==folder and output.parent.name in JOBS and
            not output.exists() and output.name.startswith('attempt-'), 'Unsafe worker output')
    output.mkdir(parents=True,exist_ok=False)
    report={'status':'RUNNING','created_utc':utc(),'scope':SCOPE,'source_head':core.BASE,
            'checkpoint_sha256':core.PRIMARY,'dataset_sha256':core.INITIAL,
            'new_ppo_updates':0,'rollout_physics_steps':0,'policy_inference_calls_after_load':0,
            'environment_construction_started':False,'stage10_complete':False,
            'policy_success_assessed':False,'observer_effect':'Readback synchronizes GPU execution',
            'video_review':'NOT_APPLICABLE_NO_ROLLOUT'}
    write(output/'result.json',report)
    raw=env=writer=capture=observer=None
    try:
        import torch
        import mujoco_warp as mjwarp
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.sim import Simulation
        from mjlab.rl import RslRlVecEnvWrapper
        from mjlab.tasks.registry import load_rl_cfg,load_runner_cls
        from mjlab.utils.os import dump_yaml
        from mjlab_microduck.double_balance_stage09 import no_optimizer_steps
        from mjlab_microduck.double_balance_stage09_state import preflight,load_checked
        from mjlab_microduck.double_balance_stage09_evaluation import (
            TASK_ID,evaluation_config,resume_agent_config,check_loaded_runner,
            seal_initial_states,require_equal,require_finite,MetricObserver)
        from mjlab_microduck.double_balance_stage09_diagnostics import observer_class,kinematic_calibration
        from mjlab_microduck.double_balance_stage08_state import assert_zero_assistance
        from stage10_capture import module_digest
        from stage10_trace import TraceWriter
        from stage10_init_capture import InitCapture,no_rollout_and_count_bootstrap,no_policy_inference
        import mjlab_microduck.tasks  # noqa: F401
        require(preflight(cloud=False)==core.BASE, 'Frozen preflight HEAD differs')
        pin_installed_sources(ManagerBasedRlEnv,Simulation,RslRlVecEnvWrapper)
        report['runtime_identity']=runtime_identity(torch,output/'runtime-check.json')
        report['backend_before']=core.backend_snapshot(torch)
        report['backend_reader_before']=core.precision_reader_audit(torch)
        write(output/'result.json',report)
        with no_optimizer_steps(), no_rollout_and_count_bootstrap(
                ManagerBasedRlEnv,Simulation,mjwarp,output) as bootstrap:
            checkpoint,state,checksum=load_checked(plan['checkpoint'],evaluation=True)
            require(checksum==core.PRIMARY,'Checkpoint identity differs')
            cfg=evaluation_config('dev')
            agent=resume_agent_config(asdict(load_rl_cfg(TASK_ID)),checkpoint)
            agent.update(logger='tensorboard',upload_model=False)
            dump_yaml(output/'params/env.yaml',asdict(cfg))
            report['task_config_sha256']=sha(output/'params/env.yaml')
            require(report['task_config_sha256']==read(HERE/'reference-b.json')['task_config_sha256'],
                    'Task configuration differs from reviewed Batch B')
            # Durable boundary: resume never automatically repeats an attempted construction.
            report['environment_construction_started']=True
            write(output/'result.json',report)
            write(output/'construction-started.json',{'utc':utc(),'repeat_requires_review':True})
            raw=ManagerBasedRlEnv(cfg=cfg,device='cuda:0',render_mode=None)
            env=RslRlVecEnvWrapper(raw,clip_actions=agent['clip_actions'])
            runner=load_runner_cls(TASK_ID)(env,deepcopy(agent),None,'cuda:0')
            runner.load(plan['checkpoint'],strict=True,map_location='cuda:0')
            check_loaded_runner(runner,raw,checkpoint)
            bootstrap['phase']='after_load'
            require(raw.num_envs==128 and raw.max_episode_length==500 and not raw.cfg.auto_reset and
                    abs(raw.step_dt-.02)<1e-12 and raw.cfg.decimation==10 and abs(raw.physics_dt-.002)<1e-12,
                    'Frozen environment definition differs')
            report['model_loaded']={k:module_digest(getattr(runner.alg,k),torch) for k in ('actor','critic')}
            before_parameters={k:module_digest_parameters(getattr(runner.alg,k)) for k in ('actor','critic')}
            report['restoration']={'adam':'Loaded, compared exactly, steps forbidden',
                                   'learning_rate':runner.alg.learning_rate,
                                   'iteration_loaded':runner.current_learning_iteration,
                                   'common_step_loaded':raw.common_step_counter,
                                   'evaluation_counters':'Reset to zero once before original final reset',
                                   'assistance':0,'initial_state_seed':8101}
            require(runner.current_learning_iteration==4999 and raw.common_step_counter==120000 and
                    runner.alg.learning_rate==5.062500000000001e-05,'Unexpected checkpoint progress or LR')
            write(output/'result.json',report)
            policy=runner.get_inference_policy()
            with no_policy_inference(runner):
                writer=TraceWriter(output/'trace',max_bytes=512*1024**2)
                capture=InitCapture(raw,runner,policy,writer,torch,output)
                report['dependency_sources_sha256']=sha(output/'dependency-sources.json')
                capture.snapshot('loaded_before_eval_counter_reset')
                raw.common_step_counter=raw._sim_step_counter=0
                raw.episode_length_buf.zero_()
                capture.install()
                bootstrap['phase']='observed_final_reset'
                raw.reset(seed=8101)
                policy.reset()
                capture.snapshot('before_initial_state_seal')
                initial=seal_initial_states(raw,'dev',folder/'initial-states')
                write(output/'initial-receipt.json',initial)
                capture.snapshot('after_initial_state_seal')
                obs=env.get_observations()
                require_finite(obs,'initialization observations')
                capture.snapshot('after_wrapper_get_observations',obs)
                # Preserve the original Batch B setup up to its first snapshot.
                alive=torch.ones(raw.num_envs,dtype=torch.bool,device='cuda:0')
                observer_type=observer_class(MetricObserver,output,
                    dict(protocol='dev',checkpoint_sha256=checksum,source_head=core.BASE,new_ppo_updates=0),
                    kinematic_calibration(core.REPO))
                observer=observer_type(raw,alive)
                raw.metrics_manager.compute=observer.compute
                capture.snapshot('reset_ready_after_original_observer_setup',obs)
                capture.boundaries.verify()
                report['boundary_counts']=dict(capture.boundaries.counts)
                require(raw._sim_step_counter==0 and raw.common_step_counter==0 and
                        not bool(raw.episode_length_buf.any()), 'Rollout counters advanced')
                require(not bool(capture.numpy(raw.sim.wp_data.time).any()), 'Simulation time advanced after final reset')
                assert_zero_assistance(raw)
                require(not bool(raw.command_manager.get_command('twist').any()),'Nonzero command')
                require(not observer.rows,'Unexpected metric computation')
                for k in ('actor','critic'):
                    require(module_digest_parameters(getattr(runner.alg,k))==before_parameters[k], 'Model parameters changed')
                    require(module_digest(getattr(runner.alg,k),torch)==report['model_loaded'][k], 'Model state_dict changed')
                require_equal(runner.alg.save()['optimizer_state_dict'],checkpoint['optimizer_state_dict'],'Adam after initialization')
                require(runner.current_learning_iteration==checkpoint['iter'] and
                        runner.alg.learning_rate==agent['algorithm']['learning_rate'],'Training progress/LR changed')
                report['backend_after']=core.backend_snapshot(torch)
                report['backend_reader_after']=core.precision_reader_audit(torch)
                require(report['backend_after']==report['backend_before'],'Backend configuration changed')
                capture.close()
                writer.close()
                report.update(recorded_events=writer.events,unique_trace_arrays=len(writer.blobs),
                              raw_unique_array_bytes=writer.total,model_parameters_unchanged=True,
                              model_buffers_unchanged=True,adam_state_unchanged=True,
                              initialization_gaps=capture.gaps,sim_time_after_final_reset=0.,
                              original_metric_calls_after_load=0)
            bootstrap['phase']='complete'
        core.verify_source()
        validate_plan(folder)
        report['source_recheck_after']='PASS_101_FROZEN_FILES'
        report['status']=COMPLETE
    except BaseException as exc:
        report.update(status='FAILED_REVIEW_REQUIRED',error=type(exc).__name__+': '+str(exc))
        raise
    finally:
        if capture is not None:
            capture.close()
            report['observed_boundary_counts']=dict(capture.boundaries.counts)
        if writer is not None:
            writer.close()
            report['recorded_events']=writer.events
        report['finished_utc']=utc()
        write(output/'result.json',report)
        if observer is not None:
            raw.metrics_manager.compute=observer.original
        if env is not None:
            env.close()
        elif raw is not None:
            raw.close()


def validate_result(output):
    report=read(output/'result.json')
    require(report['status']==COMPLETE and report['scope']==SCOPE,'Initialization result incomplete')
    require(report['new_ppo_updates']==report['rollout_physics_steps']==report['policy_inference_calls_after_load']==0,
            'Initialization budget changed')
    require(report['backend_before']==report['backend_after'] and
            report['source_recheck_after']=='PASS_101_FROZEN_FILES','Missing completion checks')
    require(report['checkpoint_sha256']==core.PRIMARY and report['dataset_sha256']==core.INITIAL,'Input mismatch')
    return report


def choose_existing(parent):
    receipt_path=parent/'completed.json'
    if receipt_path.exists():
        receipt=read(receipt_path)
        require(Path(receipt['attempt']).name==receipt['attempt'],'Unsafe receipt')
        output=parent/receipt['attempt']
        core.verify_inventory(output,receipt['files'])
        validate_result(output)
        return output
    finished=[p.parent for p in parent.glob('attempt-*/result.json') if read(p).get('status')==COMPLETE]
    require(len(finished)<=1,'Multiple completed attempts; review required')
    if finished:
        validate_result(finished[0])
        return finished[0]
    started=list(parent.glob('attempt-*/construction-started.json'))
    started += [p for p in parent.glob('attempt-*/result.json') if read(p).get('environment_construction_started')]
    require(not started,'An environment construction was already attempted; package and review before any retry')
    require(len(list(parent.glob('attempt-*.log')))<2,'Preflight attempt limit reached; return evidence')
    return None


def controller(resume=None,package_only=None):
    import fcntl
    core.verify_source()
    core.ART.mkdir(parents=True,exist_ok=True)
    with (core.ART/'local-diagnostic.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Another Stage 10 local controller holds the lock')
        if package_only:
            folder=package_only.resolve()
            validate_plan(folder)
            package_review(folder)
            return
        prior=prior_review()
        checkpoint,dataset,sidecar=core.resolve_inputs()
        checked(sidecar,SIDECAR)
        require(shutil.disk_usage(core.REPO.parent).free>=8*1024**3,'Need 8 GiB free for retained evidence')
        home=core.ART/'initialization';home.mkdir(exist_ok=True)
        active=home/'active-batch.json'
        if resume:
            folder=resume.resolve()
            require(not active.exists() or read(active)['folder']==str(folder),'Another initialization batch is registered')
            plan=validate_plan(folder)
            if not active.exists():
                write(active,{'folder':str(folder),'tools':tool_hashes()})
        elif active.exists():
            entry=read(active)
            require(entry['tools']==tool_hashes(),'Active initialization batch uses another tool version; preserve and review')
            folder=Path(entry['folder'])
            plan=validate_plan(folder)
        else:
            leftovers=[p for p in home.iterdir() if p.is_dir()]
            require(not leftovers,'Existing unregistered initialization directory; preserve and review, do not create another batch')
            folder=home/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
            (folder/'initial-states').mkdir(parents=True,exist_ok=False)
            shutil.copy2(dataset,folder/'initial-states/dev.pt')
            shutil.copy2(sidecar,folder/'initial-states/dev.json')
            (folder/'tool').mkdir()
            for name in TOOLS:
                shutil.copy2(HERE/name,folder/'tool'/name)
            plan={'schema_version':1,'batch':'stage10-initialization-C','created_utc':utc(),
                  'source_head':core.BASE,'tools':tool_hashes(),'jobs':JOBS,'scope':SCOPE,
                  'checkpoint':str(checkpoint),'checkpoint_sha256':core.PRIMARY,
                  'dataset_sha256':core.INITIAL,'dataset_receipt_sha256':SIDECAR,
                  'prior_batch_review':prior,'new_ppo_updates':0,'stage10_complete':False,
                  'cloud_status':'OFF_CONFIRMED_BY_USER','cloud_gpu_hours':0,'cloud_data_must_be_preserved':True}
            write(folder/'plan.json',plan)
            write(active,{'folder':str(folder),'tools':tool_hashes()})
        require(plan['checkpoint']==str(checkpoint),'Checkpoint path changed')
        print('Stage10Output='+str(folder),flush=True)
        print('Stage10Resume='+shlex.join(['python3',str(SELF),'--resume',str(folder)]),flush=True)
        if (folder/'run.json').exists() and read(folder/'run.json')['status']=='INITIALIZATION_BATCH_COMPLETE_REVIEW_REQUIRED':
            for job in JOBS:
                require(choose_existing(folder/job) is not None,'Completed batch is missing a verified receipt')
            print('Stage10Initialization=already complete; no new construction or rollout',flush=True)
            package_review(folder)
            print('Stage10Batch=INITIALIZATION_BATCH_COMPLETE_REVIEW_REQUIRED; stage not complete',flush=True)
            return
        run={'status':'RUNNING','created_utc':utc(),'scope':SCOPE,'completed_jobs':[],
             'new_ppo_updates':0,'rollout_physics_steps':0,'stage10_complete':False,
             'cloud_contacted':False,'source_edited':False,'stage06_smoke_run':False}
        write(folder/'run.json',run)
        try:
            outputs=[]
            for job in JOBS:
                parent=folder/job;parent.mkdir(exist_ok=True)
                output=choose_existing(parent)
                reused=output is not None
                if output is None:
                    attempt='attempt-'+uuid.uuid4().hex[:8]
                    output=parent/attempt
                    core.supervised([sys.executable,'-u',str(SELF),'--worker',str(folder/'plan.json'),
                                     '--output',str(output)],parent/(attempt+'.log'),timeout_s=900)
                result=validate_result(output)
                if not (parent/'completed.json').exists():
                    write(parent/'completed.json',{'attempt':output.name,'completed_utc':utc(),'files':core.inventory(output)})
                outputs.append(output)
                run['completed_jobs'].append(job)
                write(folder/'run.json',run)
                print(f'Stage10Initialization={job}; {"reused" if reused else "captured"}; '
                      f'{result["recorded_events"]} events; zero rollout; zero PPO',flush=True)
            from stage10_init_compare import compare_all
            comparison=compare_all(outputs,folder/'comparison.json')
            core.verify_source();validate_plan(folder)
            run['status']=comparison['status']
        except BaseException as exc:
            run.update(status='STOPPED_REVIEW_REQUIRED',error=type(exc).__name__+': '+str(exc))
            raise
        finally:
            run['finished_utc']=utc()
            write(folder/'run.json',run)
            package_review(folder)
            print('Stage10Batch='+run['status']+'; stage not complete; upload all printed review ZIP parts',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    g=p.add_mutually_exclusive_group()
    g.add_argument('--resume',type=Path)
    g.add_argument('--package-only',type=Path)
    g.add_argument('--worker',type=Path,help=argparse.SUPPRESS)
    p.add_argument('--output',type=Path,help=argparse.SUPPRESS)
    args=p.parse_args()
    require((args.worker is not None)==(args.output is not None),'--output is for supervised worker only')
    python=core.REPO/'.venv/bin/python'
    require(python.is_file(),'Existing project venv missing; no packages will be installed')
    if Path(sys.prefix).resolve()!=(core.REPO/'.venv').resolve():
        os.execv(str(python),[str(python),'-u',str(SELF),*sys.argv[1:]])
    os.environ.setdefault('MUJOCO_GL','egl');os.environ.setdefault('MPLBACKEND','Agg')
    os.chdir(core.REPO)
    if args.worker:
        worker(args.worker,args.output.resolve())
    else:
        controller(args.resume,args.package_only)


if __name__=='__main__':
    try:
        main()
    except (Exception,KeyboardInterrupt) as exc:
        print('Stage10Stopped='+type(exc).__name__+': '+str(exc),flush=True)
        traceback.print_exc()
        raise SystemExit(1)
