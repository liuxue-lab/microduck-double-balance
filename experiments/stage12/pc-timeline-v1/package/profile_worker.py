"""160-step accepted pipeline baseline and detailed host profiling, with full state parity."""
import argparse
import json
from pathlib import Path
import signal
import time
import traceback
from types import SimpleNamespace

import common as old
import evaluate
from deploy_contract import require,validate_report
from loop_contract import measured
from profile_contract import STEPS,WARMUP,valid_partial
from phase_cache import PhaseCache,Bindings
from bam_graph import BamGraphs,BatchedAuditReads,SHADOW_CHECK_CALLS
from inference_only import InferenceOnly
from host_profile import HostProfile,complete_profile


class HostTimes(Bindings):
    def __init__(self):super().__init__();self.samples={};self.enabled=False
    def attach(self,obj,name,label):
        original=getattr(obj,name);self.samples[label]=[]
        def wrapped(*args,**kwargs):
            if not self.enabled:return original(*args,**kwargs)
            started=time.perf_counter_ns()
            try:return original(*args,**kwargs)
            finally:self.samples[label].append((time.perf_counter_ns()-started)/1e6)
        self.set(obj,name,wrapped)


def run(folder,mode,cancel_file):
    folder=Path(folder);preflight=old.preflight();cp=validate_report(old.SLOT/'report.json')
    base=evaluate.load_runtime();done=[0];latencies=[];instances=[];started=time.monotonic()
    trace=(folder/'state-trace.jsonl').open('x')
    class Session(base.EvaluationSession):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.cache=None;self.graphs=None;self.reads=None;self.slim=None;self.detail=None
            self.timer=HostTimes();instances.append(self)
            try:
                self.cache=PhaseCache(self.raw,base.mdp,self.probe)
            except BaseException:
                self.close();raise
        def after_reset(self,episode):
            super().after_reset(episode)
            self.graphs=BamGraphs(self.guard.groups,base.torch)
            self.slim=InferenceOnly(self.raw,base.torch)
            self.reads=BatchedAuditReads(self.guard,base.torch)
            for obj,name,label in (
                (self.raw.action_manager,'process_action','process_action'),
                (self.raw.action_manager,'apply_action','apply_action'),
                (self.raw.scene,'write_data_to_sim','scene_write'),(self.raw.scene,'update','scene_update'),
                (self.raw.sim,'step','physics_with_force_and_bounds'),(self.raw.sim,'forward','forward'),
                (self.raw.sim,'sense','sense'),(self.raw.termination_manager,'compute','termination'),
                (self.raw.reward_manager,'compute','reward'),(self.raw.metrics_manager,'compute','metrics_and_probe'),
                (self.raw.metrics_manager,'compute_substep','metrics_substep'),
                (self.raw.observation_manager,'compute','observations'),(self.raw.command_manager,'compute','commands')):
                self.timer.attach(obj,name,label)
            if mode=='profile':
                self.detail=HostProfile();self.detail.attach_session(self)
        def step(self,env,actions):
            base.torch.cuda.synchronize()
            active=done[0]>=WARMUP
            self.timer.enabled=active
            try:
                if active and self.detail is not None:self.detail.start()
                begun=time.perf_counter_ns()
                answer=super().step(env,actions)
                base.torch.cuda.synchronize()
                latencies.append((time.perf_counter_ns()-begun)/1e6)
            finally:
                if self.detail is not None:self.detail.stop()
                self.timer.enabled=False
            obs,rewards,dones,extras=answer
            raw=self.raw
            tensors=dict(actor=obs['actor'],actions=actions,dones=dones,
                qpos=raw.sim.data.qpos,qvel=raw.sim.data.qvel,ctrl=raw.sim.data.ctrl,
                actuator_force=raw.sim.data.actuator_force,external_wrench=raw.sim.data.xfrc_applied,
                metric_values=raw.metrics_manager._step_values,stable_ticks=raw._s12_stable_ticks,
                success_ever=raw._s12_success_ever)
            # Include live BAM-written friction fields and every diagnostic in
            # the cross-trajectory comparison, not only actor-visible state.
            tensors.update(frictionloss=raw.sim.model.dof_frictionloss[...],
                           damping=raw.sim.model.dof_damping[...])
            for number,group in enumerate(self.guard.groups):
                for key,value in group.last_diagnostics.items():tensors['bam_%d_%s'%(number,key)]=value
                tensors['bam_%d_previous_torque'%number]=group._prev_motor_torque
            values={k:v.detach().reshape(-1).cpu().tolist() for k,v in tensors.items()}
            # Boolean diagnostics are represented as exact numeric 0/1 values.
            values={k:[int(v) if type(v) is bool else v for v in row] for k,row in values.items()}
            values['counters']=[int(raw.common_step_counter),int(raw._sim_step_counter)]
            trace.write(json.dumps(dict(step=done[0]+1,values=values),allow_nan=False)+'\n')
            done[0]+=1
            return answer
        def close(self):
            # Restore detail wrappers first: some wrap coarse timers, BAM or
            # BatchedAuditReads methods. Always finish all remaining cleanup.
            error=None
            for item in (self.detail,self.timer,self.slim,self.graphs,self.reads,self.cache):
                if item is not None:
                    try:item.close()
                    except BaseException as exc:
                        if error is None:error=exc
            try:super().close()
            finally:
                if error is not None:raise error
    context=SimpleNamespace(**dict(vars(base),EvaluationSession=Session))
    def boundary():
        if done[0]>=STEPS:raise evaluate.EvaluationInterrupted('PROFILE_SAMPLE_COMPLETE')
        if Path(cancel_file).exists() or time.monotonic()-started>=180:
            raise evaluate.EvaluationInterrupted('PROFILE_STOP_OR_DEADLINE')
    def progress(**_):
        old.write(folder/'progress.json',dict(mode=mode,completed=done[0],total=STEPS,percent=done[0]/STEPS*100))
        print('PC_PROFILE_PROGRESS=%s %d/%d %.1f%%' % (mode,done[0],STEPS,done[0]/STEPS*100),flush=True)
    try:
        row=evaluate.evaluate_one(context,folder,dict(case_id='regression-2500',seed=2026102407,force_seed=2026102500),
            cp,preflight['package_identity'],boundary,progress)
    finally:trace.close()
    good=valid_partial(row) and done[0]==STEPS and len(instances)==1
    snapshot={}
    if instances:
        item=instances[0]
        snapshot=dict(manager_host_time_ms={k:measured(v) for k,v in item.timer.samples.items()},
            manager_host_sum_per_measured_control_ms={k:sum(v)/max(1,len(latencies)-WARMUP) for k,v in item.timer.samples.items()},
            cache_statistics=None if item.cache is None else item.cache.stats,
            graph_statistics=None if item.graphs is None else item.graphs.stats(),
            batched_audit_calls=0 if item.reads is None else item.reads.calls,
            detailed_profile=None if item.detail is None else item.detail.report(folder/'host-profile.pstats'),
            inference_only_statistics=None if item.slim is None else item.slim.stats(),
            runtime_bindings_restored=not item.timer.saved and (item.cache is None or not item.cache.bindings.saved)
                and (item.graphs is None or not item.graphs.saved) and (item.reads is None or not item.reads.saved)
                and (item.slim is None or not item.slim.saved) and (item.detail is None or (item.detail.closed and not item.detail.saved)))
    good=good and snapshot.get('runtime_bindings_restored') is True
    good=good and snapshot.get('cache_statistics',{}).get('cache_hits',0)>0
    rows=snapshot.get('graph_statistics') or []
    good=good and bool(rows) and all(x and x['replays']==STEPS*10 and
        x['same_input_checked_calls']==SHADOW_CHECK_CALLS for x in rows)
    good=good and snapshot.get('batched_audit_calls')==STEPS*10
    if mode=='profile':
        good=good and complete_profile(snapshot.get('detailed_profile'),STEPS-WARMUP,len(rows))
    slim=snapshot.get('inference_only_statistics') or {}
    good=good and slim.get('omitted_reward_calls')==STEPS and slim.get('actor_group_calls')==STEPS
    good=good and slim.get('unexpected_critic_calls')==0
    result=dict(status='PROFILE_ARM_COMPLETE' if good else 'PROFILE_ARM_FAILED_PRESERVE',mode=mode,
        episode_result=row,preflight=preflight,control_steps=done[0],physics_steps=row.get('observed_physics_steps'),
        full_environment_step_ms=measured(latencies[WARMUP:]),all_environment_steps_ms=measured(latencies),
        excluded_initial_timing_steps=WARMUP,trace_path=str(folder/'state-trace.jsonl'),**snapshot,
        excluded_comparison_fields=['critic','rewards'],
        reward_is_compatibility_placeholder=True,diagnostic_instrumentation=mode=='profile',
        accepted_pipeline='InferenceOnly + PhaseCache + BAM Graph v2 + BatchedAuditReads',
        command_views_enabled=False,read_graph_enabled=False,
        timing_note='CUDA completion included in full environment step; numeric trace capture is outside timed span. Manager times are host wall intervals, may contain synchronization, and are not additive GPU kernel timings.',
        added_PPO_updates=0,motor_io=False,hardware_acceptance=False,full_loop_realtime_acceptance=False,
        complete_task_acceptance=False,elapsed_seconds=time.monotonic()-started)
    old.write(folder/'worker-report.json',result,exclusive=True);progress()
    print('PC_PROFILE_ARM_STATUS='+result['status'],flush=True)
    return 0 if good else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True)
    parser.add_argument('--mode',choices=('inference-only','profile'),required=True)
    parser.add_argument('--cancel-file',required=True);args=parser.parse_args()
    def stop(*_):raise evaluate.EvaluationInterrupted('PROFILE_SIGNAL')
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    try:raise SystemExit(run(args.folder,args.mode,args.cancel_file))
    except Exception as exc:
        print('PC_PROFILE_ARM_ERROR='+repr(exc),flush=True);traceback.print_exc();raise SystemExit(2)
