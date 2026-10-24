"""100 quiet control steps to attribute diagnostic time; not an acceptance run."""
import argparse
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import common as old
import evaluate
from deploy_contract import validate_report
from loop_contract import measured


def run(folder):
    folder=Path(folder)
    preflight=old.preflight()
    cp=validate_report(old.SLOT/'report.json')
    base=evaluate.load_runtime()
    stats={k:[] for k in ('control','audit_before','integrator_submit','integrator_wait','audit_after')}
    done=[0]
    started=time.monotonic()
    class Session(base.EvaluationSession):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            for attr,key in [('before_integration','audit_before'),('after_integration','audit_after')]:
                original=getattr(self.guard,attr)
                def timed(fn=original,label=key):
                    t=time.perf_counter_ns()
                    try:return fn()
                    finally:stats[label].append((time.perf_counter_ns()-t)/1e6)
                setattr(self.guard,attr,timed)
            original=self.physics_step_original
            self.uninstrumented_step=original
            def integrate():
                t=time.perf_counter_ns()
                answer=original()
                submitted=time.perf_counter_ns()
                base.torch.cuda.synchronize()
                ended=time.perf_counter_ns()
                stats['integrator_submit'].append((submitted-t)/1e6)
                stats['integrator_wait'].append((ended-submitted)/1e6)
                return answer
            self.physics_step_original=integrate
        def close(self):
            try:super().close()
            finally:
                if hasattr(self,'uninstrumented_step'):
                    self.raw.sim.step=self.uninstrumented_step
        def step(self,env,actions):
            t=time.perf_counter_ns()
            answer=super().step(env,actions)
            stats['control'].append((time.perf_counter_ns()-t)/1e6)
            done[0]+=1
            return answer
    context=SimpleNamespace(**dict(vars(base),EvaluationSession=Session))
    def boundary():
        if done[0]>=100:raise evaluate.EvaluationInterrupted('PROFILE_SAMPLE_COMPLETE')
        if (folder/'STOP.json').exists() or time.monotonic()-started>=150:
            raise evaluate.EvaluationInterrupted('PROFILE_STOP_OR_DEADLINE')
    def progress(**values):
        print('PC_PROFILE=%d/100' % done[0],flush=True)
    row=evaluate.evaluate_one(context,folder,dict(case_id='quiet-2407',seed=2026102407,force_seed=None),
        cp,preflight['package_identity'],boundary,progress)
    valid=done[0]==100 and row.get('stop_reason')=='PROFILE_SAMPLE_COMPLETE' \
        and row['observed_physics_steps']==1000 and row['frozen_inputs_unchanged'] \
        and row['policy_and_normalizer_unchanged'] and row['external_body_wrench_zero_at_close'] \
        and row['checkpoint_unchanged'] and not row.get('cleanup_error') and not row.get('environment_close_error') \
        and not row['model_limits']['any_model_bound_violation_observed']
    result=dict(status='PROFILE_COMPLETE' if valid else 'PROFILE_INCOMPLETE_PRESERVE',
        control_steps=done[0],physics_steps=row.get('observed_physics_steps'),
        measured_sections_ms={k:measured(v) for k,v in stats.items()},
        approximate_per_control_section_sum_ms={k:sum(v)/max(1,done[0]) for k,v in stats.items()},
        note='Explicit CUDA synchronization added after each integrator call for attribution; profiler overhead changes timing. Control excludes policy and network; components are nested, not additive with control.',
        task_validation_claimed=False, hardware_acceptance=False,realtime_acceptance=False,
        motor_io=False,added_PPO_updates=0,episode_result=row,preflight=preflight)
    old.write(folder/'profile-report.json',result,exclusive=True)
    print('PC_PROFILE_STATUS='+result['status'],flush=True)
    return 0 if valid else 2


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--folder',required=True);args=parser.parse_args()
    def stop(*_):raise evaluate.EvaluationInterrupted('PROFILE_SIGNAL')
    for sig in (signal.SIGTERM,signal.SIGINT):signal.signal(sig,stop)
    raise SystemExit(run(args.folder))
