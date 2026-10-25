"""Paired CPU Audit microbenchmark; no simulation, GPU, training or profiler."""
import copy
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'package'))
from model_limits import Audit
from scalar_audit import ScalarAudit,exact,state


def main():
    assert sys.getprofile() is None
    report=json.loads((ROOT/'reference/profile-v1-measured.json').read_bytes())
    bounds=report['arms']['inference-only']['episode_result']['model_limits']['boundaries']
    joints=copy.deepcopy(bounds['joints']);motors=copy.deepcopy(bounds['motors'])
    inputs=[]
    for tick in range(128):
        q=[(j['lower_rad']+j['upper_rad'])/2+.1*(j['upper_rad']-j['lower_rad'])*math.sin(tick*.021+i)
           for i,j in enumerate(joints)]
        speeds=[.01*math.cos(tick*.021+i) for i in range(len(joints))]
        points=[]
        for i,m in enumerate(motors):
            omega=speeds[m['local_joint_id']];duty=.02*math.sin(tick*.031+i)
            tau=m['kt']/m['resistance_ohm']*(5.*duty-m['kt']*omega*1.)
            points.append(dict(voltage=5.,velocity=omega,kd_scale=1.,duty=duty,torque=tau,
                ctrl=tau,position=q[m['local_joint_id']],position_error=.01))
        inputs.append((points,q,speeds,[0.]*len(motors)))
    samples={'original':[],'scalar':[]};equal_rounds=[]
    for round_id in range(7):
        states={}
        for mode in (('original','scalar') if round_id%2==0 else ('scalar','original')):
            audit=Audit(copy.deepcopy(joints),copy.deepcopy(motors),joint_violation_handling='record_and_continue_simulation')
            audit.observe_joints(inputs[0][1],inputs[0][2],initial=True)
            adapter=ScalarAudit(audit) if mode=='scalar' else None
            try:
                # 100 host-only audit updates exclude all 40+40 shadow checks.
                for i in range(100):
                    points,q,v,f=inputs[i%128];audit.before_integration(points);audit.after_integration(q,v,f)
                started=time.perf_counter_ns()
                for i in range(100,1600):
                    points,q,v,f=inputs[i%128];audit.before_integration(points);audit.after_integration(q,v,f)
                samples[mode].append((time.perf_counter_ns()-started)/1e6/150.)
                states[mode]=copy.deepcopy(state(audit))
            finally:
                if adapter is not None:adapter.close()
        equal_rounds.append(exact(states['original'],states['scalar']))
    assert all(equal_rounds)
    medians={k:statistics.median(v) for k,v in samples.items()}
    result=dict(status='CPU_AUDIT_BENCHMARK_COMPLETE',python=platform.python_version(),machine=platform.machine(),
        source_report_sha256=hashlib.sha256((ROOT/'reference/profile-v1-measured.json').read_bytes()).hexdigest(),
        measured_ms_per_10_audit_pairs=samples,median_ms_per_10_audit_pairs=medians,
        median_reduction_fraction=1-medians['scalar']/medians['original'],
        full_Audit_state_bitwise_equal_each_round=equal_rounds,alternating_arm_order=True,
        excluded_initial_audit_pairs_per_arm=100,measured_audit_pairs_per_arm=1500,
        metadata='Actual reviewed 14-joint and motor specs; deterministic synthetic host samples',
        scope='This workspace CPU Audit only; excludes packing, GPU, plant, policy, network; not PC speed or 50Hz acceptance',
        cprofile_enabled=False,actual_physics_steps=0,actual_GPU_steps=0,PPO_updates=0)
    (ROOT/'cpu-benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
