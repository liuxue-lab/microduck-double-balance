"""Exact reviewed predecessor and bounded two-arm comparison contracts."""
import json
import math
from pathlib import Path
import common as old
from deploy_contract import require,sha,ATOL,RTOL

STEPS=160
WARMUP=10
REPORT_SHA='8e48c09c82c7d8b17dbb0823f6e3f3f005c18c3052d77afacbe0f43e1e028002'
PACKAGE_ID='70e196cd67be34dc8d4ae846ccdf5570037e1706f03ad7aef68f5dcc4858ece8'
PRIOR=old.BASE/'artifacts/double-balance-stage12/native-pc-cache-v1/report.json'


def reviewed(path=PRIOR):
    require(path.is_file() and not path.is_symlink() and sha(path)==REPORT_SHA,'Reviewed PC cache report differs')
    r=old.read(path)
    require(r['status']=='PC_CACHE_MEASURED' and r['package_identity']==PACKAGE_ID
        and r['worker_exited'] and r['original_records_unchanged'] and not r['errors']
        and r['comparison']['passed'] and r['comparison']['values_outside_tolerance']==0
        and r['recommendation']['choice']=='PHASE_CACHE_CANDIDATE','Reviewed cache comparison did not pass')
    require(r['trace_files']['baseline/state-trace.jsonl']==r['trace_files']['phase-cache/state-trace.jsonl'],
            'Reviewed cache trajectories differed')
    return r


def valid_partial(row):
    limits=row.get('model_limits',{});push=row.get('disturbance',{})
    return (row.get('status')=='EPISODE_INTERRUPTED' and row.get('stop_reason')=='GRAPH_SAMPLE_COMPLETE'
        and row.get('completed_control_steps')==STEPS and row.get('observed_physics_steps')==STEPS*10
        and all(row.get(k) is True for k in ('checkpoint_unchanged','frozen_inputs_unchanged',
            'policy_and_normalizer_unchanged','external_body_wrench_zero_at_close'))
        and not row.get('cleanup_error') and not row.get('environment_close_error')
        and limits.get('any_model_bound_violation_observed') is False
        and limits.get('completed_physics_steps')==STEPS*10
        and limits.get('motor_checks_before_integration')==STEPS*10
        and limits.get('joint_state_checks')==STEPS*10+1
        and push.get('completed_pulses')==push.get('applied_pulses')==1
        and push.get('applied_physics_steps')==50 and push.get('checked_physics_steps')==STEPS*10
        and push.get('force_seed')==2026102500 and push.get('schedule_verified') is True)


def compare_traces(first,second):
    maxima={};comparisons=0;outside=0
    for path in (first,second):
        require(path.is_file() and not path.is_symlink() and path.stat().st_size<8*1024**2,'Invalid comparison trace')
    with first.open() as a,second.open() as b:
        for seq in range(STEPS):
            left,right=a.readline(65537),b.readline(65537)
            require(left.endswith('\n') and right.endswith('\n') and max(len(left),len(right))<=65536,'Incomplete trace row')
            x,y=json.loads(left),json.loads(right)
            require(x['step']==y['step']==seq+1 and x['values'].keys()==y['values'].keys(),'State trace order/schema differs')
            for key,actual in y['values'].items():
                expected=x['values'][key]
                require(isinstance(actual,list) and len(actual)==len(expected)>0,'State trace vector differs')
                require(all(type(v) in (int,float) and math.isfinite(v) for v in actual+expected),'Nonfinite state trace')
                errors=[abs(v-w) for v,w in zip(actual,expected)]
                maxima[key]=max(maxima.get(key,0.),max(errors))
                outside+=sum(e>ATOL+RTOL*abs(w) for e,w in zip(errors,expected))
                comparisons+=len(errors)
        require(not a.read(1) and not b.read(1),'Unexpected extra comparison samples')
    return dict(passed=outside==0,compared_steps=STEPS,scalar_comparisons=comparisons,
        values_outside_tolerance=outside,max_absolute_error_by_field=maxima,atol=ATOL,rtol=RTOL,
        comparison_scope='One 3.2-second matched-seed trajectory with one full 10%-weight pulse; not 60-second acceptance')
