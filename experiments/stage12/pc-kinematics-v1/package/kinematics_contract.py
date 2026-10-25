"""Exact reviewed predecessor and bounded two-arm comparison contracts."""
import json
import math
from pathlib import Path
import common as old
from deploy_contract import require,sha,ATOL,RTOL

STEPS=160
WARMUP=10
REPORT_SHA='0a7e6af238c531b3c176c436b9223f4d41c01769a71873dbd2ad052305253485'
PACKAGE_ID='8f557aac0f11957ecd83f4ac3b6b59348558e7684207141d98118e4700c7ad02'
PRIOR=old.BASE/'artifacts/double-balance-stage12/native-pc-audit-v1/report.json'


def reviewed(path=PRIOR):
    require(path.is_file() and not path.is_symlink() and sha(path)==REPORT_SHA,'Reviewed audit-v1 report differs')
    r=old.read(path)
    require(r['status']=='PC_AUDIT_MEASURED' and r['package_identity']==PACKAGE_ID
        and r['worker_exited'] and r['original_records_unchanged'] and r['last_worker_exit_code']==0,
        'Expected completed audit-v1 measurement')
    require(r['comparison']['passed'] and r['comparison']['scalar_comparisons']==87200
        and r['comparison']['values_outside_tolerance']==0
        and all(v==0 for v in r['comparison']['max_absolute_error_by_field'].values()),
        'Reviewed audit-v1 parity changed')
    require(r['recommendation']['choice']=='KEEP_BAM_GRAPH', 'Prior selection differs')
    for mode in ('bam-graph','packed-audit'):
        a=r['arms'][mode]
        require(a['status']=='AUDIT_ARM_COMPLETE' and valid_partial(a['episode_result'],'AUDIT_SAMPLE_COMPLETE')
            and a['runtime_bindings_restored'],'Prior arm or cleanup was incomplete')
    require(r['arms']['bam-graph']['episode_result']['model_limits']==
        r['arms']['packed-audit']['episode_result']['model_limits'], 'Reviewed audit snapshots differ')
    return r


def valid_partial(row,stop_reason='KINEMATICS_SAMPLE_COMPLETE'):
    limits=row.get('model_limits',{});push=row.get('disturbance',{})
    return (row.get('status')=='EPISODE_INTERRUPTED' and row.get('stop_reason')==stop_reason
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
