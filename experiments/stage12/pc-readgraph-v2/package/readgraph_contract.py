"""Exact reviewed predecessor and bounded two-arm comparison contracts."""
import json
import math
from pathlib import Path
import common as old
from deploy_contract import require,sha,ATOL,RTOL

STEPS=160
WARMUP=10
TRACE_FIELDS={'actor','actions','dones','qpos','qvel','ctrl','actuator_force','external_wrench',
    'metric_values','stable_ticks','success_ever','frictionloss','damping','counters',
    'bam_0_duty_applied','bam_0_duty_current_limited','bam_0_duty_unclipped','bam_0_effective_vin',
    'bam_0_joint_velocity','bam_0_motor_torque','bam_0_position_error','bam_0_previous_torque'}
REPORT_SHA='3b7a74f36e2c9f68b8c98aa42489519ca5ff352eadc09ea02861839d63c60f14'
PACKAGE_ID='019d160a44ca51a5294ff15a09ae2ecf2529d30af7af01d41a9b9cd69a0cc8b4'
PRIOR=old.BASE/'artifacts/double-balance-stage12/native-pc-runtime-v1/report.json'
FAILED_REPORT_SHA='4cb22947ae8bf47f9d19e14c527e8584a36a6e9b40702051dbf5537ca0a831b5'
FAILED_PRIOR=old.BASE/'artifacts/double-balance-stage12/native-pc-readgraph-v1/report.json'


def reviewed_failure(path=FAILED_PRIOR):
    require(path.is_file() and not path.is_symlink() and sha(path)==FAILED_REPORT_SHA,
            'Reviewed readgraph-v1 failure report differs')
    r=old.read(path)
    require(r['status']=='PC_READGRAPH_FAILED_PRESERVE' and r['worker_exited']
            and r['original_records_unchanged'] and r['last_worker_exit_code']==2
            and r['package_identity']=='ccb719c5e35920694d5e787b79dfc04016b8d1546c51d805bfd35eee72d55701',
            'Expected preserved v1 bridge-type failure')
    baseline=r['arms']['inference-only'];failed=r['arms']['graph-reads'];row=failed['episode_result']
    require(baseline['status']=='READGRAPH_ARM_COMPLETE' and valid_partial(baseline['episode_result'])
            and baseline['runtime_bindings_restored'],'Prior baseline incomplete')
    require(failed['status']=='READGRAPH_ARM_FAILED_PRESERVE' and failed['runtime_bindings_restored']
            and failed['control_steps']==failed['physics_steps']==0
            and row.get('error')=="RuntimeError('Unexpected captured input type: qpos')"
            and all(row.get(k) is True for k in ('checkpoint_unchanged','frozen_inputs_unchanged',
                'policy_and_normalizer_unchanged','external_body_wrench_zero_at_close'))
            and not row.get('cleanup_error') and not row.get('environment_close_error'),
            'Prior candidate failed in a different scope')
    return r


def reviewed(path=PRIOR):
    require(path.is_file() and not path.is_symlink() and sha(path)==REPORT_SHA,'Reviewed runtime-v1 report differs')
    r=old.read(path)
    require(r['status']=='PC_RUNTIME_MEASURED' and r['package_identity']==PACKAGE_ID
        and r['worker_exited'] and r['original_records_unchanged'] and r['last_worker_exit_code']==0,
        'Expected completed runtime-v1 measurement')
    require(r['comparison']['passed'] and r['comparison']['scalar_comparisons']==73440
        and r['comparison']['values_outside_tolerance']==0
        and all(v==0 for v in r['comparison']['max_absolute_error_by_field'].values()),
        'Reviewed runtime-v1 parity changed')
    require(r['recommendation']['choice']=='INFERENCE_ONLY_CANDIDATE', 'Prior selection differs')
    for mode in ('bam-graph','inference-only'):
        a=r['arms'][mode]
        require(a['status']=='RUNTIME_ARM_COMPLETE' and valid_partial(a['episode_result'],'RUNTIME_SAMPLE_COMPLETE')
            and a['runtime_bindings_restored'],'Prior arm or cleanup was incomplete')
    require(r['arms']['bam-graph']['episode_result']['model_limits']==
        r['arms']['inference-only']['episode_result']['model_limits'], 'Reviewed audit snapshots differ')
    require(all(r['comparison'].get(k) is True for k in ('full_audit_equal','trial_equal','disturbance_equal')),
        'Reviewed full audit/force comparison changed')
    a=r['arms']['inference-only']['inference_only_statistics']
    require(a['omitted_reward_calls']==STEPS and a['actor_group_calls']==STEPS and a['unexpected_critic_calls']==0,
        'Prior inference-only scope was incomplete')
    return r


def valid_partial(row,stop_reason='READGRAPH_SAMPLE_COMPLETE'):
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
            require(set(x['values'])==TRACE_FIELDS,'Required actor/plant/BAM comparison fields missing or unexpected')
            require(len(x['values']['actor'])==61 and len(x['values']['actions'])==14,'Actor/action dimension changed')
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
        excluded_fields=['critic','rewards'],
        comparison_scope='One 3.2-second matched-seed trajectory with one full 10%-weight pulse; not 60-second acceptance')


def compare_audits(first,second):
    a,b=first['episode_result'],second['episode_result']
    # Only the output path differs between arm directories; pulse contents,
    # force verification and every Audit/Trial value must match exactly.
    disturbance=lambda row:{k:v for k,v in row['disturbance'].items() if k!='force_schedule_path'}
    return dict(full_audit_equal=a['model_limits']==b['model_limits'],
        trial_equal=a['trial']==b['trial'],disturbance_equal=disturbance(a)==disturbance(b))
