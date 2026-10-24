"""Reviewed profiling evidence and unchanged accepted runtime prerequisites."""
import common as old
import readgraph_contract as runtime
from deploy_contract import require, sha
from readgraph_contract import STEPS,WARMUP,TRACE_FIELDS,compare_traces,compare_audits
from host_profile import complete_profile

REPORT_SHA='e2feba442176d65e67b321925b41dd622296fdc79e80ca45fafadc8f4eaae809'
PACKAGE_ID='76c1297484d861f18079d140b377fe2e61289bc6fcfab91ce10266e7e8f4e8c3'
PRIOR=old.BASE/'artifacts/double-balance-stage12/native-pc-profile-v1/report.json'


def reviewed(path=PRIOR,runtime_path=runtime.PRIOR):
    runtime.reviewed(runtime_path)
    require(path.is_file() and not path.is_symlink() and sha(path)==REPORT_SHA,
            'Reviewed profile-v1 report differs')
    r=old.read(path)
    require(r['status']=='PC_PROFILE_MEASURED' and r['worker_exited']
        and r['original_records_unchanged'] and r['last_worker_exit_code']==0
        and r['package_identity']==PACKAGE_ID,'Expected completed profile-v1 measurement')
    c=r['comparison']
    require(c['passed'] and c['scalar_comparisons']==73440 and c['values_outside_tolerance']==0
        and all(v==0 for v in c['max_absolute_error_by_field'].values())
        and all(c[k] for k in ('full_audit_equal','trial_equal','disturbance_equal')),
        'Reviewed profile state/audit/force parity changed')
    require(r['recommendation']['choice']=='KEEP_INFERENCE_ONLY' and r['recommendation']['diagnostic_only'],
            'Prior baseline or diagnostic scope changed')
    for mode in ('inference-only','profile'):
        a=r['arms'][mode]
        require(a['status']=='PROFILE_ARM_COMPLETE'
            and runtime.valid_partial(a['episode_result'],'PROFILE_SAMPLE_COMPLETE')
            and a['runtime_bindings_restored'],'Prior arm or cleanup incomplete')
    require(complete_profile(r['arms']['profile']['detailed_profile'],STEPS-WARMUP,1),
            'Incomplete reviewed profiling windows')
    return r


def valid_partial(row):
    return runtime.valid_partial(row,'AUDITFAST_SAMPLE_COMPLETE')
