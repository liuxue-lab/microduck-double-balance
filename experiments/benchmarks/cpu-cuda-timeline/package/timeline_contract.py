"""Pin the latest rejected optimization report and the accepted runtime."""
import common as old
import readgraph_contract as runtime
from deploy_contract import require,sha
from readgraph_contract import STEPS,WARMUP,TRACE_FIELDS,compare_traces,compare_audits

REPORT_SHA='c46fe69f73ad7f8c23e610af5b904a723aed61107719a220bb04e68a938c831d'
PACKAGE_ID='6e80fb13e0f421999d724a9918259dbc727ea21f831a76157a15b756c0594f2c'
PRIOR=old.BASE/'artifacts/double-balance-stage12/native-pc-auditfast-v1/report.json'


def reviewed(path=PRIOR,runtime_path=runtime.PRIOR):
    runtime.reviewed(runtime_path)
    require(path.is_file() and not path.is_symlink() and sha(path)==REPORT_SHA,
            'Reviewed auditfast-v1 report differs')
    r=old.read(path)
    require(r['status']=='PC_AUDITFAST_MEASURED' and r['worker_exited']
        and r['original_records_unchanged'] and r['last_worker_exit_code']==0
        and r['package_identity']==PACKAGE_ID,'Expected completed auditfast-v1 measurement')
    c=r['comparison']
    require(c['passed'] and c['scalar_comparisons']==73440 and c['values_outside_tolerance']==0
        and all(v==0 for v in c['max_absolute_error_by_field'].values())
        and all(c[k] for k in ('full_audit_equal','trial_equal','disturbance_equal')),
        'Reviewed state/audit/force parity changed')
    require(r['recommendation']['choice']=='KEEP_INFERENCE_ONLY', 'Prior runtime selection changed')
    for mode in ('inference-only','audit-fast'):
        a=r['arms'][mode]
        require(a['status']=='AUDITFAST_ARM_COMPLETE'
            and runtime.valid_partial(a['episode_result'],'AUDITFAST_SAMPLE_COMPLETE')
            and a['runtime_bindings_restored'],'Prior arm or cleanup incomplete')
    return r


def valid_partial(row):
    return runtime.valid_partial(row,'TIMELINE_SAMPLE_COMPLETE')
