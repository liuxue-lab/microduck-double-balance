"""Profile only the accepted pipeline, pinned to the latest measured report."""
import common as old
import readgraph_contract as runtime
from deploy_contract import require, sha
from readgraph_contract import STEPS, WARMUP, TRACE_FIELDS, compare_traces, compare_audits

REPORT_SHA = '718f8f83319b8fcd22158b51b045508b3c220f55b7505119ee22abeb2497a0cf'
PACKAGE_ID = 'e1fb1455ec38ede197349bfb983cb2c94b7de7f5879b074e93135219006cbc43'
PRIOR = old.BASE/'artifacts/double-balance-stage12/native-pc-commandview-v2/report.json'


def reviewed(path=PRIOR, runtime_path=runtime.PRIOR):
    runtime.reviewed(runtime_path)
    require(path.is_file() and not path.is_symlink() and sha(path)==REPORT_SHA,
            'Reviewed commandview-v2 measurement differs')
    r = old.read(path)
    require(r['status']=='PC_COMMANDVIEW_MEASURED' and r['worker_exited']
        and r['original_records_unchanged'] and r['last_worker_exit_code']==0
        and r['package_identity']==PACKAGE_ID, 'Expected completed commandview-v2 measurement')
    c = r['comparison']
    require(c['passed'] and c['scalar_comparisons']==73440 and c['values_outside_tolerance']==0
        and all(v==0 for v in c['max_absolute_error_by_field'].values())
        and all(c[k] for k in ('full_audit_equal','trial_equal','disturbance_equal')),
        'Reviewed state/audit/force parity changed')
    require(r['recommendation']['choice']=='KEEP_INFERENCE_ONLY', 'Prior baseline choice differs')
    for mode in ('inference-only','command-views'):
        a = r['arms'][mode]
        require(a['status']=='COMMANDVIEW_ARM_COMPLETE'
            and runtime.valid_partial(a['episode_result'], 'COMMANDVIEW_SAMPLE_COMPLETE')
            and a['runtime_bindings_restored'], 'Prior arm or cleanup incomplete')
    return r


def valid_partial(row):
    return runtime.valid_partial(row, 'PROFILE_SAMPLE_COMPLETE')
