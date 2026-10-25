"""Retain the measured inference-only baseline after the read-graph short test."""
import common as old
import readgraph_contract as prior
from deploy_contract import require,sha
from readgraph_contract import STEPS,WARMUP,TRACE_FIELDS,compare_traces,compare_audits

REPORT_SHA='9a49846bd3aaf8724b8082c9cca51a59c402ca583d6cc7c450d6933ab99347c5'
PRIOR=old.BASE/'artifacts/double-balance-stage12/native-pc-readgraph-v2/report.json'


def reviewed(path=PRIOR,runtime_path=prior.PRIOR):
    prior.reviewed(runtime_path)
    require(path.is_file() and not path.is_symlink() and sha(path)==REPORT_SHA,
            'Reviewed readgraph-v2 measurement differs')
    r=old.read(path)
    require(r['status']=='PC_READGRAPH_MEASURED' and r['worker_exited']
        and r['original_records_unchanged'] and r['last_worker_exit_code']==0
        and r['package_identity']=='ca031a1783d107eb3cab318fc77045283b07e5d542e8ee3f5da83fa0361a400c',
        'Expected completed readgraph-v2 measurement')
    c=r['comparison']
    require(c['passed'] and c['scalar_comparisons']==73440 and c['values_outside_tolerance']==0
        and all(v==0 for v in c['max_absolute_error_by_field'].values())
        and all(c[k] for k in ('full_audit_equal','trial_equal','disturbance_equal')),
        'Reviewed state/audit/force parity changed')
    require(r['recommendation']['choice']=='KEEP_INFERENCE_ONLY','Prior selected baseline differs')
    for mode in ('inference-only','graph-reads'):
        a=r['arms'][mode]
        require(a['status']=='READGRAPH_ARM_COMPLETE' and prior.valid_partial(a['episode_result'])
            and a['runtime_bindings_restored'],'Prior arm or cleanup incomplete')
    return r


def valid_partial(row):
    return prior.valid_partial(row,'COMMANDVIEW_SAMPLE_COMPLETE')
