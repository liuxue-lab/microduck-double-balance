"""Retain the measured inference-only baseline after the read-graph short test."""
import common as old
import readgraph_contract as prior
from deploy_contract import require,sha
from readgraph_contract import STEPS,WARMUP,TRACE_FIELDS,compare_traces,compare_audits

REPORT_SHA='9a49846bd3aaf8724b8082c9cca51a59c402ca583d6cc7c450d6933ab99347c5'
PRIOR=old.BASE/'artifacts/double-balance-stage12/native-pc-readgraph-v2/report.json'
FAILED_REPORT_SHA='f813c60aaccdd991285e24970d77282847a34634a15c5be695302c410b2020dc'
FAILED_PRIOR=old.BASE/'artifacts/double-balance-stage12/native-pc-commandview-v1/report.json'


def reviewed_failure(path=FAILED_PRIOR):
    require(path.is_file() and not path.is_symlink() and sha(path)==FAILED_REPORT_SHA,
            'Reviewed commandview-v1 failure report differs')
    r=old.read(path)
    require(r['status']=='PC_COMMANDVIEW_FAILED_PRESERVE' and r['worker_exited']
        and r['original_records_unchanged'] and r['last_worker_exit_code']==2
        and r['package_identity']=='92025f2837aa998cb5a9f544c5d643b8787b2747fa02a454b2e4bd86445166df',
        'Expected preserved commandview-v1 index-type failure')
    a=r['arms']['inference-only'];b=r['arms']['command-views'];e=b['episode_result']
    require(a['status']=='COMMANDVIEW_ARM_COMPLETE' and valid_partial(a['episode_result'])
        and a['runtime_bindings_restored'],'Prior baseline incomplete')
    require(b['status']=='COMMANDVIEW_ARM_FAILED_PRESERVE' and b['runtime_bindings_restored']
        and b['control_steps']==b['physics_steps']==0
        and e.get('error')=="RuntimeError('Unexpected joint address type')"
        and all(e.get(k) is True for k in ('checkpoint_unchanged','frozen_inputs_unchanged',
            'policy_and_normalizer_unchanged','external_body_wrench_zero_at_close'))
        and not e.get('cleanup_error') and not e.get('environment_close_error'),
        'Prior candidate failed in a different scope')
    return r


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
