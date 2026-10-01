"""Compare initialization traces without importing or invoking the simulator."""
from itertools import combinations
from pathlib import Path

from stage10_core import read, write, require
from stage10_trace import rows, different, compare_pair
from stage10_init_capture import INPUT_FIELDS


def category(name):
    if name.startswith('physical/'):
        return 'recorded_integrator_inputs' if name.split('/', 1)[1] in INPUT_FIELDS else 'derived_dynamics'
    for prefix, group in [('model/', 'compiled_model'), ('workspace/', 'padded_constraint_workspace'),
                          ('contact/', 'contact_storage'), ('rng/', 'recorded_rng'),
                          ('bam/', 'bam'), ('call_args/', 'original_call_arguments'),
                          ('observations_returned/', 'observations')]:
        if name.startswith(prefix):
            return group
    return 'other_recorded_state'


def boundary_summary(left, right):
    aa = {r['phase']: r for r in rows(left/'trace')}
    bb = {r['phase']: r for r in rows(right/'trace')}
    require(aa.keys() == bb.keys(), 'Initialization phase sequence differs')
    result = []
    for n in (1, 2):
        item = {'original_forward_number': n, 'complete_input_equivalence_proved': False}
        for side in ('before', 'after'):
            label = f'sim.forward#{n}.{side}'
            require(label in aa, 'Missing original forward boundary: ' + label)
            a,b = aa[label]['fields'],bb[label]['fields']
            changed = [k for k in sorted(set(a)|set(b)) if different(a.get(k),b.get(k))]
            model = [k for k in sorted(set(a)|set(b)) if k.startswith('model/') and not k.endswith('/capture_status')]
            inputs = ['physical/'+k for k in INPUT_FIELDS]
            require(all(k in a and k in b for k in inputs), 'Missing forward input snapshot')
            item[side] = {'phase':label,
                          'differing_recorded_inputs':[k for k in inputs if k in changed],
                          'differing_model_fields':[k for k in model if k in changed],
                          'differing_derived_fields':[k for k in changed if category(k)=='derived_dynamics'],
                          'capture_gaps': sorted({k for k in set(a)|set(b) if k.endswith('/capture_status')})}
        result.append(item)
    return result


def compare_all(folders, output):
    folders = [Path(p) for p in folders]
    require(len(folders)==3, 'Exactly three initialization traces are required')
    reports = [read(p/'result.json') for p in folders]
    from stage10_recovery_common import validate_approved_first
    for index,report in enumerate(reports):
        if index==0:
            validate_approved_first(folders[index])
        else:
            require(report['status']=='INITIALIZATION_TRACE_COMPLETE_REVIEW_REQUIRED', 'Incomplete initialization trace')
            require(report['backend_before']==report['backend_after'], 'Precision configuration changed')
            require(report['source_recheck_after']=='PASS_101_FROZEN_FILES', 'Missing final source check')

        require(report['new_ppo_updates']==0 and report['rollout_physics_steps']==0 and
                report['policy_inference_calls_after_load']==0, 'Initialization-only budget differs')
        for key in ('runtime_identity','model_loaded','source_head','checkpoint_sha256','dataset_sha256',
                    'task_config_sha256','dependency_sources_sha256','boundary_counts'):
            require(report[key]==reports[0][key], 'Runtime or source identity differs: '+key)
    pairs=[]
    for left,right in combinations(folders,2):
        pair=compare_pair(left/'trace',right/'trace')
        pair['first_observed_by_category']={}
        for field,detail in pair['first_difference_by_field'].items():
            group=category(field)
            old=pair['first_observed_by_category'].get(group)
            if old is None or detail['index']<old['index']:
                pair['first_observed_by_category'][group]={k:detail[k] for k in ('index','phase','control','substep')}
                pair['first_observed_by_category'][group]['fields']=[field]
            elif detail['index']==old['index']:
                old['fields'].append(field)
        pair['original_forward_boundaries']=boundary_summary(left,right)
        pairs.append(pair)
    result={'status':'INITIALIZATION_TRACES_COMPLETE_WITH_POSTCHECK_GAP_REVIEW_REQUIRED','pairs':pairs,
            'original_first_result':'FAILED_REVIEW_REQUIRED','first_trace_reused':True,'first_run_postchecks_incomplete':True,
            'new_ppo_updates':0,'rollout_physics_steps':0,'stage10_complete':False,
            'policy_success_assessed':False,'physics_reward_acceptance_changed':False,
            'initialization_internal_calls':'See per-run bootstrap-calls.json; graph capture is not a rollout',
            'limits':['First-run final model/Adam/backend/command/source checks were not completed and cannot be reconstructed.',
                      'No proof of full simulator/internal-state equality.',
                      'Padded EFC storage can contain inactive/stale entries; first workspace difference is not a physical root cause.',
                      'GPU readback changes synchronization; this is an instrumented condition.',
                      'An observed boundary is not a CUDA kernel or a causal intervention.',
                      'No additional forward, reset or rollout was executed to recheck a result.']}
    write(output,result)
    return result
