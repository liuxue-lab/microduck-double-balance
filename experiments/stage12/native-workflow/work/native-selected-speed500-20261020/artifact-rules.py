"""Derive only the two pinned first-episode files from the pinned evaluation."""
import re

FIXED_FILES = dict(FILES)
REMOTE_STAGE = '/root/autodl-tmp/microduck-double-balance/artifacts/double-balance-stage12/'
CAMPAIGN = 'native-speed-refine-v1'
SEEDS = (20261021, 20261022)
EXPECTED_SUMMARY = @@EXPECTED_SUMMARY@@

def expanded_files(value):
    need(value['status']=='EVALUATION_COMPLETE' and value['segment']==2 and value['model']=='5000',
         'Selected evaluation identity differs')
    checkpoint=value['checkpoint']
    need(checkpoint['sha256']==FIXED_FILES['model']['sha256'] and checkpoint['bytes']==10036828 and
         checkpoint['path']==REMOTE_STAGE+FIXED_FILES['model']['remote'], 'Evaluation checkpoint differs')
    need(checkpoint['completed_updates']==checkpoint['attempted_updates']==500 and checkpoint['policy_updates']==5000,
         'Evaluation checkpoint counters differ')
    need(value['seeds']==list(SEEDS) and value['num_envs_per_group']==128 and value['summary']==EXPECTED_SUMMARY,
         'Selected evaluation seeds, counts or summary differ')
    need(value['paired_comparison_allowed'] is False and value['selection_set_not_final_heldout_test'] is True
         and value['task_acceptance_claimed'] is False, 'Evaluation scope differs')
    expected=dict(top_center_m=.012,top_height_error_m=.006,top_speed_m_s=.08,lower_offset_m=.06,
                  lower_height_min_m=.20,lower_height_max_m=.29,robot_tilt_deg=20.,lower_ball_speed_m_s=.15,
                  hold=1e-6,consecutive_steps=250,control_dt=.02,horizon_seconds=10.)
    need(value['thresholds']==expected,'Original evaluation thresholds differ')
    groups=value['group_evidence']
    need(len(groups)==2 and sorted(g['seed'] for g in groups)==list(SEEDS),'Evaluation group identities differ')
    result=dict(FIXED_FILES)
    for group in groups:
        seed=group['seed'];remote=CAMPAIGN+'/evaluations/segment-02/group-%d.json'%seed
        need(group['path']==REMOTE_STAGE+remote and re.fullmatch('[0-9a-f]{64}',group['sha256']) is not None,
             'Unexpected completed group path or hash')
        result['group_%d'%seed]=dict(remote=remote,local='evaluation/group-%d.json'%seed,
                                   max_bytes=2*1024**2,sha256=group['sha256'])
    return result
