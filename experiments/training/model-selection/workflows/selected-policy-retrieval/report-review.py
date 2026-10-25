"""Review the user's retained campaign report; no cloud or simulator access."""
from datetime import datetime,timedelta,timezone
import hashlib,json,math
from pathlib import Path

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;checks=[]
path=ROOT/'records/velocity-reward-refinement-20261020T121521.060332Z-12132.json'
r=json.loads(path.read_bytes())
def check(name,ok):
    assert ok,name
    checks.append(name)
check('report_size_and_hash',path.stat().st_size==62848 and hashlib.sha256(path.read_bytes()).hexdigest()=='48b261c65e6fd92241b480fcb797346bb62353f5513e8b2d185757f40e3f74b6')
identity=json.loads((HERE.parent/'velocity-reward-refinement/package/manifest.json').read_text())['identity']
check('training_package_identity_matches',r['package_identity']==r['controller_package_identity']==r['started']['package_identity']==identity)
check('completed_without_reported_error',r['status']==r['exit']['status']=='REFINEMENT_COMPLETE' and r['exit']['error'] is None)
check('all_1000_attempts_completed',r['budget']['attempted_updates']==r['budget']['completed_updates']==r['exit']['added_PPO_updates']==1000 and not r['budget']['pending'] and r['budget']['interrupted_attempts']==0)
check('recorded_processes_exited',r['processes_alive']==dict(supervisor=False,child=False))
check('original_files_reported_unchanged',r['exit']['original_training_and_evaluation_files_unchanged'] is True)
check('within_original_two_hour_deadline',r['exit']['finished_utc']<r['started']['deadline_utc'] and r['started']['deadline_utc']-r['started']['started_utc']==7200)
check('4096_environment_training_restored',r['started']['num_envs']==4096 and r['started']['rollout_steps']==24 and r['started']['learning_rate']==1e-5)
history=r['selection']['history'];best=r['selection']['best']
check('all_four_candidates_completed',len(history)==4 and [x['summary']['successes'] for x in history]==[246,253,245,250] and [x['checkpoint']['completed_updates'] for x in history]==[250,500,750,1000])
check('best_is_the_second_candidate',best==history[1] and best['summary']['successes']==max(x['summary']['successes'] for x in history))
check('selected_checkpoint_identity',best['checkpoint']['sha256']=='51821a28880aba74b80e6d5a388dfd7a82098bd3dca04a1e8947b4b2a8d5539a' and best['checkpoint']['bytes']==10036828 and best['checkpoint']['policy_updates']==5000)
check('two_following_candidates_did_not_improve',r['selection']['no_improvement']==2 and r['selection']['early_stop'] is True and r['exit']['reason']=='ATTEMPT_BUDGET_EXHAUSTED')
check('all_candidate_episodes_survived_ten_seconds',all(x['summary']['episodes']==256 and x['summary']['termination_counts']=={'time_out':256} and x['summary']['physically_terminated_count']==0 and x['summary']['mean_survival_seconds']==10. for x in history))
check('success_rates_match_exact_counts',all(x['summary']['success_fraction']==x['summary']['successes']/256 for x in history))
check('original_task_used_for_evaluation',r['started']['reward_spec']['evaluation_uses_original_task'] is True and r['started']['reward_spec']['observation_and_physics_unchanged'] is True)
check('diagnostics_cover_1000_valid_rows',r['run_diagnostics']['parsed_rows']==1000 and r['run_diagnostics']['invalid_lines']==0)
segments=r['run_diagnostics']['segments']
check('four_last_50_windows',[(s['first_update'],s['last_update']) for s in segments]==[(201,250),(451,500),(701,750),(951,1000)] and all(s['collection_seconds']['samples']==s['learning_seconds']['samples']==50 for s in segments))
check('timings_match_user_terminal',[(round(s['collection_seconds']['median'],3),round(s['learning_seconds']['median'],3)) for s in segments]==[(2.099,.325),(2.010,.324),(2.006,.320),(2.076,.329)])
check('original_baseline_preserved_as_separate_checkpoint',r['started']['origin_checkpoint_sha256']=='df03202acf010680b63d3277ec879f63e2d4808e5d1286653923a314f125ac6a' and best['checkpoint']['sha256']!=r['started']['origin_checkpoint_sha256'])
old=json.loads((ROOT/'records/completed-candidate-evaluation-20261020T091955.991117Z-10680.json').read_text())
# The old report schema is kept intact; select its aggregate by model identity.
def aggregates(value):
    if isinstance(value,dict):
        if value.get('episodes')==256 and value.get('successes')==212:yield value
        for child in value.values():yield from aggregates(child)
    elif isinstance(value,list):
        for child in value:yield from aggregates(child)
baseline=list(aggregates(old))[0]
check('baseline_212_of_256_confirmed',baseline['success_fraction']==212/256 and baseline['longest_stable_seconds']['median']==7.8)
tz=timezone(timedelta(hours=8))
summary=dict(status='REPORT_CONSISTENCY_PASS',checks=len(checks),passed=checks,report_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    started_beijing=datetime.fromtimestamp(r['started']['started_utc'],tz).isoformat(),finished_beijing=datetime.fromtimestamp(r['exit']['finished_utc'],tz).isoformat(),
    wall_minutes=(r['exit']['finished_utc']-r['started']['started_utc'])/60,
    selected_checkpoint=best['checkpoint'],success_gain_episodes=253-212,success_gain_percentage_points=100*(253-212)/256,
    longest_stable_median_gain_seconds=best['summary']['longest_stable_seconds']['median']-7.8,
    lower_speed_gate_fraction_before=baseline['gate_violation_fraction']['lower_ball_speed'],
    lower_speed_gate_fraction_after=best['summary']['gate_violation_fraction']['lower_ball_speed'],
    upper_speed_gate_fraction_before=baseline['gate_violation_fraction']['top_speed'],
    upper_speed_gate_fraction_after=best['summary']['gate_violation_fraction']['top_speed'],
    timing=[dict(segment=s['segment'],collection_median=s['collection_seconds']['median'],PPO_median=s['learning_seconds']['median'],
        logged_compute_transitions_per_second_median=s['environment_transitions_per_logged_compute_second']['median'],
        sampling_fraction_of_mean_logged_compute=s['collection_seconds']['mean']/(s['collection_seconds']['mean']+s['learning_seconds']['mean']),
        between_update_wall_median=s['wall_seconds_between_updates']['median']) for s in segments],
    scope='Checks only the uploaded report and retained source identity. Per-episode cloud files are not yet downloaded to this assistant. Selection-set comparison is not a final held-out evaluation or an exactly paired comparison.',
    new_PPO_updates=0,new_simulation_steps=0)
(HERE/'report-review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
