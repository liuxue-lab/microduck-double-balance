"""Exact batch reservation; no new tolerance, RK tier or training authority."""
import ast
def require(ok,message):
    if not ok:raise RuntimeError(message)
def validate_plan(plan):
    require(plan['protocol']=='g10-native-blocks-endpoint-recovery-v3','Protocol differs')
    require((plan['canonical_completed_before'],plan['canonical_attempted_before'])==(667608,667609),'Previous actual ledger differs')
    require(plan['physical_budget_derivation']==dict(adaptive_macros=2,adaptive_microsteps=352,fixed_continuations=6,fixed_microsteps=128,integrations=1472,processes=4,setup_forward_each=2,adaptive_forward_each=1413,fixed_forward_each=514,GPU_forward=5918),'Budget derivation differs')
    require(plan['native_budget']==dict(integrations=1472,GPU_forward=5918,CPU_forward=8,PPO=0,cloud=0),'Native total differs')
    require(2*(352+2*128)+2*128==1472 and 4*2+2*1413+6*514==5918,'Internal physical accounting mismatch')
    require(plan['owner_limits']['integrations']==608 and plan['owner_limits']['GPU_forward']==2443,'Producer cap differs')
    require(plan['adaptive_trial_tiers']==[32,64,128] and plan['mandatory_accepted_tier_replay'],'Original adaptive protocol differs')
    require(plan['wall_clock_policy']=='unlimited_user_requested','Wall policy differs')
    require(plan['owner_limits']['PPO']==plan['owner_limits']['cloud']==plan['PPO']==plan['cloud_training']==0,'Training budget nonzero')
    require(plan['graph_limits']['launch']==plan['contact_ordering']['calls']==8480,'Inherited sorting hard ceiling changed')
    require(plan['fixed_reference_resolution']==128,'Fixed diagnostic tier changed')
    require(plan['process_order']==['world0-producer','world0-consumer','world1-producer','world1-consumer'] and plan['progress_units']==16,'Declared process schedule differs')
    return dict(status='G10_BLOCKS_RECOVERY_HOST_BUDGET_PASS',native_calls=0,integrations_total_max=1472)
def validate_runtime(plan,eager,graph):
    result=validate_plan(plan)
    require(eager.CALL_CAP==1424 and graph.CALL_CAP==8480,'Inherited Contact class differs')
    graph.validate_capacity(1,256,8480);return result
def verify_bundle_contract(files,plan):
    result=validate_plan(plan)
    import hashlib,json
    receipt=json.loads(files['source-reuse.json'])
    for name,h in receipt['unchanged_physical_helpers'].items():require(hashlib.sha256(files[name]).hexdigest()==h,'Reviewed helper changed: '+name)
    require(hashlib.sha256(files['worker.py']).hexdigest()==receipt['generated_worker_sha256'],'Worker body changed')
    return result
