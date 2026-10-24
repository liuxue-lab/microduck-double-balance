"""One local inference-only validation window, separate from frozen PPO ledgers."""
MAX_SECONDS=4500
MAX_EVALUATIONS=27


def require(ok,message):
    if not ok:raise RuntimeError(message)


def new(now):
    return dict(schema='stage12-final6h-local-eval-budget-v1',started_utc=now,
        deadline_utc=now+MAX_SECONDS,max_evaluation_episodes=MAX_EVALUATIONS,
        max_updates=0,attempted_updates=0,completed_updates=0,evaluations={},
        automatic_retry=False,automatic_extension=False,training_authorized=False)


def validate(state):
    require(state['schema']=='stage12-final6h-local-eval-budget-v1','Unexpected local evaluation ledger')
    require(state['deadline_utc']==state['started_utc']+MAX_SECONDS,'Local evaluation deadline changed')
    require(state['max_evaluation_episodes']==MAX_EVALUATIONS,'Evaluation ceiling changed')
    require(state['max_updates']==state['attempted_updates']==state['completed_updates']==0,
            'This task permits no PPO updates')
    require(not state['training_authorized'] and not state['automatic_extension']
            and not state['automatic_retry'],'Evaluation authorization changed')
    require(len(state['evaluations'])<=MAX_EVALUATIONS,'Evaluation ceiling exceeded')
    return True


def reserve_evaluation(state,phase,key,checkpoint_sha256,now):
    validate(state)
    require(now<state['deadline_utc'],'Local evaluation time boundary reached')
    require(phase in ('selection','heldout'),'Unknown evaluation phase')
    require(key not in state['evaluations'],'An evaluation cannot be repeated')
    require(len(state['evaluations'])<MAX_EVALUATIONS,'Evaluation budget exhausted')
    state['evaluations'][key]=dict(phase=phase,checkpoint_sha256=checkpoint_sha256,
        status='RESERVED',started_utc=now)
    return key


def finish_evaluation(state,key,result_sha256):
    require(state['evaluations'][key]['status']=='RESERVED','Evaluation already finished')
    state['evaluations'][key].update(status='COMPLETE',result_sha256=result_sha256)
    validate(state)
