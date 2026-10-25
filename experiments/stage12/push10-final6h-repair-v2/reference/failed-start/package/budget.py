"""One new six-hour authorization, with no retries or deadline renewal."""
MAX_UPDATES=10005
MAX_SECONDS=21600
TRAIN_SECONDS=17100
MAX_CLEANUP_SECONDS=90
MAX_EVALUATIONS=27
TARGETS={'smoke':5,'formal':MAX_UPDATES}
PHASE_CAPS={'smoke':5,'formal':MAX_UPDATES-5}


def require(ok,message):
    if not ok:raise RuntimeError(message)


def validate(state):
    require(state['max_updates']==MAX_UPDATES,'Update ceiling changed')
    require(state['deadline_utc']==state['started_utc']+MAX_SECONDS,'Six-hour deadline changed')
    require(state['train_stop_utc']==state['started_utc']+TRAIN_SECONDS,'Training boundary changed')
    require(0<=state['completed_updates']<=state['attempted_updates']<=MAX_UPDATES,'Invalid charged update counts')
    require(sum(state['phase_attempts'].values())==state['attempted_updates'],'Phase attempt accounting differs')
    require(sum(state['phase_completed'].values())==state['completed_updates'],'Phase commit accounting differs')
    return True


def new(now):
    return dict(schema='stage12-push10-final6h-budget-v1',started_utc=now,
        deadline_utc=now+MAX_SECONDS,train_stop_utc=now+TRAIN_SECONDS,
        max_updates=MAX_UPDATES,max_evaluation_episodes=MAX_EVALUATIONS,
        attempted_updates=0,completed_updates=0,pending=None,
        phase_attempts={k:0 for k in TARGETS},phase_completed={k:0 for k in TARGETS},
        evaluations={},old_budgets_frozen=True,automatic_retry=False,automatic_extension=False,
        approval='User authorized final training, amended from four to six hours; six hours includes selection and verification')


def reserve_update(state,phase,now):
    validate(state)
    require(now<state['train_stop_utc'],'Training time boundary reached')
    require(state['pending'] is None,'Earlier attempt remains charged and pending; no replay')
    require(phase in TARGETS,'Unknown training phase')
    require(state['attempted_updates']<MAX_UPDATES,'Maximum charged updates reached')
    require(state['phase_attempts'][phase]<PHASE_CAPS[phase],'Phase attempt cap reached')
    require(TARGETS[phase]-PHASE_CAPS[phase]<=state['completed_updates']<TARGETS[phase],
            'Training phase out of order')
    state['attempted_updates']+=1;state['phase_attempts'][phase]+=1
    state['pending']=dict(phase=phase,attempt=state['attempted_updates'],started_utc=now)


def commit_update(state,phase):
    validate(state)
    require(state['pending'] is not None and state['pending']['phase']==phase,'Commit without matching reservation')
    state['completed_updates']+=1;state['phase_completed'][phase]+=1;state['pending']=None
    validate(state)


def reserve_evaluation(state,phase,key,checkpoint_sha256,now):
    validate(state)
    require(now<state['deadline_utc'],'Six-hour compute boundary reached')
    require(phase in ('selection','heldout'),'Unknown evaluation phase')
    require(key not in state['evaluations'],'An evaluation cannot be repeated')
    require(len(state['evaluations'])<MAX_EVALUATIONS,'Evaluation budget exhausted')
    state['evaluations'][key]=dict(phase=phase,checkpoint_sha256=checkpoint_sha256,
        status='RESERVED',started_utc=now)
    return key


def finish_evaluation(state,key,result_sha256):
    require(state['evaluations'][key]['status']=='RESERVED','Evaluation already finished')
    state['evaluations'][key].update(status='COMPLETE',result_sha256=result_sha256)
