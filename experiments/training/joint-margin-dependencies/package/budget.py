"""One approved campaign: failed attempts remain spent; no renewed deadlines."""
MAX_UPDATES = 505
MAX_SECONDS = 7200
TARGETS = {'smoke': 5, 'segment-1': 255, 'segment-2': 505}
PHASE_CAPS = {'smoke': 5, 'segment-1': 250, 'segment-2': 250}
EVAL_SEEDS = {'selection-1': (2026102302, 2026102303),
              'selection-2': (2026102302, 2026102303),
              'heldout': (2026102304, 2026102305, 2026102306, 2026102307)}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def new(now):
    return dict(started_utc=now, deadline_utc=now+MAX_SECONDS,
                max_updates=MAX_UPDATES, attempted_updates=0, completed_updates=0,
                pending=None, phase_attempts={k: 0 for k in TARGETS},
                phase_completed={k: 0 for k in TARGETS}, evaluations={},
                max_evaluation_episodes=8, approval='user approved 2026-10-23')


def reserve_update(state, phase, now):
    require(state['max_updates'] == MAX_UPDATES, 'Budget ceiling changed')
    require(now < state['deadline_utc'], 'Original compute deadline reached')
    require(state['pending'] is None, 'Earlier PPO attempt remains pending; preserve')
    require(phase in TARGETS, 'Unknown phase')
    require(state['attempted_updates'] < MAX_UPDATES, 'Approved PPO attempts exhausted')
    require(state['phase_attempts'][phase] < PHASE_CAPS[phase], 'Phase attempts exhausted')
    required = 0 if phase == 'smoke' else 5 if phase == 'segment-1' else 255
    require(required <= state['completed_updates'] < TARGETS[phase], 'PPO phase out of order')
    state['attempted_updates'] += 1
    state['phase_attempts'][phase] += 1
    state['pending'] = dict(phase=phase, attempt=state['attempted_updates'], started_utc=now)


def commit_update(state, phase):
    require(state['pending'] is not None and state['pending']['phase'] == phase,
            'PPO commit has no matching reservation')
    state['completed_updates'] += 1
    state['phase_completed'][phase] += 1
    state['pending'] = None


def reserve_evaluation(state, phase, seed, checkpoint_sha256, now):
    require(now < state['deadline_utc'], 'Original compute deadline reached')
    require(state['pending'] is None, 'PPO remains incomplete')
    require(phase in EVAL_SEEDS and seed in EVAL_SEEDS[phase], 'Unapproved evaluation seed')
    key = phase+':'+str(seed)
    require(key not in state['evaluations'], 'Evaluation already attempted; never resample')
    require(len(state['evaluations']) < 8, 'Evaluation episode budget exhausted')
    state['evaluations'][key] = dict(seed=seed, phase=phase, checkpoint_sha256=checkpoint_sha256,
                                     status='RESERVED', started_utc=now)
    return key


def finish_evaluation(state, key, result_sha256):
    item = state['evaluations'][key]
    require(item['status'] == 'RESERVED', 'Evaluation commit without reservation')
    item.update(status='COMPLETE', result_sha256=result_sha256)


def resumable(state, checkpoint_completed, now):
    require(now < state['deadline_utc'], 'Original deadline expired; collect only')
    require(state['pending'] is None, 'Interrupted PPO attempt remains spent; no automatic repair')
    require(checkpoint_completed == state['completed_updates'], 'Checkpoint lags committed PPO; no replay')
    require(all(v['status'] == 'COMPLETE' for v in state['evaluations'].values()),
            'Interrupted evaluation remains spent; no resampling')
