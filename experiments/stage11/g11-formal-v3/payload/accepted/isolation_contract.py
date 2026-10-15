"""Exact candidate witnesses; no tolerance, aligned RNG or physical claims."""
import numpy as np
from lifecycle_support import require
from manager_contract import immediate_contract,compare_manager_rows
from transaction_contract import compare_records,exact
from isolation_policy import CHANNELS,canonical_ids


def stream_rows(before,after,worlds,global_stream=True):
    keys=[f'{c}/{w}' for c in CHANNELS for w in worlds]
    if global_stream:keys.append('event_global_2/0')
    for key in keys:
        n='RNGService.states.'+key
        require(exact(before['owners'][n],after['owners'][n]),'Stream bytes differ: '+key)
        for attr in ('draw_counts','seeds'):
            n='RNGService.'+attr+'.'+key
            require(before['scalars'][n]==after['scalars'][n],'Stream scalar differs: '+n)
    return len(keys)


def immediate(base,after,mask,logical_world=0):
    selected=list(canonical_ids(mask,1));other=[w for w in range(1) if w not in selected]
    if not selected:
        check=compare_records(base,after,True,False)
        return dict(empty_reset_true_noop=True,all_selected_state=check)
    check=immediate_contract(base,after,selected)
    a,b=base['owners'],after['owners']
    n='EventManager._interval_term_time_left.1'
    require(exact(a[n][other],b[n][other]),'Unselected class interval changed')
    require(np.all((b[n][selected]>=.35)&(b[n][selected]<.45)),'Class interval range differs')
    n='EventManager._interval_term_time_left.2'
    require(exact(a[n],b[n]),'Global timer phase reset')
    n='EventClass.resets';expected=a[n].copy();expected[selected]+=1
    require(exact(expected,b[n]) and exact(a['EventClass.fires'],b['EventClass.fires']),'Class reset dispatch differs')
    stream_rows(base,after,other)
    changed={'obs_phase','command_time','command_value','event_local_0','event_local_1'}
    for w in selected:
        for channel in CHANNELS:
            key=f'{channel}/{logical_world}';n='RNGService.draw_counts.'+key
            require(after['scalars'][n]==base['scalars'][n]+int(channel in changed),'Reset stream draw count differs '+key)
            if channel not in changed:
                n='RNGService.states.'+key
                require(exact(a[n],b[n]),'Unrequested stream changed '+key)
    check.update(event_reset_implementation='private explicit-stream mixed local/class/global candidate',
                 global_timer_phase_preserved=True,class_reset_dispatch_exact=True,
                 reset_stream_draws_per_selected_world=5,unselected_streams_exact=True)
    return check


def isolated(control,post,mask):
    selected=list(canonical_ids(mask,1));other=[w for w in range(1) if w not in selected]
    if not selected:
        check=compare_records(control,post,False,True)
        return dict(empty_control_post_exact=True,all_selected_state=check)
    result=compare_manager_rows(control,post,selected,False)
    require(not result['changed'],'Ordinary unselected manager rows differ: '+repr(result['changed']))
    for n in ('EventClass.resets','EventClass.fires'):
        require(exact(control['owners'][n][other],post['owners'][n][other]),'Unselected class callback state differs')
    n='EventManager._interval_term_time_left.2'
    require(exact(control['owners'][n],post['owners'][n]),'Shared global timer differs')
    streams=stream_rows(control,post,other)
    result.update(control_kind='ordinary control from identical baseline, no RNG alignment',
        explicit_streams_checked=streams,diagnostic_candidate_only=True,
        meaningful_unselected_worlds=other,global_logs_curriculum_aggregate_recorder_trace_excluded=True)
    return result


def expiry(before,post):
    a,b=before['owners'],post['owners']
    expired=np.flatnonzero(a['Command.time_left']<=0)
    expected=a['Command.command_counter'].copy();expected[expired]+=1
    require(exact(expected,b['Command.command_counter']),'Command expiry counter differs')
    expected=a['env.events'].copy();expected[1]+=1
    local=np.flatnonzero(a['EventManager._interval_term_time_left.0']<1e-6)
    expected[2,local]+=1
    global_fired=bool(a['EventManager._interval_term_time_left.2'][0]<1e-6)
    if global_fired:expected[3]+=1
    require(exact(expected,b['env.events']),'Interval/step callback dispatch differs')
    class_ids=np.flatnonzero(a['EventManager._interval_term_time_left.1']<1e-6)
    expected=a['EventClass.fires'].copy();expected[class_ids]+=1
    require(exact(expected,b['EventClass.fires']),'Class timer callback differs')
    for index,lo,hi in ((0,.25,.35),(1,.35,.45),(2,.5,.6)):
        n=f'EventManager._interval_term_time_left.{index}';ids=np.flatnonzero(a[n]<1e-6)
        other=np.flatnonzero(a[n]>=1e-6)
        require(exact(a[n][other],b[n][other]),'Nonexpired interval changed')
        require(np.all((b[n][ids]>=lo)&(b[n][ids]<hi)),'Expired interval resample bounds differ')
    return dict(command_expired_worlds=expired.tolist(),function_expired_worlds=local.tolist(),
                class_expired_worlds=class_ids.tolist(),global_fired=global_fired,dt=0.)
