"""Exact declared manager witnesses, with ordinary RNG effects kept visible."""
import numpy as np
from batch_scene_contract import exact,masks
from lifecycle_support import require

D='ObservationManager._group_obs_term_delay_buffer.diagnostic.action_history.'
H='ObservationManager._group_obs_term_history_buffer.diagnostic.action_history.'


def immediate_contract(base,after,mask):
    a,b=base['owners'],after['owners'];selected,other=masks(mask);checked=[]
    def same(n):require(exact(a[n],b[n]),'Native retained state differs: '+n);checked.append(n)
    def cleared(n):
        expected=a[n].copy();expected[selected]=0
        require(exact(expected,b[n]),'Native selected clear differs: '+n);checked.append(n)
    for n in ('ActionManager._action','ActionManager._prev_action','ActionManager._prev_prev_action',
        'JointPositionAction._raw_actions','env.episode_length_buf','env._manual_reset_pending',
        'env.fail','env.timeout','RewardManager._episode_sums.feet_on_ball','MetricsManager._episode_sums.sub',
        'MetricsManager._episode_sums.last','MetricsManager._substep_accum.0','MetricsManager._substep_episode_sums.0',
        'MetricsManager._step_count','Command.metrics.calls',D+'_current_lags',D+'_step_count',
        D+'_buffer._num_pushes',H+'_num_pushes'):cleared(n)
    for n in ('JointPositionAction._processed_actions','RewardManager._reward_buf','RewardManager._step_reward',
        'MetricsManager._step_values','TerminationManager._term_dones.failure','TerminationManager._term_dones.timeout',
        'TerminationManager._terminated_buf','TerminationManager._truncated_buf','env.reward_buf',
        'env.reset_buf','env.reset_terminated','env.reset_time_outs','env.obs_buf.diagnostic'):same(n)
    for n in (D+'_buffer._buffer',H+'_buffer'):
        expected=a[n].copy();expected[:,selected]=0
        require(exact(expected,b[n]),'Selected observation history reset differs: '+n);checked.append(n)
    for n in (D+'_buffer._pointer',H+'_pointer','MetricsManager._substep_count','Command.updates',
              'env.common_step_counter','env._sim_step_counter'):
        require(base['scalars'][n]==after['scalars'][n],'Retained scalar differs '+n)
    require(after['scalars']['ObservationManager._obs_buffer'] is None,'Observation cache not invalidated')
    for n,low,high in ((D+'_phase_offsets',0,3),('Command.time_left',.25,.35),
                       ('Command._command',0,1),('EventManager._interval_term_time_left.0',.25,.35)):
        require(exact(a[n][other],b[n][other]),'Unselected random state changed '+n)
        require(np.all((b[n][selected]>=low)&(b[n][selected]<high)),'Resample bounds differ '+n)
    expected=a['Command.command_counter'].copy();expected[selected]=1
    require(exact(expected,b['Command.command_counter']),'Command reset counter differs')
    expected=a['env.level'].copy();expected[selected]+=1
    require(exact(expected,b['env.level']),'Curriculum selected increment differs')
    expected=a['env.events'].copy();expected[0,selected]+=1
    require(exact(expected,b['env.events']),'Reset event selected increment differs')
    if not selected:
        for n in ('env.extras.log.Episode_Reward/feet_on_ball','env.extras.log.Episode_Metrics/sub','env.extras.log.Episode_Metrics/last'):
            require(b[n].shape==() and np.isnan(b[n]),'Native empty-mask NaN tensor log missing '+n)
        require(after['scalars']['env.extras.log.Metrics/diagnostic/calls']=={'nonfinite_float':'nan'},'Native empty-mask NaN scalar log missing')
    return dict(exact_fields=checked,native_processed_actions_retained=True,native_terminal_cache_retained=True,
        shared_history_pointers_retained=True,empty_mask_NaN_logs_retained=not selected,
        event_reset_implementation='unchanged diagnostic candidate, not installed native fix')


def compare_manager_rows(control,post,mask,require_exact):
    _,other=masks(mask);a,b=control['owners'],post['owners'];changed=[];checked=[]
    prefixes=('ActionManager.','JointPositionAction.','ObservationManager.','EventManager.','RewardManager.',
        'TerminationManager.','MetricsManager.','Command.','CommandManager.')
    env_fields=('env.obs_buf.diagnostic','env.reward_buf','env.reset_buf','env.reset_terminated','env.reset_time_outs',
        'env.episode_length_buf','env._manual_reset_pending','env.level','env.fail','env.timeout','env.features','env.events')
    for n in sorted(set(a)&set(b)):
        if not n.startswith(prefixes) and n not in env_fields:continue
        x,y=a[n],b[n]
        require(x.dtype==y.dtype and x.shape==y.shape,'Manager layout differs '+n)
        if n in (D+'_buffer._buffer',H+'_buffer','env.events'):x,y=x[:,other],y[:,other]
        elif x.ndim and x.shape[0]==1:x,y=x[other],y[other]
        else:continue
        checked.append(n)
        if not exact(x,y):changed.append(n)
    require(len(checked)>=35,'Manager unselected row coverage missing')
    if require_exact:require(not changed,'RNG-aligned unselected manager rows differ: '+repr(changed))
    return dict(unselected_worlds=other,checked=checked,changed=changed,all_exact=not changed,
        control_kind='post-reset RNG aligned counterfactual' if require_exact else 'ordinary shared RNG control',
        no_arbitrary_shared_RNG_isolation_claim=True)


def first_observation_contract(post,mask,noisy,delayed):
    selected,_=masks(mask);a=post['owners']
    for w in selected:
        require(exact(noisy[w],delayed[w]),'First selected observation delay is not current input')
        require(exact(a[D+'_buffer._buffer'][:,w],np.broadcast_to(noisy[w],(3,14))),'Selected noise history backfill differs')
        require(exact(a[H+'_buffer'][:,w],np.broadcast_to(delayed[w],(3,14))),'Selected observation history backfill differs')
        require(exact(a['env.obs_buf.diagnostic'][w,:42],np.tile(delayed[w],3)),'Selected flattened observation differs')
    for n in (D+'_buffer._num_pushes',D+'_step_count',H+'_num_pushes'):
        require(np.all(a[n][selected]==1),'Selected observation history counter differs '+n)
    require(exact(a['env.obs_buf.diagnostic'][:,42:],post['outputs']['foot_height_scan.heights'].reshape(1,2)),
        'Observation height differs from recorded sensor')
    return dict(first_selected_noise_delay_exact=True,first_selected_history_backfill_exact=True,foot_height_exact=True)
