"""Explicit G11 owner roster plus reuse of per-world generator/delay policies."""
from types import MethodType, SimpleNamespace
from enum import Enum
from formal_owner_keys import supported_key
import numpy as np
from runtime_accounting import require, save
from isolation_policy import (StreamBank, canonical_ids, reset_delay, sample_lags,
                              reset_event, apply_interval)


def install_public_rng(env, act, bank, gate, torch):
    bam = act._delay_buffer
    delays = {id(bam): (bam, 'bam_lag')}
    for group in env.observation_manager._group_obs_term_delay_buffer.values():
        for buffer in group.values():
            delays[id(buffer)] = (buffer, 'obs_lag')
    cls = type(bam)
    old_sample, old_reset = cls._sample_lags, cls.reset
    def sample(self, mask):
        if id(self) in delays:
            return sample_lags(self, mask, bank, delays[id(self)][1])
        return old_sample(self, mask)
    def reset(self, batch_ids=None):
        if id(self) in delays:
            return reset_delay(self, batch_ids, bank, delays[id(self)][1], torch)
        return old_reset(self, batch_ids)
    gate.patch(cls, '_sample_lags', sample)
    gate.patch(cls, 'reset', reset)
    for buffer, channel in delays.values():
        require(buffer.batch_size == 1 and buffer.per_env and buffer.hold_prob == 0.,
                'Unreviewed delay/randomness scope')
    em = env.event_manager
    require(len(em._mode_term_cfgs['interval']) == 3 and
            [c.is_global_time for c in em._mode_term_cfgs['interval']] == [False, False, True],
            'Mixed interval roster changed')
    native_apply = em.apply
    gate.patch(em, 'reset', lambda env_ids=None: reset_event(em, env_ids, bank, torch))
    def apply(mode, env_ids=None, dt=None, global_env_step_count=None):
        if mode == 'interval':
            require(env_ids is None, 'Interval ids must come from expiry')
            return apply_interval(em, dt, bank, torch)
        return native_apply(mode, env_ids=env_ids, dt=dt,
                            global_env_step_count=global_env_step_count)
    gate.patch(em, 'apply', apply)
    require(set(env.command_manager._terms) == {'twist'}, 'Command roster changed')
    cmd = env.command_manager.get_term('twist')
    c = cmd.cfg
    from formal_config import command_contract
    command_policy = command_contract(c)
    from formal_command import private_command_method
    sample_command = private_command_method(bank, torch)
    def resample(ids):
        selected = canonical_ids(ids, 1)
        if not selected:
            return
        lo, hi = c.resampling_time_range
        for w in selected:
            cmd.time_left[w] = bank.draw('command_time', w)*(hi-lo)+lo
        sample_command(cmd, ids)
        cmd.command_counter[ids] += 1
    gate.patch(cmd, '_resample', resample)
    from formal_noise import install_noise
    noise_policy = install_noise(env, bank, gate, torch)
    return dict(delay_buffers=len(delays), streams=sorted(bank.states),
                spawn_reset_rng_draws=0,
                command_policy=command_policy,
                command_sampling='Pinned native base method with private uniform draws; turn-in-place extension is natively a no-op at p=0',
                observation_noise=noise_policy,
                interval_policy='G09 reset_event/apply_interval, local0/local1/global2 streams',
                random_training_not_released=True)


def roots_for(env, bank, torch, wp):
    from sensor_state_snapshot import OwnerSnapshot
    roots, skipped, seen = [], [], set()
    def supported(v, trail=None):
        if v is None or type(v) in (bool, int, float, str) or isinstance(v, (torch.Tensor, wp.array)):
            return True
        trail = set() if trail is None else trail
        if id(v) in trail:
            return True
        trail.add(id(v))
        if isinstance(v, dict):
            return all(supported_key(k) and supported(x, trail) for k, x in v.items())
        if isinstance(v, (list, tuple)):
            return all(supported(x, trail) for x in v)
        return type(v).__module__.startswith(('mjlab.sensor.', 'mjlab.utils.buffers.'))
    def add(label, obj, names):
        actual = []
        for n in names:
            if (id(obj), n) not in seen:
                require(hasattr(obj, n), 'Owner attribute missing: '+label+'.'+n)
                require(n in OwnerSnapshot.REFERENCES or supported(getattr(obj, n)), 'Unsupported owner state: '+label+'.'+n)
                seen.add((id(obj), n)); actual.append(n)
        if actual:
            roots.append((label, obj, tuple(actual)))
    def numeric_owner(label, obj, excluded=()):
        names = []
        for n, v in vars(obj).items():
            if callable(v) or n in excluded:
                skipped.append(dict(owner=label, field=n, policy='callable or explicit immutable/native reference', type=type(v).__name__))
                continue
            if supported(v):
                names.append(n)
            elif isinstance(v, (slice, range, torch.device, torch.dtype, Enum)):
                skipped.append(dict(owner=label, field=n, policy='immutable descriptor', value=str(v)))
            else:
                raise RuntimeError('Unclassified G11 state owner '+label+'.'+n+': '+type(v).__name__)
        add(label, obj, names)
    # Environment containers, caches and public clocks are captured separately
    # from live Model/Scene/native contexts and constant geometry/configuration.
    numeric_owner('env', env, excluded=(
        'cfg','scene','sim','manager_visualizers','_g11_surface','_g11_startup_report',
        '_g11_runtime','_basketball_state','single_observation_space','single_action_space','observation_space','action_space',
        *[n+'_manager' for n in ('action','observation','event','reward','termination',
                                'metrics','command','curriculum','recorder')]))
    if hasattr(env, '_basketball_state'):
        numeric_owner('BasketballState', env._basketball_state)
    add('RNGService', bank, ('states','seeds','draw_counts'))
    configs = ('cfg','_env','_terms','_term_cfgs','_class_term_cfgs','_mode_term_cfgs',
               '_mode_class_term_cfgs','_group_obs_term_cfgs','_group_obs_class_term_cfgs',
               '_group_obs_class_instances')
    for name in ('action','observation','event','reward','termination','metrics','command','curriculum','recorder'):
        manager = getattr(env, name+'_manager')
        numeric_owner(name+'Manager', manager, configs)
        if name == 'observation':
            require(not manager._group_obs_class_instances, 'Unreviewed stateful observation noise')
            for group, terms in manager._group_obs_term_cfgs.items():
                for key, term in zip(manager._group_obs_term_names[group], terms, strict=True):
                    if term.noise is not None:
                        numeric_owner('noise:'+group+'/'+key, term.noise)
        if name in ('action','command'):
            for key, term in manager._terms.items():
                numeric_owner(name+':'+key, term, ('cfg','_env','_entity','_duck','robot','_asset'))
        if name == 'recorder':
            for i, term in enumerate(manager._terms):
                numeric_owner('Recorder:'+str(i), term, ('cfg','_env'))
    act = env.scene['robot'].actuators[0]
    add('BAM', act, ('_prev_motor_torque','_dt','_base_kp','_friction_fields_checked',
        '_delay_buffer','vin_tensor','vin_drop_gain','kp_scale','kd_scale','default_kp_scale',
        'default_kd_scale','friction_scale','default_friction_scale','diagnostics_enabled','last_diagnostics'))
    add('firmware', act._bam_model.actuator, ('kp','vin'))
    for name, ent in env.scene.entities.items():
        add('entity:'+name, ent.data, [n for n, v in vars(ent.data).items() if isinstance(v, torch.Tensor)])
    for name, sensor in env.scene.sensors.items():
        add('sensor:'+name, sensor, tuple(vars(sensor)))
    add('reset_mask', env.sim._reset_mask, ('_tensor','_wp_array'))
    return roots, dict(roots=[dict(label=k, fields=list(ns)) for k, _, ns in roots],
                      exclusions=skipped,
                      rollback_scope='Numeric Data, selected real manager/task/entity/BAM/sensor states and nine private RNG streams; pre-trial macro boundary',
                      arbitrary_mid_RL_step_restart=False)
