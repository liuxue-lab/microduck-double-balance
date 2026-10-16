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
    require(not em._mode_term_cfgs.get('interval', []),
            'Stage12 nominal config must have no diagnostic interval events')
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
                interval_policy='No interval terms; reset_event handles empty roster; unused stream seeds retained',
                random_training_not_released=True)


# A single active roster is shared by setup and every later macro.
from integration.owner_roster import roots_for
