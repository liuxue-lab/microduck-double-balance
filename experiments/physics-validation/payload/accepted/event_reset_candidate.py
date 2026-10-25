"""Diagnostic-only EventManager reset method. Never patch the installed class.

Keep class-term reset dispatch; index interval timers using the complete term
list so function terms are reset too and mixed term ordering stays aligned.
Global timers keep their phase across per-environment episode reset.
"""
import torch


def reset_consistent(self, env_ids=None):
    for mode_cfg in self._mode_class_term_cfgs.values():
        for term_cfg in mode_cfg:
            term_cfg.func.reset(env_ids=env_ids)
    num_envs = self._env.num_envs if env_ids is None else len(env_ids)
    if 'interval' in self._mode_term_cfgs:
        for index, term_cfg in enumerate(self._mode_term_cfgs['interval']):
            if not term_cfg.is_global_time:
                assert term_cfg.interval_range_s is not None
                lower, upper = term_cfg.interval_range_s
                sampled_interval = torch.rand(num_envs, device=self.device) * (upper-lower) + lower
                self._interval_term_time_left[index][env_ids] = sampled_interval
    return {}
