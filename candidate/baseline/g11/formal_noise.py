"""Original four UniformNoiseCfg amplitudes, existing per-world obs_noise RNG.

Only the actual manager-owned noise instances are patched. The formula/order
is native uniform/add before clip/scale/delay/history. Default RNG is untouched.
"""
from runtime_accounting import require
from formal_config import NOISE, noise_contract


def apply_uniform(noise, data, bank, torch, width):
    require(tuple(data.shape) == (1,width) and data.dtype == torch.float32 and
            data.device.type == 'cuda' and noise.operation == 'add', 'Noise tensor scope differs')
    lo = noise._get_cached_tensor('n_min', noise.n_min, data.device)
    hi = noise._get_cached_tensor('n_max', noise.n_max, data.device)
    random = bank.draw('obs_noise', 0, (width,)).unsqueeze(0)
    return data + (random * (hi-lo) + lo)


def install_noise(env, bank, gate, torch):
    from mjlab.utils.noise import UniformNoiseCfg
    manager = env.observation_manager
    require(not manager._group_obs_class_instances, 'Stateful noise models outside reviewed scope')
    rows = []; seen = set()
    for group, terms in manager._group_obs_term_cfgs.items():
        names = manager._group_obs_term_names[group]
        for name, term in zip(names, terms, strict=True):
            if term.noise is None:
                continue
            require(group=='actor' and id(term.noise) not in seen, 'Unexpected shared/noisy critic term')
            noise = term.noise; seen.add(id(noise))
            row = noise_contract(noise, name, UniformNoiseCfg); rows.append(row)
            # These immutable parameter tensors are warmed before owner snapshots;
            # their actual cache is included in roots_for, not hidden in cfg refs.
            noise._get_cached_tensor('n_min', noise.n_min, torch.device(env.device))
            noise._get_cached_tensor('n_max', noise.n_max, torch.device(env.device))
            def apply(data, n=noise, width=row['width']):
                return apply_uniform(n, data, bank, torch, width)
            gate.patch(noise, 'apply', gate.limited('Observation_noise', 16, apply))
    require({r['term'] for r in rows} == set(NOISE), 'Effective noise roster differs')
    return dict(terms=rows, amplitudes_changed=False, group_flags_changed=False,
                default_RNG_used=False, stream='obs_noise/0', original_formula_order=True)
