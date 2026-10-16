"""Stage12 owner roster; accepted state selection plus explicit null recorder schema.

No synthetic _terms attribute is installed on real managers. Null recorder's
active_terms list is captured normally; ordinary recorders retain every term.
Frozen G11 sources and snapshot/restore semantics are unchanged.
"""
from enum import Enum
from formal_owner_keys import supported_key
from runtime_accounting import require


def recorder_terms(manager):
    from mjlab.managers.recorder_manager import NullRecorderManager, RecorderManager
    if type(manager) is NullRecorderManager:
        require(manager.cfg is None and type(manager.active_terms) is list and not manager.active_terms and
                '_terms' not in vars(manager), 'Null recorder schema differs')
        return ()
    require(type(manager) is RecorderManager, 'Unreviewed recorder manager type')
    require(type(manager._terms) is list and len(manager._terms)==len(manager.active_terms),
            'Recorder term roster differs')
    return manager._terms


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
            for i, term in enumerate(recorder_terms(manager)):
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
