"""Aggregate effective configuration checks before environment construction.

No configuration values are rewritten here. This is a read-only preflight,
not an inference that play=True means every random consumer was disabled.
"""
import math
from runtime_accounting import require, save

NOISE = {'base_ang_vel': (-.03, .03, 3),
         'projected_gravity': (-.01, .01, 3),
         'joint_pos': (-.001, .001, 14), 'joint_vel': (-.25, .25, 14)}


def noise_contract(noise, name, UniformNoiseCfg):
    require(name in NOISE, 'Unreviewed noisy observation: '+name)
    lo, hi, width = NOISE[name]
    require(type(noise) is UniformNoiseCfg and noise.operation == 'add' and
            (noise.n_min, noise.n_max) == (lo, hi), 'Observation noise differs: '+name)
    return dict(term=name, operation='add', n_min=lo, n_max=hi, width=width)


def command_contract(c):
    require(all(tuple(getattr(c.ranges, n)) == (0., 0.) for n in
                ('lin_vel_x', 'lin_vel_y', 'ang_vel_z')), 'Only zero velocity ranges released')
    require(c.heading_command is True and c.rel_heading_envs == 0. and
            c.rel_standing_envs == 1. and c.rel_forward_envs == .2 and
            c.rel_world_envs == 0. and c.init_velocity_prob == 0. and
            c.rel_turn_in_place_envs == 0., 'Inherited nominal command flags differ')
    require(tuple(c.ranges.heading) == (-math.pi, math.pi) and
            tuple(c.resampling_time_range) == (3., 8.), 'Command sampling ranges differ')
    return dict(standing=1., forward=.2, heading=0., world=0., init_velocity=0.,
                heading_range=list(c.ranges.heading), resampling_s=list(c.resampling_time_range),
                forward_bucket_retained=True, final_standing_command_zero=True)


def preflight(cfg, out, UniformNoiseCfg):
    checks = []
    def check(name, fn):
        try:
            detail = fn()
            if detail is False:
                raise RuntimeError('Predicate failed')
            checks.append(dict(name=name, passed=True, detail=detail))
        except Exception as exc:
            checks.append(dict(name=name, passed=False, error=repr(exc)))
    check('world_cadence', lambda: cfg.scene.num_envs == 1 and
          cfg.sim.mujoco.timestep == .002 and cfg.decimation == 10 and cfg.episode_length_s == 10.)
    check('seed_manual_reset', lambda: cfg.seed == 1111003 and cfg.auto_reset is False)
    check('unassisted_actions', lambda: list(cfg.actions) == ['joint_pos', 'ball_hold'] and
          cfg.actions['ball_hold'].levels == (0.,))
    check('null_curriculum', lambda: cfg.curriculum == {})
    check('command_roster', lambda: list(cfg.commands) == ['twist'])
    check('inherited_command', lambda: command_contract(cfg.commands['twist']))
    expected = {'stage11_release_block':'startup', 'expand_bam_friction_fields':'startup',
        'reset_stage11_head_reference':'reset', 'reset_action_history':'reset',
        'g11_step_probe':'step', 'g11_local_probe':'interval',
        'g11_class_probe':'interval', 'g11_global_probe':'interval'}
    check('event_roster', lambda: {n:t.mode for n,t in cfg.events.items()} == expected)
    check('interval_timers', lambda: [(t.interval_range_s,t.is_global_time)
          for t in cfg.events.values() if t.mode=='interval'] ==
          [((.02,.02),False), ((.02,.02),False), ((.02,.02),True)])
    check('observation_groups', lambda: list(cfg.observations) == ['actor','critic'] and
          cfg.observations['actor'].enable_corruption is True and
          cfg.observations['critic'].enable_corruption is False)
    for group_name, group in cfg.observations.items():
        check(group_name+'/concatenation_history', lambda g=group:
              g.concatenate_terms and g.history_length in (None,0,1) and
              all(t.history_length in (0,1) for t in g.terms.values()))
        effective = {n:t.noise for n,t in group.terms.items()
                     if group.enable_corruption and t.noise is not None}
        check(group_name+'/noise_roster', lambda e=effective,g=group_name:
              set(e) == (set(NOISE) if g=='actor' else set()))
        for name, noise in effective.items():
            check(group_name+'/'+name+'/noise', lambda n=noise,k=name: noise_contract(n,k,UniformNoiseCfg))
        for name, term in group.terms.items():
            if term.delay_max_lag:
                check(group_name+'/'+name+'/delay', lambda t=term:
                      t.delay_per_env is True and t.delay_hold_prob == 0. and
                      0 <= t.delay_min_lag <= t.delay_max_lag <= 1)
    p = cfg.events['reset_stage11_head_reference'].params
    check('deterministic_spawn', lambda: all(p[n] == 0. for n in
          ('xy_noise','tilt_noise_deg','joint_noise','ball_vel_noise')) and
          tuple(p['yaw_range']) == (0.,0.) and p['ball_radius']==.12 and p['root_height']==.125)
    report = dict(status='PASS' if all(c['passed'] for c in checks) else 'BLOCKED',
        checks=checks, passed=sum(c['passed'] for c in checks), total=len(checks),
        physical_integrations=0, config_mutations=0, inherited_noise_preserved=True,
        constructor_IMU_bias='Native once-only lazy draw during shape probe; retained and snapshotted after constructor',
        private_RNG_boundary='After constructor and real mixed-event reset probe, before public reset/step')
    save(out/'configuration-preflight.json', report, durable=True)
    require(report['status']=='PASS', 'Configuration preflight: '+', '.join(c['name'] for c in checks if not c['passed']))
    return report
