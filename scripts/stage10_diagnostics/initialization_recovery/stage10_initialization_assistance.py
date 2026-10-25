"""Strict read-only guard for the observed, never-processed action lifecycle."""
import numpy as np
from stage10_core import require


def assert_unprocessed_zero_assistance(raw, capture):
    require(raw.num_envs == 128, 'Unexpected initialization environment count')
    term = raw.action_manager.get_term('ball_hold')
    values = vars(term)
    require('_force' in values and values['_force'] is None,
            'Initialization expects the original unprocessed _force=None sentinel')
    missing = ('_torque', '_duck_force', '_duck_torque')
    require(all(key not in values for key in missing),
            'Assistance caches were materialized; preserve for review')
    require(tuple(term.cfg.levels) == (0.,), 'Assistance configuration is nonzero')
    require(tuple(values['_raw'].shape) == (128, 0), 'Hold action width/layout differs')
    checked = {}
    for name, value in (
        ('hold', raw._basketball_state.hold),
        ('all_bodies_xfrc_applied', raw.sim.wp_data.xfrc_applied),
        ('all_dofs_qfrc_applied', raw.sim.wp_data.qfrc_applied),
    ):
        a = capture.numpy(value)
        require(np.isfinite(a).all() and not a.any(), 'Nonzero/nonfinite initialization assistance: ' + name)
        checked[name] = {'shape': list(a.shape), 'finite': True, 'all_zero': True}
    return {'status': 'PASS_UNPROCESSED_INITIALIZATION_ONLY', 'arrays': checked,
            'force_cache': 'None', 'other_three_caches': 'ABSENT',
            'caches_created_or_cleared': False, 'actions_processed_or_applied': False,
            'original_rollout_assistance_assertion_modified': False,
            'scope': 'Only this no-rollout initialization diagnostic; not policy acceptance'}
