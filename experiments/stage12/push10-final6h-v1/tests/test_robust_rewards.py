"""CPU reward algebra/config/runtime-order checks; no robot simulation or PPO."""
from copy import deepcopy
import math
from pathlib import Path
import sys
from types import SimpleNamespace as NS, ModuleType
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'package'))
import torch
import robust_costs as c
import robust_task as task


def arrays():
    q = torch.zeros(5, 13, dtype=torch.float64)
    bounds = torch.zeros(5, 13, 2, dtype=q.dtype)
    bounds[..., 0] = -1.
    bounds[..., 1] = 1.
    weights = torch.tensor(c.joint_importance(c.OTHER_JOINTS), dtype=q.dtype)
    return q, torch.zeros_like(q), bounds, weights


def fake_env(count=5):
    names = tuple(reversed(c.SERVO_JOINTS))
    gids = list(range(14, 0, -1))
    qadr = [3+2*i for i in range(15)]
    vadr = [1+3*i for i in range(15)]
    bounds = torch.zeros(1, 14, 2)
    bounds[..., 0] = -1.
    bounds[..., 1] = 1.
    robot = NS(joint_names=names, indexing=NS(joint_ids=torch.tensor(gids)),
               data=NS(joint_pos_limits=bounds, joint_pos=torch.zeros(count, 14),
                       joint_vel=torch.zeros(count, 14)))
    flags = dict(top_ball_lost=torch.zeros(count, dtype=torch.bool),
                 fell=torch.zeros(count, dtype=torch.bool),
                 time_out=torch.zeros(count, dtype=torch.bool))
    raw = NS(scene={'robot': robot}, num_envs=count,
             sim=NS(mj_model=NS(jnt_qposadr=qadr, jnt_dofadr=vadr, jnt_type=[3]*15),
                    data=NS(qpos=torch.zeros(count, 35), qvel=torch.zeros(count, 46))),
             termination_manager=NS(get_term=lambda name:flags[name]))
    return raw, flags


class AlgebraTests(unittest.TestCase):
    def test_interior_position_and_inward_recovery_cost_are_zero(self):
        q, v, bounds, weights = arrays()
        q[:, 0] = 1.-2*c.MARGIN_RAD
        v[:, 0] = -2.
        position, outward = c.costs(q, v, bounds, weights)
        self.assertTrue(torch.equal(position, torch.zeros_like(position)))
        self.assertTrue(torch.equal(outward, torch.zeros_like(outward)))

    def test_monotone_from_interior_to_material_nominal_excursion(self):
        q, v, bounds, weights = arrays()
        q[:, 0] = torch.tensor([1.-c.MARGIN_RAD, 1.-c.MARGIN_RAD/2, 1.,
                               1.+c.MARGIN_RAD/2, 1.+c.MARGIN_RAD], dtype=q.dtype)
        position, _ = c.costs(q, v, bounds, weights)
        self.assertTrue(torch.all(position[1:] > position[:-1]))
        # Priority hip: at nominal edge the actual control reward is -0.08;
        # 1.5 degrees beyond it is -0.34 per step, not a mean diluted by 13.
        self.assertAlmostEqual(float(position[2]*c.POSITION_WEIGHT*.02), -.08)
        self.assertAlmostEqual(float(position[3]*c.POSITION_WEIGHT*.02), -.34)

    def test_lower_upper_are_symmetric_and_outward_motion_only_costs(self):
        q, v, bounds, weights = arrays()
        q[:, 0] = 1.-c.MARGIN_RAD/2
        v[:, 0] = 1.
        right = c.costs(q, v, bounds, weights)
        left = c.costs(-q, -v, bounds, weights)
        self.assertTrue(all(torch.allclose(a, b) for a, b in zip(right, left)))
        self.assertTrue((right[1] > 0.).all())
        self.assertTrue((c.costs(q, -v, bounds, weights)[1] == 0.).all())

    def test_three_priority_hips_keep_two_other_joints_quarter_and_sum_not_mean(self):
        q, v, bounds, weights = arrays()
        self.assertEqual(sum(value == 2. for value in weights.tolist()), 3)
        self.assertEqual(sum(value == .25 for value in weights.tolist()), 10)
        self.assertNotIn('head_roll', c.OTHER_JOINTS)
        q[:, :] = 1.-c.MARGIN_RAD/2
        position, _ = c.costs(q, v, bounds, weights)
        self.assertTrue(torch.allclose(position, torch.ones_like(position)*.25*8.5))
        # The 13-joint sum intentionally preserves each joint's local price.
        self.assertAlmostEqual(float(position[0]*c.POSITION_WEIGHT*.02), -.085)

    def test_old_passed_neck_margin_has_small_position_tax(self):
        q, v, bounds, weights = arrays()
        neck = c.OTHER_JOINTS.index('neck_pitch')
        q[:, neck] = 1.-math.radians(2.0319949384571334)
        position, _ = c.costs(q, v, bounds, weights)
        actual = position*c.POSITION_WEIGHT*.02
        self.assertTrue(torch.all(actual > -.00105))
        self.assertTrue(torch.all(actual < 0.))

    def test_no_silent_shape_or_invalid_margin_fallback(self):
        q, v, bounds, weights = arrays()
        for function in (
            lambda:c.costs(q, v[:, :-1], bounds, weights),
            lambda:c.costs(q, v, bounds, weights[:-1]),
            lambda:c.proximity(q, bounds[:, :-1]),
            lambda:c.proximity(q, bounds, 0.),
            lambda:c.costs(q, v, bounds, weights, lookahead_seconds=float('nan')),
            lambda:c.joint_importance(c.SERVO_JOINTS),
        ):
            with self.assertRaises(ValueError):function()


class RuntimeRewardTests(unittest.TestCase):
    def test_named_direct_qpos_mapping_and_fresh_reads(self):
        raw, _ = fake_env()
        robot = raw.scene['robot']
        local = robot.joint_names.index('right_hip_roll')
        gid = int(robot.indexing.joint_ids[local])
        qid, vid = raw.sim.mj_model.jnt_qposadr[gid], raw.sim.mj_model.jnt_dofadr[gid]
        raw.sim.data.qpos[:, qid] = 1.-c.MARGIN_RAD/2
        raw.sim.data.qvel[:, vid] = 1.
        first = task.reward_costs(raw)
        self.assertTrue(torch.allclose(first['other_joint_margin'], torch.full((5,), .5), atol=1e-5))
        self.assertTrue((first['other_joint_outward'] > 0.).all())
        # Entity cache deliberately remains zero. The reward still sees direct state.
        self.assertFalse(bool(robot.data.joint_pos.any()))
        raw.sim.data.qpos[:, qid] = 0.
        raw.sim.data.qvel[:, vid] = 0.
        self.assertFalse(bool(task.other_joint_margin_cost(raw).any()))

    def test_missing_joint_or_overlapping_margin_rejected(self):
        raw, _ = fake_env();raw.scene['robot'].joint_names = ('wrong',)+raw.scene['robot'].joint_names[1:]
        with self.assertRaisesRegex(RuntimeError, 'fourteen'):task.reward_costs(raw)
        raw, _ = fake_env();raw.scene['robot'].data.joint_pos_limits[..., 1] = -.99
        with self.assertRaisesRegex(RuntimeError, 'overlapping'):task.reward_costs(raw)

    def test_failure_is_once_only_and_timeout_is_not_penalized(self):
        raw, flags = fake_env()
        flags['top_ball_lost'][:] = torch.tensor([False, True, False, True, False])
        flags['fell'][:] = torch.tensor([False, False, True, True, False])
        flags['time_out'][:] = torch.tensor([False, False, False, False, True])
        actual = task.physical_failure_cost(raw)*c.FAILURE_WEIGHT*.02
        self.assertTrue(torch.equal(actual, torch.tensor([0., -2., -2., -2., 0.])))

    def test_reward_signs_checked_and_nonfinite_measurement_rejected(self):
        raw, _ = fake_env()
        with patch('headroom_task.head_roll_position_cost', return_value=torch.zeros(5)), \
             patch('headroom_task.head_roll_outward_cost', return_value=torch.zeros(5)):
            values = task.validate_cost_signs(raw)
            self.assertEqual(set(values), {'head_roll_margin','head_roll_outward',
                                          'other_joint_margin','other_joint_outward','physical_failure'})
            self.assertTrue(all(torch.isfinite(value).all() and (value <= 0.).all() for value in values.values()))
            raw.sim.data.qpos.fill_(float('nan'))
            with self.assertRaisesRegex(RuntimeError, 'nonfinite'):task.validate_cost_signs(raw)

    def test_configuration_overlay_preserves_existing_objects_and_parameters(self):
        baseline = NS(rewards={'double_balance':NS(weight=4.), 'action_rate_l2':NS(weight=-.4),
                              'head_roll_margin':NS(weight=-6.), 'head_roll_outward':NS(weight=-.5)},
                      terminations={name:NS(time_out=False) for name in ('top_ball_lost','fell')},
                      observations=object(), actions=object(), events=object(), sim=object(),
                      episode_length_s=60., curriculum={})
        before = dict(baseline.rewards)
        managers = ModuleType('mjlab.managers');managers.RewardTermCfg = NS
        with patch.dict(sys.modules, {'mjlab.managers':managers}), \
             patch('headroom_task.make_cfg', return_value=baseline) as make:
            cfg = task.make_cfg(4096, 123)
        self.assertIs(cfg, baseline);make.assert_called_once_with(4096, 123)
        for name, value in before.items():self.assertIs(cfg.rewards[name], value)
        self.assertEqual({key:cfg.rewards[key].weight for key in task.COST_WEIGHTS}, task.COST_WEIGHTS)
        self.assertEqual(cfg.episode_length_s, 60.)


class OrderedRewardManager:
    def compute(self, dt):
        cfg = self.cfg
        return cfg.func(self.env)*cfg.weight*dt


class UnscaledRewardManager:
    def compute(self, dt):
        cfg = self.cfg
        return cfg.func(self.env)*cfg.weight


class OrderedEnvironment:
    def step(self):
        self.termination_manager.compute()
        result = self.reward_manager.compute(dt=self.step_dt)
        self._reset_idx()
        return result


class ReversedEnvironment:
    def step(self):
        result = self.reward_manager.compute(dt=self.step_dt)
        self.termination_manager.compute()
        return result


class EarlyResetEnvironment:
    def step(self):
        self.termination_manager.compute()
        self._reset_idx()
        return self.reward_manager.compute(dt=self.step_dt)


class RuntimeOrderTests(unittest.TestCase):
    def env(self, cls=OrderedEnvironment, manager=OrderedRewardManager):
        raw = cls();raw.step_dt = .02;raw.reward_manager = manager()
        raw.termination_manager = NS(get_term=lambda _:torch.zeros(3, dtype=torch.bool))
        return raw

    def test_expected_order_and_dt_are_recorded(self):
        report = task.verify_runtime_reward_order(self.env())
        self.assertEqual(report['failure_penalty_per_terminal_step'], -2.)
        self.assertTrue(report['termination_before_reward'])
        self.assertTrue(report['reward_before_auto_reset'])
        self.assertEqual(len(report['env_step_source_sha256']), 64)

    def test_reversed_or_unscaled_runtime_rejected_before_learning(self):
        with self.assertRaisesRegex(RuntimeError, 'before terminations'):
            task.verify_runtime_reward_order(self.env(ReversedEnvironment))
        with self.assertRaisesRegex(RuntimeError, 'automatic reset'):
            task.verify_runtime_reward_order(self.env(EarlyResetEnvironment))
        with self.assertRaisesRegex(RuntimeError, 'scaling'):
            task.verify_runtime_reward_order(self.env(manager=UnscaledRewardManager))
        raw = self.env();raw.step_dt = .01
        with self.assertRaisesRegex(RuntimeError, 'scaling'):task.verify_runtime_reward_order(raw)


if __name__ == '__main__':unittest.main(verbosity=2)
