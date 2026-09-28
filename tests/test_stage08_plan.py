"""CPU tests for accidental task changes, resume counters and budget admission."""
from copy import deepcopy
import math
from types import SimpleNamespace as NS
import unittest

from mjlab_microduck.double_balance_stage08_plan import (
    apply_training_profile, budget, budget_decision, campaign_manifest, progress,
)


def training_cfg():
    return NS(scene=NS(num_envs=4096, entities={"robot": "frozen", "ball": "frozen"}),
              sim=NS(mujoco=NS(timestep=.002)), decimation=10, episode_length_s=10.,
              curriculum={k: NS(params={"source": k}) for k in
                          ("standing_envs", "action_rate_weight", "com_range", "head_com_range", "basketball_hold")},
              actions={"ball_hold": NS(levels=(1, .5, .25, .1, .03, 0))},
              commands={"twist": NS(ranges=NS(lin_vel_x=(-.15, .15), lin_vel_y=(-.1, .1), ang_vel_z=(-.5, .5)),
                                    rel_standing_envs=.02, rel_turn_in_place_envs=.15)},
              rewards={"action_rate_l2": NS(weight=-.05), "double_balance": NS(weight=4)},
              events={k: NS(params={"ranges": (-.003, .003)}) for k in
                      ("randomize_com", "randomize_head_com", "unrelated")},
              terminations={"success_boundary": "frozen"}, metrics={"success": "frozen"},
              observations={"actor": 61, "critic": 85})


class ProfileTests(unittest.TestCase):
    def test_source_and_frozen_contracts_unchanged(self):
        source = training_cfg()
        original = deepcopy(source)
        for profile in "ABCDE":
            result = apply_training_profile(source, profile)
            self.assertEqual(source, original)
            for name in ("scene", "sim", "decimation", "episode_length_s", "actions",
                         "terminations", "metrics", "observations"):
                self.assertEqual(getattr(result, name), getattr(original, name))
            self.assertEqual(result.rewards["double_balance"], original.rewards["double_balance"])
            self.assertEqual(result.events["unrelated"], original.events["unrelated"])
        self.assertEqual(apply_training_profile(source, "A"), source)

    def test_zero_command_cannot_be_overwritten_by_turn_or_curriculum(self):
        result = apply_training_profile(training_cfg(), "B")
        twist = result.commands["twist"]
        self.assertEqual(vars(twist.ranges), dict(lin_vel_x=(0., 0.), lin_vel_y=(0., 0.), ang_vel_z=(0., 0.)))
        self.assertEqual(twist.rel_turn_in_place_envs, 0)
        self.assertEqual(twist.rel_standing_envs, 1)
        self.assertNotIn("standing_envs", result.curriculum)

    def test_fixed_values_survive_future_curriculum_steps(self):
        result = apply_training_profile(training_cfg(), "E")
        self.assertEqual(result.curriculum, {})
        self.assertEqual(result.rewards["action_rate_l2"].weight, -.4)
        for name in ("randomize_com", "randomize_head_com"):
            self.assertEqual(result.events[name].params["ranges"], (-.005, .005))

    def test_reject_eval_and_silent_batch_resize(self):
        with self.assertRaises(ValueError):
            apply_training_profile(training_cfg(), "E", evaluation=True)
        cfg = training_cfg()
        cfg.scene.num_envs = 2048
        with self.assertRaises(ValueError):
            apply_training_profile(cfg, "E")

    def test_reject_stale_or_already_overridden_cfg(self):
        cfg = apply_training_profile(training_cfg(), "B")
        with self.assertRaises(ValueError):
            apply_training_profile(cfg, "C")


class BudgetAndProgressTests(unittest.TestCase):
    def test_source_and_new_update_counters_are_distinct(self):
        p = progress(500)
        self.assertEqual((p["experiment_completed_updates"], p["lineage_completed_updates"]), (500, 1500))
        self.assertEqual((p["common_step_counter"], p["sim_step_counter"]), (36000, 360000))
        self.assertEqual((p["expected_adam_steps"], p["next_iteration"]), (30000, 1500))
        self.assertEqual(p["experiment_transitions"], 49_152_000)

    def test_budget_preserves_finalization_time(self):
        self.assertFalse(budget_decision("A800", spent_seconds=20*3600,
                                        additional_updates=1, seconds_per_update=4.75)["admit"])
        self.assertFalse(budget_decision("5090", spent_seconds=52*3600,
                                        additional_updates=1, seconds_per_update=4.75)["admit"])
        self.assertTrue(budget_decision("A800", spent_seconds=0,
                                       additional_updates=2500, seconds_per_update=4.75,
                                       overhead_seconds=1800)["admit"])

    def test_invalid_budget_values_are_not_treated_as_free_time(self):
        for value in (math.nan, math.inf, -1, 0):
            with self.assertRaises(ValueError):
                budget_decision("5090", spent_seconds=0, additional_updates=500, seconds_per_update=value)
        for value in (-1, 1.5, True):
            with self.assertRaises(ValueError):
                progress(value)
        with self.assertRaises(ValueError):
            budget("5060")

    def test_manifests_do_not_authorize_training_or_shrink_batch(self):
        for gpu, hours, reserve in (("A800", 24, 4), ("5090", 60, 8)):
            m = campaign_manifest(gpu)
            self.assertEqual(m["budget"]["maximum_cloud_hours"], hours)
            self.assertEqual(m["budget"]["finalization_reserve_hours"], reserve)
            self.assertFalse(m["formal_training_started"])
            self.assertFalse(m["capacity"]["automatically_reduce_environment_count"])


if __name__ == "__main__":
    unittest.main()
