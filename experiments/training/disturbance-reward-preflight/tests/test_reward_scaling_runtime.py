"""Execute the actual mjlab v1.3.0 compute method on CPU; no simulation/PPO."""
import ast
import hashlib
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'package'))
import robust_task

SOURCE=ROOT/'reference/mjlab-v1.3.0-reward_manager.py'
module=ast.parse(SOURCE.read_text())
cls=next(node for node in module.body if isinstance(node,ast.ClassDef) and node.name=='RewardManager')
method=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name=='compute')
scope={'torch':torch}
exec(compile(ast.Module(body=[method],type_ignores=[]),str(SOURCE),'exec'),scope)
InstalledManager=type('InstalledManager',(),{'compute':scope['compute']})


class ScalingRegression(unittest.TestCase):
    def test_official_scale_alias_accepts_and_does_not_modify_live_buffers(self):
        manager=InstalledManager();manager._scale_by_dt=True
        sentinel=torch.tensor([123.]);manager._reward_buf=sentinel
        before=dict(vars(manager))
        report=robust_task.verify_runtime_reward_scaling(manager,.02)
        self.assertEqual(vars(manager),before)
        self.assertTrue(torch.equal(sentinel,torch.tensor([123.])))
        self.assertAlmostEqual(report['cases'][0]['rewards'][0],-2.)
        self.assertAlmostEqual(report['cases'][1]['rewards'][0],-4.)
        self.assertEqual(report['physics_steps'],0)

    def test_dt_disabled_is_rejected_even_when_source_contains_dt(self):
        manager=InstalledManager();manager._scale_by_dt=False
        with self.assertRaisesRegex(RuntimeError,'scaling'):
            robust_task.verify_runtime_reward_scaling(manager,.02)

    def test_wrong_control_frequency_is_rejected(self):
        manager=InstalledManager();manager._scale_by_dt=True
        with self.assertRaisesRegex(RuntimeError,'50Hz'):
            robust_task.verify_runtime_reward_scaling(manager,.002)

    def test_doubled_rewards_are_rejected(self):
        class WrongManager:
            _scale_by_dt=True
            def compute(self,dt):return scope['compute'](self,dt)*2.
        with self.assertRaisesRegex(RuntimeError,'scaling'):
            robust_task.verify_runtime_reward_scaling(WrongManager(),.02)


if __name__=='__main__':unittest.main()
