"""Real PyTorch 2.9.1 regression across runner inference/logging modes."""
import importlib.util
import math
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]

class TorchStatisticsTests(unittest.TestCase):
    def test_original_runtime_error_reproduced_with_real_torch(self):
        import torch
        spec=importlib.util.spec_from_file_location('original_stats',ROOT/'reference/failure-v3/package/joint_statistics.py')
        old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
        q=torch.zeros((2,14));bounds=torch.tensor([[-1.,1.]]*14)
        stats=old.JointStatistics(torch,q,bounds,['q'+str(i) for i in range(13)]+['head_roll'])
        with torch.inference_mode():stats.observe(q)
        with self.assertRaisesRegex(RuntimeError,'Inplace update to inference tensor'):
            stats.take_rollout(1)

    def test_fixed_rollouts_cross_inference_and_normal_modes(self):
        import torch
        from statistics_runtime_check import check_statistics_runtime
        result=check_statistics_runtime(torch)
        self.assertEqual(result['status'],'TORCH_STATISTICS_TRANSITIONS_PASSED')
        self.assertEqual(result['PPO_updates'],0)

    def test_mixed_mode_updates_preserve_episode_counts_and_grad_mode(self):
        import torch
        from joint_statistics import JointStatistics
        q=torch.zeros((2,14));bounds=torch.tensor([[-math.radians(25),math.radians(25)]]*14)
        stats=JointStatistics(torch,q,bounds,['q'+str(i) for i in range(13)]+['head_roll'])
        stats.episode_steps[:]=29999
        with torch.inference_mode():
            q[1,13]=math.radians(27);stats.observe(q)
        stats.after_control_step(torch.tensor([True,True]),torch.tensor([False,False]))
        row=stats.take_rollout(1)
        self.assertEqual(row['ended_full_60s_episodes'],2)
        self.assertEqual(row['ended_full_60s_no_joint_violation'],1)
        q[:]=0;stats.observe(q)
        with torch.inference_mode():
            stats.take_rollout(1);stats.observe(q)
        stats.observe(q)
        self.assertEqual(stats.take_rollout(2)['any_joint_violation_fraction'],0)
        self.assertTrue(torch.is_grad_enabled());self.assertFalse(torch.is_inference_mode_enabled())

    def test_reward_costs_compute_on_real_torch_and_preserve_inputs(self):
        import torch
        from headroom_costs import costs,retention_factor
        q=torch.deg2rad(torch.tensor([[0.],[20.],[22.5],[25.],[27.]]));before=q.clone()
        limits=torch.deg2rad(torch.tensor([-25.,25.])).expand(5,1,2)
        with torch.inference_mode():
            pos,out=costs(q,torch.zeros_like(q),limits);factor=retention_factor(q,limits)
        self.assertTrue(torch.allclose(factor,torch.tensor([1.,1.,.25,0.,0.]),atol=1e-6))
        self.assertTrue(torch.equal(before,q));self.assertTrue(torch.all(out==0));self.assertTrue(torch.all(pos>=0))

if __name__=='__main__':unittest.main(verbosity=2)
