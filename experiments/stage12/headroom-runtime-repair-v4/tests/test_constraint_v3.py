"""Targeted CPU algebra/interface checks, not Torch/GPU or physical validation."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import contextlib
import importlib.util
import json
import math
import shutil
import sys
import tempfile
import types
import unittest
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'package'))
import common
import headroom_costs as costs
import headroom_task as task
import joint_statistics
import dependency_check


class Tensor(np.ndarray):
    def __new__(cls,value,dtype=None):return np.asarray(value,dtype=dtype).view(cls)
    @property
    def device(self):return 'cpu'
    def detach(self):return self
    def cpu(self):return self
    def zero_(self):self[...] = 0;return self


class Backend:
    long=np.int64
    zeros=staticmethod(lambda shape,dtype=float,device=None:Tensor(np.zeros(shape,dtype=dtype)))
    zeros_like=staticmethod(lambda x:Tensor(np.zeros_like(x)))
    ones_like=staticmethod(lambda x:Tensor(np.ones_like(x)))
    abs=staticmethod(lambda x:Tensor(np.abs(x)))
    maximum=staticmethod(lambda a,b:Tensor(np.maximum(a,b)))
    any=staticmethod(lambda x,dim=None:Tensor(np.any(x,axis=dim)))
    sum=staticmethod(lambda x,dim=None:Tensor(np.sum(x,axis=dim)))
    amax=staticmethod(lambda x,dim=None:Tensor(np.amax(x,axis=dim)))
    where=staticmethod(lambda x,a,b:Tensor(np.where(x,a,b)))
    stack=staticmethod(lambda x:Tensor(np.stack(x)))


def arrays(degrees,velocities=None):
    q=np.radians(np.array(degrees,dtype=float)).reshape(-1,1)
    vel=np.zeros_like(q) if velocities is None else np.array(velocities,dtype=float).reshape(q.shape)
    bounds=np.broadcast_to(np.radians([-25.,25.]),q.shape+(2,))
    return q,vel,bounds


class RewardTests(unittest.TestCase):
    def test_position_penalty_monotonic_and_task_reward_vanishes_at_nominal_limit(self):
        q,v,b=arrays([0,18,20,22.5,25,26,30])
        before=q.copy();position,outward=costs.costs(q,v,b);factor=costs.retention_factor(q,b)
        np.testing.assert_allclose(position[:5],[0,0,0,.25,1])
        self.assertTrue(np.all(np.diff(position)>=0))
        np.testing.assert_allclose(factor,[1,1,1,.25,0,0,0])
        self.assertTrue(np.all(costs.POSITION_WEIGHT*position<=0))
        np.testing.assert_equal(q,before);np.testing.assert_equal(outward,0)

    def test_predictive_cost_catches_fast_outward_motion_before_old_band(self):
        q,v,b=arrays([10,-10,24,-24],[5,-5,-.5,.5])
        _,outward=costs.costs(q,v,b)
        self.assertGreater(outward[0],0);self.assertEqual(outward[0],outward[1])
        np.testing.assert_equal(outward[2:],0)

    def test_invalid_shapes_and_nonfinite_propagation(self):
        q,v,b=arrays([25])
        with self.assertRaises(ValueError):costs.costs(q,v,b,margin=0)
        with self.assertRaises(ValueError):costs.costs(q,v,b,lookahead_seconds=float('inf'))
        with self.assertRaises(ValueError):costs.costs(q,v,b[:,0])
        q[0,0]=np.nan;self.assertTrue(np.isnan(costs.costs(q,v,b)[0][0]))

    def test_named_joint_mapping_preserves_actual_soft_range(self):
        q,v,b=arrays([20,25]);data=SimpleNamespace(joint_pos=q,joint_vel=v,
            joint_pos_limits=b,soft_joint_pos_limits=np.broadcast_to(np.radians([-22.5,22.5]),b.shape))
        robot=SimpleNamespace(data=data,find_joints=lambda _:([0],['head_roll']))
        env=SimpleNamespace(scene={'robot':robot})
        np.testing.assert_allclose(task.head_roll_position_cost(env),[0,1])
        wrong=SimpleNamespace(scene={'robot':SimpleNamespace(data=data,find_joints=lambda _:([0],['wrong']))})
        with self.assertRaises(RuntimeError):task.head_roll_position_cost(wrong)

    def test_configuration_overlay_keeps_base_objects_and_dependencies(self):
        cfg=SimpleNamespace(episode_length_s=10,rewards={'double_balance':SimpleNamespace(func=lambda _:1,weight=4)},
                            sentinel=object())
        managers=types.ModuleType('mjlab.managers');managers.RewardTermCfg=lambda **kw:SimpleNamespace(**kw)
        speed=types.ModuleType('speed_task');speed.make_cfg=lambda *args:cfg
        speed.speed_aligned_head_reward=lambda _:np.array([4.,4.])
        with patch.dict(sys.modules,{'mjlab':types.ModuleType('mjlab'),'mjlab.managers':managers,'speed_task':speed}):
            self.assertIs(task.make_cfg(),cfg);self.assertEqual(cfg.episode_length_s,60)
            self.assertEqual(cfg.rewards['double_balance'].weight,4)
            self.assertEqual(cfg.rewards['head_roll_margin'].weight,-6)
            q,v,b=arrays([0,26]);robot=SimpleNamespace(find_joints=lambda _:([0],['head_roll']),data=SimpleNamespace(
                joint_pos=q,joint_vel=v,joint_pos_limits=b,soft_joint_pos_limits=np.broadcast_to(np.radians([-22.5,22.5]),b.shape)))
            np.testing.assert_allclose(task.margin_coupled_retention(SimpleNamespace(scene={'robot':robot})),[4,0])
        dependency_check.check_local_closure(ROOT/'package')


class StatisticsTests(unittest.TestCase):
    def fixture(self,n=2):
        names=['joint-'+str(i) for i in range(13)]+['head_roll']
        bounds=Tensor(np.tile(np.radians([-25.,25.]),(14,1)))
        q=Tensor(np.zeros((n,14)))
        return joint_statistics.JointStatistics(Backend(),q,bounds,names),q

    def test_violation_survives_autoreset_and_rollout_summary(self):
        stats,q=self.fixture();q[1,13]=math.radians(27)
        stats.observe(q)
        stats.after_control_step(Tensor([False,True],bool),Tensor([False,True],bool))
        q[:]=0;stats.observe(q)
        result=stats.take_rollout(2)
        self.assertEqual(result['head_roll_violation_fraction'],.25)
        self.assertAlmostEqual(result['head_roll_peak_abs_degrees'],27)
        self.assertEqual(result['ended_episodes_with_joint_violation'],1)
        self.assertEqual(result['ended_full_60s_no_joint_violation'],0)
        self.assertEqual(stats.episode_steps.tolist(),[2,1])

    def test_full_60s_safe_and_physically_failed_episode_are_distinguished(self):
        stats,q=self.fixture();stats.episode_steps[:]=29999;stats.observe(q)
        stats.after_control_step(Tensor([True,True],bool),Tensor([False,True],bool))
        result=stats.take_rollout(1)
        self.assertEqual(result['ended_episodes'],2)
        self.assertEqual(result['ended_full_60s_episodes'],1)
        self.assertEqual(result['ended_full_60s_no_joint_violation'],1)

    def test_continuous_safe_run_does_not_bridge_reset(self):
        stats,q=self.fixture(1)
        stats.observe(q);stats.observe(q)
        stats.after_control_step(Tensor([True],bool),Tensor([False],bool));stats.observe(q)
        result=stats.take_rollout(3)
        self.assertEqual(result['longest_observed_safe_run_seconds'],.004)
        self.assertEqual(stats.safe_run.tolist(),[1])

    def test_non_head_joint_and_window_count_are_not_hidden(self):
        stats,q=self.fixture();q[0,0]=math.radians(-26);stats.observe(q)
        with self.assertRaises(RuntimeError):stats.take_rollout(240)
        result=stats.take_rollout(1)
        self.assertEqual(result['any_joint_violation_fraction'],.5)
        self.assertEqual(result['head_roll_violation_fraction'],0)
        self.assertEqual(result['joints'][0]['violation_samples'],1)


class ParentTests(unittest.TestCase):
    def test_real_returned_parent_is_verified_and_never_written(self):
        source=ROOT/'reference/parent-records'
        contract=json.loads((ROOT/'package/parent-evidence.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            for name in contract['files']:
                target=folder/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source/name,target)
            before={name:(folder/name).read_bytes() for name in contract['files']}
            with patch.object(common,'PARENT_SLOT',folder):
                self.assertEqual(common.verify_parent()['parent_completed_updates'],505)
                self.assertEqual(before,{name:(folder/name).read_bytes() for name in before})
                (folder/'budget.json').write_text('{}')
                with self.assertRaisesRegex(RuntimeError,'Parent campaign evidence changed'):common.verify_parent()

    def test_original_physics_observation_and_acceptance_modules_byte_identical(self):
        source=ROOT/'reference/parent-package'
        for name in ('native_task.py','speed_task.py','reward_spec.py','normalization.py','criteria.py',
                     'baseline.py','baseline_runtime.py','model_limits.py','model_limits_runtime.py'):
            self.assertEqual((ROOT/'package'/name).read_bytes(),(source/name).read_bytes(),name)


if __name__=='__main__':unittest.main(verbosity=2)
