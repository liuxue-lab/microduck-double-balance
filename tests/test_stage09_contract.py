"""Zero-PPO tests for reward geometry, counters, immutable metrics and budget."""
from datetime import datetime,timezone
from copy import deepcopy
from dataclasses import fields,is_dataclass
from functools import partial
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace as NS
import unittest

from mjlab_microduck.double_balance_stage09_plan import progress,PROFILES,budget_decision
from mjlab_microduck.double_balance_stage09_budget import BudgetLedger,create_ledger

ROOT=Path(__file__).resolve().parents[1]


def config_differences(actual,expected,path='cfg'):
    """Compare partials by their complete definition, not copy identity.

    Ordinary functions retain identity checks. No configuration field is
    omitted; unexpected leaf types retain Python's original equality test.
    """
    if type(actual) is not type(expected):
        yield f'{path}: type {type(actual).__name__} != {type(expected).__name__}'
    elif isinstance(actual,partial):
        for attr in ('func','args','keywords','__dict__'):
            yield from config_differences(getattr(actual,attr),getattr(expected,attr),f'{path}.{attr}')
    elif is_dataclass(actual) and not isinstance(actual,type):
        for field in fields(actual):
            yield from config_differences(getattr(actual,field.name),getattr(expected,field.name),f'{path}.{field.name}')
    elif isinstance(actual,dict):
        for key in actual.keys() | expected.keys():
            if key not in actual or key not in expected:
                yield f'{path}[{key!r}]: missing from '+('actual' if key not in actual else 'expected')
            else:
                yield from config_differences(actual[key],expected[key],f'{path}[{key!r}]')
    elif isinstance(actual,(list,tuple)):
        if len(actual)!=len(expected):yield f'{path}: length {len(actual)} != {len(expected)}'
        for index,(a,b) in enumerate(zip(actual,expected)):
            yield from config_differences(a,b,f'{path}[{index}]')
    elif actual!=expected:
        yield f'{path}: {repr(actual)[:180]} != {repr(expected)[:180]}'


class ContractTests(unittest.TestCase):
    def test_partial_copies_match_but_changed_definitions_fail(self):
        original={'spec_fn':partial(pow,2,mod=7),'weight':-.4}
        left,right=deepcopy(original),deepcopy(original)
        self.assertNotEqual(left,right)  # Reproduces the old false failure.
        self.assertEqual(list(config_differences(left,right)),[])
        for replacement in (partial(abs,2,mod=7),partial(pow,3,mod=7),partial(pow,2,mod=11)):
            changed=deepcopy(right);changed['spec_fn']=replacement
            self.assertTrue(list(config_differences(left,changed)))
        changed=deepcopy(right);changed['spec_fn'].stage=9
        self.assertTrue(list(config_differences(left,changed)))
        changed=deepcopy(right);changed['weight']=-.05
        self.assertEqual(len(list(config_differences(left,changed))),1)

    def test_progress_and_adam_continue_from_primary(self):
        self.assertEqual(progress(0)['expected_adam_steps'],100000)
        self.assertEqual(progress(0)['next_iteration'],5000)
        self.assertEqual(progress(500)['common_step_counter'],132000)
        self.assertEqual(progress(500)['sim_step_counter'],1320000)
        for n in (-1,501,True,1.5):
            with self.assertRaises(ValueError):progress(n)

    def test_four_factorial_arms(self):
        self.assertEqual(set(PROFILES.values()),{(0.,0.),(-.25,0.),(0.,-.25),(-.25,-.25)})

    def test_finalization_time_is_reserved(self):
        self.assertTrue(budget_decision('5090',spent_seconds=0,additional_updates=2000,seconds_per_update=3.1)['admit'])
        self.assertFalse(budget_decision('5090',spent_seconds=3.25*3600,additional_updates=1,seconds_per_update=3.1)['admit'])
        with self.assertRaises(ValueError):budget_decision('5060',spent_seconds=0,additional_updates=1,seconds_per_update=1)

    def test_failed_attempts_are_charged_and_caps_survive_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'ledger.json'
            create_ledger(path,'5090',datetime.now(timezone.utc).isoformat())
            ledger=BudgetLedger(path)
            self.assertEqual(ledger.charge_update('F1'),1)
            self.assertEqual(BudgetLedger(path).snapshot()['charged_updates']['F1'],1)
            state=json.loads(path.read_text());state['charged_updates']['F1']=500;path.write_text(json.dumps(state))
            with self.assertRaises(ValueError):ledger.charge_update('F1')
            with self.assertRaises(FileExistsError):create_ledger(path,'5090',datetime.now(timezone.utc).isoformat())

    def test_frozen_mdp_prefix_and_old_factories(self):
        contract=json.loads((ROOT/'docs/audits/stage-09-approved-plan.json').read_text())
        data=(ROOT/'src/mjlab_microduck/tasks/mdp.py').read_bytes()
        self.assertEqual(hashlib.sha256(data[:contract['original_mdp_bytes']]).hexdigest(),contract['original_mdp_sha256'])
        for name,digest in contract['unchanged_runtime_files'].items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),digest,name)


RUNTIME=all(importlib.util.find_spec(n) is not None for n in ('torch','mjlab'))


@unittest.skipUnless(RUNTIME,'Requires the locked laptop runtime; no PPO is run')
class RewardRuntimeTests(unittest.TestCase):
    def test_heading_is_relative_to_trunk_and_vertical_cannot_evade_cost(self):
        import torch
        from mjlab_microduck.tasks.mdp import stage09_head_yaw_cost
        def q(yaw):return [math.cos(yaw/2),0,0,math.sin(yaw/2)]
        data=NS(site_quat_w=torch.tensor([q(.7),q(.7+math.pi/2),q(.7+math.pi)])[:,None],
                body_link_quat_w=torch.tensor([q(.7)]*3)[:,None])
        robot=NS(data=data,find_sites=lambda _:([0],['tray']),find_bodies=lambda _:([0],['trunk_base']))
        env=NS(scene={'robot':robot},num_envs=3,device='cpu')
        torch.testing.assert_close(stage09_head_yaw_cost(env),torch.tensor([0.,1.,2.]),atol=1e-6,rtol=1e-6)
        data.body_link_quat_w=torch.tensor([[1.,0,0,0]]*3)[:,None]
        data.site_quat_w=torch.tensor([[math.sqrt(.5),0,math.sqrt(.5),0]]*3)[:,None]
        torch.testing.assert_close(stage09_head_yaw_cost(env),torch.full((3,),2.))

    def test_limit_cost_uses_actual_qpos_and_increases_outside_range(self):
        import torch
        from mjlab_microduck.tasks.mdp import stage09_head_margin_cost
        names=['neck_pitch','head_pitch','head_yaw','head_roll']
        hard=torch.deg2rad(torch.tensor([[-90.,60.],[-90.,90.],[-170.,170.],[-25.,25.]]))
        q=torch.zeros(4,4);q[:,3]=torch.deg2rad(torch.tensor([0.,22.5,25.,27.5]))
        data=NS(joint_pos=q,joint_pos_limits=hard[None].expand(4,-1,-1))
        robot=NS(data=data,find_joints=lambda regex:([names.index(regex[1:-1])],[regex[1:-1]]))
        env=NS(scene={'robot':robot})
        torch.testing.assert_close(stage09_head_margin_cost(env),torch.tensor([0.,0.,1.,4.]),atol=1e-5,rtol=1e-5)

    def test_overlay_changes_only_approved_terms_on_E(self):
        from dataclasses import asdict
        from mjlab.tasks.registry import load_env_cfg
        from mjlab_microduck.double_balance_stage09_plan import apply_profile,TERMS
        from mjlab_microduck.double_balance_stage08_plan import apply_training_profile
        from mjlab_microduck.double_balance_smoke import TASK_ID
        import mjlab_microduck.tasks
        base=load_env_cfg(TASK_ID);base.scene.num_envs=4096
        base_before=asdict(base)
        expected=asdict(apply_training_profile(base,'E'))
        for name in PROFILES:
            with self.subTest(profile=name):
                actual=asdict(apply_profile(base,name))
                for key,weight in zip(TERMS,PROFILES[name]):
                    if weight:
                        self.assertIn(key,actual['rewards'])
                        self.assertEqual(actual['rewards'][key]['weight'],weight)
                    else:self.assertNotIn(key,actual['rewards'])
                    actual['rewards'].pop(key,None)
                differences=list(config_differences(actual,expected))
                self.assertFalse(differences,'\n'.join(differences[:20]))
                differences=list(config_differences(asdict(base),base_before))
                self.assertFalse(differences,'Input cfg mutated:\n'+'\n'.join(differences[:20]))


if __name__=='__main__':unittest.main()
