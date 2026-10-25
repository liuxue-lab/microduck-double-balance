"""Instrumented bounded-session fixture, no simulator/GPU initialization."""
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'package'))
import profile_sim as p


class ProfileBound(unittest.TestCase):
    def exercise(self,violation):
        integrated=[];synced=[];restored=[]
        def original():integrated.append(1)
        raw=SimpleNamespace(sim=SimpleNamespace(step=original))
        class Base:
            def __init__(self,*args,**kwargs):
                self.raw=raw;self.physics_step_original=raw.sim.step
                self.guard=SimpleNamespace(before_integration=lambda:None,after_integration=lambda:None)
            def step(self,env,actions):
                for _ in range(10):
                    self.guard.before_integration();self.physics_step_original();self.guard.after_integration()
                return 'unchanged-step'
            def close(self):self.raw.sim.step=self.physics_step_original
        base=SimpleNamespace(EvaluationSession=Base,torch=SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda:synced.append(1))))
        def evaluate(context,folder,case,cp,identity,boundary,progress):
            self.assertIsNone(case['force_seed'])
            session=context.EvaluationSession()
            try:
                for _ in range(101):
                    boundary();self.assertEqual(session.step(None,None),'unchanged-step')
            except p.evaluate.EvaluationInterrupted as exc:
                reason=str(exc)
            finally:
                session.close();restored.append(raw.sim.step is original)
            return dict(stop_reason=reason,observed_physics_steps=len(integrated),frozen_inputs_unchanged=True,
                policy_and_normalizer_unchanged=True,checkpoint_unchanged=True,external_body_wrench_zero_at_close=True,
                model_limits=dict(any_model_bound_violation_observed=violation))
        with tempfile.TemporaryDirectory() as d,ExitStack() as stack:
            stack.enter_context(patch.object(p.old,'preflight',return_value=dict(package_identity='fixture')))
            stack.enter_context(patch.object(p,'validate_report',return_value={}))
            stack.enter_context(patch.object(p.evaluate,'load_runtime',return_value=base))
            stack.enter_context(patch.object(p.evaluate,'evaluate_one',side_effect=evaluate))
            code=p.run(d);result=json.loads((Path(d)/'profile-report.json').read_bytes())
        self.assertEqual(len(integrated),1000);self.assertEqual(len(synced),1000);self.assertEqual(restored,[True])
        self.assertEqual(result['control_steps'],100);self.assertEqual(result['physics_steps'],1000)
        self.assertFalse(result['task_validation_claimed']);self.assertFalse(result['motor_io'])
        self.assertEqual(result['added_PPO_updates'],0)
        self.assertEqual(code,2 if violation else 0)
        return result
    def test_100_steps_only_and_original_integrator_restored(self):self.exercise(False)
    def test_model_bound_violation_is_not_called_complete(self):self.exercise(True)


if __name__=='__main__':unittest.main()
