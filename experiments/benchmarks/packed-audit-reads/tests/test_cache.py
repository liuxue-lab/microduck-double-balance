"""CPU fixtures for cache scope, paired state checks and lifecycle; no hardware."""
import ast
from contextlib import ExitStack
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'package'))
from phase_cache import PhaseCache,Bindings
import audit_contract as contract
import audit_worker as worker
import audit_ops as ops


def scene():
    calls=[];raw=NS(epoch=0)
    def kin(env,reference_site='stage11_head_material_frame'):
        calls.append((id(env),env.epoch,reference_site));return (env.epoch,reference_site)
    mdp=NS(_stage11_head_kinematics=kin)
    def compute():return [mdp._stage11_head_kinematics(raw),mdp._stage11_head_kinematics(raw)]
    def advance():raw.epoch+=1
    raw.termination_manager=NS(compute=compute);raw.reward_manager=NS(compute=compute)
    raw.observation_manager=NS(compute=compute)
    raw.sim=NS(step=advance,forward=advance,reset=advance)
    raw.scene=NS(write_data_to_sim=advance)
    probe=NS(original=compute)
    return raw,mdp,probe,calls,kin


class ScopeTests(unittest.TestCase):
    def test_no_cache_outside_phase_and_one_calculation_inside(self):
        raw,mdp,probe,calls,original=scene();cache=PhaseCache(raw,mdp,probe)
        try:
            mdp._stage11_head_kinematics(raw);mdp._stage11_head_kinematics(raw)
            self.assertEqual(len(calls),2)
            self.assertEqual(raw.reward_manager.compute(),[(0,'stage11_head_material_frame')]*2)
            self.assertEqual(len(calls),3);self.assertFalse(cache.values)
            raw.epoch=4;self.assertEqual(raw.reward_manager.compute()[0][0],4)
            self.assertEqual(len(calls),4)
        finally:cache.close()
        self.assertIs(mdp._stage11_head_kinematics,original)
    def test_site_and_environment_identity_are_separate(self):
        raw,mdp,probe,calls,_=scene();cache=PhaseCache(raw,mdp,probe);other=NS(epoch=12)
        try:
            def invoke():
                for _ in range(2):
                    mdp._stage11_head_kinematics(raw,'one');mdp._stage11_head_kinematics(raw,'two')
                    mdp._stage11_head_kinematics(other)
            cache.scoped(invoke)()
            self.assertEqual(len(calls),4);self.assertEqual(cache.stats['cache_hits'],2)
        finally:cache.close()
    def test_every_state_write_barrier_invalidates_before_and_after(self):
        for target in ('step','forward','reset','write_data_to_sim'):
            with self.subTest(target=target):
                raw,mdp,probe,calls,_=scene()
                owner=raw.scene if target=='write_data_to_sim' else raw.sim
                def write():
                    # Even a helper read inside a state write cannot populate the later cache.
                    mdp._stage11_head_kinematics(raw);raw.epoch+=1
                setattr(owner,target,write);cache=PhaseCache(raw,mdp,probe)
                try:
                    def invoke():
                        before=mdp._stage11_head_kinematics(raw);getattr(owner,target)()
                        return before,mdp._stage11_head_kinematics(raw)
                    before,after=cache.scoped(invoke)()
                    self.assertEqual(before[0],0);self.assertEqual(after[0],1)
                    self.assertEqual(cache.stats['barrier_invalidations'],1)
                finally:cache.close()
    def test_metric_probe_recomputes_independently(self):
        raw,mdp,probe,calls,_=scene();cache=PhaseCache(raw,mdp,probe)
        try:
            probe.original();self.assertEqual(len(calls),1)
            mdp._stage11_head_kinematics(raw);self.assertEqual(len(calls),2)
            self.assertEqual(cache.depth,0)
        finally:cache.close()
    def test_exception_clears_scope_and_restores_bindings(self):
        raw,mdp,probe,calls,original=scene();cache=PhaseCache(raw,mdp,probe)
        def fail():mdp._stage11_head_kinematics(raw);raise ValueError('injected')
        with self.assertRaises(ValueError):cache.scoped(fail)()
        self.assertEqual(cache.depth,0);self.assertFalse(cache.values)
        cache.close();cache.close();self.assertIs(mdp._stage11_head_kinematics,original)
    def test_binding_cleanup_restores_class_method_without_instance_shadow(self):
        class Holder:
            def method(self):return 7
        obj=Holder();binding=Bindings();binding.set(obj,'method',lambda:9)
        self.assertEqual(obj.method(),9);binding.close()
        self.assertNotIn('method',vars(obj));self.assertEqual(obj.method(),7)


def valid_row():
    return dict(status='EPISODE_INTERRUPTED',stop_reason='AUDIT_SAMPLE_COMPLETE',completed_control_steps=160,
        observed_physics_steps=1600,checkpoint_unchanged=True,frozen_inputs_unchanged=True,
        policy_and_normalizer_unchanged=True,external_body_wrench_zero_at_close=True,
        model_limits=dict(any_model_bound_violation_observed=False,completed_physics_steps=1600,
            motor_checks_before_integration=1600,joint_state_checks=1601),
        disturbance=dict(completed_pulses=1,applied_pulses=1,applied_physics_steps=50,
            checked_physics_steps=1600,force_seed=2026102500,schedule_verified=True))


class ContractTests(unittest.TestCase):
    def test_reviewed_real_optimization_report(self):
        r=contract.reviewed(ROOT/'reference/bam-graph-snapshot-measured.json')
        self.assertEqual(r['status'],'PC_GRAPH_MEASURED')
    def test_partial_cannot_hide_missing_physics_or_force_checks(self):
        row=valid_row();self.assertTrue(contract.valid_partial(row))
        for field in ('completed_control_steps','observed_physics_steps'):
            with self.subTest(field=field):self.assertFalse(contract.valid_partial(dict(row,**{field:1})))
        for field in ('completed_pulses','applied_physics_steps','checked_physics_steps'):
            changed=dict(row,disturbance=dict(row['disturbance'],**{field:0}))
            self.assertFalse(contract.valid_partial(changed))
        self.assertFalse(contract.valid_partial(dict(row,cleanup_error='failed')))
    def test_trace_detects_recurrent_drift_and_incomplete_data(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a',Path(d)/'b'
            rows=[dict(step=i+1,values=dict(actor=[.2]*61,actions=[.3]*14)) for i in range(160)]
            a.write_text(''.join(json.dumps(r)+'\n' for r in rows));b.write_bytes(a.read_bytes())
            self.assertTrue(contract.compare_traces(a,b)['passed'])
            rows[-1]['values']['actions'][4]+=.01
            b.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            self.assertFalse(contract.compare_traces(a,b)['passed'])
            b.write_text('{}\n')
            with self.assertRaises((RuntimeError,KeyError)):contract.compare_traces(a,b)
    def test_speed_alone_cannot_activate_changed_results(self):
        base=dict(full_environment_step_ms=dict(mean=35.,p95=36.))
        fast=dict(full_environment_step_ms=dict(mean=15.,p95=16.))
        rejected=ops.recommendation(base,fast,dict(passed=False))
        self.assertEqual(rejected['choice'],'KEEP_BAM_GRAPH')
        accepted=ops.recommendation(base,fast,dict(passed=True))
        self.assertEqual(accepted['choice'],'PACKED_AUDIT_CANDIDATE')
        self.assertFalse(accepted['production_activated']);self.assertFalse(accepted['full_loop_realtime_acceptance'])
    def test_watch_continues_after_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'report.json').write_text(json.dumps(dict(status='PC_AUDIT_MEASURED',directory=d)))
            with patch.object(ops,'ROOT',root),patch.object(ops.time,'sleep',side_effect=KeyboardInterrupt),patch('builtins.print'):
                with self.assertRaises(KeyboardInterrupt):ops.show(watch=True)


class WorkerLifecycle(unittest.TestCase):
    def test_both_worker_modes_capture_equal_state_and_restore_hooks(self):
        import torch
        from test_bam_graph import fixture,OriginalArithmeticTests
        import bam_graph
        torch.set_num_threads(1)
        original=lambda env,reference_site='stage11_head_material_frame':env.sim.data.qpos.clone()
        mdp=NS(_stage11_head_kinematics=original)
        raws=[]
        from test_packed_checks import make_session
        from eval_runtime import EvaluationSession
        class Base(EvaluationSession):
            def __init__(self):
                session,set_command,events=make_session()
                self.__dict__.update(session.__dict__)
                raw=self.raw;raw.common_step_counter=10;raw._sim_step_counter=100
                raw._s12_stable_ticks=torch.zeros(1,dtype=torch.int64)
                raw._s12_success_ever=torch.zeros(1,dtype=torch.bool)
                integrate=self.physics_step_original
                def counted():
                    result=integrate();raw._sim_step_counter+=1;return result
                self.physics_step_original=counted;raw.sim.step=self.physics_step
                raw.sim.forward=lambda:None;raw.sim.sense=lambda:None;raw.sim.reset=lambda:None
                raw.scene=NS(write_data_to_sim=lambda:set_command(raw._sim_step_counter),update=lambda **_:None)
                raw.action_manager=NS(process_action=lambda *_:None,apply_action=lambda:None)
                raw.command_manager=NS(compute=lambda **_:None)
                raw.termination_manager=NS(compute=lambda:False)
                def calculate():
                    a=mdp._stage11_head_kinematics(raw);b=mdp._stage11_head_kinematics(raw)
                    return a+b
                raw.reward_manager=NS(compute=calculate)
                raw.metrics_manager=NS(compute=calculate,compute_substep=lambda:None,_step_values=torch.zeros(1,1))
                self.probe=NS(original=calculate)
                def observe_probe():
                    self.probe.original();return mdp._stage11_head_kinematics(raw)
                raw.metrics_manager.compute=observe_probe
                def obs(**_):
                    a=mdp._stage11_head_kinematics(raw);b=mdp._stage11_head_kinematics(raw)
                    return dict(actor=a[:,:1].repeat(1,61),critic=b[:,:1].repeat(1,85))
                raw.observation_manager=NS(compute=obs);raws.append(raw)
            def after_reset(self,episode):pass
            def step(self,env,actions):
                raw=self.raw;raw.action_manager.process_action(actions)
                for _ in range(10):
                    raw.action_manager.apply_action();raw.scene.write_data_to_sim();raw.sim.step()
                    raw.scene.update(dt=.002);raw.metrics_manager.compute_substep()
                raw.common_step_counter+=1;raw.termination_manager.compute()
                reward=raw.reward_manager.compute();raw.metrics_manager.compute();raw.sim.forward()
                raw.command_manager.compute(dt=.02);raw.sim.sense()
                return raw.observation_manager.compute(),reward,torch.zeros(1,dtype=torch.bool),{}
            def close(self):pass
        context=NS(EvaluationSession=Base,mdp=mdp,torch=torch)
        def evaluate(ctx,folder,case,cp,identity,boundary,progress):
            self.assertEqual(case['force_seed'],2026102500)
            session=ctx.EvaluationSession();session.after_reset(1);steps=0
            try:
                for _ in range(161):boundary();session.step(None,torch.zeros(1,14));steps+=1
            except worker.evaluate.EvaluationInterrupted as exc:self.assertEqual(str(exc),'AUDIT_SAMPLE_COMPLETE')
            finally:session.close()
            self.assertEqual(steps,160)
            self.assertEqual(session.physics_calls,1600)
            self.assertEqual(session.guard.audit.motor_checks,1600)
            self.assertEqual(session.guard.audit.joint_checks,1601)
            self.assertEqual(session.applied_physics_steps,50)
            return valid_row()
        with tempfile.TemporaryDirectory() as d,ExitStack() as stack:
            root=Path(d)
            stack.enter_context(patch.object(worker.old,'preflight',return_value=dict(package_identity='fixture')))
            stack.enter_context(patch.object(worker,'validate_report',return_value={}))
            stack.enter_context(patch.object(worker.evaluate,'load_runtime',return_value=context))
            stack.enter_context(patch.object(worker.evaluate,'evaluate_one',side_effect=evaluate))
            stack.enter_context(patch.object(torch.cuda,'synchronize',lambda:None))
            stack.enter_context(patch.object(bam_graph,'GraphedBam',side_effect=lambda group,t,cmd:OriginalArithmeticTests().emulate_capture(group,cmd)))
            for mode in ('bam-graph','packed-audit'):
                folder=root/mode;folder.mkdir()
                self.assertEqual(worker.run(folder,mode,root/'STOP'),0)
                r=json.loads((folder/'worker-report.json').read_bytes())
                self.assertEqual(r['control_steps'],160);self.assertTrue(r['runtime_bindings_restored'])
                self.assertFalse(r['complete_task_acceptance'])
                if mode=='bam-graph':self.assertGreater(r['cache_statistics']['cache_hits'],0)
                self.assertIs(mdp._stage11_head_kinematics,original)
            self.assertTrue(contract.compare_traces(root/'bam-graph/state-trace.jsonl',root/'packed-audit/state-trace.jsonl')['passed'])


if __name__=='__main__':unittest.main()
