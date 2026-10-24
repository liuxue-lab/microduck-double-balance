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
import runtime_contract as contract
import runtime_worker as worker
import runtime_ops as ops


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
    return dict(status='EPISODE_INTERRUPTED',stop_reason='RUNTIME_SAMPLE_COMPLETE',completed_control_steps=160,
        observed_physics_steps=1600,checkpoint_unchanged=True,frozen_inputs_unchanged=True,
        policy_and_normalizer_unchanged=True,external_body_wrench_zero_at_close=True,
        model_limits=dict(any_model_bound_violation_observed=False,completed_physics_steps=1600,
            motor_checks_before_integration=1600,joint_state_checks=1601),
        disturbance=dict(completed_pulses=1,applied_pulses=1,applied_physics_steps=50,
            checked_physics_steps=1600,force_seed=2026102500,schedule_verified=True))


class ContractTests(unittest.TestCase):
    def test_reviewed_real_optimization_report(self):
        r=contract.reviewed(ROOT/'reference/kinematics-v1-measured.json')
        self.assertEqual(r['status'],'PC_KINEMATICS_MEASURED')
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
            rows=[dict(step=i+1,values=dict({k:[0.] for k in contract.TRACE_FIELDS},actor=[.2]*61,actions=[.3]*14)) for i in range(160)]
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
        self.assertEqual(accepted['choice'],'INFERENCE_ONLY_CANDIDATE')
        self.assertFalse(accepted['production_activated']);self.assertFalse(accepted['full_loop_realtime_acceptance'])
    def test_trace_cannot_omit_a_plant_field_from_both_arms(self):
        with tempfile.TemporaryDirectory() as d:
            a,b=Path(d)/'a',Path(d)/'b'
            values=dict({k:[0.] for k in contract.TRACE_FIELDS},actor=[0.]*61,actions=[0.]*14)
            values.pop('external_wrench')
            a.write_text(''.join(json.dumps(dict(step=i+1,values=values))+'\n' for i in range(160)))
            b.write_bytes(a.read_bytes())
            with self.assertRaisesRegex(RuntimeError,'fields'):contract.compare_traces(a,b)
    def test_full_audit_or_pulse_difference_is_visible(self):
        from copy import deepcopy
        report=json.loads((ROOT/'reference/kinematics-v1-measured.json').read_bytes())
        first=report['arms']['bam-graph'];second=deepcopy(first)
        second['episode_result']['disturbance']['force_schedule_path']='different-arm-output-path'
        self.assertTrue(all(contract.compare_audits(first,second).values()))
        second['episode_result']['model_limits']['motor_checks_before_integration']-=1
        second['episode_result']['trial']['injected_difference']=True
        second['episode_result']['disturbance']['applied_physics_steps']-=1
        self.assertFalse(any(contract.compare_audits(first,second).values()))
    def test_watch_continues_after_completion(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'report.json').write_text(json.dumps(dict(status='PC_RUNTIME_MEASURED',directory=d)))
            with patch.object(ops,'ROOT',root),patch.object(ops.time,'sleep',side_effect=KeyboardInterrupt),patch('builtins.print'):
                with self.assertRaises(KeyboardInterrupt):ops.show(watch=True)


class WorkerLifecycle(unittest.TestCase):
    def test_both_worker_modes_capture_equal_state_and_restore_hooks(self):
        import torch
        from test_bam_graph import fixture,OriginalArithmeticTests
        import bam_graph
        torch.set_num_threads(1)
        from test_head_graph import MDP,fixture as head_fixture
        from test_inference_only import fixture as manager_fixture
        import inference_only
        original=MDP._stage11_head_kinematics
        mdp=NS(**vars(MDP))
        raws=[];current_modules={}
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
                    result=integrate();raw._sim_step_counter+=1
                    raw.scene['robot'].data.site_pose_w[0,2,0]=raw.sim.data.qpos[0,0]
                    raw.scene['top_ball'].data.root_link_pos_w[0,1]=raw.sim.data.qpos[0,1]+.002
                    return result
                self.physics_step_original=counted;raw.sim.step=self.physics_step
                raw.sim.forward=lambda:None;raw.sim.sense=lambda:None;raw.sim.reset=lambda:None
                class Scene(dict):pass
                raw.scene=Scene(head_fixture().scene)
                raw.scene.write_data_to_sim=lambda:set_command(raw._sim_step_counter)
                raw.scene.update=lambda **_:None
                raw.action_manager=NS(process_action=lambda *_:None,apply_action=lambda:None)
                raw.command_manager=NS(compute=lambda **_:None)
                raw.termination_manager=NS(compute=lambda:False)
                def calculate():
                    a=mdp._stage11_head_kinematics(raw)[0];b=mdp._stage11_head_kinematics(raw)[0]
                    return a+b
                raw.termination_manager.compute=calculate
                raw.metrics_manager=NS(compute=calculate,compute_substep=lambda:None,_step_values=torch.zeros(1,1))
                self.probe=NS(original=calculate)
                def observe_probe():
                    self.probe.original();return mdp._stage11_head_kinematics(raw)
                raw.metrics_manager.compute=observe_probe
                def actor(env):
                    a=mdp._stage11_head_kinematics(env)[0];b=mdp._stage11_head_kinematics(env)[0]
                    return (a+b)[:,:1].repeat(1,61)
                def critic(env):return mdp._stage11_head_kinematics(env)[0][:,:1].repeat(1,85)
                # Construct the original observation-method fixture on this live
                # plant so all head probes read the same current physics state.
                setup,modules=manager_fixture()
                current_modules.clear();current_modules.update(modules)
                for name in ('num_envs','device','step_dt','physics_dt','cfg','extras','epoch',
                             'actor_source_calls','critic_source_calls','observation_manager','reward_manager'):
                    setattr(raw,name,getattr(setup,name))
                obs=raw.observation_manager;obs._env=raw
                obs._group_obs_term_cfgs['actor'][0].func=actor
                for i,cfg in enumerate(obs._group_obs_term_cfgs['critic']):
                    def term(env,index=i):
                        data=critic(env);return data[:,index:index+1] if index<13 else data[:,13:]
                    cfg.func=term
                    module,name=inference_only.CRITIC[obs._group_obs_term_names['critic'][i]]
                    setattr(modules[module],name,term)
                torch.manual_seed(2026102407)
                obs._obs_buffer=obs.compute(update_history=True);raws.append(raw)
            def after_reset(self,episode):pass
            def step(self,env,actions):
                raw=self.raw;raw.action_manager.process_action(actions)
                for _ in range(10):
                    raw.action_manager.apply_action();raw.scene.write_data_to_sim();raw.sim.step()
                    raw.scene.update(dt=.002);raw.metrics_manager.compute_substep()
                raw.common_step_counter+=1;raw.termination_manager.compute()
                reward=raw.reward_manager.compute(dt=.02);raw.metrics_manager.compute();raw.sim.forward()
                raw.command_manager.compute(dt=.02);raw.sim.sense()
                return raw.observation_manager.compute(update_history=True),reward,torch.zeros(1,dtype=torch.bool),{}
            def close(self):pass
        context=NS(EvaluationSession=Base,mdp=mdp,torch=torch)
        def evaluate(ctx,folder,case,cp,identity,boundary,progress):
            self.assertEqual(case['force_seed'],2026102500)
            session=ctx.EvaluationSession();session.after_reset(1);steps=0
            try:
                for _ in range(161):boundary();session.step(None,torch.zeros(1,14));steps+=1
            except worker.evaluate.EvaluationInterrupted as exc:self.assertEqual(str(exc),'RUNTIME_SAMPLE_COMPLETE')
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
            stack.enter_context(torch.inference_mode())
            stack.enter_context(patch.object(inference_only,'verify_sources',side_effect=lambda:(current_modules,{})))
            for mode in ('bam-graph','inference-only'):
                folder=root/mode;folder.mkdir()
                self.assertEqual(worker.run(folder,mode,root/'STOP'),0)
                r=json.loads((folder/'worker-report.json').read_bytes())
                self.assertEqual(r['control_steps'],160);self.assertTrue(r['runtime_bindings_restored'])
                self.assertFalse(r['complete_task_acceptance'])
                if mode=='inference-only':
                    self.assertEqual(r['inference_only_statistics']['omitted_reward_calls'],160)
                    self.assertEqual(r['inference_only_statistics']['actor_group_calls'],160)
                    self.assertEqual(raws[-1].reward_manager.calls,0)
                if mode=='bam-graph':self.assertGreater(r['cache_statistics']['cache_hits'],0)
                self.assertIs(mdp._stage11_head_kinematics,original)
            self.assertTrue(contract.compare_traces(root/'bam-graph/state-trace.jsonl',root/'inference-only/state-trace.jsonl')['passed'])


if __name__=='__main__':unittest.main()
