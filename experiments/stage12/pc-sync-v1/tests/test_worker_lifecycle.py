"""Whole four-arm worker on original CPU fixtures; no real CUDA or speed claim."""
from contextlib import ExitStack
import json
import os
import math
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'package'))
import bootstrap
REF=Path(os.environ.get('MICRODUCK_REFERENCE_PACKAGE',str(bootstrap.BASE_PACKAGE)))
base_common=bootstrap.load_base(REF)
sys.path.append(str(Path(os.environ.get('MICRODUCK_REFERENCE_TESTS',str(bootstrap.PROJECT/'experiments/stage12/pc-timeline-v1/tests')))))
with patch.object(bootstrap,'load_base',return_value=base_common):
    import sync_worker as worker
import sync_contract as contract
import sync_adapters as adapters
import auxiliary
from test_cache import valid_row as old_valid_row

def valid_row():
    row=old_valid_row();row['stop_reason']='SYNC_SAMPLE_COMPLETE';return row

class TimelineWorkerLifecycle(unittest.TestCase):
    def test_both_worker_modes_capture_equal_state_and_restore_hooks(self):
        import torch
        from test_bam_graph import fixture,OriginalArithmeticTests
        import bam_graph
        torch.set_num_threads(1)
        from test_head_graph import MDP,MATH,fixture as head_fixture
        from test_inference_only import fixture as manager_fixture
        import inference_only
        import read_graph
        import bridge_views
        from bridge_fixture import BRIDGE,bridge_data
        from command_fixture import prepare_commands,MODULES
        entity_module=NS(quat_from_matrix=MATH.quat_from_matrix)
        class PoseData:
            def __init__(self,data):
                self.__dict__.update(vars(data));self._pose=vars(data)['site_pose_w'];del self.__dict__['site_pose_w']
            @property
            def site_pose_w(self):
                matrices=MATH.matrix_from_quat(self._pose[...,3:7])
                return torch.cat((self._pose[...,:3],entity_module.quat_from_matrix(matrices)),-1)
        original=MDP._stage11_head_kinematics
        mdp=NS(**vars(MDP))
        raws=[];current_modules={}
        from test_packed_checks import make_session
        from eval_runtime import EvaluationSession
        class Base(EvaluationSession):
            def __init__(self):
                session,set_command,events=make_session()
                self.__dict__.update(session.__dict__)
                self.raw.sim.data=bridge_data(self.raw.sim.data)
                prepare_commands(self.guard,delay=True)
                for group in self.guard.groups:group._delay_buffer.generator=None
                raw=self.raw;raw.common_step_counter=10;raw._sim_step_counter=100
                raw._s12_stable_ticks=torch.zeros(1,dtype=torch.int64)
                raw._s12_success_ever=torch.zeros(1,dtype=torch.bool)
                integrate=self.physics_step_original
                def counted():
                    result=integrate();raw._sim_step_counter+=1
                    raw.scene['robot'].data._pose[0,2,0]=raw.sim.data.qpos[0,0]
                    raw.scene['top_ball'].data.root_link_pos_w[0,1]=raw.sim.data.qpos[0,1]+.002
                    return result
                self.physics_step_original=counted;raw.sim.step=self.physics_step
                raw.sim.forward=lambda:None;raw.sim.sense=lambda:None;raw.sim.reset=lambda:None
                class Scene(dict):pass
                raw.scene=Scene(head_fixture().scene)
                raw.scene['robot'].data=PoseData(raw.scene['robot'].data)
                command_data=self.guard.robot.data
                def write_commands():
                    command_data.joint_pos_target.fill_(.02*math.sin(raw._sim_step_counter*.021))
                    group=self.guard.groups[0]
                    cmd=group.apply_delay(group.get_command(command_data))
                    raw.sim.data.ctrl[:,group.global_ctrl_ids]=group.compute(cmd)
                raw.scene.write_data_to_sim=write_commands
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
            except worker.evaluate.EvaluationInterrupted as exc:self.assertEqual(str(exc),'SYNC_SAMPLE_COMPLETE')
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
            all_modules=dict(MODULES,**{'mjlab.utils.lab_api.math':MATH,'mjlab.entity.data':entity_module})
            stack.enter_context(patch.object(adapters,'verify_sources',side_effect=lambda names:({n:all_modules[n] for n in names},{})))
            stack.enter_context(patch.object(inference_only,'verify_sources',side_effect=lambda:(current_modules,{})))
            stack.enter_context(patch.object(read_graph,'verify_entity',return_value={}))
            original_import=bridge_views.importlib.import_module
            stack.enter_context(patch.object(bridge_views.importlib,'import_module',
                side_effect=lambda name:BRIDGE if name=='mjlab.sim.sim_data' else original_import(name)))
            for mode in ('baseline','quat','delay','combined'):
                folder=root/mode;folder.mkdir()
                self.assertEqual(worker.run(folder,mode,root/'STOP'),0)
                r=json.loads((folder/'worker-report.json').read_bytes())
                self.assertEqual(r['control_steps'],160);self.assertTrue(r['runtime_bindings_restored'])
                self.assertFalse(r['complete_task_acceptance'])
                self.assertFalse(r['command_views_enabled']);self.assertFalse(r['cprofile_enabled'])
                for key in ('original_sim_step','audit_before_python','audit_after_python',
                            'before_read_and_audit','after_read_and_audit'):
                    self.assertEqual(r['manager_host_time_ms'][key]['count'],1500)
                if mode in ('quat','combined'):
                    self.assertEqual(r['static_quaternion']['checked_calls'],40)
                    self.assertTrue(r['static_quaternion']['closed'])
                if mode in ('delay','combined'):
                    self.assertEqual(r['fixed_delays']['groups'][0]['rng_comparisons'],40)
                    self.assertEqual(r['fixed_delays']['groups'][0]['calls'],1600)
                    self.assertTrue(r['fixed_delays']['closed'])
                self.assertEqual(r['batched_audit_calls'],1600)
                if mode in ('baseline','quat','delay','combined'):
                    self.assertEqual(r['inference_only_statistics']['omitted_reward_calls'],160)
                    self.assertEqual(r['inference_only_statistics']['actor_group_calls'],160)
                    self.assertEqual(raws[-1].reward_manager.calls,0)
                if mode=='baseline':self.assertGreater(r['cache_statistics']['cache_hits'],0)
                self.assertIs(mdp._stage11_head_kinematics,original)
            for mode in ('quat','delay','combined'):
                comparison=contract.compare_traces(root/'baseline/state-trace.jsonl',root/mode/'state-trace.jsonl')
                self.assertTrue(comparison['passed'])
                self.assertTrue(all(v==0 for v in comparison['max_absolute_error_by_field'].values()))
                self.assertTrue(auxiliary.compare(root/'baseline/aux-state.jsonl',root/mode/'aux-state.jsonl')['passed'])



if __name__=='__main__':unittest.main()
