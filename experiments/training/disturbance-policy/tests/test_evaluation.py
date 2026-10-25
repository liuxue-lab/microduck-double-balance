"""CPU fixtures and pure evaluation contracts; no robot simulation or training."""
from pathlib import Path
from copy import deepcopy
import contextlib
import math
import signal
import sys
import tempfile
import time
import types
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"package"))
import torch
import eval_runtime as runtime
import evaluate
from eval_plan import *
from eval_criteria import *
from budget import new as new_budget

def stable_sample():
    return dict(stable=True, root_xy=[0.0, 0.0], values=dict(top_center_m=0.001, top_height_error_m=0.001, top_speed_m_s=0.01, lower_offset_m=0.01, lower_height_m=0.25, robot_tilt_rad=0.01, lower_ball_speed_m_s=0.01, hold=0.0))

class FakeProbe:

    def __init__(self, raw, torch_module, mdp):
        self.calls = 0
        self.last = None

    def close(self):
        pass

class FakeGuard:

    def __init__(self, raw, torch_module, run):
        self.before = self.after = self.reset = 0

    def after_reset(self):
        self.reset += 1

    def before_integration(self):
        self.before += 1

    def after_integration(self):
        self.after += 1

    def snapshot(self):
        return dict(completed_physics_steps=self.after, motor_checks_before_integration=self.before, joint_state_checks=self.after + self.reset)

    def close(self):
        pass

class FakeRaw:
    """Models a scene writer followed by integration, without simulating physics."""

    def __init__(self):
        NS = types.SimpleNamespace
        self.num_envs, self.step_dt, self.physics_dt = (1, 0.02, 0.002)
        self.cfg = NS(decimation=10)
        self.scene = {'robot': NS(indexing=NS(body_ids=torch.tensor([2, 3, 4])), data=NS(root_link_pos_w=torch.zeros(1, 3)), find_bodies=lambda pattern: ([0], ['trunk_base'])), 'ball': NS(indexing=NS(body_ids=torch.tensor([0]))), 'top_ball': NS(indexing=NS(body_ids=torch.tensor([1])))}
        self.sim = NS(data=NS(xfrc_applied=torch.zeros(1, 5, 6)), model=NS(body_mass=torch.tensor([[100.0, 200.0, 0.3, 0.4, 0.1]])), mj_model=NS(opt=NS(gravity=[0.0, 0.0, -9.81])))
        self.sim.step = self.integrate
        self.integrated = 0
        self.force_observations = []
        self.raise_on_call = None

    def integrate(self):
        if self.raise_on_call == self.integrated + 1:
            raise RuntimeError('injected integrator failure')
        value = self.sim.data.xfrc_applied.clone()
        if bool((value != 0).any()):
            self.force_observations.append((self.integrated, value))
        self.integrated += 1

class FakeEnv:

    def __init__(self, raw, session):
        self.raw, self.session = (raw, session)

    def step(self, actions):
        for _ in range(10):
            self.raw.sim.data.xfrc_applied.zero_()
            self.raw.sim.step()
        self.session.probe.calls += 1
        self.session.probe.last = stable_sample()
        return 'original-step-return'

def fixture(case,checkpoint=None,margin=.4):
    checkpoint=checkpoint or dict(path='/unused/model.pt',sha256='candidate-a',bytes=1,completed_updates=1,policy_updates=6011)
    joints=[dict(name='head_roll' if i==0 else 'joint_'+str(i),
        min_q_rad=-1.+2*margin,max_q_rad=1.-2*margin,limit_violation_samples=0,
        initial_limit_violation=False,peak_nominal_excursion_rad=0.) for i in range(14)]
    bounds=[dict(name=j['name'],lower_rad=-1.,upper_rad=1.) for j in joints]
    enabled=case['force_seed'] is not None
    magnitude=.1*.8*9.81 if enabled else 0.
    pulses=[]
    for plan in make_schedule(case['force_seed']):
        vector=[magnitude*v for v in plan['unit_force_world']]
        pulses.append(dict(plan,complete=True,applied_physics_steps=50,force_write_attempts=50,
            actual_start_seconds=plan['start_seconds'],actual_end_seconds=plan['end_seconds'],
            force_world_N=vector,actual_force_N=magnitude,
            applied_impulse_N_s=magnitude*.1,applied_impulse_world_N_s=[v*.1 for v in vector]))
    row=dict(case,status='EPISODE_COMPLETE',checkpoint=checkpoint,execution_completed=True,
        completed_control_steps=3000,observed_physics_steps=30000,added_PPO_updates=0,training_started=False,
        policy_and_normalizer_unchanged=True,checkpoint_unchanged=True,frozen_inputs_unchanged=True,
        external_body_wrench_zero_at_close=True,
        trial=dict(control_steps=3000,completed_60_seconds=True,goal_60s_no_physical_termination_met=True,
            physically_terminated=False,terminations=['time_out'],duration_seconds=60.),
        model_limits=dict(status='OBSERVED_WITHIN_CHECKED_MODEL_BOUNDS',completed_physics_steps=30000,motor_checks_before_integration=30000,
            joint_state_checks=30001,any_model_bound_violation_observed=False,first_limit_event=None,
            stopping_limit_event=None,joints=joints,motors=[dict(name=j['name']) for j in joints],
            boundaries=dict(joints=bounds)),
        disturbance=dict(force_seed=case['force_seed'],disturbance_enabled=enabled,
            planned_pulses=len(pulses),applied_pulses=len(pulses),completed_pulses=len(pulses),
            applied_physics_steps=50*len(pulses),checked_physics_steps=30000,schedule_verified=True,
            force_fraction=.1 if enabled else 0.,period_seconds=2.,duration_seconds=.1,
            robot_mass_kg=.8,gravity_m_s2=9.81,force_newtons=magnitude,pulses=pulses))
    row['acceptance']=accept_case(row,case,checkpoint['sha256'])
    return row


def candidate(sha='candidate-a',margin=.4,updates=1):
    cp=dict(path='/unused/'+sha+'.pt',sha256=sha,bytes=1,completed_updates=updates,policy_updates=6010+updates)
    rows=[fixture(case,cp,margin) for case in case_plan('selection')]
    return dict(status='EVALUATION_COMPLETE',phase='selection',checkpoint=cp,cases=rows,
                summary=summarize_phase(rows,'selection',sha))


class PureEvaluation(unittest.TestCase):
    def test_fixed_disjoint_selection_and_heldout_plan(self):
        self.assertEqual(len(case_plan('selection')),3)
        heldout=case_plan('heldout')
        self.assertEqual(len(heldout),18)
        self.assertEqual(sum(c['force_seed'] is None for c in heldout),2)
        self.assertFalse({c['force_seed'] for c in heldout if c['force_seed']} &
                         {c['force_seed'] for c in case_plan('selection') if c['force_seed']})
        self.assertEqual(len(make_schedule(2026102800)),29)
        self.assertEqual(make_schedule(None),[])
        with self.assertRaises(RuntimeError):make_schedule(2026102410)

    def test_full_quiet_and_push_cases_pass(self):
        for case in case_plan('heldout'):
            row=fixture(case)
            self.assertTrue(accept_case(row,case,'candidate-a')['passed'])
        self.assertTrue(candidate()['summary']['qualified'])

    def test_drop_or_joint_violation_permanently_fails(self):
        case=case_plan('selection')[1]
        for key in ('drop','joint'):
            row=fixture(case)
            if key=='drop':
                row['trial'].update(physically_terminated=True,terminations=['top_ball_lost'])
            else:
                row['model_limits']['joints'][0]['limit_violation_samples']=1
                row['model_limits']['joints'][0]['peak_nominal_excursion_rad']=.01
            self.assertFalse(accept_case(row,case,'candidate-a')['passed'])

    def test_wrong_seed_direction_impulse_or_check_counts_fail(self):
        case=case_plan('selection')[1]
        for field,value in [('force_world_N',[0.,0.,0.]),('applied_impulse_N_s',0.),
                            ('actual_end_seconds',2.12),('applied_physics_steps',49)]:
            row=fixture(case);row['disturbance']['pulses'][0][field]=value
            self.assertFalse(accept_case(row,case,'candidate-a')['passed'])
        row=fixture(case);row['model_limits']['joint_state_checks']=30000
        self.assertFalse(accept_case(row,case,'candidate-a')['passed'])
        row=fixture(case);row['force_seed']=2026102700
        self.assertFalse(accept_case(row,case,'candidate-a')['passed'])

    def test_no_quiet_regression_and_best_actual_joint_margin(self):
        a=candidate('a',.2,20);b=candidate('b',.4,10)
        self.assertEqual(choose_candidate([a,b])['checkpoint']['sha256'],'b')
        bad=candidate('bad',.49,30)
        bad['cases'][0]['trial']['physically_terminated']=True
        bad['summary']=summarize_phase(bad['cases'],'selection','bad')
        self.assertEqual(choose_candidate([bad,a])['checkpoint']['sha256'],'a')
        self.assertFalse(choose_candidate([bad])['qualified'])

    def test_mixed_duplicate_incomplete_and_forged_summary_rejected(self):
        good=candidate()
        with self.assertRaises(RuntimeError):choose_candidate([good,good])
        bad=deepcopy(good);bad['cases'][1]['checkpoint']={'sha256':'other'}
        with self.assertRaises(RuntimeError):choose_candidate([bad])
        bad=deepcopy(good);bad['cases']=bad['cases'][:-1]
        with self.assertRaises(RuntimeError):choose_candidate([bad])
        bad=deepcopy(good);bad['summary']['minimum_joint_margin_fraction']=999.
        with self.assertRaises(RuntimeError):choose_candidate([bad])


class RuntimeEvaluation(unittest.TestCase):
    def test_quiet_and_push_use_exact_same_physics_count_and_clear_forces(self):
        for force_seed in (None,2026102800):
            with self.subTest(force_seed=force_seed),tempfile.TemporaryDirectory() as tmp:
                raw=FakeRaw()
                with patch.object(runtime,'FreshStateGuard',FakeGuard),patch.object(runtime,'Probe',FakeProbe):
                    session=runtime.EvaluationSession(raw,torch,None,Path(tmp),2026102407,force_seed)
                    session.after_reset(1)
                    env=FakeEnv(raw,session)
                    for _ in range(3000):session.step(env,None)
                    session.finish('time_out',False,['time_out'])
                    value=session.snapshot()
                    self.assertEqual(value['observed_physics_steps'],30000)
                    self.assertEqual(value['disturbance']['checked_physics_steps'],30000)
                    self.assertEqual(value['disturbance']['completed_pulses'],0 if force_seed is None else 29)
                    self.assertEqual(value['disturbance']['applied_physics_steps'],0 if force_seed is None else 1450)
                    self.assertEqual(len(raw.force_observations),0 if force_seed is None else 1450)
                    self.assertEqual(value['disturbance']['force_newtons'],0. if force_seed is None else .1*value['disturbance']['robot_mass_kg']*9.81)
                    self.assertFalse(bool(raw.sim.data.xfrc_applied.any()))
                    with self.assertRaises(RuntimeError):session.after_reset(2)
                    session.close()

    def test_unknown_force_not_hidden(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw=FakeRaw()
            with patch.object(runtime,'FreshStateGuard',FakeGuard),patch.object(runtime,'Probe',FakeProbe):
                session=runtime.EvaluationSession(raw,torch,None,Path(tmp),2026102407,None)
                session.after_reset(1)
                raw.sim.data.xfrc_applied[0,0,0]=.5
                with self.assertRaises(RuntimeError):session.step(FakeEnv(raw,session),None)
                self.assertEqual(float(raw.sim.data.xfrc_applied[0,0,0]),.5)
                raw.sim.data.xfrc_applied.zero_()
                session.close()


class EvaluationLifecycle(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.stack=contextlib.ExitStack()
        for name,value in [('SLOT',self.root),('verify_package',lambda:'fixture'),
                           ('validate_checkpoint',lambda cp:None),('load_runtime',lambda:None)]:
            self.stack.enter_context(patch.object(evaluate,name,value))
        self.cp=candidate()['checkpoint'];self.cp_path=self.root/'candidate.json'
        evaluate.write(self.cp_path,self.cp)
        evaluate.write(self.root/'budget.json',new_budget(time.time()))
        self.calls=[]
        def fake(context,run,case,checkpoint,package_id,boundary,progress):
            boundary();self.calls.append(case['case_id']);return fixture(case,checkpoint)
        self.stack.enter_context(patch.object(evaluate,'evaluate_one',side_effect=fake))

    def tearDown(self):
        signal.alarm(0);self.stack.close();self.tmp.cleanup()

    def run_phase(self,phase='selection'):
        folder=self.root/'evaluations'/('selection-001' if phase=='selection' else 'heldout')
        argv=['evaluate','--phase',phase,'--checkpoint-json',str(self.cp_path),'--run-dir',str(folder)]
        with patch.object(sys,'argv',argv):
            code=evaluate.main()
        return code,evaluate.read(folder/'evaluation-result.json')

    def prepare_lock(self,result):
        selection=choose_candidate([result])
        evaluate.write(self.root/'selection.json',selection,exclusive=True)
        evaluate.write(self.root/'selected-candidate.json',selection['checkpoint'],exclusive=True)

    def test_selection_reserves_each_case_once_and_repeat_is_readonly(self):
        code,result=self.run_phase()
        self.assertEqual(code,0);self.assertTrue(result['summary']['qualified'])
        self.assertEqual(len(evaluate.read(self.root/'budget.json')['evaluations']),3)
        self.assertEqual(len(self.calls),3)
        code,again=self.run_phase()
        self.assertEqual(code,0);self.assertEqual(len(self.calls),3);self.assertEqual(result,again)

    def test_heldout_first_failure_stops_without_replacement_or_reselection(self):
        _,selection=self.run_phase();self.prepare_lock(selection)
        def fail_first_push(context,run,case,checkpoint,package_id,boundary,progress):
            row=fixture(case,checkpoint)
            if case['force_seed'] is not None:
                row['trial'].update(physically_terminated=True,terminations=['top_ball_lost'])
                row['acceptance']=accept_case(row,case,checkpoint['sha256'])
            return row
        with patch.object(evaluate,'evaluate_one',side_effect=fail_first_push):
            code,result=self.run_phase('heldout')
        self.assertEqual(code,0);self.assertFalse(result['summary']['all_passed'])
        self.assertEqual(result['evaluated_cases'],3)
        self.assertTrue(result['stopped_on_first_failure'])
        self.assertEqual(len(evaluate.read(self.root/'budget.json')['evaluations']),6)

    def test_heldout_cannot_bypass_qualification_or_change_checkpoint(self):
        _,selection=self.run_phase();self.prepare_lock(selection)
        lock=evaluate.read(self.root/'selection.json');lock['qualified']=False
        evaluate.write(self.root/'selection.json',lock)
        with self.assertRaises(RuntimeError):self.run_phase('heldout')
        self.assertEqual(len(evaluate.read(self.root/'budget.json')['evaluations']),3)

    def test_deadline_is_incomplete_not_policy_failure_and_no_retry(self):
        with patch.object(evaluate,'load_runtime',side_effect=evaluate.EvaluationInterrupted('deadline')):
            code,result=self.run_phase()
        self.assertEqual(code,3);self.assertEqual(result['status'],'EVALUATION_INCOMPLETE_PRESERVE')
        self.assertEqual(result['evaluated_cases'],0);self.assertFalse(result['summary']['all_passed'])
        code,_=self.run_phase()
        self.assertEqual(code,3);self.assertEqual(len(self.calls),0)

    def test_partial_phase_marker_refuses_resume(self):
        folder=self.root/'evaluations'/'selection-001';folder.mkdir(parents=True)
        evaluate.write(folder/'phase-started.json',{'old':True},exclusive=True)
        with self.assertRaises(FileExistsError):self.run_phase()
        self.assertFalse(self.calls)


class EvaluationWorkerFlow(unittest.TestCase):
    def run_fixture(self,terminal_step=3000,broken=False):
        case=case_plan('selection')[1];cp=candidate()['checkpoint']
        counts=dict(reset=0,policy=0,physics=0,env_close=0,session_close=0,load=0)
        outer=self
        def obs():return {'actor':torch.zeros(1,61),'critic':torch.zeros(1,85)}
        class TD(dict):
            def __init__(self,values,batch_size):super().__init__(values)
        class Policy:
            training=False
            obs_normalizer=types.SimpleNamespace(training=False)
            def reset(self):outer.assertTrue(torch.is_inference_mode_enabled())
            def state_dict(self):return {'weight':torch.ones(1)}
            def __call__(self,observations,stochastic_output):
                outer.assertFalse(stochastic_output);counts['policy']+=1
                return torch.zeros(1,14)
        class Raw:
            def __init__(self,cfg,**kw):
                self.cfg=cfg;self.max_episode_length=3000;self.step_dt=.02;self.physics_dt=.002
                self.steps=0;self.reset_terminated=torch.tensor([False])
                self.sim=types.SimpleNamespace(data=types.SimpleNamespace(xfrc_applied=torch.zeros(1,3,6)))
                self.termination_manager=types.SimpleNamespace(active_terms=['top_ball_lost','time_out'],get_term=self.term)
            def term(self,name):
                return torch.tensor([self.steps>=terminal_step and name==('top_ball_lost' if terminal_step<3000 else 'time_out')])
            def reset(self,seed):
                outer.assertTrue(torch.is_inference_mode_enabled());counts['reset']+=1
                return obs(),{}
            def close(self):counts['env_close']+=1
        class Runner:
            def __init__(self,*a,**kw):self.alg=self
            def load(self,saved,options,strict):
                outer.assertFalse(options['optimizer']);outer.assertFalse(options['iteration'])
                outer.assertTrue(strict);counts['load']+=1
            def get_inference_policy(self,**kw):return Policy()
            def learn(self,*a,**kw):raise AssertionError('No PPO during evaluation')
        class Session:
            def __init__(self,raw,*a):
                self.raw=raw;self.physics_calls=0
                outer.assertEqual(raw.common_step_counter,6010*24)
                outer.assertEqual(raw._sim_step_counter,6010*24*10)
            def after_reset(self,n):outer.assertEqual(n,1)
            def step(self,env,actions):
                if broken:raise RuntimeError('fake integration error')
                self.raw.steps+=1;self.physics_calls+=10;counts['physics']+=10
                done=self.raw.steps>=terminal_step
                self.raw.reset_terminated[0]=done and terminal_step<3000
                return TD(obs(),[1]),torch.zeros(1),torch.tensor([done]),{}
            def finish(self,*a):pass
            def snapshot(self):
                value=fixture(case,cp)
                steps=self.raw.steps;complete=steps==3000 and terminal_step==3000
                value['trial'].update(control_steps=steps,duration_seconds=steps*.02,completed_60_seconds=complete,
                    goal_60s_no_physical_termination_met=complete,physically_terminated=terminal_step<3000,
                    terminations=['time_out'] if complete else ['top_ball_lost'])
                value['observed_physics_steps']=steps*10
                value['model_limits'].update(completed_physics_steps=steps*10,
                    motor_checks_before_integration=steps*10,joint_state_checks=steps*10+1)
                return {k:value[k] for k in ('trial','model_limits','disturbance','observed_physics_steps')}
            def close(self):counts['session_close']+=1
        context=types.SimpleNamespace(torch=torch,TensorDict=TD,ManagerBasedRlEnv=Raw,
            RslRlVecEnvWrapper=lambda raw:raw,MjlabOnPolicyRunner=Runner,MicroduckBasketballRlCfg=object(),
            make_cfg=lambda *a:types.SimpleNamespace(episode_length_s=10.,curriculum={},events={},decimation=10),
            head_scale=lambda raw:1.,EvaluationSession=Session,mdp=None)
        saved=dict(infos=dict(stage12_final6h=dict(package_identity='fixture',policy_updates=cp['policy_updates'],
            origin_checkpoint_sha256=evaluate.MODEL_SHA,common_step_counter=6010*24,sim_step_counter=6010*24*10)))
        with tempfile.TemporaryDirectory() as tmp,patch.object(evaluate,'asdict',return_value={'algorithm':{}}), \
             patch.object(evaluate,'validate_checkpoint'),patch.object(evaluate,'verify_inputs',return_value={'pinned':True}), \
             patch.object(evaluate,'sha',return_value=cp['sha256']),patch.object(torch,'load',return_value=saved):
            result=evaluate.evaluate_one(context,Path(tmp),case,cp,'fixture',lambda:None,lambda **kw:None)
        return counts,result

    def test_one_complete_frozen_episode_without_optimizer_or_retry(self):
        counts,row=self.run_fixture()
        self.assertEqual(counts,dict(reset=1,policy=3000,physics=30000,env_close=1,session_close=1,load=1))
        self.assertEqual(row['status'],'EPISODE_COMPLETE');self.assertTrue(row['acceptance']['passed'])

    def test_early_physical_failure_keeps_original_termination(self):
        counts,row=self.run_fixture(terminal_step=113)
        self.assertEqual((counts['reset'],counts['policy'],counts['physics']),(1,113,1130))
        self.assertEqual(row['status'],'EPISODE_COMPLETE');self.assertFalse(row['acceptance']['passed'])

    def test_runtime_error_preserves_and_closes(self):
        counts,row=self.run_fixture(broken=True)
        self.assertEqual(row['status'],'EPISODE_ERROR_PRESERVE')
        self.assertEqual((counts['env_close'],counts['session_close']),(1,1))
        self.assertFalse(row['acceptance']['passed'])



if __name__=='__main__':
    unittest.main(verbosity=2)


