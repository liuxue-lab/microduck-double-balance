"""New CPU checks for finite orchestration; no physical simulation or PPO."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'package'))
import budget
import criteria
import common
import agent

spec=importlib.util.spec_from_file_location('headroom_client_test',ROOT/'controller.py')
client=importlib.util.module_from_spec(spec);spec.loader.exec_module(client)


def success(seed=2026102302,digest='a'*64):
    joints=[dict(name='joint-'+str(i),limit_violation_samples=0,initial_limit_violation=False,
                 peak_nominal_excursion_rad=0.,min_q_rad=-.1,max_q_rad=.1) for i in range(14)]
    return dict(status='EPISODE_COMPLETE',seed=seed,checkpoint=dict(sha256=digest),
        baseline_trial=dict(control_steps=3000,completed_60_seconds=True,
            goal_60s_no_physical_termination_met=True,physically_terminated=False,
            terminations=['time_out'],max_root_drift_from_reset_m=1000,stable_fraction=0),
        model_limits=dict(joints=joints,motors=[dict(name='joint-'+str(i)) for i in range(14)],
            completed_physics_steps=30000,motor_checks_before_integration=30000,joint_state_checks=30001,
            any_model_bound_violation_observed=False,first_limit_event=None,stopping_limit_event=None),
        observed_physics_steps=30000,zero_wrench_checks_before_physics=30000,
        disturbance_enabled=False,applied_disturbance_pulses=0,external_body_wrench_zero_at_close=True,
        policy_and_normalizer_unchanged=True,checkpoint_unchanged=True)



def fixture_budget(now):
    value=json.loads((ROOT/'reference/failure-v3/budget.json').read_bytes())
    value['started_utc']=now;value['deadline_utc']=now+7200
    return value

class BudgetTests(unittest.TestCase):
    def test_repair_cannot_create_or_refund_budget(self):
        with self.assertRaisesRegex(RuntimeError,'cannot create'):budget.new(100)
        ledger=fixture_budget(100)
        self.assertEqual(budget.MAX_UPDATES-ledger['attempted_updates'],1504)
        self.assertEqual(budget.TARGETS['smoke']-ledger['completed_updates'],5)
        self.assertEqual(sum(v for k,v in budget.PHASE_CAPS.items() if k!='smoke'),1499)

    def test_parent_restart_separates_saved_weights_from_spent_accounting(self):
        with tempfile.TemporaryDirectory() as directory,patch.object(common,'SLOT',Path(directory)):
            checkpoint=common.input_checkpoint()
        self.assertEqual(checkpoint['completed_updates'],0)
        self.assertEqual(checkpoint['policy_updates'],5505)
        self.assertEqual(common.checkpoint_accounting_position(checkpoint),1)
        self.assertEqual(common.retained_policy_updates(6),5510)
        self.assertEqual(common.retained_policy_updates(1505),7009)

    def test_exact_1505_ceiling_and_phase_accounting(self):
        ledger=fixture_budget(10)
        for phase,n in list(budget.PHASE_CAPS.items()):
            for _ in range(n-ledger['phase_attempts'][phase]):budget.reserve_update(ledger,phase,20);budget.commit_update(ledger,phase)
        self.assertEqual((ledger['attempted_updates'],ledger['completed_updates']),(1505,1505))
        with self.assertRaises(RuntimeError):budget.reserve_update(ledger,'segment-2',20)
        self.assertEqual(ledger['phase_attempts'],budget.PHASE_CAPS)

    def test_failure_remains_spent_and_blocks_replay(self):
        ledger=fixture_budget(0);budget.reserve_update(ledger,'smoke',1)
        with self.assertRaises(RuntimeError):budget.reserve_update(ledger,'smoke',2)
        with self.assertRaises(RuntimeError):budget.resumable(ledger,0,2)
        self.assertEqual(ledger['attempted_updates'],2)
        self.assertEqual(ledger['completed_updates'],1)

    def test_phase_order_and_commit_without_reservation(self):
        ledger=fixture_budget(0)
        with self.assertRaises(RuntimeError):budget.reserve_update(ledger,'segment-1',1)
        with self.assertRaises(RuntimeError):budget.commit_update(ledger,'smoke')
        self.assertEqual(ledger['attempted_updates'],1)

    def test_deadline_cannot_be_refreshed_by_resume(self):
        ledger=fixture_budget(100)
        with self.assertRaises(RuntimeError):budget.reserve_update(ledger,'smoke',7300)
        with self.assertRaises(RuntimeError):budget.resumable(ledger,0,7300)
        self.assertEqual(ledger['deadline_utc'],7300)

    def test_checkpoint_lag_cannot_be_retrained_for_free(self):
        ledger=fixture_budget(0);budget.reserve_update(ledger,'smoke',1);budget.commit_update(ledger,'smoke')
        with self.assertRaises(RuntimeError):budget.resumable(ledger,0,2)
        budget.resumable(ledger,2,2)

    def test_resume_rejects_changed_budget_or_deadline(self):
        for key,value in [('max_updates',2000),('deadline_utc',8000)]:
            ledger=fixture_budget(0);ledger[key]=value
            with self.assertRaises(RuntimeError):budget.resumable(ledger,0,2)

    def test_evaluation_reservation_survives_interruption(self):
        ledger=fixture_budget(0)
        key=budget.reserve_evaluation(ledger,'selection-1',2026102302,'a'*64,1)
        with self.assertRaises(RuntimeError):budget.reserve_evaluation(ledger,'selection-1',2026102302,'b'*64,2)
        with self.assertRaises(RuntimeError):budget.resumable(ledger,0,2)
        budget.finish_evaluation(ledger,key,'f'*64);budget.resumable(ledger,1,2)

    def test_only_predeclared_ten_eval_slots(self):
        ledger=fixture_budget(0)
        for phase,seeds in budget.EVAL_SEEDS.items():
            for seed in seeds:
                key=budget.reserve_evaluation(ledger,phase,seed,'a'*64,1)
                budget.finish_evaluation(ledger,key,'f'*64)
        self.assertEqual(len(ledger['evaluations']),10)
        with self.assertRaises(RuntimeError):budget.reserve_evaluation(ledger,'heldout',20261021,'a'*64,1)


class AcceptanceTests(unittest.TestCase):
    def test_retention_without_position_return_or_geometric_stillness(self):
        self.assertTrue(criteria.accept_episode(success())['passed'])

    def test_any_joint_excursion_fails_even_when_retained(self):
        for field,value in [('limit_violation_samples',1),('initial_limit_violation',True),
                            ('peak_nominal_excursion_rad',.04)]:
            result=success();result['model_limits']['joints'][0][field]=value
            self.assertFalse(criteria.accept_episode(result)['passed'])

    def test_incomplete_500hz_sampling_fails(self):
        result=success();result['model_limits']['motor_checks_before_integration']=29999
        self.assertFalse(criteria.accept_episode(result)['passed'])

    def test_drop_error_wrench_or_mutated_normalizer_fails(self):
        result=success();result['baseline_trial']['physically_terminated']=True
        self.assertFalse(criteria.accept_episode(result)['passed'])
        for field,value in [('error','oops'),('policy_and_normalizer_unchanged',False),
                            ('applied_disturbance_pulses',1),('checkpoint_unchanged',False)]:
            result=success();result[field]=value
            self.assertFalse(criteria.accept_episode(result)['passed'])

    def test_duplicate_seeds_and_mixed_checkpoints_rejected(self):
        with self.assertRaises(RuntimeError):criteria.summarize([success(),success()],(2026102302,2026102303),'a'*64)
        with self.assertRaises(RuntimeError):criteria.summarize([success(2026102302),success(2026102303,'b'*64)],
                                                              (2026102302,2026102303),'a'*64)


class ClientTests(unittest.TestCase):
    def test_ssh_endpoint_comes_from_current_input_only(self):
        self.assertEqual(client.parse_endpoint('ssh -p 12345 root@example.test'),
                         dict(host='root@example.test',port=12345))
        self.assertEqual(client.parse_endpoint('ssh root@example.test -p 12345')['port'],12345)
        for command in ('ssh -p 0 root@x','ssh -p 12 root@x;uname',
                        'ssh -o ProxyCommand=foo -p 12 root@x','ssh -p 12 user@x'):
            with self.assertRaises((RuntimeError,ValueError)):client.parse_endpoint(command)

    def test_new_instance_cannot_silently_restart_approved_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(client,'LOCAL',Path(directory)):
                client.write(Path(directory)/'campaign.json',dict(started_utc=10,deadline_utc=7210))
                with self.assertRaises(RuntimeError):client.validate_same_campaign({'status':'NOT_INSTALLED'},dict(host='root@x',port=12))

    def test_uncertain_launch_cannot_move_to_other_host(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(client,'LOCAL',Path(directory)):
                client.write(Path(directory)/'launch-intent.json',dict(endpoint=dict(host='root@old',port=1)))
                with self.assertRaises(RuntimeError):client.validate_same_campaign({'status':'NOT_STARTED'},dict(host='root@new',port=2))

    def test_uncertain_launch_cannot_restart_on_same_host_without_ledger(self):
        with tempfile.TemporaryDirectory() as directory:
            endpoint=dict(host='root@old',port=1)
            with patch.object(client,'LOCAL',Path(directory)):
                client.write(Path(directory)/'launch-intent.json',dict(endpoint=endpoint))
                with self.assertRaises(RuntimeError):client.validate_same_campaign({'status':'NOT_STARTED'},endpoint)

    def test_status_and_budget_modules_do_not_import_torch(self):
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('mjlab',sys.modules)


class SupervisorTests(unittest.TestCase):
    def simulate(self,selection_pass_at,heldout_pass=True,break_smoke=False):
        with tempfile.TemporaryDirectory() as directory:
            slot=Path(directory)
            for rel in ('run/checkpoints','evaluations','logs'):(slot/rel).mkdir(parents=True,exist_ok=True)
            ledger=fixture_budget(10**12)
            common.write(slot/'budget.json',ledger)
            common.write(slot/'started.json',dict(started_utc=ledger['started_utc'],deadline_utc=ledger['deadline_utc']))
            common.write(slot/'session.json',dict(number=1))
            common.write(slot/'selection.json',dict(history=[],candidate=None))
            common.write(slot/'protected-records.json',{})
            calls=[];number=[0]
            def latest():return common.read(slot/'run/latest.json')
            def worker(kind,phase,*_):
                calls.append((kind,phase));number[0]+=1
                b=common.read(slot/'budget.json')
                if kind=='train':
                    remaining=budget.TARGETS[phase]-b['completed_updates']
                    for _ in range(remaining):budget.reserve_update(b,phase,10**12+1);budget.commit_update(b,phase)
                    common.write(slot/'budget.json',b)
                    cp=slot/'run/checkpoints'/('test-%s.pt'%b['completed_updates']);cp.write_bytes(str(b['completed_updates']).encode())
                    checkpoint=dict(path=str(cp),sha256=common.sha(cp),completed_updates=b['completed_updates'])
                    common.write(slot/'run/latest.json',checkpoint)
                    session=slot/'run'/('session-%03d'%number[0]);session.mkdir()
                    common.write(session/'result.json',dict(status='TRAINING_PHASE_COMPLETE',phase=phase,
                        completed_updates=b['completed_updates'],smoke_export=dict(passed=not break_smoke)))
                else:
                    folder=slot/'evaluations'/phase;cp=common.read(folder/'candidate.json');rows=[]
                    passed=(heldout_pass if phase=='heldout' else phase==selection_pass_at)
                    for seed in budget.EVAL_SEEDS[phase]:
                        key=budget.reserve_evaluation(b,phase,seed,cp['sha256'],10**12+1)
                        row=success(seed,cp['sha256'])
                        if not passed:row['model_limits']['joints'][0]['limit_violation_samples']=1
                        p=folder/str(seed)/'result.json';p.parent.mkdir();common.write(p,row)
                        budget.finish_evaluation(b,key,common.sha(p));rows.append(row)
                    common.write(slot/'budget.json',b)
                    common.write(folder/'evaluation-result.json',dict(status='EVALUATION_COMPLETE',checkpoint=cp,
                        summary=criteria.summarize(rows,budget.EVAL_SEEDS[phase],cp['sha256'])))
                return number[0]
            with (slot/'lease').open('a') as lock,patch.multiple(agent,SLOT=slot,
                verify_inputs=lambda:None,verify_protected=lambda _:True,
                snapshot=lambda:{},input_checkpoint=latest,run_worker=worker):
                agent.supervise(lock.fileno(),1)
            return calls,common.read(slot/'budget.json'),common.read(slot/'exit.json'),(slot/'selected.json').exists()

    def test_first_qualified_candidate_stops_at_506_and_uses_four_heldout(self):
        calls,b,exit,selected=self.simulate('selection-1')
        self.assertEqual(b['attempted_updates'],506)
        self.assertEqual(len(b['evaluations']),6)
        self.assertNotIn(('train','segment-2'),calls)
        self.assertTrue(exit['checked_baseline_goal_passed'] and selected)

    def test_second_candidate_is_bounded_at_1006_and_eight_episodes(self):
        calls,b,exit,selected=self.simulate('selection-2')
        self.assertEqual(b['attempted_updates'],1006)
        self.assertEqual(len(b['evaluations']),8)
        self.assertTrue(selected)

    def test_third_candidate_is_bounded_at_1505_and_ten_episodes(self):
        calls,b,exit,selected=self.simulate('selection-3')
        self.assertEqual((b['attempted_updates'],len(b['evaluations'])),(1505,10))
        self.assertTrue(selected and exit['checked_baseline_goal_passed'])

    def test_no_candidate_and_failed_heldout_do_not_add_training(self):
        calls,b,exit,selected=self.simulate('none')
        self.assertEqual((b['attempted_updates'],len(b['evaluations'])),(1505,6))
        self.assertFalse(selected)
        calls,b,exit,selected=self.simulate('selection-1',heldout_pass=False)
        self.assertEqual(b['attempted_updates'],506)
        self.assertFalse(selected or exit['checked_baseline_goal_passed'])
        self.assertEqual(calls[-1],('evaluate','heldout'))

    def test_smoke_failure_blocks_formal_training(self):
        calls,b,exit,selected=self.simulate('selection-1',break_smoke=True)
        self.assertEqual(calls,[('train','smoke')])
        self.assertEqual(b['attempted_updates'],6)
        self.assertEqual(exit['status'],'CAMPAIGN_ERROR_PRESERVE')


if __name__=='__main__':unittest.main(verbosity=2)
