"""Real CPU subprocess lifecycle and inference-only orchestration contracts."""
import ast
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'package'))
import common
import budget
import local_agent as agent
import evaluate
from test_evaluation import fixture,candidate
from eval_plan import case_plan
from eval_criteria import summarize_phase


class BudgetTests(unittest.TestCase):
    def test_no_ppo_and_no_extension(self):
        for field,value in [('attempted_updates',1),('completed_updates',1),('max_updates',1),
                            ('training_authorized',True),('automatic_retry',True),('deadline_utc',9999)]:
            state=budget.new(1.);state[field]=value
            with self.assertRaises(RuntimeError):budget.validate(state)

    def test_once_only_and_fixed_27_case_ceiling(self):
        state=budget.new(1.)
        for i in range(27):
            budget.reserve_evaluation(state,'selection',str(i),'sha',2.)
            budget.finish_evaluation(state,str(i),'result')
        with self.assertRaises(RuntimeError):budget.reserve_evaluation(state,'heldout','28','sha',3.)
        with self.assertRaises(RuntimeError):budget.finish_evaluation(state,'0','other')
        self.assertEqual(state['attempted_updates'],0)

    def test_deadline_and_duplicate_do_not_consume_an_extra_slot(self):
        state=budget.new(1.)
        budget.reserve_evaluation(state,'selection','one','sha',2.)
        with self.assertRaises(RuntimeError):budget.reserve_evaluation(state,'selection','one','sha',3.)
        with self.assertRaises(RuntimeError):budget.reserve_evaluation(state,'heldout','two','sha',4501.)
        self.assertEqual(len(state['evaluations']),1)


class FrozenInputTests(unittest.TestCase):
    def test_exact_local_index_format_is_the_verified_return_controllers_format(self):
        original=(ROOT/'reference/training-return-index.json').read_bytes()
        pins=common.read(ROOT/'package/return-pins.json')
        local=(json.dumps(json.loads(original),ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
        self.assertEqual(hashlib.sha256(original).hexdigest(),pins['index_sha256'])
        self.assertEqual(hashlib.sha256(local).hexdigest(),pins['local_index_sha256'])
        self.assertNotEqual(pins['local_index_sha256'],pins['index_sha256'])
        report=common.read(ROOT/'reference/training-report.json')
        self.assertEqual(report['local_return']['commit'],common.RETURN_COMMIT)
        self.assertEqual(report['local_return']['index_sha256'],pins['index_sha256'])
        self.assertEqual(report['local_return']['verified_files'],400)

    def test_candidate_translation_preserves_original_receipts_and_pins_all_three(self):
        pins=common.read(ROOT/'package/return-pins.json')['files']
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'run/candidates').mkdir(parents=True)
            originals={}
            for label,updates in common.CANDIDATES:
                relative='run/checkpoints/final6h_%06d.pt'%updates
                cp=dict(path=common.CLOUD_SLOT+'/'+relative,**pins[relative],
                    candidate_label=label,completed_updates=updates,policy_updates=6010+updates,
                    package_identity=common.TRAINING_PACKAGE_ID)
                path=root/'run/candidates'/(label+'.json');common.write(path,cp)
                originals[path]=path.read_bytes()
            with patch.object(common,'INPUTS',root):
                rows=common.candidate_inputs()
                for row,(label,updates) in zip(rows,common.CANDIDATES):
                    self.assertEqual(row['path'],str(root/('run/checkpoints/final6h_%06d.pt'%updates)))
                    self.assertEqual(row['policy_updates'],6010+updates)
                path=root/'run/candidates/candidate-final.json';bad=common.read(path)
                for key,value in [('sha256','wrong'),('completed_updates',5305),('path','/different/model.pt')]:
                    common.write(path,dict(bad,**{key:value}))
                    with self.assertRaises(RuntimeError):common.candidate_inputs()
                    path.write_bytes(originals[path])
            self.assertTrue(all(path.read_bytes()==data for path,data in originals.items()))

    def test_checkpoint_training_lineage_is_separate_from_new_evaluation_identity(self):
        cp=dict(policy_updates=11314,completed_updates=5304)
        meta=dict(package_identity=common.TRAINING_PACKAGE_ID,origin_checkpoint_sha256=common.MODEL_SHA,
            policy_updates=11314,campaign_completed_updates=5304,committed_snapshot=True,num_envs=4096,lower_assistance=0.)
        self.assertEqual(common.verify_checkpoint_metadata({'infos':{'stage12_final6h':meta}},cp),meta)
        for key,value in [('package_identity','evaluation-package'),('committed_snapshot',False),
                          ('campaign_completed_updates',5305),('num_envs',64),('lower_assistance',.1)]:
            with self.assertRaises(RuntimeError):
                common.verify_checkpoint_metadata({'infos':{'stage12_final6h':dict(meta,**{key:value})}},cp)

    def test_runtime_and_force_inputs_cannot_be_injected_through_environment(self):
        with patch.dict(os.environ,{'PYTHONPATH':'bad','PYTHONHOME':'bad','WORLD_SIZE':'8',
                                  'MICRODUCK_FAKE':'1','CUDA_VISIBLE_DEVICES':'8'}):
            env=common.clean_environment()
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'0')
        for key in ('PYTHONPATH','PYTHONHOME','WORLD_SIZE','MICRODUCK_FAKE'):self.assertNotIn(key,env)

    def test_symlinked_input_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'target').mkdir();(root/'link').symlink_to(root/'target')
            with self.assertRaises(RuntimeError):common.safe(root,'link/model.pt')


class RealWorkerLifecycle(unittest.TestCase):
    def run_child(self,body,deadline_after=.2,grace=1.):
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
            root=Path(tmp);(root/'logs').mkdir();(root/'package').mkdir()
            cp=root/'cp.json';common.write(cp,{'sha256':'fixture'})
            folder=root/'evaluation';folder.mkdir()
            source='''import argparse,json,signal,time,os
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--phase');p.add_argument('--checkpoint-json');p.add_argument('--run-dir')
a=p.parse_args();f=Path(a.run_dir)/'evaluation-result.json'
value=dict(phase=a.phase,checkpoint={'sha256':'fixture'},status='EVALUATION_COMPLETE',finished_utc=time.time())
'''+body
            (root/'package/evaluate.py').write_text(source)
            for name,value in [('SLOT',root),('PROJECT',root),('HERE',root/'package'),
                               ('PYTHON',Path(sys.executable)),('CLEANUP_SECONDS',grace)]:
                stack.enter_context(patch.object(agent,name,value))
            # The isolated build host virtualizes getpid()/Popen.pid but exposes
            # host /proc IDs. Use a fixture identity reader for these real CPU
            # subprocess/signal tests; production retains its Linux /proc guard.
            # wait(), return codes, process groups and signals remain real.
            def fixture_identity(pid):
                try:return dict(pid=pid,start='fixture-'+str(pid),state='S',group=os.getpgid(pid))
                except ProcessLookupError:return None
            stack.enter_context(patch.object(agent,'identity',side_effect=fixture_identity))
            with tempfile.TemporaryFile() as lease:
                outcome=agent.run_worker('selection',cp,folder,1,lease.fileno(),time.time()+deadline_after,[False])
            record=common.read(root/'logs/001-finished.json')
            self.assertEqual(record,outcome);self.assertFalse(agent.alive(agent.optional('child.json')))
            return outcome

    def test_pid_reuse_or_group_mismatch_never_signals_another_process(self):
        from unittest.mock import Mock
        child=Mock(pid=42);child.poll.return_value=None
        pin=dict(pid=42,start='first',group=42)
        for current in (dict(pid=42,start='second',group=42),dict(pid=42,start='first',group=43),None):
            with patch.object(agent,'identity',return_value=current),patch.object(agent.os,'killpg') as kill:
                with self.assertRaises(RuntimeError):agent.signal_owned(child,pin,signal.SIGTERM)
                kill.assert_not_called()

    def test_saved_result_allows_normal_teardown_past_deadline_without_term(self):
        outcome=self.run_child("f.write_text(json.dumps(value));time.sleep(.45)\n")
        self.assertEqual(outcome['worker_exit_code'],0);self.assertEqual(outcome['signals'],[])
        self.assertTrue(outcome['result_saved'])

    def test_nonzero_exit_is_recorded_even_with_a_complete_result(self):
        outcome=self.run_child("f.write_text(json.dumps(value));raise SystemExit(7)\n")
        self.assertEqual(outcome['worker_exit_code'],7)
        self.assertFalse(agent.phase_completed(outcome,{'status':'EVALUATION_COMPLETE'}))

    def test_deadline_interrupt_without_result_cannot_be_accepted(self):
        outcome=self.run_child('time.sleep(30)\n')
        self.assertEqual(outcome['worker_exit_code'],-signal.SIGTERM)
        self.assertEqual(outcome['signals'],['SIGTERM']);self.assertFalse(outcome['result_saved'])

    def test_unresponsive_child_has_bounded_cleanup_and_recorded_kill(self):
        outcome=self.run_child('signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(30)\n',grace=.08)
        self.assertEqual(outcome['worker_exit_code'],-signal.SIGKILL)
        self.assertEqual(outcome['signals'],['SIGTERM','SIGKILL'])
        self.assertEqual(outcome['stop_reason'],'cleanup_grace_exceeded')

    def test_completed_result_with_hung_cleanup_is_not_reported_as_success(self):
        outcome=self.run_child('f.write_text(json.dumps(value));time.sleep(30)\n',grace=.08)
        self.assertEqual(outcome['worker_exit_code'],-signal.SIGKILL)
        self.assertFalse(agent.phase_completed(outcome,{'status':'EVALUATION_COMPLETE'}))

    def test_exit_status_stop_reason_and_phase_status_all_matter(self):
        self.assertTrue(agent.phase_completed({'worker_exit_code':0,'stop_reason':'evaluation_deadline'},
                                             {'status':'EVALUATION_COMPLETE'}))
        for code,reason,status in [(0,'user_stop','EVALUATION_COMPLETE'),(-15,None,'EVALUATION_COMPLETE'),
                                   (0,None,'EVALUATION_INCOMPLETE_PRESERVE')]:
            self.assertFalse(agent.phase_completed({'worker_exit_code':code,'stop_reason':reason},{'status':status}))


class LocalOrchestration(unittest.TestCase):
    def execute(self,mode='pass'):
        with tempfile.TemporaryDirectory() as tmp,contextlib.ExitStack() as stack:
            root=Path(tmp);(root/'candidates').mkdir();(root/'evaluations').mkdir();(root/'logs').mkdir()
            stack.enter_context(patch.object(agent,'SLOT',root))
            stack.enter_context(patch.object(evaluate,'SLOT',root))
            for mod in (agent,evaluate):stack.enter_context(patch.object(mod,'verify_package',return_value='fixture'))
            stack.enter_context(patch.object(agent,'verify_inputs',return_value={'inputs':'unchanged'}))
            stack.enter_context(patch.object(evaluate,'validate_checkpoint'))
            stack.enter_context(patch.object(evaluate,'load_runtime',return_value=None))
            paths=[]
            for i in range(3):
                cp=candidate('candidate-%d'%i,updates=i+1)['checkpoint']
                path=root/'candidates'/('%d.json'%i);common.write(path,cp);paths.append(str(path))
            now=time.time();ledger=budget.new(now)
            common.write(root/'budget.json',ledger)
            common.write(root/'started.json',dict(started_utc=now,deadline_utc=ledger['deadline_utc'],
                candidates=paths,package_identity='fixture'))
            common.write(root/'preflight.json',{'inputs':{'inputs':'unchanged'}})
            phases=[]
            def one(context,run,case,checkpoint,package_id,boundary,progress):
                boundary();row=fixture(case,checkpoint)
                if mode=='no-qualified' or (mode=='heldout-failed' and case['force_seed']==2026102800):
                    row['trial'].update(physically_terminated=True,terminations=['top_ball_lost'])
                    row['acceptance']=evaluate.accept_case(row,case,checkpoint['sha256'])
                return row
            def worker(phase,receipt,folder,number,fd,deadline,stopped):
                phases.append((phase,common.read(receipt)['sha256']))
                with patch.object(sys,'argv',['evaluate','--phase',phase,'--checkpoint-json',str(receipt),'--run-dir',str(folder)]):
                    code=evaluate.main()
                return dict(worker_exit_code=code,stop_reason=None)
            stack.enter_context(patch.object(evaluate,'evaluate_one',side_effect=one))
            stack.enter_context(patch.object(agent,'run_worker',side_effect=worker))
            handlers={sig:signal.getsignal(sig) for sig in (signal.SIGINT,signal.SIGTERM)}
            try:
                with tempfile.TemporaryFile() as lease:report=agent.supervise(lease.fileno())
            finally:
                for sig,handler in handlers.items():signal.signal(sig,handler)
            self.assertEqual(report,common.read(root/'report.json'))
            return report,phases

    def test_all_27_cases_and_only_one_locked_heldout_candidate(self):
        report,phases=self.execute()
        self.assertEqual(report['status'],'VALIDATION_PASSED')
        self.assertEqual(report['evaluated_cases'],27);self.assertEqual(len(phases),4)
        self.assertEqual(phases[-1],('heldout',report['chosen_checkpoint']['sha256']))
        self.assertEqual(report['heldout']['summary']['push_passes'],16)
        self.assertEqual(report['heldout']['summary']['quiet_passes'],2)
        self.assertEqual(report['added_PPO_updates'],0);self.assertFalse(report['all_possible_sequences_proven'])

    def test_no_qualified_candidate_cannot_enter_heldout(self):
        report,phases=self.execute('no-qualified')
        self.assertEqual(report['status'],'NO_QUALIFIED_CANDIDATE')
        self.assertEqual(report['evaluated_cases'],9);self.assertEqual(len(phases),3)
        self.assertFalse(report['disturbance_goal_passed']);self.assertIsNone(report['heldout'])

    def test_heldout_failure_never_reselects_or_replaces_sequence(self):
        report,phases=self.execute('heldout-failed')
        self.assertEqual(report['status'],'HELDOUT_FAILED')
        self.assertEqual(report['evaluated_cases'],12);self.assertEqual(len(phases),4)
        self.assertEqual(report['heldout']['summary']['evaluated_cases'],3)
        self.assertFalse(report['disturbance_goal_passed'])

    def test_repeated_run_is_readonly_and_orphan_budget_never_resets(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(agent,'preflight') as preflight:
            root=Path(tmp);slot=root/'slot';slot.mkdir()
            with patch.object(agent,'SLOT',slot),patch.object(agent,'BASE',root):
                common.write(slot/'started.json',{'original':True})
                raw=(slot/'started.json').read_bytes();agent.launch()
                self.assertEqual((slot/'started.json').read_bytes(),raw);preflight.assert_not_called()
                (slot/'started.json').unlink();common.write(slot/'budget.json',{'orphan':True})
                with self.assertRaises(RuntimeError):agent.launch()
                self.assertEqual(common.read(slot/'budget.json'),{'orphan':True});preflight.assert_not_called()

    def test_all_runtime_imports_resolve_and_no_training_or_remote_execution_path(self):
        package=ROOT/'package'
        self.assertFalse((package/'train.py').exists())
        for path in package.glob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                    self.assertNotIn(node.func.attr,('learn','backward','collect_rollouts'),path.name)
            self.assertNotIn('ssh -p',path.read_text())
            self.assertNotIn('shutdown -',path.read_text())


if __name__=='__main__':unittest.main(verbosity=2)
