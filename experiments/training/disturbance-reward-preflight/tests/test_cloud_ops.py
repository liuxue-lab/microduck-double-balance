"""Offline cloud lifecycle, transport and monitoring tests; no GPU or SSH calls."""
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'package'))
import agent
import budget
import common


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


controller=load('final6h_test_controller',ROOT/'controller.py')
bootstrap=load('final6h_test_bootstrap',ROOT/'bootstrap.py')


class Child:
    pid=1234
    returncode=0
    def poll(self):return self.returncode
    def wait(self,timeout=None):return self.returncode


class Fixture(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.base=Path(self.temp.name)
        self.slot=self.base/'artifacts/final6h';self.slot.mkdir(parents=True)
        self.stack=contextlib.ExitStack()
        for name,value in [('BASE',self.base),('SLOT',self.slot),('HERE',self.slot/'package')]:
            self.stack.enter_context(patch.object(agent,name,value))
        self.stack.enter_context(patch.object(agent,'verify_package',return_value='package'))
        self.stack.enter_context(patch.object(agent,'alive',return_value=False))
        self.stack.enter_context(patch.object(agent,'preflight',return_value={
            'package_identity':'package','protected_records':{}}))
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
    def tearDown(self):self.stack.close();self.temp.cleanup()
    def reserve(self):
        now=time.time();ledger=budget.new(now)
        for p in ('run/checkpoints','run/candidates','logs','evaluations'): (self.slot/p).mkdir(parents=True)
        common.write(self.slot/'budget.json',ledger)
        common.write(self.slot/'started.json',dict(started_utc=now,deadline_utc=now+21600,
                     train_stop_utc=now+common.TRAIN_SECONDS,package_identity='package'))
        common.write(self.slot/'protected-records.json',{})


class CloudLifecycle(Fixture):
    def test_launch_reserves_before_detached_spawn_and_inherits_lock(self):
        def spawn(cmd,**kwargs):
            started=common.read(self.slot/'started.json');ledger=common.read(self.slot/'budget.json')
            self.assertEqual(started['deadline_utc']-started['started_utc'],21600)
            self.assertEqual(ledger['completed_updates'],0)
            self.assertTrue(kwargs['start_new_session']);self.assertEqual(kwargs['stdin'],agent.subprocess.DEVNULL)
            self.assertEqual(len(kwargs['pass_fds']),1);os.fstat(kwargs['pass_fds'][0])
            return Child()
        with patch.object(agent.subprocess,'Popen',side_effect=spawn), \
             patch.object(agent,'process_pin',return_value={'pid':1234,'start':'1','group':1234}):
            result=agent.launch()
        self.assertEqual(result['status'],'STARTING')
        self.assertFalse(result['started']['automatic_budget_renewal'])

    def test_repeat_start_never_runs_preflight_or_spawn(self):
        self.reserve()
        with patch.object(agent,'preflight') as preflight,patch.object(agent.subprocess,'Popen') as spawn:
            agent.launch();preflight.assert_not_called();spawn.assert_not_called()

    def test_preflight_failure_does_not_consume_budget(self):
        with patch.object(agent,'preflight',side_effect=RuntimeError('model differs')):
            with self.assertRaisesRegex(RuntimeError,'model differs'):agent.launch()
        self.assertFalse((self.slot/'started.json').exists())

    def test_spawn_failure_consumes_start_no_automatic_retry(self):
        with patch.object(agent.subprocess,'Popen',side_effect=OSError('fixture')):
            with self.assertRaises(OSError):agent.launch()
        self.assertTrue((self.slot/'started.json').exists())
        with patch.object(agent.subprocess,'Popen') as spawn:agent.launch();spawn.assert_not_called()

    def test_snapshot_is_readonly_and_distinguishes_return_from_compute(self):
        self.reserve();common.write(self.slot/'exit.json',dict(status='CAMPAIGN_COMPLETE'))
        common.write(self.slot/'supervisor.json',{'pid':1234})
        with patch.object(agent,'alive',side_effect=lambda pin:bool(pin)):
            state=agent.snapshot()
        self.assertEqual(state['status'],'COMPUTE_COMPLETE_RETURNING')
        self.assertFalse(state['compute_active'])

    def test_rejects_orphan_ledger_without_reset(self):
        common.write(self.slot/'budget.json',{'old':True})
        with self.assertRaisesRegex(RuntimeError,'Orphan'):agent.launch()
        self.assertEqual(common.read(self.slot/'budget.json'),{'old':True})

    def test_candidate_hash_dedup_and_lineage(self):
        self.reserve();model=self.slot/'run/checkpoints/push10_000020.pt';model.write_bytes(b'model')
        receipt=dict(path=str(model),sha256=common.sha(model),bytes=5,completed_updates=20,policy_updates=6030)
        for n in ('candidate-01.json','candidate-final.json'):common.write(self.slot/'run/candidates'/n,receipt)
        self.assertEqual(len(agent.candidate_receipts()),1)
        model.write_bytes(b'changed')
        with self.assertRaisesRegex(RuntimeError,'changed'):agent.candidate_receipts()

    def test_deadline_term_then_bounded_kill_never_renews_budget(self):
        self.reserve();clock=[0.];sent=[]
        class PendingChild(Child):
            returncode=None
            def wait(self,timeout=None):
                if self.returncode is not None:return self.returncode
                clock[0]+=50.
                raise agent.subprocess.TimeoutExpired('fixture',timeout)
        child=PendingChild()
        def signal_child(proc,pin,sig):
            sent.append(sig)
            if sig==agent.signal.SIGKILL:proc.returncode=-9
        before=(self.slot/'budget.json').read_bytes()
        with (self.base/'lock').open('w') as lease, \
            patch.object(agent.subprocess,'Popen',return_value=child), \
            patch.object(agent,'process_pin',return_value={'pid':1234,'start':'1','group':1234}), \
            patch.object(agent,'signal_owned',side_effect=signal_child), \
            patch.object(agent.time,'time',return_value=20.), \
            patch.object(agent.time,'monotonic',side_effect=lambda:clock[0]):
                result=agent.run_worker('train',[],1,lease.fileno(),10.,[False])
        self.assertEqual(sent,[agent.signal.SIGTERM,agent.signal.SIGKILL])
        self.assertEqual(result['stop_reason'],'compute_deadline')
        self.assertEqual((self.slot/'budget.json').read_bytes(),before)

    def test_process_identity_mismatch_never_signals_reused_pid(self):
        child=Child();child.returncode=None
        pin={'pid':1234,'start':'old','group':1234}
        with patch.object(agent,'identity',return_value={'pid':1234,'start':'new','group':1234}), \
             patch.object(agent.os,'killpg') as kill:
            with self.assertRaisesRegex(RuntimeError,'identity'):agent.signal_owned(child,pin,agent.signal.SIGTERM)
        kill.assert_not_called()

    def test_supervisor_sequence_runs_all_candidates_then_one_heldout(self):
        self.run_supervisor_fixture(True,False)

    def test_no_qualified_candidate_skips_heldout_and_does_not_retrain(self):
        self.run_supervisor_fixture(False,False)

    def test_repaired_supervisor_uses_new_sessions_and_original_deadline(self):
        self.run_supervisor_fixture(False,False,repaired=True)

    def run_supervisor_fixture(self,qualified,heldout_passed,repaired=False):
        self.reserve();calls=[];receipt={};results=[]
        if repaired:common.write(self.slot/'repair-resume-started.json',{'session_base':3})
        def worker(kind,args,number,fd,deadline,stopped):
            calls.append((kind,list(args)));phase=args[args.index('--phase')+1]
            if kind=='train':
                self.assertEqual(deadline,common.read(self.slot/'started.json')['train_stop_utc'])
                folder=self.slot/'run'/('session-%03d'%number);folder.mkdir()
                ledger=common.read(self.slot/'budget.json');ledger['completed_updates']=5 if phase=='smoke' else 20
                common.write(self.slot/'budget.json',ledger)
                common.write(folder/'result.json',dict(status='TRAINING_PHASE_COMPLETE',phase=phase,smoke_export={'passed':True}))
                if phase=='formal':
                    for i,name in enumerate(('candidate-01','candidate-02','candidate-final'),1):
                        model=self.slot/'run/checkpoints'/('model-%s.pt'%i);model.write_bytes(bytes([i]))
                        value=dict(path=str(model),sha256=common.sha(model),bytes=1,completed_updates=5+i,policy_updates=6015+i)
                        common.write(self.slot/'run/candidates'/(name+'.json'),value)
                    receipt.update(value)
            else:
                folder=Path(args[args.index('--run-dir')+1]);folder.mkdir()
                cp=common.read(args[args.index('--checkpoint-json')+1])
                value=dict(status='EVALUATION_COMPLETE',phase=phase,checkpoint=cp,cases=[],
                           summary={'all_passed':heldout_passed if phase=='heldout' else qualified})
                common.write(folder/'evaluation-result.json',value);results.append(value)
            return dict(worker_exit_code=0,stop_reason=None,worker_alive=False)
        import eval_plan,eval_criteria
        def chosen(values):
            self.assertEqual(len(values),3)
            return dict(qualified=qualified,checkpoint=receipt,selection=values[-1])
        with (self.base/'lock').open('w') as lease,patch.object(agent,'run_worker',side_effect=worker), \
            patch.object(agent,'verify_inputs'),patch.object(agent,'verify_protected'), \
            patch.object(eval_plan,'choose_candidate',side_effect=chosen), \
            patch.object(eval_criteria,'summarize_phase',side_effect=lambda rows,phase,sha:{'all_passed':heldout_passed if phase=='heldout' else qualified}), \
            patch.object(agent.subprocess,'Popen',return_value=Child()), \
            patch.object(agent,'process_pin',return_value={'pid':1234,'start':'1','group':1234}):
                agent.supervise(lease.fileno())
        self.assertEqual([kind for kind,args in calls].count('train'),2)
        if repaired:
            self.assertEqual([args[args.index('--session')+1] for kind,args in calls if kind=='train'],['3','4'])
        self.assertEqual([kind for kind,args in calls].count('evaluate'),4 if qualified else 3)
        ended=common.read(self.slot/'exit.json');self.assertFalse(ended['disturbance_goal_passed'])
        self.assertFalse(ended['automatic_training_extension'])
        self.assertTrue((self.slot/'transfer-outcome.json').exists())


class ControllerAndTransport(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.path=Path(self.temp.name)
        self.stack=contextlib.ExitStack();self.stack.enter_context(patch.object(controller,'LOCAL',self.path))
        controller.RELEASE_DATA=dict(commit='a'*40,package_identity='b'*64,
            ssh=base64.b85encode(zlib.compress(b'HOST=None\nPORT=None\n')).decode())
    def tearDown(self):self.stack.close();self.temp.cleanup()

    def test_current_endpoint_default_and_explicit_override(self):
        _,endpoint=controller.initialize_ssh(None);self.assertEqual(endpoint['port'],32185)
        _,endpoint=controller.initialize_ssh('ssh -p 1234 root@example.com');self.assertEqual(endpoint['port'],1234)
        with self.assertRaises(RuntimeError):controller.parse_endpoint('ssh -p 1 root@example.com echo bad')

    def test_existing_endpoint_only_from_new_slot(self):
        controller.write(self.path/'endpoint.json',{'host':'root@new.example','port':2222})
        _,endpoint=controller.initialize_ssh(None);self.assertEqual(endpoint['port'],2222)

    def test_uncertain_launch_cannot_switch_instances(self):
        controller.write(self.path/'launch-intent.json',{'endpoint':{'host':'root@x','port':1}})
        with self.assertRaisesRegex(RuntimeError,'不能换机'):
            controller.validate_same_campaign({'status':'NOT_INSTALLED'},{'host':'root@y','port':2})

    def test_watch_ready_only_after_formal_update_and_never_stops(self):
        first=dict(status='TRAINING',progress={'phase':'formal','status':'TRAINING'},budget={'completed_updates':6},
                   processes_alive={'child':True,'supervisor':True})
        last=dict(status='CAMPAIGN_COMPLETE',progress={},budget={},exit={'disturbance_goal_passed':False},
                  processes_alive={'child':False,'supervisor':False})
        output=io.StringIO()
        with patch.object(controller,'rpc',return_value=last) as rpc,patch.object(controller.time,'sleep'), \
             patch.object(controller,'save_report'),contextlib.redirect_stdout(output):
            controller.watch(None,first)
        self.assertIn('READY_TO_LEAVE',output.getvalue());rpc.assert_called_once_with(None,'status',timeout=60)

    def test_watch_interrupt_does_not_send_cloud_stop(self):
        with patch.object(controller.time,'sleep',side_effect=KeyboardInterrupt), \
             patch.object(controller,'rpc') as rpc,contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):controller.watch(None,{'status':'TRAINING','progress':{},'budget':{}})
        rpc.assert_not_called()

    def test_source_payload_hash_and_member_validation(self):
        files={'worker.py':b'# source\n'}
        pins={n:dict(bytes=len(v),sha256=hashlib.sha256(v).hexdigest()) for n,v in files.items()}
        identity=hashlib.sha256(json.dumps(pins,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        files['manifest.json']=json.dumps(dict(identity=identity,files=pins)).encode()
        self.assertEqual(bootstrap.verify_files(files,identity),files)
        bad=dict(files,**{'worker.py':b'changed'})
        with self.assertRaisesRegex(RuntimeError,'hash'):bootstrap.verify_files(bad,identity)

    def test_repaired_client_accepts_original_start_identity_but_rejects_new_clock(self):
        started=dict(schema='stage12-push10-final6h-v1',package_identity=controller.ORIGINAL_PACKAGE_ID,
            started_utc=controller.ORIGINAL_STARTED_UTC,deadline_utc=controller.ORIGINAL_DEADLINE_UTC,
            train_stop_utc=controller.ORIGINAL_TRAIN_STOP_UTC,max_updates=controller.MAX_UPDATES,
            parent_checkpoint_sha256=controller.PARENT_MODEL_SHA)
        state=dict(started=started,package_identity='b'*64,repair={'package_identity':'b'*64})
        controller.validate_same_campaign(state,{'host':'root@cloud','port':32185})
        started['started_utc']+=600;started['train_stop_utc']+=600;started['deadline_utc']+=600
        with self.assertRaisesRegex(RuntimeError,'预算不同'):
            controller.validate_same_campaign(state,{'host':'root@cloud','port':32185})


if __name__=='__main__':unittest.main(verbosity=2)
