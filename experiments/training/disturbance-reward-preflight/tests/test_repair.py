"""Known-failure installation and one bounded continuation, with no GPU/SSH."""
import contextlib
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'package'))
import repair
import agent
import common
import budget


class Child:
    pid=87654321
    def poll(self):return None


class RepairTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name)
        self.repo=self.base/'workspace';self.repo.mkdir()
        self.slot=self.base/'artifacts/final6h'
        shutil.copytree(ROOT/'reference/failed-start',self.slot)
        self.files={p.name:p.read_bytes() for p in (ROOT/'package').iterdir()
                    if p.is_file() and p.name!='manifest.json'}
        pins={name:dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()) for name,raw in self.files.items()}
        self.identity=hashlib.sha256(json.dumps(pins,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        self.files['manifest.json']=(json.dumps(dict(identity=self.identity,files=pins))+'\n').encode()
        self.request=dict(commit='a'*40,package_identity=self.identity)
        self.before={name:(self.slot/name).read_bytes() for name in ('budget.json','started.json')}
        self.contract=json.loads(self.files['repair-contract.json'])
        self.stack=contextlib.ExitStack()
        self.stack.enter_context(patch.object(repair,'alive',return_value=False))
        for name,value in [('BASE',self.base),('SLOT',self.slot),('HERE',self.slot/'package')]:
            self.stack.enter_context(patch.object(agent,name,value))
        self.stack.enter_context(patch.object(agent,'alive',return_value=False))
        self.stack.enter_context(patch.object(agent,'verify_package',return_value=self.identity))
        self.stack.enter_context(patch.object(agent,'preflight',return_value={'package_identity':self.identity}))
        self.stack.enter_context(patch.object(agent,'verify_protected',return_value=True))
        self.stack.enter_context(patch.object(agent.time,'time',return_value=self.contract['original_started_utc']+600))
    def tearDown(self):self.stack.close();self.tmp.cleanup()
    def install(self):return repair.apply_repair(self.repo,self.slot,self.files,self.request)
    def assert_budget_preserved(self):
        for name,raw in self.before.items():self.assertEqual((self.slot/name).read_bytes(),raw)
        self.assertTrue(budget.validate(common.read(self.slot/'budget.json')))

    def test_install_preserves_all_failed_evidence_and_budget(self):
        receipt=self.install();self.assert_budget_preserved()
        self.assertFalse(receipt['budget_renewed'])
        before=Path(receipt['evidence_directory'])
        for name,pin in self.contract['preserved_files'].items():
            self.assertTrue(repair.matches(before/name,pin),name)
        self.assertEqual(common.read(self.slot/'package/manifest.json')['identity'],self.identity)
        self.assertEqual(common.read(self.slot/'started.json')['package_identity'],self.contract['original_package_identity'])

    def test_install_is_idempotent_and_never_launches(self):
        with patch.object(agent.subprocess,'Popen') as launch:
            first=self.install();second=self.install()
        self.assertEqual(first,second);launch.assert_not_called();self.assert_budget_preserved()

    def test_final_post_transfer_supervisor_log_append_is_preserved(self):
        log=self.slot/'logs/supervisor.log';log.write_text('MD12_FINAL6H_STATE={"status":"CAMPAIGN_ERROR_PRESERVE"}\n')
        receipt=self.install()
        self.assertEqual(Path(receipt['completed_supervisor_log']['path']).read_bytes(),log.read_bytes())
        self.assertEqual((Path(receipt['evidence_directory'])/'logs/supervisor.log').read_bytes(),b'')

    def test_different_attempt_count_blocks_repair(self):
        ledger=common.read(self.slot/'budget.json');ledger['attempted_updates']=1
        common.write(self.slot/'budget.json',ledger)
        with self.assertRaisesRegex(RuntimeError,'failure changed'):self.install()
        self.assertFalse((self.slot/'repair-applied.json').exists())

    def test_live_transfer_blocks_code_replacement(self):
        common.write(self.slot/'transfer-worker.json',{'pid':123,'start':'x'})
        with patch.object(repair,'alive',side_effect=lambda pin:bool(pin and pin.get('pid')==123)):
            with self.assertRaisesRegex(RuntimeError,'still alive'):self.install()
        self.assert_budget_preserved()

    def test_exact_worker_error_is_in_status_report(self):
        state=agent.snapshot()
        self.assertEqual(state['worker_result']['error'],self.contract['allowed_error'])

    def test_continuation_spawns_once_with_old_clock_and_retains_old_session(self):
        self.install()
        old_result=(self.slot/'run/session-001/result.json').read_bytes()
        def spawn(cmd,**kw):
            self.assertTrue(kw['start_new_session'])
            self.assertTrue(kw['pass_fds'])
            self.assert_budget_preserved()
            self.assertEqual(common.read(self.slot/'repair-resume-started.json')['session_base'],3)
            self.assertFalse((self.slot/'exit.json').exists())
            return Child()
        with patch.object(agent.subprocess,'Popen',side_effect=spawn) as launch, \
             patch.object(agent,'process_pin',return_value={'pid':87654321,'start':'x','group':87654321}), \
             patch.object(agent,'alive',side_effect=lambda pin:bool(pin and pin.get('pid')==87654321)):
            first=agent.launch_repair();second=agent.launch_repair()
        self.assertEqual(launch.call_count,1)
        self.assertEqual(first['status'],'STARTING')
        self.assert_budget_preserved()
        self.assertEqual((self.slot/'run/session-001/result.json').read_bytes(),old_result)
        self.assertEqual(common.read(self.slot/'repair-resume-started.json')['original_deadline_utc'],1792800690.031889)

    def test_spawn_failure_does_not_renew_or_retry(self):
        self.install()
        with patch.object(agent.subprocess,'Popen',side_effect=OSError('fixture')) as launch:
            with self.assertRaises(OSError):agent.launch_repair()
            agent.launch_repair()
        self.assertEqual(launch.call_count,1);self.assert_budget_preserved()

    def test_expired_training_window_never_spawns(self):
        self.install()
        with patch.object(agent.time,'time',return_value=self.contract['original_train_stop_utc']), \
             patch.object(agent.subprocess,'Popen') as launch:
            with self.assertRaisesRegex(RuntimeError,'training window expired'):agent.launch_repair()
        launch.assert_not_called();self.assert_budget_preserved()

    def test_different_repair_identity_is_rejected(self):
        self.install();self.request['package_identity']='b'*64
        with self.assertRaisesRegex(RuntimeError,'receipt differs'):self.install()


if __name__=='__main__':unittest.main()
