"""Repair checks against the actual returned failure; no GPU or physical steps."""
import base64
import contextlib
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch,Mock
import zlib

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'package'))
import repair
import dependency_check
import agent
import transfer


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


boot=load(ROOT/'bootstrap.py','repair_bootstrap_test')


def package_files():return {p.name:p.read_bytes() for p in (ROOT/'package').iterdir() if p.is_file()}


def source(files,request):
    doc=dict(commit=request['commit'],package_identity=request['package_identity'],source_path=boot.CODE,
             files={n:base64.b64encode(raw).decode() for n,raw in files.items()})
    raw=json.dumps(doc,sort_keys=True,separators=(',',':')).encode()
    return dict(encoding='zlib-base85-json-v1',raw_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                data=base64.b85encode(zlib.compress(raw)).decode())


class RepairTests(unittest.TestCase):
    def setUp(self):
        t=tempfile.TemporaryDirectory();self.addCleanup(t.cleanup);self.root=Path(t.name)
        self.base=self.root/'cloud';self.repo=self.base/'workspace';self.repo.mkdir(parents=True)
        self.slot=self.base/'artifacts/double-balance-stage12/native-headroom-refine-v1'
        shutil.copytree(ROOT/'evidence/return-4823151/unpacked',self.slot)
        (self.slot/'run/checkpoints').mkdir();(self.slot/'evaluations').mkdir()
        self.files=package_files();self.package_id=json.loads(self.files['manifest.json'])['identity']
        self.contract=json.loads(self.files['repair-contract.json'])
        self.request=dict(operation='repair',commit='f'*40,package_identity=self.package_id)
        self.original={n:(self.slot/n).read_bytes() for n in self.contract['preserved_files']}
        self.old_budget=(self.slot/'budget.json').read_bytes();self.old_started=(self.slot/'started.json').read_bytes()
        self.receipt_root=self.slot/'repairs/missing-reward-spec-v2'

    def apply(self):return repair.apply_repair(self.repo,self.slot,self.files,self.request)

    def test_real_failure_repair_preserves_all_44_returned_files_and_budget(self):
        receipt=self.apply()
        self.assertEqual(receipt['status'],'REPAIR_APPLIED')
        for n,raw in self.original.items():self.assertEqual((self.receipt_root/'before'/n).read_bytes(),raw)
        self.assertEqual((self.slot/'budget.json').read_bytes(),self.old_budget)
        self.assertEqual((self.slot/'started.json').read_bytes(),self.old_started)
        self.assertEqual((self.slot/'package/reward_spec.py').read_bytes(),self.files['reward_spec.py'])
        self.assertEqual((self.receipt_root/'retired-package/speed_task.py').read_bytes(),self.files['speed_task.py'])
        self.assertFalse(receipt['deadline_extended']);self.assertFalse(receipt['automatic_training'])

    def test_idempotent_repair_does_not_change_installed_files_or_receipt(self):
        self.apply();paths=[self.slot/'repair-applied.json',*(self.slot/'package').iterdir()]
        before={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in paths}
        self.apply();self.assertEqual(before,{str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in paths})

    def test_actual_failure_is_resumeable_only_before_original_deadline(self):
        self.apply();repair.validate_resume(self.slot,self.package_id,self.contract['original_started_utc']+100)
        with self.assertRaisesRegex(RuntimeError,'deadline expired'):
            repair.validate_resume(self.slot,self.package_id,self.contract['original_deadline_utc'])
        self.assertEqual((self.slot/'budget.json').read_bytes(),self.old_budget)

    def test_spent_or_pending_budget_refuses_repair_without_any_reset(self):
        budget=json.loads(self.old_budget);budget['attempted_updates']=1;budget['pending']={'attempt':1}
        path=self.slot/'budget.json';path.write_text(json.dumps(budget));raw=path.read_bytes()
        with self.assertRaisesRegex(RuntimeError,'Original failure changed'):self.apply()
        self.assertEqual(path.read_bytes(),raw);self.assertFalse((self.slot/'repair-applied.json').exists())

    def test_unrelated_error_refuses_repair(self):
        path=self.slot/'run/session-001/result.json';value=json.loads(path.read_bytes());value['error']='Different runtime failure'
        path.write_text(json.dumps(value))
        with self.assertRaisesRegex(RuntimeError,'Original failure changed'):self.apply()
        self.assertFalse((self.slot/'package/reward_spec.py').exists())

    def test_changed_old_source_evidence_stops_before_install(self):
        path=self.slot/'package/train.py';path.write_text('preserve user change')
        with self.assertRaisesRegex(RuntimeError,'Original evidence differs'):self.apply()
        self.assertEqual(path.read_text(),'preserve user change')

    def test_live_worker_is_never_replaced(self):
        with patch.object(repair,'alive',return_value=True):
            with self.assertRaisesRegex(RuntimeError,'still alive'):self.apply()
        self.assertFalse((self.slot/'repair-applied.json').exists())

    def test_gpu_lease_prevents_concurrent_install(self):
        lockpath=self.base/'artifacts/microduck-headroom-gpu.lock'
        with lockpath.open('a') as f:
            fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError,'GPU lease'):self.apply()
        self.assertFalse((self.slot/'repair-applied.json').exists())

    def test_interrupted_directory_swap_can_finish_from_verified_backup(self):
        rename=os.rename;calls=[]
        def interrupt(src,dst):
            calls.append((src,dst))
            if len(calls)==2:raise OSError('fixture interrupted between directory swaps')
            return rename(src,dst)
        with patch.object(repair.os,'rename',side_effect=interrupt):
            with self.assertRaisesRegex(OSError,'fixture interrupted'):self.apply()
        self.assertFalse((self.slot/'package').exists());self.assertTrue((self.receipt_root/'retired-package').exists())
        self.apply();repair.validate_resume(self.slot,self.package_id,self.contract['original_started_utc']+100)
        self.assertEqual((self.slot/'budget.json').read_bytes(),self.old_budget)

    def test_symlink_evidence_is_not_followed(self):
        path=self.slot/'package/train.py';outside=self.root/'outside';outside.write_bytes(path.read_bytes())
        path.unlink();path.symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError,'Symlink'):self.apply()
        self.assertEqual(outside.read_bytes(),self.original['package/train.py'])

    def test_cloud_bootstrap_preserves_config_and_calls_repaired_resume(self):
        subprocess.run(['git','init',str(self.repo)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        subprocess.run(['git','-C',str(self.repo),'remote','add','origin','https://example.invalid/upstream.git'],check=True)
        config=(self.repo/'.git/config').read_bytes();python=self.repo/'.venv/bin/python';python.parent.mkdir(parents=True);python.touch()
        request=dict(self.request,source=source(self.files,self.request))
        with patch.object(boot,'BASE',self.base),patch.object(sys,'stdin',io.StringIO(json.dumps(request))), \
             patch.object(boot.os,'execv') as execute,contextlib.redirect_stdout(io.StringIO()) as output:
            boot.main()
        self.assertEqual(execute.call_args.args[1][-1],'resume-repaired')
        self.assertIn('budget/deadline unchanged',output.getvalue())
        self.assertEqual((self.repo/'.git/config').read_bytes(),config)

    def agent_patches(self,now):
        stack=contextlib.ExitStack()
        for name,value in (('BASE',self.base),('SLOT',self.slot),('HERE',self.slot/'package')):
            stack.enter_context(patch.object(agent,name,value))
        stack.enter_context(patch.object(agent,'preflight',return_value=dict(package_identity=self.package_id,dependencies={'physics_steps':0})))
        stack.enter_context(patch.object(agent,'verify_package',return_value=self.package_id))
        stack.enter_context(patch.object(agent,'verify_protected',return_value=True))
        stack.enter_context(patch.object(agent,'input_checkpoint',return_value={'completed_updates':0}))
        stack.enter_context(patch.object(agent,'alive',return_value=False))
        stack.enter_context(patch.object(agent,'identity',return_value=dict(pid=987654,start='fixture',state='S',group=987654)))
        stack.enter_context(patch.object(agent.time,'time',return_value=now))
        stack.enter_context(patch.object(agent.account,'new',side_effect=AssertionError('Must never create a new budget')))
        child=Mock(pid=987654);popen=stack.enter_context(patch.object(agent.subprocess,'Popen',return_value=child))
        return stack,popen

    def test_agent_explicit_repair_launches_once_in_same_campaign(self):
        self.apply()
        stack,popen=self.agent_patches(self.contract['original_started_utc']+100)
        with stack:
            state=agent.launch(resume=True,repair=True);again=agent.launch(resume=True,repair=True)
            self.assertEqual(popen.call_count,1);self.assertEqual(state['session']['number'],2)
            self.assertEqual(again['started']['deadline_utc'],self.contract['original_deadline_utc'])
        self.assertEqual((self.slot/'budget.json').read_bytes(),self.old_budget)
        self.assertEqual((self.slot/'started.json').read_bytes(),self.old_started)
        self.assertEqual(json.loads((self.slot/'worker-counter.json').read_bytes())['number'],1)
        self.assertEqual((self.slot/'exit-session-001.json').read_bytes(),self.original['exit-session-001.json'])

    def test_ordinary_resume_cannot_bypass_known_error_guard(self):
        self.apply();stack,popen=self.agent_patches(self.contract['original_started_utc']+100)
        with stack:
            with self.assertRaisesRegex(RuntimeError,'explicit audited repair'):agent.launch(resume=True)
            popen.assert_not_called()

    def test_expired_repair_cannot_run_import_checks_or_spawn(self):
        self.apply();stack,popen=self.agent_patches(self.contract['original_deadline_utc']+1)
        with stack:
            with self.assertRaisesRegex(RuntimeError,'deadline expired'):agent.launch(resume=True,repair=True)
            agent.preflight.assert_not_called();popen.assert_not_called()


class DependencyTests(unittest.TestCase):
    def test_original_missing_dependency_is_detected_without_torch(self):
        with self.assertRaisesRegex(RuntimeError,'speed_task.py -> reward_spec'):
            dependency_check.check_local_closure(ROOT/'evidence/return-4823151/unpacked/package')

    def test_complete_new_training_and_evaluation_import_closure(self):
        result=dependency_check.check_local_closure(ROOT/'package')
        self.assertIn(['speed_task.py','reward_spec'],result['local_imports'])
        self.assertEqual(result['physics_steps'],0);self.assertEqual(result['PPO_updates'],0)
        for path in (ROOT/'package').glob('*.py'):compile(path.read_bytes(),str(path),'exec')

    def test_reward_module_is_exact_original_git_blob(self):
        raw=(ROOT/'package/reward_spec.py').read_bytes()
        self.assertEqual(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest(),
                         '56321d96901a12b7fe88f798eb10dc99511e22e0')

    def test_config_import_and_overlay_with_external_runtime_stubs(self):
        # These stubs only supply the external API surface; no physics or tensors.
        def original_cfg(num_envs,seed):return types.SimpleNamespace(episode_length_s=10.,rewards={'double_balance':types.SimpleNamespace(func=None)})
        native=types.ModuleType('native_task');native.make_cfg=original_cfg;native.head_scale=lambda env:1.
        managers=types.ModuleType('mjlab.managers');managers.RewardTermCfg=lambda **kw:types.SimpleNamespace(**kw)
        tasks=types.ModuleType('mjlab_microduck.tasks');tasks.mdp=types.SimpleNamespace()
        entries={'torch':types.ModuleType('torch'),'native_task':native,'mjlab.managers':managers,
                 'mjlab_microduck':types.ModuleType('mjlab_microduck'),'mjlab_microduck.tasks':tasks}
        with patch.dict(sys.modules,entries):
            speed=load(ROOT/'package/speed_task.py','speed_task_fixture')
            with patch.dict(sys.modules,{'speed_task':speed}):
                headroom=load(ROOT/'package/headroom_task.py','headroom_task_fixture');cfg=headroom.make_cfg(64,2026102300)
        self.assertEqual(cfg.episode_length_s,60.)
        self.assertEqual(cfg.rewards['head_roll_margin'].weight,-2.)
        self.assertEqual(cfg.rewards['head_roll_outward'].weight,-.25)
        self.assertIs(cfg.rewards['double_balance'].func,speed.speed_aligned_head_reward)


class ReturnTests(unittest.TestCase):
    setUp=RepairTests.setUp
    apply=RepairTests.apply

    def test_session_return_uses_new_archive_and_branch_preserves_old_return(self):
        self.apply();old=self.slot/'transfer';old.mkdir();(old/'return-index.json').write_text('preserved first return index')
        (old/'part-0001.zip').write_bytes(b'old archive proof')
        remote=self.root/'return-fixture.git'
        subprocess.run(['git','init','--bare',str(remote)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        subprocess.run(['git','init',str(self.repo)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        context=contextlib.ExitStack()
        for name,value in (('BASE',self.base),('SLOT',self.slot),('ORIGIN_SSH',str(remote))):
            context.enter_context(patch.object(transfer,name,value))
        context.enter_context(patch.object(transfer,'verify_package',return_value=self.package_id))
        context.enter_context(patch.object(transfer,'input_checkpoint',return_value={'path':str(self.root/'original-model-already-local.pt')}))
        context.enter_context(patch.object(transfer,'snapshot',return_value=dict(status='CAMPAIGN_ERROR_PRESERVE',
            processes_alive={'supervisor':False,'child':False},exit={'checked_baseline_goal_passed':False})))
        context.enter_context(patch.dict(os.environ,{'GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':os.devnull}))
        with context,contextlib.redirect_stdout(io.StringIO()):
            receipts=[]
            for number in (2,3):
                (self.slot/'session.json').write_text(json.dumps({'number':number}))
                (self.slot/'logs'/('new-session-%03d.log'%number)).write_text('new result '+str(number))
                transfer.main();receipts.append(json.loads((self.slot/'return.json').read_bytes()))
            transfer.main()
        self.assertNotEqual(receipts[0]['commit'],receipts[1]['commit'])
        self.assertNotEqual(receipts[0]['branch'],receipts[1]['branch'])
        for number in (2,3):
            index=json.loads((old/('session-%03d'%number)/'return-index.json').read_bytes())
            self.assertEqual(index['session'],number);self.assertEqual(index['package_identity'],self.package_id)
            self.assertIn('logs/new-session-%03d.log'%number,index['inventory']['included'])
        self.assertEqual((old/'return-index.json').read_text(),'preserved first return index')
        self.assertEqual((old/'part-0001.zip').read_bytes(),b'old archive proof')


if __name__=='__main__':unittest.main(verbosity=2)
