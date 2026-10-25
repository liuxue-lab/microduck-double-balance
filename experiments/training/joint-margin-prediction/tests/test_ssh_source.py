"""Real local Git and subprocess tests; no network, GPU, physics or PPO."""
import ast
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import zlib

ROOT=Path(__file__).resolve().parents[1]
CODE='experiments/stage12/headroom-constraint-v3/package'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


boot=load(ROOT/'bootstrap.py','ssh_source_boot')
client=load(ROOT/'controller.py','ssh_source_client')


def pack(document):
    raw=json.dumps(document,sort_keys=True,separators=(',',':')).encode()
    return dict(encoding='zlib-base85-json-v1',raw_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                data=base64.b85encode(zlib.compress(raw,9)).decode())


class SourceTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.source=self.root/'local-repo';self.base=self.root/'cloud';self.repo=self.base/'workspace'
        self.slot=self.base/'artifacts/double-balance-stage12/native-headroom-constraint-v3'
        self.package=self.slot/'package';self.local=self.root/'client-artifacts'
        env=patch.dict(os.environ,{'GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':os.devnull})
        env.start();self.addCleanup(env.stop)
        for repo in (self.source,self.repo):
            repo.mkdir(parents=True);self.git(repo,'init','-b','double-balance')
            self.git(repo,'config','user.name','Fixture');self.git(repo,'config','user.email','fixture@example.invalid')
            self.git(repo,'config','core.sshCommand','false')
        self.git(self.repo,'remote','add','origin','https://example.invalid/upstream.git')
        (self.repo/'preserved.txt').write_text('cloud worktree must remain untouched')
        self.runtime=self.repo/'.venv/bin/python';self.runtime.parent.mkdir(parents=True)
        self.runtime.write_text('#!'+sys.executable+'\nimport json,sys\nprint("MD12_HEADROOM_STATE="+json.dumps({"status":"NOT_STARTED","added_PPO_updates":0,"operation":sys.argv[-1]}),flush=True)\n')
        self.runtime.chmod(0o755)
        data=b'# committed fixture code, never trained\n'
        pins={'agent.py':{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}
        self.identity=hashlib.sha256(json.dumps(pins,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        p=self.source/CODE;p.mkdir(parents=True);(p/'agent.py').write_bytes(data)
        (p/'manifest.json').write_text(json.dumps(dict(identity=self.identity,files=pins))+'\n')
        self.git(self.source,'add','.');self.git(self.source,'commit','-m','pinned fixture')
        self.commit=self.git(self.source,'rev-parse','HEAD').decode().strip()
        self.release=dict(commit=self.commit,package_identity=self.identity)
        for module,name,value in ((client,'PROJECT',self.source),(client,'LOCAL',self.local),
                                  (client,'RELEASE_DATA',self.release),(boot,'BASE',self.base)):
            p=patch.object(module,name,value,create=True);p.start();self.addCleanup(p.stop)
        self.request=dict(operation='start',commit=self.commit,package_identity=self.identity,source=client.source_payload())
        self.document=json.loads(zlib.decompress(base64.b85decode(self.request['source']['data'])))

    def git(self,repo,*args):
        return subprocess.run(['git','-C',str(repo),*args],check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout

    def invoke(self,request=None):
        with patch.object(sys,'stdin',io.StringIO(json.dumps(request or self.request))), \
             patch.object(boot.os,'execv') as execute,contextlib.redirect_stdout(io.StringIO()) as output:
            boot.main();return execute.call_args_list,output.getvalue()

    def assert_failed(self,request,pattern):
        with patch.object(boot.os,'execv') as execute:
            with self.assertRaisesRegex((RuntimeError,ValueError),pattern):self.invoke(request)
            execute.assert_not_called()
        self.assertFalse(self.package.exists());self.assertFalse((self.slot/'budget.json').exists())
        report=json.loads(sorted((self.slot/'bootstrap-source').glob('*.json'))[-1].read_bytes())
        self.assertEqual(report['status'],'SOURCE_FAILED_PRESERVE');self.assertTrue(report['configuration_unchanged'])

    def test_uses_committed_source_while_preserving_dirty_local_worktree(self):
        (self.source/CODE/'agent.py').write_text('uncommitted edit must not be sent')
        status=self.git(self.source,'status','--porcelain=v1','-z')
        self.assertEqual(client.source_payload(),self.request['source'])
        self.assertEqual(self.git(self.source,'status','--porcelain=v1','-z'),status)

    def test_missing_local_commit_fails_before_ssh(self):
        self.release['commit']='0'*40
        with self.assertRaises(subprocess.CalledProcessError):client.source_payload()
        self.assertFalse(self.slot.exists())

    def test_independent_cloud_repository_no_network_and_unchanged_configuration(self):
        config=(self.repo/'.git/config').read_bytes();local=(self.repo/'preserved.txt').read_bytes()
        original=boot.git_bytes
        def metadata_only(repo,*args):
            self.assertEqual(args[:2],('rev-parse','--git-path'));return original(repo,*args)
        with patch.object(boot,'git_bytes',side_effect=metadata_only):calls,output=self.invoke()
        self.assertEqual(len(calls),1);self.assertEqual(calls[0].args[1][-1],'start')
        self.assertEqual((self.package/'agent.py').read_bytes(),self.git(self.source,'show',self.commit+':'+CODE+'/agent.py'))
        self.assertEqual((self.repo/'.git/config').read_bytes(),config)
        self.assertEqual((self.repo/'preserved.txt').read_bytes(),local)
        self.assertIn('GitHub fetches=0',output)
        report=json.loads(next((self.slot/'bootstrap-source').glob('*.json')).read_bytes())
        self.assertEqual(report['status'],'PINNED_SOURCE_INSTALLED');self.assertEqual(report['new_PPO_updates'],0)

    def test_repeated_start_preserves_budget_deadline_prior_failures_and_file_times(self):
        self.invoke();(self.slot/'budget.json').write_text('{"attempted_updates":7,"deadline_utc":12345}\n')
        old=self.slot/'bootstrap-fetch/old-failure.json';old.parent.mkdir();old.write_text('historical timeout evidence')
        paths=[*self.package.iterdir(),self.slot/'source-commit.json',self.slot/'budget.json',old]
        before={str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in paths}
        self.invoke();self.invoke(dict(self.request,operation='resume'))
        self.assertEqual(before,{str(p):(p.read_bytes(),p.stat().st_mtime_ns) for p in paths})

    def test_readonly_status_without_install_needs_no_payload_or_git(self):
        request={k:v for k,v in self.request.items() if k!='source'};request['operation']='status'
        with patch.object(boot,'git_bytes',side_effect=AssertionError('No Git expected')):
            calls,output=self.invoke(request)
        self.assertEqual(calls,[]);self.assertIn('NOT_INSTALLED',output);self.assertFalse(self.slot.exists())

    def test_payload_transport_corruption_stops_before_install(self):
        source=dict(self.request['source'],sha256='0'*64)
        self.assert_failed(dict(self.request,source=source),'transport hash differs')

    def test_wrong_source_commit_stops_before_install(self):
        document=dict(self.document,commit='f'*40)
        self.assert_failed(dict(self.request,source=pack(document)),'provenance differs')

    def test_modified_code_stops_even_with_recomputed_transport_hash(self):
        self.document['files']['agent.py']=base64.b64encode(b'changed').decode()
        self.assert_failed(dict(self.request,source=pack(self.document)),'source hash differs')

    def test_unapproved_manifest_identity_is_rejected(self):
        self.assert_failed(dict(self.request,package_identity='a'*64),'provenance differs')

    def test_oversized_compression_rejected_before_json_parse(self):
        source=dict(self.request['source'],data=base64.b85encode(zlib.compress(b'X'*(boot.MAX_SOURCE_BYTES+100))).decode())
        self.assert_failed(dict(self.request,source=source),'oversized compressed')

    def test_path_traversal_rejected_even_with_consistent_hashes(self):
        data=b'never write';pins={'../outside.py':{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}
        identity=hashlib.sha256(json.dumps(pins,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        document=dict(self.document,package_identity=identity,files={
            '../outside.py':base64.b64encode(data).decode(),
            'manifest.json':base64.b64encode(json.dumps(dict(identity=identity,files=pins)).encode()).decode()})
        self.assert_failed(dict(self.request,package_identity=identity,source=pack(document)),'Invalid package path')
        self.assertFalse((self.slot/'outside.py').exists())

    def test_existing_conflict_does_not_partially_install_or_overwrite(self):
        self.package.mkdir(parents=True);existing=self.package/'agent.py';existing.write_text('preserve mismatch')
        with self.assertRaisesRegex(RuntimeError,'Installed campaign differs'):self.invoke()
        self.assertEqual(existing.read_text(),'preserve mismatch');self.assertFalse((self.package/'manifest.json').exists())

    def test_broken_symlink_does_not_write_outside_package(self):
        self.package.mkdir(parents=True);outside=self.root/'outside.py'
        (self.package/'agent.py').symlink_to(outside)
        with self.assertRaisesRegex(RuntimeError,'Installed campaign differs'):self.invoke()
        self.assertFalse(outside.exists());self.assertTrue((self.package/'agent.py').is_symlink())

    def test_actual_release_payload_and_rpc_subprocess_install(self):
        output_path=ROOT/'release/microduck-stage12-headroom-v3-20261023.py'
        release=ast.literal_eval(ast.parse(output_path.read_bytes()).body[0].value)
        bootstrap=zlib.decompress(base64.b85decode(release['bootstrap'])).decode()
        bootstrap=bootstrap.replace("BASE=Path('/root/autodl-tmp/microduck-double-balance')",'BASE=Path('+repr(str(self.base))+')')
        release['bootstrap']=base64.b85encode(zlib.compress(bootstrap.encode())).decode()
        class LocalChannel:
            @staticmethod
            def argv(remote):return shlex.split(remote)
        with patch.object(client,'RELEASE_DATA',release),patch.object(client,'PROJECT',ROOT/'bundle-objects.git'), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            state=client.rpc(LocalChannel(),'start',timeout=20)
        self.assertEqual(state['status'],'NOT_STARTED');self.assertEqual(state['added_PPO_updates'],0)
        self.assertIn('HEADROOM_SOURCE_VERIFIED=',output.getvalue())
        for path in (ROOT/'package').iterdir():
            if path.is_file():self.assertEqual((self.package/path.name).read_bytes(),path.read_bytes())
        self.assertFalse((self.slot/'budget.json').exists())
        request=json.loads(next((self.local/'requests').glob('*.request.json')).read_bytes())
        self.assertLess(len(json.dumps(request)),400000)


class RpcFailureTests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory();self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        release=dict(commit='a'*40,package_identity='b'*64,bootstrap='unused-fixture')
        for name,value in (('LOCAL',self.root),('RELEASE_DATA',release)):
            p=patch.object(client,name,value,create=True);p.start();self.addCleanup(p.stop)

    def channel(self,code):
        class Channel:
            @staticmethod
            def argv(remote):return [sys.executable,'-u','-c',code]
        return Channel()

    def test_reported_bootstrap_error_returns_promptly_even_if_channel_lingers(self):
        ssh=self.channel('import time\nprint(\'MD12_HEADROOM_ERROR={"error":"fixture timeout","bootstrap_failed":true}\',flush=True)\ntime.sleep(50)')
        start=time.monotonic()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            with self.assertRaisesRegex(RuntimeError,'fixture timeout'):client.rpc(ssh,'status',timeout=20)
        self.assertLess(time.monotonic()-start,4)
        self.assertTrue(json.loads((self.root/'last-error.json').read_bytes())['bootstrap_failed'])
        self.assertIn('fixture timeout',output.getvalue());self.assertEqual(len(list((self.root/'requests').glob('*.log'))),1)

    def test_timeout_records_uncertain_outcome_and_cleans_only_request_process_group(self):
        pidfile=self.root/'descendant.pid'
        code='import pathlib,subprocess,sys,time\np=subprocess.Popen([sys.executable,"-c","import time; time.sleep(50)"])\npathlib.Path('+repr(str(pidfile))+').write_text(str(p.pid))\ntime.sleep(50)'
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(TimeoutError,'仅用 --status'):client.rpc(self.channel(code),'status',timeout=.2)
        error=json.loads((self.root/'last-error.json').read_bytes())
        self.assertEqual(error['status'],'SSH_REQUEST_OUTCOME_UNCERTAIN_PRESERVE')
        pid=int(pidfile.read_text());proc=Path('/proc')/str(pid)/'stat'
        self.assertTrue(not proc.exists() or proc.read_text().split()[2]=='Z')
        self.assertFalse((self.root/'campaign.json').exists())

    def test_status_rpc_does_not_prepare_source(self):
        ssh=self.channel('print(\'MD12_HEADROOM_STATE={"status":"NOT_INSTALLED"}\')')
        with patch.object(client,'source_payload',side_effect=AssertionError('Source must not be sent for status')), \
             contextlib.redirect_stdout(io.StringIO()):state=client.rpc(ssh,'status',timeout=5)
        self.assertEqual(state['status'],'NOT_INSTALLED')
        request=json.loads(next((self.root/'requests').glob('*.request.json')).read_bytes())
        self.assertNotIn('source',request)


if __name__=='__main__':unittest.main(verbosity=2)
