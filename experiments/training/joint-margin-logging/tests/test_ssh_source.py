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
CODE='experiments/stage12/headroom-runtime-repair-v4/package'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


boot=load(ROOT/'bootstrap.py','ssh_source_boot')
client=load(ROOT/'controller.py','ssh_source_client')


def pack(document):
    raw=json.dumps(document,sort_keys=True,separators=(',',':')).encode()
    return dict(encoding='zlib-base85-json-v1',raw_bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                data=base64.b85encode(zlib.compress(raw,9)).decode())


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
