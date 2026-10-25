"""Offline fixture checks; never touches the canonical Microduck checkout."""
import base64
import contextlib
import hashlib
import importlib.util
import io
import os
import zlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

MODULE=Path(os.environ.get('STAGE10_FINALIZER_UNDER_TEST', str(Path(__file__).resolve().parents[2]/'stage10_finalize.py')))

def digest(data): return hashlib.sha256(data).hexdigest()
def js(x): return (json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode()

class FinalizerTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='stage10-finalizer-fixture-')
        self.root=Path(self.tmp.name)
        self.repo=self.root/'repo';self.repo.mkdir()
        self.stage=self.root/'artifacts'
        spec=importlib.util.spec_from_file_location('finalizer_fixture',MODULE)
        self.m=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.m)
        self.m.REPO=self.repo;self.m.STAGE=self.stage
        self.git('init','-q','-b','double-balance')
        self.git('config','user.email','fixture@example.invalid')
        self.git('config','user.name','Stage10 offline fixture')
        self.git('config','commit.gpgsign','false')
        self.git('config','tag.gpgsign','false')
        self.git('remote','add','origin',self.m.ORIGIN)
        frozen={}
        for i in range(101):
            rel='frozen/'+str(i)+'.txt';data=('baseline '+str(i)+'\n').encode()
            self.write(self.repo/rel,data);frozen[rel]=digest(data)
        self.runtime=self.repo/'docs/audits/stage-09-runtime-hashes.json'
        self.write(self.runtime,js(frozen));self.m.RUNTIME_SHA=digest(js(frozen))
        self.git('add','.');self.git('commit','-qm','fixture baseline')
        self.base=self.git('rev-parse','HEAD');self.m.BASE=self.base
        self.git('tag','-a','stage-09-complete','-m','fixture previous tag')
        self.old_tag=self.git('rev-parse','stage-09-complete')
        batches=[]
        for n in 'ABCD':
            root=self.stage/n
            data=('evidence '+n).encode();self.write(root/'data.bin',data)
            rows=[dict(path='data.bin',sha256=digest(data),size_bytes=len(data))]
            self.write(root/'files.json',js(rows))
            archive=root/'archive.zip';self.write(archive,b'fixture archived bytes')
            batches.append(dict(batch=n,root=str(root),manifest='files.json',manifest_sha256=digest(js(rows)),
                                file_count=1,archives=[dict(path=str(archive),sha256=digest(archive.read_bytes()))]))
        self.evidence=dict(batches=batches,replay_raw=[])
        for k in ['checkpoint','dataset','dataset_receipt']:
            p=self.stage/(k+'.bin');self.write(p,k.encode())
            self.evidence[k]=dict(path=str(p),sha256=digest(p.read_bytes()))
        self.rel='docs/audits/stage-10-fixture.json'
        data=js({'test_only':True})
        self.payload=dict(evidence=self.evidence,files=[dict(path=self.rel,sha256=digest(data),data=base64.b64encode(data).decode())])

    def tearDown(self): self.tmp.cleanup()
    def write(self,p,data): p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    def git(self,*args):
        return subprocess.check_output(['git','-C',str(self.repo),*args],stderr=subprocess.STDOUT).decode().strip()
    def perform(self,verify=False):
        with contextlib.redirect_stdout(io.StringIO()): return self.m.perform(self.payload,verify)
    def assert_no_commit_or_targets(self):
        self.assertEqual(self.git('rev-parse','HEAD'),self.base)
        self.assertFalse((self.repo/self.rel).exists())
        self.assertEqual(self.git('rev-parse','stage-09-complete'),self.old_tag)

    def test_verify_only_has_no_repository_or_evidence_write(self):
        before={str(p.relative_to(self.root)):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.perform(True)
        after={str(p.relative_to(self.root)):p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        self.assert_no_commit_or_targets()

    def test_success_and_idempotent_rerun(self):
        head=self.perform()
        self.assertEqual(self.git('rev-parse','HEAD^'),self.base)
        self.assertEqual(self.git('cat-file','-t',self.m.TAG),'tag')
        self.assertEqual(self.git('rev-parse',self.m.TAG+'^{commit}'),head)
        self.assertEqual(self.git('rev-parse','stage-09-complete'),self.old_tag)
        receipt=self.stage/'finalization'/head/'finalization-receipt.json'
        before=receipt.read_bytes()
        self.assertEqual(self.perform(),head)
        self.assertEqual(receipt.read_bytes(),before)
        r=json.loads(before);self.assertFalse(r['stage10_complete']);self.assertEqual(r['new_ppo_updates'],0)
        self.assertEqual(r['cloud_status'],'OFF_CONFIRMED_BY_USER')

    def test_partial_exact_payload_can_resume(self):
        self.write(self.repo/self.rel,base64.b64decode(self.payload['files'][0]['data']))
        self.git('add','--',self.rel)
        self.perform()
        self.assertEqual(self.git('cat-file','-t',self.m.TAG),'tag')

    def test_interrupted_after_commit_resumes_tag_without_second_commit(self):
        actual=self.m.run_git
        def fail_tag(*args):
            if args[0]=='tag': raise RuntimeError('simulated interruption before tag')
            actual(*args)
        with patch.object(self.m,'run_git',side_effect=fail_tag):
            with self.assertRaisesRegex(RuntimeError,'simulated interruption'): self.perform()
        head=self.git('rev-parse','HEAD')
        self.assertNotEqual(head,self.base)
        self.assertEqual(self.perform(),head)
        self.assertEqual(self.git('rev-parse',self.m.TAG+'^{commit}'),head)

    def test_changed_evidence_refuses_before_writes(self):
        self.write(self.stage/'C/data.bin',b'tampered')
        with self.assertRaisesRegex(RuntimeError,'mismatch'): self.perform()
        self.assert_no_commit_or_targets()

    def test_missing_evidence_refuses_before_writes(self):
        (self.stage/'B/archive.zip').unlink()
        with self.assertRaisesRegex(RuntimeError,'Missing evidence'): self.perform()
        self.assert_no_commit_or_targets()

    def test_unrelated_tracked_edit_is_preserved(self):
        p=self.runtime;p.write_bytes(p.read_bytes()+b' ')
        with self.assertRaisesRegex(RuntimeError,'SHA256 mismatch'): self.perform()
        self.assertTrue(p.read_bytes().endswith(b' '));self.assert_no_commit_or_targets()

    def test_unrelated_staged_file_is_preserved(self):
        self.write(self.repo/'user-note.txt',b'user edit')
        self.git('add','--','user-note.txt')
        before=self.git('diff','--cached')
        with self.assertRaisesRegex(RuntimeError,'Unrelated staged'): self.perform()
        self.assertEqual(self.git('diff','--cached'),before);self.assert_no_commit_or_targets()

    def test_conflicting_target_is_preserved(self):
        self.write(self.repo/self.rel,b'user content')
        with self.assertRaisesRegex(RuntimeError,'Existing target differs'): self.perform()
        self.assertEqual((self.repo/self.rel).read_bytes(),b'user content')
        self.assertEqual(self.git('rev-parse','HEAD'),self.base)

    def test_conflicting_staged_target_is_preserved(self):
        self.write(self.repo/self.rel,b'user index content');self.git('add','--',self.rel)
        self.write(self.repo/self.rel,base64.b64decode(self.payload['files'][0]['data']))
        before=self.git('show',':'+self.rel)
        with self.assertRaisesRegex(RuntimeError,'Staged target differs'): self.perform()
        self.assertEqual(self.git('show',':'+self.rel),before)
        self.assertEqual(self.git('rev-parse','HEAD'),self.base)

    def test_conflicting_tag_is_preserved(self):
        self.git('tag','-a',self.m.TAG,'-m','unrelated existing tag')
        tag=self.git('rev-parse',self.m.TAG)
        with self.assertRaisesRegex(RuntimeError,'Conflicting'): self.perform()
        self.assertEqual(self.git('rev-parse',self.m.TAG),tag);self.assert_no_commit_or_targets()

    def test_symlink_target_refused(self):
        target=self.root/'outside';self.write(target,b'preserve')
        (self.repo/self.rel).symlink_to(target)
        with self.assertRaisesRegex(RuntimeError,'(escapes|Linked)'): self.perform()
        self.assertEqual(target.read_bytes(),b'preserve')

    def test_unexpected_head_refuses(self):
        self.write(self.repo/'unrelated.txt',b'other work');self.git('add','.');self.git('commit','-qm','user work')
        head=self.git('rev-parse','HEAD')
        with self.assertRaisesRegex(RuntimeError,'Unexpected HEAD'): self.perform()
        self.assertEqual(self.git('rev-parse','HEAD'),head)
        self.assertFalse((self.repo/self.rel).exists())

    def test_complete_delivery_tree_stages_and_commits_cleanly(self):
        rows=json.loads(zlib.decompress(base64.b64decode(self.m.PAYLOAD)))['files']
        self.payload['files']=rows
        head=self.perform()
        self.assertEqual(self.git('rev-parse',self.m.TAG+'^{commit}'),head)
        self.assertEqual(len(self.git('diff','--name-only',self.base,head).splitlines()),len(rows)+2)

if __name__=='__main__': unittest.main(verbosity=2)
