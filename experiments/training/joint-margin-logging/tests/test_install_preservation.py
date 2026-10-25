"""Real local Git fixtures for the installer repair; no network, cloud or PPO."""
import base64
import contextlib
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import zlib

ROOT=Path(__file__).resolve().parents[1]
PREFIX='experiments/stage12/headroom-runtime-repair-v4'


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


client=load(ROOT/'controller.py','install_repair_client')


class InstallPreservationTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name);self.source=self.root/'source';self.project=self.root/'project'
        self.local=self.root/'artifacts'
        self.env=patch.dict(os.environ,{'GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':os.devnull})
        self.env.start();self.addCleanup(self.env.stop)
        self.command('git','init','-b','double-balance',str(self.source))
        self.g('config','user.name','Fixture',repo=self.source)
        self.g('config','user.email','fixture@example.invalid',repo=self.source)
        for name in ('README.txt','staged.txt','deleted.txt','renamed.txt'):
            (self.source/name).write_text('base '+name+'\n')
        (self.source/'.gitignore').write_text('ignored-user.bin\n')
        self.g('add','.',repo=self.source);self.g('commit','-m','base',repo=self.source)
        self.base=self.g('rev-parse','HEAD',repo=self.source).decode().strip()
        self.command('git','clone',str(self.source),str(self.project))
        self.g('remote','set-url','origin',client.REMOTE)
        self.g('config','user.name','Fixture');self.g('config','user.email','fixture@example.invalid')
        for name in ('README.txt','package/train.py'):
            p=self.source/PREFIX/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('incoming '+name+'\n')
        self.g('add',PREFIX,repo=self.source);self.g('commit','-m','new workflow only',repo=self.source)
        self.target=self.g('rev-parse','HEAD',repo=self.source).decode().strip()
        bundle=self.root/'new.bundle'
        self.g('bundle','create',str(bundle),'refs/heads/double-balance','^'+self.base,repo=self.source)
        self.raw=bundle.read_bytes()
        self.release=dict(commit=self.target,package_identity='a'*64,
                          bundle_sha256=hashlib.sha256(self.raw).hexdigest(),
                          bundle=base64.b85encode(zlib.compress(self.raw)).decode())
        for key,value in {'PROJECT':self.project,'LOCAL':self.local,'BASE_COMMIT':self.base,
                          'RELEASE_DATA':self.release}.items():
            p=patch.object(client,key,value,create=True);p.start();self.addCleanup(p.stop)

    def command(self,*args):
        return subprocess.run(args,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout

    def g(self,*args,repo=None):
        return self.command('git','-C',str(repo or self.project),*args)

    def install(self):
        with contextlib.redirect_stdout(io.StringIO()):client.install_code()

    def collision(self,name='package/train.py'):
        p=self.project/PREFIX/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('user-owned\n');return p

    def assert_blocked(self,p):
        before=client.local_edits();content=p.read_bytes()
        with self.assertRaisesRegex(RuntimeError,'冲突'):self.install()
        self.assertEqual(self.g('rev-parse','HEAD').decode().strip(),self.base)
        self.assertEqual(client.local_edits(),before);self.assertEqual(p.read_bytes(),content)
        self.assertFalse((self.local/'code-installed.json').exists())
        report=json.loads((self.local/'install-report.json').read_text())
        self.assertEqual(report['status'],'PATH_CONFLICT_PRESERVED')
        self.assertEqual(report['PPO_updates'],0)

    def test_install_preserves_all_edit_kinds_and_old_bundle(self):
        (self.project/'staged.txt').write_text('staged user version\n');self.g('add','staged.txt')
        (self.project/'staged.txt').write_text('different unstaged user version\n')
        (self.project/'README.txt').write_text('unstaged README\n')
        (self.project/'deleted.txt').unlink()
        self.g('mv','renamed.txt','改名 空格\n文件.txt')
        (self.project/'new-staged.txt').write_text('user staged addition\n');self.g('add','new-staged.txt')
        (self.project/'untracked.txt').write_text('untracked user note\n')
        self.g('config','merge.autoStash','true')
        before=client.local_edits()
        contents={p.name:p.read_bytes() for p in self.project.iterdir() if p.is_file()}
        cache=self.local/'microduck-stage12-headroom-from-stage12.bundle'
        cache.parent.mkdir();cache.write_bytes(b'preserved previous bundle fixture')
        self.install()
        self.assertEqual(self.g('rev-parse','HEAD').decode().strip(),self.target)
        self.assertEqual(client.local_edits(),before)
        self.assertEqual({p.name:p.read_bytes() for p in self.project.iterdir() if p.is_file()},contents)
        self.assertFalse((self.project/'deleted.txt').exists())
        self.assertEqual(cache.read_bytes(),b'preserved previous bundle fixture')
        self.assertEqual(self.g('stash','list'),b'')
        self.assertTrue(json.loads((self.local/'install-report.json').read_text())['local_edits_preserved'])

    def test_clean_install_and_repeat_are_idempotent(self):
        self.install();before=client.local_edits()
        snapshots=list((self.local/'install-preserved').iterdir())
        self.install()
        self.assertEqual(client.local_edits(),before)
        self.assertEqual(list((self.local/'install-preserved').iterdir()),snapshots)
        self.assertEqual(self.g('rev-parse','HEAD').decode().strip(),self.target)

    def test_untracked_target_is_preserved(self):
        self.assert_blocked(self.collision())

    def test_ignored_target_is_preserved(self):
        p=self.collision()
        (self.project/'.git/info/exclude').write_text(PREFIX+'/package/train.py\n')
        self.g('check-ignore',str(p));self.assert_blocked(p)

    def test_staged_target_is_preserved(self):
        p=self.collision();self.g('add',str(p));self.assert_blocked(p)

    def test_symlink_parent_is_preserved(self):
        outside=self.root/'outside';outside.mkdir();(outside/'note.txt').write_text('keep me')
        p=self.project/PREFIX;p.parent.mkdir(parents=True);p.symlink_to(outside,target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError,'冲突'):self.install()
        self.assertTrue(p.is_symlink());self.assertEqual((outside/'note.txt').read_text(),'keep me')
        self.assertEqual(list(outside.iterdir()),[outside/'note.txt'])
        self.assertEqual(self.g('rev-parse','HEAD').decode().strip(),self.base)

    def test_file_blocking_parent_directory_is_preserved(self):
        p=self.project/'experiments';p.write_text('user file, not a directory\n');self.assert_blocked(p)

    def test_unrelated_ignored_file_and_nested_note_survive(self):
        ignored=self.project/'ignored-user.bin';ignored.write_bytes(b'\0keep ignored\xff')
        p=self.project/PREFIX/'notes/user.txt';p.parent.mkdir(parents=True);p.write_text('nested note')
        before=client.local_edits();self.install()
        self.assertEqual(client.local_edits(),before)
        self.assertEqual(ignored.read_bytes(),b'\0keep ignored\xff');self.assertEqual(p.read_text(),'nested note')

    def test_existing_merge_operation_is_not_disturbed(self):
        p=self.project/'.git/MERGE_HEAD';p.write_text(self.base+'\n')
        with self.assertRaisesRegex(RuntimeError,'其他 Git 操作'):self.install()
        self.assertEqual(p.read_text(),self.base+'\n')
        self.assertEqual(self.g('rev-parse','HEAD').decode().strip(),self.base)

    def test_local_commit_divergence_is_not_overwritten(self):
        (self.project/'README.txt').write_text('local committed work\n')
        self.g('add','README.txt');self.g('commit','-m','local work')
        head=self.g('rev-parse','HEAD')
        with self.assertRaisesRegex(RuntimeError,'分歧'):self.install()
        self.assertEqual(self.g('rev-parse','HEAD'),head)
        self.assertEqual((self.project/'README.txt').read_text(),'local committed work\n')


if __name__=='__main__':unittest.main(verbosity=2)
