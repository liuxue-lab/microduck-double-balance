"""Focused boundary tests; full upstream import is tested by probe_git_import.py."""
import importlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage12.git_import import GitImportProbe, GIT, GIT_ENV, is_null_device
from stage12.host_probe import audit_boundary
from stage12.optional_imports import FontconfigFallback


class GitImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.folder = Path(self.tmp.name)
        previous = os.getcwd()
        os.chdir(self.folder)
        self.addCleanup(os.chdir, previous)
        env = patch.dict(os.environ, {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}, clear=True)
        env.start()
        self.addCleanup(env.stop)
        rows = json.loads((ROOT / 'reviewed-import-sources/gitpython-provenance.json').read_bytes())
        for row in rows['sources']:
            path = self.folder / row['installed']
            path.parent.mkdir(exist_ok=True)
            shutil.copyfile(ROOT / 'reviewed-import-sources' / row['reviewed'], path)
        self.dist = SimpleNamespace(version='3.1.46', locate_file=lambda name: self.folder / name)
        self.guard = GitImportProbe()

    def configure(self):
        self.guard.configure(ROOT, self.folder, lambda name: self.dist)

    def args(self, **overrides):
        values = dict(executable=GIT, argv=[GIT, 'version'], cwd=str(self.folder), env=dict(GIT_ENV))
        values.update(overrides)
        return tuple(values.values())

    def test_version_and_installed_source_must_match(self):
        self.dist.version = 'different'
        with self.assertRaisesRegex(RuntimeError, 'version differs'):
            self.configure()
        self.dist.version = '3.1.46'
        with (self.folder / 'git/cmd.py').open('a') as f:
            f.write('\n# different source\n')
        with self.assertRaisesRegex(RuntimeError, 'source differs'):
            self.configure()
        self.assertFalse(self.guard.active)

    def test_inactive_window_and_wrong_command_never_allow_a_process(self):
        self.configure()
        self.assertFalse(self.guard.handle('subprocess.Popen', self.args()))
        self.guard.active = True
        for argv in ([GIT, 'status'], [GIT, 'version', '--build-options'], ['/bin/sh', '-c', 'git version']):
            self.assertFalse(self.guard.handle('subprocess.Popen', self.args(executable=argv[0], argv=argv)))
        self.assertEqual(self.guard.report['permitted_queries'], [])

    def test_caller_cwd_and_environment_are_required(self):
        self.configure()
        self.guard.active = True
        for args in (self.args(), self.args(cwd='/'), self.args(env=dict(GIT_ENV, GIT_CONFIG_COUNT='1'))):
            with self.assertRaises(RuntimeError):
                self.guard.handle('subprocess.Popen', args)
        self.assertEqual(self.guard.report['permitted_queries'], [])

    def test_third_query_and_changed_source_are_rejected(self):
        self.configure()
        self.guard.active = True
        with patch.object(self.guard, '_reviewed_caller', return_value=True):
            self.assertTrue(self.guard.handle('subprocess.Popen', self.args()))
            self.assertTrue(self.guard.handle('subprocess.Popen', self.args()))
            with self.assertRaisesRegex(RuntimeError, 'exceeds import allowance'):
                self.guard.handle('subprocess.Popen', self.args())
        with (self.folder / 'git/remote.py').open('a') as f:
            f.write('\n# mutation\n')
        with self.assertRaisesRegex(RuntimeError, 'source changed'):
            self.guard.verify_unchanged()

    def test_failed_import_closes_the_window(self):
        self.configure()
        with patch.object(importlib, 'import_module', side_effect=ImportError('fixture import failure')):
            with self.assertRaisesRegex(ImportError, 'fixture import failure'):
                self.guard.preload()
        self.assertFalse(self.guard.active)
        self.assertEqual(self.guard.report['completed_queries'], 0)
        self.assertFalse(self.guard.handle('subprocess.Popen', self.args()))

    def test_null_device_requires_literal_path_character_mode_and_device_number(self):
        self.assertTrue(is_null_device('/dev/null'))
        alias = self.folder / 'alias'
        alias.symlink_to('/dev/null')
        self.assertFalse(is_null_device(str(alias)))
        for mode, device in ((stat.S_IFREG, os.makedev(1, 3)), (stat.S_IFLNK, os.makedev(1, 3)),
                             (stat.S_IFCHR, os.makedev(1, 5))):
            with patch('stage12.git_import.os.lstat', return_value=SimpleNamespace(st_mode=mode, st_rdev=device)):
                self.assertFalse(is_null_device('/dev/null'))

    def test_boundary_accepts_null_device_but_keeps_network_process_and_file_denials(self):
        hooks = []
        with patch.object(sys, 'addaudithook', side_effect=hooks.append):
            audit_boundary(self.folder, FontconfigFallback(), self.guard)
        hook = hooks[0]
        hook('open', ('/dev/null', None, os.O_RDWR))
        self.assertEqual(self.guard.report['null_device_write_opens'], 1)
        for event, args in [('open', ('/outside-stage12.txt', 'w', os.O_WRONLY | os.O_CREAT)),
                            ('subprocess.Popen', self.args()), ('socket.connect', (None, ('example.test', 443)))]:
            with self.assertRaises(RuntimeError):
                hook(event, args)


if __name__ == '__main__':
    unittest.main(verbosity=2)
