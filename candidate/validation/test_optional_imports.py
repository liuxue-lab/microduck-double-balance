"""Exercise the reviewed optional helper; never import or construct a simulator."""
import ast
import errno
import functools
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage12.optional_imports import FontconfigFallback
from stage12 import host_probe


class FontFallbackTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'matplotlib/font_manager.py'
        self.source.parent.mkdir(parents=True)
        shutil.copyfile(ROOT / 'reviewed-import-sources/matplotlib-font_manager.py', self.source)
        self.dist = SimpleNamespace(version='3.10.9', locate_file=lambda name: self.source)
        self.guard = FontconfigFallback()
        with patch.dict(os.environ, {}):
            self.guard.configure(ROOT, lambda name: self.dist)

    def function(self, check_output, filename=None, module='matplotlib.font_manager'):
        tree = ast.parse(self.source.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_get_fontconfig_fonts')
        ns = dict(__name__=module, lru_cache=functools.lru_cache, Path=Path, os=os,
                  _log=logging.getLogger('test'), subprocess=SimpleNamespace(check_output=check_output,
                  CalledProcessError=type('NeverCalledProcessError', (Exception,), {})))
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(filename or self.source), 'exec'), ns)
        return ns['_get_fontconfig_fonts']

    def test_exact_upstream_fallback_handles_denial_without_a_child(self):
        calls=[]
        def probe(argv):
            calls.append(argv)
            self.guard.handle('subprocess.Popen', ('fc-list', argv, None, None))
            self.fail('An optional child would have escaped the audit guard')
        fn=self.function(probe)
        self.assertEqual(fn(), [])
        self.assertEqual(fn(), [])
        self.assertEqual(calls, [['fc-list', '--help']])
        self.assertEqual(self.guard.report['external_processes_allowed'], 0)
        self.assertFalse(self.guard.report['optional_fontconfig_denials'][0]['spawned'])

    def test_other_caller_gets_no_special_handling(self):
        self.assertIsNone(self.guard.handle('subprocess.Popen', ('fc-list', ['fc-list','--help'], None, None)))
        self.assertEqual(self.guard.report['optional_fontconfig_denials'], [])

    def test_same_function_name_in_wrong_source_or_module_is_not_accepted(self):
        def probe(argv):
            result=self.guard.handle('subprocess.Popen', ('fc-list', argv, None, None))
            self.assertIsNone(result)
            raise RuntimeError('Generic process denial remains active')
        for fn in [self.function(probe, filename=Path(self.tmp.name)/'unreviewed.py'),
                   self.function(probe, module='unrelated')]:
            with self.assertRaisesRegex(RuntimeError, 'Generic process denial'):
                fn()

    def test_changed_source_is_rejected_before_special_handling(self):
        self.source.write_text(self.source.read_text()+'\n# changed\n')
        with self.assertRaisesRegex(RuntimeError, 'changed during preflight'):
            self.guard.verify_unchanged()
        new = FontconfigFallback()
        with self.assertRaisesRegex(RuntimeError, 'differs from reviewed'):
            new.configure(ROOT, lambda name: self.dist)

    def test_repeated_uncached_optional_probe_is_rejected(self):
        def probe(argv):
            self.guard.handle('subprocess.Popen', ('fc-list', argv, None, None))
        fn=self.function(probe)
        self.assertEqual(fn(), [])
        fn.cache_clear()
        with self.assertRaisesRegex(RuntimeError, 'Unexpected repeated'):
            fn()

    def test_other_process_network_and_writes_still_blocked_by_host_hook(self):
        hooks=[]
        with patch.object(sys, 'addaudithook', side_effect=hooks.append):
            host_probe.audit_boundary(Path(self.tmp.name), self.guard)
        hook=hooks[0]
        for event,args in [('subprocess.Popen',('/bin/true',['/bin/true'],None,None)),
                           ('subprocess.Popen',('fc-list',['fc-list','--help'],None,None)),
                           ('socket.connect',(None,('example.test',443))),('os.system',('echo test',))]:
            with self.assertRaisesRegex(RuntimeError, 'forbids network/process'):
                hook(event,args)
        with self.assertRaisesRegex(RuntimeError, 'Write outside'):
            hook('open',('/outside-run.txt','w',os.O_WRONLY|os.O_CREAT))


if __name__=='__main__':
    unittest.main(verbosity=2)
