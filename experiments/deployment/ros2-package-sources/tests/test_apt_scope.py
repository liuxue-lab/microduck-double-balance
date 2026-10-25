import contextlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'package'))
import apt_scope as a
import host_setup as h
import repair

CODE = 'https://packages.microsoft.com/repos/code'
UBUNTU = 'http://archive.ubuntu.com/ubuntu'


class SourceScopeTests(unittest.TestCase):
    def fixture(self, root):
        main = root / 'system.list'
        parts = root / 'system-parts'; parts.mkdir()
        main.write_text('deb ' + UBUNTU + ' jammy main universe\n')
        (parts / 'mixed.list').write_text(
            '# user comment\ndeb [arch=amd64 signed-by=/usr/share/keyrings/code.gpg] ' + CODE +
            ' stable main\ndeb https://dl.google.com/linux/chrome-stable/deb stable main\n')
        run = root / 'run'; run.mkdir()
        with patch.object(a, 'configured_sources', return_value=(main, parts)):
            scope = a.AptSources(run, os.environ.copy())
        return scope, main, parts

    def test_exact_host_and_path_only(self):
        for uri in (CODE, CODE + '/', 'http://packages.microsoft.com/repos/code'):
            self.assertTrue(a.code_repository(uri))
        for uri in ('https://packages.microsoft.com/repos/edge',
                    'https://packages.microsoft.com.example.invalid/repos/code',
                    'https://packages.microsoft.com/repos/code-other'):
            self.assertFalse(a.code_repository(uri))

    def test_legacy_mixed_source_file_preserves_non_code_entries(self):
        original = '# deb ' + CODE + ' stable main\n' + 'deb-src ' + CODE + ' stable main\n'
        other = 'deb [arch=arm64] http://ports.ubuntu.com/ubuntu-ports jammy main\n'
        filtered, removed = a.filter_list(original + other)
        self.assertEqual(removed, 1)
        self.assertEqual(filtered, original.splitlines(keepends=True)[0] + other)

    def test_deb822_multiline_uris_preserves_inline_key_and_other_uri(self):
        key = 'Signed-By:\n -----BEGIN PGP PUBLIC KEY BLOCK-----\n .\n ABCDEFG\n -----END PGP PUBLIC KEY BLOCK-----\n'
        text = ('Types: deb deb-src\nURIs: ' + CODE + '\n ' + UBUNTU + '\n'
                'Suites: jammy\nComponents: main universe\n' + key)
        filtered, count = a.filter_sources(text)
        self.assertEqual(count, 1)
        self.assertNotIn(CODE, filtered)
        self.assertIn('URIs: ' + UBUNTU + '\n', filtered)
        self.assertIn(key, filtered)

    def test_deb822_disabled_and_adjacent_stanzas_preserved(self):
        code = 'Types: deb\nURIs: ' + CODE + '\nSuites: stable\nComponents: main\n'
        other = 'Types: deb\nURIs: ' + UBUNTU + '\nSuites: jammy\nComponents: main\n'
        disabled = code + 'Enabled: no\n'
        filtered, count = a.filter_sources(disabled + '\n' + code + '\n' + other)
        self.assertEqual(count, 1)
        self.assertIn(disabled, filtered)
        self.assertIn(other, filtered)

    def test_snapshots_preserve_system_sources_and_refresh_bootstrap_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); scope, main, parts = self.fixture(root)
            original = {str(p):p.read_bytes() for p in (main, *parts.iterdir())}
            options = scope.refresh()
            self.assertIn('Acquire::ForceIPv4=true', options)
            self.assertIn('APT::Get::List-Cleanup=false', options)
            self.assertNotIn('Acquire::AllowInsecureRepositories=true', options)
            self.assertEqual(scope.current['excluded_code_entries'], 1)
            self.assertEqual({str(p):p.read_bytes() for p in (main, *parts.iterdir())}, original)
            ros = root / 'ros-system-managed.sources'
            ros.write_text('Types: deb\nURIs: http://packages.ros.org/ros2/ubuntu\nSuites: jammy\nComponents: main\n')
            (parts / 'ros2.sources').symlink_to(ros)
            with self.assertRaisesRegex(RuntimeError, 'sources changed'): scope.options()
            scope.refresh()
            copy = Path(scope.current['directory']) / 'sources.list.d/ros2.sources'
            self.assertFalse(copy.is_symlink())
            self.assertEqual(copy.read_bytes(), ros.read_bytes())

    def test_modified_snapshot_stops_before_apt(self):
        with tempfile.TemporaryDirectory() as d:
            scope, _, _ = self.fixture(Path(d)); scope.refresh()
            Path(scope.current['source_files'][0]['copy']).write_text('unexpected')
            with self.assertRaisesRegex(RuntimeError, 'snapshot changed'): scope.options()

    @unittest.skipUnless(shutil.which('apt-get'), 'APT not present')
    def test_real_apt_uri_planning_reads_scope_without_fetch_or_install(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d); scope, _, _ = self.fixture(root)
            options = scope.refresh()
            lists = root / 'lists'; lists.mkdir(); (lists / 'partial').mkdir()
            status = root / 'status'; status.touch()
            # --print-uris plans index requests without fetching them. All state is temporary.
            cp = subprocess.run(['apt-get', *options, '-o', 'Dir::State::lists=' + str(lists),
                                 '-o', 'Dir::State::status=' + str(status),
                                 '-o', 'Debug::NoLocking=true', '--print-uris', 'update'],
                                text=True, capture_output=True, timeout=15)
            self.assertEqual(cp.returncode, 0, cp.stderr)
            self.assertIn('archive.ubuntu.com', cp.stdout)
            self.assertIn('dl.google.com', cp.stdout)
            self.assertNotIn('packages.microsoft.com', cp.stdout)
            self.assertEqual(list(lists.iterdir()), [lists / 'partial'])


class FailureReportTests(unittest.TestCase):
    def test_failed_command_embeds_actual_error_and_stdout(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            job = h.Setup(d)
            argv = [sys.executable, '-c',
                    "import sys; print('Err:1 HTTPS InRelease'); print('  Could not handshake'); "
                    "print('E: Some index files failed to download.', file=sys.stderr); sys.exit(100)"]
            with self.assertRaises(RuntimeError): job.command(argv, live=True)
            action = job.actions[-1]
            self.assertEqual(action['returncode'], 100)
            self.assertIn('Could not handshake', action['output_tail'])
            self.assertIn('Some index files failed', action['error_context'])
            self.assertFalse(action['output_truncated'])
            self.assertEqual(Path(action['log']).read_text(), action['output_tail'])

    def test_timed_out_command_keeps_partial_diagnostics(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            job = h.Setup(d)
            with self.assertRaisesRegex(RuntimeError, 'time limit'):
                job.command([sys.executable,'-u','-c',"import time; print('partial error evidence'); time.sleep(3)"], timeout=.3)
            self.assertIn('partial error evidence', job.actions[-1]['output_tail'])
            self.assertIn('time limit', job.actions[-1]['error'])

    def test_update_policy_and_install_share_scoped_ipv4_options(self):
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            job = h.Setup(d)
            fake_options = ['-o','Dir::Etc::sourcelist=/test/scoped.list','-o','Acquire::ForceIPv4=true']
            class Scope:
                current = {'excluded_code_entries':1}
            job.apt_scope = Scope()
            with patch.object(job,'apt_options',return_value=fake_options),patch.object(job,'sudo',return_value=['sudo','-n']),\
                 patch.object(job,'room'), patch.object(job,'command',side_effect=['', 'Inst ros-humble-rclpy (1.0)\n', '']) as command:
                job.apt_update();job.install(['ros-humble-rclpy'])
            self.assertEqual(len(command.call_args_list),3)
            for call in command.call_args_list:
                self.assertIn('Dir::Etc::sourcelist=/test/scoped.list',call.args[0])
                self.assertIn('Acquire::ForceIPv4=true',call.args[0])

    def test_remote_failed_setup_is_returned_for_outer_report(self):
        result={'status':'HOST_SETUP_FAILED','actions':[{'output_tail':'E: remote apt failed'}]}
        with tempfile.TemporaryDirectory() as d,\
             patch.object(repair.ops,'remote_json',side_effect=[{}, {}, result]),\
             patch.object(repair.ops.shared_ssh,'connect',return_value='/tmp/socket'),\
             patch.object(repair.ops.shared_ssh,'argv',return_value=['ssh','jetson','command']),\
             patch.object(repair.subprocess,'run'),patch.object(repair.subprocess,'call',return_value=2):
            got=repair.remote_setup(Path(d))
            self.assertEqual(got['ssh_returncode'],2)
            self.assertEqual(got['status'],'HOST_SETUP_FAILED')
            self.assertIn('remote apt failed',got['actions'][0]['output_tail'])


if __name__ == '__main__':
    unittest.main()
