import builtins
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from stage12.headless import prepare, bundled_glfw
from stage12.source_capture import file_identity
from stage12.source_reuse import verify_rows
from stage12.runner_contract import recurrent_partition, next_iteration, rollout_cost


class HeadlessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.out = self.path / 'output'
        self.out.mkdir()
        self.dists = {}
        for row in json.loads((ROOT / 'reviewed-import-sources/provenance.json').read_text())['sources']:
            package = self.path / row['distribution']
            installed = package / row['installed']
            installed.parent.mkdir(parents=True)
            shutil.copyfile(ROOT / 'reviewed-import-sources' / row['reviewed'], installed)
            class Dist:
                def __init__(self, package, version):
                    self.package, self.version = package, version
                    self.files = []
                def locate_file(self, name):
                    return self.package / name
            self.dists[row['distribution']] = Dist(package, row['version'])
        self.dist = self.dists['glfw']
        self.dist.files = ['glfw/x11/libglfw.so']
        self.lib = self.dist.locate_file(self.dist.files[0])
        self.lib.parent.mkdir(parents=True)
        self.lib.write_bytes(b'fake library: metadata-only fixture, never loaded')

    def test_preparation_sets_exact_wheel_member_without_loading(self):
        with patch.dict(os.environ, {}, clear=True), patch('ctypes.CDLL', side_effect=AssertionError('No native load in setup')):
            r = prepare(ROOT, self.out, self.dists.__getitem__)
            self.assertEqual(os.environ['MUJOCO_GL'], 'disable')
            self.assertEqual(os.environ['PYGLFW_LIBRARY'], str(self.lib))
            self.assertEqual(r['library_identity'], file_identity(self.lib))
        self.assertEqual(json.loads((self.out / 'headless-setup.json').read_text())['status'], 'HEADLESS_DISCOVERY_CONFIGURED')

    def test_exact_upstream_glfw_env_branch_never_probes_processes(self):
        source = ROOT / 'reviewed-import-sources/glfw-library.py'
        fake = object()
        with patch.dict(os.environ, {'PYGLFW_LIBRARY': str(self.lib)}), \
             patch('subprocess.Popen', side_effect=AssertionError('Unexpected child process')), \
             patch('ctypes.CDLL', return_value=fake) as cdll:
            namespace = {'__file__': str(source)}
            exec(compile(source.read_bytes(), str(source), 'exec'), namespace)
            self.assertIs(namespace['glfw'], fake)
            cdll.assert_called_once_with(str(self.lib))

    def test_exact_upstream_disabled_context_imports_no_graphics_backend(self):
        source = ROOT / 'reviewed-import-sources/mujoco-gl_context.py'
        original = builtins.__import__
        def checked(name, *args, **kwargs):
            if name.startswith(('mujoco', 'glfw', 'OpenGL')):
                raise AssertionError('Unexpected backend import: ' + name)
            return original(name, *args, **kwargs)
        with patch.dict(os.environ, {'MUJOCO_GL': 'disable'}), patch('builtins.__import__', side_effect=checked):
            namespace = {'__file__': str(source)}
            exec(compile(source.read_bytes(), str(source), 'exec'), namespace)
            self.assertNotIn('GLContext', namespace)

    def test_changed_installed_discovery_source_is_rejected_and_reported(self):
        source = self.dist.locate_file('glfw/library.py')
        source.write_bytes(source.read_bytes() + b'\n# changed\n')
        with self.assertRaisesRegex(RuntimeError, 'Installed import source differs'):
            prepare(ROOT, self.out, self.dists.__getitem__)
        self.assertEqual(json.loads((self.out / 'headless-setup.json').read_text())['status'], 'HEADLESS_DISCOVERY_BLOCKED')

    def test_missing_or_ambiguous_member_is_rejected(self):
        for files in ([], ['glfw/x11/libglfw.so', 'glfw/x11/libglfw.so']):
            self.dist.files = files
            with self.assertRaisesRegex(RuntimeError, 'exactly one'):
                bundled_glfw(self.dist)

    def test_symlinked_library_is_rejected(self):
        target = self.path / 'target.so'
        self.lib.rename(target)
        self.lib.symlink_to(target)
        with self.assertRaisesRegex(RuntimeError, 'symlinked'):
            bundled_glfw(self.dist)


class SourceReuseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        self.repo, self.prior = self.path / 'repo', self.path / 'prior'
        (self.repo / 'src').mkdir(parents=True)
        self.prior.mkdir()
        rows = []
        for i, mode in enumerate(('CAPTURED_NEW', 'REUSED_VERIFIED_V2', 'REVIEWED_HISTORICAL_IDENTITY_ONLY')):
            source = self.repo / ('source' + str(i) + '.py')
            source.write_text('x = ' + str(i) + '\n')
            old = self.prior / ('old' + str(i) + '.py')
            if mode != 'REVIEWED_HISTORICAL_IDENTITY_ONLY':
                shutil.copyfile(source, old)
            rows.append(dict(source=str(source), relative='project/' + source.name, saved=old.name,
                             prior_file=str(old), disposition=mode, **file_identity(source)))
        asset = self.repo / 'src/model.xml'
        asset.write_text('<mujoco/>')
        self.manifest = dict(status='CAPTURE_COMPLETE_WITH_VERIFIED_REUSE', files=rows,
                             assets_hashed_not_packaged=[dict(path='src/model.xml', **file_identity(asset))])
        self.roster = {r['relative']: r['source'] for r in rows}

    def verify(self):
        return verify_rows(self.manifest, self.prior, self.roster, self.repo)

    def test_verifies_both_prior_generations_without_copy_or_mutation(self):
        before = {str(p): p.read_bytes() for p in self.path.rglob('*') if p.is_file()}
        r = self.verify()
        self.assertEqual((r['current_source_identities'], r['retained_source_files'], r['source_files_copied']), (3, 2, 0))
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.path.rglob('*') if p.is_file()})

    def test_current_source_mutation_is_rejected(self):
        (self.repo / 'source0.py').write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'Current source changed'): self.verify()

    def test_retained_source_mutation_is_rejected(self):
        (self.prior / 'old1.py').write_text('changed')
        with self.assertRaisesRegex(RuntimeError, 'Retained source changed'): self.verify()

    def test_added_source_and_added_asset_are_rejected(self):
        self.roster['project/extra.py'] = str(self.repo / 'extra.py')
        with self.assertRaisesRegex(RuntimeError, 'roster changed'): self.verify()
        del self.roster['project/extra.py']
        (self.repo / 'src/extra.xml').write_text('<mujoco/>')
        with self.assertRaisesRegex(RuntimeError, 'asset roster changed'): self.verify()

    def test_asset_change_is_rejected(self):
        (self.repo / 'src/model.xml').write_text('<changed/>')
        with self.assertRaisesRegex(RuntimeError, 'Asset identity changed'): self.verify()


class RunnerContractTests(unittest.TestCase):
    def test_no_empty_or_omitted_recurrent_environment_slices(self):
        for n, b in [(1, 4), (2, 4), (5, 2), (0, 1), (1, 0), (True, 1)]:
            with self.assertRaises(RuntimeError): recurrent_partition(n, b)
        self.assertEqual(recurrent_partition(4, 2), [[0, 2], [2, 4]])

    def test_pinned_iteration_convention_requires_committed_update_count(self):
        self.assertEqual(next_iteration(0, 1), 1)
        self.assertEqual(next_iteration(9, 10), 10)
        for index, updates in [(0, 0), (9, 9), (9, 11), (-1, 0)]:
            with self.assertRaises(RuntimeError): next_iteration(index, updates)

    def test_rollout_cost_does_not_authorize_incomplete_native_budget(self):
        a, b = rollout_cost(1, 1), rollout_cost(2, 2)
        self.assertEqual((a['max_rollout_native_integrations'], b['max_rollout_native_integrations']), (84480, 168960))
        self.assertEqual(a['authorized_PPO_updates'], 0)
        self.assertIsNone(a['setup_reset_sense_forward_budget'])
        self.assertFalse(a['full_execution_budget_released'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
