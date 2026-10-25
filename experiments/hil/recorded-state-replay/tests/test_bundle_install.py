"""Exercise the actual release bundle in isolated Git checkouts; never push."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get('MICRODUCK_BUNDLE_SOURCE', str(HERE / 'bundle-objects.git')))
SCRIPT = Path(os.environ.get('MICRODUCK_RELEASE_SCRIPT',str(HERE / 'release/microduck-stage12-hil-replay-v1-20261025.py')))


@unittest.skipUnless(SOURCE.is_dir() and SCRIPT.is_file(), 'Release builder and Git object store required')
class BundleInstall(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.repo = self.root / 'workspace'
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        def git(*args):
            return subprocess.check_output(['git', '-C', str(self.repo), *args], stderr=subprocess.PIPE)
        self.git = git
        git('config', 'user.name', 'Offline Test')
        git('config', 'user.email', 'offline-test@invalid.example')
        self.base = '38e7ee84ad7c83b023bb8dcf46b9252d60f3552f'
        # The offline review object store intentionally holds a partial source tree.
        # Use the real sparse-checkout fixture used by prior bundle validation.
        shutil.copytree(SOURCE / 'objects', self.repo / '.git/objects', dirs_exist_ok=True)
        (self.repo / '.git/shallow').write_text(self.base + '\n')
        git('update-ref', 'refs/heads/double-balance', self.base)
        git('symbolic-ref', 'HEAD', 'refs/heads/double-balance')
        git('config', 'core.sparseCheckout', 'true')
        (self.repo / '.git/info/sparse-checkout').write_text(
            '/experiments/stage12/native-mujoco-v3/\n/experiments/stage12/hil-replay-v1/\n')
        git('read-tree', '-mu', 'HEAD')
        git('remote', 'add', 'origin', 'git@github.com:liuxue-lab/microduck-double-balance.git')
        spec = importlib.util.spec_from_file_location('release_under_test', SCRIPT)
        self.installer = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.installer)
        self.installer.PROJECT = self.repo
        self.installer.LOCAL = self.root / 'artifacts/hil-replay-v1'

    def tearDown(self):
        self.tmp.cleanup()

    def test_real_bundle_preserves_dirty_tree_and_reuses_pinned_package(self):
        tracked = self.repo / 'experiments/stage12/native-mujoco-v3/README.txt'
        tracked.write_bytes(tracked.read_bytes() + b'\nTest staged user note.\n')
        self.git('add', str(tracked))
        tracked.write_bytes(tracked.read_bytes() + b'Test unstaged user note.\n')
        (self.repo / 'user-notes.txt').write_text('untracked user content')
        before = self.installer.local_edits()
        self.installer.install_code()
        self.assertEqual(self.git('rev-parse', 'HEAD').decode().strip(), self.installer.RELEASE_DATA['commit'])
        self.assertEqual(before, self.installer.local_edits())
        package = self.installer.install_package()
        self.assertEqual(package, self.installer.installed_package())
        # A second installation must not repeat, reset, stage or rewrite user work.
        self.installer.install_code()
        self.assertEqual(before, self.installer.local_edits())
        (package / 'replay_ops.py').write_text('corrupt package')
        with self.assertRaisesRegex(RuntimeError, '变化'):
            self.installer.installed_package()

    def test_ignored_incoming_conflict_is_preserved(self):
        relative = 'experiments/stage12/hil-replay-v1/README.txt'
        path = self.repo / relative
        path.parent.mkdir(parents=True)
        path.write_text('user ignored content, never overwrite')
        with (self.repo / '.git/info/exclude').open('a') as output:
            output.write('\n/experiments/stage12/hil-replay-v1/\n')
        with self.assertRaisesRegex(RuntimeError, '冲突'):
            self.installer.install_code()
        self.assertEqual(path.read_text(), 'user ignored content, never overwrite')
        self.assertEqual(self.git('rev-parse', 'HEAD').decode().strip(), self.base)

    def test_divergent_commit_stops_without_reset(self):
        (self.repo / 'user-change.txt').write_text('new local commit')
        self.git('commit', '--allow-empty', '-qm', 'Keep a user commit')
        before = self.git('rev-parse', 'HEAD')
        with self.assertRaisesRegex(RuntimeError, '新提交或分歧'):
            self.installer.install_code()
        self.assertEqual(before, self.git('rev-parse', 'HEAD'))
        self.assertEqual((self.repo / 'user-change.txt').read_text(), 'new local commit')


if __name__ == '__main__':
    unittest.main(verbosity=2)
