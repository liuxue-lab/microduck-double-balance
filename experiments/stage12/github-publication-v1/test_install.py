"""Exercise local Git imports and refusal paths without any network or HIL."""
import base64
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
import zlib

spec = importlib.util.spec_from_file_location('publication_install', Path(__file__).with_name('install.py'))
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def git(repo, *args):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_AUTHOR_NAME='Fixture', GIT_AUTHOR_EMAIL='fixture@example.invalid',
               GIT_COMMITTER_NAME='Fixture', GIT_COMMITTER_EMAIL='fixture@example.invalid',
               GIT_AUTHOR_DATE='2026-10-25T00:00:00Z', GIT_COMMITTER_DATE='2026-10-25T00:00:00Z')
    return subprocess.check_output(['git', '-c', 'core.hooksPath=/dev/null', '-c', 'protocol.allow=never',
                                    '-c', 'protocol.file.allow=always', '-C', str(repo), *args],
                                   env=env, stderr=subprocess.PIPE).decode().strip()


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='microduck-doc-import-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        writer = self.root / 'writer'
        writer.mkdir()
        git(writer, 'init', '-q', '-b', 'double-balance')
        (writer / 'README.md').write_text('# old\n')
        (writer / 'keep.txt').write_text('existing tracked content\n')
        git(writer, 'add', 'README.md', 'keep.txt')
        git(writer, 'commit', '-qm', 'base')
        parent = git(writer, 'rev-parse', 'HEAD')
        self.repo = self.root / 'client'
        self.repo.mkdir()
        git(self.repo, 'init', '-q')
        git(self.repo, 'fetch', '--no-tags', str(writer), parent)
        git(self.repo, 'checkout', '-qb', 'double-balance', 'FETCH_HEAD')
        git(self.repo, 'remote', 'add', 'origin', installer.REMOTE)
        contents = {'README.md': '# new\n', 'docs/double-balance/README.md': '# Double balance\n'}
        for name, data in contents.items():
            p = writer / name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(data)
        git(writer, 'add', *contents)
        git(writer, 'commit', '-qm', 'documentation')
        commit = git(writer, 'rev-parse', 'HEAD')
        tag = 'stage-12-hil-docs-20261025'
        git(writer, 'tag', '-a', tag, '-m', 'Documentation fixture')
        tag_object = git(writer, 'rev-parse', 'refs/tags/' + tag)
        bundle = self.root / 'fixture.bundle'
        git(writer, 'bundle', 'create', str(bundle), '^' + parent, 'refs/heads/double-balance', 'refs/tags/' + tag)
        pack = bundle.read_bytes().split(b'\n\n', 1)[1]
        header = (f'# v2 git bundle\n-{parent} native-live-v1 prerequisite\n'
                  f'{commit} refs/heads/double-balance\n{tag_object} refs/tags/{tag}\n\n').encode()
        raw = header + pack
        files = {name: {'action': 'modify' if name == 'README.md' else 'add',
                        'bytes': len(data.encode()), 'sha256': hashlib.sha256(data.encode()).hexdigest()}
                 for name, data in contents.items()}
        files['README.md']['old_sha256'] = hashlib.sha256(b'# old\n').hexdigest()
        self.release = {'parent': parent, 'commit': commit, 'tag': tag, 'tag_object': tag_object,
                        'files': files, 'new_roots': ['docs/double-balance'],
                        'bundle_sha256': hashlib.sha256(raw).hexdigest(),
                        'bundle': base64.b85encode(zlib.compress(raw)).decode()}
        self.output = self.root / 'artifacts'

    def assertRefused(self, release=None):
        before = git(self.repo, 'rev-parse', 'HEAD')
        with self.assertRaises(installer.Refusal):
            installer.apply(self.repo, self.output, release or self.release)
        self.assertEqual(git(self.repo, 'rev-parse', 'HEAD'), before)

    def test_fast_forward_and_repeat_preserve_untracked(self):
        unknown = self.repo / 'docs/audits/private-note.txt'
        unknown.parent.mkdir(parents=True)
        unknown.write_bytes(b'preserve me\x00\xff')
        fixtures = self.repo / 'tests/fixtures/local.bin'
        fixtures.parent.mkdir(parents=True)
        fixtures.write_bytes(b'local fixture')
        result = installer.apply(self.repo, self.output, self.release)
        self.assertEqual(result['status'], 'INSTALLED')
        self.assertEqual(git(self.repo, 'rev-parse', 'HEAD'), self.release['commit'])
        self.assertEqual(git(self.repo, 'rev-parse', self.release['tag']), self.release['tag_object'])
        self.assertEqual(unknown.read_bytes(), b'preserve me\x00\xff')
        self.assertEqual(fixtures.read_bytes(), b'local fixture')
        self.assertFalse(result['remote_push_performed'])
        self.assertEqual(installer.apply(self.repo, self.output, self.release)['status'], 'ALREADY_INSTALLED')

    def test_check_is_read_only(self):
        self.assertEqual(installer.preflight(self.repo, self.release), self.release['parent'])
        self.assertFalse(self.output.exists())
        self.assertEqual(git(self.repo, 'status', '--porcelain'), '')

    def test_dirty_tracked_file(self):
        (self.repo / 'keep.txt').write_text('user edit\n')
        self.assertRefused()
        self.assertEqual((self.repo / 'keep.txt').read_text(), 'user edit\n')

    def test_staged_change(self):
        (self.repo / 'new.txt').write_text('staged user file')
        git(self.repo, 'add', 'new.txt')
        self.assertRefused()

    def test_hidden_readme_edit(self):
        git(self.repo, 'update-index', '--assume-unchanged', 'README.md')
        (self.repo / 'README.md').write_text('hidden user edit\n')
        self.assertRefused()

    def test_new_path_collision(self):
        p = self.repo / 'docs/double-balance/README.md'
        p.parent.mkdir(parents=True)
        p.write_text('user document\n')
        self.assertRefused()
        self.assertEqual(p.read_text(), 'user document\n')

    def test_ignored_directory_collision(self):
        (self.repo / '.git/info/exclude').write_text('docs/double-balance/\n')
        p = self.repo / 'docs/double-balance/private.bin'
        p.parent.mkdir(parents=True)
        p.write_bytes(b'ignored asset')
        self.assertRefused()
        self.assertEqual(p.read_bytes(), b'ignored asset')

    def test_symlink_directory(self):
        outside = self.root / 'outside'
        outside.mkdir()
        (self.repo / 'docs').symlink_to(outside, target_is_directory=True)
        self.assertRefused()
        self.assertEqual(list(outside.iterdir()), [])

    def test_dirty_other_branch(self):
        git(self.repo, 'switch', '-c', 'other')
        self.assertRefused()

    def test_divergent_head(self):
        git(self.repo, 'commit', '--allow-empty', '-qm', 'unrelated new work')
        self.assertRefused()

    def test_wrong_remote(self):
        git(self.repo, 'remote', 'set-url', 'origin', 'https://github.com/liuxue-lab/microduck-double-balance.git')
        self.assertRefused()

    def test_wrong_push_remote(self):
        git(self.repo, 'config', 'remote.origin.pushurl', 'git@github.com:someone/other.git')
        self.assertRefused()

    def test_tag_conflict(self):
        git(self.repo, 'tag', self.release['tag'])
        self.assertRefused()
        self.assertEqual(git(self.repo, 'rev-parse', self.release['tag']), self.release['parent'])

    def test_corrupt_bundle(self):
        release = copy.deepcopy(self.release)
        release['bundle_sha256'] = '0' * 64
        self.assertRefused(release)
        self.assertFalse(self.output.exists())

    def test_changed_scope_manifest(self):
        release = copy.deepcopy(self.release)
        del release['files']['docs/double-balance/README.md']
        self.assertRefused(release)
        self.assertFalse((self.repo / 'docs/double-balance').exists())

    def test_wrong_file_digest(self):
        release = copy.deepcopy(self.release)
        release['files']['README.md']['sha256'] = 'f' * 64
        self.assertRefused(release)
        self.assertEqual((self.repo / 'README.md').read_text(), '# old\n')

    def test_output_inside_checkout(self):
        self.output = self.repo / 'artifacts'
        self.assertRefused()
        self.assertFalse(self.output.exists())

    def test_existing_bundle_conflict(self):
        self.output.mkdir()
        p = self.output / 'microduck-stage-12-hil-docs-from-native-live-v1.bundle'
        p.write_bytes(b'user data')
        self.assertRefused()
        self.assertEqual(p.read_bytes(), b'user data')

    def test_path_traversal(self):
        release = copy.deepcopy(self.release)
        release['files']['../escape'] = release['files']['README.md']
        self.assertRefused(release)

    def test_existing_git_operation(self):
        (self.repo / '.git/MERGE_HEAD').write_text(self.release['parent'] + '\n')
        self.assertRefused()


if __name__ == '__main__':
    unittest.main()
