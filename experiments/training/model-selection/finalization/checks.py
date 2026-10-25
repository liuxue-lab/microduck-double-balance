"""Real temporary Git repositories; synthetic source/model bytes; no GPU/network."""
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('finalizer_engine', HERE / 'engine.py')
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)
PASSED = []


def mark(name):
    PASSED.append(name)


def command(root, *args):
    return subprocess.check_output(['git', *args], cwd=root, stderr=subprocess.STDOUT).decode().strip()


def pin(data):
    return dict(bytes=len(data), sha256=hashlib.sha256(data).hexdigest())


@contextlib.contextmanager
def fixture():
    with tempfile.TemporaryDirectory(prefix='microduck-stage12-finalize-test-') as directory:
        root = Path(directory) / 'repo'
        root.mkdir()
        artifacts = Path(directory) / 'artifacts'
        artifacts.mkdir()
        command(root, 'init', '-b', 'double-balance')
        command(root, 'config', 'user.name', 'Fixture')
        command(root, 'config', 'user.email', 'fixture@example.invalid')
        command(root, 'config', 'commit.gpgsign', 'false')
        command(root, 'config', 'tag.gpgsign', 'false')
        command(root, 'remote', 'add', 'origin', 'git@github.com:liuxue-lab/microduck-double-balance.git')
        (root / 'scripts').mkdir()
        checker = b"print('PASS: fixture link check')\n"
        (root / 'scripts/check_relative_links.py').write_bytes(checker)
        (root / 'source.py').write_text('frozen = 1\n')
        (artifacts / 'model.pt').write_bytes(b'synthetic model bytes, never loaded')
        command(root, 'add', '.')
        command(root, 'commit', '-m', 'fixture base')
        base = command(root, 'rev-parse', 'HEAD')
        files = {'experiments/stage12/native-workflow/evidence.json': b'{"synthetic":true}\n',
                 'docs/handoffs/stage12.md': b'# Synthetic archive\n'}
        manifest = dict(expected_base=base, branch='double-balance', tag='stage-12-complete',
                        origin='git@github.com:liuxue-lab/microduck-double-balance.git',
                        source_pins={'source.py': pin((root / 'source.py').read_bytes()),
                                     'scripts/check_relative_links.py': pin(checker)},
                        local_artifacts={'model.pt': pin((artifacts / 'model.pt').read_bytes())},
                        commit_subject='Fixture archive')
        identity = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        yield root, artifacts, files, manifest, identity


def invoke(f, check=False):
    return engine.finalize(*f, check_only=check)


def expect_failure(f, text):
    head = command(f[0], 'rev-parse', 'HEAD')
    try:
        invoke(f)
    except (RuntimeError, BlockingIOError) as e:
        assert text in str(e), str(e)
    else:
        raise AssertionError('Expected failure: ' + text)
    assert command(f[0], 'rev-parse', 'HEAD') == head


def main():
    with fixture() as f:
        before = command(f[0], 'status', '--porcelain=v1')
        r = invoke(f, check=True)
        assert r['status'] == 'READY_READ_ONLY'
        assert command(f[0], 'status', '--porcelain=v1') == before
        assert not (f[1] / 'native-stage12-finalization-v1').exists()
        mark('read_only_preflight_writes_nothing')
        r = invoke(f)
        assert r['status'] == 'LOCAL_STAGE12_COMPLETE_READY_TO_PUSH'
        assert command(f[0], 'cat-file', '-t', 'refs/tags/stage-12-complete') == 'tag'
        assert command(f[0], 'status', '--porcelain=v1') == ''
        assert command(f[0], 'rev-list', '--count', 'HEAD') == '2'
        mark('one_commit_annotated_tag_and_incremental_bundle')
        r2 = invoke(f)
        assert (r['head'], r['tag_object'], r['bundle_sha256']) == (r2['head'], r2['tag_object'], r2['bundle_sha256'])
        assert command(f[0], 'rev-list', '--count', 'HEAD') == '2'
        mark('completed_rerun_is_idempotent')
        # Import the bundle into a checkout containing only its prerequisite.
        clone = f[0].parent / 'import-check'
        subprocess.check_output(['git', 'clone', '--quiet', '--no-local', str(f[0]), str(clone)])
        command(clone, 'reset', '--hard', f[3]['expected_base'])
        command(clone, 'tag', '-d', 'stage-12-complete')
        command(clone, 'bundle', 'verify', r['bundle'])
        command(clone, 'fetch', r['bundle'], 'refs/heads/double-balance:refs/remotes/stage12/double-balance',
                'refs/tags/stage-12-complete:refs/tags/stage-12-complete')
        command(clone, 'merge', '--ff-only', 'refs/remotes/stage12/double-balance')
        assert command(clone, 'rev-parse', 'HEAD') == r['head']
        mark('bundle_import_fast_forward_matches_commit_and_tag')

    with fixture() as f:
        (f[0] / 'my-untracked.txt').write_text('leave this alone\n')
        first = next(iter(f[2])); p = f[0] / first
        p.parent.mkdir(parents=True); p.write_bytes(f[2][first])
        command(f[0], 'add', '--', first)
        r = invoke(f)
        assert (f[0] / 'my-untracked.txt').read_text() == 'leave this alone\n'
        assert command(f[0], 'status', '--porcelain=v1') == '?? my-untracked.txt'
        assert 'my-untracked.txt' not in command(f[0], 'diff-tree', '--no-commit-id', '--name-only', '-r', r['head'])
        mark('partial_install_and_owned_staging_resume_unrelated_file_preserved')

    for case, prepare, message in (
        ('wrong_branch', lambda f: command(f[0], 'switch', '-c', 'other'), 'Branch must'),
        ('wrong_origin', lambda f: command(f[0], 'remote', 'set-url', 'origin', 'https://example.invalid/repo'), 'origin must'),
        ('wrong_model', lambda f: (f[1] / 'model.pt').write_bytes(b'wrong'), 'File size mismatch'),
        ('source_modified', lambda f: (f[0] / 'source.py').write_text('frozen = 2\n'), 'SHA256 mismatch'),
        ('conflicting_lightweight_tag', lambda f: command(f[0], 'tag', 'stage-12-complete'), 'Existing stage tag conflicts'),
        ('merge_in_progress', lambda f: (f[0] / '.git/MERGE_HEAD').write_text(f[3]['expected_base']), 'Finish existing Git operation'),
    ):
        with fixture() as f:
            prepare(f)
            expect_failure(f, message)
            assert not (f[0] / next(iter(f[2]))).exists()
            mark(case + '_stops_before_archival_write')

    with fixture() as f:
        (f[0] / 'mine.txt').write_text('staged by user')
        command(f[0], 'add', 'mine.txt')
        before = command(f[0], 'write-tree')
        expect_failure(f, 'Unrelated staged changes')
        assert command(f[0], 'write-tree') == before
        mark('unrelated_index_is_not_changed_or_committed')

    with fixture() as f:
        p = f[0] / next(iter(f[2])); p.parent.mkdir(parents=True); p.write_text('existing user bytes')
        expect_failure(f, 'Existing archive file differs')
        assert p.read_text() == 'existing user bytes'
        mark('conflicting_target_is_not_overwritten')

    with fixture() as f:
        outside = f[0].parent / 'outside'; outside.mkdir()
        (f[0] / 'experiments').symlink_to(outside, target_is_directory=True)
        expect_failure(f, 'Symlink is not an archival target')
        assert not list(outside.iterdir())
        mark('symlink_target_cannot_escape_repo')

    with fixture() as f:
        (f[0] / 'source.py').write_text('frozen = 2\n')
        command(f[0], 'add', 'source.py'); command(f[0], 'commit', '-m', 'different committed source')
        (f[0] / 'source.py').write_text('frozen = 1\n')
        expect_failure(f, 'Frozen source differs in Git HEAD')
        mark('matching_worktree_cannot_hide_different_committed_runtime')

    with fixture() as f:
        (f[0] / 'unrelated-doc.txt').write_text('prior user commit')
        command(f[0], 'add', 'unrelated-doc.txt'); command(f[0], 'commit', '-m', 'prior user docs')
        before = command(f[0], 'rev-parse', 'HEAD')
        r = invoke(f)
        assert command(f[0], 'show', '-s', '--format=%P', r['head']) == before
        assert (f[0] / 'unrelated-doc.txt').read_text() == 'prior user commit'
        mark('descendant_with_identical_runtime_is_preserved')

    with fixture() as f:
        root, artifacts, files, manifest, identity = f
        base_tree = engine.tree(root, manifest['expected_base'])
        manifest['approved_runtime_changes'] = ['source.py', 'scripts/retained.py']
        manifest['baseline_source_entries'] = {'source.py': base_tree['source.py'], 'scripts/retained.py': None}
        (root / 'source.py').write_text('frozen = 2\n')
        (root / 'scripts/retained.py').write_text('retained = True\n')
        for name in manifest['approved_runtime_changes']:
            manifest['source_pins'][name] = pin((root / name).read_bytes())
        command(root, 'add', 'source.py')  # Exact known pending runtime may already be staged.
        before = {n:(root/n).read_bytes() for n in manifest['approved_runtime_changes']}
        r = invoke(f)
        assert set(r['frozen_runtime_paths_committed']) == set(before)
        for name, data in before.items():
            assert (root/name).read_bytes() == data
            assert engine.git(root, 'show', 'HEAD:' + name) == data
        assert invoke(f)['head'] == r['head']
        mark('known_missing_and_modified_runtime_committed_without_rewriting_bytes')
        mark('runtime_integration_rerun_preserves_same_commit')

    with fixture() as f:
        root, artifacts, files, manifest, identity = f
        base_entry = engine.tree(root, manifest['expected_base'])['source.py']
        manifest['approved_runtime_changes'] = ['source.py']
        manifest['baseline_source_entries'] = {'source.py': base_entry}
        (root/'source.py').write_text('unknown = 42\n')
        command(root, 'add', 'source.py'); command(root, 'commit', '-m', 'different runtime history')
        (root/'source.py').write_text('frozen = 2\n')
        manifest['source_pins']['source.py'] = pin((root/'source.py').read_bytes())
        expect_failure(f, 'without an approved baseline transition')
        mark('approved_path_does_not_allow_unknown_committed_source_overwrite')

    for operation, test_name in (('tag', 'resume_after_commit_before_tag'), ('bundle', 'resume_after_tag_before_bundle')):
        with fixture() as f:
            original_git = engine.git
            tripped = []
            def failing_git(root, *args, **kwargs):
                matches = ('tag' in args and '-a' in args) if operation == 'tag' else args[:2] == ('bundle', 'create')
                if matches and not tripped:
                    tripped.append(True)
                    raise RuntimeError('simulated interruption')
                return original_git(root, *args, **kwargs)
            engine.git = failing_git
            try:
                try: invoke(f)
                except RuntimeError as e: assert 'simulated interruption' in str(e)
                else: raise AssertionError('fault did not run')
            finally:
                engine.git = original_git
            head = command(f[0], 'rev-parse', 'HEAD')
            r = invoke(f)
            assert r['head'] == head
            assert command(f[0], 'rev-list', '--count', 'HEAD') == '2'
            mark(test_name)

    with fixture() as f:
        hook = f[0] / '.git/hooks/pre-commit'
        hook.write_text('#!/bin/sh\nexit 1\n'); hook.chmod(0o755)
        expect_failure(f, 'Command failed')
        assert not command(f[0], 'tag', '-l', 'stage-12-complete')
        hook.unlink()  # Fixture-only failing hook; production never changes hooks.
        invoke(f)
        assert command(f[0], 'rev-list', '--count', 'HEAD') == '2'
        mark('git_hook_failure_preserves_index_for_exact_retry')

    with fixture() as f:
        out = f[1] / 'native-stage12-finalization-v1'; out.mkdir()
        with (out / 'finalization.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            expect_failure(f, 'Resource temporarily unavailable')
        mark('concurrent_finalizer_cannot_mutate_repo')

    receipt = dict(status='PASS', count=len(PASSED), checks=PASSED,
                   engine_sha256=hashlib.sha256((HERE / 'engine.py').read_bytes()).hexdigest(),
                   scope='Real disposable Git fixtures; synthetic model bytes; no user repository, cloud, Torch, CUDA, PPO or simulation accessed')
    (HERE / 'offline-review.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
