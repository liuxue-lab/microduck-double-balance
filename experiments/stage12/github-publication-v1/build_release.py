"""Build the pinned documentation-only Git bundle in an isolated object store.

The store must already contain the fixed parent commit and its ancestor tree
objects. No checkout, network operation, training, or push is performed.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zlib

PARENT = '056657aa36158fb81dca6fd2092e525222d6e632'
PARENT_TREE = '26d5d49bcc093858abb71340059429e688ad672e'
TAG = 'stage-12-hil-docs-20261025'
SCRIPT = 'microduck-stage12-github-publication-20261025.py'
BUNDLE = 'microduck-stage-12-hil-docs-from-native-live-v1.bundle'
NEW_ROOTS = ['docs/double-balance', 'experiments/double-balance',
             'experiments/stage12/github-publication-v1']
COMMIT_DATE = '2026-10-25T10:00:00+00:00'


def build(gitdir, source, output):
    objects = set()

    def git(*args, data=None, env=None):
        return subprocess.check_output(['git', '--git-dir', str(gitdir), *args], input=data, env=env)

    def put(kind, data):
        sha = git('hash-object', '-t', kind, '-w', '--stdin', data=data).decode().strip()
        objects.add(sha)
        return sha

    def tree(rows):
        rows.sort(key=lambda r: (r[1] + ('/' if r[0] == '40000' else '')).encode())
        return put('tree', b''.join((mode + ' ' + name).encode() + b'\0' + bytes.fromhex(sha)
                                   for mode, name, sha in rows))

    def entries(sha):
        raw = git('cat-file', 'tree', sha)
        rows = []
        while raw:
            header, raw = raw.split(b'\0', 1)
            mode, name = header.decode().split(' ', 1)
            rows.append((mode, name, raw[:20].hex()))
            raw = raw[20:]
        return rows

    if git('show', '-s', '--format=%T', PARENT).decode().strip() != PARENT_TREE:
        raise RuntimeError('Parent tree mismatch')
    paths = [source / 'README.md']
    for name in NEW_ROOTS:
        folder = source / name
        if not folder.is_dir() or folder.is_symlink():
            raise RuntimeError('Missing or unsafe new source directory: ' + name)
        existing = subprocess.run(['git', '--git-dir', str(gitdir), 'cat-file', '-e', PARENT + ':' + name],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if existing.returncode == 0:
            raise RuntimeError('New directory already exists in parent: ' + name)
        paths.extend(p for p in folder.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc')
    changes, pins = {}, {}
    for p in sorted(paths):
        if p.is_symlink() or any(q.is_symlink() for q in p.parents if q != source):
            raise RuntimeError('Symlink in source')
        name = p.relative_to(source).as_posix()
        data = p.read_bytes()
        if len(data) > 2 * 1024 * 1024:
            raise RuntimeError('Unexpected large source file: ' + name)
        changes[name] = put('blob', data)
        pins[name] = {'action': 'modify' if name == 'README.md' else 'add', 'bytes': len(data),
                      'sha256': hashlib.sha256(data).hexdigest()}
        if name == 'README.md':
            pins[name]['old_sha256'] = hashlib.sha256(git('show', PARENT + ':README.md')).hexdigest()

    def update(old, subset):
        rows = {name: (mode, sha) for mode, name, sha in entries(old)} if old else {}
        groups = {}
        for name, sha in subset.items():
            first, sep, rest = name.partition('/')
            if sep:
                groups.setdefault(first, {})[rest] = sha
            else:
                rows[first] = ('100644', sha)
        for name, group in groups.items():
            prior = rows.get(name)
            if prior and prior[0] != '40000':
                raise RuntimeError('Non-directory parent: ' + name)
            rows[name] = ('40000', update(prior[1] if prior else None, group))
        return tree([(mode, name, sha) for name, (mode, sha) in rows.items()])

    final_tree = update(PARENT_TREE, changes)
    env = dict(os.environ, GIT_AUTHOR_NAME='Codex', GIT_AUTHOR_EMAIL='codex@openai.com',
               GIT_COMMITTER_NAME='Codex', GIT_COMMITTER_EMAIL='codex@openai.com',
               GIT_AUTHOR_DATE=COMMIT_DATE, GIT_COMMITTER_DATE=COMMIT_DATE)
    commit = git('commit-tree', final_tree, '-p', PARENT, env=env, data=(
        'Document completed double-ball HIL results and reproducible evidence\n\n'
        'Add a project entry, bounded acceptance report, original-environment guide,\n'
        'archive scope, pinned evidence and an offline documentation importer.\n'
        'Preserve the accepted runtime, policy, upstream attribution and old tags.\n').encode()).decode().strip()
    objects.add(commit)
    timestamp = git('show', '-s', '--format=%ct', commit).decode().strip()
    tag_data = (f'object {commit}\ntype commit\ntag {TAG}\n'
                f'tagger Codex <codex@openai.com> {timestamp} +0000\n\n'
                'Stage12 HIL documentation archive.\n'
                'Two bounded simulation HIL cases; no physical robot or full backup acceptance.\n').encode()
    tag = git('mktag', data=tag_data).decode().strip()
    objects.add(tag)
    pack = git('pack-objects', '--stdout', '--delta-base-offset', '--no-reuse-delta',
               data=('\n'.join(sorted(objects)) + '\n').encode())
    header = (f'# v2 git bundle\n-{PARENT} native-live-v1 prerequisite\n'
              f'{commit} refs/heads/double-balance\n{tag} refs/tags/{TAG}\n\n').encode()
    bundle = header + pack
    release = {'schema': 'microduck-stage12-github-publication-v1', 'parent': PARENT,
               'commit': commit, 'tree': final_tree, 'tag': TAG, 'tag_object': tag,
               'bundle_sha256': hashlib.sha256(bundle).hexdigest(), 'files': pins,
               'new_roots': NEW_ROOTS, 'bundle': base64.b85encode(zlib.compress(bundle, 9)).decode()}
    output.mkdir(parents=True, exist_ok=True)
    (output / BUNDLE).write_bytes(bundle)
    installer = source / 'experiments/stage12/github-publication-v1/install.py'
    script = output / SCRIPT
    script.write_text('#!/usr/bin/env python3\nRELEASE_DATA = ' + repr(release) + '\n' +
                      installer.read_text().replace('from __future__ import annotations\n', ''), encoding='utf-8')
    receipt = {k: v for k, v in release.items() if k != 'bundle'}
    receipt.update(script_name=SCRIPT, script_bytes=script.stat().st_size,
                   script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(), bundle_name=BUNDLE,
                   bundle_bytes=len(bundle), object_count=len(objects), remote_push_performed=False)
    (output / 'build-receipt.json').write_text(json.dumps(receipt, indent=2, ensure_ascii=False) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--git-dir', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = build(args.git_dir.resolve(), args.source.resolve(), args.output.resolve())
    print(json.dumps({k: result[k] for k in ('parent', 'commit', 'tag', 'script_sha256', 'bundle_sha256', 'object_count')}, indent=2))
