"""Local-only Stage12 archival commit; invoked by the reviewed delivery wrapper.

No model loading, dependency installation, simulation, cloud access, or push.
The wrapper supplies a SHA256-verified manifest and its exact file bytes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import traceback


def need(condition, message):
    if not condition:
        raise RuntimeError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def relative(name):
    p = PurePosixPath(name)
    need(not p.is_absolute() and str(p) == name and all(x not in ('..', '.', '') for x in p.parts),
         'Unsafe relative path: ' + name)
    return p


def safe_path(root, name):
    p = root
    for part in relative(name).parts:
        p = p / part
        need(not p.is_symlink(), 'Symlink is not an archival target: ' + str(p))
    return p


def run(args, cwd, timeout=60, accepted=(0,)):
    p = subprocess.run(args, cwd=cwd, capture_output=True, timeout=timeout)
    need(p.returncode in accepted,
         'Command failed: ' + ' '.join(map(str, args[:5])) + '\n' + p.stderr.decode(errors='replace')[-4000:])
    return p.stdout


def git(root, *args, **kwargs):
    return run(['git', *args], root, **kwargs)


def value(root, *args):
    return git(root, *args).decode().strip()


def tree(root, ref):
    result = {}
    for entry in git(root, 'ls-tree', '-r', '-z', ref).split(b'\0'):
        if entry:
            info, name = entry.split(b'\t', 1)
            mode, kind, sha = info.decode().split()
            result[name.decode()] = (mode, kind, sha)
    return result


def blob_id(data):
    return hashlib.sha1(b'blob ' + str(len(data)).encode() + b'\0' + data).hexdigest()


def check_file(path, pin):
    need(path.is_file() and not path.is_symlink(), 'Missing regular file: ' + str(path))
    need(path.stat().st_size == pin['bytes'], 'File size mismatch: ' + str(path))
    data = path.read_bytes()
    need(digest(data) == pin['sha256'], 'SHA256 mismatch: ' + str(path))
    return data


def install_new(path, data):
    """Publish a complete file without replacing an existing path, including races."""
    if path.exists():
        need(path.is_file() and path.read_bytes() == data, 'Existing file differs; preserved: ' + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.stage12-write-', dir=path.parent)
    try:
        with os.fdopen(fd, 'wb') as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            need(path.is_file() and not path.is_symlink() and path.read_bytes() == data,
                 'Concurrent conflicting write; preserved: ' + str(path))
    finally:
        os.unlink(temporary)


def changed(root, before, after):
    return {x.decode() for x in git(root, 'diff', '--name-only', '-z', before, after).split(b'\0') if x}


def preflight(root, artifacts, files, manifest, identity):
    need(root.is_dir(), 'Repository missing: ' + str(root))
    need(value(root, 'rev-parse', '--show-toplevel') == str(root), 'Not the repository root')
    need(value(root, 'rev-parse', '--show-object-format') == 'sha1', 'Unexpected Git object format')
    need(value(root, 'symbolic-ref', '--short', 'HEAD') == manifest['branch'], 'Branch must be double-balance')
    for options in (('remote', 'get-url', '--all', 'origin'), ('remote', 'get-url', '--push', '--all', 'origin')):
        need(value(root, *options).splitlines() == [manifest['origin']], 'origin must retain the established SSH URL')
    git_dir = Path(value(root, 'rev-parse', '--absolute-git-dir'))
    for marker in ('MERGE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD', 'rebase-apply', 'rebase-merge', 'index.lock'):
        need(not (git_dir / marker).exists(), 'Finish existing Git operation first: ' + marker)
    head = value(root, 'rev-parse', 'HEAD')
    git(root, 'merge-base', '--is-ancestor', manifest['expected_base'], head)
    head_tree = tree(root, head)
    marker = 'Stage12-Archive-ID: ' + identity
    own_commit = marker in value(root, 'show', '-s', '--format=%B', head).splitlines()
    parents = value(root, 'show', '-s', '--format=%P', head).split()
    if own_commit:
        need(len(parents) == 1, 'Unexpected finalization commit ancestry')
        base = parents[0]
        base_tree = tree(root, base)
        for name, data in files.items():
            need(name not in base_tree, 'Finalization must only add files: ' + name)
            need(head_tree.get(name) == ('100644', 'blob', blob_id(data)), 'Committed archive mismatch: ' + name)
    else:
        base = head
        base_tree = head_tree
        for name in files:
            need(name not in head_tree, 'Archive path already tracked; preserved: ' + name)
    runtime_changes = []
    for name, pin in manifest['source_pins'].items():
        data = check_file(safe_path(root, name), pin)
        frozen_blob = blob_id(data)
        parent_entry = base_tree.get(name)
        already_frozen = parent_entry is not None and parent_entry[0] in ('100644', '100755') and parent_entry[1:] == ('blob', frozen_blob)
        if not already_frozen:
            allowed_base = manifest.get('baseline_source_entries', {}).get(name)
            allowed_base = tuple(allowed_base) if allowed_base is not None else None
            need(name in manifest.get('approved_runtime_changes', ()) and parent_entry == allowed_base,
                 'Frozen source differs in Git HEAD without an approved baseline transition: ' + name)
            runtime_changes.append(name)
        if own_commit:
            entry = head_tree.get(name)
            need(entry is not None and entry[0] in ('100644', '100755') and entry[1:] == ('blob', frozen_blob),
                 'Committed frozen runtime differs: ' + name)
    expected_changes = set(files) | set(runtime_changes)
    if own_commit:
        need(changed(root, base, head) == expected_changes, 'Finalization commit has unexpected changes')
    for name, pin in manifest['local_artifacts'].items():
        check_file(safe_path(artifacts, name), pin)
    # A staged change belonging to this exact interrupted install can be resumed.
    staged = {x.decode() for x in git(root, 'diff', '--cached', '--name-only', '-z').split(b'\0') if x}
    need(staged <= expected_changes, 'Unrelated staged changes are preserved; unstage/commit them before finalizing')
    if own_commit:
        need(not staged, 'Unexpected staged edits after finalization commit')
    for name, data in files.items():
        path = safe_path(root, name)
        if path.exists():
            need(path.is_file() and path.read_bytes() == data, 'Existing archive file differs; preserved: ' + name)
        if name in staged:
            need(git(root, 'show', ':' + name) == data, 'Staged archival bytes differ: ' + name)
    for name in set(runtime_changes) & staged:
        need(git(root, 'show', ':' + name) == safe_path(root, name).read_bytes(), 'Staged frozen source differs: ' + name)
    ref = 'refs/tags/' + manifest['tag']
    tag_object = git(root, 'rev-parse', '--verify', '--quiet', ref, accepted=(0, 1)).decode().strip()
    if tag_object:
        need(own_commit and value(root, 'cat-file', '-t', tag_object) == 'tag' and
             value(root, 'rev-parse', ref + '^{}') == head and
             marker in value(root, 'cat-file', '-p', tag_object).splitlines(),
             'Existing stage tag conflicts; no replacement or force operation')
    return {'head': head, 'base': base, 'own_commit': own_commit, 'tag_object': tag_object,
            'frozen_runtime_paths_committed': runtime_changes,
            'source_files_verified': len(manifest['source_pins']),
            'local_artifacts_verified': len(manifest['local_artifacts'])}


def finalize(root, artifacts, files, manifest, identity, check_only=False):
    state = preflight(root, artifacts, files, manifest, identity)
    if check_only:
        return dict(state, status='READY_READ_ONLY', archive_identity=identity,
                    archive_files=len(files), git_writes=0, added_PPO_updates=0, simulation_steps=0)
    output = safe_path(artifacts, 'native-stage12-finalization-v1')
    output.mkdir(parents=True, exist_ok=True)
    lock_path = safe_path(output, 'finalization.lock')
    with lock_path.open('a+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Recheck after acquiring the local finalizer lock.
        state = preflight(root, artifacts, files, manifest, identity)
        for name, data in files.items():
            install_new(safe_path(root, name), data)
        link_output = run([sys.executable, str(root / 'scripts/check_relative_links.py'), str(root)], root).decode().strip()
        need(value(root, 'rev-parse', 'HEAD') == state['head'], 'HEAD moved during preparation; preserved')
        if not state['own_commit']:
            commit_paths = set(files) | set(state['frozen_runtime_paths_committed'])
            git(root, 'add', '--', *sorted(commit_paths))
            staged = {x.decode() for x in git(root, 'diff', '--cached', '--name-only', '-z').split(b'\0') if x}
            need(staged == commit_paths, 'Index changed during finalization; no commit created')
            index_tree = value(root, 'write-tree')
            wanted = dict(tree(root, state['head']))
            wanted.update({name: ('100644', 'blob', blob_id(data)) for name, data in files.items()})
            for name in state['frozen_runtime_paths_committed']:
                path = safe_path(root, name)
                mode = '100755' if path.stat().st_mode & 0o111 else '100644'
                wanted[name] = (mode, 'blob', blob_id(path.read_bytes()))
            need(tree(root, index_tree) == wanted, 'Index contains unexpected data; preserved')
            message = (manifest['commit_subject'] + '\n\n'
                       'Preserve the pinned runtime sources and completed native simulation workflow with selected speed +500 evidence.\n'
                       'Selection set: 253/256; no held-out, old G11 strict-physics, or hardware acceptance claim.\n\n'
                       'Stage12-Archive-ID: ' + identity + '\n')
            message_path = output / ('commit-message-' + identity[:12] + '.txt')
            install_new(message_path, message.encode())
            git(root, '-c', 'user.name=Codex', '-c', 'user.email=codex@openai.com',
                'commit', '--file', str(message_path), timeout=180)
            new_head = value(root, 'rev-parse', 'HEAD')
            need(value(root, 'show', '-s', '--format=%P', new_head) == state['head'] and
                 value(root, 'rev-parse', new_head + '^{tree}') == index_tree,
                 'Post-commit tree/parent differs; retained without a stage tag')
        state = preflight(root, artifacts, files, manifest, identity)
        if not state['tag_object']:
            tag_text = (manifest['tag'] + ': completed native simulation training, selection, fetch and video archive.\n'
                        'Best: original4500 + speed reward500; selection253/256; videos2x10s.\n'
                        'No held-out, old G11 strict-physics, or real-hardware acceptance claim.\n'
                        'Stage12-Archive-ID: ' + identity)
            git(root, '-c', 'user.name=Codex', '-c', 'user.email=codex@openai.com',
                'tag', '-a', manifest['tag'], state['head'], '-m', tag_text)
        state = preflight(root, artifacts, files, manifest, identity)
        bundle = output / 'microduck-stage-12-from-stage-11.bundle'
        if bundle.exists():
            need(bundle.is_file() and not bundle.is_symlink(), 'Invalid existing bundle path')
            refs = dict((line.split()[1], line.split()[0]) for line in git(root, 'bundle', 'list-heads', str(bundle)).decode().splitlines())
            need(refs == {'refs/heads/' + manifest['branch']: state['head'],
                          'refs/tags/' + manifest['tag']: state['tag_object']}, 'Existing bundle has different refs; preserved')
        else:
            fd, temporary = tempfile.mkstemp(prefix='.stage12-bundle-', dir=output)
            os.close(fd)
            os.unlink(temporary)
            try:
                git(root, 'bundle', 'create', temporary, 'refs/heads/' + manifest['branch'],
                    'refs/tags/' + manifest['tag'], '^' + manifest['expected_base'], timeout=180)
                os.link(temporary, bundle)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        git(root, 'bundle', 'verify', str(bundle))
        final_state = preflight(root, artifacts, files, manifest, identity)
        need(final_state['head'] == state['head'] and final_state['tag_object'] == state['tag_object'],
             'Git refs changed while creating bundle; preserved')
        return dict(final_state, status='LOCAL_STAGE12_COMPLETE_READY_TO_PUSH', archive_identity=identity,
                    archive_files=len(files), relative_links=link_output, bundle=str(bundle),
                    bundle_sha256=digest(bundle.read_bytes()), added_PPO_updates=0, simulation_steps=0,
                    cloud_connections=0, push_performed=False, power_action='NONE',
                    source_and_model_unchanged=True)


def main(files, manifest, identity):
    parser = argparse.ArgumentParser(description='Stage12本机归档、提交、注释标签和增量bundle；不推送、不运行GPU。')
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--finalize', action='store_true')
    mode.add_argument('--check', action='store_true')
    args = parser.parse_args()
    root = Path(manifest['project']).resolve()
    artifacts = Path(manifest['artifacts']).resolve()
    try:
        result = finalize(root, artifacts, files, manifest, identity, check_only=args.check)
        code = 0
    except BaseException as error:
        result = {'status': 'LOCAL_FINALIZATION_STOPPED_PRESERVE', 'error': repr(error),
                  'traceback': traceback.format_exc(), 'archive_identity': identity,
                  'added_PPO_updates': 0, 'simulation_steps': 0, 'cloud_connections': 0,
                  'push_performed': False, 'power_action': 'NONE'}
        code = 1
    result['local_return_utc'] = dt.datetime.now(dt.timezone.utc).isoformat()
    if args.check:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        report_dir = safe_path(artifacts, 'native-stage12-finalization-v1/reports')
        report_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
        report = report_dir / ('microduck-stage12-finalize-' + stamp + '-' + str(os.getpid()) + '.json')
        install_new(report, (json.dumps(result, ensure_ascii=False, indent=2) + '\n').encode())
        print('STAGE12_FINALIZATION_REPORT_SAVED=' + str(report))
        print('STATUS=' + result['status'])
        if code == 0:
            print('COMMIT=' + result['head'])
            print('TAG=' + manifest['tag'] + ' (annotated)')
            print('BUNDLE=' + result['bundle'])
            print('本机封口完成；新增PPO=0；未推送。按原规范执行：')
            print('cd ' + str(root))
            print('git push origin double-balance')
            print('git push origin stage-12-complete')
        else:
            print(result['error'])
            print('已保留现场；不要强制覆盖、重置或删除文件。请回传上述小JSON。')
    return code
