"""Offline fast-forward import of the Stage12 documentation release; stdlib only.

The distributed file prepends RELEASE_DATA. Importing this module does nothing.
No training, simulation, SSH, network fetch, push, cleanup, or staging occurs.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile
import zlib

DEFAULT_REPO = Path('/home/lx/microduck-double-balance/workspace')
DEFAULT_OUTPUT = Path('/home/lx/microduck-double-balance/artifacts/double-balance-stage12/github-publication-v1')
REMOTE = 'git@github.com:liuxue-lab/microduck-double-balance.git'
BRANCH = 'double-balance'
FETCH_REF = 'refs/remotes/stage12-hil-docs/double-balance'


class Refusal(RuntimeError):
    """Leave the existing checkout intact when a precondition is not met."""


def git(repo, *args, check=True):
    env = {k: v for k, v in os.environ.items() if not k.startswith('GIT_')}
    env.update(GIT_TERMINAL_PROMPT='0', GIT_OPTIONAL_LOCKS='0', LC_ALL='C')
    result = subprocess.run(
        ['git', '-c', 'core.hooksPath=/dev/null', '-c', 'core.fsmonitor=false',
         '-c', 'protocol.allow=never', '-c', 'protocol.file.allow=always',
         '-c', 'submodule.recurse=false', '-C', str(repo), *args],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, timeout=60)
    if check and result.returncode:
        raise Refusal('Git 操作未成功：' + ' '.join(args[:3]) + '\n' +
                      result.stderr.decode('utf-8', errors='replace').strip())
    return result


def value(repo, *args):
    return git(repo, *args).stdout.decode().strip()


def release_bytes(release):
    try:
        raw = zlib.decompress(base64.b85decode(release['bundle']))
    except Exception as exc:
        raise Refusal('内嵌增量包无法解码。') from exc
    if hashlib.sha256(raw).hexdigest() != release['bundle_sha256']:
        raise Refusal('内嵌增量包 SHA256 不匹配。')
    header = ('# v2 git bundle\n-' + release['parent'] +
              ' native-live-v1 prerequisite\n' + release['commit'] +
              ' refs/heads/double-balance\n' + release['tag_object'] +
              ' refs/tags/' + release['tag'] + '\n\n').encode()
    if not raw.startswith(header):
        raise Refusal('增量包引用或前置提交不匹配。')
    return raw


def valid_path(name):
    p = PurePosixPath(name)
    if p.is_absolute() or not name or str(p) != name or any(x in ('..', '.git') for x in p.parts):
        raise Refusal('发布清单中有非法路径：' + name)
    return p


def no_symlink_ancestors(path, boundary):
    current = path
    while current != boundary:
        if current.is_symlink():
            raise Refusal('路径被符号链接占用：' + str(current))
        if current.exists() and current != path and not current.is_dir():
            raise Refusal('父路径不是目录：' + str(current))
        current = current.parent
        if current == current.parent and current != boundary:
            raise Refusal('路径不在允许目录内。')


def preflight(repo, release):
    release_bytes(release)
    if not repo.is_dir() or repo.is_symlink():
        raise Refusal('未找到正常的源码仓库目录：' + str(repo))
    if Path(value(repo, 'rev-parse', '--show-toplevel')).resolve() != repo.resolve():
        raise Refusal('指定路径不是仓库根目录。')
    if value(repo, 'symbolic-ref', '--quiet', '--short', 'HEAD') != BRANCH:
        raise Refusal('当前分支不是 double-balance。')
    if value(repo, 'config', '--get-all', 'remote.origin.url').splitlines() != [REMOTE]:
        raise Refusal('origin 不是既有 GitHub SSH 远端。')
    push_urls = git(repo, 'config', '--get-all', 'remote.origin.pushurl', check=False)
    if push_urls.returncode not in (0, 1) or (push_urls.stdout and push_urls.stdout.decode().splitlines() != [REMOTE]):
        raise Refusal('origin 的推送地址与既有 SSH 远端不同。')
    for marker in ('MERGE_HEAD', 'CHERRY_PICK_HEAD', 'REVERT_HEAD', 'rebase-merge', 'rebase-apply', 'index.lock'):
        p = Path(value(repo, 'rev-parse', '--git-path', marker))
        if not p.is_absolute():
            p = repo / p
        if p.exists():
            raise Refusal('存在未结束的 Git 操作或锁：' + marker)
    head = value(repo, 'rev-parse', 'HEAD')
    if head not in (release['parent'], release['commit']):
        raise Refusal('HEAD 与固定父提交或已安装文档提交不一致；不自动回退或合并。当前：' + head)
    if git(repo, 'status', '--porcelain=v1', '-z', '--untracked-files=no').stdout:
        raise Refusal('已跟踪文件或暂存区有改动；请先审查，导入器不会 stash/reset。')
    tag = git(repo, 'rev-parse', '--verify', 'refs/tags/' + release['tag'], check=False)
    if tag.returncode == 0 and tag.stdout.decode().strip() != release['tag_object']:
        raise Refusal('同名标签指向不同对象；不会覆盖。')
    for name, pin in release['files'].items():
        p = repo.joinpath(*valid_path(name).parts)
        no_symlink_ancestors(p, repo)
        if head == release['parent'] and pin['action'] == 'add' and os.path.lexists(p):
            raise Refusal('新增路径已存在，保留原文件：' + name)
        if head == release['parent'] and pin['action'] == 'modify':
            if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest() != pin['old_sha256']:
                raise Refusal('待修改文件与固定父提交不同，保留原文件：' + name)
    # Block ignored/untracked files anywhere inside newly introduced roots.
    if head == release['parent']:
        for name in release['new_roots']:
            p = repo.joinpath(*valid_path(name).parts)
            no_symlink_ancestors(p, repo)
            if os.path.lexists(p):
                raise Refusal('新增目录已存在，保留原目录：' + name)
    if head == release['commit']:
        verify_worktree(repo, release)
    return head


def verify_objects(repo, release):
    if value(repo, 'show', '-s', '--format=%P', release['commit']) != release['parent']:
        raise Refusal('文档提交的父提交不匹配。')
    raw = git(repo, 'diff-tree', '--no-commit-id', '-r', '--name-status', '-z',
              release['parent'], release['commit']).stdout.split(b'\0')
    rows = {raw[i + 1].decode(): raw[i].decode() for i in range(0, len(raw) - 1, 2)}
    expected = {p: 'A' if pin['action'] == 'add' else 'M' for p, pin in release['files'].items()}
    if rows != expected:
        raise Refusal('增量文件范围与发布清单不一致。')
    for name, pin in release['files'].items():
        data = git(repo, 'show', release['commit'] + ':' + name).stdout
        mode = value(repo, 'ls-tree', release['commit'], '--', name).split()[0]
        if mode != '100644' or len(data) != pin['bytes'] or hashlib.sha256(data).hexdigest() != pin['sha256']:
            raise Refusal('Git 对象内容或模式不匹配：' + name)
    raw_tag = git(repo, 'cat-file', 'tag', release['tag_object']).stdout.decode()
    if not raw_tag.startswith('object ' + release['commit'] + '\ntype commit\ntag ' + release['tag'] + '\n'):
        raise Refusal('注释标签目标不匹配。')


def verify_worktree(repo, release):
    for name, pin in release['files'].items():
        p = repo / name
        no_symlink_ancestors(p, repo)
        if not p.is_file() or p.is_symlink():
            raise Refusal('导入后的文件缺失或不安全：' + name)
        raw = p.read_bytes()
        if len(raw) != pin['bytes'] or hashlib.sha256(raw).hexdigest() != pin['sha256']:
            raise Refusal('导入后文件内容不匹配：' + name)


def apply(repo, output, release):
    head = preflight(repo, release)
    raw = release_bytes(release)
    # Do not write release files into the source checkout.
    repo_real, output_real = repo.resolve(), output.resolve()
    if output_real == repo_real or repo_real in output_real.parents:
        raise Refusal('导入回执目录必须位于源码仓库外。')
    no_symlink_ancestors(output, Path(output.anchor))
    if output.exists() and not output.is_dir():
        raise Refusal('导入回执路径不是目录。')
    output.mkdir(parents=True, exist_ok=True)
    bundle = output / 'microduck-stage-12-hil-docs-from-native-live-v1.bundle'
    if os.path.lexists(bundle):
        if bundle.is_symlink() or not bundle.is_file() or bundle.read_bytes() != raw:
            raise Refusal('已有同名 bundle 内容不同，未覆盖。')
    else:
        with bundle.open('xb') as stream:
            stream.write(raw)
    git(repo, 'bundle', 'verify', str(bundle))
    git(repo, 'fetch', '--no-tags', '--no-recurse-submodules', '--no-write-fetch-head',
        str(bundle), 'refs/heads/double-balance:' + FETCH_REF)
    verify_objects(repo, release)
    # Recheck after object import, before changing the user's checkout.
    if preflight(repo, release) != head:
        raise Refusal('导入期间 HEAD 改变，停止更新工作树。')
    if head == release['parent']:
        git(repo, 'merge', '--ff-only', '--no-edit', release['commit'])
    verify_worktree(repo, release)
    tagref = 'refs/tags/' + release['tag']
    tag = git(repo, 'rev-parse', '--verify', tagref, check=False)
    if tag.returncode:
        git(repo, 'update-ref', tagref, release['tag_object'], '0' * 40)
    elif tag.stdout.decode().strip() != release['tag_object']:
        raise Refusal('标签在导入期间改变，未覆盖。文档提交已导入，请保留现场。')
    result = {'status': 'ALREADY_INSTALLED' if head == release['commit'] else 'INSTALLED',
              'parent': release['parent'], 'commit': release['commit'], 'tag': release['tag'],
              'tag_object': release['tag_object'], 'bundle_sha256': release['bundle_sha256'],
              'file_count': len(release['files']), 'remote_push_performed': False,
              'HIL_started': False, 'PPO_updates': 0}
    with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', suffix='.json',
                                     prefix='install-receipt-', dir=output, delete=False) as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--apply', action='store_true', help='核对并导入文档增量提交；不推送')
    modes.add_argument('--check', action='store_true', help='只读检查本机前置条件')
    modes.add_argument('--status', action='store_true', help='显示预期版本和本机 HEAD')
    args = parser.parse_args(argv)
    if not any((args.apply, args.check, args.status)):
        parser.print_help()
        return 0
    release = globals().get('RELEASE_DATA')
    if not release:
        parser.error('请使用已生成且含增量包的单文件发布器。')
    try:
        if args.status:
            release_bytes(release)
            print(json.dumps({'expected_parent': release['parent'], 'expected_commit': release['commit'],
                              'tag': release['tag'], 'current_head': value(DEFAULT_REPO, 'rev-parse', 'HEAD')}, indent=2))
        elif args.check:
            head = preflight(DEFAULT_REPO, release)
            print('CHECK_PASSED: ' + head + '；未修改仓库。')
        else:
            result = apply(DEFAULT_REPO, DEFAULT_OUTPUT, release)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            print('本机文档导入完成。下一步由用户推送：')
            print('git -C ' + str(DEFAULT_REPO) + ' push origin double-balance')
            print('git -C ' + str(DEFAULT_REPO) + ' push origin ' + release['tag'])
        return 0
    except (Refusal, OSError, subprocess.TimeoutExpired) as exc:
        print('STOP: ' + str(exc), file=sys.stderr)
        print('已停止；没有执行推送、训练或 HIL。请保留当前文件和终端输出。', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
