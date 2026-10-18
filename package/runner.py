"""Local-only launch, bounded progress monitor, and explicit Git SSH return."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import tempfile
import time

ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT / 'probe-plan.json').read_text())
CANONICAL = Path('/home/lx/microduck-double-balance/workspace')
REMOTE = 'git@github.com:liuxue-lab/microduck-double-balance.git'
ZERO = dict(physics_integrations=0, simulator_forwards=0, PPO=0, cloud_training=0,
            automatic_push=False, cloud_gpu_allowed=False, production_integration_authorized=False)
EVIDENCE = ('started.json', 'progress.json', 'finished.json', 'worker.log', 'result.json',
            'warp-pins-before.json', 'warp-pins-after.json', 'original-values.npz',
            'parallel-values-1.npz', 'parallel-values-2.npz', 'benchmark-timings.json',
            'parity-failure.json', 'parity-failure.npz')


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def write(path, obj, exclusive=False):
    raw = json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if exclusive:
        with path.open('x') as handle:
            handle.write(raw)
    else:
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(raw)
        tmp.replace(path)


def verify_package():
    manifest = read(ROOT / 'MANIFEST.json')
    actual = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()}
    require(actual == set(manifest['files']) | {'MANIFEST.json'}, 'Package file set differs')
    for name, pin in manifest['files'].items():
        path = ROOT / name
        require(not path.is_symlink() and path.stat().st_size == pin['bytes']
                and sha(path) == pin['sha256'], 'Package file changed: ' + name)


def git(folder, *args, timeout=30, check=True):
    result = subprocess.run(['git', '-C', str(folder), '-c', 'core.hooksPath=/dev/null',
                             '-c', 'commit.gpgsign=false', '-c', 'core.attributesFile=/dev/null',
                             '-c', 'core.autocrlf=false', *args], capture_output=True,
                            text=True, timeout=timeout)
    if check:
        require(result.returncode == 0, 'Git failed: ' + result.stderr.strip())
    return result.stdout.strip() if check else result


def existing_return(output, meta):
    repo = output / 'return' / 'git'
    require(git(repo, 'remote', 'get-url', 'origin') == REMOTE, 'Return remote differs')
    require(git(repo, 'rev-parse', 'HEAD') == meta['commit'], 'Return commit changed')
    require(not git(repo, 'status', '--porcelain'), 'Return tree changed')
    require(sha(repo / 'RETURN-INDEX.json') == meta['evidence_index_sha256'], 'Return index changed')
    index = read(repo / 'RETURN-INDEX.json')
    require(set(git(repo, 'ls-tree', '-r', '--name-only', 'HEAD').splitlines()) ==
            set(index['files']) | {'RETURN-INDEX.json'}, 'Committed return file set differs')
    require(index['package_manifest_sha256'] == sha(ROOT / 'MANIFEST.json'), 'Return package changed')
    for name, pin in index['files'].items():
        source = ROOT / name[8:] if name.startswith('package/') else output / name[9:]
        require(source.is_file() and not source.is_symlink() and sha(source) == pin['sha256']
                and source.stat().st_size == pin['bytes'], 'Original return input changed: ' + name)
    return meta


def prepare_return(output):
    verify_package()
    require((output / 'finished.json').is_file(), 'Worker exit not recorded; preserve this slot, do not rerun')
    finished = read(output / 'finished.json')
    require(finished.get('child_reaped') is True, 'Worker exit is uncertain; preserve existing data')
    target = output / 'return'
    if (target / 'LOCAL-RETURN.json').is_file():
        return existing_return(output, read(target / 'LOCAL-RETURN.json'))
    require(git(CANONICAL, 'remote', 'get-url', 'origin') == REMOTE, 'Canonical Git SSH remote differs')
    identity = {key: git(CANONICAL, 'config', '--get', key) for key in ('user.name', 'user.email')}
    require(all(identity.values()), 'Existing local Git identity missing')
    ssh = git(CANONICAL, 'config', '--get', 'core.sshCommand', check=False)
    require(ssh.returncode in (0, 1), 'Cannot read existing Git SSH configuration')
    if ssh.returncode == 0 and ssh.stdout.strip():
        identity['core.sshCommand'] = ssh.stdout.strip()
    entries = [(p, 'package/' + p.relative_to(ROOT).as_posix())
               for p in sorted(ROOT.rglob('*')) if p.is_file()]
    entries += [(output / name, 'evidence/' + name) for name in EVIDENCE if (output / name).is_file()]
    require(sum(p.stat().st_size for p, _ in entries) <= 32 * 1024**2, 'Return exceeds fixed 32 MiB cap')
    require(all(not p.is_symlink() for p, _ in entries), 'Symlink in return inputs')
    pins = {name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p, name in entries}
    digest = hashlib.sha256(json.dumps(pins, sort_keys=True).encode()).hexdigest()
    if target.exists():
        marker = read(target / 'TRANSFER-IDENTITY.json')
        require(marker['input_digest'] == digest, 'Partial transfer inputs differ; preserve old return')
    else:
        target.mkdir()
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ').lower()
        marker = dict(input_digest=digest, branch='stage12-geometry-parallel-v1-return-' + stamp + '-' + digest[:10])
        write(target / 'TRANSFER-IDENTITY.json', marker, exclusive=True)
    repo = target / 'git'
    repo.mkdir(exist_ok=True)
    git(repo, 'init', '--initial-branch=' + marker['branch'])
    for name, value in identity.items():
        git(repo, 'config', name, value)
    git(repo, 'config', 'core.hooksPath', '/dev/null')
    git(repo, 'config', 'commit.gpgsign', 'false')
    git(repo, 'config', 'core.autocrlf', 'false')
    if 'origin' in git(repo, 'remote').splitlines():
        require(git(repo, 'remote', 'get-url', 'origin') == REMOTE, 'Return remote changed')
    else:
        git(repo, 'remote', 'add', 'origin', REMOTE)
    for path, name in entries:
        destination = repo / name
        if destination.exists():
            require(sha(destination) == pins[name]['sha256'], 'Partial transfer bytes differ: ' + name)
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(path.read_bytes())
    index = dict(format='stage12-geometry-probe-return-v1', branch=marker['branch'], files=pins,
                 package_manifest_sha256=sha(ROOT / 'MANIFEST.json'),
                 source_return_commit=PLAN['return_commit'],
                 result=read(output / 'result.json') if (output / 'result.json').is_file() else None,
                 finished=finished, **ZERO)
    if (repo / 'RETURN-INDEX.json').exists():
        require(read(repo / 'RETURN-INDEX.json') == index, 'Partial transfer index differs')
    else:
        write(repo / 'RETURN-INDEX.json', index, exclusive=True)
    git(repo, 'add', '-f', '--', 'RETURN-INDEX.json', 'package', 'evidence')
    if git(repo, 'diff', '--cached', '--name-only'):
        git(repo, 'commit', '-m', 'Stage12 local geometry-only bounded probe; independent review required')
    meta = dict(status='LOCAL_GIT_RETURN_PREPARED_NOT_PUSHED', branch=marker['branch'],
                commit=git(repo, 'rev-parse', 'HEAD'), directory=str(repo),
                evidence_index_sha256=sha(repo / 'RETURN-INDEX.json'), **ZERO)
    write(target / 'LOCAL-RETURN.json', meta, exclusive=True)
    return existing_return(output, meta)


def stop_owned_child(child):
    if child.poll() is None:
        try:
            os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            return child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return child.wait(timeout=5)
    return child.returncode


def show(output):
    for name in ('progress.json', 'result.json', 'finished.json', 'return/LOCAL-RETURN.json', 'return/PUSHED.json'):
        path = output / name
        if path.is_file():
            print(name + '=' + json.dumps(read(path), ensure_ascii=False, sort_keys=True), flush=True)
    if not (output / 'started.json').is_file():
        print('LOCAL_GEOMETRY_NOT_STARTED | 积分=0 | PPO=0', flush=True)


def launch(output, launcher):
    verify_package()
    require(Path.home() == Path('/home/lx') and os.geteuid() != 0,
            '请在 lx 本机运行；拒绝云端或 root 执行')
    require(output == ROOT.parent, 'Output/package scope differs')
    require(not (output / 'started.json').exists(), '此槽已使用，绝不重跑；用 --status 或 --push-return')
    python = Path(PLAN['runtime_python'])
    require(python.is_file(), '本机固定 Python 环境不存在；未安装或修改环境')
    require(os.statvfs(output).f_bavail * os.statvfs(output).f_frsize >= 256 * 1024**2,
            'Local disk reserve below 256 MiB')
    cache = Path(tempfile.mkdtemp(prefix='md12-geometry-', dir='/tmp'))
    (cache / 'cuda').mkdir()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               PYTHONNOUSERSITE='1', MD12_GEOMETRY_PRIVATE_CACHE=str(cache),
               CUDA_CACHE_PATH=str(cache / 'cuda'), TMPDIR=str(cache),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
    env.pop('PYTHONPATH', None)
    started = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   max_seconds=PLAN['max_seconds'], package_manifest_sha256=sha(ROOT / 'MANIFEST.json'),
                   private_cache=str(cache), **ZERO)
    write(output / 'started.json', started, exclusive=True)
    print('LOCAL_GEOMETRY_START | 仅本机 RTX 5060 | 最多300秒 | 物理积分=0 | PPO=0', flush=True)
    child, reason, child_reaped = None, 'worker-exit', False
    code = None
    t0 = time.monotonic()
    try:
        with (output / 'worker.log').open('x') as log:
            child = subprocess.Popen([str(python), '-B', '-s', '-u', str(ROOT / 'worker.py'),
                                      '--output', str(output)], cwd=str(ROOT), env=env,
                                     stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            next_print = t0
            while child.poll() is None:
                now = time.monotonic()
                if now-t0 >= PLAN['max_seconds']:
                    reason = '300-second-limit'
                    break
                if now >= next_print:
                    state = read(output / 'progress.json') if (output / 'progress.json').is_file() else {}
                    print('{phase} | 当前阶段 {done}/{total} | 原版查询 {original}/162 | 并行查询 {candidate}/12090 | {seconds:.0f}/300秒 | 积分=0'.format(
                        phase=state.get('phase', 'starting'), done=state.get('done', 0), total=state.get('total', 0),
                        original=state.get('original_queries_completed', 0),
                        candidate=state.get('candidate_queries_completed', 0), seconds=now-t0), flush=True)
                    next_print = now + 5
                time.sleep(min(0.5, max(0, PLAN['max_seconds']-(now-t0))))
            code = stop_owned_child(child)
            child_reaped = True
    except KeyboardInterrupt:
        reason = 'user-interrupt'
    except Exception as error:
        reason = 'launcher-error: ' + str(error)
    finally:
        if child is not None and not child_reaped:
            code = stop_owned_child(child)
            child_reaped = True
        if child is None:
            child_reaped = True
        write(output / 'finished.json', dict(reason=reason, child_exit_code=code,
              child_reaped=child_reaped, elapsed_seconds=time.monotonic()-t0, **ZERO), exclusive=True)
    result = read(output / 'result.json') if (output / 'result.json').is_file() else {}
    print('LOCAL_GEOMETRY_FINISHED | ' + result.get('status', reason), flush=True)
    if result.get('benchmark'):
        latency = result['benchmark']['query_and_readback_seconds']
        print('单次几何查询中位耗时：原版 %.3f ms；并行版 %.3f ms；倍率 %.3f x。完整仿真尚未测试。' % (
              latency['median']['original']*1000, latency['median']['candidate']*1000, latency['speedup']), flush=True)
    if code != 0 or reason != 'worker-exit':
        if (output / 'worker.log').is_file():
            print('\n'.join((output / 'worker.log').read_text(errors='replace').splitlines()[-35:]), flush=True)
    try:
        meta = prepare_return(output)
        print('GEOMETRY_RETURN_PREPARED=' + json.dumps(meta, ensure_ascii=False, sort_keys=True), flush=True)
    except Exception as error:
        print('RETURN_PREPARATION_PENDING | ' + str(error) + ' | 已有结果保留，回传可单独重试', flush=True)
    print('手动回传（可重复；不会重跑测试）：\npython3 ' + shlex.quote(str(launcher)) + ' --push-return', flush=True)
    return 0 if code == 0 and reason == 'worker-exit' else 1


def push_return(output):
    meta = prepare_return(output)
    repo = output / 'return' / 'git'
    # This is the only network write. Called only for the user's explicit --push-return.
    print('GEOMETRY_GIT_SSH_PUSH | 固定提交=' + meta['commit'] + ' | 积分=0', flush=True)
    command = ['git', '-C', str(repo), '-c', 'core.hooksPath=/dev/null', 'push',
               'origin', meta['commit'] + ':refs/heads/' + meta['branch']]
    result = subprocess.run(command)
    require(result.returncode == 0, '回传失败；重复 --push-return 即可，绝不重跑测试')
    refs = git(repo, 'ls-remote', '--exit-code', 'origin', 'refs/heads/' + meta['branch'], timeout=60)
    require(refs.split()[0] == meta['commit'], 'Remote commit differs after push')
    receipt = dict(**meta, push_status='COMPLETE', remote_commit_verified=True)
    write(output / 'return' / 'PUSHED.json', receipt)
    print('Stage12GeometryPush=COMPLETE\nGEOMETRY_RETURN_JSON=' + json.dumps(receipt, sort_keys=True), flush=True)
    return 0
