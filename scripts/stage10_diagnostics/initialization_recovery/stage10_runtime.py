"""Reviewed runtime identity and packaging helpers, no simulation imports."""
from __future__ import annotations
from pathlib import Path
from datetime import datetime,timezone
from importlib.metadata import version
import os,uuid,zipfile
import stage10_core as core
from stage10_core import require,read,write,sha
HERE=Path(__file__).resolve().parent

def runtime_differences(actual, expected, prefix=''):
    if isinstance(actual, dict) and isinstance(expected, dict):
        result = []
        for key in sorted(set(actual) | set(expected)):
            path = prefix + '.' + key if prefix else key
            if key not in actual or key not in expected:
                result.append({'field': path, 'actual': actual.get(key), 'expected': expected.get(key),
                               'missing_from': 'actual' if key not in actual else 'expected'})
            else:
                result.extend(runtime_differences(actual[key], expected[key], path))
        return result
    return [] if actual == expected else [{'field': prefix, 'actual': actual, 'expected': expected}]

def runtime_identity(torch, audit_path):
    # Same version source as Batch A: the loaded torch build, including +cu128.
    # Distribution metadata (e.g. 2.9.1) is separate evidence, not that build ID.
    names = ('mjlab', 'mujoco', 'mujoco-warp', 'warp-lang', 'rsl-rl-lib')
    identity = {'versions': {**{name: version(name) for name in names}, 'torch': str(torch.__version__)},
                'cuda': torch.version.cuda,
                'gpu': torch.cuda.get_device_name(0), 'num_threads': torch.get_num_threads(),
                'backend': core.backend_snapshot(torch),
                'environment': {name: os.environ.get(name) for name in
                                ('MUJOCO_GL', 'PYTHONHASHSEED', 'CUBLAS_WORKSPACE_CONFIG', 'CUDA_LAUNCH_BLOCKING')}}
    reference = read(HERE / 'reference-runtime.json')
    expected = {name: reference[name] for name in identity}
    differences = runtime_differences(identity, expected)
    write(audit_path, {'status': 'MISMATCH' if differences else 'MATCH',
                      'actual': identity, 'expected': expected, 'differences': differences,
                      'torch_distribution_version': version('torch'),
                      'torch_build_version': str(torch.__version__),
                      'comparison_version_source': 'torch.__version__, identical to Batch A',
                      'backend_settings_changed_by_check': False, 'new_ppo_updates': 0})
    summary = '; '.join(str(row['field']) + ': expected=' + repr(row['expected']) +
                        ', actual=' + repr(row['actual']) for row in differences)
    require(not differences, 'Runtime/backend differs from Batch A: ' + summary +
            '; full values saved in ' + str(audit_path))
    return identity

def module_digest_parameters(model):
    import hashlib
    digest = hashlib.sha256()
    for name, value in model.named_parameters():
        a = value.detach().cpu().numpy()
        digest.update(name.encode() + b'\0' + a.tobytes())
    return digest.hexdigest()

def package_review(folder):
    # Include every completed or partial short trace; no model copies or cloud files.
    paths = [p for p in sorted(folder.rglob('*')) if p.is_file() and
             not p.name.startswith('review-') and p != folder / 'initial-states/dev.pt']
    for p in paths:
        require(not p.is_symlink(), 'Refuse linked review input: ' + str(p))
    manifest = folder / 'review-files.json'
    write(manifest, [{'path': str(p.relative_to(folder)), 'size_bytes': p.stat().st_size, 'sha256': sha(p)} for p in paths])
    paths.append(manifest)
    groups, group, size = [], [], 0
    for p in paths:
        n = p.stat().st_size
        require(n < 48 * 1024**2, 'Review item exceeds part limit: ' + str(p))
        if group and size + n > 48 * 1024**2:
            groups.append(group)
            group, size = [], 0
        group.append(p)
        size += n
    if group:
        groups.append(group)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '-' + uuid.uuid4().hex[:6]
    outputs = []
    for number, group in enumerate(groups, 1):
        path = folder / f'review-{stamp}-part{number:02d}.zip'
        with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for p in group:
                archive.write(p, p.relative_to(folder))
        require(path.stat().st_size < 49 * 1024**2, 'Review ZIP exceeds 49 MiB')
        outputs.append({'path': str(path), 'sha256': sha(path), 'size_bytes': path.stat().st_size})
    for row in outputs:
        print('Stage10ReviewZIP=' + row['path'], flush=True)
        print('Stage10ReviewSHA256=' + row['sha256'], flush=True)
    print('Stage10ReviewParts=' + str(len(outputs)), flush=True)
    return outputs
