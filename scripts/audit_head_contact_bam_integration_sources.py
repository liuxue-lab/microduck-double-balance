"""Stage11 targeted installed-source capture. Standard library only, zero physics.

Run with the canonical venv Python -I -S. Do not import the packages being read.
No model compilation, graph capture, forward, step, manager or PPO is permitted.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import sys

MAX_BYTES = 2 * 1024 * 1024


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def relative_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or not p.parts or '..' in p.parts or '\\' in name:
        raise ValueError('Unsafe relative source name')
    return p


def capture_file(path, destination, expected):
    item = {'path': str(path), 'expected_sha256': expected}
    if not path.is_file():
        return {**item, 'status': 'MISSING'}
    if path.stat().st_size > MAX_BYTES:
        return {**item, 'status': 'TOO_LARGE'}
    before = path.read_bytes()
    if len(before) > MAX_BYTES:
        return {**item, 'status': 'TOO_LARGE'}
    digest = sha(before)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as f:
        f.write(before)
    after = path.read_bytes()
    syntax = True
    error = None
    try:
        if path.suffix == '.py':
            ast.parse(before, filename=str(path))
        elif path.suffix == '.json':
            json.loads(before)
    except (SyntaxError, ValueError, UnicodeError) as exc:
        syntax, error = False, str(exc)
    unchanged = before == after
    return {**item, 'status': 'MATCH' if digest == expected and syntax and unchanged else 'REVIEW_REQUIRED',
            'sha256': digest, 'bytes': len(before), 'syntax_valid': syntax,
            'syntax_error': error, 'unchanged_during_capture': unchanged}


def collect(repo, out, plan):
    repo = repo.resolve()
    out.mkdir(parents=True, exist_ok=False)
    # Explicit metadata search avoids executing site hooks / editable finders.
    sites = sorted((repo/'.venv/lib').glob('python*/site-packages'))
    found = list(importlib.metadata.distributions(path=[str(p) for p in sites]))
    names = {}
    for dist in found:
        name = (dist.metadata.get('Name') or '').lower().replace('_', '-')
        names.setdefault(name, []).append(dist)
    result = {'status': 'SOURCE_CAPTURE_REVIEW_REQUIRED', 'new_integrations': 0,
              'new_forwards': 0, 'graph_captures': 0, 'full_robot_steps': 0,
              'PPO': 0, 'cloud_hours': 0, 'formal_physics_adoption': False,
              'site_hooks_disabled': bool(sys.flags.no_site), 'isolated_python': bool(sys.flags.isolated),
              'versions': {}, 'sources': {}, 'errors': [], 'dependencies_imported': []}
    live_hashes = []
    for name, spec in plan['installed_sources'].items():
        candidates = names.get(name, [])
        if len(candidates) != 1:
            result['errors'].append(f'{name}: expected one installed metadata entry, got {len(candidates)}')
            continue
        dist = candidates[0]
        direct = dist.read_text('direct_url.json')
        origin = json.loads(direct) if direct else {}
        # Only retain the revision/editable status; never archive credentials or URLs.
        commit = origin.get('vcs_info', {}).get('commit_id')
        result['versions'][name] = {
            'version': dist.version, 'expected_version': spec['version'],
            'version_matches': dist.version == spec['version'],
            'vcs_commit': commit, 'expected_vcs_commit': spec.get('commit'),
            'vcs_commit_matches': not spec.get('commit') or commit == spec['commit'],
            'editable': bool(origin.get('dir_info', {}).get('editable', False)),
        }
        for filename, expected in spec['files'].items():
            relative_name(filename)
            key = 'installed/' + filename
            path = Path(dist.locate_file(filename))
            item = capture_file(path, out/'source'/key, expected)
            result['sources'][key] = item
            if 'sha256' in item:
                live_hashes.append((path, item['sha256']))
    for filename, expected in plan['project_sources'].items():
        relative_name(filename)
        path = repo/filename
        if not path.resolve().is_relative_to(repo):
            result['errors'].append('Project source resolves outside repository: '+filename)
            continue
        key = 'project/' + filename
        item = capture_file(path, out/'source'/key, expected)
        result['sources'][key] = item
        if 'sha256' in item:
            live_hashes.append((path, item['sha256']))
    for path, expected in live_hashes:
        if not path.is_file() or sha(path.read_bytes()) != expected:
            result['errors'].append('Source changed after capture: '+str(path))
    forbidden = {'mujoco', 'mujoco_warp', 'warp', 'torch', 'mjlab', 'bam', 'mjlab_microduck'}
    result['dependencies_imported'] = sorted(n for n in sys.modules if n.split('.')[0] in forbidden)
    result['all_source_bytes_match'] = (
        len(result['sources']) == sum(len(s['files']) for s in plan['installed_sources'].values()) + len(plan['project_sources'])
        and all(v['status'] == 'MATCH' for v in result['sources'].values()))
    result['all_versions_match'] = (
        len(result['versions']) == len(plan['installed_sources'])
        and all(v['version_matches'] and v['vcs_commit_matches'] for v in result['versions'].values()))
    result['capture_complete'] = all('sha256' in v for v in result['sources'].values()) and not result['errors']
    if result['dependencies_imported']:
        result['errors'].append('Unexpected physics dependency already imported')
    result['status'] = ('SOURCE_BYTES_MATCH_REVIEW_REQUIRED' if result['all_source_bytes_match']
                        and result['all_versions_match'] and not result['errors']
                        else 'SOURCE_DIFFERENCE_REVIEW_REQUIRED')
    write_json(out/'source-review.json', result)
    print(json.dumps({'status': result['status'], 'source_count': len(result['sources']),
                      'new_integrations': 0, 'new_forwards': 0, 'PPO': 0}, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    if not sys.flags.no_site or not sys.flags.isolated:
        raise SystemExit('Use -I -S; do not execute site hooks')
    collect(args.repo, args.out, json.loads(args.plan.read_text()))


if __name__ == '__main__':
    main()
