"""Per-command APT source copies; exclude only the observed failing Code repo.

Never edit the original source files or weaken APT signature/TLS checks.
Keep the system's package-list cache (including excluded sources) in place.
"""
import hashlib
from pathlib import Path
import re
import shlex
import subprocess
from urllib.parse import urlsplit


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def code_repository(uri):
    parsed = urlsplit(uri)
    return (parsed.scheme.lower() in ('https', 'http')
            and parsed.hostname == 'packages.microsoft.com'
            and parsed.path.rstrip('/') == '/repos/code')


def filter_list(text):
    kept, removed = [], 0
    for line in text.splitlines(keepends=True):
        match = re.match(r'^\s*deb(?:-src)?\s+(?:\[[^\]]*\]\s+)?(\S+)', line)
        if match and code_repository(match.group(1)):
            removed += 1
        else:
            kept.append(line)
    return ''.join(kept), removed


def filter_sources(text):
    # Preserve all non-URI fields verbatim, including embedded Signed-By keys.
    kept, removed = [], 0
    for block in re.split(r'(\n[ \t]*\n)', text):
        lines = block.splitlines(keepends=True)
        fields = {}
        last = None
        for i, line in enumerate(lines):
            if line.startswith('#') or not line.strip():
                continue
            if line[0].isspace():
                if last is not None:
                    fields[last]['indices'].append(i)
                    fields[last]['value'] += ' ' + line.strip()
                continue
            match = re.match(r'^([A-Za-z0-9-]+):\s*(.*)', line)
            if match:
                last = match.group(1).lower()
                need(last not in fields, 'Duplicate field in APT .sources entry: ' + last)
                fields[last] = dict(indices=[i], value=match.group(2).strip())
        if fields.get('enabled', {}).get('value', 'yes').lower() == 'no' or 'uris' not in fields:
            kept.append(block)
            continue
        uris = fields['uris']['value'].split()
        remaining = [uri for uri in uris if not code_repository(uri)]
        removed += len(uris) - len(remaining)
        if len(remaining) == len(uris):
            kept.append(block)
        elif remaining:
            indices = fields['uris']['indices']
            replacement = 'URIs: ' + ' '.join(remaining) + '\n'
            kept.append(''.join(replacement if i == indices[0] else line
                                for i, line in enumerate(lines)
                                if i == indices[0] or i not in indices))
        # If every URI in this entry is Code, omit only this entry.
    return ''.join(kept), removed


def configured_sources(env):
    result = subprocess.run(['apt-config', 'shell', 'MAIN', 'Dir::Etc::sourcelist/f',
                             'PARTS', 'Dir::Etc::sourceparts/d'],
                            env=env, capture_output=True, text=True, check=True)
    values = dict(token.split('=', 1) for token in shlex.split(result.stdout))
    need(set(values) == {'MAIN', 'PARTS'}, 'Cannot resolve configured APT source locations')
    main, parts = Path(values['MAIN']), Path(values['PARTS'])
    need(main.is_absolute() and parts.is_absolute(), 'APT source paths must be absolute')
    return main, parts


def inventory(main, parts):
    paths = ([main] if main.exists() else [])
    if parts.is_dir():
        paths += [p for p in sorted(parts.iterdir())
                  if re.fullmatch(r'[A-Za-z0-9_.-]+\.(list|sources)', p.name)]
    items = []
    for p in paths:
        need(p.is_file(), 'APT source is not a readable regular file: ' + str(p))
        raw = p.read_bytes()
        items.append(dict(path=str(p), raw=raw,
                          sha256=hashlib.sha256(raw).hexdigest()))
    return items


class AptSources:
    def __init__(self, run_directory, env):
        self.run_directory = Path(run_directory)
        self.main, self.parts = configured_sources(env)
        self.generations = []
        self.current = None

    def refresh(self):
        entries = inventory(self.main, self.parts)
        scope = self.run_directory / ('apt-sources-%03d' % (len(self.generations) + 1))
        scope.mkdir(mode=0o700)
        parts = scope / 'sources.list.d'
        parts.mkdir(mode=0o700)
        main = scope / 'sources.list'
        main.touch(mode=0o600)
        rows = []
        for row in entries:
            original = Path(row['path'])
            text = row['raw'].decode('utf-8')
            filtered, count = (filter_sources(text) if original.suffix == '.sources'
                               else filter_list(text))
            target = main if original == self.main else parts / original.name
            target.write_text(filtered, encoding='utf-8')
            target.chmod(0o600)
            rows.append(dict(original=str(original), original_sha256=row['sha256'],
                             copy=str(target), copy_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                             excluded_code_entries=count))
        self.current = dict(directory=str(scope), source_files=rows,
                            originals={row['path']: row['sha256'] for row in entries},
                            excluded_code_entries=sum(row['excluded_code_entries'] for row in rows),
                            original_source_files_edited=False,
                            ipv4_for_this_command_only=True, signature_and_tls_checks_preserved=True)
        self.generations.append(self.current)
        return self.options()

    def options(self):
        need(self.current is not None, 'APT source scope was not prepared')
        now = {row['path']: row['sha256'] for row in inventory(self.main, self.parts)}
        need(now == self.current['originals'], 'APT sources changed after snapshot; preserve and rerun')
        for row in self.current['source_files']:
            p = Path(row['copy'])
            need(not p.is_symlink() and hashlib.sha256(p.read_bytes()).hexdigest() == row['copy_sha256'],
                 'APT source snapshot changed: ' + str(p))
        scope = Path(self.current['directory'])
        values = ['Dir::Etc::sourcelist=' + str(scope / 'sources.list'),
                  'Dir::Etc::sourceparts=' + str(scope / 'sources.list.d'),
                  'Dir::Cache::pkgcache=', 'Dir::Cache::srcpkgcache=',
                  'APT::Get::List-Cleanup=false',
                  'Acquire::ForceIPv4=true', 'Acquire::ForceIPv6=false',
                  'Acquire::http::Timeout=20', 'Acquire::https::Timeout=20']
        return [arg for value in values for arg in ('-o', value)]
