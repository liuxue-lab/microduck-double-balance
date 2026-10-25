"""Read existing Stage12 files only. No simulator imports or campaign writes."""
import json
import math
from pathlib import Path
import re
import statistics

TAIL_BYTES = 1024 * 1024


def tail(path, limit=TAIL_BYTES):
    with Path(path).open('rb') as f:
        end = f.seek(0, 2)
        start = max(0, end - limit)
        f.seek(start)
        data = f.read(end - start)
    if start:
        data = data.split(b'\n', 1)[-1]
    return data.decode(errors='replace')


def stats(values):
    values = [x for x in values if isinstance(x, (int, float)) and math.isfinite(x)]
    if not values:
        return None
    return dict(samples=len(values), mean=statistics.mean(values),
                median=statistics.median(values), minimum=min(values), maximum=max(values))


def parse_log(text, limit=30):
    text = re.sub(r'\x1b\[[0-?]*[ -/]*[@-~]', '', text)
    blocks = re.split(r'Learning iteration\s+(\d+)\s*/\s*(\d+)', text)
    rows = []
    for i in range(1, len(blocks), 3):
        body = blocks[i+2]
        values = {}
        for name, label in [('collection_seconds', 'Collection time'), ('learning_seconds', 'Learning time')]:
            match = re.search(r'\b' + label + r':\s*([0-9]+(?:\.[0-9]+)?)s\b', body)
            if match:
                values[name] = float(match[1])
        if len(values) == 2:
            rows.append(dict(iteration=int(blocks[i]), **values))
    rows = rows[-limit:]
    c = [r['collection_seconds'] for r in rows]
    p = [r['learning_seconds'] for r in rows]
    total = sum(c) + sum(p)
    return dict(recent_iterations=rows, collection_seconds=stats(c), learning_seconds=stats(p),
                collection_fraction_of_logged_compute=sum(c)/total if total else None,
                learning_fraction_of_logged_compute=sum(p)/total if total else None,
                timing_scope='Existing runner wall timers; exclude logger/checkpoint overhead; not CUDA kernel timings.')


def parse_metrics(text, limit=31):
    rows = []
    for line in text.splitlines():
        try:
            value = json.loads(line)
        except (ValueError, TypeError):
            continue
        if (isinstance(value, dict) and value.get('campaign_phase') == 'formal' and isinstance(value.get('elapsed_seconds'), (int, float))
                and isinstance(value.get('completed_updates'), int)):
            rows.append(value)
    rows = rows[-limit:]
    rates = []
    for a, b in zip(rows, rows[1:]):
        count = b['completed_updates'] - a['completed_updates']
        elapsed = b['elapsed_seconds'] - a['elapsed_seconds']
        if count > 0 and elapsed > 0 and a.get('session') == b.get('session'):
            rates.append(elapsed/count)
    return dict(wall_seconds_per_update=stats(rates),
                window_first_update=rows[0]['completed_updates'] if rows else None,
                window_last_update=rows[-1]['completed_updates'] if rows else None,
                last_state=rows[-1] if rows else None,
                timing_scope='Differences between existing elapsed_seconds records, including intervening logging and checkpoint work.')


def inspect_slot(slot):
    slot = Path(slot)
    result = dict(slot=str(slot), read_only=True, snapshot_is_atomic=False, files={}, added_PPO_updates=0,
                  added_simulation_steps=0, task_restart=False)
    for name in ('budget.json', 'run/progress.json', 'run/latest.json', 'run/smoke-result.json',
                 'started.json', 'supervisor.json', 'child.json'):
        p = slot/name
        if p.is_file() and p.stat().st_size <= 65536:
            try:
                result['files'][name] = json.loads(p.read_bytes())
            except (ValueError, OSError) as exc:
                result.setdefault('read_errors', {})[name] = repr(exc)
    logs = list((slot/'run').glob('train-*-formal.log'))
    if not logs:
        logs = list((slot/'run').glob('train-*.log'))
    if logs:
        log = max(logs, key=lambda p: p.stat().st_mtime)
        result['log_path'] = str(log)
        result['log'] = parse_log(tail(log))
    metrics = slot/'run/metrics.jsonl'
    if metrics.is_file():
        result['metrics'] = parse_metrics(tail(metrics))
    # PID identity checks are read-only and avoid trusting a recycled process ID.
    result['processes'] = {}
    for kind in ('supervisor', 'child'):
        pin = result['files'].get(kind+'.json')
        if not pin:
            continue
        try:
            row = (Path('/proc')/str(pin['pid'])/'stat').read_text().rsplit(')', 1)[1].split()
            result['processes'][kind] = dict(pid=pin['pid'], alive=row[19] == pin['start'] and row[0] not in ('Z', 'X'))
        except (OSError, KeyError, IndexError):
            result['processes'][kind] = dict(pid=pin.get('pid'), alive=False)
    return result
