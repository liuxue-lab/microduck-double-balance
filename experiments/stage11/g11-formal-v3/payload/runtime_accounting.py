"""G11 bounded accounting. Integrations keep a durable before/after WAL.

Nested graph/kernel counters stay in RAM and are checkpointed at macro/exit
boundaries. A hard process loss is reported as an interval, never as PASS.
The native synchronization and checks in the accepted helpers are retained.
"""
import hashlib
import json
import os
import time
from pathlib import Path


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, 'tolist'):
        return value.tolist()
    return value


def save(path, value, durable=False):
    path = Path(path)
    raw = (json.dumps(jsonable(value), ensure_ascii=False, sort_keys=True,
                      indent=2, allow_nan=False) + '\n').encode()
    tmp = path.with_name(path.name + '.writing')
    with tmp.open('wb') as f:
        f.write(raw)
        if durable:
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, path)
    if durable:
        fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


class Journal:
    def __init__(self, out, counts, caps):
        self.out, self.counts, self.caps = Path(out), counts, caps
        self.stream = (self.out / 'native-ledger.jsonl').open('xb', buffering=0)
        self.seq = 0
        self.pending = {}
        self.start = time.monotonic()
        self.last_progress = 0.0
        self.wal_writes = 0
        self.wal_seconds = 0.0
        self.digest = hashlib.sha256()
        self.phase = 'initialization'
        self.macros = 0
        self.formal_steps = 0
        self.closed = False

    def event(self, kind, label, number, **extra):
        self.seq += 1
        raw = (json.dumps(dict(seq=self.seq, kind=kind, label=label,
                              number=number, **extra), sort_keys=True,
                          separators=(',', ':'), allow_nan=False) + '\n').encode()
        t = time.monotonic()
        written = self.stream.write(raw)
        require(written == len(raw), 'Partial native ledger append; stop')
        os.fsync(self.stream.fileno())
        self.wal_seconds += time.monotonic() - t
        self.wal_writes += 1
        self.digest.update(raw)

    def begin(self, label, number):
        require(label not in self.pending, 'Nested ledger entry: ' + label)
        self.event('attempt', label, number)
        self.pending[label] = number

    def end(self, label, number):
        require(self.pending.get(label) == number, 'Ledger completion has no matching attempt')
        self.event('complete', label, number)
        del self.pending[label]

    def progress(self, force=False):
        now = time.monotonic()
        if force or now - self.last_progress >= 5:
            save(self.out / 'progress.json', dict(
                phase=self.phase, elapsed_s=now-self.start,
                macro_completed=self.macros, macro_cap=20,
                formal_steps_completed=self.formal_steps, formal_steps_cap=2,
                counts=self.counts, caps={k: self.caps[k] for k in
                    ('integrations', 'GPU_forward', 'CPU_forward')},
                native_WAL_pending=dict(self.pending),
                gate_closed=10, gate_total=11, gate_percent=1000/11,
                wall_deadline=None, PPO=0, cloud_training=0,
                cumulative_completed=670872+self.counts.get('integrations_completed', 0),
                cumulative_attempted=670873+self.counts.get('integrations_attempted', 0)))
            self.last_progress = now

    def checkpoint(self, phase):
        self.phase = phase
        save(self.out / 'counter-checkpoint.json', dict(counts=self.counts,
             wal_sequence=self.seq, pending=dict(self.pending), phase=phase), durable=True)
        self.progress(True)

    def close(self):
        if self.closed:
            return
        self.checkpoint('worker-finished')
        self.stream.close()
        self.closed = True
        save(self.out / 'accounting-review.json', dict(
            durable_integrations=True, WAL_records=self.seq,
            WAL_sha256=self.digest.hexdigest(), WAL_fsyncs=self.wal_writes,
            WAL_seconds=self.wal_seconds, pending=dict(self.pending),
            hard_crash_rule='Unpaired integration attempt has completed interval [n-1,n]; never infer completion or retry.',
            nested_counter_rule='Exact on orderly exit; after hard loss use last checkpoint plus explicit per-integration upper bounds.',
            native_sync_checks_removed=0, full_per_kernel_persistence=False), durable=True)


def make_classes():
    from lifecycle_support import Gate as OriginalGate
    from macro_state_bridge import MacroBudget as OriginalBudget

    class Gate(OriginalGate):
        def __init__(self, counts, journal, sync, caps):
            super().__init__(counts, journal.progress, float('inf'), sync)
            self.journal, self.caps = journal, caps
            self.timings = {}

        def limited(self, label, limit, native, check=None):
            # Existing helper caps become this new batch's declared caps, only
            # for labels present in the reviewed manifest. No tolerance changes.
            limit = self.caps.get(label, limit)
            def call(*args, **kwargs):
                if check is not None:
                    check(args, kwargs)
                a, c = label+'_attempted', label+'_completed'
                require(self.counts.get(a, 0) < limit, 'Budget exhausted: '+label)
                durable = label == 'CPU_forward' or (label == 'GPU_forward' and
                          'integrations' not in self.journal.pending)
                number = self.counts.get(a, 0) + 1
                if durable:
                    self.journal.begin(label, number)
                self.counts[a] = number
                t = time.monotonic()
                try:
                    value = native(*args, **kwargs)
                    self.sync()
                    if durable:
                        self.journal.end(label, number)
                    self.counts[c] = self.counts.get(c, 0) + 1
                    return value
                finally:
                    row = self.timings.setdefault(label, {'calls': 0, 'inclusive_s': 0.0})
                    row['calls'] += 1
                    row['inclusive_s'] += time.monotonic() - t
                    self.journal.progress()
            return call

    class Budget(OriginalBudget):
        delegated = frozenset(('GPU_forward', 'sense', 'Scene_write', 'Scene_update',
                              'Action_apply', 'Metrics_substep'))
        def __init__(self, limits, counts, journal, sync):
            super().__init__(limits, counts=counts, persist=journal.progress,
                             sync=sync, deadline=float('inf'))
            self.journal = journal

        def run(self, name, fn):
            self.reserve(**{name: 1})
            a, c = name+'_attempted', name+'_completed'
            if name in self.delegated:
                before = (self.counts.get(a, 0), self.counts.get(c, 0))
                value = fn()
                require((self.counts.get(a, 0), self.counts.get(c, 0)) ==
                        (before[0]+1, before[1]+1), 'Double/missing delegated accounting: '+name)
                return value
            number = self.counts.get(a, 0) + 1
            if name == 'integrations':
                self.journal.begin(name, number)
            self.counts[a] = number
            value = fn()
            self.sync()
            if name == 'integrations':
                self.journal.end(name, number)
            self.counts[c] = self.counts.get(c, 0) + 1
            self.journal.progress()
            return value
    return Gate, Budget


def review_ledger(path):
    pending, complete, attempts = {}, {}, {}
    rows = 0
    incomplete_tail = False
    with Path(path).open('rb') as f:
        for raw in f:
            if not raw.endswith(b'\n'):
                incomplete_tail = True
                break
            row = json.loads(raw)
            rows += 1
            require(row['seq'] == rows, 'Ledger sequence mismatch')
            k, n = row['label'], row['number']
            if row['kind'] == 'attempt':
                require(k not in pending and n > attempts.get(k, 0), 'Ledger attempt ordering mismatch')
                pending[k] = n
                attempts[k] = n
            else:
                require(row['kind'] == 'complete' and pending.get(k) == n, 'Ledger pairing mismatch')
                complete[k] = n
                del pending[k]
    a, c = attempts.get('integrations', 0), complete.get('integrations', 0)
    return dict(rows=rows, attempts=attempts, completed=complete, pending=pending,
                incomplete_tail=incomplete_tail,
                integration_attempted_lower=a, integration_attempted_upper=a+(1 if incomplete_tail else 0),
                integration_completed_lower=c, integration_completed_upper=a,
                exact=not incomplete_tail and not pending)
