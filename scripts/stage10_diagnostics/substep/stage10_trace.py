"""Bounded, content-addressed diagnostic traces. No simulator dependency."""
from __future__ import annotations

import hashlib
import io
import itertools
import json
from pathlib import Path
import zipfile

import numpy as np

from stage10_core import require, read, write, sha, verify_inventory

RECOVERY_APPROVAL_SHA = '06e5e8a1054ef3bfb75e2e6840e186846bb771dd1a848ddb0292b583c26c3c7f'


def report_for_comparison(folder):
    """Reuse only the independently reviewed complete trace, retaining its gap."""
    folder = Path(folder)
    report = read(folder / 'result.json')
    if report['status'] == 'SHORT_TRACE_COMPLETE_REVIEW_REQUIRED':
        require(report['backend_after'] == report['backend_before'], 'Post-run backend values differ')
        return {**report, 'reviewed_trace_reuse': False, 'postcheck_verified': True}
    approval_path = folder / 'recovery-review.json'
    require(approval_path.is_file() and not approval_path.is_symlink() and sha(approval_path) == RECOVERY_APPROVAL_SHA,
            'Failed attempt lacks the exact reviewed trace-reuse record')
    approval = read(approval_path)
    require(folder.name == approval['original_attempt'] and folder.parent.name == approval['original_job']
            and folder.parent.parent.name == approval['original_batch'], 'Reviewed trace belongs to another run')
    verify_inventory(folder, approval['original_files'])
    require(report['status'] == 'FAILED_REVIEW_REQUIRED' and report['physics_steps'] == 100
            and report['control_steps'] == 10 and 'backend_after' not in report,
            'Reviewed failure boundary changed')
    # The original result.json remains FAILED_REVIEW_REQUIRED on disk.
    return {**report, 'schema': read(folder / 'capture-schema.json'), 'backend_after': None,
            'reviewed_trace_reuse': True, 'review_status': approval['review_status'],
            'postcheck_verified': False, 'postcheck_gap': approval['postcheck_gap']}


class TraceWriter:
    """Retain exact CPU bytes; chunk size keeps each return file under 49 MiB."""
    def __init__(self, folder, max_bytes=512 * 1024**2):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stream = (self.folder / 'events.jsonl').open('x', encoding='utf-8')
        self.blobs, self.total, self.chunk_bytes, self.chunk_id = {}, 0, 0, 0
        self.max_bytes, self.archive, self.events = max_bytes, None, 0

    def array(self, value, env_axis=None):
        a = np.array(value, copy=True, order='C', subok=False)
        require(not a.dtype.hasobject, 'Object arrays cannot be diagnostic evidence')
        require(a.nbytes <= 16 * 1024**2, 'One trace field exceeds 16 MiB; preserve partial evidence')
        signature = json.dumps([a.dtype.str, list(a.shape)], separators=(',', ':')).encode()
        digest = hashlib.sha256(signature + b'\0' + a.tobytes()).hexdigest()
        if digest not in self.blobs:
            stream = io.BytesIO()
            np.save(stream, a, allow_pickle=False)
            content = stream.getvalue()
            require(self.total + len(content) <= self.max_bytes, 'Trace reached the 512 MiB/run raw-byte cap')
            if self.archive is None or self.chunk_bytes + len(content) > 16 * 1024**2:
                if self.archive is not None:
                    self.archive.close()
                self.chunk_id += 1
                self.chunk_name = f'arrays-{self.chunk_id:03d}.zip'
                self.archive = zipfile.ZipFile(self.folder / self.chunk_name, 'x',
                                              compression=zipfile.ZIP_DEFLATED, compresslevel=1)
                self.chunk_bytes = 0
            entry = digest + '.npy'
            self.archive.writestr(entry, content)
            self.blobs[digest] = {'blob': digest, 'file': self.chunk_name, 'entry': entry,
                                  'dtype': a.dtype.str, 'shape': list(a.shape)}
            self.chunk_bytes += len(content)
            self.total += len(content)
        return {**self.blobs[digest], 'env_axis': env_axis}

    def record(self, phase, control, substep, fields):
        row = {'index': self.events, 'phase': phase, 'control': control,
               'substep': substep, 'fields': fields}
        self.stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + '\n')
        self.stream.flush()
        self.events += 1

    def close(self):
        if self.archive is not None:
            self.archive.close()
            self.archive = None
        if not self.stream.closed:
            self.stream.close()


def rows(folder):
    return [json.loads(line) for line in (Path(folder) / 'events.jsonl').read_text().splitlines()]


class TraceReader:
    def __init__(self, folder):
        self.folder, self.archives = Path(folder), {}

    def array(self, descriptor):
        name = descriptor['file']
        require(Path(name).name == name, 'Unsafe trace archive path')
        if name not in self.archives:
            self.archives[name] = zipfile.ZipFile(self.folder / name)
        with self.archives[name].open(descriptor['entry']) as f:
            return np.load(io.BytesIO(f.read()), allow_pickle=False)

    def close(self):
        for f in self.archives.values():
            f.close()


def different(a, b):
    if isinstance(a, dict) and isinstance(b, dict) and 'blob' in a and 'blob' in b:
        return a['blob'] != b['blob'] or a['env_axis'] != b['env_axis']
    return a != b


def detail(a, b, left, right):
    if not (isinstance(a, dict) and isinstance(b, dict) and 'blob' in a and 'blob' in b):
        return {'kind': 'metadata_or_missing', 'left': a, 'right': b}
    if a['shape'] != b['shape'] or a['dtype'] != b['dtype']:
        return {'kind': 'shape_or_dtype', 'left_shape': a['shape'], 'right_shape': b['shape'],
                'left_dtype': a['dtype'], 'right_dtype': b['dtype']}
    x, y = left.array(a), right.array(b)
    if x.dtype.kind not in 'buif':
        return {'kind': 'byte_difference', 'dtype': x.dtype.str}
    delta = np.abs(x.astype(np.float64) - y.astype(np.float64))
    finite = np.isfinite(x) & np.isfinite(y)
    out = {'kind': 'array', 'dtype': x.dtype.str, 'shape': list(x.shape),
           'bitwise_equal': a['blob'] == b['blob'],
           'numerically_equal': bool(np.array_equal(x, y, equal_nan=True)),
           'nonfinite_elements': int((~finite).sum()),
           'max_abs_finite': float(delta[finite].max()) if finite.any() else None,
           'elements_abs_gt_1e_7': int(((delta > 1e-7) & finite).sum()),
           'elements_abs_gt_1e_5': int(((delta > 1e-5) & finite).sum())}
    axis = a.get('env_axis')
    if axis is not None and axis == b.get('env_axis') and x.shape[axis] == 128:
        numeric_bad = np.not_equal(x, y)
        axes = tuple(i for i in range(x.ndim) if i != axis)
        bad = np.any(numeric_bad, axis=axes) if axes else numeric_bad
        out['env_ids_numeric_difference'] = np.flatnonzero(bad).tolist()
    return out


def compare_pair(left_folder, right_folder):
    aa, bb = rows(left_folder), rows(right_folder)
    require(len(aa) == len(bb), 'Event counts differ; trace alignment is invalid')
    left, right = TraceReader(left_folder), TraceReader(right_folder)
    first_by_field, first_core, first_contact, phases = {}, None, None, {}
    try:
        for a, b in zip(aa, bb, strict=True):
            identity = {k: a[k] for k in ('index', 'phase', 'control', 'substep')}
            require(identity == {k: b[k] for k in identity}, 'Event ordering differs')
            changed = [k for k in sorted(set(a['fields']) | set(b['fields']))
                       if different(a['fields'].get(k), b['fields'].get(k))]
            core = [k for k in changed if not k.startswith('contact/')]
            contact = [k for k in changed if k.startswith('contact/')]
            if core and first_core is None:
                first_core = {**identity, 'fields': core}
            if contact and first_contact is None:
                first_contact = {**identity, 'fields': contact}
            if changed:
                phases[a['phase']] = phases.get(a['phase'], 0) + 1
            for k in changed:
                if k not in first_by_field:
                    first_by_field[k] = {**identity, **detail(a['fields'].get(k), b['fields'].get(k), left, right)}
        return {'left': str(left_folder.parent.name), 'right': str(right_folder.parent.name),
                'aligned_events': len(aa), 'first_core_difference': first_core,
                'first_contact_storage_difference': first_contact,
                'first_difference_by_field': first_by_field,
                'differing_event_counts_by_phase': phases,
                'contact_order_is_not_a_causal_or_physical_equivalence_test': True}
    finally:
        left.close()
        right.close()


def compare_all(folders, output):
    require(len(folders) == 3, 'Expected exactly three short traces')
    reports = [report_for_comparison(p) for p in folders]
    require(all(r['status'] == 'SHORT_TRACE_COMPLETE_REVIEW_REQUIRED' or r['reviewed_trace_reuse'] for r in reports),
            'Incomplete short trace')
    for report in reports[1:]:
        for key in ('runtime_identity', 'model_loaded', 'backend_before', 'schema', 'dependency_sources_sha256'):
            require(report[key] == reports[0][key], 'Incomparable runs: ' + key)
    pairs = [compare_pair(Path(a) / 'trace', Path(b) / 'trace') for a, b in itertools.combinations(folders, 2)]
    gaps = [{'job': Path(folder).parent.name, 'attempt': Path(folder).name, 'gap': report['postcheck_gap']}
            for folder, report in zip(folders, reports, strict=True) if not report['postcheck_verified']]
    result = {'status': 'SUBSTEP_BATCH_TRACES_COMPLETE_WITH_POSTCHECK_GAP_REVIEW_REQUIRED' if gaps
                       else 'SUBSTEP_BATCH_COMPLETE_REVIEW_REQUIRED', 'new_ppo_updates': 0,
              'stage10_complete': False, 'task_success_not_evaluated': True,
              'num_runs': 3, 'num_envs': 128, 'control_steps_each': 10,
              'physics_steps_each': 100, 'seconds_each': 0.2, 'pairs': pairs,
              'postcheck_gaps': gaps, 'all_postchecks_verified': not bool(gaps),
              'original_failed_results_preserved': True,
              'limits': ['GPU readback adds synchronization and can alter nondeterminism.',
                        'A reused first trace lacks backend_after; its end-of-run precision cannot be reconstructed.',
                        'First observed difference localizes a boundary, not a CUDA kernel or a root cause.',
                        'Contacts use recorded aggregate-count prefix and unverified storage ordering.',
                        'No claim of full simulator/internal state equivalence; consult schema gaps.',
                        'Tolerances only describe numeric differences; old acceptance stays unchanged.',
                        'No 10-second episode completed, no success rate or independent benchmark result.']}
    write(output, result)
    return result
