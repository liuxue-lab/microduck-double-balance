"""Offline audit of uploaded Stage 10 B traces; never imports runtime code.

Usage: python3 review_substep_b.py EXTRACTED_REVIEW OUTPUT_JSON [SOURCE_COPY]
Requires NumPy only. All .npy loads use allow_pickle=False.
"""
from collections import Counter
from itertools import combinations
from pathlib import Path
import hashlib
import io
import json
import sys
import zipfile

import numpy as np


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text())


def verify_files(root, rows):
    for row in rows:
        p = root / row['path']
        assert p.resolve().is_relative_to(root.resolve()), p
        assert p.stat().st_size == row['size_bytes'], p
        assert sha(p) == row['sha256'], p
    return len(rows)


def identity(row):
    return {k: row[k] for k in ('index', 'phase', 'control', 'substep')}


def array_descriptor(v):
    return isinstance(v, dict) and 'blob' in v


def equal(a, b):
    if array_descriptor(a) and array_descriptor(b):
        return all(a[k] == b[k] for k in ('blob', 'dtype', 'shape', 'env_axis'))
    return a == b


def expected_order():
    result = [('reset_ready', 0, 0)]
    for c in range(1, 11):
        result += [(p, c, 0) for p in ('policy_before', 'policy_after', 'action_processed')]
        for s in range(1, 11):
            result += [(p, c, s) for p in
                       ('action_applied', 'bam_before', 'bam_after', 'physics_before', 'physics_after')]
        result += [(p, c, 10) for p in ('forward_before', 'forward_after', 'control_return')]
    return result


class Run:
    def __init__(self, path):
        self.path = path
        self.report = read(path / 'result.json')
        self.rows = [json.loads(s) for s in (path / 'trace/events.jsonl').read_text().splitlines()]
        assert len(self.rows) == 561
        assert [(e['phase'], e['control'], e['substep']) for e in self.rows] == expected_order()
        self.arrays = {}
        storage = {}
        for p in sorted((path / 'trace').glob('arrays-*.zip')):
            with zipfile.ZipFile(p) as z:
                assert z.testzip() is None
                for name in z.namelist():
                    a = np.load(io.BytesIO(z.read(name)), allow_pickle=False)
                    assert not a.dtype.hasobject and a.flags.c_contiguous
                    signature = json.dumps([a.dtype.str, list(a.shape)], separators=(',', ':')).encode()
                    h = hashlib.sha256(signature + b'\0' + a.tobytes()).hexdigest()
                    assert name == h + '.npy' and h not in self.arrays
                    assert a.dtype.kind in 'buif' and bool(np.isfinite(a).all())
                    self.arrays[h] = a
                    storage[h] = (p.name, name)
        referenced = set()
        for n, row in enumerate(self.rows):
            assert row['index'] == n
            f = row['fields']
            for v in f.values():
                if array_descriptor(v):
                    a = self.arrays[v['blob']]
                    assert v['dtype'] == a.dtype.str and v['shape'] == list(a.shape)
                    assert (v['file'], v['entry']) == storage[v['blob']]
                    if v['env_axis'] is not None:
                        assert a.shape[v['env_axis']] == 128
                    referenced.add(v['blob'])
            for key in ('physical/qfrc_applied', 'physical/xfrc_applied'):
                if key in f:
                    assert not bool(self.get(f[key]).any())
            if 'contact/count' in f:
                count = f['contact/count']
                for k, v in f.items():
                    if k.startswith('contact/') and array_descriptor(v):
                        assert self.get(v).shape[0] == count
                ids = self.get(f['contact/worldid'])
                assert bool(((ids >= 0) & (ids < 128)).all())
                assert int(self.get(f['physical/nacon'])[0]) == count
            if row['phase'] == 'physics_after':
                expected = ((row['control'] - 1) * 10 + row['substep']) * .002
                assert np.max(np.abs(self.get(f['physical/time']).astype('float64') - expected)) < 1e-6
        assert referenced == set(self.arrays)
        if 'unique_trace_arrays' in self.report:
            assert self.report['unique_trace_arrays'] == len(self.arrays)
        self.audit = {'job': path.parent.name, 'attempt': path.name,
                      'original_status': self.report['status'], 'events': len(self.rows),
                      'unique_arrays_verified': len(self.arrays), 'all_numeric_finite': True,
                      'all_references_verified': True, 'external_wrenches_zero': True,
                      'contact_prefix_counts_and_world_ids_verified': True,
                      'terminal_time_s': self.get(self.rows[-1]['fields']['physical/time']).tolist()[0],
                      'backend_after_available': 'backend_after' in self.report}

    def get(self, d):
        return self.arrays[d['blob']]


def delta(a, b, left, right):
    if not (array_descriptor(a) and array_descriptor(b)):
        return {'metadata_left': a, 'metadata_right': b}
    x, y = left.get(a), right.get(b)
    if x.dtype != y.dtype or x.shape != y.shape:
        return {'shape_or_dtype_difference':True, 'left_shape':list(x.shape),
                'right_shape':list(y.shape),'left_dtype':x.dtype.str,'right_dtype':y.dtype.str}
    d = np.abs(x.astype('float64') - y.astype('float64'))
    out = {'max_abs': float(d.max()) if d.size else 0.,
           'shape': list(x.shape), 'dtype': x.dtype.str,
           'unequal_elements': int(np.count_nonzero(x != y)),
           'numerically_equal': bool(np.array_equal(x, y))}
    if a['env_axis'] is not None:
        axis = a['env_axis']
        reduce = tuple(i for i in range(x.ndim) if i != axis)
        mask = (x != y).any(axis=reduce) if reduce else x != y
        out['env_ids'] = np.flatnonzero(mask).tolist()
    return out


def canonical_contacts(run, row, omit_address):
    fields = row['fields']
    names = sorted(k for k,v in fields.items() if k.startswith('contact/') and array_descriptor(v)
                   and not (omit_address and k == 'contact/efc_address'))
    arrays = [run.get(fields[k]) for k in names]
    signature = tuple((k, a.dtype.str, a.shape[1:]) for k,a in zip(names, arrays))
    return signature, Counter(b''.join(a[i].tobytes() for a in arrays)
                              for i in range(fields['contact/count']))


def compare(left, right, supplied):
    first = {}
    timeline = []
    contact_first = None
    canonical_first = None
    for a,b in zip(left.rows, right.rows):
        assert identity(a) == identity(b)
        assert set(a['fields']) == set(b['fields'])
        changed = [k for k in sorted(a['fields']) if not equal(a['fields'][k], b['fields'][k])]
        for k in changed:
            if k not in first:
                first[k] = {**identity(a), **delta(a['fields'][k], b['fields'][k], left, right)}
        if a['phase'] in ('reset_ready', 'physics_after', 'control_return', 'policy_after'):
            metrics = {k: delta(a['fields'][k], b['fields'][k], left, right)
                       for k in ('physical/qpos', 'physical/qvel', 'physical/qacc', 'physical/ctrl', 'policy_action')
                       if k in a['fields']}
            timeline.append({**identity(a), 'fields': metrics})
        if any(k.startswith('contact/') for k in changed):
            canon_equal = canonical_contacts(left, a, True) == canonical_contacts(right, b, True)
            all_equal = canonical_contacts(left, a, False) == canonical_contacts(right, b, False)
            detail = {**identity(a), 'counts': [a['fields']['contact/count'],b['fields']['contact/count']],
                      'all_captured_rows_equal_up_to_permutation': all_equal,
                      'rows_excluding_efc_address_equal_up_to_permutation': canon_equal}
            if contact_first is None:
                contact_first = detail
            if not canon_equal and canonical_first is None:
                canonical_first = detail
    assert set(first) == set(supplied['first_difference_by_field'])
    for k, v in first.items():
        ref = supplied['first_difference_by_field'][k]
        assert all(v[n] == ref[n] for n in ('index','phase','control','substep')), k
        if 'max_abs_finite' in ref:
            assert v['max_abs'] == ref['max_abs_finite'], k
            assert v['numerically_equal'] == ref['numerically_equal'], k
            if 'env_ids_numeric_difference' in ref:
                assert v['env_ids'] == ref['env_ids_numeric_difference'], k
    reset = left.rows[0]['fields']
    reset_changed = [k for k,v in first.items() if v['index'] == 0]
    reset_diagnostics = {}
    for key in ('physical/time','physical/qvel','physical/qacc_warmstart','physical/ctrl',
                'physical/nefc','physical/solver_niter','physical/nacon','physical/nisland'):
        x = left.get(reset[key])
        reset_diagnostics[key] = {'min': float(x.min()), 'max': float(x.max()),
                                 'nonzero_envs': np.flatnonzero(np.any(x != 0, axis=tuple(range(1,x.ndim)))
                                                              if x.ndim > 1 else x != 0).tolist()}
    return {'left': left.path.parent.name, 'right': right.path.parent.name,
            'independent_first_differences_match_supplied': True,
            'reset_field_count': len(reset), 'reset_different_fields': reset_changed,
            'recorded_rng_equal_at_all_observation_events': not any(k.startswith('rng/') for k in first),
            'first_difference_by_field': first,
            'first_contact_storage_difference': contact_first,
            'first_canonical_contact_content_difference': canonical_first,
            'reset_diagnostics_left': reset_diagnostics, 'timeline': timeline}


def main():
    root, output = map(Path, sys.argv[1:3])
    manifest = read(root/'review-files.json')
    count = verify_files(root, manifest)
    assert {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()} == {
        r['path'] for r in manifest} | {'review-files.json'}
    plan, comparison = read(root/'plan.json'), read(root/'comparison.json')
    for n, h in plan['tools'].items():
        assert sha(root/'tool'/n) == h
    runs = []
    for job in plan['jobs']:
        receipt = read(root/job/'completed.json')
        folder = root/job/receipt['attempt']
        verify_files(folder,receipt['files'])
        runs.append(Run(folder))
    assert runs[0].report['status'] == 'FAILED_REVIEW_REQUIRED'
    assert 'backend_after' not in runs[0].report
    recovery = read(runs[0].path/'recovery-review.json')
    assert sha(runs[0].path/'recovery-review.json') == '06e5e8a1054ef3bfb75e2e6840e186846bb771dd1a848ddb0292b583c26c3c7f'
    preserved = verify_files(runs[0].path, recovery['original_files'])
    assert sha(runs[0].path.parent/(runs[0].path.name+'.log')) == recovery['error_log_sha256']
    for run in runs[1:]:
        assert run.report['status'] == 'SHORT_TRACE_COMPLETE_REVIEW_REQUIRED'
        assert run.report['backend_after'] == run.report['backend_before']
        assert run.report['model_parameters_unchanged'] and run.report['adam_state_unchanged']
    for run in runs:
        for k in ('runtime_identity','model_loaded','backend_before','dependency_sources_sha256',
                  'checkpoint_sha256','dataset_sha256','restoration'):
            assert run.report[k] == runs[0].report[k]
        for name in ('capture-schema.json','dependency-sources.json','initial-receipt.json','head-initial.npz','params/env.yaml','runtime-check.json'):
            assert sha(run.path/name) == sha(runs[0].path/name)
    probe = read(root/'rnn-reader-recovery-v1/precision-probe-31341e22.json')
    assert probe['status'] == 'PRECISION_READER_PROBE_PASS_ZERO_PHYSICS'
    assert probe['before_module_import'] == probe['after_module_import'] == runs[0].report['backend_before']
    pairs = [compare(a,b,supplied) for (a,b),supplied in zip(combinations(runs,2),comparison['pairs'])]
    source_count = None
    if len(sys.argv) > 3:
        source = Path(sys.argv[3]); frozen = read(source/'docs/audits/stage-09-runtime-hashes.json')
        for name, h in frozen.items():
            assert sha(source/name) == h, name
        source_count = len(frozen)
    report = {'status':'REVIEWED_WITH_FIRST_RUN_POSTCHECK_GAP', 'stage10_complete':False,
              'manifest_files_verified':count, 'frozen_source_files_verified':source_count,
              'runs':[r.audit for r in runs], 'unique_arrays_per_run_sum':sum(len(r.arrays) for r in runs),
              'aligned_event_total':sum(len(r.rows) for r in runs),
              'interface_probe_passed_zero_physics':True,
              'first_run_original_failure_preserved':True,
              'first_run_original_files_verified_against_pinned_recovery':preserved,
              'first_run_missing_postchecks':['backend_after','final frozen-source recheck'],
              'new_ppo_updates':0,'new_simulation_executed_by_this_audit':False,
              'success_rate_assessed':False,'cloud_gpu_hours':0,
              'pairs':pairs,
              'limits':['First capture is after reset/forward/sensing/observation, so the generating call is not isolated.',
                        'Only captured state is compared; compiled model/internal device state is incomplete.',
                        'Synchronization from diagnostic readback can alter execution ordering.',
                        'Contact row permutation test is not full contact/solver equivalence.',
                        'No intervention or full 10-second rollout; no causal attribution of acceptance label changes.']}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'pairs'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
