"""Read-only reconciliation and exact frozen-criterion checks; no physics API."""
from pathlib import Path
import copy
import gzip
import hashlib
import json
import sys
import zipfile

from .durable import require, file_digest, write_json


def read(path):
    return json.loads(Path(path).read_bytes())


def safe_file(root, relative):
    root, rel = Path(root), Path(relative)
    require(not rel.is_absolute() and '..' not in rel.parts and '\\' not in str(rel), 'Unsafe evidence path')
    path = root / rel
    require(path.is_file() and all(not p.is_symlink() for p in (path, *path.parents)), 'Missing/linked evidence: '+str(path))
    return path


def verify_file(root, row):
    path = safe_file(root, row['path'])
    require(path.stat().st_size == row['bytes'] and file_digest(path) == row['sha256'], 'Evidence bytes changed: '+row['path'])
    return path


def verified_index(root):
    audit = read(Path(root)/'v5-audit.json')
    raw = (Path(root)/'v5-native-evidence-index.jsonl.gz').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == audit['evidence_index_sha256'], 'Frozen index changed')
    rows = [json.loads(x) for x in gzip.decompress(raw).splitlines()]
    require(len(rows) == audit['index_files'] and len({r['path'] for r in rows}) == len(rows), 'Index roster differs')
    require(sum(r['bytes'] for r in rows) == audit['index_bytes'], 'Index byte total differs')
    return rows


def verify_predecessor(root, base=None):
    root = Path(root)
    audit = read(root/'v5-audit.json')
    base = Path(base or audit['base'])
    run = base/audit['run']
    marker = read(safe_file(base, 'started-v1.json'))
    require(marker['run'] == str(run) and marker['package_manifest_sha256'] == audit['package_sha256'], 'v5 started identity differs')
    archive = verify_file(base, dict(path=audit['archive_filename'], bytes=audit['archive_bytes'], sha256=audit['archive_sha256']))
    receipt = read(safe_file(base, 'return-receipt.json'))
    transfer = read(safe_file(base, 'transfer-receipt.json'))
    require(receipt['run'] == str(run) and receipt['archive'] == str(archive) and receipt['sha256'] == audit['archive_sha256'] and receipt['bytes'] == audit['archive_bytes'], 'v5 return receipt differs')
    require(transfer['commit'] == audit['commit'] and transfer['branch'] == audit['branch'] and transfer['archive_sha256'] == audit['archive_sha256'], 'v5 transfer receipt differs')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        manifest = read(root/'v5-audit.json')['original_return_manifest']
        require(len(names) == len(set(names)) and set(names) == set(manifest)|{'RETURN-MANIFEST.json'}, 'v5 return roster differs')
        require(json.loads(z.read('RETURN-MANIFEST.json')) == manifest, 'v5 return manifest differs')
        for name, row in manifest.items():
            raw = z.read(name)
            require(len(raw) == row['bytes'] and hashlib.sha256(raw).hexdigest() == row['sha256'], 'v5 archive member differs: '+name)
            if name.startswith('evidence/'):
                path = safe_file(run, name[len('evidence/'):])
                require(path.read_bytes() == raw, 'Retained v5 report changed: '+name)
    counts = reconcile(root, run)
    require(counts['native_added'] == audit['native_added'] and counts['canonical_after'] == audit['canonical_after'], 'v5 accounting differs')
    report = read(run/'stage12-report.json')
    require(report['preservation'] == 'PASS' and report['status'] == 'LOCAL_COMPLETION_STOPPED_PRESERVE', 'v5 stop/preservation status differs')
    return run, counts


def reconcile(root, run):
    root, run = Path(root), Path(run)
    sys.path.insert(0, str(root/'frozen-v5/baseline/g11'))
    from runtime_accounting import review_ledger
    audit = read(root/'v5-audit.json')
    owners = []
    for name, expected in audit['integrations'].items():
        state = review_ledger(run/name/'native-ledger.jsonl')
        require(state['exact'] and state['pending'] == {} and not state['incomplete_tail'], 'Uncertain native ledger: '+name)
        require(state['attempts'] == state['completed'] == {'CPU_forward':3, 'GPU_forward':audit['GPU_forwards'][name], 'integrations':expected}, 'Owner totals differ: '+name)
        macros = sorted((run/name).glob('macro-*/macro-result.json'))
        require([p.parent.name for p in macros] == ['macro-%02d'%i for i in range(audit['accepted_macros'][name])], 'Accepted macro roster differs: '+name)
        for p in macros:
            row = read(p)
            require(row['decision']['accepted'] and row['mandatory_exact_replay'] and row['replayed'] and row['accepted_resolution'] in (64,128), 'Unaccepted recorded macro: '+str(p))
            require(row['actual_integrations'] == (160 if row['accepted_resolution'] == 64 else 352), 'Macro consumption differs')
        owners.append(dict(owner=name, integrations=expected, accepted_macros=len(macros), ledger=state))
    n1 = read(run/'single_owner_lifecycle/measurement.json')
    close = read(run/'single_owner_lifecycle/owner-closeout.json')
    require(n1['status'] == 'SHORT_NATIVE_PILOT_PASS' and close['all_clean'] is True, 'N1 cannot be reused')
    require(n1['final'][0]['macros'] == 50 and n1['final'][0]['public_steps'] == 5, 'N1 public completion differs')
    n2 = read(run/'two_process_short_pilot/owner-closeout.json')
    require(n2['all_clean'] is False and not n2['owners'][0]['report_exists'] and n2['owners'][0]['interrupted'] and n2['owners'][0]['exitcode'] == -15, 'N2 interrupted sibling differs')
    require(n2['owners'][1]['exitcode'] == 1 and not n2['owners'][1]['interrupted'], 'N2 failed owner differs')
    added = sum(row['integrations'] for row in owners)
    return dict(owners=owners, native_added=dict(completed=added,attempted=added),
                canonical_after={k:audit['canonical_before'][k]+added for k in ('completed','attempted')},
                baseline_added_once=True, historical_failure_retained=True, N1_reused=True, N2_passed=False)


def typed_worlds(worlds, np):
    result = copy.deepcopy(worlds)
    for row in result:
        for key in ('qpos','qvel'):
            row['endpoint'][key] = np.asarray(row['endpoint'][key], dtype=np.float32)
    return result


def assess_events(events, np, assess_one, assess_cohort):
    """Reconstruct float32 trial endpoints before invoking unchanged criteria."""
    previous, current, replay_candidate = None, None, None
    decisions, replays = [], []
    for event in events:
        kind, row = event['kind'], event['row']
        if kind == 'trial_complete':
            if row['role'] == 'replay':
                require(replay_candidate is not None, 'Replay lacks accepted candidate')
                exact = all(row[key] == replay_candidate[key] for key in ('stage_sha256','endpoint_sha256','endpoint_record','numeric_Data_hashes','worlds'))
                require(exact, 'Archived accepted replay differs from candidate')
                replays.append(dict(resolution=row['resolution'], exact=True))
            else:
                previous, current = current, row
        elif kind == 'decision':
            require(previous is not None and current is not None and current['resolution'] == row['resolution'], 'Decision trial order differs')
            recomputed = assess_cohort(typed_worlds(previous['worlds'],np), typed_worlds(current['worlds'],np), (0,), assess_one)
            recorded = {k:row[k] for k in recomputed}
            require(recorded == recomputed, 'Frozen float32 assessment differs from recorded decision')
            failed_checks = []
            for world in recomputed['worlds']:
                original = world['original']
                for group in ('state_error','event_agreement'):
                    for check in original[group]['checks']:
                        if not check['passed']:
                            failed_checks.append(dict(group=group, **check))
                if not original['initial_and_final_occupancy_equal']:
                    failed_checks.append(dict(group='occupancy',passed=False))
            decisions.append(dict(resolution=row['resolution'], recorded_equals_recomputed=True,
                                  accepted=recomputed['accepted'], failed_checks=failed_checks, decision=recomputed))
            if recomputed['accepted']:
                replay_candidate = current
    require(decisions, 'No decisions retained for macro')
    return dict(decisions=decisions, accepted_replays=replays)


def macro_review(folder, np, assess_one, assess_cohort):
    folder = Path(folder)
    inv = read(folder/'raw/evidence-inventory.json')
    for name, row in inv['files'].items():
        verify_file(folder/'raw', dict(path=name, **row))
    events = [read(safe_file(folder/'raw', name)) for name in inv['events']]
    require([e['sequence'] for e in events] == sorted(set(e['sequence'] for e in events)), 'Event sequence differs')
    result = assess_events(events,np,assess_one,assess_cohort)
    failure = (folder/'macro-failure.json').exists()
    if failure:
        require([r['resolution'] for r in result['decisions']] == [64,128] and not any(r['accepted'] for r in result['decisions']), 'Failure rejection roster differs')
        require(not result['accepted_replays'] and read(folder/'macro-failure.json')['driver_commits'] == 0, 'Failed macro was published')
    else:
        macro = read(folder/'macro-result.json')
        require(result['decisions'][-1]['accepted'] and len(result['accepted_replays']) == 1, 'Accepted macro lacks exact replay')
        require(result['decisions'][-1]['decision'] == macro['decision'], 'Accepted result decision differs')
    return dict(passed=True, native_acceptance=not failure, failure_expected=failure,
                recorded_rejection_is_not_native_PASS=failure, event_count=len(events), **result)


def resume_review(run, np):
    run = Path(run)
    folder = run/'two_process_short_pilot/owner-1/macro-27/raw'
    snapshot = read(folder/'record-00001-snapshot.json')
    checked = {}
    for label, key in (('data','Data'),('owners','owners')):
        with np.load(folder/('record-00001-'+label+'.npz'), allow_pickle=False) as arrays:
            require(set(arrays.files) == set(snapshot[key]), 'Snapshot array roster differs')
            for name, meta in snapshot[key].items():
                array = arrays[name]
                require(list(array.shape) == meta['shape'] and str(array.dtype) == meta['dtype'] and array.nbytes == meta['bytes'] and hashlib.sha256(array.tobytes()).hexdigest() == meta['sha256'], 'Snapshot array mismatch: '+name)
            checked[label] = len(arrays.files)
    return dict(raw_numeric_snapshot_verified=True, array_counts=checked, macro_start_tick=27,
                inflight_public_step=3, completed_public_steps=2, completed_macros_in_inflight_step=7,
                checkpoint_kind='numeric evidence and accounting; not a validated cross-process restart',
                native_resume_released=False, N1_reuse_released=True,
                reasons=['counter-checkpoint contains counts/pending/WAL sequence, not environment state',
                         'macro snapshot is inside env.step after action application, not a public step return boundary',
                         'snapshot has native/model/RNG fingerprints and process-bound topology; no validated loader for full runtime/IPC/cleanup state',
                         'failed tier128 must be resolved under original criteria before another native attempt'],
                resume_policy='Reuse completed CPU/source/N1 receipts now; never re-run N1 to repair N2; numerical continuation requires a separately verified restore contract and reconciled budget',
                PPO=0,cloud_training=0,new_integrations=0)
