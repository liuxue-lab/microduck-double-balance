"""Standard-library, read-only end-of-run collection; no simulator imports."""
import hashlib
import json
import math
from pathlib import Path
import statistics

EXPECTED_PACKAGE = 'dd6d1e1e697bcaa348dbdd92dd8c7b30ea31ae9c32599338106859ae7657523a'
MAX_MODEL_BYTES = 64*1024**2
MAX_METRICS_BYTES = 64*1024**2


def read_small(path):
    path=Path(path)
    if not path.exists():
        return None
    if path.stat().st_size>65536:
        raise ValueError('Receipt exceeds 64 KiB: '+str(path))
    value=json.loads(path.read_bytes())
    if not isinstance(value,dict):
        raise ValueError('Receipt is not an object: '+str(path))
    return value


def process_alive(pin):
    if not pin:
        return None
    try:
        row=(Path('/proc')/str(int(pin['pid']))/'stat').read_text().rsplit(')',1)[1].split()
        return row[19]==pin['start'] and row[0] not in ('Z','X')
    except FileNotFoundError:
        return False
    except (OSError,KeyError,ValueError,IndexError):
        return None


def snapshot(slot):
    slot=Path(slot)
    files={}
    for name in ('started.json','budget.json','supervisor.json','child.json','run/progress.json'):
        files[name]=read_small(slot/name)
    pins={kind:process_alive(files[kind+'.json']) for kind in ('supervisor','child')}
    exists=bool(files['started.json'] or files['budget.json'])
    if not exists:
        status='NO_CAMPAIGN_FOUND'
    elif any(v is True for v in pins.values()):
        status='RUNNING'
    elif all(v is False for v in pins.values()):
        status='STOPPED'
    else:
        status='PROCESS_STATE_UNKNOWN'
    return dict(status=status,slot=str(slot),files=files,processes_alive=pins,
                read_only=True,added_PPO_updates=0,added_simulation_steps=0,
                snapshot_is_atomic=False,automatic_power_actions=0)


def verify_checkpoint(slot, receipt):
    if not isinstance(receipt,dict):
        return dict(verified=False,error='Missing checkpoint receipt')
    try:
        base=(Path(slot)/'run/checkpoints').resolve(strict=True)
        path=Path(receipt['path'])
        if path.is_symlink() or path.resolve(strict=True).parent != base or not path.is_file():
            raise ValueError('Checkpoint is outside the recorded checkpoint directory or is a symlink')
        before=path.stat()
        if not (0<before.st_size<=MAX_MODEL_BYTES and before.st_size==receipt['bytes']):
            raise ValueError('Checkpoint size does not match receipt or exceeds read limit')
        digest=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1024**2),b''):
                digest.update(chunk)
        after=path.stat()
        if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
            raise ValueError('Checkpoint changed while being read')
        actual=digest.hexdigest()
        if actual!=receipt['sha256']:
            raise ValueError('Checkpoint SHA256 differs from saved receipt')
        return dict(verified=True,path=str(path),bytes=before.st_size,sha256=actual,
                    completed_updates=receipt.get('completed_updates'),policy_updates=receipt.get('policy_updates'))
    except (OSError,KeyError,TypeError,ValueError) as exc:
        return dict(verified=False,error=repr(exc),receipt=receipt)


def metric_history(path):
    path=Path(path)
    if not path.is_file():
        return dict(available=False,reason='No metrics history')
    if path.stat().st_size>MAX_METRICS_BYTES:
        return dict(available=False,reason='Metrics history exceeds 64 MiB read limit; file preserved')
    rows={};bad=duplicates=0
    fields={'lower':'lower_unassisted_current_fraction','joint':'joint_geometric_current_fraction',
            'reward':'mean_reward','episode_length_steps':'mean_episode_length_steps'}
    with path.open('rb') as f:
        while True:
            line=f.readline(65537)
            if not line:
                break
            if len(line)>65536:
                return dict(available=False,reason='Oversized metrics line; file preserved')
            try:
                value=json.loads(line)
                if not isinstance(value,dict) or value.get('campaign_phase')!='formal':
                    continue
                update=value['completed_updates']
                if not isinstance(update,int) or update<0:
                    raise ValueError('Invalid update number')
                if update in rows:
                    duplicates+=1
                row={key:value.get(source) for key,source in fields.items()}
                row.update(update=update,head_scale=value.get('head_reward_scale'),
                    elapsed=value.get('elapsed_seconds'),session=value.get('session'),
                    nan_resets=value.get('nan_resets_in_last_rollout'),
                    success_5s=value.get('episode_metrics',{}).get('Episode_Metrics/native_joint_geometric_success_5s'))
                rows[update]=row
            except (ValueError,KeyError,TypeError,AttributeError):
                bad+=1
    ordered=[rows[k] for k in sorted(rows)]
    def finite(x):return isinstance(x,(float,int)) and math.isfinite(x)
    def summarize(group):
        out=dict(first_update=group[0]['update'],last_update=group[-1]['update'],logged_updates=len(group))
        for name in (*fields,'success_5s'):
            values=[r[name] for r in group if finite(r.get(name))]
            out[name+'_mean']=statistics.mean(values) if values else None
            if name=='success_5s':
                out['success_5s_max']=max(values) if values else None
                out['positive_success_logged_updates']=sum(x>0 for x in values)
        return out
    groups={}
    for row in ordered:
        groups.setdefault((max(1,row['update'])-1)//250,[]).append(row)
    windows=[summarize(group) for group in groups.values()]
    totals=summarize(ordered) if ordered else None
    return dict(available=True,valid_logged_updates=len(ordered),invalid_lines=bad,duplicate_updates=duplicates,
                windows_250_updates=windows,whole_run=totals,last_logged=ordered[-1] if ordered else None,
                metrics_are_training_aggregates=True,episode_weighted_success_rate_available=False,
                interpretation='success_5s is a logged mean over reset groups, then over logged steps; it is not an independent episode success rate.')


def collect_finished(slot):
    result=snapshot(slot)
    # Never hash or scan while either worker might still be running.
    if result['status']!='STOPPED':
        return result
    slot=Path(slot)
    for name in ('initialization.json','config.json','run/latest.json','run/best-training-metric.json',
                 'run/smoke-result.json','run/transfer.json','run/formal-load.json'):
        result['files'][name]=read_small(slot/name)
    result['exit_receipts']={p.name:read_small(p) for p in sorted(slot.glob('exit-*.json'))[-8:]}
    result['session_results']={p.name:read_small(p) for p in sorted((slot/'run').glob('session-*-result.json'))[-8:]}
    result['history']=metric_history(slot/'run/metrics.jsonl')
    latest=result['files']['run/latest.json']
    best=result['files']['run/best-training-metric.json']
    result['checkpoints']={'latest':verify_checkpoint(slot,latest)}
    if best:
        if latest==best.get('checkpoint'):
            result['checkpoints']['best_training_metric']=dict(result['checkpoints']['latest'])
        else:
            result['checkpoints']['best_training_metric']=verify_checkpoint(slot,best.get('checkpoint'))
    b=result['files']['budget.json'] or {};p=result['files']['run/progress.json'] or {}
    init=result['files']['started.json'] or {};smoke=result['files']['run/smoke-result.json'] or {}
    latest_verified=result['checkpoints']['latest'].get('verified',False)
    counts_match=(isinstance(b.get('completed_updates'),int) and b.get('completed_updates')==p.get('completed_updates')
                  and (latest or {}).get('completed_updates')==b.get('completed_updates'))
    update_budget_exhausted=(isinstance(b.get('max_updates'),int) and b.get('attempted_updates')==b.get('max_updates')
                            and isinstance(b.get('completed_updates'),int) and b['completed_updates']<=b['attempted_updates'])
    result['summary']=dict(
        training_exit_status=p.get('status'),completed_updates=b.get('completed_updates'),
        attempted_updates=b.get('attempted_updates'),max_updates=b.get('max_updates'),
        original_deadline_utc=b.get('deadline_utc'),pending_attempt=b.get('pending'),
        original_package_identity_matches=init.get('package_identity')==EXPECTED_PACKAGE,
        one_smoke_pass_recorded=smoke.get('status')=='SMOKE_PASSED' and smoke.get('completed_updates')==5,
        latest_checkpoint_matches_completed_count=counts_match,
        update_budget_exhausted=update_budget_exhausted,
        available_checkpoint_bytes_verified=all(x.get('verified') for x in result['checkpoints'].values()),
        finished_training_run_confirmed=(p.get('status')=='TRAINING_BUDGET_COMPLETE' and latest_verified and counts_match and update_budget_exhausted
                                        and b.get('pending') is False and init.get('package_identity')==EXPECTED_PACKAGE),
        independent_policy_evaluation_done=False,training_success_claimed=False,
        models_downloaded=False,models_and_full_logs_retained_cloud_side=True)
    result['status']='FINISHED_REPORT_READY'
    return result
