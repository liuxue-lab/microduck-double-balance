"""Sequential four-arm pilot; only development reports can select a candidate."""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from mjlab_microduck.double_balance_stage09_budget import BudgetLedger, atomic_json, validate_local_runtime
from mjlab_microduck.double_balance_stage09_plan import PROFILES, PRIMARY_SHA


def run_campaign(args):
    from mjlab_microduck.double_balance_stage09 import MODULE,package_review
    from mjlab_microduck.double_balance_stage09_state import (
        preflight,load_checked,require,file_sha256,utc_now,
    )
    from mjlab_microduck.double_balance_stage09_evaluation import PROTOCOL_VERSION
    head=preflight(cloud=True);ledger=BudgetLedger(args.ledger)
    _,_,checksum=load_checked(args.checkpoint)
    require(checksum==PRIMARY_SHA,'Campaign source must be the approved main model')
    repo=Path(__file__).resolve().parents[2]
    local=json.loads(args.local_report.read_text())
    require(local['status']=='ZERO_PPO_RUNTIME_CHECKS_COMPLETE_REVIEW_PENDING' and
            local['source_sha256']==PRIMARY_SHA and local['new_ppo_updates']==0 and
            validate_local_runtime(repo,local['runtime_manifest_sha256']),
            'Matching successful laptop zero-PPO checks are required')
    args.output.mkdir(parents=True,exist_ok=True)
    path=args.output/'campaign.json'
    if path.exists():
        state=json.loads(path.read_text())
        require(state['campaign_id']==ledger.identity and state['source_sha256']==checksum and
                state['runtime_manifest_sha256']==local['runtime_manifest_sha256'],
                'Resume cannot switch budget/source/runtime')
    else:
        state=dict(status='INITIALIZING',campaign_id=ledger.identity,source_sha256=checksum,
                   runtime_manifest_sha256=local['runtime_manifest_sha256'],git_head=head,
                   started_utc=utc_now(),profiles={},evaluations={},stage09_complete=False)
        atomic_json(path,state)

    # Original evidence/campaign identity stays frozen; record the active code separately.
    state['active_git_head'] = head
    state['active_runtime_manifest_sha256'] = file_sha256(repo/'docs/audits/stage-09-runtime-hashes.json')
    recovery = ledger.snapshot().get('approved_recovery')
    if recovery is not None:
        state['approved_recovery'] = recovery
    atomic_json(path,state)

    def job(argv, log_path, *, training):
        remaining=ledger.remaining(training=training)
        require(remaining>180,'Insufficient remaining job time')
        with log_path.open('x') as log:
            p=subprocess.run([sys.executable,'-u','-m',MODULE,*argv,'--worker','--ledger',str(args.ledger)],
                             stdout=log,stderr=subprocess.STDOUT,timeout=max(1,remaining-90))
        return p.returncode

    def evaluation(checkpoint, label, protocol, selection=None):
        key=label+'/'+protocol
        checksum=file_sha256(checkpoint)
        if key in state['evaluations']:
            item=state['evaluations'][key]
            require(file_sha256(item['path'])==item['sha256'],'Completed evaluation report changed')
            report=json.loads(Path(item['path']).read_text())
            require(report['checkpoint_sha256']==checksum and report['status']=='PASS','Stale evaluation')
            return report
        folder=args.output/'evaluations'/label/(protocol+'-'+uuid.uuid4().hex[:8])
        folder.parent.mkdir(parents=True,exist_ok=True)
        argv=['evaluate','--checkpoint',str(checkpoint),'--output',str(folder),
              '--datasets',str(args.datasets),'--protocol',protocol]
        if selection is not None:argv+=['--selection',str(selection)]
        code=job(argv,folder.with_suffix('.log'),training=False)
        require(code==0,'Evaluation failed; keep this campaign and inspect its log')
        report=json.loads((folder/'evaluation.json').read_text())
        require(report['status']=='PASS' and report['checkpoint_sha256']==checksum,'Evaluation identity mismatch')
        state['evaluations'][key]=dict(path=str(folder/'evaluation.json'),sha256=file_sha256(folder/'evaluation.json'))
        atomic_json(path,state)
        return report

    try:
        for protocol in ('nominal','dev'):evaluation(args.checkpoint,'baseline',protocol)
        for profile in PROFILES:
            entry=state['profiles'].setdefault(profile,dict(completed=0,checkpoint=str(args.checkpoint),segments=[]))
            # Recover a validated save from an interrupted segment without rolling back charges.
            for segment in entry['segments']:
                report_path=Path(segment)/'training.json'
                if not report_path.exists():continue
                saved=json.loads(report_path.read_text()).get('latest_checkpoint')
                if saved:
                    _,meta,_=load_checked(saved,profile=profile,campaign_id=ledger.identity)
                    if meta['experiment_completed_updates']>=entry['completed']:
                        entry.update(completed=meta['experiment_completed_updates'],checkpoint=saved)
            for milestone in (250,500):
                if entry['completed']<milestone:
                    segment=args.output/profile/('segment-'+uuid.uuid4().hex[:8]);segment.parent.mkdir(parents=True,exist_ok=True)
                    entry['segments'].append(str(segment));atomic_json(path,state)
                    code=job(['train','--profile',profile,'--target',str(milestone),
                              '--checkpoint',entry['checkpoint'],'--output',str(segment),'--datasets',str(args.datasets)],
                              segment.with_suffix('.log'),training=True)
                    report=json.loads((segment/'training.json').read_text())
                    if report.get('latest_checkpoint'):
                        _,meta,_=load_checked(report['latest_checkpoint'],profile=profile,campaign_id=ledger.identity)
                        entry.update(completed=meta['experiment_completed_updates'],checkpoint=report['latest_checkpoint'])
                    atomic_json(path,state)
                    require(code==0 and report['status']=='TRAINING_COMPLETE' and entry['completed']==milestone,
                            'Training segment stopped; no automatic retry or budget reset')
                # If a resumed branch is already at 500, the completed 250 reports remain required.
                label=f'{profile}-u{milestone:06d}'
                if entry['completed']>milestone:
                    require(all(label+'/'+p in state['evaluations'] for p in ('nominal','dev')),
                            'Earlier milestone evaluation is missing; inspect campaign before continuing')
                    continue
                for protocol in ('nominal','dev'):evaluation(entry['checkpoint'],label,protocol)
        baseline=evaluation(args.checkpoint,'baseline','dev')
        candidates=[]
        for profile,entry in state['profiles'].items():
            report=evaluation(entry['checkpoint'],f'{profile}-u000500','dev')
            pose=report['posture_summary'];base_pose=baseline['posture_summary']
            finite=pose['tail_abs_yaw_median_deg'] is not None and pose['tail_head_margin_bad_fraction'] is not None
            eligible=(finite and report['successes']>=baseline['successes']-4 and
                      pose['tail_abs_yaw_median_deg']<base_pose['tail_abs_yaw_median_deg'] and
                      pose['tail_head_margin_bad_fraction']<base_pose['tail_head_margin_bad_fraction'])
            entry['eligible_by_development']=eligible
            if eligible:
                rank=(pose['successes'],-pose['tail_abs_yaw_median_deg'],
                      -pose['tail_head_margin_bad_fraction'],report['successes'])
                candidates.append((rank,profile,entry['checkpoint']))
        selection_path=args.output/'selection.json'
        checkpoints=[dict(role='frozen-baseline',path=str(args.checkpoint),sha256=PRIMARY_SHA)]
        if candidates:
            _,profile,checkpoint=max(candidates)
            checkpoints.append(dict(role='candidate',profile=profile,path=checkpoint,sha256=file_sha256(checkpoint)))
        selection=dict(schema_version=1,selection_locked=True,protocol_version=PROTOCOL_VERSION,
                       created_utc=utc_now(),checkpoints=checkpoints,
                       development_reports=[v for k,v in state['evaluations'].items() if k.endswith('/dev') or k.endswith('/nominal')],
                       rule='dev old-success >= baseline-4/128; improve yaw and margin; rank posture successes, yaw, margin, old successes',
                       single_training_seed=True,old_results_replaced=False)
        if selection_path.exists():
            existing=json.loads(selection_path.read_text())
            require(existing['checkpoints']==checkpoints,'Frozen selection cannot be replaced')
        else:atomic_json(selection_path,selection)
        state['selected_checkpoints']=checkpoints;atomic_json(path,state)
        if len(checkpoints)==2:
            for item in checkpoints:
                for protocol in ('test-930901','test-930902','test-930903'):
                    evaluation(Path(item['path']),item['role'],protocol,selection=selection_path)
            state['status']='PILOT_AND_FRESH_EVALUATIONS_COMPLETE_VIDEO_REVIEW_PENDING'
        else:
            state['status']='PILOT_COMPLETE_NO_ELIGIBLE_CANDIDATE'
        archive=[]
        for profile,entry in state['profiles'].items():
            for segment in entry['segments']:
                for n in (0,250,500):
                    cp=Path(segment)/'checkpoints'/f'update_{n:06d}.pt'
                    if cp.exists():archive.append(dict(profile=profile,updates=n,path=str(cp),sha256=file_sha256(cp),size_bytes=cp.stat().st_size))
        atomic_json(args.output/'checkpoint-archive.json',dict(source_sha256=PRIMARY_SHA,checkpoints=archive,
                    copied_off_cloud=False,video_review='PENDING',stage09_complete=False))
    except BaseException as exc:
        state.update(status='PAUSED_OR_FAILED_REVIEW_REQUIRED',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        state.update(last_utc=utc_now(),budget=ledger.snapshot(),power_operations_performed=False)
        atomic_json(path,state)
        result=package_review(args.output)
        print('Stage09Campaign='+state['status'],flush=True)
        print('Stage09Review='+result['path'],flush=True)
        print('Stage09ReviewSHA256='+result['sha256'],flush=True)
