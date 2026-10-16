"""Read-only known-failure/ledger review. No simulator imports or native calls.

This verifies the receipt failure, immutable archive, raw call evidence and
accounting before a separate repaired revision. It does not accept the prior
physical experiment or resume its published epoch.
"""
from pathlib import Path
import hashlib,json,zipfile
RUN_ID='20261014T134742Z-261213'
OLD_LAUNCHER='d158b03c59e9634aada22016c9148674660ff2ac38bbca32a33abed8041a8479'
OLD_BUNDLE='e3683e66cb8d1e99667842589212cdb65507889c55cf3f1b7ee666cd91efe7e5'
def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def integer_counts(value):
    return {k:v for k,v in value.items() if type(v) is int}
def merged(prior,current):
    return {k:prior.get(k,0)+current.get(k,0) for k in sorted(set(prior)|set(current))}
def verify_reports(finish,summary,review,durable,old_plan,reconciled):
    require(finish==dict(process_has_exited=True,exit_code=2,timed_out=False,interrupted=False,launch_error=None),'Previous execution was not the observed non-timeout failure')
    require(review['status']=='INDEPENDENT_WORLD_COORDINATOR_FAILED_REVIEW_REQUIRED' and review['owners_exited'] is True,'Previous coordinator/exit state differs')
    owner=review['owner_reports']['0']
    require(owner and owner['status']=='SINGLE_WORLD_NATIVE_FAILURE_REVIEW_REQUIRED' and owner.get('error')=="RuntimeError('Publication complete-owner receipt differs')",'Previous failure is not the reproduced receipt mismatch: '+str(owner and owner.get('error')))
    require('Owner failure: SINGLE_WORLD_NATIVE_FAILURE_REVIEW_REQUIRED' in review.get('error',''),'Previous coordinator failure chain differs')
    require(review['counts']==summary['counts']==durable and reconciled['consistent'] is True and reconciled['coordinator']==reconciled['owners']==durable,'Previous durable counters disagree')
    for name,expected in [('integrations',1344),('GPU_forward',5406),('GPU_solver',5406)]:
        require(durable[name+'_attempted']==durable[name+'_completed']==expected,'Observed partial native schedule differs: '+name)
    require(durable.get('PPO',0)==durable.get('cloud',0)==0,'Unexpected training history')
    require(summary['canonical_completed']==old_plan['canonical_completed_before']+1344==653999 and summary['canonical_attempted']==old_plan['canonical_attempted_before']+1344==654000,'Previous integration ledger differs')
    require(summary['prior_G09_call_ledger']==old_plan['prior_G09_call_ledger'] and summary['G09_calls_cumulative']==merged(old_plan['prior_G09_call_ledger'],durable),'Previous complete call ledger differs')
    for suffix in ('attempted','completed'):
        require(summary['G09_GPU_forward_cumulative_'+suffix]==old_plan['prior_GPU_forward_'+suffix]+5406==25939,'Previous forward history differs')
    series={kind+'_'+suffix:old_plan['prior_contact_graph_series'][kind+'_'+suffix]+durable.get('ContactGraph_'+new+'_'+suffix,0) for kind,new in [('capture','capture_begin'),('launch','launch')] for suffix in ('attempted','completed')}
    require(summary['contact_graph_series_cumulative']==series,'Previous graph history differs')
    require(all(summary.get(k) is True for k in ('git_state_unchanged','selected_sources_unchanged','old_evidence_rechecked_unchanged','counters_consistent')),'Previous source/evidence protection failed')
    require(summary['G09']=='OPEN' and summary['stage11_complete'] is False,'Prior failure must remain open')
    return summary

def verify_failed_run(root,current_git,expected_program):
    root=Path(root);base=root/'post-reset-independent-worlds';run=base/RUN_ID
    marker_path=base/'started-v1.json';marker=json.loads(marker_path.read_bytes())
    require(marker['run']==str(run) and marker['bundle_sha256']==OLD_BUNDLE,'Original failed-run started identity differs; retain it')
    archive=run/'post-reset-independent-worlds-return.zip'
    require(archive.is_file() and not archive.is_symlink() and 0<archive.stat().st_size<512*1024*1024,'Original return archive missing or invalid')
    archive_sha=sha(archive.read_bytes())
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();require(len(names)==len(set(names)),'Duplicate previous archive member')
        inv=json.loads(z.read('evidence-inventory.json'))
        require(set(names)==set(inv)|{'evidence-inventory.json'},'Original failed-run inventory differs')
        for name,row in inv.items():
            path=run/name
            require(path.resolve().is_relative_to(run.resolve()) and not path.is_symlink(),'Unsafe previous evidence path')
            raw=z.read(name);require(row==dict(bytes=len(raw),sha256=sha(raw)) and path.read_bytes()==raw,'Previous archived/local evidence changed: '+name)
        require((run/'evidence-inventory.json').read_bytes()==z.read('evidence-inventory.json'),'Previous inventory bytes changed')
        require(sha(z.read('launcher.py'))==OLD_LAUNCHER,'Previous launcher differs from reviewed source')
        program={n.removeprefix('program/'):dict(bytes=len(z.read(n)),sha256=sha(z.read(n))) for n in names if n.startswith('program/')}
        require(program==expected_program,'Previous executable bundle differs')
        load=lambda name:json.loads(z.read(name))
        old_plan=load('program/plan.json')
        require(load('git-before.json')==load('git-after.json')==current_git,'Project changed since previous run')
        require(load('source-after.json')['all_selected_sources_unchanged'] is True,'Previous source protection differs')
        owner_counts={str(w):integer_counts(load('probe/owner-'+str(w)+'/progress.json')) for w in (0,1)}
        durable=merged(owner_counts['0'],owner_counts['1'])
        summary=verify_reports(load('execution-finished.json'),load('run-summary.json'),load('probe/run-review.json'),durable,old_plan,load('owner-counts-reconciled.json'))
        publication=load('probe/publications/0002.json')
        require(publication['epoch']==2 and publication['protocol']=='stage11-independent-owners-publication-v1','Original failed publication differs')
        require(not any(n in names for n in ['probe/publications/0002-consumer.json','probe/publications/0002-closed.json']),'Previous failed publication unexpectedly consumed/closed')
        require(load('probe/barrier-0001.json')['abort_kind']=='predeclared_cancel_probe','Original cancellation missing')
        owner_evidence=[]
        for world in (0,1):
            prefix='probe/owner-'+str(world)+'/'
            prepared=load(prefix+'responses/0004.json');validation=load(prefix+'responses/0005.json')
            require(prepared==publication['states'][world] and prepared['sequence']==4 and prepared['state']=='prepared' and prepared['case']=='control' and prepared['mode']=='adaptive','Original first control receipt differs')
            require(validation['state']=='validated' and validation['state_sha256']==prepared['state_sha256'] and validation['unchanged'] is True,'Original complete state was not validated')
            pid=prepared['pid'];proc=Path('/proc')/str(pid)/'cmdline'
            if proc.is_file():require(str(run).encode() not in proc.read_bytes(),'Original owner still running; no overlapping execution')
            require(prepared['checks_passed'] is True,'Original control comparisons did not pass')
            counts=owner_counts[str(world)]
            require(counts['integrations_attempted']==counts['integrations_completed']==672 and counts['GPU_forward_completed']==counts['GPU_solver_completed']==2703,'Owner partial native count differs')
            trial_count=0;microstep_count=0;rows=[]
            for folder,n,steps in [('macro-raw',128,352),('continuation-cancel-control-adaptive-raw',64,160),('continuation-control-adaptive-raw',64,160)]:
                events=[load(name) for name in sorted(names) if name.startswith(prefix+folder+'/record-') and name.endswith('.json') and not name.endswith('-snapshot.json')]
                trials=[e['row'] for e in events if e['kind']=='trial_complete'];micros=[e['row'] for e in events if e['kind']=='microstep']
                require(len(micros)==steps and [t['resolution'] for t in trials]==([32,64,128,128] if n==128 else [32,64,64]),'Archived partial trial schedule differs')
                require(trials[-1]['role']=='replay' and trials[-1]['resolution']==n and all(len(m['stages'])==4 and all(len(s['qpos'])==len(s['qvel'])==1 for s in m['stages']) for m in micros),'Archived singleton replay/stage roster differs')
                trial_count+=len(trials);microstep_count+=len(micros);rows.append(dict(folder=folder,microsteps=len(micros),trials=len(trials),accepted_resolution=n))
            require(microstep_count==672 and 4*microstep_count+trial_count+5==2703,'Raw stage/forward accounting differs')
            require(len(z.read(prefix+'full-contact/index.jsonl').splitlines())==2703,'Full Contact evidence count differs')
            for name in names:
                if name.startswith(prefix+'commands/'):
                    command=load(name)
                    require(not (command['kind']=='ABORT' and command.get('epoch')==2),'Previously published epoch was rolled back')
            owner_evidence.append(dict(logical_world=world,pid=pid,raw_schedule=rows,transmitted_sequence=4,old_internal_receipt_omitted_sequence=True))
    return dict(status='KNOWN_RECEIPT_FAILURE_AND_LEDGER_VERIFIED',run=str(run),archive_sha256=archive_sha,marker_sha256=sha(marker_path.read_bytes()),launcher_sha256=OLD_LAUNCHER,bundle_sha256=OLD_BUNDLE,
        timeout_caused_failure=False,physical_acceptance=False,complete_physical_audit=False,native_calls_in_reviewer=0,owner_evidence=owner_evidence,
        summary=summary,old_started_and_published_epoch_retained=True)

def carry_forward(plan,verified):
    result=json.loads(json.dumps(plan));prior=verified['summary']
    result.update(prior_G09_call_ledger=prior['G09_calls_cumulative'],prior_contact_graph_series=prior['contact_graph_series_cumulative'],
        canonical_completed_before=prior['canonical_completed'],canonical_attempted_before=prior['canonical_attempted'],
        prior_GPU_forward_completed=prior['G09_GPU_forward_cumulative_completed'],prior_GPU_forward_attempted=prior['G09_GPU_forward_cumulative_attempted'],
        failed_predecessor=dict(run=verified['run'],archive_sha256=verified['archive_sha256'],status=verified['status'],physical_acceptance=False))
    return result
