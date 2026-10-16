"""Read-only validation of the independently reviewed, commit-pinned failure.

No simulator imports or native calls. The old archive and every local member
must still match. This does not resume the old epoch or accept its physics.
"""
from pathlib import Path,PurePosixPath
import hashlib,json,zipfile
RUN_ID='20261014T143302Z-6a8be2'
ARCHIVE_SHA='41fce460804333fdb28cc2dae47cded294f10b5109e3c27ad7c79b22ad09e12c'
LAUNCHER_SHA='9a900e6e16720ad703678393afbb0597982509d936ebd565ea8fa507617e3582'
BUNDLE_SHA='80923a948ccdc81b5ad222942c9631e195ad90a6a83113e11bf17f7fb556d4c0'
COMMIT='99dd39dc52349f43821b8334bb9b16226be5881b'
def need(ok,message):
    if not ok:raise RuntimeError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def merged(a,b):return {k:a.get(k,0)+b.get(k,0) for k in set(a)|set(b)}
def verify_reports(summary,review,plan,durable,previous):
    need(review['status']=='INDEPENDENT_WORLD_COORDINATOR_FAILED_REVIEW_REQUIRED' and review['owners_exited'] is True,'Previous failure/exit differs')
    need(review['owner_reports']['1']['error']=="RuntimeError('Exact endpoint replay failed; rollback required')",'Previous failure cause differs')
    need(summary['counts']==review['counts']==durable,'Previous durable counters differ')
    for n,v in [('integrations',2880),('GPU_forward',11575),('GPU_solver',11575)]:
        need(durable[n+'_attempted']==durable[n+'_completed']==v,'Previous raw call totals differ: '+n)
    need(plan['prior_G09_call_ledger']==previous['G09_calls_cumulative'] and summary['prior_G09_call_ledger']==plan['prior_G09_call_ledger'],'Earlier failure ledger lost')
    need(summary['G09_calls_cumulative']==merged(plan['prior_G09_call_ledger'],durable),'Complete cumulative call ledger differs')
    need(summary['canonical_completed']==plan['canonical_completed_before']+2880==656879 and summary['canonical_attempted']==plan['canonical_attempted_before']+2880==656880,'Canonical history differs')
    for suffix in ('attempted','completed'):
        need(summary['G09_GPU_forward_cumulative_'+suffix]==plan['prior_GPU_forward_'+suffix]+11575==37514,'Forward history differs')
    graph={kind+'_'+s:plan['prior_contact_graph_series'][kind+'_'+s]+durable['ContactGraph_'+native+'_'+s] for kind,native in [('capture','capture_begin'),('launch','launch')] for s in ('attempted','completed')}
    need(summary['contact_graph_series_cumulative']==graph,'Graph history differs')
    need(all(summary.get(k) is True for k in ('git_state_unchanged','selected_sources_unchanged','old_evidence_rechecked_unchanged','counters_consistent')),'Previous immutable evidence check failed')
    need(summary['G09']=='OPEN' and summary['stage11_complete'] is False and durable.get('PPO',0)==durable.get('cloud',0)==0,'Invalid prior acceptance/training claim')
    return summary
def verify_receipt_run(root,current_git,expected_program,previous):
    root=Path(root);base=root/'post-reset-independent-worlds-receipt-repair';run=base/RUN_ID
    marker_path=base/'started-v1.json';marker=json.loads(marker_path.read_bytes())
    need(marker['run']==str(run) and marker['bundle_sha256']==BUNDLE_SHA,'Receipt-repair started marker differs; preserve it')
    archive=run/'post-reset-independent-worlds-receipt-repair-return.zip'
    need(archive.is_file() and not archive.is_symlink() and archive.stat().st_size==196926339,'Reviewed failure archive size differs')
    need(sha(archive.read_bytes())==ARCHIVE_SHA,'Reviewed failure archive SHA256 differs')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();inv=json.loads(z.read('evidence-inventory.json'))
        need(len(names)==len(set(names))==26760 and set(names)==set(inv)|{'evidence-inventory.json'},'Reviewed archive roster differs')
        for name,row in inv.items():
            path=run/name;posix=PurePosixPath(name)
            need(not posix.is_absolute() and '..' not in posix.parts and path.resolve().is_relative_to(run.resolve()) and not path.is_symlink(),'Unsafe previous evidence path')
            raw=z.read(name);need(row==dict(bytes=len(raw),sha256=sha(raw)) and path.read_bytes()==raw,'Archived/local previous evidence changed: '+name)
        need((run/'evidence-inventory.json').read_bytes()==z.read('evidence-inventory.json'),'Previous inventory changed')
        need(sha(z.read('launcher.py'))==LAUNCHER_SHA,'Previous launcher differs')
        program={n.removeprefix('program/'):dict(bytes=len(z.read(n)),sha256=sha(z.read(n))) for n in names if n.startswith('program/')}
        need(program==expected_program,'Previous executable bundle differs')
        load=lambda name:json.loads(z.read(name))
        need(load('execution-finished.json')==dict(process_has_exited=True,exit_code=2,timed_out=False,interrupted=False,launch_error=None),'Previous failure was interrupted or timed out')
        need(load('git-before.json')==load('git-after.json')==current_git,'Project changed since reviewed failure')
        counts=[{k:v for k,v in load('probe/owner-'+str(w)+'/progress.json').items() if type(v) is int} for w in (0,1)]
        summary=verify_reports(load('run-summary.json'),load('probe/run-review.json'),load('runtime-plan.json'),merged(*counts),previous)
        phase=load('probe/phase-progress.json');need(phase['completed_units']==7 and phase['total_units']==12,'Previous phase count differs')
        need('probe/publications/0006.json' not in names,'Failed reset-1 case was published')
        for epoch in range(2,6):
            pub=load('probe/publications/%04d.json'%epoch);consumer=load('probe/publications/%04d-consumer.json'%epoch);closed=load('probe/publications/%04d-closed.json'%epoch)
            need(consumer['complete_pair_read'] is True and closed['leases_released'] is True,'Earlier successful publication not consumed/closed')
            for w in (0,1):
                wire=pub['states'][w];need(wire==load('probe/owner-%d/responses/%04d.json'%(w,wire['sequence'])),'Previous transmitted receipt differs')
        for w in (0,1):
            owner=load('probe/owner-%d/run-review.json'%w);proc=Path('/proc')/str(owner['pid'])/'cmdline'
            if proc.is_file():need(str(run).encode() not in proc.read_bytes(),'Previous owner remains active')
        last=load('probe/owner-1/continuation-reset-1-adaptive-raw/record-00371.json')
        need(last['kind']=='rollback' and last['row']['reason']=='exception_abort' and all(last['row'][k] is True for k in ('data_exact','owners_exact','rng_exact','model_exact','bindings_exact')),'Failed private replay was not restored')
    return dict(status='KNOWN_SOLVER_BOUNDARY_REPLAY_FAILURE_VERIFIED',run=str(run),archive_sha256=ARCHIVE_SHA,commit=COMMIT,marker_sha256=sha(marker_path.read_bytes()),summary=summary,
        timeout_caused_failure=False,physical_acceptance=False,native_calls_in_reviewer=0,phase_percent=100*7/12,completed_joint_cases=4,first_divergence=dict(world=1,case='reset-1-adaptive',micro_index=0,stage=0),old_started_preserved=True)
