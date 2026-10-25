"""Read-only identity, ledger and preservation gate for the completed old matrix."""
from pathlib import Path,PurePosixPath
import hashlib,json,zipfile
RUN_ID='20261014T154645Z-42ffbc'
ARCHIVE_SHA='526c0c8fc65f7c8c40472cf8f3aaa295643156c98238ae84ffbdddebdf86950a'
BUNDLE_SHA='c4f82567194423e536afa8ac2a0d10ee46adba4027593c663fd407915be87778'
LAUNCHER_SHA='f8c41a9aeedf0ef8812972fa474b75e58f909f1396dc49c1dc423babe6b08ef4'
def need(ok,message):
    if not ok:raise RuntimeError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def merged(a,b):return {k:a.get(k,0)+b.get(k,0) for k in set(a)|set(b)}
def verify_reports(summary,review,plan,durable,previous):
    need(review['status']=='INDEPENDENT_WORLD_MATRIX_COMPARISON_FAILURES_REVIEW_REQUIRED' and review['owners_exited'] is True,'Previous matrix/exit differs')
    need(summary['counts']==review['counts']==durable,'Previous durable counters differ')
    for n,v in [('integrations',3712),('GPU_forward',14920),('GPU_solver',14920)]:
        need(durable[n+'_attempted']==durable[n+'_completed']==v,'Previous call total differs: '+n)
    need(plan['prior_G09_call_ledger']==previous['G09_calls_cumulative'] and summary['prior_G09_call_ledger']==plan['prior_G09_call_ledger'],'Earlier ledger lost')
    need(summary['G09_calls_cumulative']==merged(plan['prior_G09_call_ledger'],durable),'Cumulative call ledger differs')
    need(summary['canonical_completed']==plan['canonical_completed_before']+3712==660591 and summary['canonical_attempted']==plan['canonical_attempted_before']+3712==660592,'Canonical history differs')
    for suffix in ('attempted','completed'):
        need(summary['G09_GPU_forward_cumulative_'+suffix]==plan['prior_GPU_forward_'+suffix]+14920==52434,'Forward history differs')
    graph={kind+'_'+s:plan['prior_contact_graph_series'][kind+'_'+s]+durable['ContactGraph_'+native+'_'+s] for kind,native in [('capture','capture_begin'),('launch','launch')] for s in ('attempted','completed')}
    need(summary['contact_graph_series_cumulative']==graph,'Graph history differs')
    failures=[]
    for world in ('0','1'):
        owner=review['owner_reports'][world]
        need(owner['all_cases_completed'] and owner['original_material_Data_Model_owners_RNG_restored'],'Old owner restoration/completion missing')
        failures += [(world,row['case'],row['mode'],key) for row in owner['continuation_cases'] for key,value in row.items() if key.startswith('check_') and not value['passed']]
    need(failures==[('1','reset-1','reference','check_historical_paired_reference')],'Original failed comparison differs')
    need(all(summary.get(k) is True for k in ('git_state_unchanged','selected_sources_unchanged','old_evidence_rechecked_unchanged','counters_consistent')),'Previous immutable evidence check failed')
    need(summary['G09']=='OPEN' and summary['stage11_complete'] is False and durable.get('PPO',0)==durable.get('cloud',0)==0,'Invalid prior acceptance/training claim')
    return summary

def verify_solver_run(root,current_git,expected_program,previous):
    root=Path(root);base=root/'post-reset-independent-worlds-solver-order';run=base/RUN_ID
    marker_path=base/'started-v1.json';marker=json.loads(marker_path.read_bytes())
    need(marker['run']==str(run) and marker['bundle_sha256']==BUNDLE_SHA,'Solver-order started marker differs; preserve it')
    archive=run/'post-reset-independent-worlds-solver-order-return.zip'
    need(archive.is_file() and not archive.is_symlink() and archive.stat().st_size==172458001 and sha(archive.read_bytes())==ARCHIVE_SHA,'Reviewed solver archive identity differs')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();inv=json.loads(z.read('evidence-inventory.json'))
        need(len(names)==len(set(names))==25864 and set(names)==set(inv)|{'evidence-inventory.json'},'Reviewed archive roster differs')
        for name,row in inv.items():
            path=run/name;posix=PurePosixPath(name)
            need(not posix.is_absolute() and '..' not in posix.parts and path.resolve().is_relative_to(run.resolve()) and not path.is_symlink(),'Unsafe previous evidence path')
            raw=z.read(name);need(row==dict(bytes=len(raw),sha256=sha(raw)) and path.read_bytes()==raw,'Archived/local solver evidence changed: '+name)
        need((run/'evidence-inventory.json').read_bytes()==z.read('evidence-inventory.json'),'Previous inventory changed')
        need(sha(z.read('launcher.py'))==LAUNCHER_SHA,'Previous launcher differs')
        program={n.removeprefix('program/'):dict(bytes=len(z.read(n)),sha256=sha(z.read(n))) for n in names if n.startswith('program/')}
        need(program==expected_program,'Previous executable bundle differs')
        load=lambda name:json.loads(z.read(name))
        need(load('execution-finished.json')==dict(process_has_exited=True,exit_code=2,timed_out=False,interrupted=False,launch_error=None),'Previous execution was interrupted or timed out')
        need(load('git-before.json')==load('git-after.json')==current_git,'Project changed since reviewed matrix')
        counters=[{k:v for k,v in load('probe/owner-'+str(w)+'/progress.json').items() if type(v) is int} for w in (0,1)]
        summary=verify_reports(load('run-summary.json'),load('probe/run-review.json'),load('runtime-plan.json'),merged(*counters),previous)
        phase=load('probe/phase-progress.json');need(phase['completed_units']==phase['total_units']==12,'Previous matrix incomplete')
        need('probe/publications/0007.json' not in names,'Failed historical comparison was jointly published')
        for epoch in (2,3,4,5,6,8,9):
            pub=load('probe/publications/%04d.json'%epoch);consumer=load('probe/publications/%04d-consumer.json'%epoch);closed=load('probe/publications/%04d-closed.json'%epoch)
            need(consumer['complete_pair_read'] is True and closed['leases_released'] is True,'Previous publication not consumed/closed')
            for w in (0,1):
                wire=pub['states'][w];need(wire==load('probe/owner-%d/responses/%04d.json'%(w,wire['sequence'])),'Previous transmitted receipt differs')
        for w in (0,1):
            owner=load('probe/owner-%d/run-review.json'%w);proc=Path('/proc')/str(owner['pid'])/'cmdline'
            if proc.is_file():need(str(run).encode() not in proc.read_bytes(),'Previous owner remains active')
    return dict(status='KNOWN_INPUT_PRECONDITION_DEFECT_AND_OLD_FAILURE_PRESERVED',run=str(run),archive_sha256=ARCHIVE_SHA,
        commit='be69afb31022e035c1ced663f1754cc4b39adeab',marker_sha256=sha(marker_path.read_bytes()),summary=summary,
        timeout_caused_failure=False,physical_acceptance=False,native_calls_in_reviewer=0,phase_percent=100.0,
        first_divergence=dict(world=1,case='reset-1',boundary='startup BAM parameters and pre-integration control'),old_started_preserved=True)
