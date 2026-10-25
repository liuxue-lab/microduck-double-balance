"""Read-only preservation gate for the independently audited startup target."""
from pathlib import Path,PurePosixPath
import hashlib,json,zipfile
RUN_ID='20261014T172835Z-1dfc5c'
ARCHIVE_SHA='c135aed5d7a167c90a29ce9402de40ce3e16bf50a82e02a20c23c70aadcd5758'
BUNDLE_SHA='26a4774dc4959b78006fb344e1c0c01b98fcc1b539ad2913a51872b586a8aeca'
LAUNCHER_SHA='e26a620fd7f3219d7905e9e6317556f36f300414430760bf041af1f8013bac73'
def need(ok,message):
    if not ok:raise RuntimeError(message)
def sha(raw):return hashlib.sha256(raw).hexdigest()
def merged(a,b):return {k:a.get(k,0)+b.get(k,0) for k in set(a)|set(b)}

def verify_reports(summary,review,plan,durable,previous):
    need(review['status']==summary['worker_status']=='TARGET_STARTUP_COMPARISON_MEASURED_REVIEW_REQUIRED','Previous target status differs')
    need(review['owners_exited'] is True and review['owner_exit_codes']=={'1':0} and set(review['owner_reports'])=={'1'},'Previous target owner/exit differs')
    need(review['joint_publications']==[] and review['full_matrix_rerun'] is False,'Previous scoped result differs')
    need(summary['counts']==review['counts']==durable,'Previous durable counters differ')
    for n,v in [('integrations',832),('GPU_forward',3342),('GPU_solver',3342),('sense',18),('CPU_forward',2),('Startup_BAM_parameter_write',2)]:
        need(durable[n+'_attempted']==durable[n+'_completed']==v,'Previous call total differs: '+n)
    need(plan['prior_G09_call_ledger']==previous['G09_calls_cumulative'] and summary['prior_G09_call_ledger']==plan['prior_G09_call_ledger'],'Earlier ledger lost')
    need(summary['G09_calls_cumulative']==merged(plan['prior_G09_call_ledger'],durable),'Cumulative call ledger differs')
    need(summary['canonical_completed']==plan['canonical_completed_before']+832==661423 and summary['canonical_attempted']==plan['canonical_attempted_before']+832==661424,'Canonical history differs')
    for suffix in ('attempted','completed'):
        need(summary['G09_GPU_forward_cumulative_'+suffix]==plan['prior_GPU_forward_'+suffix]+3342==55776,'Forward history differs')
    graph={kind+'_'+s:plan['prior_contact_graph_series'][kind+'_'+s]+durable['ContactGraph_'+native+'_'+s] for kind,native in [('capture','capture_begin'),('launch','launch')] for s in ('attempted','completed')}
    need(summary['contact_graph_series_cumulative']==graph,'Graph history differs')
    owner=review['owner_reports']['1']
    need(all(owner[k] is True for k in ('all_cases_completed','strict_comparisons_passed','target_candidates_baseline_restored','original_material_Data_Model_owners_RNG_restored')),'Target completion/restoration missing')
    need(owner['contact_ordering']['graph_closed'] is True and owner['schedule_check']['status']=='COMPLETE_SINGLETON_NATIVE_ACCOUNTING_PASS','Target graph/accounting incomplete')
    need(owner['counts']==durable,'Target owner counters differ')
    rows=owner['continuation_cases'];need([(r['case'],r['mode']) for r in rows]==[('reset-1','adaptive'),('reset-1','reference')],'Target cases differ')
    checks=[v['passed'] for r in rows for k,v in r.items() if k.startswith('check_')]
    need(len(checks)==10 and all(v is True for v in checks),'Previously verified comparison changed')
    for row in rows:
        proof=row['target_input_preflight']
        need(row['checks_passed'] is True and row['joint_publication'] is False and 'aborted_restore' in row,'Target candidate not restored')
        need(proof['passed'] is True and len(proof['checks'])==10 and all(v['passed'] is True for v in proof['checks'].values()),'Previously verified input differs')
    need(owner['conditioning_result']['mandatory_exact_replay'] is True and rows[0]['physical']['mandatory_exact_replay'] is True,'Required target replay missing')
    need(all(summary.get(k) is True for k in ('git_state_unchanged','selected_sources_unchanged','old_evidence_rechecked_unchanged','counters_consistent')),'Previous immutable evidence check failed')
    need(summary['G09']=='OPEN' and summary['stage11_complete'] is False and durable.get('PPO',0)==durable.get('cloud',0)==0,'Invalid prior acceptance/training claim')
    return summary

def verify_startup_run(root,current_git,expected_program,previous):
    root=Path(root);base=root/'post-reset-logical-startup-resume';run=base/RUN_ID
    marker_path=base/'started-v1.json';marker=json.loads(marker_path.read_bytes())
    need(marker['run']==str(run) and marker['bundle_sha256']==BUNDLE_SHA,'Startup started marker differs; preserve it')
    archive=run/'post-reset-logical-startup-resume-return.zip'
    need(archive.is_file() and not archive.is_symlink() and archive.stat().st_size==56806947 and sha(archive.read_bytes())==ARCHIVE_SHA,'Reviewed startup archive identity differs')
    with zipfile.ZipFile(archive) as z:
        names=z.namelist();inv=json.loads(z.read('evidence-inventory.json'))
        need(len(names)==len(set(names))==9471 and set(names)==set(inv)|{'evidence-inventory.json'},'Reviewed archive roster differs')
        for name,row in inv.items():
            path=run/name;posix=PurePosixPath(name)
            need(not posix.is_absolute() and '..' not in posix.parts and path.resolve().is_relative_to(run.resolve()) and not path.is_symlink(),'Unsafe previous evidence path')
            raw=z.read(name);need(row==dict(bytes=len(raw),sha256=sha(raw)) and path.read_bytes()==raw,'Archived/local startup evidence changed: '+name)
        need((run/'evidence-inventory.json').read_bytes()==z.read('evidence-inventory.json'),'Previous inventory changed')
        need(sha(z.read('launcher.py'))==LAUNCHER_SHA,'Previous launcher differs')
        program={n.removeprefix('program/'):dict(bytes=len(z.read(n)),sha256=sha(z.read(n))) for n in names if n.startswith('program/')}
        need(program==expected_program,'Previous executable bundle differs')
        load=lambda name:json.loads(z.read(name))
        need(load('execution-finished.json')==dict(process_has_exited=True,exit_code=0,timed_out=False,interrupted=False,launch_error=None),'Previous execution was interrupted or timed out')
        need(load('git-before.json')==load('git-after.json')==current_git,'Project changed since reviewed target')
        durable={k:v for k,v in load('probe/owner-1/progress.json').items() if type(v) is int}
        review=load('probe/run-review.json')
        summary=verify_reports(load('run-summary.json'),review,load('runtime-plan.json'),durable,previous)
        phase=load('probe/phase-progress.json');need(phase['completed_units']==phase['total_units']==6,'Previous target incomplete')
        need(not any(n.startswith(('probe/owner-0/','probe/publications/')) for n in names),'Previous execution scope differs')
        need(load('probe/owner-1/run-review.json')==review['owner_reports']['1'],'Previous owner report differs')
        for mode in ('adaptive','reference'):
            result=load('probe/target-'+mode+'.json')
            need(result['comparisons_passed'] is True and result['restored']['full_baseline_restored'] is True,'Previous target transaction incomplete')
        owner=review['owner_reports']['1'];proc=Path('/proc')/str(owner['pid'])/'cmdline'
        if proc.is_file():need(str(run).encode() not in proc.read_bytes(),'Previous owner remains active')
    return dict(status='INDEPENDENTLY_VERIFIED_TARGET_BYTES_PRESERVED',run=str(run),archive_sha256=ARCHIVE_SHA,
        commit='33780ae1e471e9f63d142b66bbb37b3ee3e24235',marker_sha256=sha(marker_path.read_bytes()),summary=summary,
        timeout_caused_failure=False,physical_acceptance=False,scoped_target_physical_acceptance=True,full_corrected_matrix_acceptance=False,
        native_calls_in_reviewer=0,phase_percent=100.0,target_check_percent=100.0,target_checks='10/10',old_started_preserved=True)
