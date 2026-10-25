"""Exercise evidence chaining and the one-command boundary with isolated fixtures."""
import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from types import SimpleNamespace
from unittest.mock import patch

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;passed=[]
def check(name,ok):
    assert ok,name
    passed.append(name)
def rejected(operation):
    try:operation();return False
    except (RuntimeError,KeyError):return True
digest=lambda data:hashlib.sha256(data).hexdigest()
outer={'__name__':'integration_fixture'}
exec((ROOT/'microduck-stage12-selected-speed500-v1-20261020.py').read_text(),outer)
fetch=outer['fetch'];replay=outer['replay'];audit={'__name__':'audit_fixture'};exec(outer['SOURCES']['audit'],audit)
actual=json.loads((HERE/'selection-data.json').read_text())
meta=dict(package_identity=replay['TRAIN_ID'],origin_package_identity=replay['ORIGIN_ID'],origin_checkpoint_sha256=replay['ORIGIN_MODEL_SHA'],
          policy_updates=5000,refine_updates=500,campaign_attempted_updates=500,num_envs=4096,lower_assistance=0.,reward_spec=replay['REWARD_SPEC'])
saved=dict(infos=dict(stage12_native=deepcopy(meta),stage12_refine=deepcopy(meta)))
check('new_checkpoint_lineage_is_accepted',replay['verify_checkpoint_metadata'](saved)['policy_updates']==5000)
for field,value in [('package_identity',replay['ORIGIN_ID']),('refine_updates',1000),('origin_checkpoint_sha256','0'*64)]:
    bad=deepcopy(saved);bad['infos']['stage12_native'][field]=value
    check('checkpoint_rejects_wrong_'+field,rejected(lambda:replay['verify_checkpoint_metadata'](bad)))

with tempfile.TemporaryDirectory(prefix='md12-selected-fixture-') as temporary:
    root=Path(temporary);cloud=root/'cloud';cloud.mkdir();selected=root/'selected';selected.mkdir()
    outer['SELECTED']=selected;outer['REPLAY']=root/'replay'
    replay['SLOT']=outer['REPLAY']
    groups=[];bodies={};rows=[]
    for seed in (20261021,20261022):
        episodes=[]
        for i in range(128):
            longest=200 if seed==20261022 and i<3 else 474
            episodes.append(dict(seed=seed,env_id=i,steps=500,stable_steps=480,longest_stable_steps=longest,
                ever_5s=longest>=250,terminated=False,success=longest>=250,terminations=['time_out'],
                gate_failure_counts=[0]*9,run_break_counts=[0]*9,value_sums=[0.]*8))
        rows+=episodes
        group=dict(status='EVALUATED',seed=seed,model='5000',checkpoint_sha256=digest(b'c'*10036828),
                   policy_and_normalizer_unchanged=True,episodes=episodes,summary=audit['summarize'](episodes))
        remote='native-speed-refine-v1/evaluations/segment-02/group-%d.json'%seed
        bodies[remote]=json.dumps(group).encode()
        groups.append(dict(seed=seed,path=fetch['REMOTE_STAGE']+remote,sha256=digest(bodies[remote]),summary=group['summary']))
    model_path=fetch['FIXED_FILES']['model']['remote']
    model=b'c'*10036828;bodies[model_path]=model
    summary=audit['summarize'](rows)
    manifest=dict(status='EVALUATION_COMPLETE',segment=2,model='5000',seeds=[20261021,20261022],num_envs_per_group=128,
        summary=summary,thresholds=audit['THRESHOLDS'],group_evidence=groups,paired_comparison_allowed=False,
        selection_set_not_final_heldout_test=True,task_acceptance_claimed=False,
        checkpoint=dict(path=fetch['REMOTE_STAGE']+model_path,sha256=digest(model),bytes=len(model),completed_updates=500,attempted_updates=500,policy_updates=5000))
    manifest_path=fetch['FIXED_FILES']['evaluation']['remote'];bodies[manifest_path]=json.dumps(manifest).encode()
    for name in ('native_task.py','source-pins.json'):bodies[fetch['FIXED_FILES'][name]['remote']]=('fixture '+name).encode()
    files=deepcopy(fetch['FIXED_FILES'])
    for item in files.values():
        body=bodies[item['remote']];item.update(bytes=len(body),max_bytes=len(body),sha256=digest(body))
    for rel,body in bodies.items():
        path=cloud/rel;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(body)
    fetch['FIXED_FILES']=deepcopy(files);fetch['FILES']=deepcopy(files);fetch['EXPECTED_SUMMARY']=summary
    expanded=fetch['expanded_files'](manifest)
    check('group_pins_derived_from_selected_manifest',len(expanded)==6 and expanded['group_20261021']['sha256']==groups[0]['sha256'])
    bad=deepcopy(manifest);bad['group_evidence'][0]['path']='/etc/passwd'
    check('manifest_cannot_escape_selected_group_directory',rejected(lambda:fetch['expanded_files'](bad)))
    bad=deepcopy(manifest);bad['group_evidence'][1]['seed']=20261021
    check('duplicate_group_seed_rejected',rejected(lambda:fetch['expanded_files'](bad)))
    bad=deepcopy(manifest);bad['thresholds']['consecutive_steps']=249
    check('success_threshold_relaxation_rejected',rejected(lambda:fetch['expanded_files'](bad)))
    remote_source=fetch['decode'](fetch['REMOTE_B85'],fetch['REMOTE_SHA']);requests=[]
    real_run=subprocess.run
    def simulated_transport(argv,**kwargs):
        if argv[:3]==[sys.executable,'-B','-c'] and argv[3]==outer['SOURCES']['audit']:
            return real_run(argv,**kwargs)
        assert argv[:3]==['python3','-B','-c'] and argv[3].encode()==remote_source
        args=argv[4:];requests.append(args)
        # Each RPC has a fresh interpreter, as the real SSH -c invocation does.
        remote={'__name__':'remote_fixture'};exec(remote_source,remote)
        remote.update(ROOT=cloud,FILES=deepcopy(files),FIXED_FILES=deepcopy(files),EXPECTED_SUMMARY=summary)
        buffer=io.BytesIO();stream=io.TextIOWrapper(buffer,encoding='utf-8')
        try:
            with contextlib.redirect_stdout(stream):remote['main'](args)
            stream.flush();body=buffer.getvalue()
        finally:stream.detach()
        return SimpleNamespace(returncode=0,stdout=body,stderr=b'')
    original_decode=fetch['decode']
    # The real SSH authentication helper is retained in the artifact; this fixture never connects.
    fetch['decode']=lambda value,sha:b'import shlex\ndef argv(command):return shlex.split(command)\n' if sha==fetch['SSH_SHA'] else original_decode(value,sha)
    fetch['local_replay_inventory']=lambda folder:dict(existing_python=sys.executable,matching_files=365,missing_count=0,different_count=0,frozen_source_count=365)
    before={rel:digest(body) for rel,body in bodies.items()}
    with patch.object(subprocess,'run',side_effect=simulated_transport),contextlib.redirect_stdout(io.StringIO()):
        receipt=outer['download_and_verify']()
    check('two_phase_download_fetches_all_six_hash_verified_files',len(receipt['files'])==6 and all(fetch['sha'](selected/item['local'])==item['sha256'] for item in expanded.values()))
    check('dynamic_groups_work_in_fresh_remote_interpreters',any(a[0]=='inventory' and 'group_20261021' in a for a in requests))
    check('no_remote_fixture_artifact_changes',all(fetch['sha'](cloud/rel)==h for rel,h in before.items()))
    check('retained_episode_audit_uses_existing_python_without_gpu',receipt['evidence_audit']['summary']['successes']==253 and receipt['evidence_audit']['added_simulation_steps']==0)
    requests.clear()
    with patch.object(subprocess,'run',side_effect=simulated_transport),contextlib.redirect_stdout(io.StringIO()):outer['download_and_verify']()
    check('complete_repeat_performs_no_ssh_reads',not requests)
    path=selected/'evaluation/group-20261022.json';original=path.read_bytes()
    changed=json.loads(original);changed['episodes'][0]['success']=True
    path.write_text(json.dumps(changed))
    check('episode_hash_corruption_detected',rejected(lambda:audit['audit_completed_evidence'](selected)))
    edited_manifest=deepcopy(manifest);edited_manifest['group_evidence'][1]['sha256']=fetch['sha'](path)
    local_manifest=selected/'evaluation/evaluation-result.json';original_manifest=local_manifest.read_bytes()
    local_manifest.write_text(json.dumps(edited_manifest))
    check('wrong_success_decision_rejected_even_with_matching_fixture_hash',rejected(lambda:audit['audit_completed_evidence'](selected)))
    path.write_bytes(original);local_manifest.write_bytes(original_manifest)
    tiny=deepcopy(summary);tiny['values_mean_over_first_episode_steps']['top_speed_m_s']+=1e-15
    check('summary_roundoff_is_recorded_without_changing_decisions',audit['summary_audit'](tiny,summary)['valid'] and not audit['summary_audit'](tiny,summary)['exact_equal'])
    bad=deepcopy(summary);bad['successes']-=1
    check('count_differences_never_use_float_tolerance',not audit['summary_audit'](bad,summary)['valid'])
    calls=[]
    def launch():
        calls.append('launch');replay['REPORT_CALLBACK'](dict(status='LOCAL_VIDEO_COMPLETE',result=dict(status='LOCAL_VIDEO_COMPLETE',clips=[]),controller_package_identity=replay['PACKAGE_ID'],local_return_utc='fixture',cloud_connections=0));return 0
    replay['launch']=launch
    with patch.object(subprocess,'run',side_effect=simulated_transport),contextlib.redirect_stdout(io.StringIO()):code=outer['main'](['--run'])
    report=json.loads(sorted((selected/'reports').glob('*.json'))[-1].read_bytes())
    check('one_command_launches_replay_only_after_fetch_and_evidence_audit',code==0 and calls==['launch'] and report['fetch']['evidence_audit']['summary']['successes']==253 and report['local_replay']['controller_package_identity']==replay['PACKAGE_ID'])
    calls.clear()
    with patch.object(subprocess,'run',side_effect=simulated_transport),contextlib.redirect_stdout(io.StringIO()):code=outer['main'](['--download'])
    check('download_only_does_not_launch_replay',code==0 and calls==[])
    (selected/'checkpoints/refine_000500.pt').write_bytes(b'corrupted; preserve')
    with patch.object(subprocess,'run',side_effect=AssertionError('must not contact remote')),contextlib.redirect_stdout(io.StringIO()):code=outer['main'](['--run'])
    check('failed_local_integrity_blocks_replay_and_preserves_file',code==2 and calls==[] and (selected/'checkpoints/refine_000500.pt').read_bytes()==b'corrupted; preserve')
    outer['REPLAY'].mkdir();(outer['REPLAY']/'started.json').write_text('{}')
    with patch.object(subprocess,'run',side_effect=AssertionError('must not fetch again')),contextlib.redirect_stdout(io.StringIO()):code=outer['main'](['--run'])
    check('existing_replay_dispatches_to_duplicate_guard_without_fetch',code==0 and calls==['launch'])
    replay['watch']=lambda:0
    with patch.object(subprocess,'run',side_effect=AssertionError('watch contacted cloud')),contextlib.redirect_stdout(io.StringIO()):code=outer['main'](['--watch'])
    check('watch_uses_local_state_only',code==0)

out=dict(status='PASS',count=len(passed),checks=passed,real_network_requests=0,real_model_loaded=False,real_simulation_steps=0,PPO=0,
         scope='Isolated transfer, first-episode JSON, lineage and orchestration fixtures; actual GPU replay pending user run')
(HERE/'integration-offline-review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(out,ensure_ascii=False))
