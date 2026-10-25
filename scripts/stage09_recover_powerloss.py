#!/usr/bin/env python3
"""Resume the specifically approved F2 outage, on the already-running cloud GPU."""
import fcntl,hashlib,json,os,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path

ROOT=Path('/root/autodl-tmp/microduck-double-balance')
REPO=ROOT/'stage09-workspace'
ART=ROOT/'artifacts/double-balance-stage09'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    os.chdir(REPO)
    os.environ.setdefault('MUJOCO_GL','egl')
    os.environ.setdefault('UV_CACHE_DIR',str(ROOT/'cache/uv'))
    os.environ.setdefault('UV_PYTHON_INSTALL_DIR',str(ROOT/'cache/python'))
    os.environ.setdefault('XDG_CACHE_HOME',str(ROOT/'cache/xdg'))
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'GPU is already in use; no process stopped'
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:args=(p/'cmdline').read_bytes().split(b'\0')
        except OSError:continue
        assert b'mjlab_microduck.double_balance_stage09' not in args,'An existing Stage 09 process must be inspected'
    from mjlab_microduck.double_balance_stage09_state import preflight,load_checked
    from mjlab_microduck.double_balance_stage09_budget import (
        RECOVERY_ID,RECOVERY_COUNTS,RECOVERY_CHECKPOINT,ORIGINAL_RUNTIME_SHA,
        activate_powerloss_recovery,atomic_json,validate_local_runtime,
    )
    head=preflight(cloud=True)
    recovery_lock=(ART/'recovery-launch.lock').open('a')
    fcntl.flock(recovery_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    job_lock=(ART/'ledger.job.lock').open('a')
    fcntl.flock(job_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    ledger_path=ART/'ledger.json';campaign_path=ART/'pilot/campaign.json'
    ledger=json.loads(ledger_path.read_text());campaign=json.loads(campaign_path.read_text())
    assert ledger['campaign_id']==campaign['campaign_id'],'Campaign identity differs'
    assert ledger['charged_updates']==RECOVERY_COUNTS,'Counts advanced or another interruption occurred; this allowance cannot be enlarged'
    assert ledger['maximum_cloud_hours']==4 and ledger['finalization_reserve_hours']==.75
    original_start=(ART/'setup-start.txt').read_text().strip()
    assert abs(datetime.fromisoformat(original_start).timestamp()-ledger['started_epoch'])<1e-4
    now=time.time();assert now+1>=ledger['last_observed_epoch'],'Clock moved backwards'
    elapsed=max(ledger['elapsed_seconds'],now-ledger['started_epoch'])
    remaining=3.25*3600-elapsed
    print('Stage09BudgetStart='+original_start,flush=True)
    print('Stage09TrainingMinutesRemaining='+str(round(max(0,remaining)/60,1)),flush=True)
    assert remaining>850*3.1+600,'Insufficient original training window for 850 updates plus 10 minutes overhead; no budget extension performed'
    local_path=ART/'inputs/local-preflight.json'
    assert sha(local_path)=='ff513074b3797968a2bdcbeb1637a5ca53da9206bb04f2ba4d06ca0d2efb342f'
    local=json.loads(local_path.read_text());validate_local_runtime(REPO,local['runtime_manifest_sha256'])
    assert campaign['runtime_manifest_sha256']==ORIGINAL_RUNTIME_SHA
    _,_,source_sha=load_checked(ART/'inputs/primary.pt')
    assert source_sha==campaign['source_sha256']
    for name in ('F0','F1'):
        entry=campaign['profiles'][name]
        assert entry['completed']==500,name+' is not complete'
        _,state,_=load_checked(entry['checkpoint'],profile=name,campaign_id=ledger['campaign_id'])
        assert state['experiment_completed_updates']==500
        for step in (250,500):
            for protocol in ('nominal','dev'):
                item=campaign['evaluations'][f'{name}-u{step:06d}/{protocol}']
                assert sha(item['path'])==item['sha256'],'Completed evaluation changed'
                assert json.loads(Path(item['path']).read_text())['status']=='PASS'
    for protocol in ('nominal','dev'):
        item=campaign['evaluations']['baseline/'+protocol]
        assert sha(item['path'])==item['sha256'],'Baseline evaluation changed'
    checkpoint=ART/RECOVERY_CHECKPOINT
    segment=checkpoint.parent.parent
    assert str(segment) in campaign['profiles']['F2']['segments']
    report=json.loads((segment/'training.json').read_text())
    assert report['experiment_completed_updates']==175 and Path(report['latest_checkpoint'])==checkpoint
    assert not campaign.get('profiles',{}).get('F3',{}).get('segments'),'F3 already started'
    for item in segment.joinpath('checkpoints').glob('update_*.json'):
        assert json.loads(item.read_text())['experiment_completed_updates']<=150,'A later checkpoint exists; inspect it first'
    _,state,checkpoint_sha=load_checked(checkpoint,profile='F2',campaign_id=ledger['campaign_id'])
    assert state['experiment_completed_updates']==150 and state['expected_adam_steps']==103000
    print('Stage09RecoveryCheckpoint='+str(checkpoint),flush=True)
    print('Stage09RecoveryLearningRate='+str(state['learning_rate']),flush=True)
    print('Stage09RecoveryAdamSteps='+str(state['expected_adam_steps']),flush=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    audit=ART/'recovery'/RECOVERY_ID/stamp;audit.mkdir(parents=True,exist_ok=False)
    (audit/'ledger-before.json').write_bytes(ledger_path.read_bytes())
    (audit/'campaign-before.json').write_bytes(campaign_path.read_bytes())
    (audit/'interrupted-training.json').write_bytes((segment/'training.json').read_bytes())
    activated=activate_powerloss_recovery(ledger_path,checkpoint_sha)
    after=json.loads(ledger_path.read_text())
    for key in ('campaign_id','charged_updates','started_epoch','started_at','last_observed_epoch','elapsed_seconds'):
        assert after[key]==ledger[key],'Recovery changed historical accounting: '+key
    atomic_json(audit/'recovery.json',dict(id=RECOVERY_ID,activated=activated,git_head=head,
        old_runtime_manifest_sha256=ORIGINAL_RUNTIME_SHA,
        new_runtime_manifest_sha256=sha(REPO/'docs/audits/stage-09-runtime-hashes.json'),
        checkpoint_sha256=checkpoint_sha,checkpoint=str(checkpoint),restored_updates=150,
        observed_attempts=175,lost_updates=25,restored_learning_rate=state['learning_rate'],
        restored_adam_steps=103000,episode_rng_rnn_delays='RESET',assistance=0,
        exact_trajectory_resume=False,counts_reset=False,budget_reset=False,power_operations=False))
    print('Stage09RecoveryAudit='+str(audit/'recovery.json'),flush=True)
    print('Stage09AttemptCaps=F0:500,F1:500,F2:525,F3:500; Total:2025',flush=True)
    fcntl.flock(job_lock,fcntl.LOCK_UN);job_lock.close()
    command=[sys.executable,'-u','-m','mjlab_microduck.double_balance_stage09','campaign',
        '--checkpoint',str(ART/'inputs/primary.pt'),'--datasets',str(ART/'initial-states'),
        '--output',str(ART/'pilot'),'--ledger',str(ledger_path),'--local-report',str(local_path)]
    log_path=ART/('campaign-recovery-'+stamp+'.log')
    with log_path.open('x') as log:
        process=subprocess.Popen(['nohup',*command],cwd=REPO,stdin=subprocess.DEVNULL,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('Stage09RecoverySupervisorPID='+str(process.pid),flush=True)
    print('Stage09RecoveryLog='+str(log_path),flush=True)
    print('Stage09Recovery=LAUNCHED; verify actual updates in monitor',flush=True)
    print('Stage09InstancePowerOperation=NONE',flush=True)

if __name__=='__main__':main()
