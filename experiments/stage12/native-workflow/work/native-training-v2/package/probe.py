"""Read-only campaign discovery; standard library, no simulator imports."""
import hashlib
from pathlib import Path
from common import SLOT, LEGACY_SLOT, read, alive, has_campaign, validation_remaining

def snapshot(slot):
    def get(name, default=None):
        p=slot/name
        return read(p) if p.exists() else default
    progress=get('run/progress.json', {'status':'PREPARED'})
    s=dict(slot=str(slot), status=progress['status'], progress=progress,
        supervisor_alive=alive(get('supervisor.json')), training_alive=alive(get('child.json')),
        new_training_budget=get('budget.json'), latest_checkpoint=get('run/latest.json'),
        initialization=get('initialization.json'), configuration=get('config.json'),
        legacy_attached=slot==LEGACY_SLOT, campaign_exists=has_campaign(slot),
        remaining_validation_runs=validation_remaining(slot) if slot==SLOT else 0,
        historical_PPO_reservations_preserved=4, automatic_power_actions=0, power_state='UNCONFIRMED')
    s['smoke']=get('run/smoke-result.json')
    if s['campaign_exists'] and not s['supervisor_alive'] and not s['training_alive'] and progress['status'] in ('PREPARED','INITIALIZING','TRAINING','SMOKE_PASSED'):
        s['status']='PROCESS_EXITED_PRESERVE_REVIEW_LOG'
    return s

def discover():
    slots=[s for s in (LEGACY_SLOT,SLOT) if has_campaign(s)]
    if len(slots)>1:
        return dict(status='MULTIPLE_CAMPAIGNS_PRESERVED', conflicts=[snapshot(s) for s in slots],
            supervisor_alive=False,training_alive=False,campaign_exists=True,remaining_validation_runs=0)
    return snapshot(slots[0]) if slots else dict(status='NO_EXISTING_CAMPAIGN',campaign_exists=False,
        supervisor_alive=False,training_alive=False,remaining_validation_runs=1)

def collect_slot(slot):
    x=snapshot(slot);files={}
    for name in ('budget.json','started.json','initialization.json','smoke-started.json','run/smoke-result.json',
                 'run/transfer.json','run/latest.json','run/best-training-metric.json','run/formal-load.json'):
        p=slot/name
        if p.exists() and p.stat().st_size<=64*1024:
            b=p.read_bytes();files[name]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),text=b.decode())
    x['small_receipts']=files
    logs=sorted((slot/'run').glob('train-*.log'),key=lambda p:p.stat().st_mtime)
    if logs:
        with logs[-1].open('rb') as f:
            f.seek(max(0,logs[-1].stat().st_size-16000));x['log_tail']=f.read().decode(errors='replace')
    supervisors=sorted(slot.glob('supervisor-*.log'),key=lambda p:p.stat().st_mtime)
    if supervisors:
        with supervisors[-1].open('rb') as f:
            f.seek(max(0,supervisors[-1].stat().st_size-8000));x['supervisor_tail']=f.read().decode(errors='replace')
    x['models_and_full_logs_retained_cloud_side']=True
    return x
