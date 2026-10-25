#!/usr/bin/env python3
"""Launch on a user-provided running cloud RTX 5090. Never powers an instance."""
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('/root/autodl-tmp/microduck-double-balance')
ART=ROOT/'artifacts/double-balance-stage09'
REPO=ROOT/'stage09-workspace'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name','--format=csv,noheader'],text=True).strip()
    if gpu!='NVIDIA GeForce RTX 5090':raise ValueError('Launch requires exactly one cloud RTX 5090')
    package=Path(__file__).resolve().parent
    data=json.loads((package/'inputs.json').read_text())
    for relative,digest in data['files'].items():
        if sha(package/relative)!=digest:raise ValueError('Cloud input checksum differs: '+relative)
    ART.mkdir(parents=True,exist_ok=True)
    start=ART/'setup-start.txt'
    if not start.exists():
        with start.open('x') as stream:stream.write(datetime.now(timezone.utc).isoformat()+'\n')
    if not REPO.exists():
        subprocess.run(['git','clone','-b','double-balance',str(package/'source.bundle'),str(REPO)],check=True)
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    if head!=data['git_head']:raise ValueError('Existing Stage 09 checkout has another commit; do not overwrite it')
    subprocess.run(['uv','sync','--locked'],cwd=REPO,check=True)
    python=REPO/'.venv/bin/python'
    if sha(REPO/'docs/audits/stage-09-runtime-hashes.json')!=data['runtime_manifest_sha256']:
        raise ValueError('Code manifest differs from local runtime check')
    inputs=ART/'inputs';inputs.mkdir(exist_ok=True)
    for name in ('primary.pt','local-preflight.json'):
        target=inputs/name
        if target.exists() and sha(target)!=data['files'][name]:raise ValueError('Existing cloud input differs')
        if not target.exists():shutil.copy2(package/name,target)
    datasets=ART/'initial-states';datasets.mkdir(exist_ok=True)
    for source in (package/'initial-states').iterdir():
        target=datasets/source.name
        if target.exists() and sha(target)!=sha(source):raise ValueError('Existing initial states differ')
        if not target.exists():shutil.copy2(source,target)
    ledger=ART/'ledger.json'
    if not ledger.exists():
        subprocess.run([str(python),'-m','mjlab_microduck.double_balance_stage09','init-budget',
                        '--ledger',str(ledger),'--started-at',start.read_text().strip()],cwd=REPO,check=True)
    # A single ledger and job lock are retained across launches/retries.
    timestamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    log_path=ART/('campaign-'+timestamp+'.log')
    command=[str(python),'-u','-m','mjlab_microduck.double_balance_stage09','campaign',
             '--checkpoint',str(inputs/'primary.pt'),'--datasets',str(datasets),'--output',str(ART/'pilot'),
             '--ledger',str(ledger),'--local-report',str(inputs/'local-preflight.json')]
    with log_path.open('x') as log:
        process=subprocess.Popen(['nohup',*command],cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('Stage09CampaignSupervisorPID='+str(process.pid))
    print('Stage09CampaignLog='+str(log_path))
    print('Stage09CampaignState='+str(ART/'pilot/campaign.json'))
    print('Stage09InstancePowerOperation=NONE')


if __name__=='__main__':main()
