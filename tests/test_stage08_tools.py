"""CLI rejection before PPO, batch continuation and archive integrity tests."""
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

import pytest
from mjlab_microduck.double_balance_stage08_budget import create_ledger
from datetime import datetime,timezone

REPO=Path(__file__).resolve().parents[1]


def script(name):
    spec=importlib.util.spec_from_file_location(name,REPO/'scripts'/f'{name}.py')
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def test_cloud_cli_rejects_cpu_before_loading_checkpoint_or_ppo(tmp_path):
    ledger=tmp_path/'budget.json'; now=time.time()
    create_ledger(ledger,'5090',datetime.fromtimestamp(now-1,timezone.utc).isoformat(),now=now)
    output=tmp_path/'run'
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(REPO/'src'))
    result=subprocess.run([sys.executable,'-m','mjlab_microduck.double_balance_stage08','train',
        '--gpu','5090','--ledger',str(ledger),'--checkpoint',str(tmp_path/'unused.pt'),
        '--output',str(output),'--datasets',str(tmp_path/'states'),'--profile','E',
        '--seed','20260928','--target-updates','500'],capture_output=True,text=True,env=env,timeout=60)
    assert result.returncode!=0
    assert ('restricted to the cloud' in result.stderr or 'One CUDA GPU required' in result.stderr)
    assert 'Stage08Watchdog=' in result.stdout
    assert not output.exists()


def test_batch_sizes_and_endpoint_rules(tmp_path):
    batch=script('run_stage08_batch')
    screen=batch.jobs_for('screening',list('ABCDE'))
    replica=batch.jobs_for('replication',['D','E'])
    extension=batch.jobs_for('extension',['E'])
    assert len(screen)==5 and len(replica)==6 and len(extension)==3
    core=sum(j['target_updates'] for j in screen)+sum(j['target_updates'] for j in replica)-2*500
    assert core==10500
    assert core+sum(j['target_updates']-1500 for j in extension)==18000
    segment=tmp_path/'segment'; segment.mkdir()
    report={'status':'PAUSED','experiment_completed_updates':400,'latest_checkpoint':'endpoint.pt'}
    (segment/'training.json').write_text(json.dumps(report))
    previous={'jobs':[{'key':'D-20260928','segments':[{'directory':str(segment)}]}]}
    with pytest.raises(ValueError,match='not completed'):
        batch.prior_endpoint(previous,'D-20260928',500)
    report.update(status='TRAINING_COMPLETE',experiment_completed_updates=500)
    (segment/'training.json').write_text(json.dumps(report))
    assert batch.prior_endpoint(previous,'D-20260928',500)=='endpoint.pt'


def test_archive_verifies_every_member_and_rejects_unsafe_paths(tmp_path):
    module=script('archive_stage08_results')
    data=tmp_path/'data.json'; data.write_text('{"status":"PASS"}')
    manifest={'files':[{'path':'reports/data.json','size_bytes':data.stat().st_size,'sha256':module.sha(data)}]}
    archive=tmp_path/'result.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        tar.add(data,arcname='reports/data.json')
        content=json.dumps(manifest).encode(); info=tarfile.TarInfo('MANIFEST.json'); info.size=len(content)
        tar.addfile(info,io.BytesIO(content))
    module.verify_archive(archive,module.sha(archive),tmp_path/'extracted')
    assert (tmp_path/'extracted/reports/data.json').read_bytes()==data.read_bytes()
    with pytest.raises(ValueError,match='SHA-256'):
        module.verify_archive(archive,'wrong',tmp_path/'wrong')
    malicious=tmp_path/'unsafe.tar.gz'
    with tarfile.open(malicious,'w:gz') as tar:
        info=tarfile.TarInfo('../escape'); info.size=1; tar.addfile(info,io.BytesIO(b'x'))
    with pytest.raises(ValueError,match='unsafe'):
        module.verify_archive(malicious,module.sha(malicious),tmp_path/'unsafe')
    assert not (tmp_path/'escape').exists()


def test_paired_evaluation_uses_identical_initial_states():
    module=script('evaluate_stage08_batch')
    first={'outcomes_by_initial_state':{'a':True,'b':False}}
    source={'outcomes_by_initial_state':{'a':False,'b':False}}
    result=module.paired(first,source)
    assert result['candidate_only_success']==1 and result['paired_success_fraction_difference']==.5
    with pytest.raises(ValueError,match='identical'):
        module.paired(first,{'outcomes_by_initial_state':{'c':False}})
