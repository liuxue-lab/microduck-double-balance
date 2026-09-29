#!/usr/bin/env python3
"""Run screening/replication/extension sequentially under one existing budget."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mjlab_microduck.double_balance_stage08_budget import BudgetLedger,atomic_json
from mjlab_microduck.double_balance_stage08_plan import SOURCE_SHA256,TRAIN_SEEDS
from mjlab_microduck.double_balance_stage08_review import checksum,validate_decision


def jobs_for(phase,profiles):
    seeds=TRAIN_SEEDS[:1] if phase=='screening' else TRAIN_SEEDS
    target={'screening':500,'replication':1500,'extension':4000}[phase]
    return [{'key':f'{profile}-{seed}','profile':profile,'seed':seed,'target_updates':target,'segments':[]}
            for profile in profiles for seed in seeds]


def prior_endpoint(previous,key,target):
    jobs=[j for j in previous['jobs'] if j['key']==key]
    if len(jobs)!=1 or not jobs[0]['segments']:
        raise ValueError(f'Missing previous lineage: {key}')
    report=json.loads((Path(jobs[0]['segments'][-1]['directory'])/'training.json').read_text())
    if report['status']!='TRAINING_COMPLETE' or report['experiment_completed_updates']!=target:
        raise ValueError(f'Previous {key} has not completed {target} updates')
    return report['latest_checkpoint']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('screening','replication','extension'))
    p.add_argument('--gpu',choices=('A800','5090'),required=True)
    p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--capacity-summary',type=Path,required=True)
    p.add_argument('--datasets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--decision',type=Path)
    p.add_argument('--previous-batch',type=Path)
    p.add_argument('--resume-batch',action='store_true')
    args=p.parse_args()
    for key in ('ledger','source','capacity_summary','datasets','output','decision','previous_batch'):
        if getattr(args,key) is not None:
            setattr(args,key,getattr(args,key).resolve())
    if checksum(args.source)!=SOURCE_SHA256:
        raise ValueError('Source must be the archived update 1000 checkpoint')
    ledger=BudgetLedger(args.ledger)
    if ledger.snapshot()['gpu']!=args.gpu:
        raise ValueError('GPU budget differs')
    previous=None
    if args.phase=='screening':
        profiles=list('ABCDE')
    else:
        if args.decision is None or args.previous_batch is None:
            raise ValueError('Replication/extension require decision and previous batch files')
        decision=json.loads(args.decision.read_text())
        profiles=decision['selected_profiles']
        previous=json.loads(args.previous_batch.read_text())
        expected='screening' if args.phase=='replication' else 'replication'
        if previous['contract']['campaign_id']!=ledger.identity or previous['contract']['phase']!=expected:
            raise ValueError('Previous phase or budget differs')
        for profile in profiles:
            validate_decision(args.decision,profile=profile,target=1500 if args.phase=='replication' else 4000,
                              campaign_id=ledger.identity)
    if args.phase=='extension' and args.gpu!='5090':
        raise ValueError('The A800 plan has no conditional extension')
    contract={'schema_version':1,'phase':args.phase,'gpu':args.gpu,'campaign_id':ledger.identity,
              'source_sha256':SOURCE_SHA256,'capacity_sha256':checksum(args.capacity_summary),
              'decision_sha256':checksum(args.decision) if args.decision else None,
              'previous_batch_sha256':checksum(args.previous_batch) if args.previous_batch else None,
              'datasets':str(args.datasets)}
    manifest_path=args.output/'batch.json'
    if args.resume_batch:
        manifest=json.loads(manifest_path.read_text())
        if manifest['contract']!=contract:
            raise ValueError('Batch contract changed; do not reuse its output directory')
    else:
        args.output.mkdir(parents=True,exist_ok=False)
        manifest={'contract':contract,'status':'RUNNING','stage08_complete':False,'jobs':jobs_for(args.phase,profiles)}
    atomic_json(manifest_path,manifest)
    for job in manifest['jobs']:
        checkpoint=str(args.source); resume=False
        if job['segments']:
            report_path=Path(job['segments'][-1]['directory'])/'training.json'
            report=json.loads(report_path.read_text()) if report_path.exists() else {}
            if report.get('status')=='TRAINING_COMPLETE' and report['experiment_completed_updates']==job['target_updates']:
                continue
            if report.get('latest_checkpoint'):
                checkpoint=report['latest_checkpoint']; resume=True
        elif previous is not None and (args.phase=='extension' or job['seed']==TRAIN_SEEDS[0]):
            checkpoint=prior_endpoint(previous,job['key'],1500 if args.phase=='extension' else 500)
            resume=True
        if ledger.remaining(training=True)<180:
            manifest['status']='PAUSED_BUDGET'; break
        directory=args.output/job['key']/f'segment-{len(job["segments"])+1:03d}'
        job['segments'].append({'directory':str(directory),'input_checkpoint':checkpoint,'resume':resume,'status':'RUNNING'})
        atomic_json(manifest_path,manifest)
        argv=[sys.executable,'-u','-m','mjlab_microduck.double_balance_stage08','train',
              '--gpu',args.gpu,'--ledger',str(args.ledger),'--checkpoint',checkpoint,
              '--output',str(directory),'--datasets',str(args.datasets),'--profile',job['profile'],
              '--seed',str(job['seed']),'--target-updates',str(job['target_updates']),
              '--capacity-summary',str(args.capacity_summary)]
        if resume: argv.append('--resume')
        if args.decision: argv.extend(('--decision',str(args.decision)))
        directory.parent.mkdir(parents=True,exist_ok=True)
        with directory.with_suffix('.log').open('x') as log:
            result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT)
        report_path=directory/'training.json'
        report=json.loads(report_path.read_text()) if report_path.exists() else {'status':'FAIL'}
        job['segments'][-1].update(exit_code=result.returncode,status=report['status'])
        print('Stage08BatchJob='+json.dumps({'key':job['key'],'status':report['status'],'log':str(directory.with_suffix('.log'))}),flush=True)
        atomic_json(manifest_path,manifest)
        if result.returncode or report['status']!='TRAINING_COMPLETE':
            manifest['status']='PAUSED' if report['status']=='PAUSED' else 'NEEDS_REVIEW'
            break
    else:
        manifest['status']='BATCH_COMPLETE'
    atomic_json(manifest_path,manifest)
    print('Stage08Batch='+manifest['status'])
    if manifest['status']=='NEEDS_REVIEW':
        raise SystemExit(1)


if __name__=='__main__':
    main()
