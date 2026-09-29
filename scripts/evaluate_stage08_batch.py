#!/usr/bin/env python3
"""Evaluate the frozen model list sequentially; never select using test data."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mjlab_microduck.double_balance_stage08_budget import BudgetLedger,atomic_json
from mjlab_microduck.double_balance_stage08_evaluation import PROTOCOLS,validate_selection,wilson_interval
from mjlab_microduck.double_balance_stage08_review import checksum


def aggregate(reports):
    episodes=[episode for report in reports for episode in report['episodes']]
    ids=[e['initial_state_id'] for e in episodes]
    if len(ids)!=len(set(ids)):
        raise ValueError('Held-out initial-state sets overlap')
    n=len(episodes); successes=sum(e['success'] for e in episodes)
    return {'episodes':n,'successes':successes,'success_fraction':successes/n,
            'wilson95':wilson_interval(successes,n),'scope':'one checkpoint, 768 randomized initial states',
            'outcomes_by_initial_state':{e['initial_state_id']:bool(e['success']) for e in episodes}}


def paired(candidate,reference):
    a,b=candidate['outcomes_by_initial_state'],reference['outcomes_by_initial_state']
    if set(a)!=set(b): raise ValueError('Paired comparison needs identical initial states')
    gain=sum(a[k] and not b[k] for k in a); loss=sum(b[k] and not a[k] for k in a)
    return {'paired_episodes':len(a),'candidate_only_success':gain,'reference_only_success':loss,
            'paired_success_fraction_difference':(gain-loss)/len(a)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gpu',choices=('A800','5090'),required=True)
    p.add_argument('--ledger',type=Path,required=True)
    p.add_argument('--selection',type=Path,required=True)
    p.add_argument('--datasets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--resume-batch',action='store_true')
    args=p.parse_args()
    args.selection=args.selection.resolve(); args.output=args.output.resolve()
    selection=json.loads(args.selection.read_text())
    ledger=BudgetLedger(args.ledger)
    if selection['campaign_id']!=ledger.identity or ledger.snapshot()['gpu']!=args.gpu:
        raise ValueError('Selection and budget do not match')
    if not args.resume_batch: args.output.mkdir(parents=True,exist_ok=False)
    if args.resume_batch:
        previous=json.loads((args.output/'batch-evaluation.json').read_text())
        if previous['selection_sha256']!=checksum(args.selection):
            raise ValueError('Cannot reuse held-out results with a changed selection')
    summary={'status':'RUNNING','selection_sha256':checksum(args.selection),'models':{},'stage08_complete':False}
    for model in selection['checkpoints']:
        sha=model['sha256']; validate_selection(args.selection,sha)
        if checksum(model['path'])!=sha: raise ValueError('Checkpoint changed after selection')
        reports={}
        for protocol in PROTOCOLS:
            directory=args.output/sha/protocol
            report_path=directory/'evaluation.json'
            if report_path.exists():
                report=json.loads(report_path.read_text())
                if (report.get('status')!='PASS' or report['checkpoint_sha256']!=sha or
                        report['protocol']!=protocol or
                        (protocol.startswith('test-') and report['selection_sha256']!=checksum(args.selection))):
                    raise ValueError('Existing evaluation failed; retain it and use a new output directory')
                reports[protocol]=report; continue
            if ledger.remaining(training=False)<120:
                summary['status']='PAUSED_BUDGET'; atomic_json(args.output/'batch-evaluation.json',summary); return
            argv=[sys.executable,'-u','-m','mjlab_microduck.double_balance_stage08','evaluate',
                  '--gpu',args.gpu,'--ledger',str(args.ledger.resolve()),'--checkpoint',model['path'],
                  '--output',str(directory),'--datasets',str(args.datasets.resolve()),
                  '--protocol',protocol,'--selection',str(args.selection)]
            directory.parent.mkdir(parents=True,exist_ok=True)
            with directory.with_suffix('.log').open('x') as stream:
                subprocess.run(argv,stdout=stream,stderr=subprocess.STDOUT,check=True)
            reports[protocol]=json.loads(report_path.read_text())
        summary['models'][sha]={'identity':model,'nominal':reports['nominal']['success_fraction'],
                               'random_test':aggregate([reports[p] for p in PROTOCOLS if p.startswith('test-')]),
                               'reports':{p:str(args.output/sha/p/'evaluation.json') for p in PROTOCOLS}}
        atomic_json(args.output/'batch-evaluation.json',summary)
    source=next(v for v in summary['models'].values() if v['identity']['role']=='source-1000')
    for result in summary['models'].values():
        result['paired_vs_source_1000']=paired(result['random_test'],source['random_test'])
    summary['status']='PASS'
    summary['note']='PASS means evaluator completed. Policy objective requires separate result and video review.'
    atomic_json(args.output/'batch-evaluation.json',summary)
    print('Stage08BatchEvaluation=PASS')


if __name__=='__main__': main()
