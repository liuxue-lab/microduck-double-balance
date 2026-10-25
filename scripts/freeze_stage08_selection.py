#!/usr/bin/env python3
"""Freeze per-seed best checkpoints using only nominal/development reports."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mjlab_microduck.double_balance_stage08_evaluation import PROTOCOL_VERSION,selection_rank
from mjlab_microduck.double_balance_stage08_plan import TRAIN_SEEDS,SOURCE_SHA256
from mjlab_microduck.double_balance_stage08_state import FINAL_STAGE07_SHA256
from mjlab_microduck.double_balance_stage08_review import checksum,reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--profile',choices=tuple('ABCDE'),required=True)
    p.add_argument('--runs',type=Path,nargs='+',required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--stage07-final',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--screening-only',action='store_true',help='Close an unsuccessful screening without claiming multi-seed replication')
    p.add_argument('--stop-reason')
    args=p.parse_args()
    if checksum(args.source)!=SOURCE_SHA256 or checksum(args.stage07_final)!=FINAL_STAGE07_SHA256:
        raise ValueError('Stage 07 reference checksums differ')
    best={}
    for root in args.runs:
        for dev_path in root.rglob('dev/evaluation.json'):
            nominal_path=dev_path.parent.parent/'nominal/evaluation.json'
            if not nominal_path.exists(): continue
            dev=json.loads(dev_path.read_text()); nominal=json.loads(nominal_path.read_text())
            if dev.get('status')!='PASS' or nominal.get('status')!='PASS' or dev.get('profile')!=args.profile: continue
            k=dev['experiment_completed_updates']; seed=dev['seed']
            if seed not in TRAIN_SEEDS or k is None or k<=0: continue
            rank=selection_rank(nominal,dev,k)
            if seed not in best or rank>best[seed]['rank']:
                best[seed]={'rank':rank,'dev':dev,'reports':[reference(nominal_path),reference(dev_path)]}
    required_seeds=TRAIN_SEEDS[:1] if args.screening_only else TRAIN_SEEDS
    if args.screening_only and not args.stop_reason:
        raise ValueError('Screening-only closure requires the documented reason for stopping')
    if not set(required_seeds)<=set(best):
        raise ValueError('Final selection requires all three fine-tuning seeds')
    checkpoints=[]; evidence=[]; campaigns=set()
    for seed in required_seeds:
        entry=best[seed]; report=entry['dev']
        path=Path(report['checkpoint'])
        if checksum(path)!=report['checkpoint_sha256']:
            raise ValueError('Selected checkpoint changed')
        checkpoints.append({**reference(path),'role':'candidate','profile':args.profile,'seed':seed,
                            'experiment_completed_updates':report['experiment_completed_updates'],'rank':entry['rank']})
        evidence.extend(entry['reports']); campaigns.add(report['campaign_id'])
    if len(campaigns)!=1:
        raise ValueError('Candidates come from different campaign budgets')
    for path,role in ((args.source,'source-1000'),(args.stage07_final,'reference-6000')):
        checkpoints.append({**reference(path),'role':role})
    selection={'schema_version':1,'protocol_version':PROTOCOL_VERSION,'selection_locked':True,
               'created_utc':datetime.now(timezone.utc).isoformat(),'campaign_id':campaigns.pop(),
               'checkpoints':checkpoints,'development_reports':evidence,
               'held_out_used_for_selection':False,'stage08_complete':False}
    selection.update(training_seeds=list(required_seeds),screening_only=args.screening_only,stop_reason=args.stop_reason)
    with args.output.open('x') as stream:
        json.dump(selection,stream,indent=2,allow_nan=False); stream.write('\n')
    print('Stage08Selection=FROZEN')


if __name__=='__main__': main()
