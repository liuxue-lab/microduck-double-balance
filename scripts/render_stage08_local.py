#!/usr/bin/env python3
"""Local inference/video only, using a persisted Stage 08 evaluation protocol."""
import argparse
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from mjlab_microduck.double_balance_stage08_state import environment_preflight,file_sha256
from mjlab_microduck.double_balance_stage08_evaluation import evaluate_one


def choose_episodes(report):
    episodes=report['episodes']
    median=statistics.median(e['longest_stable_seconds'] for e in episodes)
    choices={'fixed-env0':0,
             'median-stability':min(episodes,key=lambda e:(abs(e['longest_stable_seconds']-median),e['env_id']))['env_id'],
             'worst-speed':max(episodes,key=lambda e:(e['lower_speed_violation_fraction'],-e['env_id']))['env_id']}
    success=next((e for e in episodes if e['success']),None)
    if success is not None: choices['success']=success['env_id']
    return choices


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--reference-report',type=Path,required=True)
    p.add_argument('--datasets',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    args=p.parse_args()
    reference=json.loads(args.reference_report.read_text())
    if reference['protocol'] not in ('nominal','dev'):
        raise ValueError('Use nominal/development videos; held-out data stays out of model selection')
    if file_sha256(args.checkpoint)!=reference['checkpoint_sha256']:
        raise ValueError('Video model differs from the reference report')
    dataset=args.datasets/f'{reference["protocol"]}.pt'
    if not dataset.exists() or file_sha256(dataset)!=reference['initial_states']['sha256']:
        raise ValueError('Download the matching initial-state set before local rendering')
    head=environment_preflight(None,cloud=False)
    args.output.mkdir(parents=True,exist_ok=False)
    choices=choose_episodes(reference)
    for env_id in sorted(set(choices.values())):
        evaluate_one(args.checkpoint.resolve(),args.output/f'env-{env_id}',reference['protocol'],
                     args.datasets.resolve(),device=args.device,head=head,video_env_id=env_id)
    (args.output/'video-selection.json').write_text(json.dumps({
        'choices':choices,'reference_report':str(args.reference_report.resolve()),
        'purpose':'local inference diagnostic; does not replace cloud evaluation',
        'ppo_updates':0,'human_video_review':'PENDING'},indent=2)+'\n')


if __name__=='__main__': main()
