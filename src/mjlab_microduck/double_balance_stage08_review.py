"""Evidence gates for replication/extension; no training or task changes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from mjlab_microduck.double_balance_stage08_plan import PROFILES, SOURCE_SHA256, TRAIN_SEEDS


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def reference(path):
    return {'path':str(Path(path).resolve()),'sha256':checksum(path)}


def read_reference(item):
    if checksum(item['path']) != item['sha256']:
        raise ValueError('Decision evidence has changed')
    return json.loads(Path(item['path']).read_text())


def comparable(report, baseline):
    if (report['status'] != 'PASS' or baseline['status'] != 'PASS' or
            report['protocol'] != baseline['protocol'] or report['protocol'] != 'dev' or
            report['protocol_version'] != baseline['protocol_version'] or
            report['initial_states']['sha256'] != baseline['initial_states']['sha256']):
        raise ValueError('Reports are not comparable development evaluations')


def improvement_gate(pair, baseline):
    if len(pair) != 2:
        raise ValueError('Two adjacent development evaluations are required')
    first,last = sorted(pair,key=lambda r:r['experiment_completed_updates'])
    for report in (first,last):
        comparable(report,baseline)
    if (last['experiment_completed_updates']-first['experiment_completed_updates'] != 250 or
            first['profile'] != last['profile'] or first['seed'] != last['seed']):
        raise ValueError('Development pair must be adjacent points in the same lineage')
    # Operational budget gates, not new success definitions or significance tests.
    strict = (last['success_fraction'] > baseline['success_fraction'] and
              last['termination_fraction'] <= baseline['termination_fraction']+.05)
    trajectory = all(
        r['median_terminal_stable_seconds'] > baseline['median_terminal_stable_seconds'] and
        r['median_longest_stable_seconds'] > baseline['median_longest_stable_seconds'] and
        r['mean_lower_speed_violation_fraction'] <= .75*baseline['mean_lower_speed_violation_fraction'] and
        r['mean_survival_seconds'] >= 9.5
        for r in (first,last))
    return {'passed':bool(strict or trajectory),'new_strict_success':bool(strict),
            'two_point_stability_and_speed_improvement':bool(trajectory),
            'not_a_statistical_significance_test':True}


def deteriorated(report,baseline):
    comparable(report,baseline)
    return (report['success_fraction'] <= baseline['success_fraction'] and
            ((report['mean_survival_seconds'] < baseline['mean_survival_seconds']-.5) or
             (report['median_terminal_stable_seconds'] <= baseline['median_terminal_stable_seconds'] and
              report['median_longest_stable_seconds'] < baseline['median_longest_stable_seconds'] and
              report['mean_lower_speed_violation_fraction'] >= baseline['mean_lower_speed_violation_fraction'])))


def validate_decision(path, *, profile, target, campaign_id):
    if target <= 500:
        return None
    if path is None:
        raise ValueError('Beyond screening, supply a hashed development decision')
    decision = json.loads(Path(path).read_text()) if not isinstance(path,dict) else path
    if decision['schema_version'] != 1 or decision['campaign_id'] != campaign_id:
        raise ValueError('Decision belongs to a different campaign')
    selected = decision['selected_profiles']
    if profile not in selected or any(p not in PROFILES for p in selected):
        raise ValueError('Profile is not selected')
    baseline = read_reference(decision['baseline_report'])
    if not (baseline.get('experiment_completed_updates') == 0 or baseline['checkpoint_sha256'] == SOURCE_SHA256):
        raise ValueError('Gate baseline must be the source update 1000 model')
    reports = [read_reference(r) for r in decision['candidate_reports']]
    if any(r.get('campaign_id') != campaign_id for r in reports):
        raise ValueError('Candidate evidence belongs to a different campaign')
    if target <= 1500:
        if decision['kind'] != 'replication' or len(set(selected)) != 2:
            raise ValueError('Replication needs two configurations')
        if abs(ord(selected[0])-ord(selected[1])) != 1:
            raise ValueError('Use adjacent nested controls; propose new treatments separately')
        if (len(reports) != 2 or reports[0]['profile'] not in selected or
                {r['experiment_completed_updates'] for r in reports} != {250,500} or
                {r['seed'] for r in reports} != {TRAIN_SEEDS[0]}):
            raise ValueError('Replication gate needs screening points 250/500')
        gates = [improvement_gate(reports,baseline)]
    else:
        if decision['kind'] != 'extension' or len(selected) != 1 or target > 4000:
            raise ValueError('Extension needs one configuration and at most 4000 updates')
        if len(reports) != 6 or any(r['profile'] != profile for r in reports):
            raise ValueError('Extension needs three seeds and two points per seed')
        gates = []
        for seed in TRAIN_SEEDS:
            pair = [r for r in reports if r['seed']==seed]
            if {r['experiment_completed_updates'] for r in pair} != {1250,1500}:
                raise ValueError('Extension requires every seed at 1250/1500')
            gates.append(improvement_gate(pair,baseline))
    if not all(g['passed'] for g in gates):
        raise ValueError('Development improvement gate did not pass; analyze failures before more training')
    return {**decision,'computed_gates':gates}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind',choices=('replication','extension'),required=True)
    parser.add_argument('--campaign-id',required=True)
    parser.add_argument('--profiles',nargs='+',choices=tuple(PROFILES),required=True)
    parser.add_argument('--baseline',type=Path,required=True)
    parser.add_argument('--reports',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    value={'schema_version':1,'kind':args.kind,'campaign_id':args.campaign_id,
           'selected_profiles':args.profiles,'baseline_report':reference(args.baseline),
           'candidate_reports':[reference(p) for p in args.reports],
           'created_utc':datetime.now(timezone.utc).isoformat()}
    checked=validate_decision(value,profile=args.profiles[0],target=1500 if args.kind=='replication' else 4000,
                              campaign_id=args.campaign_id)
    with args.output.open('x') as stream:
        json.dump(checked,stream,indent=2,allow_nan=False)
        stream.write('\n')
    print('Stage08DevelopmentGate=PASS')
    print('FormalTrainingStarted=False')


if __name__=='__main__':
    main()
