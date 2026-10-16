"""One newly authorized cloud acceptance point. No policy, optimizer or checkpoint I/O."""
from pathlib import Path
import argparse
import collections
import json
import os
import statistics
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT),str(ROOT/'candidate')]
from common import BASELINE, read, write, require, sha, GIB
from integration.planning import owner_plan, fingerprint


POINTS=[('n1-pilot',1,3),('n2-pilot',2,3),('n1-rollout',1,24),('n2-rollout',2,24),('n1-full-episode',1,500)]


def point_plan(name):
    found=[x for x in POINTS if x[0]==name]
    require(len(found)==1,'Unknown point')
    _,n,steps=found[0]
    plans=[owner_plan(steps,steps+3,i) for i in range(n)]
    return dict(name=name,owners=plans,full_episode=name=='n1-full-episode',public_steps_per_owner=steps,
                max_integrations=sum(x['caps']['integrations'] for x in plans),
                max_GPU_forward=sum(x['caps']['GPU_forward'] for x in plans),
                max_CPU_forward=sum(x['caps']['CPU_forward'] for x in plans),
                evidence_policy='Retain all inherited boundary/microstep/WAL evidence; live 8 GiB return reserve; no full-space reservation claim')


def auth(plan):
    return dict(authorization='USER_AUTHORIZED_NATIVE_MEASUREMENT',scope='STAGE12_CLOUD_5090_EXECUTION_V1',
                owner_plan_sha256=fingerprint(plan),canonical_before=BASELINE,PPO=0,cloud_training=0)


def reconcile(folder, point):
    sys.path.insert(0,str(ROOT/'candidate/baseline/g11'))
    from runtime_accounting import review_ledger
    rows=[]
    for p in point['owners']:
        base=Path(folder)/('owner-%d'%p['logical_world'])
        wal=base/'native-ledger.jsonl'
        if wal.is_file():
            try:
                state=review_ledger(wal)
            except Exception as exc:
                state=dict(exact=False,error=repr(exc),integration_attempted_lower=0,
                           integration_attempted_upper=p['caps']['integrations'],integration_completed_lower=0,
                           integration_completed_upper=p['caps']['integrations'])
        else:
            # No integration is possible before native ledger creation; bootstrap
            # failure is still not a successfully completed owner.
            state=dict(exact=not base.exists(),ledger_absent=True,integration_attempted_lower=0,
                       integration_attempted_upper=0,integration_completed_lower=0,integration_completed_upper=0)
        rows.append(dict(owner=p['logical_world'],ledger=state))
    total={k:sum(x['ledger']['integration_'+k] for x in rows) for k in
           ('attempted_lower','attempted_upper','completed_lower','completed_upper')}
    return dict(owners=rows,native_added=total,all_ledgers_exact=all(x['ledger']['exact'] for x in rows))


def independent_review(folder, expected):
    sys.path[:0]=[str(ROOT),str(ROOT/'candidate/baseline/g11/accepted')]
    import numpy as np
    from recovery.audit import macro_review
    from ground_scalar_contract import assess
    from batch_position_candidate import assess_cohort
    rows=[]; tiers=collections.Counter()
    for i,n in enumerate(expected):
        owner=Path(folder)/('owner-%d'%i)
        actual={x.parent.name for x in owner.glob('macro-*/macro-result.json')}
        require(actual=={'macro-%02d'%k for k in range(n)},'Completed macro roster differs')
        for k in range(n):
            macro=owner/('macro-%02d'%k)
            r=macro_review(macro,np,assess,assess_cohort)
            require(r['passed'] and r['native_acceptance'] and len(r['accepted_replays'])==1,'Independent frozen-criterion readback rejected')
            accepted=read(macro/'macro-result.json')['accepted_resolution']; tiers[accepted]+=1
            rows.append(dict(path=macro.relative_to(folder).as_posix(),accepted_resolution=accepted,
                             result_sha256=sha(macro/'macro-result.json'),raw_inventory_sha256=sha(macro/'raw/evidence-inventory.json'),
                             decisions_recomputed=len(r['decisions']),exact_replays=1,passed=True))
    result=dict(status='ALL_COMPLETED_MACROS_RECOMPUTED_EXACT',completed=len(rows),expected=sum(expected),
                tiers=dict(tiers),cases=rows,new_native_operations=0,PPO=0,cloud_training=0,
                assessor_sha256=sha(ROOT/'candidate/baseline/g11/accepted/ground_scalar_contract.py'))
    write(Path(folder)/'independent-review.json',result)
    return {k:v for k,v in result.items() if k!='cases'}


def measure(repo, folder, point, deadline):
    import torch
    from tensordict import TensorDict
    from integration.process_owners import OwnerProcesses
    start=time.monotonic(); rows=[]; final=[]
    plans=point['owners']; steps=point['public_steps_per_owner']; full=point['full_episode']
    timeout=max(1,min(3600,deadline-time.time()))
    with OwnerProcesses(repo,folder,plans,[auth(x) for x in plans],torch,TensorDict,timeout_seconds=timeout) as env:
        initialized=time.monotonic()
        before=env.inspect()
        env.get_observations(); env.get_observations()
        cached=env.inspect()
        for a,b in zip(before,cached,strict=True):
            for key in ('counts','rng_sha256','observation_sha256','resets','episode_length'):
                require(a[key]==b[key],'Cached observation mutates owner '+key)
        action=torch.zeros(len(plans),14)
        for i in range(steps):
            require(time.time()<deadline,'Native campaign deadline reached')
            env.transport.timeout=max(1,min(3600,deadline-time.time()))
            t=time.monotonic()
            obs,reward,done,extras=env.step(action)
            witness=env.inspect()
            transition=extras['stage12_transition']
            row=dict(index=i,wall_seconds=time.monotonic()-t,reward=reward.tolist(),done=done.tolist(),
                     terminated=transition['terminated'].tolist(),truncated=transition['truncated'].tolist(),
                     bootstrap_mask=transition['bootstrap_mask'].tolist(),witnesses=witness)
            write(Path(folder)/('transition-%04d.json'%i),row); rows.append(row)
            counts=sum(x['counts'].get('integrations_completed',0) for x in witness)
            attempted=sum(x['counts'].get('integrations_attempted',0) for x in witness)
            print('Stage12CloudPoint=%s; public=%d/%d; integrations=%d/%d; elapsed=%.1fs; PPO=0'%
                  (point['name'],i+1,steps,counts,attempted,time.monotonic()-start),flush=True)
            if full and any(row['done']):
                # Never force a timeout or suppress a physical termination to pass.
                break
        if not full:
            old=env.inspect(); env.reset([0]); selected=env.inspect()
            require(selected[0]['resets']==old[0]['resets']+1,'Selective reset count differs')
            for a,b in zip(old[1:],selected[1:],strict=True):
                for key in ('counts','rng_sha256','observation_sha256','resets','episode_length'):
                    require(a[key]==b[key],'Selective reset modified sibling '+key)
            env.reset([]); empty=env.inspect()
            for a,b in zip(selected,empty,strict=True):
                for key in ('counts','rng_sha256','observation_sha256','resets','episode_length'):
                    require(a[key]==b[key],'Empty reset modified owner '+key)
        final=env.inspect()
        require(all(x['public_steps']==len(rows) and x['macros']==10*len(rows) for x in final),'Public/macro count differs')
    close=read(Path(folder)/'owner-closeout.json'); require(close['all_clean'],'Owner cleanup not complete')
    review=independent_review(folder,[x['macros'] for x in final])
    timed=rows[1:]; times=[x['wall_seconds'] for x in timed]
    elapsed=sum(times); transitions=len(timed)*len(plans)
    completed=sum(x['counts'].get('integrations_completed',0) for x in rows[-1]['witnesses'])
    warm=sum(x['counts'].get('integrations_completed',0) for x in rows[0]['witnesses'])
    full_ok=(full and len(rows)==500 and not any(any(r['done']) for r in rows[:-1]) and
             rows[-1]['terminated']==[False] and rows[-1]['truncated']==[True] and final[0]['episode_length']==[0])
    status='NATIVE_POINT_PASS' if not full or full_ok else 'FULL_EPISODE_NOT_COMPLETED'
    return dict(status=status,point=point['name'],owners=len(plans),public_steps_per_owner=len(rows),
                requested_public_steps_per_owner=steps,startup_seconds=initialized-start,total_seconds=time.monotonic()-start,
                timed_steps_per_owner=len(timed),timed_wall_seconds=elapsed,
                public_transitions_per_second=transitions/elapsed if elapsed else None,
                integrations_per_second=(completed-warm)/elapsed if elapsed else None,
                step_median_seconds=statistics.median(times) if times else None,
                step_p95_seconds=sorted(times)[min(len(times)-1,int(.95*len(times)))] if times else None,
                maximum_step_seconds=max(x['wall_seconds'] for x in rows),
                full_10s_episode_proven=full_ok,sustained_sample=not full and steps==24,
                production_training_throughput_proven=False,physical_stability_with_policy_proven=False,
                native_nworld_per_owner=1,final=final,review=review,closeout=close,
                PPO=0,cloud_training=0,training_ready=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--work',type=Path,required=True);p.add_argument('--point',choices=[x[0] for x in POINTS],required=True)
    p.add_argument('--deadline',type=float,required=True);a=p.parse_args()
    point=point_plan(a.point); folder=a.work/'native'/a.point
    require(not folder.exists(),'Native point already started; never repeat')
    os.environ['MD12_NATIVE_DEADLINE']=str(a.deadline)
    report=dict(status='NATIVE_STOPPED_PRESERVE',point=a.point,errors=[],PPO=0,cloud_training=0,training_ready=False)
    try:
        report.update(measure(a.repo,folder,point,a.deadline))
    except BaseException as exc:
        report['errors'].append(dict(error=repr(exc),traceback=traceback.format_exc()))
    finally:
        report['accounting']=reconcile(folder,point)
        if folder.exists():
            report['evidence_bytes']=sum(x.stat().st_size for x in folder.rglob('*') if x.is_file())
        write(a.work/'reports'/(a.point+'.json'),report)
    return int(bool(report['errors']))


if __name__=='__main__':
    raise SystemExit(main())
