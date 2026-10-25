"""Host-only partial and combined review; retain passed cells on any failure."""
from pathlib import Path
import argparse,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from offline_review import j,need,segments,resources,trajectories,read,exact_arrays
from report_io import save
from aux_raw_review import audit as audit_aux

def pair(left,lmode,right,rmode,world):
    a,am=read(left/(lmode+'-endpoint'));b,bm=read(right/(rmode+'-endpoint'))
    e=exact_arrays(a,b);t=trajectories(left/(lmode+'-raw'),right/(rmode+'-raw'))
    return dict(world=world,left=left.name+'/'+lmode,right=right.name+'/'+rmode,
                endpoint=e,schema_exact=am==bm,trajectory=t,passed=e['passed'] and am==bm and t['passed'])

def partial(previous):
    p=previous/'probe/world0-producer';c=previous/'probe/world0-consumer'
    r=pair(p,'eager',p,'graph',0);need(r['passed'],'Previously reviewed pair differs')
    seg=segments(p);life=[resources(x) for x in (p,c)];aux=[audit_aux(x) for x in (p,c)]
    need(life[0]['logical']==life[1]['logical'],'Old reconstructed resource descriptions differ')
    counts=j(previous/'run-summary.json')['counts']
    need(counts['integrations_completed']==counts['integrations_attempted']==704,'Previous count changed')
    return dict(status='G10_WORLD0_PRODUCER_RAW_REVIEW_PASS',pair=r,segment=seg,
                resources=[dict(process=x.name,targets=len(y['targets'])) for x,y in zip((p,c),life)],auxiliary=aux,
                consumer_status='FAILED_AFTER_EXACT_LOAD_BEFORE_ANY_INTEGRATION',
                comparisons_passed=1,comparisons_total=6,canonical_completed=666200,canonical_attempted=666201,
                native_integrations=0,G10='OPEN',G11='OPEN')

def audit(run,previous):
    prior=partial(previous);pairs=[];processes=[];issues=[];total=attempted=0
    for name,count in (('world0-consumer',352),('world1-producer',704),('world1-consumer',352)):
        path=run/'probe'/name;item=dict(process=name,passed=False)
        try:
            report=j(path/'run-review.json');finish=j(path/'execution-finished.json')
            c=report['counts'];total+=c.get('integrations_completed',0);attempted+=c.get('integrations_attempted',0)
            need(report['status']=='G10_SEGMENTS_REBUILD_MEASURED_REVIEW_REQUIRED' and finish['exit_code']==0 and finish['process_has_exited'],'Worker incomplete: '+name)
            need(c['integrations_completed']==c['integrations_attempted']==count,'Native schedule differs')
            seg=segments(path);life=resources(path);aux=audit_aux(path)
            for label,n in (('capture_begin',seg['captures']),('capture_end',seg['captures']),('launch',seg['launches'])):
                need(c['Segment_'+label+'_completed']==c['Segment_'+label+'_attempted']==n,'Graph count differs')
            need(report['whole_run_restore']['all_numeric_Data_exact'] and report['whole_run_restore']['selected_owners_exact'],'Whole-run restoration incomplete')
            item.update(passed=True,segments=seg,resource_targets=len(life['targets']),auxiliary=aux)
        except Exception as exc:item['error']=repr(exc);issues.append(dict(process=name,error=repr(exc)))
        processes.append(item)
    for world in (0,1):
        producer=(previous if world==0 else run)/'probe'/('world%d-producer'%world)
        consumer=run/'probe'/('world%d-consumer'%world)
        try:
            pp=j(producer/'run-review.json');cp=j(consumer/'run-review.json');order=j(run/('world%d-process-order.json'%world))
            need(order['producer_exited_before_consumer_started'] and order['producer_exit_code']==0,'Process dependency violated')
            # Monotonic timestamps from different boots must never be ordered.
            if world==1:need(order['producer_exit_monotonic']<=order['consumer_start_monotonic'],'Same-run ordering differs')
            else:need(order['producer_archive_sha256']=='4556cc3189d312cff1390fabbe04652e45508e38f9910d6b450284cdfb3089c2','Reused process source differs')
            need(pp['identity']['owner_id']!=cp['identity']['owner_id'],'Process nonce reused')
            a,am=read(producer/'logical-start');b,bm=read(consumer/'logical-start')
            need(exact_arrays(a,b)['passed'] and am==bm,'Rebuilt start differs')
            boundary=j(consumer/'rebuild-boundary.json')
            need(boundary['passed'] and boundary['sense_calls']==1 and boundary['physics_integrations']==0 and boundary['fields_exempted']==[],'Boundary proof incomplete')
            raw,meta=read(consumer/'receiver-refreshed')
            need(exact_arrays(a,raw)['passed'] and am==meta,'Raw full rebuild boundary differs')
            loaded,lmeta=read(consumer/'receiver-loaded')
            need(exact_arrays(a,loaded)['passed'] and am==lmeta,'Raw loaded checkpoint differs')
            need(resources(producer)['logical']==resources(consumer)['logical'],'Resource geometry/schema differs')
        except Exception as exc:issues.append(dict(world=world,kind='reconstruction',error=repr(exc)))
        for left,lmode,right,rmode in ((producer,'eager',producer,'graph'),(producer,'eager',consumer,'restored-graph'),(producer,'graph',consumer,'restored-graph')):
            try:p=pair(left,lmode,right,rmode,world)
            except Exception as exc:p=dict(world=world,left=left.name+'/'+lmode,right=right.name+'/'+rmode,passed=False,error=repr(exc))
            pairs.append(p)
    complete=all(x['passed'] for x in pairs) and all(x['passed'] for x in processes) and not issues and total==attempted==1408
    for n in ('torch','warp','mujoco','mujoco_warp','mjlab'):need(n not in sys.modules,'Native import in host audit')
    return dict(status='G10_RESUME_RAW_HOST_REVIEW_PASS' if complete else 'G10_RESUME_PARTIAL_REVIEW_REQUIRED',
                reused_prior_review=prior,pairs=pairs,processes=processes,issues=issues,
                comparisons_passed=sum(x['passed'] for x in pairs),comparisons_total=6,
                new_integrations_completed=total,new_integrations_attempted=attempted,
                canonical_completed=666200+total,canonical_attempted=666201+attempted,
                G09='CLOSED_SCOPED',G10='OPEN',G11='OPEN',Stage11_gate_percent=100*9/11,
                full_native_RK4_graph=False,arbitrary_mid_rollout_checkpoint=False,formal_environment=False,
                auditor_native_calls=0,PPO=0,cloud_training=0)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);parser.add_argument('--previous-run',type=Path,required=True);a=parser.parse_args()
    try:r=audit(a.run,a.previous_run)
    except Exception as exc:r=dict(status='G10_RESUME_AUDIT_BLOCKED',error=repr(exc),G10='OPEN',G11='OPEN',auditor_native_calls=0)
    save(a.run/'new-raw-review.json',r);print(json.dumps(r,ensure_ascii=False,indent=2))
    return 0 if r['status']=='G10_RESUME_RAW_HOST_REVIEW_PASS' else 2
if __name__=='__main__':raise SystemExit(main())
