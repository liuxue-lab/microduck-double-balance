"""Independent raw host auditor for expanded graphs and live endpoint recovery."""
from pathlib import Path
import argparse,collections,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
from offline_review import j,need,segments,resources,trajectories,read,exact_arrays,arrays
from aux_raw_review import audit as audit_aux
from report_io import save

BLOCKS={'forward':('fwd_velocity','fwd_actuation','fwd_acceleration','_rk_perturb_state','_rk_accumulate','_advance'),
        'smooth':('kinematics','com_pos','camlight','crb','transmission'),'constraint':('make_constraint',),
        'collision':('collision',),'sensor':('sensor_pos','sensor_vel','sensor_acc'),
        'solver':('_update_constraint','_update_gradient','_update_gradient_incremental','_linesearch')}
BASE={'Norm_kernel','Cost_rows_kernel','Cost_reduce_kernel','Cost_override_copy','SolverOrder_kernel','Rigid_metadata_kernel'}

def fixed_journal(path):
    inv=j(path/'evidence-inventory.json')
    for name,row in inv['files'].items():
        raw=(path/name).read_bytes();need(row==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),'Fixed journal file changed')
    events=[j(path/n) for n in inv['events']];micros=[e['row'] for e in events if e['kind']=='microstep']
    need(len(micros)==128 and [r['micro_index'] for r in micros]==list(range(128)) and all(r['resolution']==128 for r in micros),'Fixed128 microstep schedule differs')
    trials=[e['row'] for e in events if e['kind']=='trial_start']
    need([(x['resolution'],x['role']) for x in trials]==[(128,'fixed_continuation')],'Fixed continuation trial differs')
    need(not any(e['kind'] in ('decision','exact_replay','rollback') for e in events),'Fixed diagnostic claimed adaptive acceptance')
    commits=[e['row'] for e in events if e['kind']=='commit_prepared']
    need(len(commits)==1 and commits[0]['adaptive_acceptance'] is False and commits[0]['diagnostic_clock_publication_only'],'Fixed commit semantics differ')
    return events

def graph_audit(path):
    summary=j(path/'segment-summary.json');coverage=j(path/'native-graph-coverage.json')
    required=BASE|{m+'.'+n for m,ns in BLOCKS.items() for n in ns};actual=set(summary['per_label'])
    need(required<=actual and actual<=required|{'native.'+n for n in ('launch','launch_tiled','copy','zero_','fill_')},'Native graph coverage incomplete or undeclared')
    lifecycle=segments(path,expected_labels=actual)
    rows=[json.loads(line) for line in (path/'segments.jsonl').read_text().splitlines()]
    captures={str(r['serial']):r for r in rows if r['event']=='capture'};counts=collections.Counter()
    need(set(captures)==set(coverage['captured_operations'])==set(coverage['operation_specs']),'Graph operation census roster differs')
    for key,row in captures.items():
        spec=coverage['operation_specs'][key]
        need(spec['label']==row['label'] and all(type(v) is int and v>=0 for v in coverage['captured_operations'][key].values()),'Invalid captured operation census')
    for row in rows:
        if row['event']=='launch':counts.update(coverage['captured_operations'][str(row['serial'])])
        if row['event']=='stale_rejected':need(row['actual_Data_qpos_temporarily_rebound'] and row['original_binding_restored'],'No actual stale-binding challenge')
    need(dict(counts)==coverage['logical_device_operations'],'Logical capture API census differs')
    need(counts['launch']>0 and counts['launch_tiled']>0 and counts['copy']>0 and counts['zero_']>0,'Missing native device operation family')
    need(coverage['numeric_fields_exempted']==[],'Graph comparator exemptions present')
    return dict(**lifecycle,native_blocks=20,labels=len(actual),logical_device_operations=dict(counts))

def pair(left,lmode,right,rmode,world,fixed):
    a,am=read(left/(lmode+'-endpoint'));b,bm=read(right/(rmode+'-endpoint'))
    endpoint=exact_arrays(a,b)
    trajectory=trajectories(left/(lmode+'-raw'),right/(rmode+'-raw'),fixed_journal) if fixed else trajectories(left/(lmode+'-raw'),right/(rmode+'-raw'))
    return dict(world=world,left=left.name+'/'+lmode,right=right.name+'/'+rmode,endpoint=endpoint,schema_exact=am==bm,
                trajectory=trajectory,passed=endpoint['passed'] and am==bm and trajectory['passed'])

def audit(run,previous,cache):
    processes=[];pairs=[];issues=[];total=attempted=0
    for world in (0,1):
        producer=run/f'probe/world{world}-producer';consumer=run/f'probe/world{world}-consumer'
        old=(previous if world==0 else cache)/f'probe/world{world}-producer'
        for path,native_count,forward_count in ((producer,608,2443),(consumer,128,516)):
            item=dict(process=path.name,passed=False)
            try:
                c=j(path/'progress.json');total+=c.get('integrations_completed',0);attempted+=c.get('integrations_attempted',0)
                r=j(path/'run-review.json');finish=j(path/'execution-finished.json')
                need({k:v for k,v in c.items() if k!='elapsed_s'}==r['counts'],'Durable/reported counters differ')
                need(r['status']=='G10_BLOCKS_RECOVERY_MEASURED_REVIEW_REQUIRED' and finish['exit_code']==0 and finish['process_has_exited'],'Worker incomplete')
                need(r['counts']['integrations_completed']==r['counts']['integrations_attempted']==native_count,'Integration schedule differs')
                need(r['counts']['GPU_forward_completed']==r['counts']['GPU_forward_attempted']==forward_count,'Forward schedule differs')
                graph=graph_audit(path);life=resources(path);aux=audit_aux(path)
                tables=arrays(path/'model-nested-initial.npz');table_review=j(path/'model-nested-review.json')
                need(tables and table_review['fields']==len(tables) and table_review['native_handle_imports']==0,'Nested Model table inventory missing')
                for check in table_review['checks']:
                    need(exact_arrays(tables,arrays(path/('model-nested-'+check['label']+'.npz')))['passed'],'Nested Model raw bytes changed')
                need({'loaded','final'}<={x['label'] for x in table_review['checks']},'Nested Model boundaries incomplete')
                for label,n in (('capture_begin',graph['captures']),('capture_end',graph['captures']),('launch',graph['launches'])):
                    need(r['counts']['Segment_'+label+'_completed']==r['counts']['Segment_'+label+'_attempted']==n,'Graph API ledger mismatch')
                need(r['whole_run_restore']['all_numeric_Data_exact'] and r['whole_run_restore']['selected_owners_exact'],'Whole-run restoration incomplete')
                a,am=read(path/'logical-endpoint');b,bm=read(path/'post-continuation-restored')
                need(exact_arrays(a,b)['passed'] and am==bm,'Post-continuation raw restore differs')
                item.update(passed=True,graph=graph,resource_targets=len(life['targets']),auxiliary=aux,nested_Model_fields=len(tables))
            except Exception as exc:item['error']=repr(exc);issues.append(dict(process=path.name,error=repr(exc)))
            processes.append(item)
        try:
            pp=j(producer/'run-review.json');cp=j(consumer/'run-review.json');order=j(run/f'world{world}-process-order.json')
            old_start,old_schema=read(old/'logical-start')
            for name in ('logical-start','receiver-loaded','receiver-refreshed'):
                actual,metadata=read(producer/name)
                need(exact_arrays(old_start,actual)['passed'] and old_schema==metadata,'Expanded producer archived start differs: '+name)
            need(order['producer_exited_before_consumer_started'] and order['producer_exit_code']==0 and order['producer_exit_monotonic']<=order['consumer_start_monotonic'],'Producer still alive at rebuild')
            need(pp['identity']['owner_id']!=cp['identity']['owner_id'],'Owner nonce reused')
            a,am=read(producer/'logical-endpoint')
            for name in ('logical-endpoint','receiver-loaded','receiver-refreshed'):
                b,bm=read(consumer/name);need(len(b)==636 and exact_arrays(a,b)['passed'] and am==bm,'Complete live endpoint rebuild differs: '+name)
            material=j(consumer/'endpoint-materialization.json');boundary=j(consumer/'rebuild-boundary.json')
            need(material['schema_exact'] and len(material['new_tensors'])==3 and len(material['scalar_slots'])==5 and material['physics_integrations']==material['action_applications']==material['history_pushes']==0,'Materialization changed unapproved work')
            need(boundary['passed'] and boundary['sense_calls']==1 and boundary['physics_integrations']==0 and boundary['fields_exempted']==[],'Rebuild sense/caches incomplete')
            need(resources(producer)['logical']==resources(consumer)['logical'],'Native resource geometry differs')
            need(exact_arrays(arrays(producer/'model-nested-initial.npz'),arrays(consumer/'model-nested-initial.npz'))['passed'],'Rebuilt static nested Model tables differ')
        except Exception as exc:issues.append(dict(world=world,kind='endpoint_recovery',error=repr(exc)))
        for left,lmode,right,rmode,fixed in ((old,'eager',producer,'adaptive-graph',False),(old,'graph',producer,'adaptive-graph',False),
            (producer,'fixed-eager',producer,'fixed-graph',True),(producer,'fixed-eager',consumer,'restored-fixed-graph',True),(producer,'fixed-graph',consumer,'restored-fixed-graph',True)):
            try:r=pair(left,lmode,right,rmode,world,fixed)
            except Exception as exc:r=dict(world=world,left=left.name+'/'+lmode,right=right.name+'/'+rmode,passed=False,error=repr(exc))
            pairs.append(r)
    complete=not issues and all(x['passed'] for x in processes+pairs) and total==attempted==1472
    for n in ('torch','warp','mujoco','mujoco_warp','mjlab'):need(n not in sys.modules,'Native import in offline review')
    return dict(status='G10_BLOCKS_RECOVERY_RAW_HOST_REVIEW_PASS' if complete else 'G10_BLOCKS_RECOVERY_PARTIAL_REVIEW_REQUIRED',
                processes=processes,pairs=pairs,issues=issues,comparisons_passed=sum(x['passed'] for x in pairs),comparisons_total=10,
                reused_previous_comparisons=6,new_integrations_completed=total,new_integrations_attempted=attempted,
                canonical_completed=667608+total,canonical_attempted=667609+attempted,
                G09='CLOSED_SCOPED',G10='OPEN',G11='OPEN',Stage11_gate_percent=100*9/11,
                full_native_RK4_graph=False,arbitrary_mid_rollout_checkpoint=False,formal_environment=False,
                auditor_native_calls=0,PPO=0,cloud_training=0)

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--previous-run',type=Path,required=True);p.add_argument('--cache-run',type=Path,required=True);a=p.parse_args()
    try:r=audit(a.run,a.previous_run,a.cache_run)
    except Exception as exc:r=dict(status='G10_BLOCKS_AUDIT_BLOCKED',error=repr(exc),G10='OPEN',G11='OPEN',auditor_native_calls=0)
    save(a.run/'new-raw-review.json',r);print(json.dumps(r,ensure_ascii=False,indent=2))
    return 0 if r['status']=='G10_BLOCKS_RECOVERY_RAW_HOST_REVIEW_PASS' else 2
if __name__=='__main__':raise SystemExit(main())
