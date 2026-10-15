"""Pure NumPy audit of new raw bytes. No simulator or CUDA imports."""
from pathlib import Path
import argparse,hashlib,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from portable_state import read,exact_arrays
from report_io import save

def need(ok,msg):
    if not ok:raise RuntimeError(msg)
def j(p):return json.loads(p.read_bytes())
def arrays(p):
    with np.load(p,allow_pickle=False) as d:return {n:d[n].copy() for n in d.files}
def journal(path):
    inv=j(path/'evidence-inventory.json')
    for name,row in inv['files'].items():
        b=(path/name).read_bytes();need(len(b)==row['bytes'] and hashlib.sha256(b).hexdigest()==row['sha256'],'Journal file changed: '+name)
    events=[j(path/n) for n in inv['events']]
    micros=[e for e in events if e['kind']=='microstep']
    need(len(micros)==352,'Incomplete declared trajectory')
    trials=[e['row'] for e in events if e['kind']=='trial_start']
    need([(x['resolution'],x['role']) for x in trials]==[(32,'estimate'),(64,'candidate'),(128,'retry'),(128,'replay')],'Adaptive branch schedule differs')
    need(len([x for x in events if x['kind']=='rollback'])==3 and len([x for x in events if x['kind']=='exact_replay'])==1,'Actual rollback/replay missing')
    decisions=[x['row'] for x in events if x['kind']=='decision']
    need([x['accepted'] for x in decisions]==[False,True],'Actual rejection/acceptance missing')
    return events
def trajectories(a,b,journal_reader=journal):
    left,right=journal_reader(a),journal_reader(b);differences=[];snapshots=0;microsteps=0
    need(len(left)==len(right),'Journal length differs')
    for i,(x,y) in enumerate(zip(left,right)):
        need(x['kind']==y['kind'] and x['sequence']==y['sequence'],'Journal event sequence differs')
        kind=x['kind'];u,v=x['row'],y['row']
        if kind=='microstep':
            keys=('resolution','micro_index','micro_dt_s','stage_sha256','post_integrator','stages','correction','velocity_bounds')
            if any(u[k]!=v[k] for k in keys):differences.append(dict(event=i,kind=kind))
            microsteps+=1
        elif kind in ('decision','rollback','exact_replay'):
            if u!=v:differences.append(dict(event=i,kind=kind))
        if x['snapshots']:
            need(x['snapshots']==y['snapshots'],'Snapshot roster differs')
            for name in x['snapshots']:
                if name.endswith('.npz'):
                    proof=exact_arrays(arrays(a/name),arrays(b/name));snapshots+=1
                    if not proof['passed']:differences.append(dict(event=i,file=name,proof=proof))
                elif name.endswith('-snapshot.json'):
                    xm,ym=j(a/name),j(b/name)
                    for key in ('Data','owners','owner_scalars','global_rng','model_hashes'):
                        if xm[key]!=ym[key]:differences.append(dict(event=i,file=name,metadata=key))
    return dict(passed=not differences,microsteps=microsteps,solved_stages=microsteps*4,raw_snapshots=snapshots,differences=differences[:32],difference_count=len(differences),numeric_fields_exempted=[])

def segments(path,expected_labels=None):
    summary=j(path/'segment-summary.json');rows=[json.loads(line) for line in (path/'segments.jsonl').read_text().splitlines()]
    captured={};released=set();launches=0;stale=0
    for row in rows:
        event=row['event'];serial=row['serial']
        if event=='capture':
            need(serial not in captured and row['physics_integrations']==0,'Graph capture counted as physics or duplicated');captured[serial]=row
        elif event=='launch':
            need(serial in captured and serial not in released and row['graph']==captured[serial]['graph'] and row['exec'],'Unowned graph launch');launches+=1
        elif event=='stale_rejected':need(row['before_dispatch'],'Stale guard did not precede dispatch');stale+=1
        elif event=='release':
            need(serial in captured and serial not in released and row['weakref_dead'],'Duplicate or unowned graph release')
            need(row['graph']==captured[serial]['graph'],'Graph release identity mismatch')
            for api,key in (('wp_cuda_graph_destroy','graph'),('wp_cuda_graph_exec_destroy','exec')):
                matches=[x for x in row['native_callbacks'] if x['api']==api and x['handle']==row[key]]
                need(len(matches)==1 and matches[0]['result'] and matches[0]['returned_normally'] and not matches[0]['capturing'],'Missing raw native destroy')
            released.add(serial)
    need(len(captured)==len(released)==summary['captures'] and launches==summary['launches'] and stale==1 and summary['closed'],'Segment schedule or release incomplete')
    need(set(summary['per_label'])==(set(expected_labels) if expected_labels is not None else {'Norm_kernel','Cost_rows_kernel','Cost_reduce_kernel','Cost_override_copy','SolverOrder_kernel','Rigid_metadata_kernel'}),'A declared numerical segment was never executed')
    return dict(captures=len(captured),launches=launches,releases=len(released),stale_rejections=stale)

def resources(path):
    life=j(path/'resource-lifetime.json');need(life['closed'] and life['targets'],'No closed resource inventory')
    for target in life['targets']:
        found=[x for x in life['releases'] if x['api']==target['api'] and target['handle'] in x['args']]
        need(target['weakref_dead'] and target['released'] and len(found)==1 and found[0]['returned_normally'] and not found[0]['capturing'],'Resource native release incomplete')
    return life

def audit(run):
    probe=run/'probe';pairs=[];processes=[];total=attempted=0
    for world in (0,1):
        producer=probe/('world%d-producer'%world);consumer=probe/('world%d-consumer'%world)
        pp,cp=j(producer/'run-review.json'),j(consumer/'run-review.json')
        identity=j(run/('world%d-process-order.json'%world))
        need(identity['producer_exited_before_consumer_started'] and identity['producer_exit_code']==0,'Old process was not exited before rebuild')
        need(identity['producer_exit_monotonic']<=identity['consumer_start_monotonic'],'Process ordering timestamps differ')
        need(pp['identity']['owner_id']!=cp['identity']['owner_id'],'New process identity not unique')
        # PIDs and virtual addresses may be reused; identity is a nonce plus an
        # observed exited process, never a requirement for unequal pointer bits.
        for path,report,count in ((producer,pp,704),(consumer,cp,352)):
            need(report['status']=='G10_SEGMENTS_REBUILD_MEASURED_REVIEW_REQUIRED','Incomplete worker')
            need(report['counts']['integrations_completed']==report['counts']['integrations_attempted']==count,'Integration count differs')
            total+=report['counts']['integrations_completed'];attempted+=report['counts']['integrations_attempted']
            segment=segments(path)
            for label,expected in (('capture_begin',segment['captures']),('capture_end',segment['captures']),('launch',segment['launches'])):
                need(report['counts']['Segment_'+label+'_attempted']==report['counts']['Segment_'+label+'_completed']==expected,'Graph API counter mismatch')
            processes.append(dict(path=path.name,segment=segment,resource_targets=len(resources(path)['targets'])))
        start_a,meta_a=read(producer/'logical-start');start_b,meta_b=read(consumer/'logical-start')
        need(exact_arrays(start_a,start_b)['passed'] and meta_a==meta_b,'Restored initial state differs')
        need(cp['portable_restore']['challenge_restored'] and cp['portable_restore']['native_pointers_imported']==0 and cp['rebuild_sense_exact']['passed'],'Portable restore/refit proof missing')
        need(resources(producer)['logical']==resources(consumer)['logical'],'Rebuilt resource geometry/roster differs')
        for left,lmode,right,rmode in ((producer,'eager',producer,'graph'),(producer,'eager',consumer,'restored-graph'),(producer,'graph',consumer,'restored-graph')):
            a,am=read(left/(lmode+'-endpoint'));b,bm=read(right/(rmode+'-endpoint'))
            endpoint=exact_arrays(a,b);trajectory=trajectories(left/(lmode+'-raw'),right/(rmode+'-raw'))
            pairs.append(dict(world=world,left=left.name+'/'+lmode,right=right.name+'/'+rmode,endpoint=endpoint,schema_exact=am==bm,trajectory=trajectory,passed=endpoint['passed'] and am==bm and trajectory['passed']))
    need(total==attempted==2112,'Aggregate native count differs')
    for name in ('torch','warp','mujoco','mujoco_warp','mjlab'):need(name not in sys.modules,'Unexpected native import in auditor')
    return dict(status='G10_NEW_RAW_HOST_REVIEW_PASS' if all(x['passed'] for x in pairs) else 'G10_NEW_RAW_DIFFERENCES_REVIEW_REQUIRED',
                pairs=pairs,comparisons_passed=sum(x['passed'] for x in pairs),comparisons_total=6,processes=processes,
                integrations_completed=total,integrations_attempted=attempted,canonical_completed=665496+total,canonical_attempted=665497+attempted,
                auditor_native_calls=0,G09='CLOSED_SCOPED',G10='OPEN',G11='OPEN',Stage11_gate_percent=100*9/11,
                full_native_RK4_graph=False,arbitrary_mid_rollout_checkpoint=False,formal_environment=False,PPO=0,cloud_training=0)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',type=Path,required=True);a=parser.parse_args()
    try:result=audit(a.run)
    except Exception as exc:result=dict(status='G10_RAW_AUDIT_INCOMPLETE',error=repr(exc),auditor_native_calls=0,G10='OPEN',G11='OPEN')
    save(a.run/'new-raw-review.json',result);print(json.dumps(result,ensure_ascii=False,indent=2))
    return 0 if result['status']=='G10_NEW_RAW_HOST_REVIEW_PASS' else 2
if __name__=='__main__':raise SystemExit(main())
