"""Read-only independent v3 audit. No simulator imports or native calls."""
from pathlib import Path
import argparse, collections, hashlib, json, sys, zipfile
import numpy as np

parser=argparse.ArgumentParser(description='Read-only raw v3 evidence audit; NumPy only')
parser.add_argument('--return-root',type=Path,required=True)
parser.add_argument('--v2-audit',type=Path,required=True)
cli=parser.parse_args()
R=cli.return_root.resolve()
E=R/'extracted'; P=E/'program'
sys.path.insert(0,str(P/'accepted'))
from ground_scalar_contract import assess
from batch_position_candidate import assess_cohort
def sha(b): return hashlib.sha256(b).hexdigest()
def read(p): return json.loads(p.read_bytes())
checks=[]
archive=R/'g11-formal-environment-return-20261015T144753Z-9d3a47.zip'
assert sha(archive.read_bytes())=='8b7bfce5389308ea02e1260a161f550df34a06dde0110ce405f99eff4229de06'
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==len(set(z.namelist()))==1033
    inv=read(E/'evidence-inventory.json')
    assert set(z.namelist())==set(inv)|{'evidence-inventory.json'}
    for n,row in inv.items():
        b=z.read(n); assert row==dict(bytes=len(b),sha256=sha(b)) and b==(E/n).read_bytes()
checks.append('1033 ZIP members CRC, SHA256, extracted bytes')
files={str(p.relative_to(P)):p.read_bytes() for p in P.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
assert len(files)==149 and set(files)=={str(p.relative_to(E/'program')) for p in (E/'program').rglob('*') if p.is_file() and '__pycache__' not in p.parts}
# Compare to original delivered payload, not producer claims.
original=Path(__file__).resolve().parent/'payload'
assert all(b==(original/n).read_bytes() for n,b in files.items())
checks.append('149 delivered program members exact, including 92 accepted helpers')
assert read(E/'repository-before.json')==read(E/'repository-after.json')
pins=read(E/'evidence/source-pins-before.json')
assert pins==read(E/'evidence/source-pins-after.json') and len(pins)==219
accepted=read(P/'accepted/plan.json')
expected=list(accepted['project_source_hashes'].values())
for rows in accepted['installed_source_hashes'].values(): expected+=list(rows.values())
expected+=[r['sha256'] for r in read(P/'accepted/integration-source-pins.json')]+[accepted['new_source_pin']['sha256']]
assert set(pins.values())==set(expected)
checks.append('219 pinned sources unchanged, HEAD/branch/SSH/status/diff unchanged')
report=read(E/'evidence/report.json'); counts=report['counts']
assert report==read(R/'g11-report.json')
ledger=(E/'evidence/native-ledger.jsonl').read_bytes(); assert ledger.endswith(b'\n')
pending={}; completed=collections.Counter(); attempted=collections.Counter()
for seq,line in enumerate(ledger.splitlines(),1):
    row=json.loads(line); assert row['seq']==seq
    key=row['label'];number=row['number']
    if row['kind']=='attempt':
        assert key not in pending and number>attempted[key]
        if key=='integrations': assert number==attempted[key]+1
        attempted[key]=number;pending[key]=number
    else:
        assert row['kind']=='complete' and pending.pop(key)==number
        completed[key]=number
assert seq==7352 and not pending and completed['integrations']==attempted['integrations']==3584
assert counts['integrations_attempted']==counts['integrations_completed']==3584
assert counts['RK4_integrator_attempted']==counts['RK4_integrator_completed']==3584
accounting=read(E/'evidence/accounting-review.json')
assert sha(ledger)==accounting['WAL_sha256'] and not accounting['pending']
for k,v in counts.items():
    if k.endswith('_attempted'): assert v==counts[k[:-10]+'_completed'], k
checks.append('7352 WAL records paired; 3584 completed/attempted integrations; all native counters paired')

macros=[];npz_arrays=0
for i in range(20):
    base=E/'evidence'/('macro-%02d'%i);raw=base/'raw'; result=read(base/'macro-result.json')
    events=[read(raw/n) for n in read(raw/'evidence-inventory.json')['events']]
    trials=[e['row'] for e in events if e['kind']=='trial_complete']
    decisions=[e['row'] for e in events if e['kind']=='decision']
    # JSON erases dtype; pinned trial construction supplies float32 native arrays.
    for t in trials:
        for world in t['worlds']:
            for key in ('qpos','qvel'):
                world['endpoint'][key]=np.asarray(world['endpoint'][key],dtype=np.float32)
    assert [x['resolution'] for x in trials] in ([32,64,64],[32,64,128,128])
    for a,b,d in zip(trials,trials[1:-1],decisions):
        computed=assess_cohort(a['worlds'],b['worlds'],(0,),assess)
        assert computed=={k:v for k,v in d.items() if k not in ('resolution','ticks')}
    for t in trials: assert sha(t['endpoint_record'].encode())==t['endpoint_sha256']
    a,b=trials[-2:]
    assert a['endpoint_record']==b['endpoint_record'] and a['stage_sha256']==b['stage_sha256']
    assert a['numeric_Data_hashes']==b['numeric_Data_hashes']
    assert result['accepted_stage_sha256']==b['stage_sha256']
    commit=[e for e in events if e['kind']=='commit_prepared'][0]
    assert result['committed_endpoint_sha256']==commit['row']['endpoint_sha256']
    assert result['replayed'] and result['mandatory_exact_replay'] and result['decision']['accepted']
    assert result['public_ticks']==[i%10+1]
    for e in events:
        if e['kind']=='rollback':
            assert all(e['row'][k] for k in ('bindings_exact','data_exact','model_exact','owners_exact','rng_exact'))
    micro=np.load(raw/'microsteps.npz',allow_pickle=False)
    assert len(micro['qpos'])==result['actual_integrations']==sum(t['resolution'] for t in trials)
    for t in trials:
        mask=micro['trial']==t['trial']; assert mask.sum()==t['resolution']
        for k in ('qpos','qvel','time'): assert np.isfinite(micro[k][mask]).all()
    # Replay raw trajectories, not just producer pass booleans.
    for k in ('qpos','qvel','time','stage_sha256'):
        assert np.array_equal(micro[k][micro['trial']==a['trial']],micro[k][micro['trial']==b['trial']])
    for path in raw.glob('*-snapshot.json'):
        meta=read(path); prefix=path.name[:-len('-snapshot.json')]
        for tag,key in [('data','Data'),('owners','owners')]:
            with np.load(raw/(prefix+'-'+tag+'.npz'),allow_pickle=False) as data:
                assert set(data.files)==set(meta[key])
                for n in data.files:
                    v=data[n]; assert meta[key][n]==dict(bytes=v.nbytes,shape=list(v.shape),dtype=str(v.dtype),sha256=sha(v.tobytes()))
                    npz_arrays+=1
    macros.append(dict(index=i,resolution=result['accepted_resolution'],integrations=result['actual_integrations'],elapsed_s=result['elapsed_s']))
checks.append('20 macros: original decisions recomputed, full endpoint records and replay raw microtrajectories exact')
checks.append('All 40 macro boundary Data/owner NPZ archives match per-array metadata')

def metadata(a):
    return dict(bytes=a.nbytes,shape=list(a.shape),dtype=str(a.dtype),sha256=sha(a.tobytes()))
def key(s):
    return '/d'+json.dumps(['str',s],separators=(',',':')).encode().hex()
def same(a,b):
    assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def verify_npz(base):
    global npz_arrays
    m=read(base.with_suffix('.json'))
    with np.load(base.with_suffix('.npz'),allow_pickle=False) as z:
        assert set(z.files)==set(m['arrays'])
        data={n:z[n] for n in z.files}
        for n,a in data.items():
            assert m['arrays'][n]==metadata(a)
            npz_arrays+=1
    return m,data

ev=E/'evidence'
cache=read(ev/'tuple-cache-preflight.json')
assert cache['status']=='PASS' and cache['changed_values_and_rebinding_tested'] and cache['original_dict_and_key_restored']
assert cache['native_integrations']==cache['observation_computes']==cache['reward_computes']==0
cache_root='env._double_balance_tray_site_ids'
boundaries={}
for label in ('initial-reset','step-0-before','step-0-after','step-1-before','step-1-after','final-reset'):
    row=read(ev/(label+'.json'))
    dm,da=verify_npz(ev/(label+'-data'))
    om,oa=verify_npz(ev/(label+'-owners'))
    assert dm['arrays']==row['data_hashes']
    assert om['topology'][cache_root]==cache['topology'][cache_root]
    assert all(om['scalars'][n]==v for n,v in cache['scalars'].items())
    assert row['owner_roster']['arbitrary_mid_RL_step_restart'] is False
    boundaries[label]=(row,da,om,oa)
checks.append('Six public Data/owner boundaries hashed; original tuple-key cache owner, typed key and site ID preserved')

normal_methods=collections.Counter(dict(Action_process_action=1,Action_apply=10,Metrics_substep=10,
    Termination_compute=1,Reward_compute=1,Metrics_compute=1,Command_compute=1,Event_apply=2,
    Observation_compute=1,Recorder_post_step=1))
step_rows=[]
for i,case in enumerate(('auto-timeout','manual-failure')):
    step=read(ev/('formal-step-%d.json'%i));step_rows.append(step)
    assert step['case']==case and step['observation_history_checked'] and not step['numerical_thresholds_changed']
    trace=step['trace'];assert all(t['completed'] for t in trace)
    expected=normal_methods.copy()
    if i==0:
        expected.update(dict(Recorder_pre_reset=1,Curriculum_compute=1,Event_apply=1,Recorder_post_reset=1))
        expected.update({n+'_reset':1 for n in ('Observation','Action','Reward','Metrics','Curriculum','Command','Event','Termination')})
    assert collections.Counter(t['method'] for t in trace)==expected
    assert step['actual_calls']['integrations_completed']==step['actual_calls']['integrations_attempted']==1792
    assert step['actual_calls']['BAM_compute_completed']==11-i
    assert step['terminated']==[bool(i)] and step['timed_out']==[not bool(i)]
    assert step['terminal_fault_is_test_injection']==bool(i)
    assert step['state']==boundaries['step-%d-after'%i][0]['state']
    assert step['state']['events']==[[i+1]]*4
    assert all(v==[i+1] for v in step['state']['history'].values())
    raw=ev/('macro-%02d'%(i*10+9))/'raw'
    snapshots=sorted(raw.glob('*-snapshot.json'))
    endpoint=read(snapshots[-1])['Data']
    for t in trace:
        if t['method'] in ('Termination_compute','Reward_compute'):
            assert all(endpoint['Data.'+k]['sha256']==v for k,v in t['accepted_derived_endpoint_hashes'].items())
    _,out=verify_npz(ev/('formal-step-%d-outputs'%i))
    assert {k:v.shape for k,v in out.items()}==dict(actor=(1,61),critic=(1,85),reward=(1,))
    assert all(np.isfinite(v).all() for v in out.values())
    assert out['reward'].tolist()==step['reward']
    owners=boundaries['step-%d-after'%i][3]
    for name in ('actor','critic'):same(out[name],owners['env.obs_buf'+key(name)])
checks.append('Both real manager traces, 61D/85D finite outputs, event/history counts, reward/termination at accepted endpoints')

pre=boundaries['step-0-before'];auto=boundaries['step-0-after'];manual=boundaries['step-1-after'];final=boundaries['final-reset']
assert pre[0]['state']['episode']==[499]
assert auto[0]['state']['episode']==[0] and auto[0]['state']['pending']==[False] and auto[0]['state']['tick']==0
assert manual[0]['state']['episode']==[1] and manual[0]['state']['pending']==[True] and manual[0]['state']['tick']==10
assert final[0]['state']['episode']==[0] and final[0]['state']['pending']==[False] and final[0]['state']['tick']==0
assert final[0]['state']['sim_counter']==20 and final[0]['state']['control_counter']==2
assert final[0]['state']['events']==[[2]]*4 and final[0]['state']['bam_pushes']==[1]
assert all(v==[1] for v in final[0]['state']['history'].values())
assert counts['Action_process_action_completed']==2 and counts['Action_apply_completed']==20
assert counts['Observation_compute_completed']==4 and counts['Observation_noise_completed']==16
assert counts['BAM_compute_completed']==counts['BAM_apply_delay_completed']==23
def rec(i,k):return 'Recorder:0.entries/s%d'%i+key(k)
assert [auto[2]['scalars'][rec(i,'hook')] for i in range(4)]==['post_reset','pre_reset','post_reset','post_step']
for name in ('actor','critic'):same(auto[3][rec(1,'observations')+key(name)],pre[3]['env.obs_buf'+key(name)])
assert np.any(auto[3][rec(1,'action')]) and not np.any(auto[3][rec(2,'action')])
for i in (2,3):
    same(auto[3][rec(i,'qpos')],auto[1]['Data.qpos'])
    same(auto[3][rec(i,'time')],auto[1]['Data.time'])
for k in ('qpos','time'):
    endpoint_path=sorted((ev/'macro-09/raw').glob('*-snapshot.json'))[-1]
    prefix=endpoint_path.name[:-len('-snapshot.json')]
    with np.load(endpoint_path.parent/(prefix+'-data.npz'),allow_pickle=False) as z:same(auto[3][rec(1,k)],z['Data.'+k])
assert manual[2]['scalars'][rec(4,'hook')]=='post_step'
for k in ('qpos','time'):same(manual[3][rec(4,k)],manual[1]['Data.'+k])
same(manual[3][rec(4,'action')],manual[3]['actionManager._action'])
same(manual[3][rec(4,'action')],-auto[3][rec(1,'action')])
assert not np.any(final[3]['actionManager._action'])
checks.append('Raw recorder terminal input/old observation and reset output matched; manual pending refusal evidenced by pinned guard and no extra counted action; final explicit reset clears pending')

from contact_order_contract import FIELDS,permutation,evidence,verify_evidence
plan=read(P/'plan.json')
contacts=sorted(ev.glob('contact-sample-*.json'));assert len(contacts)==21
for path in contacts:
    row=read(path)
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
        phases={phase:{n:z[phase+'.'+n] for n in FIELDS} for phase in ('before_metadata','before_sort','after_sort')}
    for phase,fields in phases.items():
        assert all(sha(a.tobytes())==row[phase][n] for n,a in fields.items())
    before=phases['before_sort'];after=phases['after_sort'];n=row['proof']['nacon']
    proposal,order=permutation(before,n,plan['contact_layout'])
    assert evidence(before,after,order,n,row['serial'])==row['proof'] and verify_evidence(row['proof'])
    for name in FIELDS:
        same(proposal[name],after[name])
        m=phases['before_metadata'][name]
        if name in ('geomcollisionid','type'):
            # Rigid active metadata is explicitly populated before canonical ordering.
            same(m[n:],before[name][n:])
        else:same(m,before[name])
checks.append('21 raw full-Contact samples independently permuted; active rows and inactive tails exact')

from solver_order_candidate import expected_ids,verify_sorted
samples=sorted((ev/'solver-order').glob('sample-*.json'));assert len(samples)==21
for path in samples:
    row=read(path)
    with np.load(path.with_suffix('.npz'),allow_pickle=False) as z:
        for k,col in [('old','before_state'),('new','after_state'),('before','before'),('after','after')]:assert sha(z[col].tobytes())==row[k]
        assert int(z['count'][0])==row['count']
        expected=expected_ids(z['before_state'],z['after_state'],z['before'],row['count'],row['nefc'],row['done'],1)
        assert expected==row['expected'] and verify_sorted(z['before'],z['after'],row['count'],expected)
checks.append('21 raw changed-constraint samples match quadratic transitions, sorted IDs and untouched tails; live all-call proofs retained as digests')

resources=read(ev/'resource-lifetime.json')
assert resources['closed'] and len(resources['targets'])==len(resources['releases'])==42
assert all(r['weakref_dead'] and r['released'] for r in resources['targets'])
assert collections.Counter((r['api'],r['handle']) for r in resources['targets'])==collections.Counter((r['api'],r['args'][1 if r['api']=='wp_texture2d_destroy_device' else 0]) for r in resources['releases'])
assert all(r['returned_normally'] and not r['capturing'] for r in resources['releases'])
graph=read(ev/'live-contact-graph.json')['summary']
assert graph['graph_closed'] and graph['calls']==counts['GPU_forward_completed']==14425
assert counts['ContactGraph_capture_begin_completed']==counts['ContactGraph_graph_destroy_completed']==counts['ContactGraph_exec_destroy_completed']==1
assert report['errors']==[] and report['remaining_matrix_completed'] and not report['full_matrix_completed']
assert report['macro_completed']==20 and sum(m['integrations'] for m in macros)==3584
assert report['cumulative_completed']==674456 and report['cumulative_attempted']==674457
assert report['PPO']==report['cloud_training']==0 and report['resource_close']['all_released']
assert read(ev/'configuration-preflight.json')['passed']==21
assert report['checkpoint']['direct_resume']=='BLOCKED_TASK_AND_TIMING_DIFFER'
assert report['checkpoint']['observed_sha256']=='86d55c3703c18fcf4817db49c6e4dcf839b3f5195d68ae2c559db7e222bded97'
assert report['noise_execution']['default_RNG_unchanged'] and report['checkpoint']['torch_load_calls_this_batch']==0
v2=read(P/'reused-v2/audit.json');assert v2==read(cli.v2_audit)
assert v2['canonical_after']==dict(completed=670872,attempted=670873,historical_failed_native_attempts=1)
assert v2['normal_step_completed_scoped'] and not v2['normal_poststep_full_owner_boundary_available']
assert report['reset_checks_reused']==v2['reset_checks']
checks.append('42 native resources matched actual releases; sorting graph destroyed; v1 checkpoint and v2 normal/reset proofs reused unchanged')

audit=dict(status='V3_REMAINING_MATRIX_INDEPENDENT_AUDIT_PASS',commit='4fc857853b988ebac7d87742925ffcb919748003',
    tree='bb2ab4db70f99c5a962a6c64969b82cc2aa833cc',archive_sha256=sha(archive.read_bytes()),archive_bytes=archive.stat().st_size,
    members=1033,checks=checks,per_array_hashes_checked=npz_arrays,macro_results=macros,
    canonical_before=dict(completed=670872,attempted=670873),added=dict(completed=3584,attempted=3584),
    canonical_after=dict(completed=674456,attempted=674457,historical_failed_native_attempts=1),
    elapsed_s=report['elapsed_s'],WAL_seconds=accounting['WAL_seconds'],WAL_wall_fraction=accounting['WAL_seconds']/report['elapsed_s'],
    cross_run_speedup_claimed=False,normal_case_reexecuted=False,contiguous_three_case_run=False,
    v2_missing_owner_boundary_recovered=False,inner_contact_solver_raw_scope='21 samples each plus live all-call byte checks/digests; G10 exhaustive raw evidence reused',
    pending_refusal_scope='Pinned worker guard succeeded; two counted action processes; no standalone post-refusal full boundary saved',
    runtime_default_RNG_scope='Pinned worker equality guards passed; only initial digest independently archived',
    G11_review='PASS_FOR_COMBINED_SCOPED_CLOSURE',producer_report_preserved=True,PPO=0,cloud_training=0,auditor_native_calls=0)
(R/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k not in ('macro_results',)},ensure_ascii=False,indent=2))
