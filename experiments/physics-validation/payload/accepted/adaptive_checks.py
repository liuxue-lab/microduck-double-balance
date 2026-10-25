"""Original bit-exact stage comparison on the final accepted trial; no grid or byte relaxation."""
from pathlib import Path
import hashlib,json
import numpy as np
from lifecycle_support import require
from transaction_contract import exact
from transaction_contract import compare_records
from convergence_contract import compare
from ground_scalar_contract import assess

def reference_agreement(adaptive,reference,adaptive_post,reference_post):
    require(adaptive['resolution'] in (64,128) and reference['resolution']==128,
        'Reference agreement needs an actually accepted adaptive tier and fixed128')
    worlds=[]
    for w,(a,b) in enumerate(zip(adaptive['worlds'],reference['worlds'])):
        def point(value,post):
            return dict(tick=1,native_time_s=float(post['data']['Data.time'][w]),
                qpos=value['endpoint']['qpos'],qvel=value['endpoint']['qvel'])
        local=assess(a,b)
        physical=compare([point(a,adaptive_post)],[point(b,reference_post)],
            a['integral'],b['integral'],expected_macros=1,max_resolution=128)
        worlds.append(dict(world=w,local_state_event=local,state_contact=physical,
            passed=local['accepted'] and physical['passed']))
    require(len(worlds)==1,'Reference world roster differs')
    return dict(passed=all(row['passed'] for row in worlds),worlds=worlds,
        criteria_unchanged=True,scope='one post-reset2ms macro, one owned world, all seven contact pairs',
        original_G08_200ms_acceptance_claimed=False)

def same_resolution_exact(adaptive,reference,n):
    require(n in (64,128),'Undeclared adaptive tier')
    if n==64:return dict(applicable=False,reason='64vs128 uses declared numerical agreement; no exact-state claim')
    return dict(applicable=True,comparison=compare_records(adaptive,reference,False,True))

def strict_results_pass(cases):
    from adaptive_driver import CASES
    if len(cases)!=len(CASES):return False
    for row,(name,mask,mode) in zip(cases,CASES):
        if row.get('case')!=name or row.get('mode')!=mode:return False
        expected=set()
        if mask is not None:expected|={'check_Data_reset','check_Manager_reset','check_Scene_reset'}
        if name in ('reset-0','reset-1'):
            expected|={'check_unselected_manager_RNG','check_unselected_scene','check_unselected_physical_stages'}
        if name=='empty':expected|={'check_exact_replay','check_trajectory_replay','check_Model_replay_exact'}
        if mode=='reference':expected|={'check_reference_input','check_reference_agreement','check_same_resolution_exact'}
        if {k for k in row if k.startswith('check_')}!=expected:return False
        if not all(row[k].get('passed') is True for k in expected):return False
        if mode=='adaptive' and not (row.get('physical',{}).get('mandatory_exact_replay') and row['physical'].get('replayed')):return False
        if mode=='reference' and row.get('reference_agreement',{}).get('passed') is not True:return False
    return True

def microsteps(path):
    events=[]
    for p in sorted(Path(path).glob('record-*.json')):
        if not p.name.endswith('-snapshot.json'):events.append(json.loads(p.read_text()))
    trials=[e['row'] for e in events if e['kind']=='trial_complete']
    require(trials and trials[-1]['role'] in ('replay','fixed_continuation'),'Last trajectory must be accepted replay or fixed reference')
    trial=trials[-1];n=trial['resolution'];require(n in (64,128),'Undeclared accepted trajectory resolution')
    rows=[e['row'] for e in events if e['kind']=='microstep' and e['row']['trial']==trial['trial']]
    require(len(rows)==n and [r['micro_index'] for r in rows]==list(range(n)),'Incomplete accepted trajectory')
    require(all(r['resolution']==n and len(r['stages'])==4 for r in rows),'Accepted stage roster differs')
    return rows

def compare_accepted_trajectories(control,actual,mask,*,replay=False):
    left,right=microsteps(control),microsteps(actual)
    require(len(left)==len(right),'Different accepted grids; strict stage isolation unavailable, no approximate substitution')
    n=len(left)
    other=[w for w in range(1) if w not in mask];require(other,'At least one unselected world required')
    remaps=0
    def canonical(c):return json.dumps({k:v for k,v in c.items() if k!='packed_id'},sort_keys=True,separators=(',',':')).encode()
    for a,b in zip(left,right):
        require(a['micro_dt_s']==b['micro_dt_s']==float(np.float32(.002/n)),'Shared physical dt differs')
        if replay:require(a['stage_sha256']==b['stage_sha256'],'Exact replay full-stage digest differs')
        for key in ('qpos','qvel','time'):
            aa=np.asarray(a['post_integrator'][key],np.float32);bb=np.asarray(b['post_integrator'][key],np.float32)
            require(exact(aa[other],bb[other]),'Unselected post-integrator '+key+' differs')
        for sa,sb in zip(a['stages'],b['stages']):
            if replay:require(sa['full_Data_hashes']==sb['full_Data_hashes'],'Replay all Data fingerprints differ')
            for key in ('qpos','qvel','qacc'):
                aa=np.asarray(sa[key],np.float32);bb=np.asarray(sb[key],np.float32)
                require(exact(aa[other],bb[other]),'Unselected physical stage '+key+' differs')
            for w in other:
                ca,cb=sa['contacts'][w],sb['contacts'][w]
                ma,mb={canonical(c):c['packed_id'] for c in ca},{canonical(c):c['packed_id'] for c in cb}
                require(len(ma)==len(ca) and len(mb)==len(cb) and ma.keys()==mb.keys(),
                        'Unselected physical Contact/solver force differs')
                remaps+=sum(ma[k]!=mb[k] for k in ma)
    return dict(worlds=other,microsteps=n,stages=4*n,primary_state_bit_exact=True,
        observed_contact_geometry_and_forces_exact=True,packed_ID_remaps=remaps,
        full_stage_Data_fingerprints_exact=replay,post_reset_adaptive_closure_claimed=False)
