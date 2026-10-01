"""Independent NumPy review of the returned, immutable Stage 10 batch A evidence."""
from pathlib import Path
import hashlib
import itertools
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'stage10-batch-a-review'
OUT = Path(__file__).resolve().parent

def read(path):
    return json.loads(path.read_text())

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    plan = read(DATA / 'plan.json')
    comparison = read(DATA / 'comparison.json')
    names = [j['name'] for j in plan['jobs']]
    arrays, reports, initials, early = {}, {}, {}, {}
    jobs, boundaries = [], []
    for name in names:
        r = read(DATA / name / 'evaluation.json')
        reports[name] = r
        assert r['checkpoint_sha256'] == plan['checkpoint_sha256']
        assert r['git_head'] == plan['source_head'] and r['initial_states']['sha256'] == plan['dataset_sha256']
        assert r['status'] == 'PASS' and r['protocol'] == 'dev' and r['num_envs'] == 128
        assert r['assistance'] == 0 and r['horizon_seconds'] == 10
        assert [e['env_id'] for e in r['episodes']] == list(range(128))
        for file in ('evaluation.json', 'head-diagnostic.json', 'head-initial.npz', 'posture-summary.json', 'params/env.yaml'):
            receipt = {x['path']: x for x in read(DATA / name / 'raw-local-files.json')['files']}
            assert sha(DATA / name / file) == receipt[file]['sha256']
        with np.load(DATA / name / 'all-env-metrics.npz', allow_pickle=False) as z:
            values = z['values']; columns = list(z['columns']); active = z['active_first_episode']
            times = z['time_s']
        assert values.shape == (500, 128, 21) and active.all()
        assert np.isfinite(values).all() and np.array_equal(times, np.arange(1, 501)*.02)
        col = {k: i for i, k in enumerate(columns)}
        get = lambda k: values[:, :, col[k]]
        gates = values[:, :, 9:18].astype(bool)
        stable = gates.all(axis=-1)
        assert np.array_equal(stable, get('cached_stable').astype(bool))
        assert np.array_equal(get('lower_ball_speed_m_s') <= np.float32(.15), get('pass_lower_ball_speed').astype(bool))
        assert (get('hold') == 0).all()
        timer = np.zeros(128, dtype=np.float32)
        count = np.zeros(128, dtype=np.int64)
        longest = count.copy()
        timer_max_error = 0.
        for step in range(500):
            timer = np.where(stable[step], timer + np.float32(.02), np.float32(0.))
            count = np.where(stable[step], count + 1, 0)
            longest = np.maximum(longest, count)
            timer_max_error = max(timer_max_error, float(abs(timer-get('official_timer_s')[step]).max()))
            assert np.array_equal(timer >= np.float32(5.), get('cached_success')[step].astype(bool))
        assert timer_max_error == 0
        expected = get('cached_success')[-1].astype(bool)
        labels = np.array([e['success'] for e in r['episodes']])
        assert np.array_equal(expected, labels)
        assert int(labels.sum()) == r['successes']
        failures = []
        for env, e in enumerate(r['episodes']):
            assert e['steps'] == 500 and not e['terminated']
            assert abs(e['terminal_stable_seconds'] - count[env]*.02) < 1e-12
            assert abs(e['longest_stable_seconds'] - longest[env]*.02) < 1e-12
            assert e['official_timer_seconds'] == float(timer[env])
            assert abs(e['lower_speed_violation_fraction'] - (~gates[:,env,7]).mean()) < 1e-7
            if not labels[env]:
                bad = np.flatnonzero(~stable[:,env])
                last = int(bad[-1]) if len(bad) else None
                failures.append({'env_id':env, 'terminal_samples':int(count[env]),
                                 'last_unstable_time_s':float(times[last]) if last is not None else None,
                                 'last_unstable_gates':[str(columns[9+i])[5:] for i in np.flatnonzero(~gates[last,env])] if last is not None else [],
                                 'last_unstable_lower_speed_m_s':float(get('lower_ball_speed_m_s')[last,env]) if last is not None else None})
            if labels[env] != (count[env] >= 250):
                boundaries.append({'job':name,'env_id':env, 'official_success':bool(labels[env]),
                                   'count_diagnostic_success':bool(count[env]>=250),
                                   'terminal_samples':int(count[env]),'official_timer_s':float(timer[env]),
                                   'preceding_unstable_time_s':float(times[499-count[env]])})
        arrays[name] = (values, stable, labels)
        with np.load(DATA/name/'head-initial.npz', allow_pickle=False) as z:
            initials[name] = {k:z[k] for k in z.files}
        with np.load(DATA/name/'early-all-envs.npz', allow_pickle=False) as z:
            early[name] = {k:z[k] for k in z.files}
        head = read(DATA/name/'head-diagnostic.json')
        head99 = head['episodes'][99]
        jobs.append({'name':name,'successes':int(labels.sum()),'episodes':128,
                     'percent':100*float(labels.mean()),'physical_terminations':0,
                     'failure_last_gates':failures,
                     'violating_sample_counts':dict(zip([str(x)[5:] for x in columns[9:18]], (~gates).sum(axis=(0,1)).tolist())),
                     'terminal_5s_violating_sample_counts':dict(zip([str(x)[5:] for x in columns[9:18]], (~gates[250:]).sum(axis=(0,1)).tolist())),
                     'float32_timer_reproduction_max_error':timer_max_error,
                     'env99':r['episodes'][99],
                     'env99_head_tail':head99['relative_head_5_to_10s'],
                     'env99_head_roll':head99['joints']['head_roll']})
    label_stack = np.stack([arrays[n][2] for n in names])
    label_counts = label_stack.sum(axis=0)
    matrix = np.count_nonzero(label_stack[:,None,:] != label_stack[None,:,:], axis=2)
    assert (matrix == matrix.T).all() and (matrix.diagonal()==0).all()
    pairs=[]
    for ai,bi in itertools.combinations(range(6),2):
        a,b=names[ai],names[bi]
        same_initial = all(np.array_equal(initials[a][k],initials[b][k]) for k in initials[a])
        assert same_initial
        assert [e['initial_state_id'] for e in reports[a]['episodes']] == [e['initial_state_id'] for e in reports[b]['episodes']]
        old=next(p for p in comparison['pairs'] if p['left']==a and p['right']==b)
        assert matrix[ai,bi] == old['label_disagreements']
        first={}
        for field in old['first_recorded_differences']:
            av,bv=early[a][field],early[b][field]
            diff=np.abs(av.astype(np.float64)-bv.astype(np.float64))
            by_env=diff.reshape(10,128,-1).max(axis=2)
            rows=np.flatnonzero((by_env>0).any(axis=1))
            if len(rows):
                step=int(rows[0]); ids=np.flatnonzero(by_env[step]>0).tolist()
                result={'sample':step,'time_s':float(early[a]['time_s'][step]),'env_ids':ids,
                        'max_abs_at_first_sample':float(by_env[step].max())}
                previous=old['first_recorded_differences'][field]
                assert result['time_s']==previous['time_s'] and ids==previous['env_ids']
                assert result['max_abs_at_first_sample']==previous['max_abs_at_first_sample']
                first[field]=result
            else: first[field]=None
        pairs.append({'left':a,'right':b,'same_render_condition':ai%2==bi%2,
                      'label_disagreements':int(matrix[ai,bi]),'initial_recorded_fields_equal':same_initial,
                      'first_action_equal':bool(np.array_equal(early[a]['policy_action'][0],early[b]['policy_action'][0])),
                      'first_recorded_differences':first})
    out={'status':'INDEPENDENT_RETURNED_EVIDENCE_REVIEW_PASS','zip_sha256':sha(ROOT/'upload/review-20261001T061125Z-75b299-part01.zip'),
         'manifest_files_verified':117,'jobs':jobs,'pair_order':names,'label_disagreement_matrix':matrix.tolist(),
         'all_six_success_env_ids':np.flatnonzero(label_counts==6).tolist(),
         'all_six_failure_env_ids':np.flatnonzero(label_counts==0).tolist(),
         'mixed_label_env_ids':np.flatnonzero((label_counts>0)&(label_counts<6)).tolist(),
         'success_frequency_histogram':{str(i):int((label_counts==i).sum()) for i in range(7)},
         'boundary_cases':boundaries,'pairs':pairs,'new_ppo_updates':0,'stage10_complete':False,
         'limits':['Same 128 development initial states repeated six times, not 768 independent trials.',
                   'All-env detailed state returned only for first 0.2s; complete metrics for 10s.',
                   'Initial head fields match; full integration, solver, observation, RNN and RNG states are unrecorded.',
                   'Differences at 0.02s are first telemetry samples, not the exact first 0.002s divergence.',
                   'Backend settings are observed, not a causal test. No pooled success rate or old-result replacement.']}
    (OUT/'independent-review.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print('INDEPENDENT REVIEW PASS')
    print('COUNTS',out['success_frequency_histogram'])
    print('MATRIX',matrix.tolist())
    print('BOUNDARIES',boundaries)
    for j in jobs:
        print(j['name'],j['successes'],'FAILURE_LAST_GATES',dict((g,sum(g in x['last_unstable_gates'] for x in j['failure_last_gates'])) for g in ['lower_ball_speed','top_speed','top_height','top_center']))
        print('LAST5_GATE_COUNTS',j['terminal_5s_violating_sample_counts'])
        print('ENV99',j['env99']['success'],j['env99']['official_timer_seconds'],j['env99']['terminal_stable_seconds'])

if __name__=='__main__':main()
