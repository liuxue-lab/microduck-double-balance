"""Failure injection for GPU telemetry and hash-guarded batch continuation."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace as NS

import pytest
import torch

from mjlab_microduck import double_balance_stage08 as worker
from mjlab_microduck import double_balance_stage08_state as state_module
from test_stage08_tools import script
from test_stage08_runtime import runtime
from mjlab_microduck.double_balance_stage08_plan import SOURCE_SHA256, TRAIN_SEEDS
from mjlab_microduck.double_balance_stage08_state import metadata, restore


GPU = {'total_mib':32607, 'used_mib':22837, 'uuid':'same-gpu'}


def exercise_sampler(tmp_path, monkeypatch, events):
    queue = [dict(GPU), *events, dict(GPU)]  # initial query, thread, final query
    calls = []
    def sample():
        value = queue.pop(0)
        calls.append(value)
        if len(queue) == 1:
            sampler.stop.set()
        if isinstance(value, Exception):
            raise value
        return value
    monkeypatch.setattr(worker, 'gpu_sample', sample)
    monkeypatch.setattr(worker.GPUSampler, 'interval_seconds', 0)
    sampler = worker.GPUSampler(tmp_path)
    sampler.thread.start()
    sampler.thread.join(timeout=2)
    assert not sampler.thread.is_alive()
    return sampler, calls


def timeout():
    return subprocess.TimeoutExpired(['nvidia-smi'], 5)


def test_transient_timeout_recovers_but_cannot_certify_capacity(tmp_path, monkeypatch):
    sampler, calls = exercise_sampler(tmp_path, monkeypatch, [dict(GPU), timeout(), dict(GPU)])
    result = sampler.close()
    assert result['samples'] == 3 and result['query_timeouts'] == 1
    assert result['sampling_status'] == 'RECOVERED_WITH_GAPS'
    assert not result['eligible_with_15_percent_headroom']
    rows = [json.loads(line) for line in sampler.path.read_text().splitlines()]
    assert [r['event'] for r in rows] == ['sample', 'query_timeout', 'sample', 'sample']
    assert rows[-1]['final'] is True and len(calls) == 5


def test_timeout_streak_resets_after_a_success(tmp_path, monkeypatch):
    sampler, _ = exercise_sampler(tmp_path, monkeypatch,
        [timeout(), timeout(), dict(GPU), timeout(), dict(GPU)])
    result = sampler.close()
    assert result['query_timeouts'] == 3 and result['max_consecutive_timeouts'] == 2


@pytest.mark.parametrize('events,message', [
    ([dict(GPU), timeout(), timeout(), timeout()], 'three consecutive'),
    ([dict(GPU), {**GPU, 'uuid':'different-gpu'}], 'allocation changed'),
    ([dict(GPU), {**GPU, 'total_mib':80000}], 'allocation changed'),
    ([dict(GPU), subprocess.CalledProcessError(9, ['nvidia-smi'])], 'non-zero exit'),
])
def test_sustained_or_non_timeout_failure_stays_fatal(tmp_path, monkeypatch, events, message):
    sampler, _ = exercise_sampler(tmp_path, monkeypatch, events)
    with pytest.raises(ValueError, match=message):
        sampler.close()


def test_final_query_must_succeed_and_keep_identity(tmp_path, monkeypatch):
    sampler, _ = exercise_sampler(tmp_path, monkeypatch, [dict(GPU)])
    monkeypatch.setattr(worker, 'gpu_sample', lambda: {**GPU, 'uuid':'changed-at-exit'})
    with pytest.raises(ValueError, match='allocation changed'):
        sampler.close()


def test_clean_sampler_certifies_sampled_headroom(tmp_path, monkeypatch):
    sampler, _ = exercise_sampler(tmp_path, monkeypatch, [dict(GPU)])
    result = sampler.close()
    assert result['sampling_status'] == 'PASS' and result['eligible_with_15_percent_headroom']


def test_resume_refreshes_only_capacity_after_live_validation(tmp_path, monkeypatch):
    batch = script('run_stage08_batch')
    old_file = tmp_path / 'old.json'; old_file.write_text('old measured report')
    new_file = tmp_path / 'new.json'; new_file.write_text('new measured report')
    contract = {'capacity_sha256':batch.checksum(old_file), 'gpu':'5090', 'campaign_id':'kept',
                'datasets':'unchanged', 'source_sha256':SOURCE_SHA256}
    manifest = {'contract':contract, 'status':'NEEDS_REVIEW', 'jobs':[{'segments':[{'status':'FAIL'}]}]}
    wanted = {**contract, 'capacity_sha256':batch.checksum(new_file)}
    calls = []
    monkeypatch.setattr(state_module, 'environment_preflight', lambda gpu: 'new-head')
    monkeypatch.setattr(worker, 'measured_capacity', lambda *args: calls.append(args))
    result = batch.resume_manifest(deepcopy(manifest), wanted, capacity_summary=new_file,
                                   previous_capacity_summary=old_file)
    assert calls == [(new_file, '5090', 'new-head')]
    assert result['status'] == 'RUNNING' and result['jobs'] == manifest['jobs']
    assert result['resume_history'][0]['previous_status'] == 'NEEDS_REVIEW'
    assert result['capacity_renewals'][0]['previous_contract'] == contract
    for change in ({'campaign_id':'reset'}, {'datasets':'new'}, {'source_sha256':'new'}):
        with pytest.raises(ValueError, match='beyond capacity'):
            batch.resume_manifest(deepcopy(manifest), {**wanted, **change}, capacity_summary=new_file,
                                  previous_capacity_summary=old_file)
    with pytest.raises(ValueError, match='original capacity'):
        batch.resume_manifest(deepcopy(manifest), wanted, capacity_summary=new_file)
    def reject(*_):
        raise ValueError('capacity wrong source head')
    monkeypatch.setattr(worker, 'measured_capacity', reject)
    original = deepcopy(manifest)
    with pytest.raises(ValueError, match='wrong source head'):
        batch.resume_manifest(manifest, wanted, capacity_summary=new_file, previous_capacity_summary=old_file)
    assert manifest == original


def test_same_contract_resume_clears_stale_needs_review():
    batch = script('run_stage08_batch')
    manifest = {'contract':{'same':'contract'}, 'status':'NEEDS_REVIEW', 'jobs':[]}
    result = batch.resume_manifest(manifest, manifest['contract'], capacity_summary=Path('unused'))
    assert result['status'] == 'RUNNING' and 'capacity_renewals' not in result


def a_evidence(root):
    recovery = script('recover_stage08_screening')
    batch = script('run_stage08_batch')
    runner, env, initial = runtime('A')
    restore(runner, env, initial, profile='A', seed=TRAIN_SEEDS[0], campaign_id='kept', resume=False)
    runner.current_learning_iteration = 1499
    env.unwrapped.common_step_counter = 36000; env.unwrapped._sim_step_counter = 360000
    runner.alg.learning_rate = 1.5e-5
    for group in runner.alg.optimizer.param_groups: group['lr'] = 1.5e-5
    for entry in runner.alg.optimizer.state.values(): entry['step'].fill_(30000)
    state = metadata(runner, env.unwrapped, 500, profile='A', seed=TRAIN_SEEDS[0],
                     campaign_id='kept', head=recovery.BASE_HEAD, purpose='formal')
    checkpoint = deepcopy(runner.alg.save())
    checkpoint.update(iter=1499, infos={'stage08':state, 'env_state':{'common_step_counter':36000}})
    directory = root / f'screening/A-{TRAIN_SEEDS[0]}/segment-001'
    (directory / 'checkpoints').mkdir(parents=True)
    def write(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
    for count in (0, 100, 200, 250, 300, 400, 500):
        path = directory / f'checkpoints/update_{count:06d}.pt'
        # Only the endpoint is loaded; the others exercise exact byte receipts.
        if count == 500: torch.save(checkpoint, path)
        else: path.write_bytes(f'opaque earlier checkpoint {count}'.encode())
        write(path.with_suffix('.json'), {'sha256':batch.checksum(path)})
        if count in (0, 250, 500):
            for protocol in ('nominal', 'dev'):
                write(directory / f'evaluations/update_{count:06d}/{protocol}/evaluation.json',
                      {'status':'PASS', 'protocol':protocol, 'checkpoint_sha256':batch.checksum(path)})
    old_capacity = root / 'old-capacity.json'; write(old_capacity, {'status':'PASS'})
    report = {'status':'FAIL', 'experiment_completed_updates':500, 'git_head':recovery.BASE_HEAD,
              'sampler_error':'nvidia-smi timed out after 5 seconds',
              'measured_capacity':{'summary':str(old_capacity)},
              'latest_checkpoint':str(directory / 'checkpoints/update_000500.pt')}
    write(directory / 'training.json', report)
    write(directory / 'watchdog.json', {'exit_code':1, 'watchdog_reason':None,
                                       'child_exited':True, 'shutdown_performed':False})
    jobs = batch.jobs_for('screening', list('ABCDE'))
    jobs[0]['segments'].append({'directory':str(directory), 'status':'FAIL'})
    manifest = {'status':'NEEDS_REVIEW', 'jobs':jobs, 'contract':{
        'phase':'screening', 'gpu':'5090', 'campaign_id':'kept', 'source_sha256':SOURCE_SHA256,
        'datasets':str(root / 'initial-states'), 'capacity_sha256':batch.checksum(old_capacity)}}
    write(root / 'screening/batch.json', manifest)
    return recovery, directory


def test_completed_a_is_verified_without_changing_old_failure(tmp_path):
    recovery, directory = a_evidence(tmp_path)
    old = (directory / 'training.json').read_bytes()
    result = recovery.verify_completed_a(tmp_path, 'kept')
    assert result['completed_updates'] == 500 and result['learning_rate'] == 1.5e-5
    assert result['expected_adam_steps'] == 30000 and result['common_step_counter'] == 36000
    assert len(result['evaluations']) == 6 and (directory / 'training.json').read_bytes() == old
    endpoint = directory / 'checkpoints/update_000500.pt'
    endpoint.write_bytes(endpoint.read_bytes() + b'corrupted')
    with pytest.raises(ValueError, match='receipt SHA mismatch'):
        recovery.verify_completed_a(tmp_path, 'kept')


@pytest.mark.parametrize('damage', ('ppo_error', 'wrong_evaluation', 'watchdog_timeout'))
def test_recovery_rejects_other_failures_and_wrong_evaluation(tmp_path, damage):
    recovery, directory = a_evidence(tmp_path)
    if damage == 'ppo_error':
        path = directory / 'training.json'; key, value = 'error', 'RuntimeError: nonfinite'
    elif damage == 'wrong_evaluation':
        path = directory / 'evaluations/update_000500/dev/evaluation.json'; key, value = 'checkpoint_sha256', 'other'
    else:
        path = directory / 'watchdog.json'; key, value = 'watchdog_reason', 'budget_or_job_timeout'
    data = json.loads(path.read_text()); data[key] = value; path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        recovery.verify_completed_a(tmp_path, 'kept')


def test_completed_endpoint_worker_never_calls_learn(tmp_path, monkeypatch):
    import mjlab.envs
    import mjlab.rl
    import mjlab.tasks.registry
    import mjlab.utils.os
    _, directory = a_evidence(tmp_path)
    runner, env, _ = runtime('A')
    raw = env.unwrapped
    runner.logger = NS(writer=None)
    def forbidden(*args, **kwargs):
        raise AssertionError('Completed A must never enter PPO or env.step')
    runner.learn = forbidden; env.step = forbidden; env.close = lambda: None
    def load(path, **kwargs):
        data = torch.load(path, map_location='cpu', weights_only=False)
        runner.alg.actor.load_state_dict(data['actor_state_dict'])
        runner.alg.critic.load_state_dict(data['critic_state_dict'])
        runner.alg.optimizer.load_state_dict(data['optimizer_state_dict'])
        runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]['lr']
        runner.current_learning_iteration = data['iter']
        raw.common_step_counter = data['infos']['env_state']['common_step_counter']
    runner.load = load
    def save(path, infos):
        data = deepcopy(runner.alg.save())
        data.update(iter=runner.current_learning_iteration,
                    infos={**infos, 'env_state':{'common_step_counter':raw.common_step_counter}})
        torch.save(data, path)
    runner.save = save
    monkeypatch.setattr(state_module, 'ARTIFACTS', tmp_path)
    monkeypatch.setattr(state_module, 'environment_preflight', lambda _: 'new-head')
    monkeypatch.setattr(state_module, 'build_training_config', lambda *a: (raw.cfg, {'clip_actions':None}))
    ledger = NS(identity='kept', path=tmp_path / 'budget.json', snapshot=lambda: {'gpu':'5090'},
                admit=lambda *a: True)
    monkeypatch.setattr(worker, 'BudgetLedger', lambda _: ledger)
    monkeypatch.setattr(worker, 'measured_capacity', lambda *a: {'seconds_per_update':3.1})
    monkeypatch.setattr(worker, 'GPUSampler', lambda _: NS(thread=NS(start=lambda:None), close=lambda:dict(GPU)))
    monkeypatch.setattr(mjlab.envs, 'ManagerBasedRlEnv', lambda **kwargs: raw)
    monkeypatch.setattr(mjlab.rl, 'RslRlVecEnvWrapper', lambda *a, **k: env)
    monkeypatch.setattr(mjlab.tasks.registry, 'load_runner_cls', lambda _: lambda *a: runner)
    monkeypatch.setattr(mjlab.utils.os, 'dump_yaml', lambda *a:None)
    evaluations = []
    def evaluate(checkpoint, output, args, completed, budget):
        evaluations.append((completed, checkpoint))
        return {'dev':{'status':'PASS'}}
    monkeypatch.setattr(worker, 'run_development', evaluate)
    output = tmp_path / 'segment-002'
    args = NS(gpu='5090', ledger=ledger.path, resume=True, profile='A', seed=TRAIN_SEEDS[0],
              checkpoint=directory / 'checkpoints/update_000500.pt', target_updates=500,
              capacity_summary=tmp_path / 'capacity.json', decision=None,
              output=output, datasets=tmp_path / 'initial-states')
    report = worker.train_worker(args)
    assert report['status'] == 'TRAINING_COMPLETE' and report['completion_recovery_only']
    assert report['optimizer_steps_this_segment'] == 0 and report['experiment_completed_updates'] == 500
    assert evaluations == [(500, output / 'checkpoints/update_000500.pt')]
    assert runner.alg.learning_rate == 1.5e-5 and raw.common_step_counter == 36000
    assert all(int(x['step']) == 30000 for x in runner.alg.optimizer.state.values())
