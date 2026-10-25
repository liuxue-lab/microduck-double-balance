"""Use the accepted native task and audits; apply Orin actions, never train."""
import argparse
import json
import math
from pathlib import Path
import signal
import time
import traceback
from types import SimpleNamespace

from bootstrap import load_base
old = load_base()
import native_evaluate as evaluate
from deploy_contract import MODEL_SHA, comparison, validate_report
from fingerprint import fingerprint
from native_runtime import session_type, adapter_checks, PIPELINE
from telemetry import Timing
from live_progress import emit
import live_progress
from realtime_io import DeferredTrace,export_models
from loop_evidence import reviewed_evidence, verify_own_package
from loop_contract import CASES, STEPS, TOTAL, WORKER_LIMIT, frozen, measured
from reference_overlap import LoopClient


def check_runtime_contract(env, agent, contract, torch):
    from mjlab.rl.exporter_utils import get_base_metadata
    raw = env.unwrapped
    old.require(env.clip_actions is None and agent['clip_actions'] is None
                and contract['action_clip'] is None and contract['action_filter'] is None,
                'Validated unclipped/unfiltered action path differs')
    metadata = get_base_metadata(raw, run_path='inference-only')
    old.require(metadata['joint_names'] == contract['runtime_metadata']['joint_names'], 'Joint order differs')
    joint = raw.action_manager.get_term('joint_pos')
    def values(x):
        return x[0].detach().cpu().tolist() if isinstance(x, torch.Tensor) else x
    old.require(values(joint._offset) == contract['action_offset']
                and values(joint._scale) == contract['action_scale'], 'Action scale or offset differs')
    terms, offset = [], 0
    for name, dim in zip(raw.observation_manager.active_terms['actor'], raw.observation_manager.group_obs_term_dim['actor']):
        width = math.prod(dim)
        term = raw.observation_manager.get_term_cfg('actor', name)
        terms.append(dict(name=name, start=offset, stop=offset + width,
                          function=term.func.__module__ + '.' + term.func.__name__, scale=term.scale, clip=term.clip))
        offset += width
    old.require(json.loads(json.dumps(terms)) == contract['actor_terms'] and offset == 61,
                'Original observation layout/scale/clip differs')
    old.require(any(t['name'] == 'body_command' and t['start'] == 55 and t['stop'] == 61
                    and t['function'] == 'native_task.head_actor_observation' for t in terms),
                'Head-ball sensor block must remain the original six observations')


class RemotePolicy:
    """Reference actor observes the same sequence; only the remote action is applied."""
    def __init__(self, reference, client, torch, case_index, stats):
        self.reference, self.client, self.torch = reference, client, torch
        self.case_index, self.stats, self.reset_calls = case_index, stats, 0
        self.prepared = False

    def __getattr__(self, name):
        return getattr(self.reference, name)

    def reset(self):
        old.require(self.reset_calls == 0 and self.client.seq == self.case_index * STEPS,
                    'Unexpected policy reset/retry')
        self.reference.reset()
        self.reset_calls += 1

    def prepare(self, observations, boundary):
        old.require(not self.prepared and self.reset_calls == 1 and
                    self.client.seq == self.case_index * STEPS and
                    not self.stats['timing'].rows and self.stats['timing'].epoch is None,
                    'Reference preparation must precede every formal release')
        from policy_prepare import prepare_reference
        self.stats['reference_preparation'] = prepare_reference(
            self.reference, observations, self.torch, boundary)
        self.prepared = True
        return self.stats['reference_preparation']

    def __call__(self, observations, stochastic_output=False):
        old.require(self.prepared, 'Actual reference must be prepared before first control release')
        old.require(self.reset_calls == 1 and stochastic_output is False, 'Expected deterministic reset policy')
        old.require(self.client.seq // STEPS == self.case_index, 'Case request budget exceeded')
        timing = self.stats['timing']
        begun = timing.begin()
        raw = observations['actor']
        old.require(tuple(raw.shape) == (1, 61), 'Actor observation shape differs')
        observation = raw[0].detach().cpu().tolist()
        now = timing.clock()
        timing.add(obs_copy_ms=(now - begun) / 1e6)
        expected=None
        def reference():
            nonlocal expected
            start=timing.clock()
            expected=self.reference(observations.to('cpu'),stochastic_output=False)
            timing.add(reference_inference_ms=(timing.clock()-start)/1e6)
        actions = self.client.step_with_reference(observation,reference)
        after_reply = timing.clock()
        timing.add(request_call_ms=(after_reply - now) / 1e6,
            ipc_roundtrip_ms=self.client.ipc_ms[-1], ros_roundtrip_ms=self.client.ros_ms[-1],
            orin_inference_ms=self.client.inference_ms[-1])
        old.require(expected is not None,'Missing original reference action')
        error = comparison(actions, expected[0].detach().cpu().tolist())
        self.stats['max_action_absolute_error'] = max(self.stats['max_action_absolute_error'], error)
        self.stats['compared_actions'] += 1
        # Conversion only. No normalization, clipping, filtering, or fallback.
        after_reference = timing.clock()
        timing.add(reference_compare_ms=(after_reference - after_reply) / 1e6)
        action = self.torch.tensor([actions], dtype=expected.dtype, device=raw.device)
        
        timing.add(action_to_cpu_ms=(timing.clock() - after_reference) / 1e6)
        timing.policy_returned()
        return action


def linked_context(base, client, case_index, stats, contract):
    class Runner:
        def __init__(self, env, agent, **kwargs):
            check_runtime_contract(env, agent, contract, base.torch)
            self.inner = base.MjlabOnPolicyRunner(env, agent, **kwargs)
            self.alg = self.inner.alg

        def get_inference_policy(self, **kwargs):
            # Use the same CPU PyTorch reference path as the accepted export
            # parity check; the simulated plant is native MuJoCo on CPU.
            policy = self.inner.get_inference_policy(device='cpu')
            old.require(policy.is_recurrent and fingerprint(policy.state_dict()) == contract['actor_state_sha256']
                        and fingerprint(policy.obs_normalizer.state_dict()) == contract['normalizer_sha256'],
                        'Live reference actor or normalizer differs from accepted export')
            # The actual instance is prepared after the final environment reset,
            # on the real input layout, immediately before control is armed.
            return RemotePolicy(policy, client, base.torch, case_index, stats)

    return SimpleNamespace(**dict(vars(base), MjlabOnPolicyRunner=Runner,
                                  EvaluationSession=session_type(base, stats), timing=stats['timing']))


def run(args):
    live_progress.configure()
    run = Path(args.run)
    started = time.monotonic()
    stopped = [False]
    case_deadline = [started + WORKER_LIMIT]
    client = trace = None
    pending_models=[]
    before = None
    result = dict(status='LOOP_SIM_FAILED_PRESERVE', cases=[], planned_cases=list(CASES),
                  reference_preparation_required=True,
                  added_PPO_updates=0, hardware_acceptance=False, realtime_acceptance=False,
                  motor_io=False, auto_retry=False, auto_training=False,
                  mode='fixed 20ms release/deadline; validated Orin response required before each native step',
                  active_pipeline=PIPELINE, original_delay_clock_ms=2., physics_hz=200, all_native_substeps_audited=True,
                  full_loop_realtime_acceptance=False, physical_robot_in_scope=False)
    def boundary():
        # The supervisor polls STOP.json and forwards SIGTERM to this owned
        # process group. Control boundaries inspect memory and the clock only.
        if stopped[0]:
            raise evaluate.EvaluationInterrupted('user_stop_requested')
        if time.monotonic() >= min(case_deadline[0], started + WORKER_LIMIT):
            raise evaluate.EvaluationInterrupted('bounded_case_or_worker_deadline')
    def interrupt(signum, frame):
        if signum == signal.SIGALRM:
            raise evaluate.EvaluationInterrupted('bounded_case_deadline')
        stopped[0] = True
    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGALRM):
        signal.signal(sig, interrupt)
    completed = 0
    index = 0
    def progress(**values):
        count = completed + values.get('control_steps', 0)
        row = dict(status='RUNNING', elapsed_seconds=time.monotonic() - started,
                   completed_control_steps=count, total_control_steps=TOTAL, percent=100. * count / TOTAL,
                   case_number=index + 1, cases_total=len(CASES), **values)
        emit(row)
    try:
        boundary()
        result['package_identity'] = verify_own_package()
        result['reviewed_evidence'] = reviewed_evidence()
        from native_preflight import preflight
        result['preflight'] = preflight()
        before = old.verify_inputs()
        result['export'] = frozen(args.export)
        contract = old.read(Path(args.export) / 'contract.json')
        checkpoint = validate_report(old.SLOT / 'report.json')
        old.require(Path(checkpoint['path']) == old.INPUTS / 'run/checkpoints/final6h_005304.pt',
                    'Unexpected selected checkpoint path')
        progress(detail='loading_native_mujoco_CPU_runtime')
        base = evaluate.load_runtime()
        trace = DeferredTrace(run/'loop-transcript.jsonl',24*1024**2,TOTAL)
        client = LoopClient(args.socket, args.session, trace)
        for index, case in enumerate(CASES):
            boundary()
            folder = run / ('case-%02d' % (index + 1))
            folder.mkdir()
            stats = dict(compared_actions=0, max_action_absolute_error=0., sessions=[], timing=Timing(folder),
                         pending_models=pending_models)
            context = linked_context(base, client, index, stats, contract)
            case_deadline[0] = time.monotonic() + 900
            signal.alarm(900)
            row = evaluate.evaluate_one(context, folder, case, checkpoint,
                                        result['preflight']['package_identity'], boundary, progress)
            signal.alarm(0)
            row['live_action_parity'] = dict(compared_actions=stats['compared_actions'],
                max_action_absolute_error=stats['max_action_absolute_error'], atol=2e-5, rtol=2e-4,
                every_applied_action_came_from_Orin=True, local_fallback=False)
            row['timing'] = stats['timing'].export()
            row['runtime_adapters'] = stats.get('adapters', {})
            row['reference_preparation'] = stats.get('reference_preparation', {})
            row['adapter_checks'] = adapter_checks(row['runtime_adapters'])
            row['realtime_passed'] = row['timing']['realtime_acceptance']
            row['loop_passed'] = (row['acceptance']['passed'] and stats['compared_actions'] == STEPS
                and len(stats['sessions']) == 1 and all(row['adapter_checks'].values())
                and row.get('model_limits',{}).get('simulated_seconds')==60.
                and row['timing']['completed_ticks'] == STEPS and not row['timing']['incomplete_tick'])
            row['reward_is_compatibility_placeholder'] = True
            row['excluded_training_outputs'] = ['critic', 'reward']
            old.write(folder / 'result.json', row, exclusive=True)
            result['cases'].append(row)
            completed += row['completed_control_steps']
            old.require(row['loop_passed'], 'Closed-loop case failed; preserve first result and stop')
        client.finish()
        from review_buffered_loop import compare_states
        result['prior_state_comparison'] = compare_states(run, result['cases'])
        old.require(result['prior_state_comparison']['all_bit_identical'],
                    'Native state trajectories differ from the reviewed v2 run; preserve')
        from live_support import compare_v3
        result['accepted_v3_comparison'] = compare_v3(run, result['cases'])
        old.require(result['accepted_v3_comparison']['all_bit_identical'],
                    'Live display changed accepted v3 state trajectories')
        result['status'] = 'LOOP_SIM_FUNCTIONAL_PASSED'
        result['full_loop_realtime_acceptance'] = all(c['realtime_passed'] for c in result['cases'])
        result['realtime_acceptance'] = result['full_loop_realtime_acceptance']
    except BaseException as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc()[-12000:])
    finally:
        signal.alarm(0)
        try:
            result['model_exports']=export_models(pending_models)
        except BaseException as exc:
            result.update(status='LOOP_SIM_FAILED_PRESERVE',model_export_error=repr(exc),
                realtime_acceptance=False,full_loop_realtime_acceptance=False)
        if client is not None:
            result['transport'] = client.snapshot()
            client.close()
        if trace is not None:
            try:
                trace.close()
            except BaseException as exc:
                result.update(status='LOOP_SIM_FAILED_PRESERVE',trace_export_error=repr(exc),
                    realtime_acceptance=False,full_loop_realtime_acceptance=False)
            result['trace_export'] = trace.snapshot()
        result['live_progress'] = live_progress.snapshot()
        try:
            result['original_records_unchanged'] = before is not None and old.verify_inputs() == before
            if before is not None:
                frozen(args.export)
                old.require(reviewed_evidence() == result['reviewed_evidence'], 'Reviewed evidence changed during loop')
                verify_own_package()
            old.require(result['original_records_unchanged'], 'Could not confirm unchanged frozen inputs')
        except BaseException as exc:
            result.update(status='LOOP_SIM_FAILED_PRESERVE', cleanup_error=repr(exc),
                          realtime_acceptance=False, full_loop_realtime_acceptance=False)
        result.update(completed_control_steps=completed, elapsed_seconds=time.monotonic() - started,
                      completed_cases=len(result['cases']))
        old.write(run / 'simulator-report.json', result, exclusive=True)
        old.write(run / 'progress.json', dict(status=result['status'], completed_control_steps=completed,
            total_control_steps=TOTAL, percent=100. * completed / TOTAL,
            elapsed_seconds=result['elapsed_seconds'], error=result.get('error')))
    return 0 if result['status'] == 'LOOP_SIM_FUNCTIONAL_PASSED' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--socket', required=True)
    parser.add_argument('--export', required=True)
    raise SystemExit(run(parser.parse_args()))
