"""Interleaved Orin CPU comparisons on accepted observations; no plant output."""
import argparse
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

from deploy_contract import OnnxPolicy, require, comparison, sha, ATOL, RTOL
from fast_policy import BufferedPolicy
from loop_contract import TOTAL, frozen, measured


def benchmark(folder, cancel_file=None):
    folder = Path(folder)
    frozen(folder)
    recording = json.loads((folder / 'observations.json').read_bytes())
    values = recording['values']
    require(len(values) == TOTAL and recording['source']['report_sha256'] ==
            '615d9b29e1c85433eebec2a988031ba01f14fc53ffeebf9906ae9355e6ac28e5', 'Unexpected benchmark observations')
    result = dict(status='POLICY_BENCHMARK_COMPLETE', variants={}, source=recording['source'],
                  original_onnx_sha256=sha(folder / 'final6h_005304.onnx'),
                  added_PPO_updates=0, motor_io=False, candidate_model_activated=False,
                  new_physics_steps=0, hardware_acceptance=False, realtime_acceptance=False)
    import onnxruntime as ort
    result['runtime_environment'] = dict(python=sys.version,onnxruntime=ort.__version__,
        architecture=platform.machine(),cpu_affinity=sorted(os.sched_getaffinity(0)),
        execution_provider='CPUExecutionProvider',intra_op_threads=1,inter_op_threads=1,
        clocks_or_power_mode_modified=False)
    policies = {}
    specs = [('fp32', OnnxPolicy, folder / 'final6h_005304.onnx'),
             ('buffered_fp32', BufferedPolicy, folder / 'final6h_005304.onnx')]
    candidate = json.loads((folder / 'candidate-report.json').read_bytes())
    result['candidate_preparation'] = candidate
    if candidate['status'] == 'QUANTIZATION_CANDIDATE_PREPARED':
        path = folder / 'mlp-dynamic-int8.onnx'
        require(sha(path) == candidate['candidate_sha256'], 'INT8 candidate changed')
        specs.append(('mlp_int8', OnnxPolicy, path))
    records = {}
    for name, factory, path in specs:
        try:
            policy = factory(path)
            for i, row in enumerate(values[:32]):
                policy.step(row['obs'], reset=i == 0)
            policy.reset()
            policies[name] = policy
            records[name] = dict(latency=[], errors=[], state_max=0., violating_actions=0, compared=0)
        except Exception as exc:
            result['variants'][name] = dict(status='UNAVAILABLE', error=repr(exc), eligible=False)
    require('fp32' in policies, 'Original FP32 runtime unavailable')
    started = time.monotonic()
    for seq, row in enumerate(values):
        require(cancel_file is None or not Path(cancel_file).exists(), 'Controller requested stop')
        require(time.monotonic() - started < 180, 'Bounded policy benchmark exceeded 180 seconds')
        names = list(policies)
        offset = seq % len(names)
        outputs = {}
        for name in names[offset:] + names[:offset]:
            policy = policies[name]
            try:
                before = time.perf_counter_ns()
                output = policy.step(row['obs'], reset=seq in (0, 3000))
                records[name]['latency'].append((time.perf_counter_ns() - before) / 1e6)
                outputs[name] = output
            except Exception as exc:
                if name == 'fp32':
                    raise
                result['variants'][name] = dict(status='FAILED', error=repr(exc), eligible=False)
                del policies[name]
        comparison(outputs['fp32'], row['actions'])
        baseline = policies['fp32']
        for name, output in outputs.items():
            errors = [abs(a-b) for a, b in zip(output, row['actions'])]
            data = records[name]
            data['errors'].extend(errors)
            data['violating_actions'] += sum(e > ATOL + RTOL * abs(v) for e, v in zip(errors, row['actions']))
            policy = policies[name]
            state_error = max(float(abs(policy.h-baseline.h).max()), float(abs(policy.c-baseline.c).max()))
            data['state_max'] = max(data['state_max'], state_error)
            data['compared'] += 1
        if (seq + 1) % 1000 == 0:
            print('POLICY_BENCHMARK=%d/6000 %.1f%%' % (seq+1, (seq+1)/60.), flush=True)
    for name, data in records.items():
        if name in result['variants']:
            continue
        errors = sorted(data['errors'])
        result['variants'][name] = dict(status='MEASURED', compared_steps=data['compared'],
            latency_ms=measured(data['latency']), max_action_absolute_error=max(errors),
            p95_action_absolute_error=errors[int(.95*len(errors))-1],
            max_state_absolute_error=data['state_max'], actions_outside_original_tolerance=data['violating_actions'],
            strict_numerical_parity=data['compared'] == TOTAL and data['violating_actions'] == 0
                                    and data['state_max'] <= ATOL,
            eligible_for_FP32_runtime_selection=name in ('fp32', 'buffered_fp32'),
            needs_new_physical_closed_loop_validation=name == 'mlp_int8')
    base = result['variants']['fp32']['latency_ms']
    fast = result['variants'].get('buffered_fp32', {})
    use_fast = fast.get('strict_numerical_parity') is True \
        and fast['latency_ms']['p95'] <= .95 * base['p95'] \
        and fast['latency_ms']['mean'] < base['mean']
    result.update(selected_FP32_runtime='buffered_fp32' if use_fast else 'fp32',
        selection_rule='same-model parity, at least 5% lower p95 and lower mean in this benchmark',
        int8_deployment_selected=False, elapsed_seconds=time.monotonic()-started)
    frozen(folder)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--folder',required=True)
    parser.add_argument('--cancel-file')
    args=parser.parse_args()
    try:
        result=benchmark(args.folder,args.cancel_file)
    except Exception as exc:
        result=dict(status='POLICY_BENCHMARK_FAILED_PRESERVE',error=repr(exc),traceback=traceback.format_exc(),
                    motor_io=False,added_PPO_updates=0,candidate_model_activated=False)
    with (Path(args.folder)/'policy-benchmark.json').open('x') as stream:
        json.dump(result,stream,indent=2,allow_nan=False)
    print('POLICY_BENCHMARK_STATUS='+result['status'],flush=True)
    raise SystemExit(0 if result['status']=='POLICY_BENCHMARK_COMPLETE' else 2)
