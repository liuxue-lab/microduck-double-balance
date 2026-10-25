"""Humble nodes for a bounded inference-link test. No simulator or motor I/O."""
import argparse
import json
import math
from pathlib import Path
import statistics
import time
import traceback

from deploy_contract import (SCHEMA, OnnxPolicy, comparison, request, require,
                             valid_session, verify_export, vector)


def save(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, ensure_ascii=False, allow_nan=False, indent=2)
        f.write('\n')


def percentile(values, fraction):
    return sorted(values)[max(0, math.ceil(fraction * len(values)) - 1)]


def run(args):
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
    from std_msgs.msg import String
    folder = Path(args.folder)
    report = verify_export(folder)
    model_sha = report['files']['final6h_005304.onnx']['sha256']
    session = valid_session(args.session)
    root = '/microduck/stage12/check_' + session
    qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=1,
                     reliability=ReliabilityPolicy.RELIABLE, durability=DurabilityPolicy.VOLATILE)
    rclpy.init()
    node = Node('stage12_' + args.role + '_' + session)
    result = dict(status='LINK_TEST_FAILED', role=args.role, session=session, onnx_sha256=model_sha,
                  hardware_acceptance=False, realtime_acceptance=False, plant_simulation_run=False,
                  added_PPO_updates=0, motor_io=False, expected_samples=128)
    started = time.monotonic()
    try:
        if args.role == 'server':
            policy = OnnxPolicy(folder / 'final6h_005304.onnx')
            publisher = node.create_publisher(String, root + '/actions', qos)
            state = dict(seq=0, fatal=None, finished_at=None)
            def receive(message):
                try:
                    p = request(message.data, session, model_sha, state['seq'])
                    require(state['seq'] < 128, 'Inference test sample limit exceeded')
                    begin = time.perf_counter_ns()
                    actions = policy.step(p['obs'], reset=p['reset'])
                    elapsed = (time.perf_counter_ns() - begin) / 1e6
                    reply = dict(schema=SCHEMA, session=session, model=model_sha, seq=p['seq'],
                                 actions=actions, h=policy.h.reshape(-1).tolist(),
                                 c=policy.c.reshape(-1).tolist(), inference_ms=elapsed)
                    publisher.publish(String(data=json.dumps(reply, allow_nan=False, separators=(',', ':'))))
                    state['seq'] += 1
                    if state['seq'] == 128:
                        state['finished_at'] = time.monotonic()
                except Exception as exc:
                    state['fatal'] = repr(exc)
                    publisher.publish(String(data=json.dumps(dict(schema=SCHEMA, session=session,
                        model=model_sha, seq=state['seq'], error=state['fatal']))))
            subscription = node.create_subscription(String, root + '/observations', receive, qos)
            print('ROS_SERVER_READY', flush=True)
            while time.monotonic() - started < 80:
                if (folder / 'cancel').exists():
                    raise RuntimeError('Cancelled by local controller')
                rclpy.spin_once(node, timeout_sec=.05)
                if state['fatal']:
                    raise RuntimeError(state['fatal'])
                if state['finished_at'] is not None and time.monotonic() - state['finished_at'] > 1:
                    break
            require(state['seq'] == 128, 'Bounded server deadline expired before all observations')
            result.update(status='LINK_SERVER_COMPLETE', completed_samples=state['seq'])
        else:
            data = json.loads((folder / 'golden.json').read_bytes())
            require(data['onnx_sha256'] == model_sha and len(data['cases']) == 128, 'Golden fixture identity differs')
            received = {}
            expected = {'seq': None, 'error': None}
            def receive(message):
                try:
                    require(len(message.data) < 40000, 'Oversized response')
                    p = json.loads(message.data)
                    require(p['schema'] == SCHEMA and p['session'] == session and p['model'] == model_sha,
                            'Response identity differs')
                    require(type(p['seq']) is int and p['seq'] == expected['seq'] and not received,
                            'Stale, repeated or reordered response')
                    require('error' not in p, 'Orin: ' + str(p.get('error')))
                    received.update(p)
                except Exception as exc:
                    expected['error'] = repr(exc)
            publisher = node.create_publisher(String, root + '/observations', qos)
            subscription = node.create_subscription(String, root + '/actions', receive, qos)
            while node.count_subscribers(root + '/observations') < 1 or node.count_publishers(root + '/actions') < 1:
                require(time.monotonic() - started < 25, 'ROS 2 wired discovery timed out; settings preserved')
                rclpy.spin_once(node, timeout_sec=.05)
            rtt, inference, dispatch_late = [], [], []
            max_action = max_state = 0.
            schedule = time.monotonic()
            for case in data['cases']:
                seq = case['seq']
                expected['seq'] = seq
                received.clear()
                due = schedule + seq * .02
                while time.monotonic() < due:
                    rclpy.spin_once(node, timeout_sec=min(.002, max(0., due - time.monotonic())))
                sent = time.monotonic()
                dispatch_late.append(max(0., (sent - due) * 1000))
                packet = dict(schema=SCHEMA, session=session, model=model_sha, seq=seq,
                              reset=case['reset'], obs=case['obs'])
                publisher.publish(String(data=json.dumps(packet, allow_nan=False, separators=(',', ':'))))
                while not received:
                    require(expected['error'] is None, str(expected['error']))
                    require(time.monotonic() - sent < 2., 'Response timed out; no stale action or automatic retry')
                    rclpy.spin_once(node, timeout_sec=.001)
                rtt.append((time.monotonic() - sent) * 1000)
                require(expected['error'] is None, str(expected['error']))
                max_action = max(max_action, comparison(received['actions'], case['actions']))
                max_state = max(max_state, comparison(received['h'], case['h']), comparison(received['c'], case['c']))
                require(type(received['inference_ms']) in (int, float)
                        and math.isfinite(received['inference_ms']) and received['inference_ms'] >= 0,
                        'Invalid inference timing')
                inference.append(received['inference_ms'])
                if (seq + 1) % 32 == 0:
                    print('ROS_PROGRESS=%d/128 RTT=%.3fms' % (seq + 1, rtt[-1]), flush=True)
            result.update(status='LINK_PARITY_PASSED', completed_samples=128,
                          max_action_absolute_error=max_action, max_state_absolute_error=max_state,
                          target_dispatch_hz=50, rtt_ms=dict(max=max(rtt), p95=percentile(rtt,.95),
                          p99=percentile(rtt,.99), mean=statistics.mean(rtt)),
                          inference_ms=dict(max=max(inference), p95=percentile(inference,.95)),
                          rtt_over_20ms=sum(x > 20 for x in rtt),
                          max_dispatch_lateness_ms=max(dispatch_late),
                          measured_all_requests_within_20ms=all(x <= 20 for x in rtt),
                          measured_dispatch_lateness_within_2ms=all(x <= 2 for x in dispatch_late),
                          warning='128 fixed vectors only; not a plant-loop or hard-real-time acceptance test')
    except BaseException as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc()[-8000:])
    finally:
        node.destroy_node()
        rclpy.shutdown()
        result['elapsed_seconds'] = time.monotonic() - started
        save(args.report, result)
    print('ROS_RESULT=' + result['status'], flush=True)
    return 0 if result['status'] in ('LINK_PARITY_PASSED', 'LINK_SERVER_COMPLETE') else 2


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('role', choices=('client', 'server'))
    p.add_argument('--folder', required=True)
    p.add_argument('--session', required=True)
    p.add_argument('--report', required=True)
    raise SystemExit(run(p.parse_args()))
