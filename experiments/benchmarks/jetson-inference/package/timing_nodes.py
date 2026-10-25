"""Fixed-rate trace replay and isolated synthetic network fault scenarios."""
import argparse
import json
from pathlib import Path
import signal
import time
import traceback

from deploy_contract import OnnxPolicy, require
from loop_contract import TOTAL, ONNX_SHA, EMPTY_CHAIN, chain, encode, frozen, packet, reply
import loop_ros
from timing_contract import PERIOD, SCENARIOS, DeadlineGate
from fast_policy import BufferedPolicy
from wire_protocol import Wire


def server(args, result):
    frozen(args.folder)
    policy = (BufferedPolicy if args.runtime == 'buffered_fp32' else OnnxPolicy)(Path(args.folder) / 'final6h_005304.onnx')
    policy.step([0.] * 61, reset=True)
    policy.reset()
    ros, node, qos, _, root = loop_ros.endpoints(args)
    wire = Wire(args.wire)
    publisher = node.create_publisher(wire.Message, root + '/actions', qos)
    total = TOTAL if args.scenario == 'replay' else 2
    state = dict(seq=0, previous=None, pending=None, emit_at=None, finished=None, fatal=None,
                 emitted=0, injected=False, suppressed=0, chain=EMPTY_CHAIN, resets=[])
    def publish(value):
        publisher.publish(wire.response(value))
        state['emitted'] += 1
    def receive(message):
        try:
            require(state['fatal'] is None and state['seq'] < total, 'Request budget exceeded')
            request = wire.decode_request(message, args.session, state['seq'])
            started = time.monotonic()
            actions = policy.step(request['obs'], reset=request['reset'])
            response = reply(request, actions, (time.monotonic() - started) * 1000)
            state['seq'] += 1
            state['chain'] = chain(state['chain'], request, response)
            if request['reset']:
                state['resets'].append(request['seq'])
            if args.scenario == 'replay' or request['seq'] == 0:
                publish(response)
                state['previous'] = response
                if state['seq'] == total:
                    state['finished'] = time.monotonic()
                return
            state['injected'] = True
            if args.scenario == 'delay':
                state.update(pending=response, emit_at=time.monotonic() + .100)
            elif args.scenario == 'silence':
                state['suppressed'] += 1
                state['finished'] = time.monotonic()
            elif args.scenario == 'duplicate':
                publish(state['previous'])
                state['finished'] = time.monotonic()
            else:
                raise RuntimeError('Unknown synthetic fault')
        except Exception as exc:
            state['fatal'] = repr(exc)
    subscription = node.create_subscription(wire.Message, root + '/observations', receive, qos)
    started = time.monotonic()
    try:
        print('TIMING_SERVER_READY=' + args.scenario, flush=True)
        while time.monotonic() - started < 210:
            require(not Path(args.cancel_file).exists(), 'Controller requested stop')
            ros.spin_once(node, timeout_sec=.001)
            require(state['fatal'] is None, str(state['fatal']))
            if state['pending'] is not None and time.monotonic() >= state['emit_at']:
                publish(state['pending'])
                state.update(pending=None, finished=time.monotonic())
            if state['finished'] is not None and time.monotonic() - state['finished'] > 1.:
                break
        require(state['seq'] == total, 'Server request count incomplete')
        if args.scenario != 'replay':
            require(state['injected'], 'Fault was not exercised')
        else:
            require(state['resets'] == [i for i in (0, 3000) if i < total], 'LSTM reset schedule differs')
        frozen(args.folder)
        result.update(status='REPLAY_SERVER_COMPLETE' if args.scenario == 'replay' else 'FAULT_SERVER_COMPLETE', model_unchanged=True)
    finally:
        result.update(completed_requests=state['seq'], emitted_replies=state['emitted'],
            suppressed_replies=state['suppressed'], fault_injected=state['injected'],
            fault_request_sequence=1, injected_delay_ms=100 if args.scenario == 'delay' else 0,
            recurrent_reset_sequences=state['resets'], transcript_sha256=state['chain'])
        node.destroy_node()
        ros.shutdown()


def client(args, result):
    from timing_source import load_recording
    values, source = load_recording()
    result['source_recording'] = source
    total = TOTAL if args.scenario == 'replay' else 2
    # Encode all payloads before starting the fixed-rate clock.
    requests = [packet(args.session, i, v['obs']) for i, v in enumerate(values[:total])]
    wire = Wire(args.wire)
    payloads = [wire.request(p) for p in requests]
    sizes = [len(m.data.encode()) if args.wire == 'json' else len(m.data) for m in payloads]
    result.update(request_application_payload_bytes=dict(min=min(sizes),max=max(sizes),mean=sum(sizes)/len(sizes)),
                  dds_serialization_and_network_headers_included_in_byte_count=False)
    # Binary float32 must preserve the exact recorded observation digest.
    if args.wire == 'binary':
        require(all(encode(wire.decode_request(m, args.session, i)) == encode(requests[i])
                    for i, m in enumerate(payloads)), 'Recording is not exactly float32 representable')
    ros, node, qos, _, root = loop_ros.endpoints(args)
    publisher = node.create_publisher(wire.Message, root + '/observations', qos)
    trace = (Path(args.folder) / 'timing-transcript.jsonl').open('x', encoding='utf-8')
    gate = DeadlineGate(args.session, clock=time.monotonic, trace=trace)
    def receive(message):
        try:
            payload = wire.decode_response(message)
        except Exception as exc:
            gate.fail('INVALID_REPLY', repr(exc))
            gate.discard('INVALID_REPLY', repr(exc))
            return
        gate.receive(payload)
    subscription = node.create_subscription(wire.Message, root + '/actions', receive, qos)
    started = time.monotonic()
    sent_messages = 0
    def boundary():
        require(not Path(args.cancel_file).exists(), 'Controller requested stop')
        require(time.monotonic() - started < 195, 'Timing client wall-clock limit exceeded')
    try:
        while node.count_subscribers(root + '/observations') != 1 or node.count_publishers(root + '/actions') != 1:
            boundary()
            require(time.monotonic() - started < 40, 'ROS wired discovery timed out')
            ros.spin_once(node, timeout_sec=.02)
        schedule = time.monotonic() + .1
        for seq in range(total):
            due = schedule + seq * PERIOD
            while time.monotonic() < due:
                boundary()
                ros.spin_once(node, timeout_sec=min(.001, max(0., due - time.monotonic())))
                require(gate.fault is None, 'Unexpected reply before next scheduled tick')
            boundary()
            gate.begin(requests[seq], values[seq]['actions'], due)
            publisher.publish(payloads[seq])
            sent_messages += 1
            while gate.seq == seq and gate.fault is None:
                boundary()
                gate.poll()
                if gate.fault is None:
                    ros.spin_once(node, timeout_sec=.001)
            if gate.fault is not None:
                break
            if (seq + 1) % 250 == 0 or seq + 1 == total:
                trace.flush()
                print('50Hz %d/%d | %.2f%% | 最近有效回复 %.3f ms' %
                      (seq + 1, total, 100. * (seq + 1) / total, gate.age_ms[-1]), flush=True)
                progress = Path(args.folder) / 'progress.json'
                temp = progress.with_suffix('.tmp')
                temp.write_text(json.dumps(dict(completed=seq+1,total=total,percent=100.*(seq+1)/total)))
                temp.replace(progress)
        if args.scenario == 'replay':
            require(gate.fault is None and gate.seq == TOTAL, 'Fixed-rate deadline/identity/parity check failed')
            # Drain an already queued duplicate before declaring completion.
            ros.spin_once(node, timeout_sec=0.)
            require(gate.fault is None, 'Unexpected reply after final tick')
            result.update(status='SCHEDULE_REPLAY_PASSED', scheduled_frequency_hz=50,
                          elapsed_schedule_seconds=time.monotonic() - schedule,
                          simulator_integrated=False, complete_control_loop_realtime_acceptance=False,
                          payloads_preencoded_before_clock=True, physical_observation_capture_included=False)
        else:
            fault_deadline = schedule + PERIOD + PERIOD
            # Keep listening after the fault so a 100-ms late reply is observed
            # and demonstrably discarded. No additional observations are sent.
            drain_until = max(time.monotonic(), fault_deadline) + .3
            while time.monotonic() < drain_until:
                boundary()
                gate.poll()
                ros.spin_once(node, timeout_sec=.001)
            refused = False
            try:
                gate.begin(packet(args.session, 2, values[2]['obs']), values[2]['actions'], time.monotonic())
            except RuntimeError as exc:
                refused = 'FAULT_LATCHED' in str(exc)
            fault = gate.fault or {}
            detection_ms = (fault.get('monotonic_seconds', float('inf')) - (schedule + PERIOD)) * 1000
            conditions = dict(first_reply_accepted=gate.seq == 1, exactly_two_requests=sent_messages == 2,
                fault_on_second_request=fault.get('seq') == 1, later_publication_refused=refused,
                detected_within_25ms=0 <= detection_ms <= 25.)
            if args.scenario == 'duplicate':
                conditions.update(expected_fault=fault.get('reason') == 'INVALID_REPLY', stale_discarded=gate.discarded >= 1)
            else:
                conditions['expected_fault'] = fault.get('reason') in ('RESPONSE_DEADLINE', 'LATE_REPLY')
                if args.scenario == 'delay':
                    conditions['late_reply_discarded'] = gate.discarded >= 1
                else:
                    conditions['no_reply_received'] = gate.discarded == 0
            result.update(fault_conditions=conditions, fault_detection_from_due_ms=detection_ms,
                          diagnostic_detection_tolerance_ms=5.)
            require(all(conditions.values()), 'Synthetic fault did not satisfy its checked conditions')
            result['status'] = 'FAULT_GATE_PASSED'
    finally:
        result.update(receive_gate=gate.snapshot(), sent_messages=sent_messages)
        trace.close()
        node.destroy_node()
        ros.shutdown()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('role', choices=('server', 'client'))
    parser.add_argument('--scenario', choices=SCENARIOS, required=True)
    parser.add_argument('--folder', required=True)
    parser.add_argument('--session', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--cancel-file', required=True)
    parser.add_argument('--wire', choices=('json', 'binary'), required=True)
    parser.add_argument('--runtime', choices=('fp32', 'buffered_fp32'), required=True)
    args = parser.parse_args()
    result = dict(status='TIMING_NODE_FAILED_PRESERVE', role=args.role, scenario=args.scenario,
        session=args.session, wire=args.wire, runtime=args.runtime, onnx_sha256=ONNX_SHA, motor_io=False, added_PPO_updates=0,
        new_physics_steps=0, hardware_acceptance=False, realtime_acceptance=False,
        synthetic_fault_injection=args.scenario != 'replay')
    def stop(signum, frame):
        raise RuntimeError('Stop signal: ' + str(signum))
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, stop)
    started = time.monotonic()
    try:
        if args.role == 'client':
            client(args, result)
        else:
            server(args, result)
    except BaseException as exc:
        result.update(status='TIMING_NODE_FAILED_PRESERVE', error=repr(exc), traceback=traceback.format_exc()[-10000:])
    finally:
        result['elapsed_seconds'] = time.monotonic() - started
        loop_ros.save(args.report, result)
    print('TIMING_NODE_STATUS=' + result['status'], flush=True)
    return 0 if result['status'] in ('SCHEDULE_REPLAY_PASSED', 'FAULT_GATE_PASSED',
                                    'REPLAY_SERVER_COMPLETE', 'FAULT_SERVER_COMPLETE') else 2


if __name__ == '__main__':
    raise SystemExit(main())
